"""
Symbol-Specific Parameter Calculator
====================================

This script implements PRIORITY 5, Task 10 from AI_AGENT_INSTRUCTIONS.md:
- Calculate actual average spread from data
- Calculate average daily volume per symbol
- Calculate beta to SPY for each symbol
- Update symbols.yaml with real values

Accurate transaction costs and risk parameters are essential for:
- Realistic backtesting (wrong costs = misleading results)
- Proper position sizing (beta-adjusted risk)
- Execution planning (volume-based sizing)

Usage:
    python scripts/update_symbol_params.py --calculate    # Calculate all parameters
    python scripts/update_symbol_params.py --update       # Update symbols.yaml
    python scripts/update_symbol_params.py --validate     # Validate current config

Author: AlphaTrade System
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SymbolMetrics:
    """Calculated metrics for a single symbol"""
    symbol: str

    # Price characteristics
    avg_price: float
    price_volatility: float           # Daily price volatility

    # Spread analysis
    avg_spread_bps: float             # Average spread in basis points
    median_spread_bps: float
    spread_volatility: float

    # Volume analysis
    avg_daily_volume: float           # Shares per day
    avg_daily_dollar_volume: float    # Dollar volume per day
    volume_volatility: float

    # Risk metrics
    beta_to_spy: float                # Beta relative to S&P 500
    correlation_to_spy: float
    annualized_volatility: float

    # Classification
    volatility_group: str             # "low", "medium", "high"
    liquidity_group: str              # "low", "medium", "high"


@dataclass
class SymbolConfig:
    """Final configuration for a symbol"""
    symbol: str
    sector: str

    # From symbols.yaml or calculated
    weight_limit: float
    spread_bps: float
    correlation_group: int

    # Calculated values
    avg_daily_volume: float
    avg_daily_dollar_volume: float
    beta: float
    volatility_group: str
    liquidity_group: str


# ============================================================================
# METRIC CALCULATORS
# ============================================================================

class SpreadCalculator:
    """
    Calculate spread from OHLC data.

    Since we don't have bid-ask data, we estimate spread from:
    1. High-Low range (proxy for intraday spread)
    2. Close-to-Close volatility (for overnight gaps)

    The high-low spread estimate is: (High - Low) / Close * 10000 bps
    This overestimates true spread but captures execution cost range.
    """

    def calculate_spread_estimate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate spread estimates from OHLC data.

        Returns dict with spread statistics.
        """
        # High-Low spread proxy
        hl_spread = (df['high'] - df['low']) / df['close'] * 10000  # In basis points

        # Filter outliers (earnings, news events)
        hl_spread_clean = hl_spread[hl_spread < hl_spread.quantile(0.99)]

        return {
            'avg_spread_bps': hl_spread_clean.mean(),
            'median_spread_bps': hl_spread_clean.median(),
            'spread_std': hl_spread_clean.std(),
            'spread_p95': hl_spread_clean.quantile(0.95)
        }


class VolumeAnalyzer:
    """
    Analyze trading volume characteristics.

    Volume metrics help with:
    - Position sizing (don't be > 1% of daily volume)
    - Execution planning (VWAP/TWAP scheduling)
    - Liquidity risk assessment
    """

    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume metrics from OHLC data.
        """
        # Group by day for daily metrics
        df_daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Daily volume
        avg_daily_volume = df_daily['volume'].mean()
        volume_std = df_daily['volume'].std()

        # Dollar volume
        avg_price = df_daily['close'].mean()
        avg_daily_dollar_volume = avg_daily_volume * avg_price

        return {
            'avg_daily_volume': avg_daily_volume,
            'avg_daily_dollar_volume': avg_daily_dollar_volume,
            'volume_std': volume_std,
            'volume_cv': volume_std / avg_daily_volume if avg_daily_volume > 0 else 0,
            'avg_price': avg_price
        }


class BetaCalculator:
    """
    Calculate beta to market benchmark (SPY).

    Beta measures:
    - Systematic risk exposure
    - How much the stock moves with the market
    - Important for position sizing and hedging

    Beta = Cov(stock, market) / Var(market)
    """

    def __init__(self, market_data: pd.DataFrame = None):
        """
        Initialize with market benchmark data.

        Args:
            market_data: DataFrame with market returns (e.g., SPY)
        """
        self.market_data = market_data

    def calculate_beta(
        self,
        stock_data: pd.DataFrame,
        window: int = 60
    ) -> Dict[str, float]:
        """
        Calculate beta and correlation to market.

        Args:
            stock_data: OHLCV DataFrame for stock
            window: Lookback window in days

        Returns:
            Dict with beta, correlation, and volatility
        """
        if self.market_data is None:
            # Return default values if no market data
            return {
                'beta': 1.0,
                'correlation': 0.5,
                'annualized_vol': 0.2
            }

        # Calculate returns
        stock_returns = stock_data['close'].pct_change().dropna()
        market_returns = self.market_data['close'].pct_change().dropna()

        # Align indices
        common_idx = stock_returns.index.intersection(market_returns.index)
        stock_ret = stock_returns.loc[common_idx]
        market_ret = market_returns.loc[common_idx]

        if len(common_idx) < window:
            return {
                'beta': 1.0,
                'correlation': 0.5,
                'annualized_vol': stock_ret.std() * np.sqrt(252 * 26)  # 26 bars per day
            }

        # Calculate beta
        covariance = stock_ret.cov(market_ret)
        market_variance = market_ret.var()
        beta = covariance / market_variance if market_variance > 0 else 1.0

        # Calculate correlation
        correlation = stock_ret.corr(market_ret)

        # Annualized volatility
        annualized_vol = stock_ret.std() * np.sqrt(252 * 26)

        return {
            'beta': beta,
            'correlation': correlation,
            'annualized_vol': annualized_vol
        }


# ============================================================================
# SYMBOL ANALYZER
# ============================================================================

class SymbolAnalyzer:
    """
    Comprehensive symbol analysis combining all metrics.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        market_symbol: str = "SPY"
    ):
        self.data_dir = Path(data_dir)
        self.market_symbol = market_symbol

        self.spread_calc = SpreadCalculator()
        self.volume_analyzer = VolumeAnalyzer()

        # Load market data for beta calculation
        market_path = self.data_dir / f"{market_symbol}_15min.csv"
        if market_path.exists():
            market_data = pd.read_csv(market_path, parse_dates=['timestamp'], index_col='timestamp')
            self.beta_calc = BetaCalculator(market_data)
        else:
            logger.warning(f"Market data ({market_symbol}) not found, using default beta=1.0")
            self.beta_calc = BetaCalculator(None)

    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load data for a symbol."""
        # Try processed first, then raw
        for suffix in ['_15min_clean.csv', '_15min.csv']:
            for directory in [Path("data/processed"), self.data_dir]:
                path = directory / f"{symbol}{suffix}"
                if path.exists():
                    df = pd.read_csv(path)
                    # Ensure timestamp is properly parsed as DatetimeIndex
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                        df = df.set_index('timestamp')
                        df.index = df.index.tz_localize(None)  # Remove timezone
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], utc=True)
                        df = df.set_index('date')
                        df.index = df.index.tz_localize(None)
                    # Ensure index is DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                    return df

        raise FileNotFoundError(f"No data found for {symbol}")

    def classify_volatility(self, annualized_vol: float) -> str:
        """
        Classify symbol into volatility bucket.

        Thresholds:
        - Low: < 20% annualized
        - Medium: 20-40%
        - High: > 40%
        """
        if annualized_vol < 0.20:
            return "low"
        elif annualized_vol < 0.40:
            return "medium"
        else:
            return "high"

    def classify_liquidity(self, avg_dollar_volume: float) -> str:
        """
        Classify symbol into liquidity bucket.

        Thresholds (daily dollar volume):
        - Low: < $50M
        - Medium: $50M - $500M
        - High: > $500M
        """
        if avg_dollar_volume < 50_000_000:
            return "low"
        elif avg_dollar_volume < 500_000_000:
            return "medium"
        else:
            return "high"

    def analyze_symbol(self, symbol: str) -> SymbolMetrics:
        """
        Perform comprehensive analysis of a single symbol.
        """
        logger.info(f"Analyzing {symbol}...")

        # Load data
        df = self.load_symbol_data(symbol)

        # Calculate spread
        spread_metrics = self.spread_calc.calculate_spread_estimate(df)

        # Calculate volume
        volume_metrics = self.volume_analyzer.calculate_volume_metrics(df)

        # Calculate beta
        beta_metrics = self.beta_calc.calculate_beta(df)

        # Classify
        vol_group = self.classify_volatility(beta_metrics['annualized_vol'])
        liq_group = self.classify_liquidity(volume_metrics['avg_daily_dollar_volume'])

        return SymbolMetrics(
            symbol=symbol,
            avg_price=volume_metrics['avg_price'],
            price_volatility=beta_metrics['annualized_vol'],
            avg_spread_bps=spread_metrics['avg_spread_bps'],
            median_spread_bps=spread_metrics['median_spread_bps'],
            spread_volatility=spread_metrics['spread_std'],
            avg_daily_volume=volume_metrics['avg_daily_volume'],
            avg_daily_dollar_volume=volume_metrics['avg_daily_dollar_volume'],
            volume_volatility=volume_metrics['volume_cv'],
            beta_to_spy=beta_metrics['beta'],
            correlation_to_spy=beta_metrics['correlation'],
            annualized_volatility=beta_metrics['annualized_vol'],
            volatility_group=vol_group,
            liquidity_group=liq_group
        )

    def analyze_all_symbols(
        self,
        symbols: List[str] = None
    ) -> Dict[str, SymbolMetrics]:
        """
        Analyze all symbols in the universe.
        """
        if symbols is None:
            symbols = [f.stem.replace('_15min', '') for f in self.data_dir.glob('*_15min.csv')]

        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")

        return results


# ============================================================================
# CONFIG UPDATER
# ============================================================================

class SymbolConfigUpdater:
    """
    Update symbols.yaml with calculated parameters.
    """

    def __init__(
        self,
        config_path: str = "config/symbols.yaml",
        metrics: Dict[str, SymbolMetrics] = None
    ):
        self.config_path = Path(config_path)
        self.metrics = metrics or {}

    def load_current_config(self) -> Dict:
        """Load current symbols.yaml."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def update_config(self) -> Dict:
        """
        Update configuration with calculated values.
        """
        config = self.load_current_config()

        if 'symbols' not in config:
            config['symbols'] = {}

        for symbol, metrics in self.metrics.items():
            if symbol not in config['symbols']:
                config['symbols'][symbol] = {}

            symbol_config = config['symbols'][symbol]

            # Update with calculated values
            symbol_config['calculated_params'] = {
                'spread_bps': round(metrics.avg_spread_bps, 2),
                'avg_daily_volume': int(metrics.avg_daily_volume),
                'avg_daily_dollar_volume': int(metrics.avg_daily_dollar_volume),
                'beta': round(metrics.beta_to_spy, 3),
                'correlation_to_spy': round(metrics.correlation_to_spy, 3),
                'annualized_volatility': round(metrics.annualized_volatility, 4),
                'volatility_group': metrics.volatility_group,
                'liquidity_group': metrics.liquidity_group
            }

            # Update spread if it was placeholder
            if symbol_config.get('spread_bps') in [None, 0, 1, 5]:
                symbol_config['spread_bps'] = round(metrics.median_spread_bps, 1)

        # Add metadata
        config['calculated_params_metadata'] = {
            'last_updated': datetime.now().isoformat(),
            'data_source': 'OHLCV bar data',
            'notes': 'Spread estimated from High-Low range. Beta calculated vs SPY.'
        }

        return config

    def save_config(self, config: Dict):
        """Save updated configuration."""
        # Backup existing
        if self.config_path.exists():
            backup_path = self.config_path.with_suffix('.yaml.backup')
            self.config_path.rename(backup_path)
            logger.info(f"Backed up existing config to {backup_path}")

        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved updated config to {self.config_path}")

    def export_summary(
        self,
        output_path: str = "config/symbol_metrics_summary.json"
    ):
        """
        Export detailed metrics summary.
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'symbols': {
                symbol: asdict(metrics)
                for symbol, metrics in self.metrics.items()
            },
            'aggregate_stats': {}
        }

        if self.metrics:
            # Calculate aggregate statistics
            spreads = [m.avg_spread_bps for m in self.metrics.values()]
            betas = [m.beta_to_spy for m in self.metrics.values()]
            volumes = [m.avg_daily_dollar_volume for m in self.metrics.values()]

            summary['aggregate_stats'] = {
                'avg_spread_bps': round(np.mean(spreads), 2),
                'median_spread_bps': round(np.median(spreads), 2),
                'avg_beta': round(np.mean(betas), 3),
                'avg_daily_dollar_volume': int(np.mean(volumes)),
                'volatility_distribution': {
                    'low': sum(1 for m in self.metrics.values() if m.volatility_group == 'low'),
                    'medium': sum(1 for m in self.metrics.values() if m.volatility_group == 'medium'),
                    'high': sum(1 for m in self.metrics.values() if m.volatility_group == 'high')
                },
                'liquidity_distribution': {
                    'low': sum(1 for m in self.metrics.values() if m.liquidity_group == 'low'),
                    'medium': sum(1 for m in self.metrics.values() if m.liquidity_group == 'medium'),
                    'high': sum(1 for m in self.metrics.values() if m.liquidity_group == 'high')
                }
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported summary to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and update symbol-specific parameters"
    )
    parser.add_argument(
        '--calculate',
        action='store_true',
        help='Calculate parameters for all symbols'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update symbols.yaml with calculated values'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate current configuration'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Analyze specific symbol'
    )

    args = parser.parse_args()

    if args.calculate or args.symbol:
        logger.info("=" * 60)
        logger.info("SYMBOL PARAMETER CALCULATION")
        logger.info("=" * 60)

        analyzer = SymbolAnalyzer()

        if args.symbol:
            # Single symbol
            metrics = analyzer.analyze_symbol(args.symbol)

            print(f"\n{args.symbol} Metrics:")
            print("-" * 60)
            print(f"  Average Price: ${metrics.avg_price:.2f}")
            print(f"  Spread: {metrics.avg_spread_bps:.1f} bps (median: {metrics.median_spread_bps:.1f})")
            print(f"  Daily Volume: {metrics.avg_daily_volume:,.0f} shares")
            print(f"  Daily $ Volume: ${metrics.avg_daily_dollar_volume:,.0f}")
            print(f"  Beta to SPY: {metrics.beta_to_spy:.3f}")
            print(f"  Correlation to SPY: {metrics.correlation_to_spy:.3f}")
            print(f"  Annualized Volatility: {metrics.annualized_volatility*100:.1f}%")
            print(f"  Volatility Group: {metrics.volatility_group}")
            print(f"  Liquidity Group: {metrics.liquidity_group}")
        else:
            # All symbols
            all_metrics = analyzer.analyze_all_symbols()

            print(f"\nCalculated metrics for {len(all_metrics)} symbols:")
            print("-" * 90)
            print(f"{'Symbol':<8} {'Price':<10} {'Spread':<10} {'Volume':<15} {'Beta':<8} {'Vol%':<8} {'Groups'}")
            print("-" * 90)

            for symbol, m in sorted(all_metrics.items()):
                print(
                    f"{symbol:<8} "
                    f"${m.avg_price:<9.2f} "
                    f"{m.median_spread_bps:<10.1f} "
                    f"{m.avg_daily_volume/1e6:<15.2f}M "
                    f"{m.beta_to_spy:<8.2f} "
                    f"{m.annualized_volatility*100:<8.1f} "
                    f"{m.volatility_group}/{m.liquidity_group}"
                )

            # Export summary
            updater = SymbolConfigUpdater(metrics=all_metrics)
            updater.export_summary()

    elif args.update:
        logger.info("=" * 60)
        logger.info("UPDATING SYMBOLS.YAML")
        logger.info("=" * 60)

        # Calculate all metrics first
        analyzer = SymbolAnalyzer()
        all_metrics = analyzer.analyze_all_symbols()

        # Update config
        updater = SymbolConfigUpdater(metrics=all_metrics)
        config = updater.update_config()
        updater.save_config(config)
        updater.export_summary()

        print(f"\nUpdated symbols.yaml with {len(all_metrics)} symbol parameters")

    elif args.validate:
        logger.info("=" * 60)
        logger.info("CONFIGURATION VALIDATION")
        logger.info("=" * 60)

        config_path = Path("config/symbols.yaml")
        if not config_path.exists():
            print("ERROR: symbols.yaml not found")
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        symbols = config.get('symbols', {})

        print(f"\nValidating {len(symbols)} symbols:")
        print("-" * 60)

        issues = []
        for symbol, sym_config in symbols.items():
            symbol_issues = []

            # Check for placeholder values
            spread = sym_config.get('spread_bps')
            if spread in [None, 0, 1]:
                symbol_issues.append(f"placeholder spread ({spread})")

            # Check for calculated params
            if 'calculated_params' not in sym_config:
                symbol_issues.append("missing calculated_params")

            if symbol_issues:
                issues.append((symbol, symbol_issues))
                print(f"  {symbol}: {', '.join(symbol_issues)}")

        if not issues:
            print("  All symbols validated successfully!")
        else:
            print(f"\n{len(issues)} symbols have issues. Run --update to fix.")

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START:")
        print("=" * 60)
        print("1. Calculate for all:     python scripts/update_symbol_params.py --calculate")
        print("2. Analyze one symbol:    python scripts/update_symbol_params.py --symbol AAPL")
        print("3. Update config:         python scripts/update_symbol_params.py --update")
        print("4. Validate config:       python scripts/update_symbol_params.py --validate")


if __name__ == "__main__":
    main()
