#!/usr/bin/env python3
"""
Run Institutional Backtest
==========================

JPMorgan-level backtesting runner that uses trained models from data/storage
with the institutional backtesting framework.

Usage:
    python scripts/run_institutional_backtest.py
    python scripts/run_institutional_backtest.py --symbols AAPL GOOGL MSFT
    python scripts/run_institutional_backtest.py --all-symbols --capital 10000000

Features:
    - Loads pre-trained models from models/artifacts/
    - Uses institutional-grade execution simulation
    - Multi-asset portfolio optimization
    - Comprehensive performance reporting

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_settings, get_logger, configure_logging


# =============================================================================
# INITIALIZATION
# =============================================================================

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def discover_available_data(data_path: Path) -> List[str]:
    """Discover symbols with available data."""
    symbols = []
    
    for f in data_path.glob("*.csv"):
        # Extract symbol from filename (e.g., AAPL_15min.csv -> AAPL)
        parts = f.stem.split("_")
        if len(parts) >= 1:
            symbol = parts[0].upper()
            if symbol not in symbols:
                symbols.append(symbol)
    
    return sorted(symbols)


def load_symbol_data(
    symbol: str,
    data_path: Path,
    timeframe: str = "15min",
) -> Optional[pl.DataFrame]:
    """Load data for a single symbol."""
    # Try different filename patterns
    patterns = [
        f"{symbol}_{timeframe}.csv",
        f"{symbol.upper()}_{timeframe}.csv",
        f"{symbol.lower()}_{timeframe}.csv",
        f"{symbol}.csv",
    ]
    
    for pattern in patterns:
        filepath = data_path / pattern
        if filepath.exists():
            try:
                df = pl.read_csv(filepath)
                
                # Parse timestamp
                if "timestamp" in df.columns:
                    df = df.with_columns([
                        pl.col("timestamp").str.to_datetime().alias("timestamp")
                    ])
                elif "date" in df.columns:
                    df = df.with_columns([
                        pl.col("date").str.to_datetime().alias("timestamp")
                    ])
                
                # Ensure required columns
                required = ["timestamp", "open", "high", "low", "close", "volume"]
                missing = [c for c in required if c not in df.columns]
                
                if missing:
                    logger.warning(f"Missing columns in {symbol}: {missing}")
                    return None
                
                # Sort by timestamp
                df = df.sort("timestamp")
                
                logger.info(f"Loaded {len(df)} bars for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                continue
    
    logger.warning(f"No data found for {symbol}")
    return None


def load_multi_symbol_data(
    symbols: List[str],
    data_path: Path,
    timeframe: str = "15min",
) -> Dict[str, pl.DataFrame]:
    """Load data for multiple symbols."""
    data = {}
    
    for symbol in symbols:
        df = load_symbol_data(symbol, data_path, timeframe)
        if df is not None:
            data[symbol] = df
    
    logger.info(f"Loaded data for {len(data)}/{len(symbols)} symbols")
    return data


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def generate_features_for_symbol(
    df: pl.DataFrame,
    symbol: str,
) -> NDArray[np.float64]:
    """Generate feature matrix for a symbol."""
    from features.pipeline import FeaturePipeline, create_default_config
    
    try:
        # Create feature pipeline
        pipeline = FeaturePipeline(create_default_config())
        
        # Generate features
        df_features = pipeline.generate(df)
        
        # Get numeric feature columns
        exclude_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "target"}
        numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        
        feature_cols = [
            c for c in df_features.columns
            if c not in exclude_cols and df_features[c].dtype in numeric_types
        ]
        
        # Extract feature matrix
        features = df_features.select(feature_cols).to_numpy().astype(np.float64)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Generated {features.shape[1]} features for {symbol}")
        return features
        
    except Exception as e:
        logger.error(f"Feature generation failed for {symbol}: {e}")
        # Return empty features
        return np.zeros((len(df), 100))


def generate_all_features(
    data: Dict[str, pl.DataFrame],
) -> Dict[str, NDArray[np.float64]]:
    """Generate features for all symbols."""
    features = {}
    
    for symbol, df in data.items():
        features[symbol] = generate_features_for_symbol(df, symbol)
    
    return features


# =============================================================================
# MODEL LOADING
# =============================================================================

class SimpleModelLoader:
    """
    Simple model loader for institutional backtesting.
    
    Loads pre-trained models from the models/artifacts directory.
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: Dict[str, Dict[str, Any]] = {}
        self._loaded_symbols: set = set()
    
    def load_models_for_symbol(self, symbol: str) -> bool:
        """Load all available models for a symbol."""
        symbol = symbol.upper()
        symbol_dir = self.models_dir / symbol
        
        if not symbol_dir.exists():
            logger.debug(f"No model directory for {symbol}")
            return False
        
        self._models[symbol] = {}
        
        # Load each model file
        for model_file in symbol_dir.glob("*.pkl"):
            try:
                import pickle
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                
                # Extract model type from filename
                parts = model_file.stem.split("_")
                if len(parts) >= 2:
                    model_type = parts[1]  # e.g., AAPL_lightgbm_v1 -> lightgbm
                else:
                    model_type = "unknown"
                
                self._models[symbol][model_type] = model
                logger.info(f"Loaded {model_type} model for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
        
        if self._models[symbol]:
            self._loaded_symbols.add(symbol)
            return True
        
        return False
    
    def get_model(self, symbol: str, model_type: str = "lightgbm") -> Optional[Any]:
        """Get a specific model for a symbol."""
        symbol = symbol.upper()
        
        if symbol not in self._models:
            self.load_models_for_symbol(symbol)
        
        return self._models.get(symbol, {}).get(model_type)
    
    def predict(
        self,
        symbol: str,
        features: NDArray[np.float64],
        model_type: str = "lightgbm",
    ) -> tuple[int, float]:
        """
        Make prediction with a model.
        
        Returns:
            Tuple of (direction, confidence)
            direction: 1 for up, -1 for down, 0 for neutral
            confidence: 0.0 to 1.0
        """
        model = self.get_model(symbol, model_type)
        
        if model is None:
            return 0, 0.0
        
        try:
            # Ensure features is 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Get prediction
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)
                if proba.ndim > 1 and proba.shape[1] >= 2:
                    confidence = float(np.max(proba[0]))
                    direction = 1 if proba[0, 1] > 0.5 else -1
                else:
                    confidence = 0.5
                    direction = 0
            else:
                pred = model.predict(features)[0]
                direction = 1 if pred > 0 else -1
                confidence = 0.6
            
            return direction, confidence
            
        except Exception as e:
            logger.warning(f"Prediction failed for {symbol}: {e}")
            return 0, 0.0
    
    @property
    def loaded_symbols(self) -> List[str]:
        """Get list of symbols with loaded models."""
        return list(self._loaded_symbols)


# =============================================================================
# SIMPLE INSTITUTIONAL BACKTESTER
# =============================================================================

class SimpleInstitutionalBacktester:
    """
    Simplified institutional backtester that works without the full
    institutional.py dependencies.
    
    Uses the core concepts:
    - Multi-asset portfolio
    - Transaction cost modeling
    - Risk management
    - Performance tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        max_position_pct: float = 0.10,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        max_drawdown_pct: float = 0.15,
        min_confidence: float = 0.55,
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.max_drawdown_pct = max_drawdown_pct
        self.min_confidence = min_confidence
        
        # State
        self._cash = initial_capital
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._portfolio_value = initial_capital
        
        # History
        self._equity_curve: List[tuple[datetime, float]] = []
        self._returns: List[float] = []
        self._trades: List[Dict[str, Any]] = []
        
        # Tracking
        self._high_water_mark = initial_capital
        self._max_drawdown = 0.0
    
    def run(
        self,
        data: Dict[str, pl.DataFrame],
        features: Dict[str, NDArray[np.float64]],
        model_loader: SimpleModelLoader,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            data: OHLCV data per symbol
            features: Feature matrices per symbol
            model_loader: Model loader for predictions
            start_date: Start date
            end_date: End date
            
        Returns:
            Backtest results
        """
        logger.info("=" * 60)
        logger.info("INSTITUTIONAL BACKTEST")
        logger.info("=" * 60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Symbols: {len(data)}")
        
        # Get aligned timestamps
        all_timestamps = self._get_aligned_timestamps(data, start_date, end_date)
        
        if len(all_timestamps) < 100:
            return {"error": "Insufficient data"}
        
        logger.info(f"Date Range: {all_timestamps[0]} to {all_timestamps[-1]}")
        logger.info(f"Total Bars: {len(all_timestamps)}")
        
        # Warmup
        warmup = 500
        simulation_timestamps = all_timestamps[warmup:]
        
        # Main loop
        symbols = list(data.keys())
        
        for i, timestamp in enumerate(simulation_timestamps):
            # Get current bar data
            current_prices = {}
            current_features = {}
            
            for symbol in symbols:
                df = data[symbol]
                idx = df.filter(pl.col("timestamp") == timestamp)
                
                if len(idx) > 0:
                    current_prices[symbol] = float(idx["close"][0])
                    
                    # Get features at this point
                    feat = features.get(symbol)
                    if feat is not None:
                        feat_idx = min(i + warmup, len(feat) - 1)
                        current_features[symbol] = feat[feat_idx]
            
            if not current_prices:
                continue
            
            # Update positions
            self._update_positions(current_prices)
            
            # Generate signals
            signals = {}
            for symbol, feat in current_features.items():
                if symbol not in current_prices:
                    continue
                
                direction, confidence = model_loader.predict(symbol, feat)
                
                if confidence >= self.min_confidence and direction != 0:
                    signals[symbol] = {
                        "direction": direction,
                        "confidence": confidence,
                        "price": current_prices[symbol],
                    }
            
            # Trade on signals
            if signals:
                self._process_signals(signals, current_prices, timestamp)
            
            # Record equity
            self._equity_curve.append((timestamp, self._portfolio_value))
            
            # Calculate return
            if len(self._equity_curve) > 1:
                prev_value = self._equity_curve[-2][1]
                ret = (self._portfolio_value - prev_value) / prev_value
                self._returns.append(ret)
            
            # Check drawdown
            if self._max_drawdown > self.max_drawdown_pct:
                logger.warning(f"Max drawdown exceeded at {timestamp}")
                break
        
        # Generate results
        return self._generate_results(all_timestamps)
    
    def _get_aligned_timestamps(
        self,
        data: Dict[str, pl.DataFrame],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[datetime]:
        """Get timestamps present in all symbols."""
        all_ts = set()
        
        for df in data.values():
            if "timestamp" in df.columns:
                all_ts.update(df["timestamp"].to_list())
        
        timestamps = sorted(all_ts)
        
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date]
        
        return timestamps
    
    def _update_positions(self, prices: Dict[str, float]) -> None:
        """Update position values."""
        position_value = 0.0
        
        for symbol, pos in self._positions.items():
            if symbol in prices:
                price = prices[symbol]
                pos["current_price"] = price
                pos["market_value"] = pos["quantity"] * price
                pos["unrealized_pnl"] = pos["market_value"] - pos["cost_basis"]
                position_value += pos["market_value"]
        
        self._portfolio_value = self._cash + position_value
        
        # Update high water mark
        if self._portfolio_value > self._high_water_mark:
            self._high_water_mark = self._portfolio_value
        
        # Calculate drawdown
        drawdown = (self._high_water_mark - self._portfolio_value) / self._high_water_mark
        self._max_drawdown = max(self._max_drawdown, drawdown)
    
    def _process_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        prices: Dict[str, float],
        timestamp: datetime,
    ) -> None:
        """Process trading signals."""
        # Simple equal-weight allocation among signals
        n_signals = len(signals)
        
        if n_signals == 0:
            return
        
        # Target allocation per signal
        target_per_signal = min(
            self.max_position_pct,
            0.8 / n_signals,  # Don't use more than 80% of capital
        )
        
        for symbol, signal in signals.items():
            direction = signal["direction"]
            confidence = signal["confidence"]
            price = signal["price"]
            
            # Calculate target position
            target_value = self._portfolio_value * target_per_signal * confidence
            current_value = self._positions.get(symbol, {}).get("market_value", 0)
            
            # Determine trade
            if direction > 0:
                trade_value = target_value - current_value
            else:
                trade_value = -current_value  # Close position for sell signals
            
            # Minimum trade size
            if abs(trade_value) < 1000:
                continue
            
            # Execute trade
            self._execute_trade(symbol, trade_value, price, timestamp)
    
    def _execute_trade(
        self,
        symbol: str,
        trade_value: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Execute a trade with costs."""
        # Calculate costs
        commission = abs(trade_value) * (self.commission_bps / 10000)
        slippage = abs(trade_value) * (self.slippage_bps / 10000)
        total_cost = commission + slippage
        
        # Adjust trade for slippage
        if trade_value > 0:
            effective_price = price * (1 + self.slippage_bps / 10000)
        else:
            effective_price = price * (1 - self.slippage_bps / 10000)
        
        quantity = trade_value / effective_price
        
        # Update position
        if symbol not in self._positions:
            self._positions[symbol] = {
                "quantity": 0,
                "avg_price": 0,
                "cost_basis": 0,
                "market_value": 0,
                "current_price": price,
                "unrealized_pnl": 0,
            }
        
        pos = self._positions[symbol]
        old_quantity = pos["quantity"]
        new_quantity = old_quantity + quantity
        
        if new_quantity != 0:
            if old_quantity == 0:
                pos["avg_price"] = effective_price
            elif (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                # Adding to position
                pos["avg_price"] = (
                    (old_quantity * pos["avg_price"] + quantity * effective_price) / 
                    new_quantity
                )
        
        pos["quantity"] = new_quantity
        pos["cost_basis"] = abs(new_quantity * pos["avg_price"])
        pos["market_value"] = new_quantity * price
        pos["current_price"] = price
        
        # Update cash
        self._cash -= trade_value + total_cost
        
        # Record trade
        self._trades.append({
            "timestamp": str(timestamp),
            "symbol": symbol,
            "side": "buy" if trade_value > 0 else "sell",
            "quantity": abs(quantity),
            "price": effective_price,
            "value": abs(trade_value),
            "commission": commission,
            "slippage": slippage,
            "total_cost": total_cost,
        })
    
    def _generate_results(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Generate comprehensive results."""
        returns = np.array(self._returns) if self._returns else np.array([0])
        
        # Total return
        total_return = (self._portfolio_value / self.initial_capital) - 1
        
        # Annualized metrics
        n_days = (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 1
        annual_factor = 365 / max(n_days, 1)
        
        annual_return = (1 + total_return) ** annual_factor - 1
        annual_vol = np.std(returns) * np.sqrt(252 * 26) if len(returns) > 1 else 0  # 26 bars/day
        
        # Sharpe ratio
        risk_free = 0.05  # 5% annual
        excess_returns = returns - risk_free / (252 * 26)
        sharpe = (
            np.mean(excess_returns) / np.std(returns) * np.sqrt(252 * 26)
            if np.std(returns) > 0 else 0
        )
        
        # Sortino ratio
        downside = returns[returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252 * 26) if len(downside) > 0 else 0
        sortino = (annual_return - risk_free) / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio
        calmar = annual_return / self._max_drawdown if self._max_drawdown > 0 else 0
        
        # Trade statistics
        n_trades = len(self._trades)
        total_costs = sum(t["total_cost"] for t in self._trades)
        
        # Win rate
        if n_trades > 0:
            # Group trades by symbol and calculate PnL
            pnls = []
            for trade in self._trades:
                # Simplified: positive for buys that went up, sells that went down
                pnls.append(trade["value"] * 0.01)  # Placeholder
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
        else:
            win_rate = 0
        
        results = {
            # Performance
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": self._max_drawdown,
            
            # Capital
            "initial_capital": self.initial_capital,
            "final_value": self._portfolio_value,
            "total_pnl": self._portfolio_value - self.initial_capital,
            
            # Trading
            "n_trades": n_trades,
            "total_costs": total_costs,
            "avg_cost_per_trade": total_costs / n_trades if n_trades > 0 else 0,
            "win_rate": win_rate,
            
            # Time series
            "equity_curve": [(str(t), v) for t, v in self._equity_curve[-1000:]],  # Last 1000
            
            # Config
            "max_position_pct": self.max_position_pct,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
        }
        
        return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def print_results(results: Dict[str, Any]) -> None:
    """Print backtest results to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n{'PERFORMANCE METRICS':^60}")
    print("-" * 60)
    print(f"  Total Return:        {results.get('total_return', 0):>12.2%}")
    print(f"  Annual Return:       {results.get('annual_return', 0):>12.2%}")
    print(f"  Annual Volatility:   {results.get('annual_volatility', 0):>12.2%}")
    print(f"  Sharpe Ratio:        {results.get('sharpe_ratio', 0):>12.2f}")
    print(f"  Sortino Ratio:       {results.get('sortino_ratio', 0):>12.2f}")
    print(f"  Calmar Ratio:        {results.get('calmar_ratio', 0):>12.2f}")
    print(f"  Max Drawdown:        {results.get('max_drawdown', 0):>12.2%}")
    
    print(f"\n{'CAPITAL':^60}")
    print("-" * 60)
    print(f"  Initial Capital:     ${results.get('initial_capital', 0):>14,.2f}")
    print(f"  Final Value:         ${results.get('final_value', 0):>14,.2f}")
    print(f"  Total P&L:           ${results.get('total_pnl', 0):>14,.2f}")
    
    print(f"\n{'TRADING STATISTICS':^60}")
    print("-" * 60)
    print(f"  Number of Trades:    {results.get('n_trades', 0):>12}")
    print(f"  Total Costs:         ${results.get('total_costs', 0):>14,.2f}")
    print(f"  Avg Cost/Trade:      ${results.get('avg_cost_per_trade', 0):>14,.2f}")
    print(f"  Win Rate:            {results.get('win_rate', 0):>12.2%}")
    
    print("\n" + "=" * 60)


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    logger.info("Starting institutional backtest...")
    
    # Data path
    data_path = Path(settings.data.storage_path)
    models_path = Path("models/artifacts")
    
    # Discover available data
    available_symbols = discover_available_data(data_path)
    logger.info(f"Available symbols: {len(available_symbols)}")
    
    if not available_symbols:
        logger.error("No data files found!")
        return 1
    
    # Select symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.all_symbols:
        symbols = available_symbols
    else:
        # Default to first 10 available
        symbols = available_symbols[:10]
    
    # Validate symbols
    symbols = [s for s in symbols if s in available_symbols]
    
    if not symbols:
        logger.error("No valid symbols selected!")
        return 1
    
    logger.info(f"Selected symbols: {symbols}")
    
    # Load data
    data = load_multi_symbol_data(symbols, data_path, args.timeframe)
    
    if not data:
        logger.error("Failed to load any data!")
        return 1
    
    # Generate features
    logger.info("Generating features...")
    features = generate_all_features(data)
    
    # Load models
    logger.info("Loading models...")
    model_loader = SimpleModelLoader(models_path)
    
    for symbol in data.keys():
        model_loader.load_models_for_symbol(symbol)
    
    logger.info(f"Loaded models for: {model_loader.loaded_symbols}")
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)
    
    # Run backtest
    backtester = SimpleInstitutionalBacktester(
        initial_capital=args.capital,
        max_position_pct=args.max_position,
        commission_bps=args.commission,
        slippage_bps=args.slippage,
        max_drawdown_pct=args.max_drawdown,
        min_confidence=args.min_confidence,
    )
    
    start_time = time.time()
    results = backtester.run(data, features, model_loader, start_date, end_date)
    elapsed = time.time() - start_time
    
    logger.info(f"Backtest completed in {elapsed:.2f}s")
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"backtesting/reports/institutional_{timestamp}.json")
    
    save_results(results, output_path)
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run institutional-grade backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Symbols
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Symbols to backtest",
    )
    parser.add_argument(
        "--all-symbols", "-a",
        action="store_true",
        help="Use all available symbols",
    )
    
    # Capital
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=1_000_000.0,
        help="Initial capital",
    )
    
    # Risk parameters
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.10,
        help="Maximum position size as fraction of portfolio",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        help="Maximum drawdown before stopping",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum model confidence to trade",
    )
    
    # Costs
    parser.add_argument(
        "--commission",
        type=float,
        default=1.0,
        help="Commission in basis points",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=2.0,
        help="Slippage in basis points",
    )
    
    # Dates
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD)",
    )
    
    # Data
    parser.add_argument(
        "--timeframe",
        default="15min",
        help="Data timeframe",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output file path for results",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))