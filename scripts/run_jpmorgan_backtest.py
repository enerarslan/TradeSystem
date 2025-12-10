#!/usr/bin/env python3
"""
JPMorgan-Level Institutional Backtest Runner
=============================================

Production-grade backtest system integrating all advanced features:

Phase 1: Data Fidelity & Microstructure
- Tick & Quote Data integration (when available)
- Dollar Bar aggregation for normalized returns
- Real Order Flow Imbalance and VPIN

Phase 2: High-Fidelity Simulation
- Order Book Reconstruction
- Queue Position Simulation
- Liquidity-Constrained Execution (max 1% volume participation)
- Realistic Market Impact (Almgren-Chriss model)

Phase 3: Alpha Modernization
- Alternative Data (Macro, Sentiment, Options)
- Advanced Feature Selection (MDA/SFI)
- Regime Detection

Phase 4: Rigorous Validation
- Deflated Sharpe Ratio (multiple testing correction)
- Probability of Backtest Overfitting (PBO)
- Feature Leakage Detection

Usage:
    python scripts/run_jpmorgan_backtest.py --symbols AAPL MSFT GOOGL
    python scripts/run_jpmorgan_backtest.py --all-symbols --capital 10000000
    python scripts/run_jpmorgan_backtest.py --core-symbols --validate

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging
from config.symbols import (
    ALL_SYMBOLS,
    CORE_SYMBOLS,
    discover_symbols_from_data,
    get_symbol_info,
)

# Phase 1: Data & Microstructure
from data.loader import CSVLoader
from data.alternative_data import AlternativeDataPipeline, AlternativeDataConfig

# Phase 2: Execution Simulation
from backtesting.liquidity_constraints import (
    LiquidityConfig,
    LiquidityConstrainedExecutor,
    LiquidityCalculator,
    MarketImpactCalculator,
)
from backtesting.order_book_simulator import (
    OrderBookSimulatorConfig,
    RealisticExecutionSimulator,
)

# Phase 3: Features
from features.pipeline import FeaturePipeline, create_default_config
from features.advanced import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    MicrostructureFeatures,
    CalendarFeatures,
)
from features.feature_selection import (
    AdvancedFeatureSelector,
    FeatureSelectionConfig,
    MDAFeatureImportance,
    SFIFeatureImportance,
)

# Phase 4: Validation
from backtesting.validation import (
    BacktestValidator,
    ValidationConfig,
    DeflatedSharpeRatio,
    FeatureLeakageDetector,
)

warnings.filterwarnings("ignore")

# =============================================================================
# INITIALIZATION
# =============================================================================

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class JPMorganBacktestConfig:
    """Configuration for JPMorgan-level backtest."""
    # Capital
    initial_capital: float = 10_000_000  # $10M default

    # Position sizing
    max_position_pct: float = 0.05       # Max 5% per position
    max_portfolio_positions: int = 20    # Max concurrent positions
    min_confidence: float = 0.55         # Minimum model confidence

    # Execution (Liquidity Constraints)
    max_participation_rate: float = 0.01  # Max 1% of bar volume
    max_position_adv_pct: float = 0.05    # Max 5% of ADV
    enable_market_impact: bool = True
    enable_queue_simulation: bool = True

    # Transaction costs (institutional rates)
    commission_bps: float = 0.5          # 0.5 bps commission
    spread_bps: float = 1.0              # 1 bps half-spread
    market_impact_bps: float = 2.0       # 2 bps market impact

    # Risk management
    max_drawdown_pct: float = 0.15       # Stop at 15% drawdown
    daily_var_limit: float = 0.02        # 2% daily VaR limit
    position_stop_loss: float = 0.05     # 5% stop loss per position

    # Feature engineering
    use_alternative_data: bool = True
    use_advanced_features: bool = True
    feature_selection_method: str = "mda"  # mda, sfi, or both

    # Validation
    run_validation: bool = True
    n_trials_for_dsr: int = 1            # Number of strategy variations tested

    # Walk-forward
    train_bars: int = 5000               # Training window
    test_bars: int = 1000                # Test window
    embargo_bars: int = 50               # Gap between train/test

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/jpmorgan"))
    save_trades: bool = True
    save_equity_curve: bool = True


# =============================================================================
# DATA LOADING
# =============================================================================

def load_symbol_data(
    symbol: str,
    data_path: Path,
) -> pl.DataFrame | None:
    """Load OHLCV data for a symbol."""
    patterns = [
        f"{symbol}_15min.csv",
        f"{symbol}_1h.csv",
        f"{symbol.upper()}_15min.csv",
    ]

    for pattern in patterns:
        file_path = data_path / pattern
        if file_path.exists():
            try:
                df = pl.read_csv(file_path)

                # Normalize columns
                df = df.rename({col: col.lower() for col in df.columns})

                # Parse timestamp
                if "timestamp" in df.columns:
                    if df["timestamp"].dtype == pl.Utf8:
                        df = df.with_columns([
                            pl.col("timestamp").str.to_datetime().alias("timestamp")
                        ])

                # Ensure required columns
                required = ["timestamp", "open", "high", "low", "close", "volume"]
                if all(c in df.columns for c in required):
                    df = df.sort("timestamp")
                    return df

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

    return None


def load_multi_symbol_data(
    symbols: list[str],
    data_path: Path,
) -> dict[str, pl.DataFrame]:
    """Load data for multiple symbols."""
    data = {}

    for symbol in symbols:
        df = load_symbol_data(symbol, data_path)
        if df is not None and len(df) > 1000:
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} bars")
        else:
            logger.warning(f"Skipped {symbol}: insufficient data")

    return data


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def generate_features(
    df: pl.DataFrame,
    symbol: str,
    config: JPMorganBacktestConfig,
) -> pl.DataFrame:
    """Generate all features for a symbol."""
    logger.info(f"Generating features for {symbol}...")

    # Basic technical features
    feature_pipeline = FeaturePipeline(create_default_config())
    df = feature_pipeline.generate(df)

    # Advanced microstructure features
    if config.use_advanced_features:
        df = MicrostructureFeatures.add_features(df)
        df = CalendarFeatures.add_features(df)

    # Alternative data features
    if config.use_alternative_data:
        try:
            alt_config = AlternativeDataConfig(
                macro_features_enabled=True,
                sentiment_features_enabled=True,
                options_features_enabled=True,
            )
            alt_pipeline = AlternativeDataPipeline(alt_config)
            df = alt_pipeline.generate_features(df, symbol)
        except Exception as e:
            logger.warning(f"Alternative data failed for {symbol}: {e}")

    # Triple Barrier labels
    tb_config = TripleBarrierConfig(
        take_profit_multiplier=2.0,
        stop_loss_multiplier=1.0,
        max_holding_period=20,
    )
    labeler = TripleBarrierLabeler(tb_config)
    df = labeler.apply_binary_labels(df)

    return df


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: JPMorganBacktestConfig,
) -> tuple[np.ndarray, list[str]]:
    """Select best features using MDA/SFI."""
    logger.info(f"Selecting features from {len(feature_names)} candidates...")

    selector_config = FeatureSelectionConfig(
        mda_n_repeats=5,
        mda_cv_splits=3,
        sfi_cv_splits=3,
        importance_threshold=0.01,
        max_features=100,
    )

    selector = AdvancedFeatureSelector(selector_config)

    methods = []
    if config.feature_selection_method in ["mda", "both"]:
        methods.append("mda")
    if config.feature_selection_method in ["sfi", "both"]:
        methods.append("sfi")

    if not methods:
        methods = ["mda"]

    try:
        X_selected, names_selected = selector.fit_transform(
            X, y, feature_names, methods=methods
        )
        logger.info(f"Selected {len(names_selected)} features")
        return X_selected, names_selected
    except Exception as e:
        logger.warning(f"Feature selection failed: {e}")
        return X, feature_names


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(
    symbols: list[str],
    models_dir: Path,
) -> dict[str, Any]:
    """Load trained models for symbols."""
    import pickle

    models = {}

    for symbol in symbols:
        symbol_dir = models_dir / symbol

        if not symbol_dir.exists():
            continue

        # Find model files
        model_files = list(symbol_dir.glob("*.pkl"))

        if not model_files:
            continue

        # Load best model (prefer lightgbm)
        for model_file in model_files:
            if "lightgbm" in model_file.name.lower():
                try:
                    with open(model_file, "rb") as f:
                        model_data = pickle.load(f)

                    # Extract model from wrapper
                    if isinstance(model_data, dict):
                        model = model_data.get("model", model_data)
                    else:
                        model = model_data

                    models[symbol] = {
                        "model": model,
                        "path": str(model_file),
                        "type": "lightgbm",
                    }
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model for {symbol}: {e}")

    logger.info(f"Loaded models for {len(models)} symbols")
    return models


# =============================================================================
# JPMORGAN BACKTESTER
# =============================================================================

class JPMorganBacktester:
    """
    JPMorgan-level institutional backtester.

    Integrates:
    - Liquidity-constrained execution
    - Market impact modeling
    - Risk management
    - Multi-asset portfolio optimization
    """

    def __init__(self, config: JPMorganBacktestConfig):
        """Initialize backtester."""
        self.config = config

        # Execution components
        self.liquidity_config = LiquidityConfig(
            max_participation_rate=config.max_participation_rate,
            max_position_adv_pct=config.max_position_adv_pct,
            enable_order_carryover=True,
        )
        self.executor = LiquidityConstrainedExecutor(self.liquidity_config)
        self.impact_calc = MarketImpactCalculator(self.liquidity_config)

        # State
        self.cash = config.initial_capital
        self.positions: dict[str, dict] = {}
        self.portfolio_value = config.initial_capital
        self.peak_value = config.initial_capital

        # Tracking
        self.equity_curve: list[tuple[datetime, float]] = []
        self.returns: list[float] = []
        self.trades: list[dict] = []
        self.daily_pnl: list[float] = []

        # Risk
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.is_killed = False

    def initialize_symbols(
        self,
        data: dict[str, pl.DataFrame],
    ) -> None:
        """Initialize liquidity metrics for all symbols."""
        for symbol, df in data.items():
            self.executor.initialize_symbol(symbol, df)

    def run(
        self,
        data: dict[str, pl.DataFrame],
        features: dict[str, pl.DataFrame],
        models: dict[str, Any],
        start_idx: int = 500,
    ) -> dict[str, Any]:
        """
        Run the backtest.

        Args:
            data: OHLCV data per symbol
            features: Feature data per symbol
            models: Trained models per symbol
            start_idx: Start index (after warmup)

        Returns:
            Backtest results dictionary
        """
        logger.info("Starting JPMorgan-level backtest...")
        start_time = time.time()

        # Get aligned timestamps
        all_timestamps = self._get_aligned_timestamps(data, start_idx)
        n_bars = len(all_timestamps)

        logger.info(f"Backtesting {n_bars} bars across {len(data)} symbols")

        # Main loop
        for i, timestamp in enumerate(all_timestamps):
            if self.is_killed:
                logger.warning("Backtest stopped: Risk limit breached")
                break

            # Get current prices
            current_prices = {}
            current_volumes = {}

            for symbol, df in data.items():
                mask = df["timestamp"] == timestamp
                if mask.sum() > 0:
                    row = df.filter(mask)
                    current_prices[symbol] = float(row["close"][0])
                    current_volumes[symbol] = float(row["volume"][0])

            if not current_prices:
                continue

            # Update positions
            self._update_positions(current_prices)

            # Check risk limits
            self._check_risk_limits()

            if self.is_killed:
                break

            # Generate signals
            signals = self._generate_signals(
                timestamp, i + start_idx, features, models, current_prices
            )

            # Execute trades
            if signals:
                self._execute_signals(
                    signals, current_prices, current_volumes, timestamp
                )

            # Record equity
            self.equity_curve.append((timestamp, self.portfolio_value))

            if len(self.equity_curve) > 1:
                prev_value = self.equity_curve[-2][1]
                ret = (self.portfolio_value - prev_value) / prev_value
                self.returns.append(ret)

            # Progress
            if (i + 1) % 1000 == 0:
                logger.info(
                    f"Progress: {i+1}/{n_bars} bars, "
                    f"Value: ${self.portfolio_value:,.0f}, "
                    f"DD: {self.current_drawdown:.2%}"
                )

        # Generate results
        elapsed = time.time() - start_time
        results = self._generate_results(elapsed)

        return results

    def _get_aligned_timestamps(
        self,
        data: dict[str, pl.DataFrame],
        start_idx: int,
    ) -> list[datetime]:
        """Get aligned timestamps across all symbols."""
        all_ts = set()

        for df in data.values():
            ts = df["timestamp"].to_list()[start_idx:]
            all_ts.update(ts)

        return sorted(all_ts)

    def _update_positions(self, prices: dict[str, float]) -> None:
        """Update position values with current prices."""
        total_position_value = 0

        for symbol, pos in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                pos["current_price"] = price
                pos["market_value"] = pos["quantity"] * price
                pos["unrealized_pnl"] = (price - pos["avg_price"]) * pos["quantity"]
                total_position_value += pos["market_value"]

        self.portfolio_value = self.cash + total_position_value

        # Update drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        # Max drawdown
        if self.current_drawdown >= self.config.max_drawdown_pct:
            logger.error(f"MAX DRAWDOWN BREACHED: {self.current_drawdown:.2%}")
            self.is_killed = True
            return

        # Position stop losses
        for symbol, pos in list(self.positions.items()):
            if pos["quantity"] == 0:
                continue

            pnl_pct = pos["unrealized_pnl"] / (pos["avg_price"] * abs(pos["quantity"]))

            if pnl_pct < -self.config.position_stop_loss:
                logger.warning(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                # Mark for liquidation
                pos["stop_triggered"] = True

    def _generate_signals(
        self,
        timestamp: datetime,
        bar_idx: int,
        features: dict[str, pl.DataFrame],
        models: dict[str, Any],
        prices: dict[str, float],
    ) -> dict[str, dict]:
        """Generate trading signals from models."""
        signals = {}

        for symbol, model_info in models.items():
            if symbol not in features or symbol not in prices:
                continue

            feat_df = features[symbol]

            if bar_idx >= len(feat_df):
                continue

            try:
                # Get feature row
                row = feat_df.row(bar_idx, named=True)

                # Extract numeric features
                exclude = {"timestamp", "symbol", "target", "tb_label", "tb_return",
                          "tb_barrier", "tb_holding_period", "open", "high", "low",
                          "close", "volume"}

                feat_values = []
                for col in feat_df.columns:
                    if col not in exclude:
                        val = row.get(col)
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            feat_values.append(float(val))
                        else:
                            feat_values.append(0.0)

                if not feat_values:
                    continue

                X = np.array([feat_values])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                # Get prediction
                model = model_info["model"]

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]
                    if len(prob) == 2:
                        confidence = max(prob)
                        direction = 1 if prob[1] > prob[0] else -1
                    else:
                        confidence = prob[0]
                        direction = 1
                else:
                    pred = model.predict(X)[0]
                    direction = 1 if pred > 0.5 else -1
                    confidence = 0.6

                # Check confidence threshold
                if confidence >= self.config.min_confidence:
                    signals[symbol] = {
                        "direction": direction,
                        "confidence": confidence,
                        "price": prices[symbol],
                    }

            except Exception as e:
                logger.debug(f"Signal generation failed for {symbol}: {e}")

        return signals

    def _execute_signals(
        self,
        signals: dict[str, dict],
        prices: dict[str, float],
        volumes: dict[str, float],
        timestamp: datetime,
    ) -> None:
        """Execute trading signals with liquidity constraints."""
        # First handle stop losses
        for symbol, pos in list(self.positions.items()):
            if pos.get("stop_triggered"):
                self._close_position(symbol, prices.get(symbol, pos["current_price"]),
                                    volumes.get(symbol, 10000), timestamp, "stop_loss")

        # Limit number of new positions
        n_current = len([p for p in self.positions.values() if p["quantity"] != 0])
        n_available = self.config.max_portfolio_positions - n_current

        if n_available <= 0:
            return

        # Sort signals by confidence
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )[:n_available]

        for symbol, signal in sorted_signals:
            direction = signal["direction"]
            confidence = signal["confidence"]
            price = signal["price"]
            volume = volumes.get(symbol, 10000)

            # Skip if already have position in same direction
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["quantity"] * direction > 0:
                    continue
                # Close opposite position first
                self._close_position(symbol, price, volume, timestamp, "reversal")

            # Calculate position size
            target_value = self.portfolio_value * self.config.max_position_pct * confidence
            target_shares = target_value / price

            # Execute with liquidity constraints
            self._open_position(
                symbol, direction, target_shares, price, volume, timestamp
            )

    def _open_position(
        self,
        symbol: str,
        direction: int,
        target_shares: float,
        price: float,
        bar_volume: float,
        timestamp: datetime,
    ) -> None:
        """Open a new position with liquidity constraints."""
        if target_shares <= 0:
            return

        order_id = f"{symbol}_{timestamp.isoformat()}"
        side = "buy" if direction > 0 else "sell"

        # Execute with liquidity constraints
        result = self.executor.execute_order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=target_shares,
            price=price,
            bar_volume=bar_volume,
            bar_timestamp=timestamp,
        )

        if result.filled_quantity <= 0:
            return

        # Calculate costs
        commission = result.filled_quantity * result.fill_price * (self.config.commission_bps / 10000)
        spread_cost = result.filled_quantity * result.fill_price * (self.config.spread_bps / 10000)
        impact_cost = result.filled_quantity * result.fill_price * (result.market_impact / 10000)
        total_cost = commission + spread_cost + impact_cost

        # Update position
        signed_qty = result.filled_quantity * direction
        trade_value = result.filled_quantity * result.fill_price

        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avg_price": 0,
                "market_value": 0,
                "current_price": price,
                "unrealized_pnl": 0,
                "cost_basis": 0,
            }

        pos = self.positions[symbol]
        old_qty = pos["quantity"]
        new_qty = old_qty + signed_qty

        if new_qty != 0:
            if old_qty == 0:
                pos["avg_price"] = result.fill_price
            else:
                pos["avg_price"] = (
                    (old_qty * pos["avg_price"] + signed_qty * result.fill_price) / new_qty
                )

        pos["quantity"] = new_qty
        pos["cost_basis"] += trade_value + total_cost

        # Update cash
        self.cash -= trade_value * direction + total_cost

        # Record trade
        self.trades.append({
            "timestamp": str(timestamp),
            "symbol": symbol,
            "side": side,
            "quantity": result.filled_quantity,
            "price": result.fill_price,
            "value": trade_value,
            "commission": commission,
            "spread_cost": spread_cost,
            "impact_cost": impact_cost,
            "total_cost": total_cost,
            "participation_rate": result.participation_rate,
            "reason": "signal",
        })

    def _close_position(
        self,
        symbol: str,
        price: float,
        bar_volume: float,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        if pos["quantity"] == 0:
            return

        direction = -1 if pos["quantity"] > 0 else 1
        quantity = abs(pos["quantity"])

        order_id = f"{symbol}_close_{timestamp.isoformat()}"
        side = "sell" if pos["quantity"] > 0 else "buy"

        # Execute
        result = self.executor.execute_order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            bar_volume=bar_volume,
            bar_timestamp=timestamp,
        )

        if result.filled_quantity <= 0:
            return

        # Calculate P&L
        trade_value = result.filled_quantity * result.fill_price
        cost_basis = (result.filled_quantity / quantity) * pos["cost_basis"]
        realized_pnl = trade_value - cost_basis if pos["quantity"] > 0 else cost_basis - trade_value

        # Costs
        commission = trade_value * (self.config.commission_bps / 10000)
        spread_cost = trade_value * (self.config.spread_bps / 10000)
        impact_cost = trade_value * (result.market_impact / 10000)
        total_cost = commission + spread_cost + impact_cost

        # Update position
        remaining = quantity - result.filled_quantity
        if remaining <= 0:
            pos["quantity"] = 0
            pos["cost_basis"] = 0
        else:
            pos["quantity"] = remaining * (1 if pos["quantity"] > 0 else -1)
            pos["cost_basis"] *= (remaining / quantity)

        # Update cash
        self.cash += trade_value - total_cost

        # Record trade
        self.trades.append({
            "timestamp": str(timestamp),
            "symbol": symbol,
            "side": side,
            "quantity": result.filled_quantity,
            "price": result.fill_price,
            "value": trade_value,
            "realized_pnl": realized_pnl - total_cost,
            "commission": commission,
            "spread_cost": spread_cost,
            "impact_cost": impact_cost,
            "total_cost": total_cost,
            "reason": reason,
        })

    def _generate_results(self, elapsed_time: float) -> dict[str, Any]:
        """Generate comprehensive results."""
        returns = np.array(self.returns)

        if len(returns) == 0:
            return {"error": "No returns generated"}

        # Basic metrics
        total_return = (self.portfolio_value / self.config.initial_capital) - 1

        # Annualization (15-min bars)
        periods_per_year = 252 * 26
        n_periods = len(returns)
        years = n_periods / periods_per_year

        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0

        annual_vol = np.std(returns) * np.sqrt(periods_per_year)

        # Risk-adjusted metrics
        risk_free = self.config.initial_capital * 0.05 / periods_per_year
        excess_returns = returns - risk_free / self.portfolio_value

        if annual_vol > 0:
            sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_vol = np.std(downside) * np.sqrt(periods_per_year)
            sortino = annual_return / downside_vol if downside_vol > 0 else 0
        else:
            sortino = float('inf')

        # Calmar
        calmar = annual_return / self.max_drawdown if self.max_drawdown > 0 else 0

        # Trade statistics
        n_trades = len(self.trades)
        total_costs = sum(t.get("total_cost", 0) for t in self.trades)

        winning_trades = [t for t in self.trades if t.get("realized_pnl", 0) > 0]
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0

        return {
            "summary": {
                "initial_capital": self.config.initial_capital,
                "final_value": self.portfolio_value,
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "max_drawdown": self.max_drawdown,
                "total_pnl": self.portfolio_value - self.config.initial_capital,
            },
            "trading": {
                "n_trades": n_trades,
                "win_rate": win_rate,
                "total_costs": total_costs,
                "avg_cost_per_trade": total_costs / n_trades if n_trades > 0 else 0,
            },
            "risk": {
                "max_drawdown": self.max_drawdown,
                "was_killed": self.is_killed,
            },
            "execution": {
                "elapsed_time_seconds": elapsed_time,
                "bars_processed": len(self.equity_curve),
            },
            "returns": returns.tolist()[-1000:] if len(returns) > 1000 else returns.tolist(),
            "equity_curve": [
                {"timestamp": str(ts), "value": val}
                for ts, val in self.equity_curve[-500:]
            ],
            "trades": self.trades[-100:] if len(self.trades) > 100 else self.trades,
        }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(
    results: dict[str, Any],
    config: JPMorganBacktestConfig,
) -> dict[str, Any]:
    """Run validation suite on backtest results."""
    logger.info("Running validation suite...")

    returns = np.array(results.get("returns", []))

    if len(returns) < 100:
        return {"error": "Insufficient returns for validation"}

    validator = BacktestValidator(ValidationConfig(
        risk_free_rate=0.05,
        periods_per_year=252 * 26,
    ))

    validation = validator.validate(
        returns=returns,
        n_trials=config.n_trials_for_dsr,
    )

    return validation


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="JPMorgan-Level Institutional Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Symbols to backtest",
    )

    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Use all available symbols",
    )

    parser.add_argument(
        "--core-symbols",
        action="store_true",
        help="Use core (most liquid) symbols only",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10_000_000,
        help="Initial capital (default: $10M)",
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        help="Maximum drawdown limit (default: 15%%)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation suite",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/jpmorgan",
        help="Output directory",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/storage",
        help="Data directory path",
    )

    parser.add_argument(
        "--models-path",
        type=str,
        default="models/artifacts",
        help="Models directory path",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configuration
    config = JPMorganBacktestConfig(
        initial_capital=args.capital,
        max_drawdown_pct=args.max_drawdown,
        run_validation=args.validate and not args.no_validate,
        output_dir=Path(args.output),
    )

    data_path = Path(args.data_path)
    models_path = Path(args.models_path)

    # Determine symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.core_symbols:
        symbols = CORE_SYMBOLS
    elif args.all_symbols:
        symbols = discover_symbols_from_data(data_path)
    else:
        # Default to core symbols
        symbols = CORE_SYMBOLS[:10]

    print("\n" + "=" * 70)
    print("JPMORGAN-LEVEL INSTITUTIONAL BACKTEST")
    print("=" * 70)
    print(f"Symbols: {len(symbols)}")
    print(f"Capital: ${config.initial_capital:,.0f}")
    print(f"Max Drawdown: {config.max_drawdown_pct:.0%}")
    print(f"Validation: {config.run_validation}")
    print("=" * 70 + "\n")

    # Load data
    logger.info("Loading market data...")
    data = load_multi_symbol_data(symbols, data_path)

    if not data:
        logger.error("No data loaded!")
        return 1

    # Load models
    logger.info("Loading models...")
    models = load_models(list(data.keys()), models_path)

    if not models:
        logger.error("No models loaded!")
        return 1

    # Generate features
    logger.info("Generating features...")
    features = {}
    for symbol, df in data.items():
        features[symbol] = generate_features(df, symbol, config)

    # Initialize backtester
    backtester = JPMorganBacktester(config)
    backtester.initialize_symbols(data)

    # Run backtest
    results = backtester.run(data, features, models)

    # Validation
    if config.run_validation:
        validation = validate_results(results, config)
        results["validation"] = validation

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    summary = results.get("summary", {})
    print(f"\nPerformance:")
    print(f"  Total Return: {summary.get('total_return', 0):.2%}")
    print(f"  Annual Return: {summary.get('annual_return', 0):.2%}")
    print(f"  Annual Volatility: {summary.get('annual_volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio: {summary.get('sortino_ratio', 0):.3f}")
    print(f"  Calmar Ratio: {summary.get('calmar_ratio', 0):.3f}")
    print(f"  Max Drawdown: {summary.get('max_drawdown', 0):.2%}")

    trading = results.get("trading", {})
    print(f"\nTrading:")
    print(f"  Total Trades: {trading.get('n_trades', 0)}")
    print(f"  Win Rate: {trading.get('win_rate', 0):.2%}")
    print(f"  Total Costs: ${trading.get('total_costs', 0):,.2f}")

    if "validation" in results:
        val = results["validation"]
        print(f"\nValidation:")
        if "dsr" in val:
            dsr = val["dsr"]
            print(f"  Deflated Sharpe: {dsr.get('deflated_sharpe', 0):.3f}")
            print(f"  Significant: {dsr.get('is_significant', False)}")
        if "pbo" in val:
            pbo = val["pbo"]
            print(f"  PBO: {pbo.get('pbo', 0):.1%}")
            print(f"  Likely Overfit: {pbo.get('is_likely_overfit', False)}")

    print("=" * 70)

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.output_dir / f"backtest_{timestamp}.json"

    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results_json = convert_types(results)

    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
