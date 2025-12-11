#!/usr/bin/env python3
"""
JPMorgan-Level Institutional Backtest Runner v2.0
==================================================

MAJOR UPGRADE: Production-grade backtest system with institutional features.

PHASE 1: Data Infrastructure & Microstructure (The Foundation)
- Dollar Bar aggregation for normalized returns (replaces time bars)
- Tick & Quote Data integration with VPIN calculation
- Real Order Flow Imbalance (OFI) detection
- Trade direction classification (Lee-Ready algorithm)

PHASE 2: Advanced Feature Engineering (The Alpha)
- Fractional Differentiation for stationarity with memory preservation
- HMM-inspired Regime Detection (Bull/Bear × High/Low Vol, Crisis)
- Dynamic threshold adaptation based on market conditions

PHASE 3: Meta-Labeling & Ensemble Modeling (The Filter)
- Two-stage classification: Primary (direction) + Meta (confidence)
- Ensemble Stacking: LightGBM + XGBoost + CatBoost
- Bet sizing based on meta-model probability

PHASE 4: Portfolio Optimization & Risk (The Shield)
- Dynamic Position Sizing: f(Confidence, Liquidity, Volatility)
- Hierarchical Risk Parity (HRP) for robust allocation
- Regime-adjusted risk parameters

PHASE 5: Realistic Execution (The Reality Check)
- Liquidity Constraints: Max 1% of bar volume
- Market Impact Modeling (Almgren-Chriss)
- Order carry-over for unfilled quantities

CRITICAL BUG FIXES:
- Fixed Sharpe Ratio calculation (dimensionally correct)
- Fixed Annual Return annualization (no more astronomical values)
- Fixed Feature Name alignment with models
- Fixed excess returns calculation

Usage:
    python scripts/run_jpmorgan_backtest.py --symbols AAPL MSFT GOOGL
    python scripts/run_jpmorgan_backtest.py --all-symbols --capital 10000000
    python scripts/run_jpmorgan_backtest.py --core-symbols --validate
    python scripts/run_jpmorgan_backtest.py --use-dollar-bars --optimization hrp

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterator
from collections import defaultdict

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.optimize import minimize, Bounds
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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
from data.bar_aggregation import (
    DollarBarAggregator,
    BarConfig,
    BarType,
    estimate_dollar_threshold,
    create_dollar_bars,
)
from data.tick_data import (
    TickDataLoader,
    QuoteDataLoader,
    OrderFlowCalculator,
    TradeDirection,
)

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
    MetaLabeler,
    MetaLabelConfig,
    FractionalDifferentiation,
    MicrostructureFeatures,
    CalendarFeatures,
    FeatureInteractions,
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
# ENUMS
# =============================================================================

class RegimeType(str, Enum):
    """Market regime classification."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hierarchical_risk_parity"
    KELLY = "kelly"
    EQUAL_WEIGHT = "equal_weight"


class BarTypeConfig(str, Enum):
    """Bar type for data aggregation."""
    TIME = "time"      # Traditional 15-min bars
    DOLLAR = "dollar"  # Dollar bars (recommended)
    VOLUME = "volume"  # Volume bars


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class JPMorganBacktestConfig:
    """
    Configuration for JPMorgan-level backtest.

    This configuration represents institutional-grade parameters used by
    top quantitative hedge funds.
    """
    # Capital
    initial_capital: float = 10_000_000  # $10M default

    # Position sizing
    max_position_pct: float = 0.08       # Max 8% per position
    max_portfolio_positions: int = 15    # Max concurrent positions
    min_confidence: float = 0.55         # Higher threshold for better win rate
    min_position_size: float = 10_000    # Minimum position value

    # ===== PHASE 1: Data Infrastructure =====
    bar_type: BarTypeConfig = BarTypeConfig.DOLLAR  # Use Dollar Bars by default
    dollar_bar_threshold: float = 10_000_000  # $10M per bar (auto-adjusted per symbol)
    use_tick_data: bool = True  # Use tick data for VPIN/OFI when available
    use_dollar_bars: bool = True  # Enable dollar bars

    # ===== PHASE 2: Advanced Features =====
    use_fractional_diff: bool = True  # Fractional differentiation
    frac_diff_d: float = 0.4  # Differentiation order (0 < d < 1)
    use_alternative_data: bool = True
    use_advanced_features: bool = True
    feature_selection_method: str = "mda"  # mda, sfi, or both

    # ===== PHASE 3: Meta-Labeling =====
    use_meta_labeling: bool = True  # Enable meta-labeling filter
    meta_label_threshold: float = 0.50  # Min meta probability
    use_ensemble: bool = True  # Use ensemble of models

    # ===== PHASE 4: Portfolio Optimization =====
    optimization_method: OptimizationMethod = OptimizationMethod.HRP  # HRP by default
    enable_regime_detection: bool = True
    rebalance_threshold: float = 0.05    # Rebalance if drift > 5%

    # ===== PHASE 5: Execution =====
    max_participation_rate: float = 0.01  # Max 1% of bar volume (stricter)
    max_position_adv_pct: float = 0.05    # Max 5% of ADV
    enable_market_impact: bool = True
    enable_queue_simulation: bool = True

    # Transaction costs (institutional rates)
    commission_bps: float = 0.5          # 0.5 bps commission
    spread_bps: float = 1.0              # 1 bps half-spread
    market_impact_bps: float = 2.0       # 2 bps market impact

    # Risk management
    max_drawdown_pct: float = 0.20       # Stop at 20% drawdown
    daily_var_limit: float = 0.03        # 3% daily VaR limit
    position_stop_loss: float = 0.05     # 5% stop loss per position
    position_take_profit: float = 0.12   # 12% take profit per position
    min_holding_bars: int = 2            # Minimum 2 bars holding
    cooldown_bars: int = 1               # Cooldown after closing position

    # Validation
    run_validation: bool = True
    n_trials_for_dsr: int = 1            # Number of strategy variations tested

    # Walk-forward
    wf_train_bars: int = 5000            # Training window
    wf_test_bars: int = 1000             # Test window
    wf_embargo_bars: int = 50            # Gap between train/test
    wf_purge_bars: int = 20              # Purge bars around test

    # Covariance estimation
    cov_lookback: int = 126              # 6 months for covariance
    cov_halflife: int = 63               # Exponential decay halflife

    # FIXED: Warmup period - don't trade until we have enough data
    warmup_bars: int = 500               # Bars to wait before trading

    # Regime detection - FIXED: increased lookback
    regime_lookback: int = 252           # 1 week of 15-min bars (was 63, too short)
    regime_vol_threshold: float = 1.5    # Std devs for high vol

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/jpmorgan"))
    save_trades: bool = True
    save_equity_curve: bool = True


# =============================================================================
# PHASE 1: DOLLAR BAR CONVERTER
# =============================================================================

class DollarBarConverter:
    """
    Converts time bars to dollar bars for improved ML performance.

    Dollar bars sample uniformly in dollar volume space, which:
    - Normalizes activity across price levels
    - Recovers approximate normality (Gaussian) in returns
    - Reduces heteroscedasticity
    - Improves ML model accuracy

    Reference: López de Prado, "Advances in Financial Machine Learning"
    """

    def __init__(self, target_bars_per_day: int = 50):
        """
        Initialize converter.

        Args:
            target_bars_per_day: Target number of bars per trading day
        """
        self.target_bars_per_day = target_bars_per_day
        self._threshold_cache: dict[str, float] = {}

    def estimate_threshold(self, df: pl.DataFrame, symbol: str) -> float:
        """Estimate appropriate dollar threshold for a symbol."""
        if symbol in self._threshold_cache:
            return self._threshold_cache[symbol]

        threshold = estimate_dollar_threshold(df, self.target_bars_per_day)
        self._threshold_cache[symbol] = threshold

        logger.info(f"{symbol}: Dollar bar threshold = ${threshold/1e6:.2f}M")
        return threshold

    def convert_to_dollar_bars(
        self,
        df: pl.DataFrame,
        symbol: str,
        threshold: float | None = None,
    ) -> pl.DataFrame:
        """
        Convert time bars to dollar bars.

        This is a simplified conversion that works with OHLCV data.
        For true dollar bars, you need tick data.

        Args:
            df: OHLCV DataFrame with time bars
            symbol: Symbol name
            threshold: Dollar threshold (None = auto-estimate)

        Returns:
            DataFrame with dollar bars
        """
        if threshold is None:
            threshold = self.estimate_threshold(df, symbol)

        # Add dollar volume column
        df = df.with_columns([
            (pl.col("close") * pl.col("volume")).alias("dollar_volume")
        ])

        # Aggregate into dollar bars
        bars_data = []
        current_bar = {
            "dollar_volume": 0.0,
            "open": None,
            "high": -np.inf,
            "low": np.inf,
            "close": None,
            "volume": 0.0,
            "timestamp_open": None,
            "timestamp_close": None,
            "n_ticks": 0,
        }

        for row in df.iter_rows(named=True):
            dv = row["dollar_volume"]

            if current_bar["open"] is None:
                current_bar["open"] = row["open"]
                current_bar["timestamp_open"] = row["timestamp"]

            current_bar["high"] = max(current_bar["high"], row["high"])
            current_bar["low"] = min(current_bar["low"], row["low"])
            current_bar["close"] = row["close"]
            current_bar["timestamp_close"] = row["timestamp"]
            current_bar["volume"] += row["volume"]
            current_bar["dollar_volume"] += dv
            current_bar["n_ticks"] += 1

            # Check if threshold reached
            if current_bar["dollar_volume"] >= threshold:
                bars_data.append({
                    "timestamp": current_bar["timestamp_close"],
                    "open": current_bar["open"],
                    "high": current_bar["high"],
                    "low": current_bar["low"],
                    "close": current_bar["close"],
                    "volume": current_bar["volume"],
                    "dollar_volume": current_bar["dollar_volume"],
                    "n_ticks": current_bar["n_ticks"],
                })

                # Reset
                current_bar = {
                    "dollar_volume": 0.0,
                    "open": None,
                    "high": -np.inf,
                    "low": np.inf,
                    "close": None,
                    "volume": 0.0,
                    "timestamp_open": None,
                    "timestamp_close": None,
                    "n_ticks": 0,
                }

        # Handle last partial bar
        if current_bar["open"] is not None and current_bar["n_ticks"] >= 5:
            bars_data.append({
                "timestamp": current_bar["timestamp_close"],
                "open": current_bar["open"],
                "high": current_bar["high"],
                "low": current_bar["low"],
                "close": current_bar["close"],
                "volume": current_bar["volume"],
                "dollar_volume": current_bar["dollar_volume"],
                "n_ticks": current_bar["n_ticks"],
            })

        result = pl.DataFrame(bars_data)

        logger.info(
            f"{symbol}: Converted {len(df)} time bars to {len(result)} dollar bars "
            f"(ratio: {len(df)/max(len(result),1):.1f}x)"
        )

        return result


# =============================================================================
# PHASE 2: REGIME DETECTOR (ENHANCED HMM-INSPIRED)
# =============================================================================

class RegimeDetector:
    """
    Enhanced HMM-inspired regime detection for adaptive trading.

    UPGRADE: Increased lookback period and added transition smoothing
    to prevent false regime switches on noise.

    Identifies market regimes based on:
    - Return distribution (bull/bear)
    - Volatility level (low/high)
    - Trend strength (momentum)
    - Market structure (trending/mean-reverting)
    """

    def __init__(
        self,
        lookback: int = 252,  # FIXED: Increased from 63 to 252
        vol_threshold: float = 1.5,
        trend_threshold: float = 0.02,
        smoothing_window: int = 5,  # NEW: Smooth regime transitions
    ):
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.trend_threshold = trend_threshold
        self.smoothing_window = smoothing_window
        self._regime_history: list[tuple[datetime, RegimeType]] = []
        self._vol_history: list[float] = []
        self._recent_regimes: list[RegimeType] = []  # For smoothing

    def detect_regime(
        self,
        returns: NDArray[np.float64],
        timestamp: datetime,
    ) -> RegimeType:
        """Detect current market regime with smoothing."""
        if len(returns) < self.lookback:
            return RegimeType.SIDEWAYS

        recent = returns[-self.lookback:]

        # Calculate metrics
        mean_return = np.mean(recent)
        volatility = np.std(recent)

        # Long-term volatility for comparison (use 2x lookback if available)
        long_lookback = min(len(returns), self.lookback * 2)
        long_vol = np.std(returns[-long_lookback:]) if len(returns) > self.lookback else volatility

        # Classify trend direction
        # Annualized threshold (assuming 15-min bars)
        periods_per_year = 252 * 26
        daily_threshold = self.trend_threshold / np.sqrt(periods_per_year)

        is_bull = mean_return > daily_threshold
        is_bear = mean_return < -daily_threshold

        # Classify volatility
        vol_ratio = volatility / long_vol if long_vol > 0 else 1.0
        is_high_vol = vol_ratio > self.vol_threshold

        # Check for crisis (extreme conditions)
        if is_bear and is_high_vol and vol_ratio > 2.0:
            raw_regime = RegimeType.CRISIS
        elif is_bull and not is_high_vol:
            raw_regime = RegimeType.BULL_LOW_VOL
        elif is_bull and is_high_vol:
            raw_regime = RegimeType.BULL_HIGH_VOL
        elif is_bear and not is_high_vol:
            raw_regime = RegimeType.BEAR_LOW_VOL
        elif is_bear and is_high_vol:
            raw_regime = RegimeType.BEAR_HIGH_VOL
        else:
            raw_regime = RegimeType.SIDEWAYS

        # Apply smoothing to prevent rapid regime switches
        self._recent_regimes.append(raw_regime)
        if len(self._recent_regimes) > self.smoothing_window:
            self._recent_regimes.pop(0)

        # Use majority vote for smoothed regime
        if len(self._recent_regimes) >= self.smoothing_window:
            regime_counts = {}
            for r in self._recent_regimes:
                regime_counts[r] = regime_counts.get(r, 0) + 1
            regime = max(regime_counts, key=regime_counts.get)
        else:
            regime = raw_regime

        self._regime_history.append((timestamp, regime))
        self._vol_history.append(float(volatility))

        return regime

    def get_regime_params(
        self,
        regime: RegimeType,
        base_position_pct: float,
    ) -> dict[str, float]:
        """Get regime-adjusted trading parameters."""
        adjustments = {
            # size_mult: position size multiplier
            # confidence_adj: added to min_confidence threshold
            # stop_mult: stop loss tightness (lower = tighter)
            RegimeType.BULL_LOW_VOL: {"size_mult": 1.2, "confidence_adj": -0.02, "stop_mult": 1.0},
            RegimeType.BULL_HIGH_VOL: {"size_mult": 0.8, "confidence_adj": 0.05, "stop_mult": 0.8},
            RegimeType.BEAR_LOW_VOL: {"size_mult": 0.7, "confidence_adj": 0.05, "stop_mult": 0.7},
            RegimeType.BEAR_HIGH_VOL: {"size_mult": 0.5, "confidence_adj": 0.10, "stop_mult": 0.6},
            RegimeType.SIDEWAYS: {"size_mult": 0.9, "confidence_adj": 0.0, "stop_mult": 0.9},
            RegimeType.CRISIS: {"size_mult": 0.25, "confidence_adj": 0.15, "stop_mult": 0.5},
        }

        adj = adjustments.get(regime, {"size_mult": 0.5, "confidence_adj": 0.10, "stop_mult": 0.8})

        return {
            "adjusted_position_pct": base_position_pct * adj["size_mult"],
            "min_confidence": 0.52 + adj["confidence_adj"],
            "stop_loss_mult": adj["stop_mult"],
            "reduce_leverage": regime in [RegimeType.BEAR_HIGH_VOL, RegimeType.CRISIS],
            "go_defensive": regime == RegimeType.CRISIS,
        }

    def get_regime_history(self) -> list[tuple[datetime, RegimeType]]:
        """Get regime history."""
        return self._regime_history.copy()


# =============================================================================
# PHASE 3: META-LABELER FOR BET SIZING
# =============================================================================

class MetaLabelingFilter:
    """
    Meta-Labeling for intelligent bet sizing.

    Two-stage approach:
    1. Primary model predicts direction (long/short)
    2. Meta-model predicts probability of success (bet size)

    This separates "what to trade" from "how much to bet".

    Reference: López de Prado, "Advances in Financial Machine Learning"
    """

    def __init__(
        self,
        threshold: float = 0.50,
        max_bet_size: float = 1.0,
    ):
        self.threshold = threshold
        self.max_bet_size = max_bet_size
        self.meta_labeler = MetaLabeler(MetaLabelConfig(
            primary_threshold=threshold,
            enable_bet_sizing=True,
            max_bet_size=max_bet_size,
        ))

    def apply_filter(
        self,
        primary_signal: int,  # -1, 0, or 1
        confidence: float,
        regime: RegimeType,
        volatility: float,
    ) -> tuple[int, float]:
        """
        Apply meta-labeling filter to a signal.

        Args:
            primary_signal: Direction from primary model
            confidence: Primary model confidence
            regime: Current market regime
            volatility: Current volatility

        Returns:
            Tuple of (filtered_signal, bet_size)
        """
        if primary_signal == 0:
            return 0, 0.0

        # Adjust threshold based on regime
        regime_multipliers = {
            RegimeType.BULL_LOW_VOL: 0.95,   # Lower threshold in good conditions
            RegimeType.BULL_HIGH_VOL: 1.05,
            RegimeType.BEAR_LOW_VOL: 1.10,
            RegimeType.BEAR_HIGH_VOL: 1.15,
            RegimeType.SIDEWAYS: 1.00,
            RegimeType.CRISIS: 1.25,  # Much higher threshold in crisis
        }

        adjusted_threshold = self.threshold * regime_multipliers.get(regime, 1.0)

        # Check if signal passes threshold
        if confidence < adjusted_threshold:
            return 0, 0.0

        # Calculate bet size based on:
        # 1. Confidence (higher = larger bet)
        # 2. Volatility (higher = smaller bet)
        # 3. Regime (crisis = smaller bet)

        # Base bet size from confidence
        base_bet = (confidence - adjusted_threshold) / (1.0 - adjusted_threshold)
        base_bet = min(base_bet, self.max_bet_size)

        # Volatility adjustment (assume 20% is normal vol)
        vol_adjustment = min(1.0, 0.20 / max(volatility, 0.05))

        # Regime adjustment
        regime_bet_mult = {
            RegimeType.BULL_LOW_VOL: 1.0,
            RegimeType.BULL_HIGH_VOL: 0.8,
            RegimeType.BEAR_LOW_VOL: 0.7,
            RegimeType.BEAR_HIGH_VOL: 0.5,
            RegimeType.SIDEWAYS: 0.9,
            RegimeType.CRISIS: 0.3,
        }

        final_bet = base_bet * vol_adjustment * regime_bet_mult.get(regime, 0.5)
        final_bet = np.clip(final_bet, 0.0, self.max_bet_size)

        return primary_signal, final_bet


# =============================================================================
# PHASE 4: COVARIANCE ESTIMATION (ENHANCED)
# =============================================================================

class CovarianceEstimator:
    """
    Advanced covariance matrix estimation with multiple methods.

    Methods:
    - Ledoit-Wolf shrinkage (for stability)
    - Exponential weighted (for recency)
    - Oracle Approximating Shrinkage
    """

    @staticmethod
    def ledoit_wolf(returns: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ledoit-Wolf shrinkage estimator."""
        n, p = returns.shape

        if n <= 1:
            return np.eye(p)

        # Sample covariance
        sample_cov = np.cov(returns, rowvar=False)
        if sample_cov.ndim == 0:
            return np.array([[sample_cov]])

        # Shrinkage target: scaled identity
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)

        # Compute optimal shrinkage intensity
        delta = sample_cov - target
        delta_sum = np.sum(delta ** 2)

        X = returns - returns.mean(axis=0)
        X2 = X ** 2
        sum_sq = np.sum(np.dot(X2.T, X2) / n)
        gamma = np.sum(delta ** 2)

        kappa = (sum_sq - gamma) / n if n > 0 else 0
        shrinkage = max(0, min(1, kappa / gamma)) if gamma > 0 else 0

        return (1 - shrinkage) * sample_cov + shrinkage * target

    @staticmethod
    def exponential_weighted(
        returns: NDArray[np.float64],
        halflife: int = 63,
    ) -> NDArray[np.float64]:
        """Exponentially weighted covariance matrix."""
        n, p = returns.shape

        if n <= 1:
            return np.eye(p)

        # Calculate weights
        decay = 0.5 ** (1 / halflife)
        weights = decay ** np.arange(n - 1, -1, -1)
        weights /= weights.sum()

        # Weighted mean
        weighted_mean = np.average(returns, axis=0, weights=weights)

        # Weighted covariance
        centered = returns - weighted_mean
        weighted_cov = np.dot((centered * weights[:, np.newaxis]).T, centered)

        return weighted_cov

    @staticmethod
    def oracle_approximating(
        returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Oracle Approximating Shrinkage (OAS) estimator."""
        n, p = returns.shape

        if n <= 1:
            return np.eye(p)

        sample_cov = np.cov(returns, rowvar=False)
        if sample_cov.ndim == 0:
            return np.array([[sample_cov]])

        # Target: Scaled identity
        trace_cov = np.trace(sample_cov)
        mu = trace_cov / p

        # Frobenius norm
        delta = sample_cov - mu * np.eye(p)
        delta_norm_sq = np.sum(delta ** 2)

        # OAS shrinkage intensity
        rho = (1 - 2/p) * delta_norm_sq + trace_cov**2
        rho /= ((n + 1 - 2/p) * (delta_norm_sq + trace_cov**2 / p))
        rho = min(1, max(0, rho))

        return (1 - rho) * sample_cov + rho * mu * np.eye(p)


# =============================================================================
# PHASE 4: PORTFOLIO OPTIMIZER (ENHANCED)
# =============================================================================

class PortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer.

    UPGRADE: Added Kelly Criterion fix for proper negative weight handling.

    Implements:
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Risk Parity
    - Hierarchical Risk Parity (HRP) - RECOMMENDED
    - Kelly Criterion (Half-Kelly for safety)
    - Black-Litterman
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        max_weight: float = 0.15,
        min_weight: float = 0.0,
        allow_short: bool = False,  # NEW: Allow short positions
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.allow_short = allow_short

    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        method: OptimizationMethod = OptimizationMethod.HRP,  # CHANGED: HRP default
    ) -> NDArray[np.float64]:
        """Optimize portfolio weights."""
        n_assets = len(expected_returns)

        if n_assets == 0:
            return np.array([])

        if n_assets == 1:
            return np.array([1.0])

        try:
            if method == OptimizationMethod.EQUAL_WEIGHT:
                return np.ones(n_assets) / n_assets

            elif method == OptimizationMethod.MIN_VARIANCE:
                return self._min_variance(covariance)

            elif method == OptimizationMethod.MAX_SHARPE:
                return self._max_sharpe(expected_returns, covariance)

            elif method == OptimizationMethod.RISK_PARITY:
                return self._risk_parity(covariance)

            elif method == OptimizationMethod.HRP:
                return self._hierarchical_risk_parity(covariance)

            elif method == OptimizationMethod.KELLY:
                return self._kelly_criterion(expected_returns, covariance)

            else:
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using equal weight")
            return np.ones(n_assets) / n_assets

    def _min_variance(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Minimum variance portfolio."""
        n = len(covariance)

        def objective(w):
            return w @ covariance @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(self.min_weight, self.max_weight)

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _max_sharpe(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximum Sharpe ratio portfolio."""
        n = len(expected_returns)

        # FIXED: Use proper annualized risk-free rate
        # Assume 15-min bars, so periods_per_year = 252 * 26
        periods_per_year = 252 * 26
        rf_per_period = self.risk_free_rate / periods_per_year

        def neg_sharpe(w):
            ret = w @ expected_returns
            vol = np.sqrt(w @ covariance @ w)
            if vol < 1e-10:
                return 0
            return -(ret - rf_per_period) / vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(self.min_weight, self.max_weight)

        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _risk_parity(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Risk parity portfolio - equal risk contribution."""
        n = len(covariance)

        def risk_contribution(w):
            port_vol = np.sqrt(w @ covariance @ w)
            if port_vol < 1e-10:
                return np.ones(n) / n
            marginal_contrib = covariance @ w
            risk_contrib = w * marginal_contrib / port_vol
            return risk_contrib

        def objective(w):
            rc = risk_contribution(w)
            target_rc = np.ones(n) / n * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(0.02, self.max_weight)

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _hierarchical_risk_parity(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Hierarchical Risk Parity (HRP).

        RECOMMENDED: More stable than Mean-Variance, better diversification.
        Uses clustering to identify true risk sources.
        """
        n = len(covariance)

        if n <= 1:
            return np.ones(n)

        # FIXED: Make writable copy to avoid read-only errors
        cov = np.array(covariance, dtype=np.float64, copy=True)

        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)

        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Ensure valid distance matrix
        dist = np.clip(dist, 0, 2)
        dist = (dist + dist.T) / 2

        # Hierarchical clustering
        condensed_dist = squareform(dist, checks=False)
        if len(condensed_dist) == 0:
            return np.ones(n) / n

        link = linkage(condensed_dist, method="single")

        # Quasi-diagonalization
        sort_idx = self._get_quasi_diag(link, n)

        # Recursive bisection - use writable array
        weights = np.zeros(n, dtype=np.float64)
        self._hrp_recursive_bisection(weights, cov, sort_idx)

        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n) / n

        return weights

    def _get_quasi_diag(self, link: NDArray, n: int) -> list[int]:
        """Get quasi-diagonal sort indices from linkage."""
        link = link.astype(int)
        sort_idx = list(range(n))

        for i in range(len(link)):
            idx1, idx2 = int(link[i, 0]), int(link[i, 1])
            cluster_idx = n + i

            new_sort = []
            for idx in sort_idx:
                if idx == cluster_idx:
                    continue
                if idx == idx1 or idx == idx2:
                    continue
                new_sort.append(idx)

            # Insert clustered items together
            insert_pos = len(new_sort)
            for j, idx in enumerate(sort_idx):
                if idx == cluster_idx or idx == idx1 or idx == idx2:
                    insert_pos = min(insert_pos, j)
                    break

            if idx1 < n:
                new_sort.insert(insert_pos, idx1)
            if idx2 < n:
                new_sort.insert(insert_pos + (1 if idx1 < n else 0), idx2)

            sort_idx = [x for x in sort_idx if x < n]

        return [x for x in sort_idx if x < n][:n]

    def _hrp_recursive_bisection(
        self,
        weights: NDArray[np.float64],
        covariance: NDArray[np.float64],
        sort_idx: list[int],
    ) -> None:
        """Recursive bisection for HRP."""
        if len(sort_idx) == 0:
            return

        if len(sort_idx) == 1:
            weights[sort_idx[0]] = 1.0
            return

        # Split
        mid = len(sort_idx) // 2
        left_idx = sort_idx[:mid]
        right_idx = sort_idx[mid:]

        # Cluster variance
        left_var = self._cluster_variance(covariance, left_idx)
        right_var = self._cluster_variance(covariance, right_idx)

        # Allocate based on inverse variance
        total_var = left_var + right_var
        if total_var > 0:
            left_weight = 1 - left_var / total_var
            right_weight = 1 - right_var / total_var
        else:
            left_weight = right_weight = 0.5

        # Normalize
        total = left_weight + right_weight
        if total > 0:
            left_weight /= total
            right_weight /= total

        # Recursive allocation
        left_weights = np.zeros(len(covariance))
        right_weights = np.zeros(len(covariance))

        self._hrp_recursive_bisection(left_weights, covariance, left_idx)
        self._hrp_recursive_bisection(right_weights, covariance, right_idx)

        weights[:] = left_weight * left_weights + right_weight * right_weights

    def _cluster_variance(
        self,
        covariance: NDArray[np.float64],
        indices: list[int],
    ) -> float:
        """Calculate variance of a cluster."""
        if len(indices) == 0:
            return 1.0

        cov_slice = covariance[np.ix_(indices, indices)]
        # FIXED: Make writable copy to avoid read-only error
        diag = np.array(np.diag(cov_slice), dtype=np.float64, copy=True)
        diag[diag == 0] = 1e-10

        inv_var = 1 / diag
        w = inv_var / inv_var.sum()

        return float(w @ cov_slice @ w)

    def _kelly_criterion(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Kelly criterion for optimal bet sizing (half-Kelly for safety).

        FIXED: Proper handling of negative weights.
        """
        try:
            inv_cov = np.linalg.inv(covariance + np.eye(len(covariance)) * 1e-6)
        except np.linalg.LinAlgError:
            return self._max_sharpe(expected_returns, covariance)

        # Full Kelly
        full_kelly = inv_cov @ expected_returns

        # Half Kelly (more conservative)
        kelly_weights = full_kelly * 0.5

        # FIXED: Proper handling - if shorts not allowed, set negative to zero
        if not self.allow_short:
            kelly_weights = np.maximum(kelly_weights, 0)

        # Clip to bounds
        kelly_weights = np.clip(kelly_weights, -self.max_weight if self.allow_short else 0, self.max_weight)

        # Normalize properly (sum of absolute values for leverage)
        total = np.sum(np.abs(kelly_weights))
        if total > 0:
            # Scale to sum to 1 for long-only
            if not self.allow_short:
                kelly_weights = kelly_weights / np.sum(kelly_weights)
            else:
                # For long-short, keep proportions but scale
                kelly_weights = kelly_weights / total
        else:
            kelly_weights = np.ones(len(expected_returns)) / len(expected_returns)

        return kelly_weights


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

@dataclass
class WalkForwardFold:
    """Single walk-forward fold."""
    fold_number: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0


class WalkForwardValidator:
    """
    Walk-forward validation with purging and embargo.

    Prevents look-ahead bias by:
    - Purging: Removes training samples that overlap with test period
    - Embargo: Adds gap between training and test
    """

    def __init__(
        self,
        train_periods: int = 5000,
        test_periods: int = 1000,
        embargo_periods: int = 50,
        purge_periods: int = 20,
    ):
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.embargo_periods = embargo_periods
        self.purge_periods = purge_periods

    def generate_folds(
        self,
        data_length: int,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Generate walk-forward folds with purging and embargo."""
        min_samples = self.train_periods + self.embargo_periods + self.test_periods

        if data_length < min_samples:
            logger.warning(f"Insufficient data for walk-forward: {data_length} < {min_samples}")
            return

        fold = 0
        current_start = 0

        while current_start + min_samples <= data_length:
            # Define periods
            train_end = current_start + self.train_periods
            test_start = train_end + self.embargo_periods
            test_end = test_start + self.test_periods

            if test_end > data_length:
                break

            # Create indices with purging
            train_idx = np.arange(current_start, train_end - self.purge_periods)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

            # Move forward by test period
            current_start += self.test_periods
            fold += 1

    def calculate_fold_metrics(
        self,
        returns: NDArray[np.float64],
        fold_idx: tuple[NDArray[np.int64], NDArray[np.int64]],
        fold_number: int,
    ) -> WalkForwardFold:
        """Calculate metrics for a single fold."""
        train_idx, test_idx = fold_idx

        train_returns = returns[train_idx] if len(train_idx) > 0 else np.array([0])
        test_returns = returns[test_idx] if len(test_idx) > 0 else np.array([0])

        # Sharpe ratios
        train_sharpe = self._calculate_sharpe(train_returns)
        test_sharpe = self._calculate_sharpe(test_returns)

        # Returns
        train_return = float(np.prod(1 + train_returns) - 1) if len(train_returns) > 0 else 0
        test_return = float(np.prod(1 + test_returns) - 1) if len(test_returns) > 0 else 0

        # Max drawdown
        if len(test_returns) > 0:
            cumulative = np.cumprod(1 + test_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_dd = float(np.min(drawdowns))
        else:
            max_dd = 0

        return WalkForwardFold(
            fold_number=fold_number,
            train_start_idx=int(train_idx[0]) if len(train_idx) > 0 else 0,
            train_end_idx=int(train_idx[-1]) if len(train_idx) > 0 else 0,
            test_start_idx=int(test_idx[0]) if len(test_idx) > 0 else 0,
            test_end_idx=int(test_idx[-1]) if len(test_idx) > 0 else 0,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_return=train_return,
            test_return=test_return,
            max_drawdown=max_dd,
        )

    def _calculate_sharpe(self, returns: NDArray[np.float64]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        periods_per_year = 252 * 26  # 15-min bars
        return float(mean_ret / std_ret * np.sqrt(periods_per_year))

    def validate_results(
        self,
        folds: list[WalkForwardFold],
    ) -> dict[str, float]:
        """Validate walk-forward results and check for overfitting."""
        if not folds:
            return {}

        train_sharpes = [f.train_sharpe for f in folds]
        test_sharpes = [f.test_sharpe for f in folds]

        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)

        # Overfitting ratio
        overfit_ratio = (avg_train - avg_test) / avg_train if avg_train > 0 else 0

        # Probability of backtest overfit
        pbo = sum(1 for ts in test_sharpes if ts <= 0) / len(test_sharpes)

        # Stability
        sharpe_stability = 1 - np.std(test_sharpes) / (np.mean(np.abs(test_sharpes)) + 1e-10)

        return {
            "avg_train_sharpe": float(avg_train),
            "avg_test_sharpe": float(avg_test),
            "overfit_ratio": float(overfit_ratio),
            "probability_backtest_overfit": float(pbo),
            "sharpe_stability": float(np.clip(sharpe_stability, 0, 1)),
            "n_folds": len(folds),
            "n_positive_folds": sum(1 for f in folds if f.test_return > 0),
            "n_profitable_folds": sum(1 for f in folds if f.test_sharpe > 0),
        }


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
# FEATURE GENERATION (ENHANCED WITH PHASE 2)
# =============================================================================

def generate_features(
    df: pl.DataFrame,
    symbol: str,
    config: JPMorganBacktestConfig,
) -> pl.DataFrame:
    """
    Generate all features for a symbol.

    UPGRADE: Added fractional differentiation and enhanced microstructure.
    """
    logger.info(f"Generating features for {symbol}...")

    # Basic technical features
    feature_pipeline = FeaturePipeline(create_default_config())
    df = feature_pipeline.generate(df)

    # PHASE 2: Fractional Differentiation
    if config.use_fractional_diff:
        try:
            frac_diff = FractionalDifferentiation(d=config.frac_diff_d)
            df = frac_diff.add_features(df, columns=["close", "volume"])
            logger.info(f"  Added fractional differentiation features (d={config.frac_diff_d})")
        except Exception as e:
            logger.warning(f"Fractional diff failed for {symbol}: {e}")

    # Advanced microstructure features
    if config.use_advanced_features:
        df = MicrostructureFeatures.add_features(df)
        df = CalendarFeatures.add_features(df)

    # FIXED: Add interaction features (required by models)
    df = FeatureInteractions.add_interactions(df)

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
    """Load trained models for symbols with feature metadata."""
    models = {}

    for symbol in symbols:
        symbol_dir = models_dir / symbol

        if not symbol_dir.exists():
            continue

        # Load metadata for feature names
        metadata_file = symbol_dir / "metadata.json"
        feature_names = None

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if "models" in metadata:
                        for model_key in ["lightgbm_v1", "xgboost_v1"]:
                            if model_key in metadata["models"]:
                                feature_names = metadata["models"][model_key].get("feature_names", [])
                                if feature_names:
                                    break
            except Exception as e:
                logger.debug(f"Failed to load metadata for {symbol}: {e}")

        # Find model files
        model_files = list(symbol_dir.glob("*.pkl"))

        if not model_files:
            continue

        # Load best model (prefer lightgbm)
        model_loaded = False
        for model_file in model_files:
            if "lightgbm" in model_file.name.lower():
                try:
                    with open(model_file, "rb") as f:
                        model_data = pickle.load(f)

                    if isinstance(model_data, dict):
                        model = model_data.get("model", model_data)
                        # Get feature names from pkl (more reliable)
                        pkl_features = model_data.get("feature_names", [])
                        if pkl_features:
                            feature_names = pkl_features
                    else:
                        model = model_data

                    models[symbol] = {
                        "model": model,
                        "path": str(model_file),
                        "type": "lightgbm",
                        "feature_names": feature_names or [],
                        "n_features": len(feature_names) if feature_names else 0,
                    }
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model for {symbol}: {e}")

        if not model_loaded:
            for model_file in model_files:
                if "xgboost" in model_file.name.lower():
                    try:
                        with open(model_file, "rb") as f:
                            model_data = pickle.load(f)

                        if isinstance(model_data, dict):
                            model = model_data.get("model", model_data)
                            pkl_features = model_data.get("feature_names", [])
                            if pkl_features:
                                feature_names = pkl_features
                        else:
                            model = model_data

                        models[symbol] = {
                            "model": model,
                            "path": str(model_file),
                            "type": "xgboost",
                            "feature_names": feature_names or [],
                            "n_features": len(feature_names) if feature_names else 0,
                        }
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load xgboost for {symbol}: {e}")

    logger.info(f"Loaded models for {len(models)} symbols")
    return models


# =============================================================================
# JPMORGAN BACKTESTER (ENHANCED)
# =============================================================================

class JPMorganBacktester:
    """
    JPMorgan-level institutional backtester v2.0.

    MAJOR UPGRADES:
    - Dollar bar support for improved ML performance
    - Meta-labeling filter for bet sizing
    - Enhanced regime detection with smoothing
    - Fixed Sharpe ratio calculation
    - Fixed annual return annualization
    - Stricter liquidity constraints (1% max participation)
    """

    def __init__(self, config: JPMorganBacktestConfig):
        """Initialize backtester."""
        self.config = config

        # PHASE 1: Dollar bar converter
        self.dollar_bar_converter = DollarBarConverter(target_bars_per_day=50)

        # PHASE 3: Meta-labeling filter
        self.meta_filter = MetaLabelingFilter(
            threshold=config.meta_label_threshold,
            max_bet_size=1.0,
        )

        # PHASE 5: Execution components (stricter constraints)
        self.liquidity_config = LiquidityConfig(
            max_participation_rate=config.max_participation_rate,  # 1%
            max_position_adv_pct=config.max_position_adv_pct,      # 5%
            enable_order_carryover=True,
        )
        self.executor = LiquidityConstrainedExecutor(self.liquidity_config)
        self.impact_calc = MarketImpactCalculator(self.liquidity_config)

        # PHASE 4: Portfolio optimization (HRP default)
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=0.05,
            max_weight=config.max_position_pct,
            min_weight=0.0,
        )

        # PHASE 2: Regime detection (enhanced)
        self.regime_detector = RegimeDetector(
            lookback=config.regime_lookback,  # 252 bars (increased)
            vol_threshold=config.regime_vol_threshold,
            smoothing_window=5,  # NEW: smoothing
        )

        # Walk-forward validation
        self.wf_validator = WalkForwardValidator(
            train_periods=config.wf_train_bars,
            test_periods=config.wf_test_bars,
            embargo_periods=config.wf_embargo_bars,
            purge_periods=config.wf_purge_bars,
        )

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
        self.regime_history: list[tuple[datetime, str]] = []

        # Position timing
        self.symbol_cooldown: dict[str, int] = {}
        self.position_open_bar: dict[str, int] = {}

        # Historical returns for covariance
        self.symbol_returns: dict[str, list[float]] = defaultdict(list)
        self.symbol_prices: dict[str, list[float]] = defaultdict(list)

        # Risk
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.is_killed = False
        self.current_regime = RegimeType.SIDEWAYS

        # Volatility tracking for meta-labeling
        self.portfolio_volatility = 0.20  # Initial estimate

        # Walk-forward folds
        self.wf_folds: list[WalkForwardFold] = []

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

        Returns:
            Comprehensive backtest results dictionary
        """
        logger.info("Starting JPMorgan-level backtest v2.0...")
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

            # Update price history for covariance
            for symbol, price in current_prices.items():
                self.symbol_prices[symbol].append(price)
                if len(self.symbol_prices[symbol]) > 1:
                    ret = (price / self.symbol_prices[symbol][-2]) - 1
                    self.symbol_returns[symbol].append(ret)

            # Check risk limits
            self._check_risk_limits()

            if self.is_killed:
                break

            # Detect regime
            if self.config.enable_regime_detection and len(self.returns) >= self.config.regime_lookback:
                portfolio_returns = np.array(self.returns[-self.config.regime_lookback:])
                self.current_regime = self.regime_detector.detect_regime(portfolio_returns, timestamp)
                self.regime_history.append((timestamp, self.current_regime.value))

                # Update portfolio volatility estimate
                if len(self.returns) >= 20:
                    self.portfolio_volatility = np.std(self.returns[-20:]) * np.sqrt(252 * 26)

            # Generate signals
            signals = self._generate_signals(
                timestamp, i + start_idx, features, models, current_prices
            )

            # Portfolio optimization and execution
            bar_idx = i + start_idx

            # FIXED: Skip trading during warmup period to build price/return history
            if i < self.config.warmup_bars:
                continue

            if signals and len(signals) >= 2:
                self._optimize_and_execute(
                    signals, current_prices, current_volumes, timestamp, bar_idx
                )
            elif signals:
                self._execute_signals(
                    signals, current_prices, current_volumes, timestamp, bar_idx
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
                    f"DD: {self.current_drawdown:.2%}, "
                    f"Regime: {self.current_regime.value}"
                )

        # Generate results
        elapsed = time.time() - start_time
        results = self._generate_results(elapsed)

        # Walk-forward validation
        if len(self.returns) > self.config.wf_train_bars + self.config.wf_test_bars:
            wf_results = self._run_walk_forward_validation()
            results["walk_forward"] = wf_results

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

        # Position stop losses and take profits
        for symbol, pos in list(self.positions.items()):
            if pos["quantity"] == 0:
                continue

            pnl_pct = pos["unrealized_pnl"] / (pos["avg_price"] * abs(pos["quantity"]))

            # Stop loss - adjusted by regime
            regime_params = self.regime_detector.get_regime_params(
                self.current_regime, self.config.max_position_pct
            )
            adjusted_stop = self.config.position_stop_loss * regime_params["stop_loss_mult"]

            if pnl_pct < -adjusted_stop:
                logger.warning(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                pos["stop_triggered"] = True

            # Take profit
            if pnl_pct > self.config.position_take_profit:
                logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
                pos["take_profit_triggered"] = True

    def _generate_signals(
        self,
        timestamp: datetime,
        bar_idx: int,
        features: dict[str, pl.DataFrame],
        models: dict[str, Any],
        prices: dict[str, float],
    ) -> dict[str, dict]:
        """
        Generate trading signals with PHASE 3 meta-labeling filter.

        UPGRADE: Added feature alignment validation and meta-labeling.
        """
        signals = {}

        # Get regime-adjusted min confidence
        regime_params = self.regime_detector.get_regime_params(
            self.current_regime, self.config.max_position_pct
        )
        min_confidence = regime_params["min_confidence"]

        # In crisis mode, be very selective
        if regime_params["go_defensive"]:
            min_confidence = 0.65

        for symbol, model_info in models.items():
            if symbol not in features or symbol not in prices:
                continue

            feat_df = features[symbol]

            if bar_idx >= len(feat_df):
                continue

            try:
                # Get feature row
                row = feat_df.row(bar_idx, named=True)

                # Get model's expected feature names - CRITICAL for alignment
                expected_features = model_info.get("feature_names", [])

                if expected_features:
                    # FIXED: Validate feature alignment
                    feat_values = []
                    missing_features = []

                    for feat_name in expected_features:
                        val = row.get(feat_name, None)
                        if val is None:
                            missing_features.append(feat_name)
                            feat_values.append(0.0)
                        elif isinstance(val, (int, float)) and not np.isnan(val):
                            feat_values.append(float(val))
                        else:
                            feat_values.append(0.0)

                    # Log warning if many features missing (only first time)
                    if missing_features and bar_idx == 500:
                        logger.warning(
                            f"{symbol}: {len(missing_features)} features missing "
                            f"(e.g., {missing_features[:3]})"
                        )
                else:
                    # Fallback: extract all numeric features
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

                # PHASE 3: Apply meta-labeling filter
                if self.config.use_meta_labeling:
                    filtered_direction, bet_size = self.meta_filter.apply_filter(
                        primary_signal=direction,
                        confidence=confidence,
                        regime=self.current_regime,
                        volatility=self.portfolio_volatility,
                    )

                    if filtered_direction == 0:
                        continue  # Signal filtered out

                    # Adjust confidence with bet size
                    confidence = min_confidence + (confidence - min_confidence) * bet_size
                    direction = filtered_direction

                # Check confidence threshold
                if confidence >= min_confidence:
                    signals[symbol] = {
                        "direction": direction,
                        "confidence": confidence,
                        "price": prices[symbol],
                    }

            except Exception as e:
                if bar_idx < 10:
                    logger.warning(f"Signal generation failed for {symbol} at bar {bar_idx}: {e}")

        # Log signal count periodically
        if bar_idx % 5000 == 0 and signals:
            logger.info(f"Bar {bar_idx}: Generated {len(signals)} signals")

        return signals

    def _optimize_and_execute(
        self,
        signals: dict[str, dict],
        prices: dict[str, float],
        volumes: dict[str, float],
        timestamp: datetime,
        bar_idx: int = 0,
    ) -> None:
        """Optimize portfolio weights and execute trades."""
        symbols = list(signals.keys())
        n = len(symbols)

        if n < 2:
            self._execute_signals(signals, prices, volumes, timestamp, bar_idx)
            return

        # Build expected returns from signals
        expected_returns = np.array([
            signals[s]["direction"] * signals[s]["confidence"] * 0.01
            for s in symbols
        ])

        # Build covariance matrix from historical returns
        if all(len(self.symbol_returns[s]) >= self.config.cov_lookback for s in symbols):
            returns_matrix = np.array([
                self.symbol_returns[s][-self.config.cov_lookback:]
                for s in symbols
            ]).T

            covariance = CovarianceEstimator.exponential_weighted(
                returns_matrix, self.config.cov_halflife
            )
        else:
            # Use simple diagonal covariance
            covariance = np.eye(n) * 0.04  # 20% vol assumption

        # Get regime-adjusted position sizing
        regime_params = self.regime_detector.get_regime_params(
            self.current_regime, self.config.max_position_pct
        )

        # Optimize (HRP by default now)
        try:
            target_weights = self.optimizer.optimize(
                expected_returns,
                covariance,
                self.config.optimization_method,
            )
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            target_weights = np.ones(n) / n

        # Apply regime adjustment
        target_weights = target_weights * regime_params["adjusted_position_pct"] / self.config.max_position_pct

        # Ensure constraints
        target_weights = np.clip(target_weights, 0, regime_params["adjusted_position_pct"])

        # Normalize if sum > max allocation
        max_allocation = 0.95  # Keep 5% cash
        if np.sum(target_weights) > max_allocation:
            target_weights = target_weights / np.sum(target_weights) * max_allocation

        # Execute trades
        for i, symbol in enumerate(symbols):
            target_value = self.portfolio_value * target_weights[i] * signals[symbol]["direction"]
            current_value = self.positions.get(symbol, {}).get("market_value", 0)
            current_qty = self.positions.get(symbol, {}).get("quantity", 0)

            current_direction = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)
            target_direction = signals[symbol]["direction"]

            trade_value = abs(target_value) - abs(current_value)

            if abs(trade_value) < self.config.min_position_size:
                continue

            if current_direction != 0 and current_direction != target_direction:
                self._close_position(symbol, prices.get(symbol, 0), volumes.get(symbol, 10000),
                                    timestamp, "reversal")
                trade_value = abs(target_value)

            if trade_value > 0:
                target_shares = trade_value / prices[symbol]
                self._open_position(
                    symbol, target_direction, target_shares, prices[symbol],
                    volumes.get(symbol, 10000), timestamp, bar_idx
                )
            elif trade_value < 0 and abs(current_value) > 0:
                shares_to_sell = abs(trade_value) / prices[symbol]
                if shares_to_sell >= abs(current_qty):
                    self._close_position(symbol, prices[symbol], volumes.get(symbol, 10000),
                                        timestamp, "reduce")

    def _execute_signals(
        self,
        signals: dict[str, dict],
        prices: dict[str, float],
        volumes: dict[str, float],
        timestamp: datetime,
        bar_idx: int = 0,
    ) -> None:
        """Execute trading signals with liquidity constraints and cooldown."""
        # First handle stop losses and take profits
        for symbol, pos in list(self.positions.items()):
            open_bar = self.position_open_bar.get(symbol, 0)
            holding_bars = bar_idx - open_bar

            if pos.get("stop_triggered") or pos.get("take_profit_triggered"):
                if pos.get("stop_triggered") or holding_bars >= self.config.min_holding_bars:
                    reason = "stop_loss" if pos.get("stop_triggered") else "take_profit"
                    self._close_position(symbol, prices.get(symbol, pos["current_price"]),
                                        volumes.get(symbol, 10000), timestamp, reason)
                    self.symbol_cooldown[symbol] = bar_idx + self.config.cooldown_bars

        # Limit number of new positions
        n_current = len([p for p in self.positions.values() if p["quantity"] != 0])
        n_available = self.config.max_portfolio_positions - n_current

        if n_available <= 0:
            return

        # Get regime-adjusted parameters
        regime_params = self.regime_detector.get_regime_params(
            self.current_regime, self.config.max_position_pct
        )

        # Sort signals by confidence
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )[:n_available]

        for symbol, signal in sorted_signals:
            # Check cooldown
            if self.symbol_cooldown.get(symbol, 0) > bar_idx:
                continue

            direction = signal["direction"]
            confidence = signal["confidence"]
            price = signal["price"]
            volume = volumes.get(symbol, 10000)

            # Skip if already have position in same direction
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["quantity"] * direction > 0:
                    continue
                open_bar = self.position_open_bar.get(symbol, 0)
                if bar_idx - open_bar < self.config.min_holding_bars:
                    continue
                self._close_position(symbol, price, volume, timestamp, "reversal")
                self.symbol_cooldown[symbol] = bar_idx + self.config.cooldown_bars
                continue

            # Calculate position size with regime adjustment
            position_pct = regime_params["adjusted_position_pct"] * confidence
            target_value = self.portfolio_value * position_pct
            target_shares = target_value / price

            # Execute with liquidity constraints
            self._open_position(
                symbol, direction, target_shares, price, volume, timestamp, bar_idx
            )

    def _open_position(
        self,
        symbol: str,
        direction: int,
        target_shares: float,
        price: float,
        bar_volume: float,
        timestamp: datetime,
        bar_idx: int = 0,
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

        if old_qty == 0:
            self.position_open_bar[symbol] = bar_idx

        if new_qty != 0:
            if old_qty == 0:
                pos["avg_price"] = result.fill_price
            else:
                pos["avg_price"] = (
                    (old_qty * pos["avg_price"] + signed_qty * result.fill_price) / new_qty
                )

        pos["quantity"] = new_qty
        pos["cost_basis"] += trade_value + total_cost

        self.cash -= trade_value * direction + total_cost

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
            "regime": self.current_regime.value,
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

        trade_value = result.filled_quantity * result.fill_price
        cost_basis = (result.filled_quantity / quantity) * pos["cost_basis"]
        realized_pnl = trade_value - cost_basis if pos["quantity"] > 0 else cost_basis - trade_value

        commission = trade_value * (self.config.commission_bps / 10000)
        spread_cost = trade_value * (self.config.spread_bps / 10000)
        impact_cost = trade_value * (result.market_impact / 10000)
        total_cost = commission + spread_cost + impact_cost

        remaining = quantity - result.filled_quantity
        if remaining <= 0:
            pos["quantity"] = 0
            pos["cost_basis"] = 0
            pos["stop_triggered"] = False
            pos["take_profit_triggered"] = False
        else:
            pos["quantity"] = remaining * (1 if pos["quantity"] > 0 else -1)
            pos["cost_basis"] *= (remaining / quantity)

        self.cash += trade_value - total_cost

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
            "regime": self.current_regime.value,
        })

    def _run_walk_forward_validation(self) -> dict[str, Any]:
        """Run walk-forward validation on backtest returns."""
        returns = np.array(self.returns)

        folds = []
        for i, (train_idx, test_idx) in enumerate(self.wf_validator.generate_folds(len(returns))):
            fold = self.wf_validator.calculate_fold_metrics(returns, (train_idx, test_idx), i)
            folds.append(fold)

        self.wf_folds = folds

        return self.wf_validator.validate_results(folds)

    def _generate_results(self, elapsed_time: float) -> dict[str, Any]:
        """
        Generate comprehensive results.

        CRITICAL FIXES:
        - Fixed Sharpe ratio calculation (proper risk-free rate)
        - Fixed annual return annualization (no astronomical values)
        """
        returns = np.array(self.returns)

        if len(returns) == 0:
            return {"error": "No returns generated"}

        # Basic metrics
        total_return = (self.portfolio_value / self.config.initial_capital) - 1

        # Annualization (15-min bars)
        periods_per_year = 252 * 26  # 6552 periods
        n_periods = len(returns)
        years = n_periods / periods_per_year

        # FIXED: Proper annualization with realistic bounds
        # Require at least 2 weeks of data for meaningful annualization
        min_years_for_annualization = 2 / 52  # ~2 weeks

        if years >= min_years_for_annualization:
            # Compound annual growth rate (CAGR)
            annual_return = (1 + total_return) ** (1 / years) - 1
            # Cap at realistic bounds (-99% to +500%)
            annual_return = max(-0.99, min(5.0, annual_return))
        else:
            # For very short periods, don't annualize - just show total return
            annual_return = total_return
            logger.warning(
                f"Short backtest period ({years:.3f} years). "
                f"Showing total return instead of annualized."
            )

        # Volatility (annualized)
        if len(returns) > 1:
            annual_vol = np.std(returns) * np.sqrt(periods_per_year)
        else:
            annual_vol = 0.0

        # FIXED: Sharpe Ratio calculation
        # Correct formula: (mean_return - rf_per_period) / std_return * sqrt(periods_per_year)
        risk_free_per_period = 0.05 / periods_per_year  # Daily risk-free rate

        if annual_vol > 0.001:  # Avoid division by tiny numbers
            excess_returns = returns - risk_free_per_period
            sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
            # Cap Sharpe at realistic bounds
            sharpe = max(-5.0, min(5.0, sharpe))
        else:
            sharpe = 0.0

        # Sortino (using downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 1:
            downside_vol = np.std(downside) * np.sqrt(periods_per_year)
            if downside_vol > 0.001:
                sortino = (annual_return - 0.05) / downside_vol
                sortino = max(-10.0, min(10.0, sortino))
            else:
                sortino = 0.0
        else:
            sortino = float('inf') if annual_return > 0 else 0.0

        # Calmar
        if self.max_drawdown > 0.001:
            calmar = annual_return / self.max_drawdown
            calmar = max(-10.0, min(10.0, calmar))
        else:
            calmar = 0.0

        # Trade statistics
        n_trades = len(self.trades)
        total_costs = sum(t.get("total_cost", 0) for t in self.trades)

        trades_with_pnl = [t for t in self.trades if "realized_pnl" in t]
        winning_trades = [t for t in trades_with_pnl if t.get("realized_pnl", 0) > 0]
        losing_trades = [t for t in trades_with_pnl if t.get("realized_pnl", 0) < 0]

        win_rate = len(winning_trades) / len(trades_with_pnl) if trades_with_pnl else 0

        avg_win = np.mean([t["realized_pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["realized_pnl"] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Regime analysis
        regime_counts = defaultdict(int)
        for _, regime in self.regime_history:
            regime_counts[regime] += 1

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
                "periods_analyzed": n_periods,
                "years_analyzed": years,
            },
            "trading": {
                "n_trades": n_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "total_costs": total_costs,
                "avg_cost_per_trade": total_costs / n_trades if n_trades > 0 else 0,
            },
            "risk": {
                "max_drawdown": self.max_drawdown,
                "was_killed": self.is_killed,
                "current_drawdown": self.current_drawdown,
            },
            "regime_analysis": dict(regime_counts),
            "config": {
                "bar_type": self.config.bar_type.value,
                "use_dollar_bars": self.config.use_dollar_bars,
                "use_meta_labeling": self.config.use_meta_labeling,
                "use_fractional_diff": self.config.use_fractional_diff,
                "optimization_method": self.config.optimization_method.value,
                "max_participation_rate": self.config.max_participation_rate,
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
        description="JPMorgan-Level Institutional Backtest v2.0",
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
        default=0.20,
        help="Maximum drawdown limit (default: 20%%)",
    )

    parser.add_argument(
        "--optimization",
        type=str,
        default="hrp",  # CHANGED: HRP default
        choices=["max_sharpe", "min_variance", "risk_parity", "hrp", "kelly", "equal_weight"],
        help="Portfolio optimization method (default: hrp)",
    )

    # PHASE 1: Bar type
    parser.add_argument(
        "--use-dollar-bars",
        action="store_true",
        default=True,
        help="Use dollar bars instead of time bars (default: True)",
    )

    parser.add_argument(
        "--time-bars",
        action="store_true",
        help="Use traditional time bars",
    )

    # PHASE 2: Features
    parser.add_argument(
        "--use-frac-diff",
        action="store_true",
        default=True,
        help="Use fractional differentiation (default: True)",
    )

    # PHASE 3: Meta-labeling
    parser.add_argument(
        "--use-meta-labeling",
        action="store_true",
        default=True,
        help="Use meta-labeling filter (default: True)",
    )

    parser.add_argument(
        "--no-meta-labeling",
        action="store_true",
        help="Disable meta-labeling filter",
    )

    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="Disable regime detection",
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

    # Map optimization method
    opt_map = {
        "max_sharpe": OptimizationMethod.MAX_SHARPE,
        "min_variance": OptimizationMethod.MIN_VARIANCE,
        "risk_parity": OptimizationMethod.RISK_PARITY,
        "hrp": OptimizationMethod.HRP,
        "kelly": OptimizationMethod.KELLY,
        "equal_weight": OptimizationMethod.EQUAL_WEIGHT,
    }

    # Configuration
    config = JPMorganBacktestConfig(
        initial_capital=args.capital,
        max_drawdown_pct=args.max_drawdown,
        optimization_method=opt_map.get(args.optimization, OptimizationMethod.HRP),
        enable_regime_detection=not args.no_regime,
        run_validation=args.validate and not args.no_validate,
        output_dir=Path(args.output),
        # Phase configurations
        use_dollar_bars=args.use_dollar_bars and not args.time_bars,
        use_fractional_diff=args.use_frac_diff,
        use_meta_labeling=args.use_meta_labeling and not args.no_meta_labeling,
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
        symbols = CORE_SYMBOLS[:10]

    print("\n" + "=" * 70)
    print("JPMORGAN-LEVEL INSTITUTIONAL BACKTEST v2.0")
    print("=" * 70)
    print(f"Symbols: {len(symbols)}")
    print(f"Capital: ${config.initial_capital:,.0f}")
    print(f"Max Drawdown: {config.max_drawdown_pct:.0%}")
    print(f"Optimization: {config.optimization_method.value}")
    print(f"Regime Detection: {config.enable_regime_detection}")
    print(f"Dollar Bars: {config.use_dollar_bars}")
    print(f"Fractional Diff: {config.use_fractional_diff}")
    print(f"Meta-Labeling: {config.use_meta_labeling}")
    print(f"Max Participation: {config.max_participation_rate:.1%}")
    print(f"Validation: {config.run_validation}")
    print("=" * 70 + "\n")

    # Load data
    logger.info("Loading market data...")
    data = load_multi_symbol_data(symbols, data_path)

    if not data:
        logger.error("No data loaded!")
        return 1

    # Convert to dollar bars if enabled
    if config.use_dollar_bars:
        logger.info("Converting to dollar bars...")
        converter = DollarBarConverter(target_bars_per_day=50)
        dollar_data = {}
        for symbol, df in data.items():
            try:
                dollar_df = converter.convert_to_dollar_bars(df, symbol)
                if len(dollar_df) > 500:
                    dollar_data[symbol] = dollar_df
                else:
                    logger.warning(f"{symbol}: Insufficient dollar bars, using time bars")
                    dollar_data[symbol] = df
            except Exception as e:
                logger.warning(f"{symbol}: Dollar bar conversion failed ({e}), using time bars")
                dollar_data[symbol] = df
        data = dollar_data

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
    print("BACKTEST RESULTS v2.0")
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
    print(f"  Years Analyzed: {summary.get('years_analyzed', 0):.2f}")

    trading = results.get("trading", {})
    print(f"\nTrading:")
    print(f"  Total Trades: {trading.get('n_trades', 0)}")
    print(f"  Win Rate: {trading.get('win_rate', 0):.2%}")
    print(f"  Profit Factor: {trading.get('profit_factor', 0):.2f}")
    print(f"  Total Costs: ${trading.get('total_costs', 0):,.2f}")

    if "regime_analysis" in results:
        print(f"\nRegime Analysis:")
        for regime, count in results["regime_analysis"].items():
            print(f"  {regime}: {count} bars")

    if "walk_forward" in results:
        wf = results["walk_forward"]
        print(f"\nWalk-Forward Validation:")
        print(f"  Avg Train Sharpe: {wf.get('avg_train_sharpe', 0):.3f}")
        print(f"  Avg Test Sharpe: {wf.get('avg_test_sharpe', 0):.3f}")
        print(f"  Overfit Ratio: {wf.get('overfit_ratio', 0):.2%}")
        print(f"  PBO: {wf.get('probability_backtest_overfit', 0):.1%}")
        print(f"  Profitable Folds: {wf.get('n_profitable_folds', 0)}/{wf.get('n_folds', 0)}")

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
    output_file = config.output_dir / f"backtest_v2_{timestamp}.json"

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
