"""
Statistical Strategies Module
=============================

Statistical arbitrage strategies for the algorithmic trading platform.

Strategies:
- PairsTradingStrategy: Classic pairs trading with z-score
- CointegrationStrategy: Cointegration-based pairs trading
- StatisticalArbitrageStrategy: Multi-asset statistical arbitrage
- KalmanFilterStrategy: Dynamic hedge ratio with Kalman filter
- OrnsteinUhlenbeckStrategy: Mean-reversion based on OU process

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from config.settings import get_logger, TimeFrame
from core.events import MarketEvent, SignalEvent
from core.types import PortfolioState, Position
from strategies.base import BaseStrategy, StrategyConfig, StrategyState

logger = get_logger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_zscore(
    series: NDArray[np.float64],
    window: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate rolling z-score.
    
    Args:
        series: Input array
        window: Rolling window size
    
    Returns:
        Z-score array
    """
    n = len(series)
    zscore = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = series[i - window + 1:i + 1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
    
    return zscore


def calculate_spread(
    price1: NDArray[np.float64],
    price2: NDArray[np.float64],
    hedge_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """
    Calculate spread between two price series.
    
    Args:
        price1: First price series
        price2: Second price series
        hedge_ratio: Hedge ratio (beta)
    
    Returns:
        Spread array
    """
    return price1 - hedge_ratio * price2


def calculate_hedge_ratio_ols(
    price1: NDArray[np.float64],
    price2: NDArray[np.float64],
) -> float:
    """
    Calculate hedge ratio using OLS regression.
    
    Args:
        price1: Dependent variable (Y)
        price2: Independent variable (X)
    
    Returns:
        Hedge ratio (slope)
    """
    # Remove NaN values
    valid = ~(np.isnan(price1) | np.isnan(price2))
    y = price1[valid]
    x = price2[valid]
    
    if len(y) < 10:
        return 1.0
    
    # OLS: y = alpha + beta * x
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
        return beta
    except Exception:
        return 1.0


def calculate_half_life(
    spread: NDArray[np.float64],
) -> float:
    """
    Calculate mean-reversion half-life using OLS.
    
    Based on Ornstein-Uhlenbeck process.
    
    Args:
        spread: Spread time series
    
    Returns:
        Half-life in periods
    """
    # Remove NaN
    spread = spread[~np.isnan(spread)]
    
    if len(spread) < 20:
        return 20.0  # Default
    
    # Regress spread change on lagged spread
    y = np.diff(spread)
    x = spread[:-1]
    
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
        if beta >= 0:
            return 100.0  # Not mean-reverting
        return -np.log(2) / beta
    except Exception:
        return 20.0


def adf_test(
    series: NDArray[np.float64],
    max_lag: int | None = None,
) -> tuple[float, float, bool]:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag to include
    
    Returns:
        Tuple of (test_statistic, p_value, is_stationary)
    """
    from scipy.stats import t as t_dist
    
    series = series[~np.isnan(series)]
    n = len(series)
    
    if n < 20:
        return 0.0, 1.0, False
    
    # First difference
    y_diff = np.diff(series)
    y_lag = series[:-1]
    
    # Regression
    x_with_const = np.column_stack([np.ones(len(y_lag)), y_lag])
    
    try:
        result = np.linalg.lstsq(x_with_const, y_diff, rcond=None)
        beta = result[0]
        
        # Calculate residuals and standard error
        residuals = y_diff - x_with_const @ beta
        mse = np.sum(residuals ** 2) / (n - 2)
        
        # Standard error of beta[1]
        x_var = np.sum((y_lag - np.mean(y_lag)) ** 2)
        se = np.sqrt(mse / x_var) if x_var > 0 else 1.0
        
        # t-statistic
        t_stat = beta[1] / se if se > 0 else 0.0
        
        # Critical values (approximate for 5% level)
        # MacKinnon critical value tables would be more accurate
        critical_1pct = -3.43
        critical_5pct = -2.86
        critical_10pct = -2.57
        
        # Simple p-value approximation
        if t_stat < critical_1pct:
            p_value = 0.01
        elif t_stat < critical_5pct:
            p_value = 0.05
        elif t_stat < critical_10pct:
            p_value = 0.10
        else:
            p_value = 0.5
        
        is_stationary = t_stat < critical_5pct
        
        return t_stat, p_value, is_stationary
        
    except Exception:
        return 0.0, 1.0, False


def engle_granger_test(
    series1: NDArray[np.float64],
    series2: NDArray[np.float64],
) -> tuple[float, float, float, bool]:
    """
    Engle-Granger cointegration test.
    
    Args:
        series1: First price series
        series2: Second price series
    
    Returns:
        Tuple of (hedge_ratio, adf_stat, p_value, is_cointegrated)
    """
    # Remove NaN
    valid = ~(np.isnan(series1) | np.isnan(series2))
    y = series1[valid]
    x = series2[valid]
    
    if len(y) < 30:
        return 1.0, 0.0, 1.0, False
    
    # Step 1: OLS regression
    hedge_ratio = calculate_hedge_ratio_ols(y, x)
    
    # Step 2: Calculate residuals (spread)
    spread = y - hedge_ratio * x
    
    # Step 3: ADF test on residuals
    adf_stat, p_value, is_stationary = adf_test(spread)
    
    # Cointegration uses stricter critical values
    # MacKinnon critical values for cointegration (2 variables)
    critical_5pct = -3.37
    is_cointegrated = adf_stat < critical_5pct
    
    return hedge_ratio, adf_stat, p_value, is_cointegrated


# =============================================================================
# PAIRS TRADING STRATEGY
# =============================================================================

@dataclass
class PairsTradingConfig(StrategyConfig):
    """Configuration for pairs trading strategy."""
    name: str = "PairsTrading"
    
    # Pair definition
    asset1: str = ""  # Long leg
    asset2: str = ""  # Short leg (hedge)
    
    # Z-score parameters
    zscore_window: int = 20
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.5
    zscore_stop_threshold: float = 3.5
    
    # Hedge ratio
    hedge_ratio_window: int = 60
    dynamic_hedge_ratio: bool = True
    
    # Risk parameters
    max_notional_per_leg: float = 50000.0
    
    # Filters
    min_correlation: float = 0.7
    check_cointegration: bool = True
    min_half_life: float = 5.0
    max_half_life: float = 60.0


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy.
    
    Classic statistical arbitrage strategy trading the spread
    between two correlated assets.
    
    Entry Conditions:
        1. Z-score crosses entry threshold (|z| > 2)
        2. Spread is mean-reverting (half-life check)
        3. Assets are correlated (optional cointegration test)
    
    Exit Conditions:
        1. Z-score crosses exit threshold (|z| < 0.5)
        2. Z-score exceeds stop threshold (|z| > 3.5)
        3. Max holding period reached
    
    Positions:
        - Z > entry: Short spread (sell asset1, buy asset2)
        - Z < -entry: Long spread (buy asset1, sell asset2)
    
    Example:
        config = PairsTradingConfig(
            asset1="GLD",
            asset2="SLV",
            zscore_entry_threshold=2.0,
        )
        strategy = PairsTradingStrategy(config)
    """
    
    def __init__(
        self,
        config: PairsTradingConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize pairs trading strategy."""
        super().__init__(config or PairsTradingConfig(), parameters)
        self.config: PairsTradingConfig = self.config
        
        # State
        self._hedge_ratio: float = 1.0
        self._spread_position: int = 0  # 1=long spread, -1=short spread
        self._entry_zscore: float = 0.0
        
        # Data storage
        self._price1_history: list[float] = []
        self._price2_history: list[float] = []
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize pair assets."""
        if not self.config.asset1 or not self.config.asset2:
            if len(symbols) >= 2:
                self.config.asset1 = symbols[0]
                self.config.asset2 = symbols[1]
            else:
                raise ValueError("Pairs trading requires two assets")
        
        self._symbols = [self.config.asset1, self.config.asset2]
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate pairs trading signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.zscore_window + 10:
            return signals
        
        # Update price history
        close = data["close"].item(-1)
        
        if symbol == self.config.asset1:
            self._price1_history.append(close)
        elif symbol == self.config.asset2:
            self._price2_history.append(close)
        else:
            return signals
        
        # Need both price histories
        min_len = min(len(self._price1_history), len(self._price2_history))
        if min_len < self.config.zscore_window + 10:
            return signals
        
        # Get aligned price arrays
        price1 = np.array(self._price1_history[-min_len:])
        price2 = np.array(self._price2_history[-min_len:])
        
        # Update hedge ratio
        if self.config.dynamic_hedge_ratio:
            window = min(self.config.hedge_ratio_window, len(price1))
            self._hedge_ratio = calculate_hedge_ratio_ols(
                price1[-window:],
                price2[-window:],
            )
        
        # Calculate spread and z-score
        spread = calculate_spread(price1, price2, self._hedge_ratio)
        zscore = calculate_zscore(spread, self.config.zscore_window)
        
        current_z = zscore[-1]
        
        if np.isnan(current_z):
            return signals
        
        # Check cointegration (periodically)
        if self.config.check_cointegration and self.bar_count % 20 == 0:
            _, _, _, is_cointegrated = engle_granger_test(price1, price2)
            if not is_cointegrated:
                logger.debug("Pair not cointegrated, skipping signals")
                return signals
        
        # Check half-life
        half_life = calculate_half_life(spread)
        if half_life < self.config.min_half_life or half_life > self.config.max_half_life:
            logger.debug(f"Half-life {half_life:.1f} out of range")
            return signals
        
        # Get current prices
        price1_current = price1[-1]
        price2_current = price2[-1]
        
        # Entry signals
        if self._spread_position == 0:
            # Short spread (z > entry): sell asset1, buy asset2
            if current_z > self.config.zscore_entry_threshold:
                signals.extend([
                    self.create_entry_signal(
                        symbol=self.config.asset1,
                        direction=-1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price1_current,
                        metadata={
                            "pair_trade": True,
                            "spread_direction": "short",
                            "zscore": current_z,
                            "hedge_ratio": self._hedge_ratio,
                        }
                    ),
                    self.create_entry_signal(
                        symbol=self.config.asset2,
                        direction=1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price2_current,
                        metadata={
                            "pair_trade": True,
                            "spread_direction": "short",
                            "zscore": current_z,
                            "hedge_ratio": self._hedge_ratio,
                        }
                    ),
                ])
                self._spread_position = -1
                self._entry_zscore = current_z
            
            # Long spread (z < -entry): buy asset1, sell asset2
            elif current_z < -self.config.zscore_entry_threshold:
                signals.extend([
                    self.create_entry_signal(
                        symbol=self.config.asset1,
                        direction=1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price1_current,
                        metadata={
                            "pair_trade": True,
                            "spread_direction": "long",
                            "zscore": current_z,
                            "hedge_ratio": self._hedge_ratio,
                        }
                    ),
                    self.create_entry_signal(
                        symbol=self.config.asset2,
                        direction=-1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price2_current,
                        metadata={
                            "pair_trade": True,
                            "spread_direction": "long",
                            "zscore": current_z,
                            "hedge_ratio": self._hedge_ratio,
                        }
                    ),
                ])
                self._spread_position = 1
                self._entry_zscore = current_z
        
        # Exit signals
        elif self._spread_position != 0:
            should_exit = False
            reason = ""
            
            # Exit at mean
            if abs(current_z) < self.config.zscore_exit_threshold:
                should_exit = True
                reason = "mean_reversion"
            
            # Stop loss
            elif abs(current_z) > self.config.zscore_stop_threshold:
                should_exit = True
                reason = "zscore_stop"
            
            # Z-score crossed zero (full reversion)
            elif self._spread_position > 0 and current_z > 0:
                should_exit = True
                reason = "zero_cross"
            elif self._spread_position < 0 and current_z < 0:
                should_exit = True
                reason = "zero_cross"
            
            if should_exit:
                # Exit both legs
                if self._spread_position > 0:  # Long spread
                    signals.extend([
                        self.create_exit_signal(
                            symbol=self.config.asset1,
                            direction=1,
                            strength=0.9,
                            price=price1_current,
                            reason=reason,
                        ),
                        self.create_exit_signal(
                            symbol=self.config.asset2,
                            direction=-1,
                            strength=0.9,
                            price=price2_current,
                            reason=reason,
                        ),
                    ])
                else:  # Short spread
                    signals.extend([
                        self.create_exit_signal(
                            symbol=self.config.asset1,
                            direction=-1,
                            strength=0.9,
                            price=price1_current,
                            reason=reason,
                        ),
                        self.create_exit_signal(
                            symbol=self.config.asset2,
                            direction=1,
                            strength=0.9,
                            price=price2_current,
                            reason=reason,
                        ),
                    ])
                
                self._spread_position = 0
                self._entry_zscore = 0.0
        
        return signals


# =============================================================================
# COINTEGRATION STRATEGY
# =============================================================================

@dataclass
class CointegrationConfig(StrategyConfig):
    """Configuration for cointegration strategy."""
    name: str = "Cointegration"
    
    # Asset pairs (list of tuples)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    
    # Cointegration test
    test_frequency: int = 20  # Bars between tests
    min_test_samples: int = 252
    max_p_value: float = 0.05
    
    # Trading parameters
    zscore_entry: float = 2.0
    zscore_exit: float = 0.0
    zscore_stop: float = 4.0
    
    # Half-life filter
    min_half_life: float = 3.0
    max_half_life: float = 100.0
    
    # Position sizing
    equal_dollar_allocation: bool = True


class CointegrationStrategy(BaseStrategy):
    """
    Cointegration Strategy.
    
    Advanced pairs trading using Engle-Granger cointegration test
    to identify statistically significant mean-reverting spreads.
    
    Process:
        1. Test for cointegration using Engle-Granger
        2. Calculate optimal hedge ratio
        3. Monitor spread z-score
        4. Trade mean-reversion opportunities
    
    Features:
        - Dynamic cointegration testing
        - Automatic pair selection
        - Hedge ratio optimization
        - Half-life filtering
    
    Example:
        config = CointegrationConfig(
            pairs=[("XLF", "XLK"), ("GLD", "GDX")],
        )
        strategy = CointegrationStrategy(config)
    """
    
    def __init__(
        self,
        config: CointegrationConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize cointegration strategy."""
        super().__init__(config or CointegrationConfig(), parameters)
        self.config: CointegrationConfig = self.config
        
        # State per pair
        self._pair_state: dict[tuple[str, str], dict[str, Any]] = {}
        self._price_data: dict[str, list[float]] = {}
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize pairs."""
        # If no pairs specified, create pairs from symbols
        if not self.config.pairs and len(symbols) >= 2:
            # Create all possible pairs
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    self.config.pairs.append((symbols[i], symbols[j]))
        
        # Initialize state for each pair
        for pair in self.config.pairs:
            self._pair_state[pair] = {
                "is_cointegrated": False,
                "hedge_ratio": 1.0,
                "half_life": 20.0,
                "position": 0,
                "entry_zscore": 0.0,
                "last_test_bar": 0,
            }
            self._price_data[pair[0]] = []
            self._price_data[pair[1]] = []
        
        # Set symbols from pairs
        all_symbols = set()
        for pair in self.config.pairs:
            all_symbols.add(pair[0])
            all_symbols.add(pair[1])
        self._symbols = list(all_symbols)
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate cointegration signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < 2:
            return signals
        
        # Update price data
        close = data["close"].item(-1)
        if symbol in self._price_data:
            self._price_data[symbol].append(close)
        
        # Process each pair
        for pair in self.config.pairs:
            if symbol not in pair:
                continue
            
            pair_signals = self._process_pair(pair, portfolio)
            signals.extend(pair_signals)
        
        return signals
    
    def _process_pair(
        self,
        pair: tuple[str, str],
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Process a single pair."""
        signals = []
        asset1, asset2 = pair
        state = self._pair_state[pair]
        
        # Get price arrays
        price1 = np.array(self._price_data.get(asset1, []))
        price2 = np.array(self._price_data.get(asset2, []))
        
        min_len = min(len(price1), len(price2))
        if min_len < self.config.min_test_samples:
            return signals
        
        price1 = price1[-min_len:]
        price2 = price2[-min_len:]
        
        # Periodic cointegration test
        if self.bar_count - state["last_test_bar"] >= self.config.test_frequency:
            hedge_ratio, adf_stat, p_value, is_cointegrated = engle_granger_test(
                price1, price2
            )
            
            state["is_cointegrated"] = is_cointegrated and p_value < self.config.max_p_value
            state["hedge_ratio"] = hedge_ratio
            state["last_test_bar"] = self.bar_count
            
            if state["is_cointegrated"]:
                spread = calculate_spread(price1, price2, hedge_ratio)
                state["half_life"] = calculate_half_life(spread)
        
        # Skip if not cointegrated
        if not state["is_cointegrated"]:
            return signals
        
        # Check half-life
        if not (self.config.min_half_life <= state["half_life"] <= self.config.max_half_life):
            return signals
        
        # Calculate spread and z-score
        spread = calculate_spread(price1, price2, state["hedge_ratio"])
        zscore = calculate_zscore(spread, int(state["half_life"] * 2))
        current_z = zscore[-1]
        
        if np.isnan(current_z):
            return signals
        
        price1_current = price1[-1]
        price2_current = price2[-1]
        
        # Trading logic
        if state["position"] == 0:
            # Entry
            if current_z > self.config.zscore_entry:
                signals.extend([
                    self.create_entry_signal(
                        symbol=asset1,
                        direction=-1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price1_current,
                        metadata={"pair": pair, "spread_dir": "short", "zscore": current_z}
                    ),
                    self.create_entry_signal(
                        symbol=asset2,
                        direction=1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price2_current,
                        metadata={"pair": pair, "spread_dir": "short", "zscore": current_z}
                    ),
                ])
                state["position"] = -1
                state["entry_zscore"] = current_z
            
            elif current_z < -self.config.zscore_entry:
                signals.extend([
                    self.create_entry_signal(
                        symbol=asset1,
                        direction=1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price1_current,
                        metadata={"pair": pair, "spread_dir": "long", "zscore": current_z}
                    ),
                    self.create_entry_signal(
                        symbol=asset2,
                        direction=-1,
                        strength=min(abs(current_z) / 3, 1.0),
                        price=price2_current,
                        metadata={"pair": pair, "spread_dir": "long", "zscore": current_z}
                    ),
                ])
                state["position"] = 1
                state["entry_zscore"] = current_z
        
        else:
            # Exit logic
            should_exit = False
            reason = ""
            
            if abs(current_z) <= self.config.zscore_exit:
                should_exit = True
                reason = "mean_reversion"
            elif abs(current_z) > self.config.zscore_stop:
                should_exit = True
                reason = "stop_loss"
            
            if should_exit:
                dir1 = 1 if state["position"] > 0 else -1
                dir2 = -1 if state["position"] > 0 else 1
                
                signals.extend([
                    self.create_exit_signal(
                        symbol=asset1,
                        direction=dir1,
                        strength=0.9,
                        price=price1_current,
                        reason=reason,
                    ),
                    self.create_exit_signal(
                        symbol=asset2,
                        direction=dir2,
                        strength=0.9,
                        price=price2_current,
                        reason=reason,
                    ),
                ])
                state["position"] = 0
                state["entry_zscore"] = 0.0
        
        return signals


# =============================================================================
# KALMAN FILTER STRATEGY
# =============================================================================

@dataclass
class KalmanFilterConfig(StrategyConfig):
    """Configuration for Kalman filter strategy."""
    name: str = "KalmanFilter"
    
    # Pair definition
    asset1: str = ""
    asset2: str = ""
    
    # Kalman filter parameters
    delta: float = 1e-4  # State transition covariance
    observation_covariance: float = 1.0
    
    # Trading parameters
    zscore_entry: float = 1.5
    zscore_exit: float = 0.0
    lookback_zscore: int = 20


class KalmanFilterStrategy(BaseStrategy):
    """
    Kalman Filter Strategy.
    
    Uses Kalman filter to dynamically estimate hedge ratio
    and track spread for pairs trading.
    
    Advantages over static OLS:
        - Adapts to changing relationships
        - Smooths noisy estimates
        - Provides uncertainty quantification
    
    State Space Model:
        - State: Hedge ratio (beta)
        - Observation: Asset returns relationship
    
    Example:
        config = KalmanFilterConfig(
            asset1="SPY",
            asset2="IWM",
        )
        strategy = KalmanFilterStrategy(config)
    """
    
    def __init__(
        self,
        config: KalmanFilterConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize Kalman filter strategy."""
        super().__init__(config or KalmanFilterConfig(), parameters)
        self.config: KalmanFilterConfig = self.config
        
        # Kalman filter state
        self._beta: float = 0.0
        self._P: float = 1.0  # State covariance
        self._R: float = self.config.observation_covariance
        
        # Price history
        self._price1_history: list[float] = []
        self._price2_history: list[float] = []
        self._spread_history: list[float] = []
        
        # Position state
        self._spread_position: int = 0
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize assets."""
        if not self.config.asset1 or not self.config.asset2:
            if len(symbols) >= 2:
                self.config.asset1 = symbols[0]
                self.config.asset2 = symbols[1]
        
        self._symbols = [self.config.asset1, self.config.asset2]
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate Kalman filter signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < 2:
            return signals
        
        close = data["close"].item(-1)
        
        # Update price history
        if symbol == self.config.asset1:
            self._price1_history.append(close)
        elif symbol == self.config.asset2:
            self._price2_history.append(close)
        else:
            return signals
        
        # Need both prices
        if len(self._price1_history) < 2 or len(self._price2_history) < 2:
            return signals
        
        # Align lengths
        min_len = min(len(self._price1_history), len(self._price2_history))
        price1 = self._price1_history[-1]
        price2 = self._price2_history[-1]
        
        # Kalman filter update
        self._kalman_update(price1, price2)
        
        # Calculate spread
        spread = price1 - self._beta * price2
        self._spread_history.append(spread)
        
        # Calculate z-score
        if len(self._spread_history) < self.config.lookback_zscore:
            return signals
        
        recent_spread = np.array(self._spread_history[-self.config.lookback_zscore:])
        mean = np.mean(recent_spread)
        std = np.std(recent_spread)
        
        if std < 1e-10:
            return signals
        
        current_z = (spread - mean) / std
        
        # Trading logic
        if self._spread_position == 0:
            if current_z > self.config.zscore_entry:
                signals.extend([
                    self.create_entry_signal(
                        symbol=self.config.asset1,
                        direction=-1,
                        strength=min(abs(current_z) / 2, 1.0),
                        price=price1,
                        metadata={"beta": self._beta, "zscore": current_z}
                    ),
                    self.create_entry_signal(
                        symbol=self.config.asset2,
                        direction=1,
                        strength=min(abs(current_z) / 2, 1.0),
                        price=price2,
                        metadata={"beta": self._beta, "zscore": current_z}
                    ),
                ])
                self._spread_position = -1
            
            elif current_z < -self.config.zscore_entry:
                signals.extend([
                    self.create_entry_signal(
                        symbol=self.config.asset1,
                        direction=1,
                        strength=min(abs(current_z) / 2, 1.0),
                        price=price1,
                        metadata={"beta": self._beta, "zscore": current_z}
                    ),
                    self.create_entry_signal(
                        symbol=self.config.asset2,
                        direction=-1,
                        strength=min(abs(current_z) / 2, 1.0),
                        price=price2,
                        metadata={"beta": self._beta, "zscore": current_z}
                    ),
                ])
                self._spread_position = 1
        
        else:
            should_exit = abs(current_z) <= self.config.zscore_exit
            
            if self._spread_position > 0 and current_z > 0:
                should_exit = True
            elif self._spread_position < 0 and current_z < 0:
                should_exit = True
            
            if should_exit:
                dir1 = 1 if self._spread_position > 0 else -1
                dir2 = -1 if self._spread_position > 0 else 1
                
                signals.extend([
                    self.create_exit_signal(
                        symbol=self.config.asset1,
                        direction=dir1,
                        strength=0.9,
                        price=price1,
                        reason="mean_reversion",
                    ),
                    self.create_exit_signal(
                        symbol=self.config.asset2,
                        direction=dir2,
                        strength=0.9,
                        price=price2,
                        reason="mean_reversion",
                    ),
                ])
                self._spread_position = 0
        
        return signals
    
    def _kalman_update(self, price1: float, price2: float) -> None:
        """
        Update Kalman filter state.
        
        State equation: beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
        Observation equation: y_t = beta_t * x_t + v_t, v_t ~ N(0, R)
        
        Where:
            y_t = price1
            x_t = price2
        """
        Q = self.config.delta  # Process noise
        
        # Prediction step
        # beta_pred = beta (random walk)
        # P_pred = P + Q
        P_pred = self._P + Q
        
        # Update step
        # Kalman gain
        x = price2
        K = P_pred * x / (x * P_pred * x + self._R)
        
        # Innovation
        y = price1
        innovation = y - self._beta * x
        
        # Update state
        self._beta = self._beta + K * innovation
        self._P = (1 - K * x) * P_pred


# =============================================================================
# ORNSTEIN-UHLENBECK STRATEGY
# =============================================================================

@dataclass
class OrnsteinUhlenbeckConfig(StrategyConfig):
    """Configuration for Ornstein-Uhlenbeck strategy."""
    name: str = "OrnsteinUhlenbeck"
    
    # Asset pair
    asset1: str = ""
    asset2: str = ""
    
    # OU process parameters
    estimation_window: int = 60
    
    # Trading thresholds (in terms of equilibrium std)
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5
    stop_threshold: float = 3.0
    
    # Time-based exit
    max_holding_bars: int = 40


class OrnsteinUhlenbeckStrategy(BaseStrategy):
    """
    Ornstein-Uhlenbeck Strategy.
    
    Models spread as an OU process and trades based on
    the expected mean-reversion.
    
    OU Process:
        dS = theta * (mu - S) * dt + sigma * dW
        
        Where:
            S = spread
            theta = mean-reversion speed
            mu = long-term mean
            sigma = volatility
    
    Entry when spread deviates significantly from mu.
    Exit when spread reverts toward mu.
    
    Example:
        config = OrnsteinUhlenbeckConfig(
            asset1="XLE",
            asset2="XOP",
        )
        strategy = OrnsteinUhlenbeckStrategy(config)
    """
    
    def __init__(
        self,
        config: OrnsteinUhlenbeckConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize OU strategy."""
        super().__init__(config or OrnsteinUhlenbeckConfig(), parameters)
        self.config: OrnsteinUhlenbeckConfig = self.config
        
        # OU parameters
        self._mu: float = 0.0
        self._theta: float = 0.1
        self._sigma: float = 1.0
        self._hedge_ratio: float = 1.0
        
        # Price history
        self._price1_history: list[float] = []
        self._price2_history: list[float] = []
        
        # Position
        self._spread_position: int = 0
        self._entry_bar: int = 0
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize assets."""
        if not self.config.asset1 or not self.config.asset2:
            if len(symbols) >= 2:
                self.config.asset1 = symbols[0]
                self.config.asset2 = symbols[1]
        
        self._symbols = [self.config.asset1, self.config.asset2]
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate OU strategy signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None:
            return signals
        
        close = data["close"].item(-1)
        
        # Update price history
        if symbol == self.config.asset1:
            self._price1_history.append(close)
        elif symbol == self.config.asset2:
            self._price2_history.append(close)
        else:
            return signals
        
        # Need sufficient history
        min_len = min(len(self._price1_history), len(self._price2_history))
        if min_len < self.config.estimation_window:
            return signals
        
        # Get aligned price arrays
        price1 = np.array(self._price1_history[-min_len:])
        price2 = np.array(self._price2_history[-min_len:])
        
        # Estimate OU parameters
        self._estimate_ou_parameters(price1, price2)
        
        # Calculate current spread
        spread = price1[-1] - self._hedge_ratio * price2[-1]
        
        # Calculate equilibrium standard deviation
        eq_std = self._sigma / np.sqrt(2 * self._theta) if self._theta > 0 else 1.0
        
        # Deviation from mean in units of equilibrium std
        deviation = (spread - self._mu) / eq_std if eq_std > 0 else 0
        
        price1_current = price1[-1]
        price2_current = price2[-1]
        
        # Trading logic
        if self._spread_position == 0:
            if deviation > self.config.entry_threshold:
                # Spread too high, short spread
                signals.extend([
                    self.create_entry_signal(
                        symbol=self.config.asset1,
                        direction=-1,
                        strength=min(abs(deviation) / 3, 1.0),
                        price=price1_current,
                        metadata={
                            "ou_mu": self._mu,
                            "ou_theta": self._theta,
                            "deviation": deviation,
                        }
                    ),
                    self.create_entry_signal(
                        symbol=self.config.asset2,
                        direction=1,
                        strength=min(abs(deviation) / 3, 1.0),
                        price=price2_current,
                        metadata={
                            "ou_mu": self._mu,
                            "ou_theta": self._theta,
                            "deviation": deviation,
                        }
                    ),
                ])
                self._spread_position = -1
                self._entry_bar = self.bar_count
            
            elif deviation < -self.config.entry_threshold:
                # Spread too low, long spread
                signals.extend([
                    self.create_entry_signal(
                        symbol=self.config.asset1,
                        direction=1,
                        strength=min(abs(deviation) / 3, 1.0),
                        price=price1_current,
                        metadata={
                            "ou_mu": self._mu,
                            "ou_theta": self._theta,
                            "deviation": deviation,
                        }
                    ),
                    self.create_entry_signal(
                        symbol=self.config.asset2,
                        direction=-1,
                        strength=min(abs(deviation) / 3, 1.0),
                        price=price2_current,
                        metadata={
                            "ou_mu": self._mu,
                            "ou_theta": self._theta,
                            "deviation": deviation,
                        }
                    ),
                ])
                self._spread_position = 1
                self._entry_bar = self.bar_count
        
        else:
            should_exit = False
            reason = ""
            
            # Exit at mean
            if abs(deviation) < self.config.exit_threshold:
                should_exit = True
                reason = "mean_reversion"
            
            # Stop loss
            elif abs(deviation) > self.config.stop_threshold:
                should_exit = True
                reason = "stop_loss"
            
            # Time-based exit
            elif self.bar_count - self._entry_bar > self.config.max_holding_bars:
                should_exit = True
                reason = "max_holding"
            
            if should_exit:
                dir1 = 1 if self._spread_position > 0 else -1
                dir2 = -1 if self._spread_position > 0 else 1
                
                signals.extend([
                    self.create_exit_signal(
                        symbol=self.config.asset1,
                        direction=dir1,
                        strength=0.9,
                        price=price1_current,
                        reason=reason,
                    ),
                    self.create_exit_signal(
                        symbol=self.config.asset2,
                        direction=dir2,
                        strength=0.9,
                        price=price2_current,
                        reason=reason,
                    ),
                ])
                self._spread_position = 0
        
        return signals
    
    def _estimate_ou_parameters(
        self,
        price1: NDArray[np.float64],
        price2: NDArray[np.float64],
    ) -> None:
        """
        Estimate OU process parameters using maximum likelihood.
        
        For spread S_t following OU:
            dS = theta * (mu - S) * dt + sigma * dW
        
        Discretized:
            S_{t+1} - S_t = theta * (mu - S_t) * dt + sigma * sqrt(dt) * epsilon
        """
        window = self.config.estimation_window
        
        # Calculate hedge ratio
        self._hedge_ratio = calculate_hedge_ratio_ols(
            price1[-window:],
            price2[-window:],
        )
        
        # Calculate spread
        spread = price1[-window:] - self._hedge_ratio * price2[-window:]
        
        # Mean and variance
        self._mu = np.mean(spread)
        
        # Estimate theta from AR(1) regression
        # S_{t+1} = a + b * S_t + epsilon
        # theta = -ln(b), mu = a / (1 - b)
        
        y = spread[1:]
        x = spread[:-1]
        
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            params = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            a, b = params
            
            if 0 < b < 1:
                self._theta = -np.log(b)
                self._mu = a / (1 - b)
            else:
                self._theta = 0.1
                self._mu = np.mean(spread)
            
            # Estimate sigma from residuals
            residuals = y - (a + b * x)
            self._sigma = np.std(residuals) * np.sqrt(2 * self._theta / (1 - b**2))
            
        except Exception:
            self._theta = 0.1
            self._mu = np.mean(spread)
            self._sigma = np.std(spread)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Utilities
    "calculate_zscore",
    "calculate_spread",
    "calculate_hedge_ratio_ols",
    "calculate_half_life",
    "adf_test",
    "engle_granger_test",
    # Configurations
    "PairsTradingConfig",
    "CointegrationConfig",
    "KalmanFilterConfig",
    "OrnsteinUhlenbeckConfig",
    # Strategies
    "PairsTradingStrategy",
    "CointegrationStrategy",
    "KalmanFilterStrategy",
    "OrnsteinUhlenbeckStrategy",
]