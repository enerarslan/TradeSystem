"""
PAIRS TRADING STRATEGY
Statistical Arbitrage - Cointegration Based
JPMorgan Quantitative Strategies Division Style

This module implements pairs trading:
- Cointegration detection
- Spread calculation
- Z-score based entry/exit
- Dynamic hedge ratio
- Half-life estimation
- Mean reversion signals

Usage:
    strategy = PairsTradingStrategy(
        symbol_a="AAPL",
        symbol_b="MSFT",
        lookback=60,
        zscore_entry=2.0,
        zscore_exit=0.5
    )
    signal = await strategy.on_tick(tick)
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from strategies.base import BaseStrategy
from data.models import (
    MarketTick, TradeSignal, Side,
    PairSpread, PairSignal
)

# Optional imports
try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy import stats
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# LOGGING
# ============================================================================

try:
    from utils.logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class PairStatus(Enum):
    """Pair trading status"""
    FLAT = "flat"
    LONG_SPREAD = "long_spread"    # Long A, Short B
    SHORT_SPREAD = "short_spread"  # Short A, Long B


class CointMethod(Enum):
    """Cointegration test method"""
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"
    SIMPLE_RATIO = "simple_ratio"


@dataclass
class CointResult:
    """Cointegration analysis result"""
    is_cointegrated: bool = False
    pvalue: float = 1.0
    hedge_ratio: float = 1.0
    half_life: float = 0.0
    correlation: float = 0.0
    spread_std: float = 0.0
    spread_mean: float = 0.0
    test_statistic: float = 0.0
    critical_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class PairPosition:
    """Current pair position"""
    status: PairStatus = PairStatus.FLAT
    entry_zscore: float = 0.0
    entry_spread: float = 0.0
    entry_time: Optional[datetime] = None
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    quantity_a: float = 0.0
    quantity_b: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PairMetrics:
    """Pair trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    avg_holding_period: float = 0.0
    max_zscore_reached: float = 0.0
    convergence_rate: float = 0.0


# ============================================================================
# COINTEGRATION ANALYZER
# ============================================================================

class CointegrationAnalyzer:
    """
    Analyzes cointegration between two price series.
    
    Methods:
    - Engle-Granger two-step method
    - Simple ratio-based spread
    - Half-life estimation
    """
    
    def __init__(
        self,
        min_correlation: float = 0.5,
        max_pvalue: float = 0.05,
        min_half_life: float = 1,
        max_half_life: float = 100
    ):
        self.min_correlation = min_correlation
        self.max_pvalue = max_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
    
    def analyze(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        method: CointMethod = CointMethod.ENGLE_GRANGER
    ) -> CointResult:
        """
        Run cointegration analysis.
        
        Args:
            prices_a: Price series A
            prices_b: Price series B
            method: Cointegration test method
        
        Returns:
            CointResult with analysis results
        """
        result = CointResult()
        
        if len(prices_a) < 30 or len(prices_b) < 30:
            return result
        
        if len(prices_a) != len(prices_b):
            min_len = min(len(prices_a), len(prices_b))
            prices_a = prices_a[-min_len:]
            prices_b = prices_b[-min_len:]
        
        # Calculate correlation
        result.correlation = float(np.corrcoef(prices_a, prices_b)[0, 1])
        
        if abs(result.correlation) < self.min_correlation:
            return result
        
        # Calculate hedge ratio
        if method == CointMethod.ENGLE_GRANGER:
            result = self._engle_granger(prices_a, prices_b, result)
        elif method == CointMethod.SIMPLE_RATIO:
            result = self._simple_ratio(prices_a, prices_b, result)
        else:
            result = self._simple_ratio(prices_a, prices_b, result)
        
        # Calculate half-life
        if result.is_cointegrated:
            result.half_life = self._calculate_half_life(prices_a, prices_b, result.hedge_ratio)
            
            # Validate half-life
            if not (self.min_half_life <= result.half_life <= self.max_half_life):
                result.is_cointegrated = False
        
        return result
    
    def _engle_granger(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        result: CointResult
    ) -> CointResult:
        """Engle-Granger cointegration test"""
        if not HAS_STATSMODELS:
            return self._simple_ratio(prices_a, prices_b, result)
        
        try:
            # Step 1: Regress A on B to get hedge ratio
            model = OLS(prices_a, prices_b).fit()
            result.hedge_ratio = float(model.params[0])
            
            # Step 2: Calculate spread
            spread = prices_a - result.hedge_ratio * prices_b
            result.spread_mean = float(np.mean(spread))
            result.spread_std = float(np.std(spread))
            
            # Step 3: Test spread for stationarity (ADF test)
            adf_result = adfuller(spread, maxlag=1)
            result.test_statistic = float(adf_result[0])
            result.pvalue = float(adf_result[1])
            result.critical_values = {k: float(v) for k, v in adf_result[4].items()}
            
            # Check if cointegrated
            result.is_cointegrated = result.pvalue < self.max_pvalue
            
            # Alternative: Use coint function
            coint_stat, coint_pvalue, _ = coint(prices_a, prices_b)
            if coint_pvalue < result.pvalue:
                result.pvalue = float(coint_pvalue)
                result.is_cointegrated = result.pvalue < self.max_pvalue
            
        except Exception as e:
            log.warning(f"Engle-Granger test failed: {e}")
            return self._simple_ratio(prices_a, prices_b, result)
        
        return result
    
    def _simple_ratio(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        result: CointResult
    ) -> CointResult:
        """Simple ratio-based spread analysis"""
        # Calculate hedge ratio as price ratio
        result.hedge_ratio = float(np.mean(prices_a) / np.mean(prices_b))
        
        # Calculate spread
        spread = prices_a - result.hedge_ratio * prices_b
        result.spread_mean = float(np.mean(spread))
        result.spread_std = float(np.std(spread))
        
        # Simple mean reversion test: spread should oscillate around mean
        # Use variance ratio test approximation
        if result.spread_std > 0:
            normalized_spread = (spread - result.spread_mean) / result.spread_std
            
            # Check for mean reversion tendency
            spread_returns = np.diff(normalized_spread)
            if len(spread_returns) > 0:
                autocorr = np.corrcoef(
                    normalized_spread[:-1],
                    normalized_spread[1:]
                )[0, 1]
                
                # Negative autocorrelation suggests mean reversion
                if autocorr < -0.1:
                    result.is_cointegrated = True
                    result.pvalue = 0.05 * (1 + autocorr)  # Approximate
        
        return result
    
    def _calculate_half_life(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        hedge_ratio: float
    ) -> float:
        """Calculate half-life of mean reversion"""
        spread = prices_a - hedge_ratio * prices_b
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        
        if len(spread_lag) == 0 or len(spread_diff) == 0:
            return 0.0
        
        try:
            # OLS regression: spread_diff = lambda * spread_lag + error
            if HAS_STATSMODELS:
                model = OLS(spread_diff, spread_lag).fit()
                lambda_param = model.params[0]
            else:
                # Simple regression
                lambda_param = np.sum(spread_diff * spread_lag) / np.sum(spread_lag ** 2)
            
            # Half-life = -ln(2) / lambda
            if lambda_param < 0:
                half_life = -np.log(2) / lambda_param
            else:
                half_life = 0.0
            
            return float(max(0, min(1000, half_life)))
            
        except Exception:
            return 0.0


# ============================================================================
# SPREAD CALCULATOR
# ============================================================================

class SpreadCalculator:
    """Calculates and tracks spread metrics"""
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.spread_history = deque(maxlen=lookback)
        self.zscore_history = deque(maxlen=lookback)
    
    def calculate(
        self,
        price_a: float,
        price_b: float,
        hedge_ratio: float,
        spread_mean: float = None,
        spread_std: float = None
    ) -> Tuple[float, float]:
        """
        Calculate spread and z-score.
        
        Returns:
            (spread, zscore)
        """
        spread = price_a - hedge_ratio * price_b
        self.spread_history.append(spread)
        
        # Use provided or calculate from history
        if spread_mean is None:
            spread_mean = np.mean(self.spread_history)
        if spread_std is None:
            spread_std = np.std(self.spread_history)
        
        # Calculate z-score
        if spread_std > 0:
            zscore = (spread - spread_mean) / spread_std
        else:
            zscore = 0.0
        
        self.zscore_history.append(zscore)
        
        return spread, zscore
    
    def get_stats(self) -> Dict[str, float]:
        """Get spread statistics"""
        if len(self.spread_history) == 0:
            return {'mean': 0, 'std': 0, 'zscore': 0}
        
        return {
            'mean': float(np.mean(self.spread_history)),
            'std': float(np.std(self.spread_history)),
            'zscore': float(self.zscore_history[-1]) if self.zscore_history else 0.0,
            'min_zscore': float(np.min(self.zscore_history)) if self.zscore_history else 0.0,
            'max_zscore': float(np.max(self.zscore_history)) if self.zscore_history else 0.0
        }


# ============================================================================
# PAIRS TRADING STRATEGY
# ============================================================================

class PairsTradingStrategy(BaseStrategy):
    """
    Statistical Arbitrage Pairs Trading Strategy.
    
    Entry Logic:
    - When z-score exceeds threshold, enter position
    - If z > entry_threshold: SHORT spread (short A, long B)
    - If z < -entry_threshold: LONG spread (long A, short B)
    
    Exit Logic:
    - When z-score reverts toward mean
    - Stop loss at extreme z-score
    - Time-based exit (max holding period)
    """
    
    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        lookback: int = 60,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 4.0,
        min_half_life: float = 1,
        max_half_life: float = 50,
        max_holding_days: int = 30,
        recalc_interval: int = 20,
        base_quantity: float = 10.0,
        use_dynamic_hedge: bool = True
    ):
        """
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            lookback: Lookback period for calculations
            zscore_entry: Z-score threshold for entry
            zscore_exit: Z-score threshold for exit
            zscore_stop: Z-score for stop loss
            min_half_life: Minimum acceptable half-life
            max_half_life: Maximum acceptable half-life
            max_holding_days: Maximum holding period
            recalc_interval: Recalculate cointegration every N bars
            base_quantity: Base position size
            use_dynamic_hedge: Dynamically adjust hedge ratio
        """
        super().__init__(name=f"PairsTrading_{symbol_a}_{symbol_b}")
        
        self.symbol_a = symbol_a.upper()
        self.symbol_b = symbol_b.upper()
        self.lookback = lookback
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.zscore_stop = zscore_stop
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_holding_days = max_holding_days
        self.recalc_interval = recalc_interval
        self.base_quantity = base_quantity
        self.use_dynamic_hedge = use_dynamic_hedge
        
        # Data buffers
        self.prices_a = deque(maxlen=lookback * 2)
        self.prices_b = deque(maxlen=lookback * 2)
        self.timestamps = deque(maxlen=lookback * 2)
        
        # Cointegration analyzer
        self.coint_analyzer = CointegrationAnalyzer(
            min_half_life=min_half_life,
            max_half_life=max_half_life
        )
        
        # Spread calculator
        self.spread_calc = SpreadCalculator(lookback=lookback)
        
        # Current state
        self.coint_result: Optional[CointResult] = None
        self.position = PairPosition()
        self.metrics = PairMetrics()
        
        # Tick counters
        self.tick_count_a = 0
        self.tick_count_b = 0
        self.bars_since_recalc = 0
        
        # Last prices
        self.last_price_a = 0.0
        self.last_price_b = 0.0
        
        log.info(f"📊 Pairs Trading initialized: {self.symbol_a}/{self.symbol_b}")
        log.info(f"   Entry Z: ±{zscore_entry}, Exit Z: ±{zscore_exit}")
    
    @property
    def pair_name(self) -> str:
        return f"{self.symbol_a}/{self.symbol_b}"
    
    async def on_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """Process tick and generate signals"""
        # Route tick to appropriate symbol
        if tick.symbol.upper() == self.symbol_a:
            self.prices_a.append(tick.price)
            self.last_price_a = tick.price
            self.tick_count_a += 1
        elif tick.symbol.upper() == self.symbol_b:
            self.prices_b.append(tick.price)
            self.last_price_b = tick.price
            self.tick_count_b += 1
        else:
            return None
        
        self.timestamps.append(tick.timestamp)
        
        # Need data for both symbols
        if not self._has_sufficient_data():
            return None
        
        # Recalculate cointegration periodically
        self.bars_since_recalc += 1
        if self.bars_since_recalc >= self.recalc_interval or self.coint_result is None:
            self._update_cointegration()
            self.bars_since_recalc = 0
        
        # Check if pair is tradeable
        if not self._is_pair_tradeable():
            return None
        
        # Calculate current spread and z-score
        spread, zscore = self.spread_calc.calculate(
            self.last_price_a,
            self.last_price_b,
            self.coint_result.hedge_ratio,
            self.coint_result.spread_mean,
            self.coint_result.spread_std
        )
        
        # Update max z-score metric
        if abs(zscore) > self.metrics.max_zscore_reached:
            self.metrics.max_zscore_reached = abs(zscore)
        
        # Generate signal based on position status
        signal = self._generate_signal(zscore, tick.timestamp)
        
        # Update position P&L
        self._update_position_pnl()
        
        return signal
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data for both symbols"""
        min_data = self.lookback
        return len(self.prices_a) >= min_data and len(self.prices_b) >= min_data
    
    def _update_cointegration(self):
        """Update cointegration analysis"""
        prices_a = np.array(self.prices_a)[-self.lookback:]
        prices_b = np.array(self.prices_b)[-self.lookback:]
        
        self.coint_result = self.coint_analyzer.analyze(
            prices_a, prices_b,
            method=CointMethod.ENGLE_GRANGER
        )
        
        if self.coint_result.is_cointegrated:
            log.debug(
                f"📈 {self.pair_name} cointegrated: "
                f"p={self.coint_result.pvalue:.4f}, "
                f"hedge={self.coint_result.hedge_ratio:.4f}, "
                f"half_life={self.coint_result.half_life:.1f}"
            )
    
    def _is_pair_tradeable(self) -> bool:
        """Check if pair is suitable for trading"""
        if self.coint_result is None:
            return False
        
        if not self.coint_result.is_cointegrated:
            return False
        
        if self.coint_result.half_life < self.min_half_life:
            return False
        
        if self.coint_result.half_life > self.max_half_life:
            return False
        
        return True
    
    def _generate_signal(
        self,
        zscore: float,
        timestamp: datetime
    ) -> Optional[TradeSignal]:
        """Generate trading signal based on z-score"""
        
        # Check for exit signals first
        if self.position.status != PairStatus.FLAT:
            exit_signal = self._check_exit_conditions(zscore, timestamp)
            if exit_signal:
                return exit_signal
        
        # Check for entry signals
        if self.position.status == PairStatus.FLAT:
            entry_signal = self._check_entry_conditions(zscore, timestamp)
            if entry_signal:
                return entry_signal
        
        return None
    
    def _check_entry_conditions(
        self,
        zscore: float,
        timestamp: datetime
    ) -> Optional[TradeSignal]:
        """Check for entry conditions"""
        
        # Short spread entry (z-score too high)
        if zscore > self.zscore_entry:
            return self._create_entry_signal(
                PairStatus.SHORT_SPREAD,
                zscore,
                timestamp
            )
        
        # Long spread entry (z-score too low)
        if zscore < -self.zscore_entry:
            return self._create_entry_signal(
                PairStatus.LONG_SPREAD,
                zscore,
                timestamp
            )
        
        return None
    
    def _check_exit_conditions(
        self,
        zscore: float,
        timestamp: datetime
    ) -> Optional[TradeSignal]:
        """Check for exit conditions"""
        
        # Stop loss
        if abs(zscore) > self.zscore_stop:
            log.warning(f"⚠️ {self.pair_name} stop loss triggered at z={zscore:.2f}")
            return self._create_exit_signal(zscore, timestamp, "stop_loss")
        
        # Mean reversion exit
        if self.position.status == PairStatus.LONG_SPREAD:
            if zscore > -self.zscore_exit:
                return self._create_exit_signal(zscore, timestamp, "mean_reversion")
        
        if self.position.status == PairStatus.SHORT_SPREAD:
            if zscore < self.zscore_exit:
                return self._create_exit_signal(zscore, timestamp, "mean_reversion")
        
        # Time-based exit
        if self.position.entry_time:
            holding_days = (timestamp - self.position.entry_time).days
            if holding_days >= self.max_holding_days:
                log.info(f"⏰ {self.pair_name} time exit after {holding_days} days")
                return self._create_exit_signal(zscore, timestamp, "time_exit")
        
        return None
    
    def _create_entry_signal(
        self,
        status: PairStatus,
        zscore: float,
        timestamp: datetime
    ) -> TradeSignal:
        """Create entry signal"""
        
        # Calculate quantities with hedge ratio
        qty_a = self.base_quantity
        qty_b = self.base_quantity * self.coint_result.hedge_ratio
        
        # Determine sides
        if status == PairStatus.LONG_SPREAD:
            side_a = Side.BUY   # Long A
            side_b = Side.SELL  # Short B
        else:  # SHORT_SPREAD
            side_a = Side.SELL  # Short A
            side_b = Side.BUY   # Long B
        
        # Update position
        self.position = PairPosition(
            status=status,
            entry_zscore=zscore,
            entry_spread=self.last_price_a - self.coint_result.hedge_ratio * self.last_price_b,
            entry_time=timestamp,
            entry_price_a=self.last_price_a,
            entry_price_b=self.last_price_b,
            quantity_a=qty_a,
            quantity_b=qty_b
        )
        
        log.info(
            f"📊 {self.pair_name} ENTRY {status.value}: "
            f"z={zscore:.2f}, hedge={self.coint_result.hedge_ratio:.4f}"
        )
        
        # Return signal for primary symbol (A)
        # In production, you'd want to return both legs
        return TradeSignal(
            symbol=self.symbol_a,
            side=side_a,
            price=self.last_price_a,
            quantity=qty_a,
            strategy_name=self.name,
            timestamp=timestamp,
            confidence=min(1.0, abs(zscore) / self.zscore_entry),
            metadata={
                'pair_symbol': self.symbol_b,
                'pair_side': side_b.value,
                'pair_quantity': qty_b,
                'pair_price': self.last_price_b,
                'zscore': zscore,
                'hedge_ratio': self.coint_result.hedge_ratio,
                'signal_type': 'entry',
                'spread_status': status.value
            }
        )
    
    def _create_exit_signal(
        self,
        zscore: float,
        timestamp: datetime,
        reason: str
    ) -> TradeSignal:
        """Create exit signal"""
        
        # Calculate P&L
        current_spread = self.last_price_a - self.coint_result.hedge_ratio * self.last_price_b
        spread_pnl = current_spread - self.position.entry_spread
        
        if self.position.status == PairStatus.SHORT_SPREAD:
            spread_pnl = -spread_pnl
        
        # Update metrics
        self.metrics.total_trades += 1
        self.metrics.total_pnl += spread_pnl * self.position.quantity_a
        
        if spread_pnl > 0:
            self.metrics.winning_trades += 1
        
        # Calculate holding period
        if self.position.entry_time:
            holding_hours = (timestamp - self.position.entry_time).total_seconds() / 3600
        else:
            holding_hours = 0
        
        # Update average holding period
        n = self.metrics.total_trades
        self.metrics.avg_holding_period = (
            (self.metrics.avg_holding_period * (n - 1) + holding_hours) / n
        )
        
        # Determine exit sides (reverse of entry)
        if self.position.status == PairStatus.LONG_SPREAD:
            side_a = Side.SELL  # Close long A
            side_b = Side.BUY   # Close short B
        else:
            side_a = Side.BUY   # Close short A
            side_b = Side.SELL  # Close long B
        
        log.info(
            f"📊 {self.pair_name} EXIT ({reason}): "
            f"z={zscore:.2f}, spread_pnl={spread_pnl:.4f}"
        )
        
        # Reset position
        old_position = self.position
        self.position = PairPosition()
        
        return TradeSignal(
            symbol=self.symbol_a,
            side=side_a,
            price=self.last_price_a,
            quantity=old_position.quantity_a,
            strategy_name=self.name,
            timestamp=timestamp,
            confidence=1.0,
            metadata={
                'pair_symbol': self.symbol_b,
                'pair_side': side_b.value,
                'pair_quantity': old_position.quantity_b,
                'pair_price': self.last_price_b,
                'zscore': zscore,
                'exit_reason': reason,
                'spread_pnl': spread_pnl,
                'signal_type': 'exit'
            }
        )
    
    def _update_position_pnl(self):
        """Update unrealized P&L"""
        if self.position.status == PairStatus.FLAT:
            return
        
        current_spread = self.last_price_a - self.coint_result.hedge_ratio * self.last_price_b
        spread_diff = current_spread - self.position.entry_spread
        
        if self.position.status == PairStatus.SHORT_SPREAD:
            spread_diff = -spread_diff
        
        self.position.unrealized_pnl = spread_diff * self.position.quantity_a
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        spread_stats = self.spread_calc.get_stats()
        
        return {
            'pair': self.pair_name,
            'is_tradeable': self._is_pair_tradeable(),
            'position_status': self.position.status.value,
            'current_zscore': spread_stats['zscore'],
            'unrealized_pnl': self.position.unrealized_pnl,
            'cointegration': {
                'is_cointegrated': self.coint_result.is_cointegrated if self.coint_result else False,
                'pvalue': self.coint_result.pvalue if self.coint_result else 1.0,
                'hedge_ratio': self.coint_result.hedge_ratio if self.coint_result else 1.0,
                'half_life': self.coint_result.half_life if self.coint_result else 0.0,
                'correlation': self.coint_result.correlation if self.coint_result else 0.0
            },
            'spread': spread_stats,
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'win_rate': (
                    self.metrics.winning_trades / self.metrics.total_trades
                    if self.metrics.total_trades > 0 else 0.0
                ),
                'total_pnl': self.metrics.total_pnl,
                'avg_holding_hours': self.metrics.avg_holding_period
            }
        }


# ============================================================================
# PAIRS PORTFOLIO STRATEGY
# ============================================================================

class PairsPortfolioStrategy(BaseStrategy):
    """
    Manages multiple pair trading strategies.
    
    Features:
    - Multiple pairs tracking
    - Dynamic pair selection
    - Portfolio-level risk management
    - Correlation across pairs
    """
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        lookback: int = 60,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        max_pairs: int = 5,
        base_quantity: float = 10.0
    ):
        """
        Args:
            pairs: List of (symbol_a, symbol_b) tuples
            lookback: Lookback period
            zscore_entry: Entry threshold
            zscore_exit: Exit threshold
            max_pairs: Maximum simultaneous pairs
            base_quantity: Base quantity per pair
        """
        super().__init__(name="PairsPortfolio")
        
        self.max_pairs = max_pairs
        self.base_quantity = base_quantity
        
        # Create individual pair strategies
        self.pair_strategies: Dict[str, PairsTradingStrategy] = {}
        
        for symbol_a, symbol_b in pairs:
            pair_name = f"{symbol_a}/{symbol_b}"
            self.pair_strategies[pair_name] = PairsTradingStrategy(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                lookback=lookback,
                zscore_entry=zscore_entry,
                zscore_exit=zscore_exit,
                base_quantity=base_quantity
            )
        
        # Active pairs count
        self.active_pairs_count = 0
        
        log.info(f"📊 Pairs Portfolio initialized with {len(pairs)} pairs")
    
    async def on_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """Process tick across all pairs"""
        signals = []
        
        for pair_name, strategy in self.pair_strategies.items():
            # Check if either symbol matches
            if tick.symbol.upper() in [strategy.symbol_a, strategy.symbol_b]:
                signal = await strategy.on_tick(tick)
                
                if signal:
                    # Check max pairs limit
                    if self._can_open_new_pair(strategy, signal):
                        signals.append(signal)
        
        # Return first signal (could be modified to return best)
        return signals[0] if signals else None
    
    def _can_open_new_pair(
        self,
        strategy: PairsTradingStrategy,
        signal: TradeSignal
    ) -> bool:
        """Check if we can open a new pair position"""
        is_entry = signal.metadata.get('signal_type') == 'entry'
        
        if is_entry:
            # Count active pairs
            active = sum(
                1 for s in self.pair_strategies.values()
                if s.position.status != PairStatus.FLAT
            )
            return active < self.max_pairs
        
        return True  # Always allow exits
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get portfolio-wide status"""
        pair_statuses = {
            name: strategy.get_status()
            for name, strategy in self.pair_strategies.items()
        }
        
        active_pairs = sum(
            1 for s in pair_statuses.values()
            if s['position_status'] != 'flat'
        )
        
        total_pnl = sum(
            s['metrics']['total_pnl']
            for s in pair_statuses.values()
        )
        
        tradeable_pairs = sum(
            1 for s in pair_statuses.values()
            if s['is_tradeable']
        )
        
        return {
            'total_pairs': len(self.pair_strategies),
            'active_pairs': active_pairs,
            'tradeable_pairs': tradeable_pairs,
            'total_pnl': total_pnl,
            'pairs': pair_statuses
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PairsTradingStrategy',
    'PairsPortfolioStrategy',
    'CointegrationAnalyzer',
    'SpreadCalculator',
    'PairStatus',
    'CointMethod',
    'CointResult',
    'PairPosition',
    'PairMetrics'
]