"""
Correlation Circuit Breaker
===========================
JPMorgan-Level Correlation Regime Monitoring

Detects when correlation structure breaks down and triggers risk reduction:

1. Mean correlation spike (all assets moving together)
2. Single factor dominance (first PC explains too much)
3. Correlation regime change (rolling window analysis)
4. VIX correlation (flight to quality detection)

Why This Matters:
- HRP and other allocation methods assume stable correlations
- In crisis, correlations spike to 1.0
- Diversification benefit disappears
- "Diversified" portfolio becomes single concentrated bet

March 2020 Example:
- Normal correlation: 0.3-0.4 between sectors
- Crisis correlation: 0.8-0.95 (everything drops together)
- HRP weights became meaningless within days

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 5
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationState(Enum):
    """Correlation regime state"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class CircuitBreakerAction(Enum):
    """Actions to take when circuit breaker triggers"""
    NONE = "none"
    REDUCE_50 = "reduce_50"       # Reduce all positions by 50%
    REDUCE_75 = "reduce_75"       # Reduce all positions by 75%
    FLATTEN = "flatten"           # Close all positions
    HALT_NEW = "halt_new"         # Only halt new positions


@dataclass
class CorrelationAlert:
    """Alert when correlation anomaly detected"""
    timestamp: datetime
    state: CorrelationState
    action: CircuitBreakerAction
    mean_correlation: float
    baseline_correlation: float
    correlation_change: float
    first_pc_explained: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'state': self.state.value,
            'action': self.action.value,
            'mean_correlation': self.mean_correlation,
            'baseline_correlation': self.baseline_correlation,
            'correlation_change': self.correlation_change,
            'first_pc_explained': self.first_pc_explained,
            'details': self.details
        }


@dataclass
class CorrelationMetrics:
    """Correlation analysis metrics"""
    mean_correlation: float
    median_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_std: float
    first_pc_explained: float
    second_pc_explained: float
    n_high_corr_pairs: int  # Pairs with corr > 0.8
    correlation_matrix: pd.DataFrame

    def to_dict(self) -> Dict[str, float]:
        return {
            'mean_correlation': self.mean_correlation,
            'median_correlation': self.median_correlation,
            'max_correlation': self.max_correlation,
            'min_correlation': self.min_correlation,
            'correlation_std': self.correlation_std,
            'first_pc_explained': self.first_pc_explained,
            'second_pc_explained': self.second_pc_explained,
            'n_high_corr_pairs': self.n_high_corr_pairs
        }


class CorrelationCircuitBreaker:
    """
    Monitors correlation structure and triggers circuit breaker on breakdown.

    Detection Methods:
    1. Mean correlation increase vs baseline
    2. PCA eigenvalue concentration
    3. Rolling correlation regime change
    4. Cross-asset correlation spikes
    """

    def __init__(
        self,
        # Baseline correlation (from training period)
        baseline_correlation: Optional[pd.DataFrame] = None,

        # Trigger thresholds
        correlation_spike_threshold: float = 0.25,   # 25% increase triggers warning
        crisis_threshold: float = 0.40,              # 40% increase triggers crisis
        first_pc_threshold: float = 0.55,            # First PC > 55% is concerning
        first_pc_crisis_threshold: float = 0.70,     # First PC > 70% is crisis

        # Rolling window settings
        lookback_period: int = 20,
        minimum_samples: int = 20,

        # Cooldown settings
        cooldown_periods: int = 10,  # Periods before resetting

        # Alert handlers
        alert_handlers: Optional[List[Callable]] = None
    ):
        self.baseline = baseline_correlation
        self.correlation_spike_threshold = correlation_spike_threshold
        self.crisis_threshold = crisis_threshold
        self.first_pc_threshold = first_pc_threshold
        self.first_pc_crisis_threshold = first_pc_crisis_threshold
        self.lookback = lookback_period
        self.min_samples = minimum_samples
        self.cooldown_periods = cooldown_periods

        # State
        self._state = CorrelationState.NORMAL
        self._is_triggered = False
        self._trigger_time: Optional[datetime] = None
        self._cooldown_counter = 0

        # History
        self._returns_buffer: deque = deque(maxlen=100)
        self._correlation_history: List[CorrelationMetrics] = []
        self._alerts: List[CorrelationAlert] = []

        # Alert handlers
        self._alert_handlers = alert_handlers or []

        # Baseline correlation stats
        self._baseline_mean_corr: float = 0.0
        self._baseline_first_pc: float = 0.0

        if baseline_correlation is not None:
            self._compute_baseline_stats()

    def _compute_baseline_stats(self) -> None:
        """Compute baseline correlation statistics"""
        if self.baseline is None or self.baseline.empty:
            return

        upper_triangle = self.baseline.values[np.triu_indices(len(self.baseline), k=1)]
        self._baseline_mean_corr = np.mean(upper_triangle)

        try:
            eigenvalues = np.linalg.eigvalsh(self.baseline)
            self._baseline_first_pc = eigenvalues[-1] / eigenvalues.sum()
        except Exception:
            self._baseline_first_pc = 0.0

        logger.info(
            f"Baseline correlation: mean={self._baseline_mean_corr:.3f}, "
            f"first_pc={self._baseline_first_pc:.3f}"
        )

    def set_baseline(self, correlation_matrix: pd.DataFrame) -> None:
        """Set or update baseline correlation matrix"""
        self.baseline = correlation_matrix
        self._compute_baseline_stats()

    def update(self, returns: pd.Series) -> None:
        """
        Update with new returns observation.

        Args:
            returns: Series of asset returns for current period
        """
        self._returns_buffer.append(returns)

    def check(
        self,
        returns_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[CorrelationAlert]]:
        """
        Check if correlation circuit breaker should trigger.

        Args:
            returns_df: DataFrame of returns (optional, uses buffer if not provided)

        Returns:
            Tuple of (should_trigger, alert)
        """
        # Get returns data
        if returns_df is not None:
            recent_returns = returns_df
        else:
            if len(self._returns_buffer) < self.min_samples:
                return False, None
            recent_returns = pd.DataFrame(list(self._returns_buffer))

        # Check minimum data
        if len(recent_returns) < self.min_samples:
            return False, None

        # Use lookback window
        if len(recent_returns) > self.lookback:
            recent_returns = recent_returns.iloc[-self.lookback:]

        # Calculate current correlation metrics
        metrics = self._calculate_correlation_metrics(recent_returns)

        # Store for history
        self._correlation_history.append(metrics)
        if len(self._correlation_history) > 100:
            self._correlation_history = self._correlation_history[-100:]

        # Check triggers
        alert = self._check_triggers(metrics)

        if alert:
            self._alerts.append(alert)
            self._send_alert(alert)
            return True, alert

        # Check cooldown / recovery
        if self._is_triggered:
            self._check_recovery(metrics)

        return False, None

    def _calculate_correlation_metrics(
        self,
        returns: pd.DataFrame
    ) -> CorrelationMetrics:
        """Calculate comprehensive correlation metrics"""
        # Correlation matrix
        corr_matrix = returns.corr()

        # Upper triangle values (excluding diagonal)
        n = len(corr_matrix)
        upper_indices = np.triu_indices(n, k=1)
        upper_values = corr_matrix.values[upper_indices]

        # Basic stats
        mean_corr = np.mean(upper_values)
        median_corr = np.median(upper_values)
        max_corr = np.max(upper_values)
        min_corr = np.min(upper_values)
        std_corr = np.std(upper_values)

        # High correlation pairs
        n_high_corr = np.sum(upper_values > 0.8)

        # PCA analysis
        try:
            # Eigenvalue decomposition
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

            total = np.sum(eigenvalues)
            first_pc = eigenvalues[0] / total if total > 0 else 0
            second_pc = eigenvalues[1] / total if total > 0 and len(eigenvalues) > 1 else 0
        except Exception:
            first_pc = 0.0
            second_pc = 0.0

        return CorrelationMetrics(
            mean_correlation=mean_corr,
            median_correlation=median_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
            correlation_std=std_corr,
            first_pc_explained=first_pc,
            second_pc_explained=second_pc,
            n_high_corr_pairs=n_high_corr,
            correlation_matrix=corr_matrix
        )

    def _check_triggers(
        self,
        metrics: CorrelationMetrics
    ) -> Optional[CorrelationAlert]:
        """Check all trigger conditions"""
        details = {}

        # 1. Mean correlation spike
        if self._baseline_mean_corr > 0:
            corr_change = (metrics.mean_correlation - self._baseline_mean_corr) / self._baseline_mean_corr
        else:
            corr_change = 0.0

        # 2. PCA concentration
        first_pc = metrics.first_pc_explained

        # Determine state and action
        new_state = CorrelationState.NORMAL
        action = CircuitBreakerAction.NONE

        # Crisis detection
        if corr_change >= self.crisis_threshold or first_pc >= self.first_pc_crisis_threshold:
            new_state = CorrelationState.CRISIS
            action = CircuitBreakerAction.REDUCE_75

            if first_pc >= 0.80:  # Extreme crisis
                action = CircuitBreakerAction.FLATTEN

            details['trigger'] = 'crisis'
            details['recommendation'] = 'Immediately reduce all positions'

        # Elevated detection
        elif corr_change >= self.correlation_spike_threshold or first_pc >= self.first_pc_threshold:
            new_state = CorrelationState.ELEVATED
            action = CircuitBreakerAction.REDUCE_50

            details['trigger'] = 'elevated'
            details['recommendation'] = 'Reduce position sizes by 50%'

        # High correlation pairs
        if metrics.n_high_corr_pairs > len(metrics.correlation_matrix) * 0.5:
            # More than half of pairs have high correlation
            if new_state == CorrelationState.NORMAL:
                new_state = CorrelationState.ELEVATED
                action = CircuitBreakerAction.HALT_NEW

            details['high_corr_pairs'] = metrics.n_high_corr_pairs
            details['warning'] = 'Many asset pairs highly correlated'

        # State transition
        if new_state != CorrelationState.NORMAL and new_state != self._state:
            self._state = new_state
            self._is_triggered = True
            self._trigger_time = datetime.now()
            self._cooldown_counter = 0

            alert = CorrelationAlert(
                timestamp=datetime.now(),
                state=new_state,
                action=action,
                mean_correlation=metrics.mean_correlation,
                baseline_correlation=self._baseline_mean_corr,
                correlation_change=corr_change,
                first_pc_explained=first_pc,
                details=details
            )

            logger.warning(
                f"Correlation circuit breaker triggered: {new_state.value}, "
                f"action={action.value}, mean_corr={metrics.mean_correlation:.3f}, "
                f"change={corr_change:.1%}, first_pc={first_pc:.1%}"
            )

            return alert

        return None

    def _check_recovery(self, metrics: CorrelationMetrics) -> None:
        """Check if correlation has recovered"""
        if not self._is_triggered:
            return

        # Calculate current deviation from baseline
        if self._baseline_mean_corr > 0:
            corr_change = (metrics.mean_correlation - self._baseline_mean_corr) / self._baseline_mean_corr
        else:
            corr_change = 0.0

        # Recovery conditions
        is_recovered = (
            corr_change < self.correlation_spike_threshold * 0.5 and
            metrics.first_pc_explained < self.first_pc_threshold * 0.9
        )

        if is_recovered:
            self._cooldown_counter += 1

            if self._cooldown_counter >= self.cooldown_periods:
                # Full recovery
                self._state = CorrelationState.RECOVERY
                self._is_triggered = False
                self._cooldown_counter = 0

                logger.info(
                    f"Correlation circuit breaker recovered after "
                    f"{self.cooldown_periods} normal periods"
                )
        else:
            self._cooldown_counter = 0

    def _send_alert(self, alert: CorrelationAlert) -> None:
        """Send alert to handlers"""
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(self, handler: Callable[[CorrelationAlert], None]) -> None:
        """Add an alert handler"""
        self._alert_handlers.append(handler)

    # =========================================================================
    # STATE AND QUERIES
    # =========================================================================

    @property
    def is_triggered(self) -> bool:
        """Whether circuit breaker is currently triggered"""
        return self._is_triggered

    @property
    def current_state(self) -> CorrelationState:
        """Current correlation state"""
        return self._state

    def get_recommended_action(self) -> CircuitBreakerAction:
        """Get current recommended action"""
        if not self._is_triggered:
            return CircuitBreakerAction.NONE

        if self._state == CorrelationState.CRISIS:
            return CircuitBreakerAction.REDUCE_75
        elif self._state == CorrelationState.ELEVATED:
            return CircuitBreakerAction.REDUCE_50
        else:
            return CircuitBreakerAction.NONE

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on correlation state.

        Returns value between 0 and 1 to multiply position sizes.
        """
        if not self._is_triggered:
            return 1.0

        action = self.get_recommended_action()

        if action == CircuitBreakerAction.FLATTEN:
            return 0.0
        elif action == CircuitBreakerAction.REDUCE_75:
            return 0.25
        elif action == CircuitBreakerAction.REDUCE_50:
            return 0.50
        elif action == CircuitBreakerAction.HALT_NEW:
            return 0.0  # For new positions
        else:
            return 1.0

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            'is_triggered': self._is_triggered,
            'state': self._state.value,
            'trigger_time': self._trigger_time.isoformat() if self._trigger_time else None,
            'cooldown_counter': self._cooldown_counter,
            'recommended_action': self.get_recommended_action().value,
            'position_multiplier': self.get_position_multiplier(),
            'baseline_mean_corr': self._baseline_mean_corr,
            'baseline_first_pc': self._baseline_first_pc
        }

    def get_recent_alerts(self, n: int = 10) -> List[CorrelationAlert]:
        """Get recent alerts"""
        return self._alerts[-n:]

    def get_correlation_trend(self) -> Dict[str, List[float]]:
        """Get correlation metrics over time"""
        if not self._correlation_history:
            return {}

        return {
            'mean_correlation': [m.mean_correlation for m in self._correlation_history],
            'first_pc_explained': [m.first_pc_explained for m in self._correlation_history],
            'max_correlation': [m.max_correlation for m in self._correlation_history],
            'n_high_corr_pairs': [m.n_high_corr_pairs for m in self._correlation_history]
        }

    def reset(self) -> None:
        """Reset circuit breaker state"""
        self._state = CorrelationState.NORMAL
        self._is_triggered = False
        self._trigger_time = None
        self._cooldown_counter = 0
        logger.info("Correlation circuit breaker reset")


# =============================================================================
# VIX-BASED CORRELATION MONITOR
# =============================================================================

class VixCorrelationMonitor:
    """
    Monitor correlation with VIX for flight-to-quality detection.

    When all assets become correlated with VIX, it indicates
    risk-off sentiment and potential correlation breakdown.
    """

    def __init__(
        self,
        vix_corr_threshold: float = -0.5,  # Strong negative correlation with VIX
        lookback: int = 20
    ):
        self.vix_corr_threshold = vix_corr_threshold
        self.lookback = lookback

        self._returns_buffer: deque = deque(maxlen=100)
        self._vix_buffer: deque = deque(maxlen=100)

    def update(self, asset_returns: pd.Series, vix_change: float) -> None:
        """Update with new data"""
        self._returns_buffer.append(asset_returns)
        self._vix_buffer.append(vix_change)

    def check(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if assets are becoming correlated with VIX.

        Returns:
            (is_risk_off, correlation_details)
        """
        if len(self._returns_buffer) < self.lookback:
            return False, {}

        returns_df = pd.DataFrame(list(self._returns_buffer)[-self.lookback:])
        vix = pd.Series(list(self._vix_buffer)[-self.lookback:])

        # Correlation of each asset with VIX
        correlations = {}
        for col in returns_df.columns:
            corr = returns_df[col].corr(vix)
            correlations[col] = corr

        # Count assets with strong negative VIX correlation
        n_correlated = sum(1 for c in correlations.values() if c < self.vix_corr_threshold)
        pct_correlated = n_correlated / len(correlations)

        is_risk_off = pct_correlated > 0.5  # More than half correlated with VIX

        return is_risk_off, {
            'correlations': correlations,
            'n_correlated': n_correlated,
            'pct_correlated': pct_correlated,
            'mean_vix_corr': np.mean(list(correlations.values()))
        }


# =============================================================================
# INTEGRATED CORRELATION RISK MANAGER
# =============================================================================

class CorrelationRiskManager:
    """
    Integrated correlation risk management.

    Combines:
    - Correlation circuit breaker
    - VIX correlation monitor
    - Dynamic correlation-based position limits
    """

    def __init__(
        self,
        circuit_breaker: Optional[CorrelationCircuitBreaker] = None,
        vix_monitor: Optional[VixCorrelationMonitor] = None,
        max_correlated_positions: int = 5
    ):
        self.circuit_breaker = circuit_breaker or CorrelationCircuitBreaker()
        self.vix_monitor = vix_monitor or VixCorrelationMonitor()
        self.max_correlated_positions = max_correlated_positions

    def update(
        self,
        asset_returns: pd.Series,
        vix_change: Optional[float] = None
    ) -> None:
        """Update all monitors"""
        self.circuit_breaker.update(asset_returns)

        if vix_change is not None:
            self.vix_monitor.update(asset_returns, vix_change)

    def check_all(
        self,
        returns_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Run all correlation checks"""
        # Circuit breaker
        cb_triggered, cb_alert = self.circuit_breaker.check(returns_df)

        # VIX correlation (if data available)
        vix_risk_off, vix_details = self.vix_monitor.check()

        # Combined assessment
        risk_level = "normal"
        if cb_triggered:
            if self.circuit_breaker.current_state == CorrelationState.CRISIS:
                risk_level = "crisis"
            else:
                risk_level = "elevated"
        elif vix_risk_off:
            risk_level = "elevated"

        return {
            'risk_level': risk_level,
            'circuit_breaker': {
                'triggered': cb_triggered,
                'state': self.circuit_breaker.current_state.value,
                'action': self.circuit_breaker.get_recommended_action().value,
                'position_multiplier': self.circuit_breaker.get_position_multiplier()
            },
            'vix_monitor': {
                'risk_off': vix_risk_off,
                **vix_details
            },
            'alert': cb_alert.to_dict() if cb_alert else None
        }

    def get_position_limit(
        self,
        symbol: str,
        current_positions: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """
        Get maximum allowed position considering correlations.

        Limits positions in highly correlated assets.
        """
        if symbol not in correlation_matrix.columns:
            return 1.0

        # Count positions highly correlated with this symbol
        high_corr_count = 0
        for other_symbol, position in current_positions.items():
            if other_symbol == symbol or other_symbol not in correlation_matrix.columns:
                continue

            corr = correlation_matrix.loc[symbol, other_symbol]
            if abs(corr) > 0.7 and position != 0:
                high_corr_count += 1

        # Reduce allowed position if too many correlated positions
        if high_corr_count >= self.max_correlated_positions:
            return 0.0  # No new position allowed

        # Gradual reduction
        reduction = high_corr_count / self.max_correlated_positions
        return max(0.0, 1.0 - reduction * 0.5)
