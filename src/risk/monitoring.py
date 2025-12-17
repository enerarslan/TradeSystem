"""
Risk Monitoring Module.

This module provides real-time and periodic risk monitoring:
- Liquidity Risk Monitoring
- Model Decay Detection
- Position Concentration Monitoring
- Drawdown Alerts

JPMorgan-level requirements:
- Real-time monitoring capability
- Automatic alert generation
- Historical comparison
- Regulatory metrics (LCR, NSFR proxies)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """Single risk alert."""
    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    recommended_action: Optional[str] = None


@dataclass
class LiquidityMetrics:
    """Liquidity risk metrics."""
    # Portfolio-level
    avg_days_to_liquidate: float
    pct_liquidatable_1_day: float
    pct_liquidatable_5_days: float
    concentration_in_illiquid: float

    # Position-level
    positions_illiquid: List[str]
    position_liquidity_scores: Dict[str, float]

    # Market impact estimates
    estimated_market_impact_pct: float
    liquidation_cost_pct: float


@dataclass
class ModelDecayMetrics:
    """Model decay monitoring metrics."""
    # Rolling performance
    sharpe_30d: float
    sharpe_90d: float
    sharpe_365d: float

    # Decay indicators
    ic_30d: float  # Information Coefficient
    ic_90d: float
    ic_decay_pct: float  # Recent IC vs historical

    # Statistical tests
    return_stability_pvalue: float
    is_decaying: bool
    decay_severity: str  # "none", "mild", "moderate", "severe"


@dataclass
class MonitoringReport:
    """Complete monitoring report."""
    timestamp: str
    alerts: List[RiskAlert]
    liquidity_metrics: Optional[LiquidityMetrics]
    model_decay: Optional[ModelDecayMetrics]
    summary: Dict[str, Any]


class LiquidityMonitor:
    """
    Monitors portfolio liquidity risk.

    Tracks:
    - Days to liquidate positions
    - Market impact of liquidation
    - Concentration in illiquid assets
    """

    def __init__(
        self,
        max_participation_rate: float = 0.10,  # Max 10% of ADV
        days_to_liquidate_threshold: int = 5,
    ):
        """
        Initialize liquidity monitor.

        Args:
            max_participation_rate: Maximum acceptable daily participation
            days_to_liquidate_threshold: Warning threshold for liquidation time
        """
        self.max_participation = max_participation_rate
        self.days_threshold = days_to_liquidate_threshold

    def calculate_days_to_liquidate(
        self,
        position_value: float,
        avg_daily_volume: float,
        avg_price: float,
    ) -> float:
        """Calculate days needed to liquidate a position."""
        if avg_daily_volume <= 0 or avg_price <= 0:
            return float('inf')

        adv_value = avg_daily_volume * avg_price
        max_daily_liquidation = adv_value * self.max_participation
        days = position_value / max_daily_liquidation

        return days

    def estimate_market_impact(
        self,
        position_value: float,
        avg_daily_volume: float,
        volatility: float,
    ) -> float:
        """
        Estimate market impact using square-root model.

        Impact = sigma * sqrt(Q / ADV)
        """
        if avg_daily_volume <= 0:
            return 0.20  # 20% impact for very illiquid

        participation = position_value / avg_daily_volume
        impact = volatility * np.sqrt(participation)

        return min(impact, 0.20)  # Cap at 20%

    def assess_liquidity(
        self,
        positions: Dict[str, float],  # Symbol -> position value
        volume_data: Dict[str, float],  # Symbol -> ADV
        price_data: Dict[str, float],  # Symbol -> price
        volatility_data: Optional[Dict[str, float]] = None,
    ) -> LiquidityMetrics:
        """
        Assess portfolio liquidity.

        Returns:
            Complete liquidity metrics
        """
        total_value = sum(positions.values())
        if total_value <= 0:
            return LiquidityMetrics(
                avg_days_to_liquidate=0,
                pct_liquidatable_1_day=1.0,
                pct_liquidatable_5_days=1.0,
                concentration_in_illiquid=0,
                positions_illiquid=[],
                position_liquidity_scores={},
                estimated_market_impact_pct=0,
                liquidation_cost_pct=0,
            )

        days_by_position = {}
        liquidity_scores = {}
        illiquid_positions = []
        total_impact = 0

        for symbol, value in positions.items():
            adv = volume_data.get(symbol, 0)
            price = price_data.get(symbol, 1)
            vol = volatility_data.get(symbol, 0.20) if volatility_data else 0.20

            # Days to liquidate
            days = self.calculate_days_to_liquidate(value, adv, price)
            days_by_position[symbol] = days

            # Liquidity score (1 = very liquid, 0 = illiquid)
            score = max(0, 1 - days / 30)  # Normalize to 30 days
            liquidity_scores[symbol] = score

            if days > self.days_threshold:
                illiquid_positions.append(symbol)

            # Market impact
            impact = self.estimate_market_impact(value, adv * price, vol)
            total_impact += impact * (value / total_value)

        # Calculate metrics
        avg_days = np.average(
            list(days_by_position.values()),
            weights=[positions[s] for s in days_by_position]
        )

        # Percentage liquidatable in N days
        liq_1d = sum(
            v for s, v in positions.items()
            if days_by_position.get(s, float('inf')) <= 1
        ) / total_value

        liq_5d = sum(
            v for s, v in positions.items()
            if days_by_position.get(s, float('inf')) <= 5
        ) / total_value

        # Concentration in illiquid
        illiq_conc = sum(
            positions[s] for s in illiquid_positions
        ) / total_value if illiquid_positions else 0

        return LiquidityMetrics(
            avg_days_to_liquidate=avg_days,
            pct_liquidatable_1_day=liq_1d,
            pct_liquidatable_5_days=liq_5d,
            concentration_in_illiquid=illiq_conc,
            positions_illiquid=illiquid_positions,
            position_liquidity_scores=liquidity_scores,
            estimated_market_impact_pct=total_impact,
            liquidation_cost_pct=total_impact * 2,  # Bid-ask + impact
        )


class ModelDecayMonitor:
    """
    Monitors ML model decay over time.

    Detects:
    - Performance degradation
    - Feature drift
    - Prediction quality decline
    """

    def __init__(
        self,
        decay_threshold: float = 0.30,  # 30% IC decline
        lookback_days: List[int] = [30, 90, 365],
    ):
        """
        Initialize decay monitor.

        Args:
            decay_threshold: Threshold for decay detection
            lookback_days: Windows for comparison
        """
        self.threshold = decay_threshold
        self.lookbacks = lookback_days

    def calculate_information_coefficient(
        self,
        predictions: pd.Series,
        returns: pd.Series,
    ) -> float:
        """Calculate Spearman IC between predictions and returns."""
        from scipy.stats import spearmanr

        common_idx = predictions.index.intersection(returns.index)
        if len(common_idx) < 10:
            return 0.0

        ic, _ = spearmanr(predictions.loc[common_idx], returns.loc[common_idx])
        return ic if not np.isnan(ic) else 0.0

    def assess_decay(
        self,
        returns: pd.Series,
        predictions: Optional[pd.Series] = None,
        historical_sharpe: Optional[float] = None,
    ) -> ModelDecayMetrics:
        """
        Assess model decay.

        Args:
            returns: Strategy return series
            predictions: Model predictions (optional)
            historical_sharpe: Historical Sharpe for comparison

        Returns:
            Model decay metrics
        """
        # Calculate rolling Sharpe ratios
        periods = 252 * 26  # 15-min bars per year

        def rolling_sharpe(rets, window_days):
            window = window_days * 26
            if len(rets) < window:
                return 0.0
            recent = rets.iloc[-window:]
            return recent.mean() / recent.std() * np.sqrt(periods) if recent.std() > 0 else 0

        sharpe_30d = rolling_sharpe(returns, 30)
        sharpe_90d = rolling_sharpe(returns, 90)
        sharpe_365d = rolling_sharpe(returns, 365) if len(returns) > 365 * 26 else sharpe_90d

        # Calculate IC if predictions available
        if predictions is not None:
            ic_30d = self.calculate_information_coefficient(
                predictions.iloc[-30*26:] if len(predictions) > 30*26 else predictions,
                returns.iloc[-30*26:] if len(returns) > 30*26 else returns,
            )
            ic_90d = self.calculate_information_coefficient(
                predictions.iloc[-90*26:] if len(predictions) > 90*26 else predictions,
                returns.iloc[-90*26:] if len(returns) > 90*26 else returns,
            )
        else:
            # Use autocorrelation of returns as proxy
            ic_30d = returns.iloc[-30*26:].autocorr(1) if len(returns) > 30*26 else 0
            ic_90d = returns.iloc[-90*26:].autocorr(1) if len(returns) > 90*26 else 0

        # Calculate decay
        if ic_90d != 0:
            ic_decay = (ic_90d - ic_30d) / abs(ic_90d)
        else:
            ic_decay = 0

        # Statistical test for return stability
        from scipy.stats import ks_2samp

        mid_point = len(returns) // 2
        if mid_point > 100:
            first_half = returns.iloc[:mid_point]
            second_half = returns.iloc[mid_point:]
            _, p_value = ks_2samp(first_half, second_half)
        else:
            p_value = 1.0

        # Determine if decaying
        is_decaying = (
            ic_decay > self.threshold or
            (historical_sharpe and sharpe_30d < historical_sharpe * 0.5) or
            p_value < 0.05
        )

        # Severity
        if ic_decay > 0.5 or sharpe_30d < 0:
            severity = "severe"
        elif ic_decay > 0.3 or sharpe_30d < sharpe_90d * 0.5:
            severity = "moderate"
        elif ic_decay > 0.1:
            severity = "mild"
        else:
            severity = "none"

        return ModelDecayMetrics(
            sharpe_30d=sharpe_30d,
            sharpe_90d=sharpe_90d,
            sharpe_365d=sharpe_365d,
            ic_30d=ic_30d,
            ic_90d=ic_90d,
            ic_decay_pct=ic_decay * 100,
            return_stability_pvalue=p_value,
            is_decaying=is_decaying,
            decay_severity=severity,
        )


class RiskMonitor:
    """
    Comprehensive risk monitoring system.

    Aggregates all monitoring components and generates alerts.
    """

    def __init__(
        self,
        alert_callbacks: Optional[List[Callable[[RiskAlert], None]]] = None,
    ):
        """
        Initialize risk monitor.

        Args:
            alert_callbacks: Functions to call when alerts are generated
        """
        self.liquidity_monitor = LiquidityMonitor()
        self.decay_monitor = ModelDecayMonitor()
        self.alert_callbacks = alert_callbacks or []
        self.alert_history: List[RiskAlert] = []

    def _emit_alert(self, alert: RiskAlert):
        """Emit an alert to all callbacks."""
        self.alert_history.append(alert)
        logger.log(
            logging.CRITICAL if alert.level == AlertLevel.CRITICAL else
            logging.WARNING if alert.level == AlertLevel.WARNING else
            logging.INFO,
            f"RISK ALERT [{alert.category}]: {alert.message}"
        )
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def check_drawdown(
        self,
        equity_curve: pd.Series,
        warn_threshold: float = -0.10,
        critical_threshold: float = -0.20,
    ) -> List[RiskAlert]:
        """Check current drawdown level."""
        alerts = []

        # Calculate current drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        current_dd = drawdown.iloc[-1]

        if current_dd < critical_threshold:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                category="drawdown",
                message=f"CRITICAL: Portfolio drawdown at {current_dd:.1%}",
                metric_name="drawdown",
                metric_value=current_dd,
                threshold=critical_threshold,
                recommended_action="Consider reducing positions immediately",
            )
            alerts.append(alert)
            self._emit_alert(alert)
        elif current_dd < warn_threshold:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                category="drawdown",
                message=f"WARNING: Portfolio drawdown at {current_dd:.1%}",
                metric_name="drawdown",
                metric_value=current_dd,
                threshold=warn_threshold,
                recommended_action="Review positions and risk exposure",
            )
            alerts.append(alert)
            self._emit_alert(alert)

        return alerts

    def check_concentration(
        self,
        positions: Dict[str, float],
        max_single_position: float = 0.10,
        max_sector_exposure: float = 0.30,
        sectors: Optional[Dict[str, str]] = None,
    ) -> List[RiskAlert]:
        """Check position concentration limits."""
        alerts = []
        total_value = sum(positions.values())

        if total_value <= 0:
            return alerts

        # Single position concentration
        for symbol, value in positions.items():
            weight = value / total_value
            if weight > max_single_position:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    category="concentration",
                    message=f"{symbol} concentration at {weight:.1%} (limit: {max_single_position:.1%})",
                    metric_name=f"position_weight_{symbol}",
                    metric_value=weight,
                    threshold=max_single_position,
                    recommended_action=f"Reduce {symbol} position",
                )
                alerts.append(alert)
                self._emit_alert(alert)

        # Sector concentration
        if sectors:
            sector_exposure = {}
            for symbol, value in positions.items():
                sector = sectors.get(symbol, "other")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + value

            for sector, value in sector_exposure.items():
                weight = value / total_value
                if weight > max_sector_exposure:
                    alert = RiskAlert(
                        timestamp=datetime.now(),
                        level=AlertLevel.WARNING,
                        category="concentration",
                        message=f"{sector} sector at {weight:.1%} (limit: {max_sector_exposure:.1%})",
                        metric_name=f"sector_weight_{sector}",
                        metric_value=weight,
                        threshold=max_sector_exposure,
                        recommended_action=f"Reduce {sector} sector exposure",
                    )
                    alerts.append(alert)
                    self._emit_alert(alert)

        return alerts

    def run_full_check(
        self,
        positions: Dict[str, float],
        volume_data: Dict[str, float],
        price_data: Dict[str, float],
        returns: pd.Series,
        equity_curve: pd.Series,
        predictions: Optional[pd.Series] = None,
        sectors: Optional[Dict[str, str]] = None,
    ) -> MonitoringReport:
        """
        Run complete risk monitoring check.

        Returns:
            Full monitoring report with all metrics and alerts
        """
        alerts = []

        # Drawdown check
        alerts.extend(self.check_drawdown(equity_curve))

        # Concentration check
        alerts.extend(self.check_concentration(positions, sectors=sectors))

        # Liquidity assessment
        liquidity = self.liquidity_monitor.assess_liquidity(
            positions, volume_data, price_data
        )

        # Liquidity alerts
        if liquidity.pct_liquidatable_1_day < 0.50:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                category="liquidity",
                message=f"Only {liquidity.pct_liquidatable_1_day:.0%} of portfolio liquidatable in 1 day",
                metric_name="liquidity_1d",
                metric_value=liquidity.pct_liquidatable_1_day,
                threshold=0.50,
                recommended_action="Reduce illiquid positions",
            )
            alerts.append(alert)
            self._emit_alert(alert)

        # Model decay assessment
        decay = self.decay_monitor.assess_decay(returns, predictions)

        # Decay alerts
        if decay.is_decaying and decay.decay_severity in ["moderate", "severe"]:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING if decay.decay_severity == "moderate" else AlertLevel.CRITICAL,
                category="model_decay",
                message=f"Model decay detected: {decay.decay_severity} ({decay.ic_decay_pct:.0f}% IC decline)",
                metric_name="ic_decay",
                metric_value=decay.ic_decay_pct,
                threshold=30,
                recommended_action="Consider model retraining or parameter update",
            )
            alerts.append(alert)
            self._emit_alert(alert)

        # Summary
        summary = {
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a.level == AlertLevel.CRITICAL),
            "warning_alerts": sum(1 for a in alerts if a.level == AlertLevel.WARNING),
            "current_drawdown": (equity_curve.iloc[-1] - equity_curve.max()) / equity_curve.max(),
            "liquidity_score": liquidity.pct_liquidatable_1_day,
            "model_health": "healthy" if not decay.is_decaying else decay.decay_severity,
        }

        return MonitoringReport(
            timestamp=datetime.now().isoformat(),
            alerts=alerts,
            liquidity_metrics=liquidity,
            model_decay=decay,
            summary=summary,
        )
