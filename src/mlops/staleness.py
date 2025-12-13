"""
Model Staleness Detection System
================================
JPMorgan-Level Model Health Monitoring

Detects when ML models need retraining based on:
1. Model age (time since training)
2. Rolling prediction accuracy
3. Prediction confidence degradation
4. Feature importance stability
5. Prediction distribution shift

Why Staleness Detection Matters:
- Markets are non-stationary
- Relationships between features and targets decay
- Old models can lose edge or become harmful
- Proactive retraining prevents losses

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class StalenessLevel(Enum):
    """Model staleness severity"""
    FRESH = "fresh"           # Model is performing well
    AGING = "aging"           # Early signs of degradation
    STALE = "stale"           # Needs attention
    CRITICAL = "critical"     # Should not be used


@dataclass
class PredictionRecord:
    """Single prediction outcome record"""
    timestamp: datetime
    symbol: str
    prediction: int  # 0, 1, or 2 for direction
    probability: float  # Model confidence
    actual: int  # Actual outcome
    correct: bool

    @property
    def confidence_error(self) -> float:
        """Error between predicted probability and actual outcome"""
        actual_prob = 1.0 if self.correct else 0.0
        return abs(self.probability - actual_prob)


@dataclass
class StalenessReport:
    """Report on model staleness"""
    timestamp: datetime
    model_name: str
    staleness_level: StalenessLevel
    is_stale: bool
    issues: List[str]
    metrics: Dict[str, float]
    recommendation: str
    retrain_urgency: str  # "none", "low", "medium", "high", "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'staleness_level': self.staleness_level.value,
            'is_stale': self.is_stale,
            'issues': self.issues,
            'metrics': self.metrics,
            'recommendation': self.recommendation,
            'retrain_urgency': self.retrain_urgency
        }


@dataclass
class AccuracyWindow:
    """Accuracy metrics for a time window"""
    window_name: str
    n_predictions: int
    accuracy: float
    avg_confidence: float
    calibration_error: float  # Brier score proxy


class ModelStalenessDetector:
    """
    Detects when ML model needs retraining.

    Monitors:
    1. Model age (days since training)
    2. Rolling accuracy vs threshold
    3. Confidence calibration
    4. Accuracy trend (improving or degrading?)
    5. Per-symbol performance
    """

    def __init__(
        self,
        model_name: str,
        model_trained_date: datetime,

        # Age thresholds
        max_age_days: int = 30,
        warning_age_days: int = 21,

        # Accuracy thresholds
        min_accuracy_threshold: float = 0.52,  # Random is 0.33 for 3-class
        warning_accuracy_threshold: float = 0.55,

        # Minimum samples for evaluation
        min_samples_for_eval: int = 100,

        # Confidence calibration threshold
        max_calibration_error: float = 0.15,

        # Trend detection
        trend_window_size: int = 50,
        min_trend_for_alert: float = -0.02,  # 2% accuracy decline

        # History size
        max_history_size: int = 10000
    ):
        self.model_name = model_name
        self.trained_date = model_trained_date

        # Age thresholds
        self.max_age_days = max_age_days
        self.warning_age_days = warning_age_days

        # Accuracy thresholds
        self.min_accuracy_threshold = min_accuracy_threshold
        self.warning_accuracy_threshold = warning_accuracy_threshold

        # Sample requirements
        self.min_samples_for_eval = min_samples_for_eval

        # Calibration
        self.max_calibration_error = max_calibration_error

        # Trend
        self.trend_window_size = trend_window_size
        self.min_trend_for_alert = min_trend_for_alert

        # Prediction history
        self._history: deque = deque(maxlen=max_history_size)

        # Per-symbol tracking
        self._symbol_history: Dict[str, deque] = {}

        # Alert callbacks
        self._alert_handlers: List[Callable] = []

        # Last check
        self._last_report: Optional[StalenessReport] = None

    def record_prediction(
        self,
        symbol: str,
        prediction: int,
        probability: float,
        actual: int
    ) -> None:
        """
        Record a prediction outcome.

        Args:
            symbol: Trading symbol
            prediction: Model's prediction (0, 1, 2)
            probability: Model's confidence
            actual: Actual outcome
        """
        record = PredictionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction=prediction,
            probability=probability,
            actual=actual,
            correct=(prediction == actual)
        )

        self._history.append(record)

        # Per-symbol tracking
        if symbol not in self._symbol_history:
            self._symbol_history[symbol] = deque(maxlen=1000)
        self._symbol_history[symbol].append(record)

    def check_staleness(self) -> StalenessReport:
        """
        Perform comprehensive staleness check.

        Returns:
            StalenessReport with analysis results
        """
        issues = []
        metrics = {}

        # 1. Age check
        age_days = (datetime.now() - self.trained_date).days
        metrics['age_days'] = age_days

        if age_days > self.max_age_days:
            issues.append(f"Model is {age_days} days old (max: {self.max_age_days})")
        elif age_days > self.warning_age_days:
            issues.append(f"Model approaching max age: {age_days}/{self.max_age_days} days")

        # 2. Overall accuracy check
        if len(self._history) >= self.min_samples_for_eval:
            # Recent accuracy (last 100)
            recent = list(self._history)[-100:]
            recent_accuracy = sum(1 for p in recent if p.correct) / len(recent)
            metrics['recent_accuracy_100'] = recent_accuracy

            if recent_accuracy < self.min_accuracy_threshold:
                issues.append(
                    f"Recent accuracy {recent_accuracy:.1%} below minimum "
                    f"{self.min_accuracy_threshold:.1%}"
                )
            elif recent_accuracy < self.warning_accuracy_threshold:
                issues.append(
                    f"Recent accuracy {recent_accuracy:.1%} approaching threshold"
                )

            # Medium-term accuracy (last 500)
            if len(self._history) >= 500:
                medium = list(self._history)[-500:]
                medium_accuracy = sum(1 for p in medium if p.correct) / len(medium)
                metrics['accuracy_500'] = medium_accuracy

            # Long-term accuracy (all available)
            all_accuracy = sum(1 for p in self._history if p.correct) / len(self._history)
            metrics['overall_accuracy'] = all_accuracy

        # 3. Confidence calibration check
        if len(self._history) >= self.min_samples_for_eval:
            calibration_error = self._compute_calibration_error()
            metrics['calibration_error'] = calibration_error

            if calibration_error > self.max_calibration_error:
                issues.append(
                    f"Model miscalibrated: error {calibration_error:.3f} "
                    f"(max: {self.max_calibration_error})"
                )

        # 4. Accuracy trend check
        if len(self._history) >= self.trend_window_size * 2:
            trend = self._compute_accuracy_trend()
            metrics['accuracy_trend'] = trend

            if trend < self.min_trend_for_alert:
                issues.append(
                    f"Accuracy declining: {trend:.1%} per {self.trend_window_size} predictions"
                )

        # 5. Per-symbol performance check
        worst_symbols = self._check_symbol_performance()
        if worst_symbols:
            symbols_str = ', '.join(worst_symbols)
            issues.append(f"Poor performance on symbols: {symbols_str}")
            metrics['struggling_symbols'] = len(worst_symbols)

        # 6. Confidence degradation check
        if len(self._history) >= self.min_samples_for_eval:
            conf_trend = self._compute_confidence_trend()
            metrics['confidence_trend'] = conf_trend

            # If confidence is dropping on correct predictions, model is losing signal
            if conf_trend < -0.05:
                issues.append(f"Model confidence degrading on correct predictions")

        # Determine staleness level
        staleness_level = self._determine_staleness_level(issues, metrics)

        # Generate recommendation
        recommendation, urgency = self._generate_recommendation(staleness_level, issues)

        report = StalenessReport(
            timestamp=datetime.now(),
            model_name=self.model_name,
            staleness_level=staleness_level,
            is_stale=staleness_level in [StalenessLevel.STALE, StalenessLevel.CRITICAL],
            issues=issues,
            metrics=metrics,
            recommendation=recommendation,
            retrain_urgency=urgency
        )

        self._last_report = report

        # Alert if stale
        if report.is_stale:
            self._send_alert(report)

        return report

    def _compute_calibration_error(self) -> float:
        """
        Compute calibration error (simplified Brier score).

        Well-calibrated model: when it says 70% confidence,
        it should be correct ~70% of the time.
        """
        recent = list(self._history)[-500:]

        # Group by confidence buckets
        buckets = {}
        for p in recent:
            bucket = int(p.probability * 10) / 10  # 0.0, 0.1, ..., 1.0
            if bucket not in buckets:
                buckets[bucket] = {'correct': 0, 'total': 0}
            buckets[bucket]['total'] += 1
            if p.correct:
                buckets[bucket]['correct'] += 1

        # Calculate weighted calibration error
        total_error = 0
        total_weight = 0

        for bucket_conf, data in buckets.items():
            if data['total'] >= 10:  # Need enough samples
                actual_accuracy = data['correct'] / data['total']
                error = abs(bucket_conf - actual_accuracy)
                weight = data['total']

                total_error += error * weight
                total_weight += weight

        return total_error / total_weight if total_weight > 0 else 0

    def _compute_accuracy_trend(self) -> float:
        """
        Compute accuracy trend (recent vs earlier).

        Returns change in accuracy (negative = declining).
        """
        history = list(self._history)

        # Compare last N to previous N
        n = self.trend_window_size

        recent = history[-n:]
        earlier = history[-2*n:-n]

        recent_acc = sum(1 for p in recent if p.correct) / len(recent)
        earlier_acc = sum(1 for p in earlier if p.correct) / len(earlier)

        return recent_acc - earlier_acc

    def _compute_confidence_trend(self) -> float:
        """
        Compute trend in confidence for correct predictions.

        If model is losing signal, confidence on correct predictions drops.
        """
        history = list(self._history)

        if len(history) < 200:
            return 0.0

        n = 100

        recent = [p for p in history[-n:] if p.correct]
        earlier = [p for p in history[-2*n:-n] if p.correct]

        if len(recent) < 20 or len(earlier) < 20:
            return 0.0

        recent_conf = np.mean([p.probability for p in recent])
        earlier_conf = np.mean([p.probability for p in earlier])

        return recent_conf - earlier_conf

    def _check_symbol_performance(self) -> List[str]:
        """
        Check for symbols where model is underperforming.

        Returns list of struggling symbols.
        """
        struggling = []

        for symbol, history in self._symbol_history.items():
            if len(history) >= 30:  # Need enough data
                recent = list(history)[-30:]
                accuracy = sum(1 for p in recent if p.correct) / len(recent)

                if accuracy < self.min_accuracy_threshold:
                    struggling.append(symbol)

        return struggling

    def _determine_staleness_level(
        self,
        issues: List[str],
        metrics: Dict[str, float]
    ) -> StalenessLevel:
        """Determine overall staleness level"""
        if not issues:
            return StalenessLevel.FRESH

        # Count severity
        critical_issues = 0
        major_issues = 0

        for issue in issues:
            if 'below minimum' in issue.lower() or 'critical' in issue.lower():
                critical_issues += 1
            elif 'approaching' in issue.lower() or 'degrading' in issue.lower():
                major_issues += 1

        # Check specific metrics
        age = metrics.get('age_days', 0)
        accuracy = metrics.get('recent_accuracy_100', 1.0)

        if critical_issues >= 2 or accuracy < 0.45:
            return StalenessLevel.CRITICAL
        elif critical_issues >= 1 or age > self.max_age_days:
            return StalenessLevel.STALE
        elif major_issues >= 2 or age > self.warning_age_days:
            return StalenessLevel.AGING
        else:
            return StalenessLevel.FRESH

    def _generate_recommendation(
        self,
        level: StalenessLevel,
        issues: List[str]
    ) -> Tuple[str, str]:
        """Generate recommendation based on staleness level"""
        if level == StalenessLevel.CRITICAL:
            return (
                "URGENT: Model performance critically degraded. "
                "Stop live trading with this model and retrain immediately.",
                "critical"
            )
        elif level == StalenessLevel.STALE:
            return (
                "Model is stale. Schedule retraining within 1-2 days. "
                "Consider reducing position sizes until retrained.",
                "high"
            )
        elif level == StalenessLevel.AGING:
            return (
                "Early signs of model degradation detected. "
                "Plan retraining for this week.",
                "medium"
            )
        else:
            return "Model performing within expected parameters.", "none"

    def _send_alert(self, report: StalenessReport) -> None:
        """Send alert to handlers"""
        logger.warning(f"Model staleness alert: {report.staleness_level.value}")
        logger.warning(f"Issues: {report.issues}")

        for handler in self._alert_handlers:
            try:
                handler(report)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(self, handler: Callable[[StalenessReport], None]) -> None:
        """Add an alert handler"""
        self._alert_handlers.append(handler)

    def get_accuracy_windows(self) -> List[AccuracyWindow]:
        """Get accuracy for different time windows"""
        windows = []
        history = list(self._history)

        for window_size, name in [(100, 'last_100'), (500, 'last_500'), (1000, 'last_1000')]:
            if len(history) >= window_size:
                window = history[-window_size:]
                n = len(window)
                accuracy = sum(1 for p in window if p.correct) / n
                avg_conf = np.mean([p.probability for p in window])
                cal_error = np.mean([p.confidence_error for p in window])

                windows.append(AccuracyWindow(
                    window_name=name,
                    n_predictions=n,
                    accuracy=accuracy,
                    avg_confidence=avg_conf,
                    calibration_error=cal_error
                ))

        return windows

    def get_symbol_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by symbol"""
        result = {}

        for symbol, history in self._symbol_history.items():
            if len(history) >= 20:
                h = list(history)
                n = len(h)
                accuracy = sum(1 for p in h if p.correct) / n

                result[symbol] = {
                    'n_predictions': n,
                    'accuracy': accuracy,
                    'avg_confidence': np.mean([p.probability for p in h])
                }

        return result

    def get_last_report(self) -> Optional[StalenessReport]:
        """Get last staleness report"""
        return self._last_report

    def reset(self) -> None:
        """Reset all history"""
        self._history.clear()
        self._symbol_history.clear()
        self._last_report = None


# =============================================================================
# MULTI-MODEL STALENESS MANAGER
# =============================================================================

class ModelStalenessManager:
    """
    Manages staleness detection for multiple models.

    Use when you have:
    - Multiple strategies with different models
    - Ensemble models
    - Symbol-specific models
    """

    def __init__(self, check_interval_minutes: int = 60):
        self.check_interval = check_interval_minutes
        self._detectors: Dict[str, ModelStalenessDetector] = {}
        self._last_check: datetime = datetime.now()

    def register_model(
        self,
        model_name: str,
        trained_date: datetime,
        **kwargs
    ) -> None:
        """Register a model for staleness monitoring"""
        self._detectors[model_name] = ModelStalenessDetector(
            model_name=model_name,
            model_trained_date=trained_date,
            **kwargs
        )
        logger.info(f"Registered model '{model_name}' for staleness monitoring")

    def record_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: int,
        probability: float,
        actual: int
    ) -> None:
        """Record prediction for a model"""
        if model_name not in self._detectors:
            logger.warning(f"Unknown model: {model_name}")
            return

        self._detectors[model_name].record_prediction(
            symbol, prediction, probability, actual
        )

    def check_all_models(self) -> Dict[str, StalenessReport]:
        """Check staleness for all registered models"""
        reports = {}

        for name, detector in self._detectors.items():
            reports[name] = detector.check_staleness()

        self._last_check = datetime.now()

        return reports

    def should_check(self) -> bool:
        """Whether it's time for a staleness check"""
        elapsed = (datetime.now() - self._last_check).total_seconds() / 60
        return elapsed >= self.check_interval

    def get_stale_models(self) -> List[str]:
        """Get list of models that are stale"""
        stale = []
        for name, detector in self._detectors.items():
            report = detector.get_last_report()
            if report and report.is_stale:
                stale.append(name)
        return stale

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status summary for all models"""
        status = {}

        for name, detector in self._detectors.items():
            report = detector.get_last_report()
            if report:
                status[name] = {
                    'staleness_level': report.staleness_level.value,
                    'is_stale': report.is_stale,
                    'retrain_urgency': report.retrain_urgency,
                    'issues_count': len(report.issues)
                }
            else:
                status[name] = {
                    'staleness_level': 'unknown',
                    'is_stale': False,
                    'retrain_urgency': 'unknown',
                    'issues_count': 0
                }

        return status


# =============================================================================
# AUTOMATED RETRAINING TRIGGER
# =============================================================================

class RetrainingTrigger:
    """
    Automatically trigger model retraining based on staleness.

    Integrates with CI/CD or training pipelines.
    """

    def __init__(
        self,
        staleness_manager: ModelStalenessManager,
        retrain_callback: Optional[Callable[[str], None]] = None
    ):
        self.staleness_manager = staleness_manager
        self.retrain_callback = retrain_callback

        # Track retraining requests
        self._pending_retrains: Dict[str, datetime] = {}
        self._completed_retrains: Dict[str, datetime] = {}

    def check_and_trigger(self) -> List[str]:
        """
        Check models and trigger retraining if needed.

        Returns list of models that were triggered for retraining.
        """
        triggered = []

        reports = self.staleness_manager.check_all_models()

        for model_name, report in reports.items():
            # Skip if already pending
            if model_name in self._pending_retrains:
                continue

            # Check if retraining needed
            should_retrain = False

            if report.retrain_urgency == 'critical':
                should_retrain = True
            elif report.retrain_urgency == 'high':
                # Check if we haven't retrained recently
                last_retrain = self._completed_retrains.get(model_name)
                if last_retrain is None or (datetime.now() - last_retrain).days > 7:
                    should_retrain = True

            if should_retrain:
                self._trigger_retrain(model_name)
                triggered.append(model_name)

        return triggered

    def _trigger_retrain(self, model_name: str) -> None:
        """Trigger retraining for a model"""
        logger.info(f"Triggering retraining for model: {model_name}")

        self._pending_retrains[model_name] = datetime.now()

        if self.retrain_callback:
            try:
                self.retrain_callback(model_name)
            except Exception as e:
                logger.error(f"Retrain callback failed: {e}")

    def mark_retrain_complete(self, model_name: str) -> None:
        """Mark a model's retraining as complete"""
        self._pending_retrains.pop(model_name, None)
        self._completed_retrains[model_name] = datetime.now()

        logger.info(f"Retraining complete for model: {model_name}")

    def get_pending_retrains(self) -> Dict[str, datetime]:
        """Get models pending retraining"""
        return self._pending_retrains.copy()
