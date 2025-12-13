"""
Concept Drift Monitoring System
MLOps Drift Detection and Model Health Monitoring

Implements Population Stability Index (PSI) and other drift detection
methods to identify when production data has shifted from training data.

Features:
- PSI (Population Stability Index) calculation
- Feature-level drift detection
- Automatic drift alerts
- Model performance degradation detection
- Trading halt triggers for severe drift
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..utils.logger import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class DriftType(Enum):
    """Types of drift detected"""
    NONE = "NONE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


@dataclass
class DriftAlert:
    """
    Drift alert raised when distribution shift is detected.

    Attributes:
        alert_id: Unique alert identifier
        timestamp: When drift was detected
        drift_type: Severity of drift
        psi_value: Calculated PSI value
        feature_name: Which feature(s) drifted
        threshold: Threshold that was exceeded
        recommendation: Suggested action
        halt_trading: Whether to halt trading
    """
    alert_id: str
    timestamp: datetime
    drift_type: DriftType
    psi_value: float
    feature_name: str
    threshold: float
    recommendation: str
    halt_trading: bool = False
    acknowledged: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        return (
            f"DriftAlert({self.drift_type.value}): "
            f"{self.feature_name} PSI={self.psi_value:.4f} "
            f"[halt={self.halt_trading}]"
        )


@dataclass
class FeatureDriftReport:
    """
    Detailed report of drift analysis for a feature.

    Attributes:
        feature_name: Name of the feature
        psi: Population Stability Index
        ks_statistic: Kolmogorov-Smirnov test statistic
        mean_shift: Change in mean (percentage)
        std_shift: Change in standard deviation (percentage)
        training_stats: Statistics from training data
        production_stats: Statistics from production data
        drift_detected: Whether drift was detected
    """
    feature_name: str
    psi: float
    ks_statistic: float
    mean_shift: float
    std_shift: float
    training_stats: Dict[str, float]
    production_stats: Dict[str, float]
    drift_detected: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def severity(self) -> DriftType:
        """Determine drift severity based on PSI"""
        if self.psi < 0.1:
            return DriftType.NONE
        elif self.psi < 0.2:
            return DriftType.MINOR
        elif self.psi < 0.25:
            return DriftType.MODERATE
        elif self.psi < 0.4:
            return DriftType.SEVERE
        else:
            return DriftType.CRITICAL


class PSICalculator:
    """
    Population Stability Index (PSI) Calculator.

    PSI measures how much a distribution has shifted between two time periods.
    It's widely used in credit scoring and financial ML to detect model drift.

    PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)

    Interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 ≤ PSI < 0.2: Minor shift (monitor)
    - PSI ≥ 0.2: Significant shift (action required)
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_pct: float = 0.0001
    ):
        """
        Initialize PSI Calculator.

        Args:
            n_bins: Number of bins for discretization
            min_pct: Minimum percentage to avoid log(0)
        """
        self.n_bins = n_bins
        self.min_pct = min_pct

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate PSI between expected and actual distributions.

        Args:
            expected: Expected (training) distribution
            actual: Actual (production) distribution
            buckets: Optional pre-defined bucket boundaries

        Returns:
            PSI value
        """
        expected = np.asarray(expected).flatten()
        actual = np.asarray(actual).flatten()

        # Remove NaN/Inf
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create bins based on expected distribution
        if buckets is None:
            buckets = self._create_bins(expected)

        # Calculate bucket percentages
        expected_pct = self._bucket_percentages(expected, buckets)
        actual_pct = self._bucket_percentages(actual, buckets)

        # Calculate PSI for each bucket
        psi = 0.0
        for i in range(len(expected_pct)):
            exp_pct = max(expected_pct[i], self.min_pct)
            act_pct = max(actual_pct[i], self.min_pct)

            psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)

        return psi

    def _create_bins(self, data: np.ndarray) -> np.ndarray:
        """Create bins based on quantiles of the data"""
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.percentile(data, percentiles)

        # Ensure bins are unique
        bins = np.unique(bins)

        # Add small margins to include all data
        bins[0] = bins[0] - 0.001
        bins[-1] = bins[-1] + 0.001

        return bins

    def _bucket_percentages(
        self,
        data: np.ndarray,
        bins: np.ndarray
    ) -> np.ndarray:
        """Calculate percentage of data in each bucket"""
        counts, _ = np.histogram(data, bins=bins)
        percentages = counts / len(data)

        # Replace zeros with minimum percentage
        percentages = np.maximum(percentages, self.min_pct)

        # Renormalize
        percentages = percentages / percentages.sum()

        return percentages

    def calculate_feature_psi(
        self,
        training_data: pd.DataFrame,
        production_data: pd.DataFrame,
        feature_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate PSI for a specific feature.

        Args:
            training_data: Training data DataFrame
            production_data: Production data DataFrame
            feature_name: Name of the feature to analyze

        Returns:
            Tuple of (PSI value, bucket breakdown)
        """
        if feature_name not in training_data.columns:
            raise ValueError(f"Feature {feature_name} not in training data")
        if feature_name not in production_data.columns:
            raise ValueError(f"Feature {feature_name} not in production data")

        expected = training_data[feature_name].values
        actual = production_data[feature_name].values

        psi = self.calculate_psi(expected, actual)

        # Calculate bucket breakdown for diagnostics
        buckets = self._create_bins(expected)
        expected_pct = self._bucket_percentages(expected, buckets)
        actual_pct = self._bucket_percentages(actual, buckets)

        breakdown = {}
        for i in range(len(buckets) - 1):
            bucket_name = f"[{buckets[i]:.2f}, {buckets[i+1]:.2f})"
            breakdown[bucket_name] = {
                "expected_pct": expected_pct[i],
                "actual_pct": actual_pct[i],
                "contribution": (actual_pct[i] - expected_pct[i]) * np.log(
                    max(actual_pct[i], self.min_pct) / max(expected_pct[i], self.min_pct)
                )
            }

        return psi, breakdown


class ConceptDriftMonitor:
    """
    Comprehensive Concept Drift Monitor.

    Monitors for distribution shifts between training and production data
    across all model features. Raises alerts and can trigger trading halts
    when significant drift is detected.

    Use Cases:
    - Detect when market regime has changed
    - Identify features that have become stale/outdated
    - Trigger model retraining pipelines
    - Halt trading during abnormal conditions
    """

    # PSI thresholds
    PSI_THRESHOLD_MINOR = 0.1
    PSI_THRESHOLD_MODERATE = 0.2
    PSI_THRESHOLD_SEVERE = 0.25
    PSI_THRESHOLD_CRITICAL = 0.4

    def __init__(
        self,
        training_data: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        halt_threshold: float = 0.2,
        check_interval_minutes: int = 15,
        window_size: int = 1000,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None
    ):
        """
        Initialize Concept Drift Monitor.

        Args:
            training_data: Reference training data distribution
            features: List of features to monitor
            halt_threshold: PSI threshold to trigger trading halt
            check_interval_minutes: How often to check for drift
            window_size: Number of recent samples for production distribution
            alert_callback: Callback function for drift alerts
        """
        self.training_data = training_data
        self.features = features or []
        self.halt_threshold = halt_threshold
        self.check_interval_minutes = check_interval_minutes
        self.window_size = window_size
        self.alert_callback = alert_callback

        self.psi_calculator = PSICalculator()

        # Production data buffer
        self._production_buffer: List[pd.Series] = []

        # Alert tracking
        self._alerts: List[DriftAlert] = []
        self._alert_counter = 0

        # State
        self._is_halted = False
        self._last_check_time: Optional[datetime] = None
        self._training_stats: Dict[str, Dict[str, float]] = {}

        # Precompute training statistics
        if training_data is not None:
            self._compute_training_stats()

        logger.info(
            f"ConceptDriftMonitor initialized. "
            f"Features: {len(self.features)}, Halt threshold: {halt_threshold}"
        )

    def set_training_data(self, training_data: pd.DataFrame) -> None:
        """Set or update training data reference"""
        self.training_data = training_data

        if not self.features:
            # Auto-detect numeric features
            self.features = training_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        self._compute_training_stats()
        logger.info(f"Training data updated. Features: {len(self.features)}")

    def _compute_training_stats(self) -> None:
        """Precompute statistics from training data"""
        if self.training_data is None:
            return

        for feature in self.features:
            if feature not in self.training_data.columns:
                continue

            data = self.training_data[feature].dropna()
            self._training_stats[feature] = {
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "q25": data.quantile(0.25),
                "q50": data.quantile(0.50),
                "q75": data.quantile(0.75)
            }

    def add_production_sample(self, sample: pd.Series) -> None:
        """
        Add a production data sample to the monitoring buffer.

        Args:
            sample: Single row of production data
        """
        self._production_buffer.append(sample)

        # Maintain window size
        if len(self._production_buffer) > self.window_size:
            self._production_buffer.pop(0)

    def add_production_batch(self, batch: pd.DataFrame) -> None:
        """
        Add a batch of production data samples.

        Args:
            batch: DataFrame of production data
        """
        for _, row in batch.iterrows():
            self.add_production_sample(row)

    def check_drift(self, force: bool = False) -> List[FeatureDriftReport]:
        """
        Check for concept drift in all monitored features.

        Args:
            force: Force check even if interval hasn't passed

        Returns:
            List of feature drift reports
        """
        now = datetime.utcnow()

        # Check if enough time has passed since last check
        if not force and self._last_check_time is not None:
            elapsed = (now - self._last_check_time).total_seconds() / 60
            if elapsed < self.check_interval_minutes:
                return []

        self._last_check_time = now

        if self.training_data is None:
            logger.warning("No training data set for drift monitoring")
            return []

        if len(self._production_buffer) < 100:
            logger.debug("Insufficient production samples for drift check")
            return []

        # Create production DataFrame from buffer
        production_df = pd.DataFrame(self._production_buffer)

        reports = []
        drift_detected = False

        for feature in self.features:
            if feature not in production_df.columns:
                continue
            if feature not in self.training_data.columns:
                continue

            report = self._analyze_feature_drift(feature, production_df)
            reports.append(report)

            if report.drift_detected:
                drift_detected = True
                self._raise_drift_alert(report)

        # Log summary
        drifted_features = [r.feature_name for r in reports if r.drift_detected]
        if drifted_features:
            logger.warning(
                f"Drift detected in {len(drifted_features)} features: "
                f"{drifted_features[:5]}{'...' if len(drifted_features) > 5 else ''}"
            )
        else:
            logger.info("No significant drift detected")

        return reports

    def _analyze_feature_drift(
        self,
        feature: str,
        production_df: pd.DataFrame
    ) -> FeatureDriftReport:
        """Analyze drift for a single feature"""
        training_values = self.training_data[feature].dropna().values
        production_values = production_df[feature].dropna().values

        # Calculate PSI
        psi = self.psi_calculator.calculate_psi(training_values, production_values)

        # Calculate KS statistic
        try:
            from scipy import stats
            ks_stat, _ = stats.ks_2samp(training_values, production_values)
        except ImportError:
            ks_stat = 0.0

        # Calculate mean and std shifts
        train_mean = np.mean(training_values)
        prod_mean = np.mean(production_values)
        mean_shift = (prod_mean - train_mean) / (train_mean + 1e-10) * 100

        train_std = np.std(training_values)
        prod_std = np.std(production_values)
        std_shift = (prod_std - train_std) / (train_std + 1e-10) * 100

        # Production statistics
        prod_stats = {
            "mean": prod_mean,
            "std": prod_std,
            "min": np.min(production_values),
            "max": np.max(production_values)
        }

        # Determine if drift detected
        drift_detected = psi >= self.PSI_THRESHOLD_MODERATE

        report = FeatureDriftReport(
            feature_name=feature,
            psi=psi,
            ks_statistic=ks_stat,
            mean_shift=mean_shift,
            std_shift=std_shift,
            training_stats=self._training_stats.get(feature, {}),
            production_stats=prod_stats,
            drift_detected=drift_detected
        )

        return report

    def _raise_drift_alert(self, report: FeatureDriftReport) -> DriftAlert:
        """Raise a drift alert for a feature"""
        self._alert_counter += 1
        alert_id = f"DRIFT-{self._alert_counter:05d}"

        # Determine severity and recommendation
        severity = report.severity
        halt_trading = report.psi >= self.halt_threshold

        if severity == DriftType.CRITICAL:
            recommendation = "IMMEDIATE: Halt trading and retrain model"
        elif severity == DriftType.SEVERE:
            recommendation = "URGENT: Investigate feature and consider retraining"
        elif severity == DriftType.MODERATE:
            recommendation = "MONITOR: Feature may need attention soon"
        else:
            recommendation = "INFO: Minor shift detected, continue monitoring"

        alert = DriftAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow(),
            drift_type=severity,
            psi_value=report.psi,
            feature_name=report.feature_name,
            threshold=self.halt_threshold,
            recommendation=recommendation,
            halt_trading=halt_trading
        )

        self._alerts.append(alert)

        # Update halt status
        if halt_trading and not self._is_halted:
            self._is_halted = True
            logger.critical(
                f"TRADING HALTED due to severe drift in {report.feature_name}. "
                f"PSI={report.psi:.4f}"
            )
            audit_logger.log_risk_event(
                event_type="DRIFT_HALT",
                severity="CRITICAL",
                details={
                    "feature": report.feature_name,
                    "psi": report.psi,
                    "alert_id": alert_id
                }
            )

        # Fire callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(str(alert))

        return alert

    def is_halted(self) -> bool:
        """Check if trading is halted due to drift"""
        return self._is_halted

    def resume_trading(self, reason: str = "Manual resume") -> None:
        """Resume trading after drift investigation"""
        if self._is_halted:
            self._is_halted = False
            logger.info(f"Trading resumed: {reason}")
            audit_logger.log_system_event(
                event_type="DRIFT_HALT_CLEARED",
                details={"reason": reason}
            )

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        unacknowledged_only: bool = False
    ) -> List[DriftAlert]:
        """
        Get drift alerts.

        Args:
            since: Only return alerts after this time
            unacknowledged_only: Only return unacknowledged alerts

        Returns:
            List of drift alerts
        """
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a drift alert"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary of feature drift statistics.

        Returns:
            DataFrame with feature drift metrics
        """
        if self.training_data is None:
            return pd.DataFrame()

        if len(self._production_buffer) < 100:
            return pd.DataFrame()

        production_df = pd.DataFrame(self._production_buffer)
        reports = []

        for feature in self.features:
            if feature not in production_df.columns:
                continue
            if feature not in self.training_data.columns:
                continue

            report = self._analyze_feature_drift(feature, production_df)
            reports.append({
                "feature": feature,
                "psi": report.psi,
                "ks_statistic": report.ks_statistic,
                "mean_shift_pct": report.mean_shift,
                "std_shift_pct": report.std_shift,
                "severity": report.severity.value,
                "drift_detected": report.drift_detected
            })

        return pd.DataFrame(reports)

    def reset(self) -> None:
        """Reset the monitor state"""
        self._production_buffer.clear()
        self._alerts.clear()
        self._is_halted = False
        self._last_check_time = None
        logger.info("Drift monitor reset")
