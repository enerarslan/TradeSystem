"""
Tests for drift detection module.

Tests verify that PSI, KS tests, and adversarial validation correctly
detect distribution shifts between training and test data.

Section 9: Required test coverage for Directive 3.4.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


class TestPSICalculation:
    """Test Population Stability Index calculation."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 1000

        # Reference distribution (standard normal)
        self.reference = np.random.randn(self.n_samples)

        # Current distribution with no drift
        self.current_no_drift = np.random.randn(self.n_samples)

        # Current distribution with slight drift (shift mean)
        self.current_slight_drift = np.random.randn(self.n_samples) + 0.3

        # Current distribution with significant drift (shift mean + variance)
        self.current_significant_drift = np.random.randn(self.n_samples) * 1.5 + 0.8

    def test_psi_no_drift(self):
        """Test PSI is low when distributions are similar."""
        from src.training.drift_detection import DriftDetector, DriftThresholds

        ref_df = pd.DataFrame({"feature": self.reference})
        cur_df = pd.DataFrame({"feature": self.current_no_drift})

        detector = DriftDetector(
            reference_data=ref_df,
            thresholds=DriftThresholds(),
        )

        psi = detector.calculate_psi(self.reference, self.current_no_drift)

        # PSI should be low (< 0.1) for similar distributions
        assert psi < 0.1, f"PSI should be low for similar distributions, got {psi}"

    def test_psi_slight_drift(self):
        """Test PSI detects slight drift."""
        from src.training.drift_detection import DriftDetector

        ref_df = pd.DataFrame({"feature": self.reference})

        detector = DriftDetector(reference_data=ref_df)

        psi = detector.calculate_psi(self.reference, self.current_slight_drift)

        # PSI should be moderate (0.1 - 0.25) for slight drift
        assert 0.05 < psi < 0.5, f"PSI should detect slight drift, got {psi}"

    def test_psi_significant_drift(self):
        """Test PSI detects significant drift."""
        from src.training.drift_detection import DriftDetector

        ref_df = pd.DataFrame({"feature": self.reference})

        detector = DriftDetector(reference_data=ref_df)

        psi = detector.calculate_psi(self.reference, self.current_significant_drift)

        # PSI should be high (> 0.25) for significant drift
        assert psi > 0.2, f"PSI should detect significant drift, got {psi}"


class TestKSTest:
    """Test Kolmogorov-Smirnov test for distribution comparison."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 500

        # Same distribution
        self.sample1 = np.random.randn(self.n_samples)
        self.sample2 = np.random.randn(self.n_samples)

        # Different distribution
        self.sample_different = np.random.exponential(1, self.n_samples)

    def test_ks_same_distribution(self):
        """Test KS test for samples from same distribution."""
        from src.training.drift_detection import DriftDetector

        ref_df = pd.DataFrame({"feature": self.sample1})
        detector = DriftDetector(reference_data=ref_df)

        ks_stat, p_value = detector.calculate_ks_statistic(self.sample1, self.sample2)

        # p-value should be high (fail to reject null hypothesis)
        assert p_value > 0.05, f"KS test should not reject same distribution, p={p_value}"

    def test_ks_different_distribution(self):
        """Test KS test for samples from different distributions."""
        from src.training.drift_detection import DriftDetector

        ref_df = pd.DataFrame({"feature": self.sample1})
        detector = DriftDetector(reference_data=ref_df)

        ks_stat, p_value = detector.calculate_ks_statistic(self.sample1, self.sample_different)

        # p-value should be low (reject null hypothesis)
        assert p_value < 0.01, f"KS test should reject different distribution, p={p_value}"


class TestDriftDetector:
    """Test comprehensive drift detection."""

    def setup_method(self):
        """Create multi-feature test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 10

        # Reference data
        self.reference = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"feature_{i}" for i in range(self.n_features)],
        )

        # Current data with no drift
        self.current_no_drift = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"feature_{i}" for i in range(self.n_features)],
        )

        # Current data with drift in some features
        self.current_partial_drift = self.current_no_drift.copy()
        # Add drift to features 0, 1, 2
        self.current_partial_drift["feature_0"] += 0.8
        self.current_partial_drift["feature_1"] *= 2.0
        self.current_partial_drift["feature_2"] += 1.0

    def test_detect_no_drift(self):
        """Test detector correctly identifies no drift."""
        from src.training.drift_detection import DriftDetector

        detector = DriftDetector(reference_data=self.reference)
        result = detector.detect_feature_drift(self.current_no_drift)

        # Should detect minimal drift
        assert len(result.affected_features) <= 2, "Should detect few drifted features"

    def test_detect_partial_drift(self):
        """Test detector identifies drifted features."""
        from src.training.drift_detection import DriftDetector, DriftThresholds

        thresholds = DriftThresholds(psi_warning=0.10)
        detector = DriftDetector(reference_data=self.reference, thresholds=thresholds)

        result = detector.detect_feature_drift(self.current_partial_drift)

        # Should detect drift
        assert result.is_drift_detected, "Should detect drift"
        assert len(result.affected_features) >= 2, "Should identify drifted features"

        # Should identify feature_0, feature_1, feature_2 as drifted
        drifted_set = set(result.affected_features)
        expected_drifted = {"feature_0", "feature_1", "feature_2"}
        assert len(drifted_set & expected_drifted) >= 2, "Should identify correct drifted features"

    def test_drift_severity_levels(self):
        """Test drift severity classification."""
        from src.training.drift_detection import DriftSeverity

        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MEDIUM.value == "medium"
        assert DriftSeverity.HIGH.value == "high"
        assert DriftSeverity.CRITICAL.value == "critical"


class TestAdversarialValidation:
    """Test adversarial validation for train/test leakage detection."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 20

        # Similar distributions (should have low AUC)
        self.train_similar = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"f_{i}" for i in range(self.n_features)],
        )
        self.test_similar = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"f_{i}" for i in range(self.n_features)],
        )

        # Different distributions (should have high AUC)
        self.train_different = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"f_{i}" for i in range(self.n_features)],
        )
        self.test_different = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features) + 1.0,  # Shifted
            columns=[f"f_{i}" for i in range(self.n_features)],
        )

    def test_adversarial_similar_distributions(self):
        """Test adversarial AUC is low for similar distributions."""
        from src.training.validation import AdversarialValidator

        validator = AdversarialValidator(warning_threshold=0.55, critical_threshold=0.60)
        result = validator.validate(self.train_similar, self.test_similar)

        # AUC should be close to 0.5 (random guessing)
        assert result["auc"] < 0.6, f"AUC should be low for similar distributions, got {result['auc']}"

    def test_adversarial_different_distributions(self):
        """Test adversarial AUC is high for different distributions."""
        from src.training.validation import AdversarialValidator

        validator = AdversarialValidator(warning_threshold=0.55, critical_threshold=0.60)
        result = validator.validate(self.train_different, self.test_different)

        # AUC should be high (can distinguish train from test)
        assert result["auc"] > 0.6, f"AUC should be high for different distributions, got {result['auc']}"


class TestDriftRecommendations:
    """Test drift detection recommendations."""

    def test_recommendations_generated(self):
        """Test that recommendations are generated based on drift analysis."""
        from src.training.drift_detection import DriftDetector

        np.random.seed(42)
        reference = pd.DataFrame(np.random.randn(1000, 5), columns=[f"f_{i}" for i in range(5)])
        current = pd.DataFrame(np.random.randn(1000, 5) + 0.5, columns=[f"f_{i}" for i in range(5)])

        detector = DriftDetector(reference_data=reference)
        result = detector.detect_feature_drift(current)

        # Get drift report
        report = detector.get_drift_report()

        # Report should contain recommendations
        assert "recommendations" in report
        assert len(report["recommendations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
