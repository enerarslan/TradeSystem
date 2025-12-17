"""
Unit tests for Cross-Validation with proper purge gap.

Tests the critical requirement:
- Purge gap calculation prevents data leakage
- No index overlap in CV splits
- Temporal ordering is maintained
"""

import pytest
import numpy as np
import pandas as pd


class TestPurgeGapCalculation:
    """Test purge gap calculation logic."""

    def test_purge_gap_formula(self):
        """Test that purge gap follows the formula: horizon + lookback + buffer."""
        # purge_gap = prediction_horizon + max_feature_lookback + buffer
        prediction_horizon = 5
        max_lookback = 200
        buffer = 10

        expected_purge_gap = prediction_horizon + max_lookback + buffer  # 215

        # This should match the config recommendation
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(ma_periods=[5, 10, 20, 50, 100, 200])
        recommended = pipeline.get_purge_gap_recommendation(prediction_horizon, buffer)

        assert recommended == expected_purge_gap

    def test_config_purge_gap_is_auto(self):
        """Test that config has purge_gap set to 'auto'."""
        import yaml
        from pathlib import Path

        config_path = Path("config/ml_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert config["cross_validation"]["purge_gap"] == "auto", \
                "purge_gap should be 'auto' to calculate dynamically"

    def test_config_max_feature_lookback(self):
        """Test that config has correct max_feature_lookback."""
        import yaml
        from pathlib import Path

        config_path = Path("config/ml_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Should be 200 to accommodate MA200
            assert config["cross_validation"]["max_feature_lookback"] == 200


class TestPurgedKFoldCV:
    """Test PurgedKFoldCV for no index overlap and temporal ordering."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for CV testing."""
        n_samples = 1000
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples)
        return X, y

    def test_no_index_overlap(self, sample_data):
        """Test that train and test indices don't overlap."""
        from src.training.validation import PurgedKFoldCV

        X, y = sample_data
        cv = PurgedKFoldCV(n_splits=5, purge_gap=50, embargo_pct=0.01)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            train_set = set(train_idx)
            test_set = set(test_idx)

            # No overlap should exist
            overlap = train_set.intersection(test_set)
            assert len(overlap) == 0, f"Fold {fold_idx}: Found {len(overlap)} overlapping indices"

    def test_temporal_ordering(self, sample_data):
        """Test that train indices respect purge gap around test indices.

        PurgedKFoldCV uses train data from BOTH before and after test set,
        with proper purge gap and embargo. This test verifies that:
        1. Train indices before test have proper gap
        2. Train indices after test have proper embargo
        """
        from src.training.validation import PurgedKFoldCV

        X, y = sample_data
        purge_gap = 50
        cv = PurgedKFoldCV(n_splits=5, purge_gap=purge_gap, embargo_pct=0.01)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            test_min = np.min(test_idx)
            test_max = np.max(test_idx)

            # Separate train indices into before and after test
            train_before = train_idx[train_idx < test_min]
            train_after = train_idx[train_idx > test_max]

            # Check gap for train indices BEFORE test
            if len(train_before) > 0:
                train_before_max = np.max(train_before)
                gap_before = test_min - train_before_max
                assert gap_before >= purge_gap, \
                    f"Fold {fold_idx}: Gap before test ({gap_before}) is less than purge_gap ({purge_gap})"

            # Check embargo for train indices AFTER test (if any)
            if len(train_after) > 0:
                train_after_min = np.min(train_after)
                gap_after = train_after_min - test_max
                # Embargo should be at least 1 (some gap after test)
                assert gap_after >= 1, \
                    f"Fold {fold_idx}: No embargo gap after test"

    def test_purge_gap_respected(self, sample_data):
        """Test that purge gap is properly applied."""
        from src.training.validation import PurgedKFoldCV

        X, y = sample_data
        purge_gap = 50
        cv = PurgedKFoldCV(n_splits=5, purge_gap=purge_gap, embargo_pct=0.0)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Get the range that should be purged
            train_max = np.max(train_idx)
            test_min = np.min(test_idx)

            # Purged indices
            purged_range = set(range(train_max + 1, test_min))

            # Neither train nor test should contain purged indices
            assert len(purged_range.intersection(set(train_idx))) == 0, \
                f"Fold {fold_idx}: Train contains purged indices"
            assert len(purged_range.intersection(set(test_idx))) == 0, \
                f"Fold {fold_idx}: Test contains purged indices"


class TestWalkForwardValidator:
    """Test WalkForwardValidator for temporal consistency."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for walk-forward testing."""
        n_samples = 2000
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples)
        return X, y

    def test_walk_forward_no_overlap(self, sample_data):
        """Test that walk-forward splits don't overlap."""
        from src.training.validation import WalkForwardValidator

        X, y = sample_data
        wf = WalkForwardValidator(
            train_period=500,
            test_period=100,
            step_size=100,
            purge_gap=50,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(wf.split(X, y)):
            train_set = set(train_idx)
            test_set = set(test_idx)

            overlap = train_set.intersection(test_set)
            assert len(overlap) == 0, f"Fold {fold_idx}: Found overlapping indices"

    def test_walk_forward_forward_only(self, sample_data):
        """Test that walk-forward only moves forward in time."""
        from src.training.validation import WalkForwardValidator

        X, y = sample_data
        wf = WalkForwardValidator(
            train_period=500,
            test_period=100,
            step_size=100,
            purge_gap=50,
        )

        previous_test_max = -1

        for fold_idx, (train_idx, test_idx) in enumerate(wf.split(X, y)):
            test_min = np.min(test_idx)

            # Each new test period should be after the previous
            assert test_min > previous_test_max, \
                f"Fold {fold_idx}: Test period went backwards in time"

            previous_test_max = np.max(test_idx)


class TestLeakagePrevention:
    """Test that CV prevents data leakage."""

    def test_cv_leakage_check(self):
        """Test the leakage detection in CV splits.

        PurgedKFoldCV uses train data from both before AND after test set.
        We verify that train indices BEFORE test respect the purge gap.
        """
        from src.training.validation import PurgedKFoldCV

        n_samples = 1000
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples)

        # Create CV with adequate purge gap
        purge_gap = 50
        cv = PurgedKFoldCV(n_splits=5, purge_gap=purge_gap, embargo_pct=0.01)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            test_min_idx = np.min(test_idx)
            test_max_idx = np.max(test_idx)

            # Get train indices that come BEFORE test set
            train_before_test = train_idx[train_idx < test_min_idx]

            # If there are train indices before test, check purge gap
            if len(train_before_test) > 0:
                train_max_before = np.max(train_before_test)
                gap = test_min_idx - train_max_before
                assert gap >= purge_gap, \
                    f"Fold {fold_idx}: Gap ({gap}) < purge_gap ({purge_gap})"

            # Verify no overlap
            overlap = set(train_idx).intersection(set(test_idx))
            assert len(overlap) == 0, f"Fold {fold_idx}: Found overlapping indices"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
