"""
Tests for Purged Cross-Validation implementation.

These tests verify that purged CV correctly prevents information leakage
in financial time series cross-validation.

Reference:
    Lopez de Prado (2018) - "Advances in Financial Machine Learning"
    Chapter 7: Cross-Validation in Finance
"""

import numpy as np
import pandas as pd
import pytest


class TestPurgedKFoldCV:
    """Test PurgedKFoldCV implementation."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.X = np.random.randn(self.n_samples, 10)
        self.y = np.random.randn(self.n_samples)

    def test_basic_split(self):
        """Test basic splitting functionality."""
        from src.training.validation import PurgedKFoldCV

        cv = PurgedKFoldCV(n_splits=5, purge_gap=20)
        splits = list(cv.split(self.X))

        # Should have 5 splits
        assert len(splits) == 5

        # Each split should have train and test
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_purge_gap_correct(self):
        """Test that purge gap is correctly applied."""
        from src.training.validation import PurgedKFoldCV

        purge_gap = 50
        cv = PurgedKFoldCV(n_splits=5, purge_gap=purge_gap)

        for train_idx, test_idx in cv.split(self.X):
            # Find the gap between train and test
            # Train indices before test should be < test_min - purge_gap
            test_min = test_idx.min()
            train_before_test = train_idx[train_idx < test_min]

            if len(train_before_test) > 0:
                train_max_before = train_before_test.max()
                gap = test_min - train_max_before

                assert gap >= purge_gap, (
                    f"Purge gap not enforced: "
                    f"train_max={train_max_before}, test_min={test_min}, "
                    f"gap={gap}, required={purge_gap}"
                )

    def test_embargo_applied(self):
        """Test that embargo is applied after test set."""
        from src.training.validation import PurgedKFoldCV

        embargo_pct = 0.05
        cv = PurgedKFoldCV(n_splits=5, purge_gap=20, embargo_pct=embargo_pct)

        splits = list(cv.split(self.X))

        for i, (train_idx, test_idx) in enumerate(splits):
            if i == len(splits) - 1:
                continue  # Skip last fold (no data after)

            test_max = test_idx.max()
            train_after_test = train_idx[train_idx > test_max]

            if len(train_after_test) > 0:
                train_min_after = train_after_test.min()
                embargo_size = train_min_after - test_max

                # Embargo should be at least embargo_pct of fold size
                fold_size = len(test_idx)
                expected_embargo = int(embargo_pct * fold_size)

                assert embargo_size >= expected_embargo, (
                    f"Embargo not applied: "
                    f"test_max={test_max}, train_min_after={train_min_after}, "
                    f"embargo={embargo_size}, expected>={expected_embargo}"
                )

    def test_no_data_leakage(self):
        """Test that there is no overlap between train and test."""
        from src.training.validation import PurgedKFoldCV

        cv = PurgedKFoldCV(n_splits=5, purge_gap=20)

        for train_idx, test_idx in cv.split(self.X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Data leakage: overlap indices = {overlap}"

    def test_all_test_samples_covered(self):
        """Test that all samples appear in test set exactly once."""
        from src.training.validation import PurgedKFoldCV

        cv = PurgedKFoldCV(n_splits=5, purge_gap=10)

        all_test_indices = []
        for _, test_idx in cv.split(self.X):
            all_test_indices.extend(test_idx)

        # Should cover all samples
        assert len(all_test_indices) == self.n_samples
        # Each sample appears exactly once
        assert len(set(all_test_indices)) == self.n_samples

    def test_calculated_purge_gap(self):
        """Test automatic purge gap calculation."""
        from src.training.validation import PurgedKFoldCV

        prediction_horizon = 10
        max_lookback = 50

        cv = PurgedKFoldCV(
            n_splits=5,
            prediction_horizon=prediction_horizon,
            max_feature_lookback=max_lookback,
        )

        expected_gap = prediction_horizon + max_lookback
        assert cv.get_purge_gap() == expected_gap


class TestCombinatorialPurgedKFoldCV:
    """Test Combinatorial Purged K-Fold CV."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 1200
        self.X = np.random.randn(self.n_samples, 10)
        self.y = np.random.randn(self.n_samples)

    def test_correct_number_of_paths(self):
        """Test that correct number of paths is generated."""
        from src.training.validation import CombinatorialPurgedKFoldCV
        from math import comb

        n_splits = 6
        n_test_splits = 2

        cv = CombinatorialPurgedKFoldCV(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            purge_gap=20,
        )

        paths = list(cv.split(self.X))
        expected_paths = comb(n_splits, n_test_splits)

        assert len(paths) == expected_paths

    def test_multiple_test_folds_per_path(self):
        """Test that CPCV generates paths with varying test fold configurations."""
        from src.training.validation import CombinatorialPurgedKFoldCV

        cv = CombinatorialPurgedKFoldCV(
            n_splits=6,
            n_test_splits=2,
            purge_gap=20,
        )

        total_paths = 0
        paths_with_gaps = 0

        for train_idx, test_idx in cv.split(self.X):
            total_paths += 1
            # Test indices should span multiple regions in some paths
            test_diff = np.diff(np.sort(test_idx))
            large_gaps = np.sum(test_diff > 10)  # Gaps larger than 10

            if large_gaps >= 1:
                paths_with_gaps += 1

        # At least some paths should have non-contiguous test sets
        # (when n_test_splits=2, some combinations will create gaps)
        assert paths_with_gaps > 0, "At least some paths should have non-contiguous test sets"
        assert total_paths == 15, f"Expected 15 paths (C(6,2)), got {total_paths}"

    def test_purge_gap_all_test_regions(self):
        """Test that purge gap is applied around all test regions."""
        from src.training.validation import CombinatorialPurgedKFoldCV

        purge_gap = 30
        cv = CombinatorialPurgedKFoldCV(
            n_splits=6,
            n_test_splits=2,
            purge_gap=purge_gap,
        )

        for train_idx, test_idx in cv.split(self.X):
            train_set = set(train_idx)
            test_set = set(test_idx)

            # Check that purge region is not in training
            for t in test_set:
                for offset in range(1, purge_gap + 1):
                    purge_idx = t - offset
                    if 0 <= purge_idx < self.n_samples:
                        assert purge_idx not in train_set or purge_idx in test_set, (
                            f"Purge gap violated: index {purge_idx} "
                            f"is in training but within {offset} of test"
                        )


class TestWalkForwardValidator:
    """Test Walk-Forward Validation."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.X = np.random.randn(self.n_samples, 10)
        self.y = np.random.randn(self.n_samples)

    def test_train_before_test(self):
        """Test that training data always precedes test data."""
        from src.training.validation import WalkForwardValidator

        wf = WalkForwardValidator(
            train_period=200,
            test_period=50,
            step_size=50,
            purge_gap=10,
        )

        for train_idx, test_idx in wf.split(self.X):
            train_max = train_idx.max()
            test_min = test_idx.min()

            assert train_max < test_min, (
                f"Training data not before test: "
                f"train_max={train_max}, test_min={test_min}"
            )

    def test_purge_gap_walk_forward(self):
        """Test purge gap in walk-forward validation."""
        from src.training.validation import WalkForwardValidator

        purge_gap = 20
        wf = WalkForwardValidator(
            train_period=200,
            test_period=50,
            step_size=50,
            purge_gap=purge_gap,
        )

        for train_idx, test_idx in wf.split(self.X):
            gap = test_idx.min() - train_idx.max()

            assert gap >= purge_gap, (
                f"Purge gap violated in walk-forward: "
                f"gap={gap}, required={purge_gap}"
            )

    def test_expanding_window(self):
        """Test expanding window mode."""
        from src.training.validation import WalkForwardValidator

        wf = WalkForwardValidator(
            train_period=200,
            test_period=50,
            step_size=50,
            expanding=True,
        )

        train_sizes = []
        for train_idx, _ in wf.split(self.X):
            train_sizes.append(len(train_idx))

        # Training size should grow
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                "Training size did not grow in expanding mode"
            )

    def test_sliding_window(self):
        """Test sliding window mode."""
        from src.training.validation import WalkForwardValidator

        train_period = 200
        wf = WalkForwardValidator(
            train_period=train_period,
            test_period=50,
            step_size=50,
            expanding=False,
        )

        for train_idx, _ in wf.split(self.X):
            # Training size should be approximately constant
            assert abs(len(train_idx) - train_period) < 50, (
                f"Training size not constant in sliding mode: "
                f"got {len(train_idx)}, expected ~{train_period}"
            )


class TestCVValidation:
    """Test CV validation utilities."""

    def test_validate_cv_splits_pass(self):
        """Test validation passes for good splits."""
        from src.training.validation import PurgedKFoldCV, validate_cv_splits

        X = np.random.randn(1000, 10)
        cv = PurgedKFoldCV(n_splits=5, purge_gap=20)

        is_valid = validate_cv_splits(
            cv, X,
            min_train_samples=100,
            min_test_samples=20,
        )

        assert is_valid

    def test_validate_cv_splits_fail_small_train(self):
        """Test validation fails for insufficient training samples."""
        from src.training.validation import PurgedKFoldCV, validate_cv_splits

        # Small dataset
        X = np.random.randn(100, 10)
        cv = PurgedKFoldCV(n_splits=5, purge_gap=50)  # Large purge eats into training

        is_valid = validate_cv_splits(
            cv, X,
            min_train_samples=50,
            min_test_samples=5,
        )

        # Should fail due to insufficient training after purging
        assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
