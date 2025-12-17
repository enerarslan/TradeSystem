"""
Data preprocessing module for AlphaTrade system.

This module provides data preprocessing capabilities:
- Missing value handling
- Outlier treatment
- Time-based train/test splitting
- Walk-forward analysis windows
- Data normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


@dataclass
class SplitInfo:
    """Information about a train/test split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int
    fold_index: int


class DataProcessor:
    """
    Data processor for OHLCV data.

    Provides preprocessing functionality including:
    - Missing value imputation
    - Outlier clipping
    - Time-based splitting
    - Walk-forward analysis
    """

    def __init__(
        self,
        fill_method: Literal["ffill", "bfill", "interpolate"] = "ffill",
        fill_limit: int = 10,
        clip_outliers: bool = True,
        outlier_std: float = 5.0,
    ) -> None:
        """
        Initialize the DataProcessor.

        Args:
            fill_method: Method for filling missing values
            fill_limit: Maximum consecutive values to fill
            clip_outliers: Whether to clip outliers
            outlier_std: Number of standard deviations for outlier threshold
        """
        self.fill_method = fill_method
        self.fill_limit = fill_limit
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std

    def process(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to OHLCV data.

        Args:
            df: Input DataFrame with OHLCV data
            inplace: Whether to modify in place

        Returns:
            Processed DataFrame
        """
        if not inplace:
            df = df.copy()

        # Sort by index
        df = df.sort_index()

        # Handle missing values
        df = self._handle_missing(df)

        # Handle OHLC consistency
        df = self._fix_ohlc_consistency(df)

        # Clip outliers in returns (not absolute prices)
        if self.clip_outliers:
            df = self._clip_return_outliers(df)

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="first")]

        logger.debug(f"Processed DataFrame: {len(df)} rows")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        if self.fill_method == "ffill":
            df = df.ffill(limit=self.fill_limit)
        elif self.fill_method == "bfill":
            df = df.bfill(limit=self.fill_limit)
        elif self.fill_method == "interpolate":
            df = df.interpolate(method="time", limit=self.fill_limit)

        # Fill any remaining NaN at the start with bfill
        df = df.bfill()

        return df

    def _fix_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC relationships are consistent."""
        # High should be max of OHLC
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)

        # Low should be min of OHLC
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        return df

    def _clip_return_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme price movements (outliers in returns)."""
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                # Calculate returns
                returns = df[col].pct_change()

                # Calculate thresholds
                mean_ret = returns.mean()
                std_ret = returns.std()
                lower = mean_ret - self.outlier_std * std_ret
                upper = mean_ret + self.outlier_std * std_ret

                # Identify outliers
                outlier_mask = (returns < lower) | (returns > upper)

                if outlier_mask.any():
                    # Clip the returns
                    clipped_returns = returns.clip(lower=lower, upper=upper)

                    # Reconstruct prices where outliers occurred
                    for idx in df.index[outlier_mask]:
                        if idx == df.index[0]:
                            continue
                        prev_idx = df.index[df.index.get_loc(idx) - 1]
                        df.loc[idx, col] = df.loc[prev_idx, col] * (
                            1 + clipped_returns.loc[idx]
                        )

        return df

    def adjust_for_splits(
        self,
        df: pd.DataFrame,
        splits: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjust prices for stock splits.

        Args:
            df: OHLCV DataFrame
            splits: DataFrame with split information (date, ratio)

        Returns:
            Split-adjusted DataFrame
        """
        df = df.copy()

        for _, split in splits.iterrows():
            split_date = pd.Timestamp(split["date"])
            ratio = split["ratio"]

            # Adjust all prices before the split
            mask = df.index < split_date
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio

            # Adjust volume (inverse ratio)
            if "volume" in df.columns:
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio

        return df


class TimeSeriesSplitter:
    """
    Time-based splitter for training and testing.

    Supports:
    - Simple train/test split
    - Walk-forward analysis
    - Purging and embargo for ML
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        """
        Initialize the splitter.

        Args:
            train_ratio: Ratio of data for training (simple split)
            n_splits: Number of walk-forward splits
            purge_gap: Number of samples to purge between train/test
            embargo_pct: Percentage of test data to embargo
        """
        self.train_ratio = train_ratio
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple time-based train/test split.

        Args:
            df: Input DataFrame
            train_ratio: Override default train ratio

        Returns:
            Tuple of (train_df, test_df)
        """
        ratio = train_ratio or self.train_ratio
        split_idx = int(len(df) * ratio)

        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        logger.info(
            f"Train/test split: {len(train)} train, {len(test)} test "
            f"({ratio:.0%}/{1-ratio:.0%})"
        )

        return train, test

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_size: int | None = None,
        test_size: int | None = None,
        anchored: bool = False,
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame, SplitInfo], None, None]:
        """
        Generate walk-forward analysis splits.

        Args:
            df: Input DataFrame
            train_size: Training window size (samples)
            test_size: Test window size (samples)
            anchored: If True, training window expands from start

        Yields:
            Tuples of (train_df, test_df, split_info)
        """
        n = len(df)

        # Default sizes if not specified
        if train_size is None:
            train_size = n // (self.n_splits + 1)
        if test_size is None:
            test_size = train_size // 4

        fold_idx = 0

        if anchored:
            # Anchored (expanding window)
            test_start = train_size
            while test_start + test_size <= n:
                train_end = test_start - self.purge_gap
                actual_test_end = min(test_start + test_size, n)

                train = df.iloc[:train_end].copy()
                test = df.iloc[test_start:actual_test_end].copy()

                info = SplitInfo(
                    train_start=train.index[0],
                    train_end=train.index[-1],
                    test_start=test.index[0],
                    test_end=test.index[-1],
                    train_size=len(train),
                    test_size=len(test),
                    fold_index=fold_idx,
                )

                yield train, test, info

                fold_idx += 1
                test_start += test_size
        else:
            # Rolling window
            for i in range(self.n_splits):
                # Calculate split boundaries
                test_start = train_size + i * test_size
                if test_start >= n:
                    break

                train_start = i * test_size
                train_end = test_start - self.purge_gap
                actual_test_end = min(test_start + test_size, n)

                if train_end <= train_start:
                    continue

                train = df.iloc[train_start:train_end].copy()
                test = df.iloc[test_start:actual_test_end].copy()

                info = SplitInfo(
                    train_start=train.index[0],
                    train_end=train.index[-1],
                    test_start=test.index[0],
                    test_end=test.index[-1],
                    train_size=len(train),
                    test_size=len(test),
                    fold_index=fold_idx,
                )

                yield train, test, info
                fold_idx += 1

    def purged_kfold(
        self,
        df: pd.DataFrame,
        n_splits: int | None = None,
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame, SplitInfo], None, None]:
        """
        Generate purged K-fold splits for time series.

        Implements purging and embargo to prevent leakage in ML.

        Args:
            df: Input DataFrame
            n_splits: Number of folds

        Yields:
            Tuples of (train_df, test_df, split_info)
        """
        n_splits = n_splits or self.n_splits
        n = len(df)
        fold_size = n // n_splits

        embargo_size = int(fold_size * self.embargo_pct)

        for i in range(n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            # Apply purge and embargo
            train_end = max(0, test_start - self.purge_gap)
            train_start_after = min(n, test_end + embargo_size)

            # Combine train portions
            train_before = df.iloc[:train_end]
            train_after = df.iloc[train_start_after:]
            train = pd.concat([train_before, train_after])

            test = df.iloc[test_start:test_end].copy()

            info = SplitInfo(
                train_start=train.index[0] if len(train) > 0 else df.index[0],
                train_end=train.index[-1] if len(train) > 0 else df.index[0],
                test_start=test.index[0],
                test_end=test.index[-1],
                train_size=len(train),
                test_size=len(test),
                fold_index=i,
            )

            yield train, test, info


def preprocess_ohlcv(
    df: pd.DataFrame,
    fill_method: str = "ffill",
    clip_outliers: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to preprocess OHLCV data.

    Args:
        df: Input DataFrame
        fill_method: Method for filling missing values
        clip_outliers: Whether to clip outliers

    Returns:
        Processed DataFrame
    """
    processor = DataProcessor(
        fill_method=fill_method,
        clip_outliers=clip_outliers,
    )
    return processor.process(df)


def create_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for simple train/test split.

    Args:
        df: Input DataFrame
        train_ratio: Ratio of data for training

    Returns:
        Tuple of (train_df, test_df)
    """
    splitter = TimeSeriesSplitter(train_ratio=train_ratio)
    return splitter.train_test_split(df)


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.

    Provides structured approach to:
    - Parameter optimization on training data
    - Out-of-sample testing
    - Result aggregation
    """

    def __init__(
        self,
        train_period_bars: int,
        test_period_bars: int,
        anchored: bool = False,
        purge_bars: int = 10,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            train_period_bars: Training period in bars
            test_period_bars: Test period in bars
            anchored: Use expanding (anchored) window
            purge_bars: Purge gap between train/test
        """
        self.train_period_bars = train_period_bars
        self.test_period_bars = test_period_bars
        self.anchored = anchored
        self.purge_bars = purge_bars

    def get_windows(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, SplitInfo]]:
        """
        Get all walk-forward windows.

        Args:
            df: Input DataFrame

        Returns:
            List of (train, test, info) tuples
        """
        splitter = TimeSeriesSplitter(purge_gap=self.purge_bars)

        windows = list(
            splitter.walk_forward_split(
                df,
                train_size=self.train_period_bars,
                test_size=self.test_period_bars,
                anchored=self.anchored,
            )
        )

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def get_window_summary(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get summary of walk-forward windows.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with window information
        """
        windows = self.get_windows(df)

        summary = []
        for train, test, info in windows:
            summary.append(
                {
                    "fold": info.fold_index,
                    "train_start": info.train_start,
                    "train_end": info.train_end,
                    "test_start": info.test_start,
                    "test_end": info.test_end,
                    "train_size": info.train_size,
                    "test_size": info.test_size,
                }
            )

        return pd.DataFrame(summary)
