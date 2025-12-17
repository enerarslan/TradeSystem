"""
Data validation module for AlphaTrade system.

This module provides comprehensive data quality checks:
- Missing value detection
- OHLC relationship validation
- Outlier detection
- Time series gap detection
- Corporate action detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


@dataclass
class ValidationResult:
    """
    Result of data validation.

    Attributes:
        is_valid: Overall validation status
        symbol: Symbol that was validated
        errors: List of critical errors
        warnings: List of warnings
        stats: Validation statistics
    """

    is_valid: bool
    symbol: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        msg = f"{self.symbol}: {status}"
        if self.errors:
            msg += f" | Errors: {len(self.errors)}"
        if self.warnings:
            msg += f" | Warnings: {len(self.warnings)}"
        return msg

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "symbol": self.symbol,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class DataValidator:
    """
    Comprehensive data validator for OHLCV data.

    Performs multiple validation checks including:
    - Missing value analysis
    - OHLC relationship validation
    - Price and volume outlier detection
    - Time series gap detection
    """

    def __init__(
        self,
        max_missing_pct: float = 5.0,
        max_price_change_pct: float = 50.0,
        min_price: float = 0.01,
        max_gap_minutes: int = 60,
        volume_outlier_std: float = 5.0,
    ) -> None:
        """
        Initialize the DataValidator.

        Args:
            max_missing_pct: Maximum allowed missing value percentage
            max_price_change_pct: Maximum allowed single-bar price change
            min_price: Minimum valid price
            max_gap_minutes: Maximum allowed gap between timestamps
            volume_outlier_std: Standard deviations for volume outlier
        """
        self.max_missing_pct = max_missing_pct
        self.max_price_change_pct = max_price_change_pct
        self.min_price = min_price
        self.max_gap_minutes = max_gap_minutes
        self.volume_outlier_std = volume_outlier_std

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> ValidationResult:
        """
        Perform comprehensive validation on OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for reporting

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True, symbol=symbol)

        # Basic checks
        if df.empty:
            result.is_valid = False
            result.errors.append("DataFrame is empty")
            return result

        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            result.is_valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            return result

        # Run all validation checks
        self._check_missing_values(df, result)
        self._check_ohlc_relationships(df, result)
        self._check_price_validity(df, result)
        self._check_price_changes(df, result)
        self._check_volume_validity(df, result)
        self._check_time_gaps(df, result)
        self._check_duplicates(df, result)

        # Collect statistics
        result.stats = self._collect_stats(df)

        logger.debug(f"Validation complete for {symbol}: {result}")
        return result

    def _check_missing_values(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for missing values in critical columns."""
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df) * 100

                if missing_pct > self.max_missing_pct:
                    result.is_valid = False
                    result.errors.append(
                        f"Column '{col}' has {missing_pct:.2f}% missing values "
                        f"(max allowed: {self.max_missing_pct}%)"
                    )
                elif missing_pct > 0:
                    result.warnings.append(
                        f"Column '{col}' has {missing_pct:.2f}% missing values"
                    )

    def _check_ohlc_relationships(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check that OHLC relationships are valid."""
        # High should be >= Open, Close, Low
        high_violations = (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["high"] < df["low"])
        ).sum()

        if high_violations > 0:
            pct = high_violations / len(df) * 100
            result.warnings.append(
                f"OHLC violation: High < other prices in {high_violations} rows ({pct:.2f}%)"
            )

        # Low should be <= Open, Close, High
        low_violations = (
            (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["low"] > df["high"])
        ).sum()

        if low_violations > 0:
            pct = low_violations / len(df) * 100
            result.warnings.append(
                f"OHLC violation: Low > other prices in {low_violations} rows ({pct:.2f}%)"
            )

    def _check_price_validity(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for invalid price values."""
        for col in ["open", "high", "low", "close"]:
            # Check for negative prices
            negative = (df[col] < 0).sum()
            if negative > 0:
                result.is_valid = False
                result.errors.append(f"Negative prices in '{col}': {negative} rows")

            # Check for zero or very low prices
            too_low = (df[col] < self.min_price).sum()
            if too_low > 0:
                result.warnings.append(
                    f"Prices below {self.min_price} in '{col}': {too_low} rows"
                )

            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                result.is_valid = False
                result.errors.append(f"Infinite values in '{col}': {inf_count} rows")

    def _check_price_changes(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for extreme price changes (potential spikes or data errors)."""
        returns = df["close"].pct_change().abs() * 100

        extreme_changes = returns[returns > self.max_price_change_pct]

        if len(extreme_changes) > 0:
            result.warnings.append(
                f"Extreme price changes (>{self.max_price_change_pct}%): "
                f"{len(extreme_changes)} occurrences. "
                f"Max change: {returns.max():.2f}%"
            )

            # Store dates of extreme changes
            result.stats["extreme_change_dates"] = extreme_changes.index.tolist()

    def _check_volume_validity(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for volume anomalies."""
        if "volume" not in df.columns:
            return

        # Check for negative volume
        negative = (df["volume"] < 0).sum()
        if negative > 0:
            result.is_valid = False
            result.errors.append(f"Negative volume: {negative} rows")

        # Check for zero volume (warning only)
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            pct = zero_vol / len(df) * 100
            result.warnings.append(f"Zero volume: {zero_vol} rows ({pct:.2f}%)")

        # Check for volume outliers
        vol_mean = df["volume"].mean()
        vol_std = df["volume"].std()
        threshold = vol_mean + self.volume_outlier_std * vol_std

        outliers = (df["volume"] > threshold).sum()
        if outliers > 0:
            result.warnings.append(
                f"Volume outliers (>{self.volume_outlier_std} std): {outliers} rows"
            )

    def _check_time_gaps(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for gaps in the time series."""
        if not isinstance(df.index, pd.DatetimeIndex):
            result.warnings.append("Index is not DatetimeIndex, skipping gap check")
            return

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Expected gap is 15 minutes, allow up to max_gap_minutes
        max_gap = pd.Timedelta(minutes=self.max_gap_minutes)

        # Filter out overnight/weekend gaps (expected)
        # Gaps within the same trading day
        gaps = time_diffs[time_diffs > max_gap]

        # Filter out expected overnight gaps (> 15 hours typically market close to open)
        overnight_threshold = pd.Timedelta(hours=15)
        intraday_gaps = gaps[gaps < overnight_threshold]

        if len(intraday_gaps) > 0:
            result.warnings.append(
                f"Intraday time gaps (>{self.max_gap_minutes} min): "
                f"{len(intraday_gaps)} occurrences"
            )
            result.stats["gap_dates"] = intraday_gaps.index.tolist()

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for duplicate timestamps."""
        if isinstance(df.index, pd.DatetimeIndex):
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                result.warnings.append(f"Duplicate timestamps: {duplicates}")
                result.stats["duplicate_count"] = duplicates

    def _collect_stats(self, df: pd.DataFrame) -> dict:
        """Collect summary statistics."""
        stats = {
            "row_count": len(df),
            "start_date": df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
            "end_date": df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None,
            "missing_total": df.isnull().sum().sum(),
            "missing_pct": df.isnull().sum().sum() / df.size * 100,
        }

        # Price statistics
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                stats[f"{col}_min"] = df[col].min()
                stats[f"{col}_max"] = df[col].max()
                stats[f"{col}_mean"] = df[col].mean()

        # Volume statistics
        if "volume" in df.columns:
            stats["volume_min"] = df["volume"].min()
            stats["volume_max"] = df["volume"].max()
            stats["volume_mean"] = df["volume"].mean()
            stats["volume_total"] = df["volume"].sum()

        return stats

    def generate_report(
        self,
        results: list[ValidationResult],
    ) -> pd.DataFrame:
        """
        Generate a validation report from multiple results.

        Args:
            results: List of ValidationResult objects

        Returns:
            DataFrame with validation summary
        """
        rows = []
        for result in results:
            row = {
                "symbol": result.symbol,
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
            }
            row.update(result.stats)
            rows.append(row)

        return pd.DataFrame(rows)


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    strict: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate OHLCV data.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name
        strict: Use stricter validation thresholds

    Returns:
        ValidationResult
    """
    if strict:
        validator = DataValidator(
            max_missing_pct=1.0,
            max_price_change_pct=20.0,
            max_gap_minutes=30,
        )
    else:
        validator = DataValidator()

    return validator.validate(df, symbol)


class CorporateActionDetector:
    """
    Detector for corporate actions in price data.

    Identifies potential:
    - Stock splits
    - Reverse splits
    - Dividends
    """

    def __init__(
        self,
        split_threshold: float = 0.4,
        reverse_split_threshold: float = 1.5,
    ) -> None:
        """
        Initialize the detector.

        Args:
            split_threshold: Return threshold for split detection
            reverse_split_threshold: Return threshold for reverse split
        """
        self.split_threshold = split_threshold
        self.reverse_split_threshold = reverse_split_threshold

    def detect_splits(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect potential stock splits in price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with detected splits
        """
        # Calculate overnight returns (close to open)
        overnight_ret = df["open"] / df["close"].shift(1) - 1

        # Detect forward splits (price drops significantly)
        forward_splits = overnight_ret[overnight_ret < -self.split_threshold]

        # Detect reverse splits (price jumps significantly)
        reverse_splits = overnight_ret[overnight_ret > self.reverse_split_threshold]

        results = []

        for idx, ret in forward_splits.items():
            ratio = 1 / (1 + ret)
            results.append(
                {
                    "date": idx,
                    "type": "forward_split",
                    "return": ret,
                    "estimated_ratio": f"1:{ratio:.0f}",
                }
            )

        for idx, ret in reverse_splits.items():
            ratio = 1 + ret
            results.append(
                {
                    "date": idx,
                    "type": "reverse_split",
                    "return": ret,
                    "estimated_ratio": f"{ratio:.0f}:1",
                }
            )

        return pd.DataFrame(results)

    def detect_dividends(
        self,
        df: pd.DataFrame,
        min_drop_pct: float = 0.5,
        max_drop_pct: float = 10.0,
    ) -> pd.DataFrame:
        """
        Detect potential ex-dividend dates.

        Args:
            df: DataFrame with OHLCV data
            min_drop_pct: Minimum price drop percentage
            max_drop_pct: Maximum price drop percentage

        Returns:
            DataFrame with potential dividend dates
        """
        # Calculate overnight returns
        overnight_ret = (df["open"] / df["close"].shift(1) - 1) * 100

        # Filter for dividend-like drops
        dividends = overnight_ret[
            (overnight_ret < -min_drop_pct) & (overnight_ret > -max_drop_pct)
        ]

        results = []
        for idx, drop in dividends.items():
            prev_close = df["close"].shift(1).loc[idx]
            estimated_div = -drop / 100 * prev_close

            results.append(
                {
                    "date": idx,
                    "price_drop_pct": drop,
                    "prev_close": prev_close,
                    "estimated_dividend": estimated_div,
                }
            )

        return pd.DataFrame(results)
