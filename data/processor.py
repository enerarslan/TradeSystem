"""
Data Processor Module
=====================

Data cleaning, validation, normalization, and resampling.
Ensures data quality for reliable backtesting and trading.

Features:
- OHLCV validation (price relationships, volume)
- Missing data detection and handling
- Outlier detection and treatment
- Timeframe resampling
- Data normalization for ML
- Gap filling and forward-fill

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, TimeFrame
from core.types import DataValidationError

logger = get_logger(__name__)


# =============================================================================
# VALIDATION TYPES
# =============================================================================

class ValidationLevel(str, Enum):
    """Validation strictness level."""
    STRICT = "strict"      # Fail on any issue
    MODERATE = "moderate"  # Warn and fix minor issues
    LENIENT = "lenient"    # Fix everything silently


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]
    
    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    symbol: str
    total_rows: int
    date_range: tuple[datetime, datetime]
    missing_bars: int
    missing_pct: float
    duplicate_bars: int
    invalid_ohlc: int
    zero_volume_bars: int
    outliers_detected: int
    gaps: list[tuple[datetime, datetime]]
    validation_result: ValidationResult


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    OHLCV data validator.
    
    Checks for:
    - Price relationship validity (H >= L, H >= O/C, L <= O/C)
    - Missing values
    - Duplicate timestamps
    - Volume validity
    - Price outliers
    - Data gaps
    """
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.MODERATE,
        max_price_change_pct: float = 50.0,
        min_volume: float = 0.0,
    ):
        """
        Initialize validator.
        
        Args:
            level: Validation strictness level
            max_price_change_pct: Maximum allowed price change %
            min_volume: Minimum required volume
        """
        self.level = level
        self.max_price_change_pct = max_price_change_pct
        self.min_volume = min_volume
    
    def validate(self, df: pl.DataFrame) -> ValidationResult:
        """
        Validate OHLCV data.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            ValidationResult with details
        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}
        
        # Check required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = set(required) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)
        
        # Basic stats
        stats["total_rows"] = len(df)
        stats["date_range"] = (
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
        
        # Check for empty data
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, stats)
        
        # Check for null values
        null_counts = df.null_count()
        total_nulls = sum(null_counts.row(0))
        if total_nulls > 0:
            stats["null_counts"] = null_counts.to_dict()
            if self.level == ValidationLevel.STRICT:
                errors.append(f"Found {total_nulls} null values")
            else:
                warnings.append(f"Found {total_nulls} null values")
        
        # Check for duplicate timestamps
        duplicate_count = len(df) - df.n_unique("timestamp")
        stats["duplicates"] = duplicate_count
        if duplicate_count > 0:
            if self.level == ValidationLevel.STRICT:
                errors.append(f"Found {duplicate_count} duplicate timestamps")
            else:
                warnings.append(f"Found {duplicate_count} duplicate timestamps")
        
        # Check OHLC relationships
        invalid_ohlc = df.filter(
            (pl.col("high") < pl.col("low")) |
            (pl.col("high") < pl.col("open")) |
            (pl.col("high") < pl.col("close")) |
            (pl.col("low") > pl.col("open")) |
            (pl.col("low") > pl.col("close"))
        )
        stats["invalid_ohlc"] = len(invalid_ohlc)
        if len(invalid_ohlc) > 0:
            if self.level == ValidationLevel.STRICT:
                errors.append(f"Found {len(invalid_ohlc)} bars with invalid OHLC relationships")
            else:
                warnings.append(f"Found {len(invalid_ohlc)} bars with invalid OHLC relationships")
        
        # Check for zero/negative prices
        invalid_prices = df.filter(
            (pl.col("open") <= 0) |
            (pl.col("high") <= 0) |
            (pl.col("low") <= 0) |
            (pl.col("close") <= 0)
        )
        stats["invalid_prices"] = len(invalid_prices)
        if len(invalid_prices) > 0:
            errors.append(f"Found {len(invalid_prices)} bars with zero/negative prices")
        
        # Check volume
        zero_volume = df.filter(pl.col("volume") <= self.min_volume)
        stats["zero_volume"] = len(zero_volume)
        if len(zero_volume) > len(df) * 0.1:  # More than 10%
            warnings.append(f"Found {len(zero_volume)} bars with zero volume ({100*len(zero_volume)/len(df):.1f}%)")
        
        # Check for extreme price changes (potential outliers)
        if len(df) > 1:
            df_with_returns = df.with_columns(
                (pl.col("close").pct_change().abs() * 100).alias("pct_change")
            )
            extreme_changes = df_with_returns.filter(
                pl.col("pct_change") > self.max_price_change_pct
            )
            stats["extreme_changes"] = len(extreme_changes)
            if len(extreme_changes) > 0:
                warnings.append(
                    f"Found {len(extreme_changes)} bars with price changes > {self.max_price_change_pct}%"
                )
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)
    
    def generate_report(
        self,
        df: pl.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "15min",
    ) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            symbol: Symbol name
            timeframe: Data timeframe
        
        Returns:
            DataQualityReport with all metrics
        """
        validation = self.validate(df)
        
        # Calculate expected bars
        if len(df) > 0:
            start = df["timestamp"].min()
            end = df["timestamp"].max()
            
            # Estimate expected bars (rough approximation)
            tf_minutes = TimeFrame(timeframe).minutes if hasattr(TimeFrame, timeframe) else 15
            total_minutes = (end - start).total_seconds() / 60
            expected_bars = int(total_minutes / tf_minutes)
            missing_bars = max(0, expected_bars - len(df))
            missing_pct = missing_bars / expected_bars * 100 if expected_bars > 0 else 0
        else:
            start = end = datetime.now()
            missing_bars = 0
            missing_pct = 0
        
        # Detect gaps
        gaps = self._detect_gaps(df, timeframe)
        
        return DataQualityReport(
            symbol=symbol,
            total_rows=len(df),
            date_range=(start, end),
            missing_bars=missing_bars,
            missing_pct=missing_pct,
            duplicate_bars=validation.stats.get("duplicates", 0),
            invalid_ohlc=validation.stats.get("invalid_ohlc", 0),
            zero_volume_bars=validation.stats.get("zero_volume", 0),
            outliers_detected=validation.stats.get("extreme_changes", 0),
            gaps=gaps,
            validation_result=validation,
        )
    
    def _detect_gaps(
        self,
        df: pl.DataFrame,
        timeframe: str,
    ) -> list[tuple[datetime, datetime]]:
        """Detect gaps in time series."""
        if len(df) < 2:
            return []
        
        # Get timeframe in minutes
        tf_map = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "1hour": 60, "4hour": 240, "1day": 1440,
        }
        expected_delta = timedelta(minutes=tf_map.get(timeframe, 15))
        
        # Calculate time differences
        df_sorted = df.sort("timestamp")
        timestamps = df_sorted["timestamp"].to_list()
        
        gaps = []
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i-1]
            # Allow 2x expected delta before flagging as gap
            if delta > expected_delta * 2:
                gaps.append((timestamps[i-1], timestamps[i]))
        
        return gaps


# =============================================================================
# DATA PROCESSOR
# =============================================================================

class DataProcessor:
    """
    Data cleaning and preprocessing pipeline.
    
    Operations:
    - Remove duplicates
    - Handle missing values
    - Fix OHLC relationships
    - Remove outliers
    - Fill gaps
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
        fill_method: str = "forward",
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
    ):
        """
        Initialize processor.
        
        Args:
            validation_level: Validation strictness
            fill_method: Method for filling missing data
            outlier_method: Outlier detection method
            outlier_threshold: Outlier threshold multiplier
        """
        self.validator = DataValidator(level=validation_level)
        self.fill_method = fill_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
    
    def process(
        self,
        df: pl.DataFrame,
        symbol: str = "",
        validate: bool = True,
        remove_duplicates: bool = True,
        fill_missing: bool = True,
        fix_ohlc: bool = True,
        remove_outliers: bool = False,
    ) -> pl.DataFrame:
        """
        Process OHLCV data.
        
        Args:
            df: Input DataFrame
            symbol: Symbol name for logging
            validate: Run validation
            remove_duplicates: Remove duplicate timestamps
            fill_missing: Fill missing values
            fix_ohlc: Fix invalid OHLC relationships
            remove_outliers: Remove outlier bars
        
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {symbol}: {len(df)} rows")
        original_len = len(df)
        
        # Initial validation
        if validate:
            result = self.validator.validate(df)
            if not result.is_valid:
                logger.warning(f"Validation issues for {symbol}: {result.errors}")
        
        # Sort by timestamp
        df = df.sort("timestamp")
        
        # Remove duplicates
        if remove_duplicates:
            df = df.unique(subset=["timestamp"], keep="last")
            if len(df) < original_len:
                logger.info(f"Removed {original_len - len(df)} duplicates")
        
        # Fill missing values
        if fill_missing:
            df = self._fill_missing(df)
        
        # Fix OHLC relationships
        if fix_ohlc:
            df = self._fix_ohlc(df)
        
        # Remove outliers
        if remove_outliers:
            df = self._remove_outliers(df)
        
        logger.info(f"Processed {symbol}: {original_len} -> {len(df)} rows")
        return df
    
    def _fill_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing values."""
        if self.fill_method == "forward":
            # Forward fill for OHLC, zero for volume
            df = df.with_columns([
                pl.col("open").forward_fill(),
                pl.col("high").forward_fill(),
                pl.col("low").forward_fill(),
                pl.col("close").forward_fill(),
                pl.col("volume").fill_null(0),
            ])
        elif self.fill_method == "interpolate":
            # Linear interpolation
            df = df.with_columns([
                pl.col("open").interpolate(),
                pl.col("high").interpolate(),
                pl.col("low").interpolate(),
                pl.col("close").interpolate(),
                pl.col("volume").interpolate(),
            ])
        elif self.fill_method == "drop":
            df = df.drop_nulls()
        
        return df
    
    def _fix_ohlc(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fix invalid OHLC relationships."""
        # High must be max of OHLC
        # Low must be min of OHLC
        df = df.with_columns([
            pl.max_horizontal("open", "high", "close").alias("high"),
            pl.min_horizontal("open", "low", "close").alias("low"),
        ])
        return df
    
    def _remove_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove outlier bars."""
        if self.outlier_method == "iqr":
            # IQR method for returns
            returns = df["close"].pct_change()
            q1 = returns.quantile(0.25)
            q3 = returns.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr
            
            df = df.with_columns(
                returns.alias("_returns")
            ).filter(
                (pl.col("_returns").is_null()) |
                ((pl.col("_returns") >= lower) & (pl.col("_returns") <= upper))
            ).drop("_returns")
            
        elif self.outlier_method == "zscore":
            # Z-score method
            returns = df["close"].pct_change()
            mean = returns.mean()
            std = returns.std()
            
            df = df.with_columns(
                ((returns - mean) / std).abs().alias("_zscore")
            ).filter(
                (pl.col("_zscore").is_null()) |
                (pl.col("_zscore") <= self.outlier_threshold)
            ).drop("_zscore")
        
        return df


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_ohlcv(
    df: pl.DataFrame,
    target_timeframe: str,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: Input DataFrame
        target_timeframe: Target timeframe (e.g., "1hour", "1day")
        timestamp_col: Timestamp column name
    
    Returns:
        Resampled DataFrame
    """
    # Map timeframe to Polars duration
    tf_map = {
        "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
        "1hour": "1h", "4hour": "4h", "1day": "1d", "1week": "1w",
    }
    
    duration = tf_map.get(target_timeframe)
    if duration is None:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")
    
    # Resample using group_by_dynamic
    resampled = df.sort(timestamp_col).group_by_dynamic(
        timestamp_col,
        every=duration,
    ).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ])
    
    # Keep symbol column if present
    if "symbol" in df.columns:
        symbol = df["symbol"].first()
        resampled = resampled.with_columns(pl.lit(symbol).alias("symbol"))
    
    return resampled


# =============================================================================
# NORMALIZATION
# =============================================================================

class DataNormalizer:
    """
    Data normalization for machine learning.
    
    Methods:
    - Z-score (standardization)
    - Min-max scaling
    - Robust scaling (IQR-based)
    - Log transformation
    """
    
    def __init__(self, method: str = "zscore"):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method
        """
        self.method = method
        self._params: dict[str, dict[str, float]] = {}
    
    def fit(self, df: pl.DataFrame, columns: list[str]) -> "DataNormalizer":
        """
        Fit normalizer to data.
        
        Args:
            df: Training data
            columns: Columns to normalize
        
        Returns:
            Self for chaining
        """
        for col in columns:
            if col not in df.columns:
                continue
            
            values = df[col]
            
            if self.method == "zscore":
                self._params[col] = {
                    "mean": values.mean(),
                    "std": values.std(),
                }
            elif self.method == "minmax":
                self._params[col] = {
                    "min": values.min(),
                    "max": values.max(),
                }
            elif self.method == "robust":
                self._params[col] = {
                    "median": values.median(),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                }
        
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            df: Data to transform
        
        Returns:
            Transformed DataFrame
        """
        transforms = []
        
        for col, params in self._params.items():
            if col not in df.columns:
                continue
            
            if self.method == "zscore":
                expr = (pl.col(col) - params["mean"]) / params["std"]
            elif self.method == "minmax":
                range_val = params["max"] - params["min"]
                expr = (pl.col(col) - params["min"]) / range_val if range_val > 0 else pl.col(col)
            elif self.method == "robust":
                iqr = params["iqr"]
                expr = (pl.col(col) - params["median"]) / iqr if iqr > 0 else pl.col(col)
            else:
                expr = pl.col(col)
            
            transforms.append(expr.alias(col))
        
        if transforms:
            df = df.with_columns(transforms)
        
        return df
    
    def fit_transform(
        self,
        df: pl.DataFrame,
        columns: list[str],
    ) -> pl.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Data to fit and transform
            columns: Columns to normalize
        
        Returns:
            Transformed DataFrame
        """
        return self.fit(df, columns).transform(df)
    
    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reverse the normalization.
        
        Args:
            df: Normalized data
        
        Returns:
            Original scale DataFrame
        """
        transforms = []
        
        for col, params in self._params.items():
            if col not in df.columns:
                continue
            
            if self.method == "zscore":
                expr = pl.col(col) * params["std"] + params["mean"]
            elif self.method == "minmax":
                range_val = params["max"] - params["min"]
                expr = pl.col(col) * range_val + params["min"]
            elif self.method == "robust":
                expr = pl.col(col) * params["iqr"] + params["median"]
            else:
                expr = pl.col(col)
            
            transforms.append(expr.alias(col))
        
        if transforms:
            df = df.with_columns(transforms)
        
        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def clean_ohlcv_data(
    df: pl.DataFrame,
    symbol: str = "",
) -> pl.DataFrame:
    """
    Convenience function to clean OHLCV data.
    
    Args:
        df: Input DataFrame
        symbol: Symbol name
    
    Returns:
        Cleaned DataFrame
    """
    processor = DataProcessor()
    return processor.process(df, symbol=symbol)


def normalize_data(
    df: pl.DataFrame,
    columns: list[str],
    method: str = "zscore",
) -> tuple[pl.DataFrame, DataNormalizer]:
    """
    Convenience function to normalize data.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
        method: Normalization method
    
    Returns:
        Tuple of (normalized DataFrame, fitted normalizer)
    """
    normalizer = DataNormalizer(method=method)
    normalized = normalizer.fit_transform(df, columns)
    return normalized, normalizer


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "DataQualityReport",
    "DataValidator",
    "DataProcessor",
    "DataNormalizer",
    "resample_ohlcv",
    "clean_ohlcv_data",
    "normalize_data",
]