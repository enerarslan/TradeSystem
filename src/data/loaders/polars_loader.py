"""
High-performance data loader using Polars.

This module provides institutional-grade data loading capabilities using Polars,
offering 10-100x performance improvements over pandas for large datasets.

Key Features:
- Lazy evaluation for memory efficiency
- Parallel file reading
- Zero-copy data conversion
- Schema validation
- Seamless pandas interoperability

Based on JPMorgan-level requirements for:
- Sub-second loading of multi-GB datasets
- Memory-efficient processing
- Type-safe schema validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


logger = logging.getLogger(__name__)


# Standard OHLCV schema
OHLCV_SCHEMA = {
    "timestamp": pl.Datetime("us") if POLARS_AVAILABLE else None,
    "open": pl.Float64 if POLARS_AVAILABLE else None,
    "high": pl.Float64 if POLARS_AVAILABLE else None,
    "low": pl.Float64 if POLARS_AVAILABLE else None,
    "close": pl.Float64 if POLARS_AVAILABLE else None,
    "volume": pl.Int64 if POLARS_AVAILABLE else None,
}


@dataclass
class LoadResult:
    """Result container for data loading operations."""
    data: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]
    rows: int
    columns: int
    load_time_ms: float
    memory_mb: float
    schema_valid: bool
    warnings: list[str]


class SchemaValidator:
    """Validate data against expected schema."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema

    def validate(self, df: "pl.DataFrame") -> tuple[bool, list[str]]:
        """
        Validate DataFrame against schema.

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        if not POLARS_AVAILABLE:
            return True, []

        warnings = []

        # Check required columns
        for col in self.schema:
            if col not in df.columns:
                warnings.append(f"Missing required column: {col}")

        # Check data types
        for col, expected_type in self.schema.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if expected_type is not None and actual_type != expected_type:
                    # Allow compatible types
                    if not self._types_compatible(actual_type, expected_type):
                        warnings.append(
                            f"Column {col}: expected {expected_type}, got {actual_type}"
                        )

        # Check for nulls in critical columns
        critical = ['open', 'high', 'low', 'close']
        for col in critical:
            if col in df.columns and df[col].null_count() > 0:
                null_pct = df[col].null_count() / len(df) * 100
                warnings.append(
                    f"Column {col} has {null_pct:.2f}% null values"
                )

        return len(warnings) == 0, warnings

    def _types_compatible(self, actual, expected) -> bool:
        """Check if actual type is compatible with expected."""
        # Float types are generally compatible
        if "Float" in str(actual) and "Float" in str(expected):
            return True
        # Int types are generally compatible
        if "Int" in str(actual) and "Int" in str(expected):
            return True
        # Datetime types
        if "Datetime" in str(actual) and "Datetime" in str(expected):
            return True
        return False


class PolarsDataLoader:
    """
    High-performance data loader using Polars.

    Provides lazy evaluation by default for memory efficiency,
    with options for eager loading when needed.

    Example:
        loader = PolarsDataLoader()

        # Load single file (lazy)
        lf = loader.load_csv("data/AAPL.csv")
        df = lf.collect()

        # Load multiple files in parallel
        df = loader.load_directory("data/raw/", pattern="*.csv")

        # Load from TimescaleDB
        df = loader.load_from_timescale(query)
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        validate_schema: bool = True,
        default_schema: dict[str, Any] = None,
        n_threads: Optional[int] = None,
        file_format: str = "csv",
    ):
        """
        Initialize PolarsDataLoader.

        Args:
            data_dir: Directory containing data files (for DataLoader API compatibility)
            validate_schema: Validate data against schema
            default_schema: Default schema for validation
            n_threads: Number of threads for parallel loading
            file_format: File format (csv, parquet)
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is required for PolarsDataLoader. "
                "Install with: pip install polars"
            )

        self.data_dir = Path(data_dir) if data_dir else None
        self.validate_schema = validate_schema
        self.default_schema = default_schema or OHLCV_SCHEMA
        self.n_threads = n_threads
        self.file_format = file_format
        self.validator = SchemaValidator(self.default_schema)
        self._symbols: Optional[list[str]] = None
        self._data_cache: dict[str, "pl.DataFrame"] = {}

    @property
    def symbols(self) -> list[str]:
        """Get list of available symbols (DataLoader API compatibility)."""
        if self._symbols is None:
            self._symbols = self._discover_symbols()
        return self._symbols

    def _discover_symbols(self) -> list[str]:
        """Discover available symbols from data directory."""
        if self.data_dir is None:
            return []

        # Try primary format first (SYMBOL_15min.csv)
        pattern_15min = f"*_15min.{self.file_format}"
        files_15min = list(self.data_dir.glob(pattern_15min))

        symbols = []
        for f in files_15min:
            symbol = f.stem.replace("_15min", "")
            if symbol and not symbol.startswith("."):
                symbols.append(symbol)

        # Fallback to simple format (SYMBOL.csv)
        if not symbols:
            pattern_simple = f"*.{self.file_format}"
            files_simple = list(self.data_dir.glob(pattern_simple))
            for f in files_simple:
                symbol = f.stem
                if symbol and not symbol.startswith(".") and symbol not in ["symbol_metadata"]:
                    symbols.append(symbol)

        symbols.sort()
        logger.info(f"PolarsDataLoader discovered {len(symbols)} symbols")
        return symbols

    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data for a single symbol (DataLoader API compatibility).

        Returns pandas DataFrame for compatibility with existing code.

        Args:
            symbol: Symbol to load
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            pandas DataFrame with OHLCV data
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to use load_symbol")

        # Check cache
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key].to_pandas()

        # Find file
        file_path = self.data_dir / f"{symbol}_15min.{self.file_format}"
        if not file_path.exists():
            file_path = self.data_dir / f"{symbol}.{self.file_format}"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {symbol}")

        # Load with Polars (fast)
        if self.file_format == "csv":
            lf = self.load_csv(file_path, lazy=True)
        else:
            lf = self.load_parquet(file_path, lazy=True)

        # Apply date filters
        if start_date:
            lf = lf.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_datetime())
        if end_date:
            lf = lf.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_datetime())

        # Collect and cache
        df = lf.collect()
        self._data_cache[cache_key] = df

        # Convert to pandas with timestamp as index (for compatibility)
        pdf = df.to_pandas()
        if "timestamp" in pdf.columns:
            pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
            pdf = pdf.set_index("timestamp")

        return pdf

    def load_csv(
        self,
        path: Union[str, Path],
        lazy: bool = True,
        columns: Optional[list[str]] = None,
        dtypes: Optional[dict] = None,
        parse_dates: bool = True,
    ) -> Union["pl.LazyFrame", "pl.DataFrame"]:
        """
        Load CSV file with optimized settings.

        Args:
            path: Path to CSV file
            lazy: If True, return LazyFrame for deferred execution
            columns: Specific columns to load (None = all)
            dtypes: Column data type overrides
            parse_dates: Automatically parse date columns

        Returns:
            LazyFrame (lazy=True) or DataFrame (lazy=False)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Build scan options
        scan_kwargs = {
            "source": str(path),
            "has_header": True,
            "ignore_errors": True,
            "try_parse_dates": parse_dates,
        }

        if dtypes:
            scan_kwargs["dtypes"] = dtypes

        if self.n_threads:
            scan_kwargs["n_threads"] = self.n_threads

        # Create lazy frame
        lf = pl.scan_csv(**scan_kwargs)

        # Select columns if specified
        if columns:
            available = lf.collect_schema().names()
            cols_to_select = [c for c in columns if c in available]
            lf = lf.select(cols_to_select)

        # Standardize column names (lowercase)
        lf = lf.rename({c: c.lower() for c in lf.collect_schema().names()})

        # Handle timestamp column
        lf = self._standardize_timestamp(lf)

        if lazy:
            return lf
        return lf.collect()

    def load_parquet(
        self,
        path: Union[str, Path],
        lazy: bool = True,
        columns: Optional[list[str]] = None,
        row_groups: Optional[list[int]] = None,
    ) -> Union["pl.LazyFrame", "pl.DataFrame"]:
        """
        Load Parquet file with optimized settings.

        Parquet format is preferred for large datasets due to:
        - Columnar storage (faster column reads)
        - Built-in compression
        - Schema preservation

        Args:
            path: Path to Parquet file
            lazy: If True, return LazyFrame
            columns: Specific columns to load
            row_groups: Specific row groups to load

        Returns:
            LazyFrame or DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        scan_kwargs = {"source": str(path)}

        if self.n_threads:
            scan_kwargs["n_rows"] = None  # Read all rows
            scan_kwargs["parallel"] = "auto"

        lf = pl.scan_parquet(**scan_kwargs)

        if columns:
            available = lf.collect_schema().names()
            cols_to_select = [c for c in columns if c in available]
            lf = lf.select(cols_to_select)

        # Standardize
        lf = lf.rename({c: c.lower() for c in lf.collect_schema().names()})
        lf = self._standardize_timestamp(lf)

        if lazy:
            return lf
        return lf.collect()

    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.csv",
        lazy: bool = False,
        add_symbol_column: bool = True,
    ) -> "pl.DataFrame":
        """
        Load all matching files from a directory in parallel.

        Args:
            directory: Directory path
            pattern: Glob pattern for file matching
            lazy: If True, return concatenated LazyFrame
            add_symbol_column: Add symbol column from filename

        Returns:
            Combined DataFrame with all data
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        if not files:
            raise ValueError(f"No files matching {pattern} in {directory}")

        logger.info(f"Loading {len(files)} files from {directory}")

        frames = []
        for file_path in files:
            try:
                if file_path.suffix.lower() == '.csv':
                    lf = self.load_csv(file_path, lazy=True)
                elif file_path.suffix.lower() == '.parquet':
                    lf = self.load_parquet(file_path, lazy=True)
                else:
                    continue

                # Add symbol column from filename
                if add_symbol_column:
                    symbol = file_path.stem.split('_')[0].upper()
                    lf = lf.with_columns(pl.lit(symbol).alias("symbol"))

                frames.append(lf)

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

        if not frames:
            raise ValueError("No valid files loaded")

        # Concatenate all frames
        combined = pl.concat(frames, how="diagonal")

        if lazy:
            return combined
        return combined.collect()

    def load_from_timescale(
        self,
        connection_string: str,
        query: str,
    ) -> "pl.DataFrame":
        """
        Load data from TimescaleDB using a SQL query.

        Args:
            connection_string: PostgreSQL connection string
            query: SQL query to execute

        Returns:
            DataFrame with query results
        """
        df = pl.read_database(query, connection_string)
        df = df.rename({c: c.lower() for c in df.columns})
        return df

    def _standardize_timestamp(
        self,
        lf: "pl.LazyFrame",
    ) -> "pl.LazyFrame":
        """Standardize timestamp column name and type."""
        schema = lf.collect_schema()
        columns = schema.names()

        # Find timestamp column
        time_cols = ['timestamp', 'time', 'datetime', 'date']
        time_col = None
        for col in time_cols:
            if col in columns:
                time_col = col
                break

        if time_col is None:
            return lf

        # Rename to 'timestamp' if needed
        if time_col != 'timestamp':
            lf = lf.rename({time_col: 'timestamp'})

        # Ensure datetime type
        if schema[time_col] != pl.Datetime:
            lf = lf.with_columns(
                pl.col('timestamp').str.to_datetime().alias('timestamp')
            )

        return lf

    def to_pandas(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame"],
        set_index: Optional[str] = 'timestamp',
    ) -> pd.DataFrame:
        """
        Convert Polars DataFrame to pandas.

        Args:
            df: Polars DataFrame or LazyFrame
            set_index: Column to set as index (None for no index)

        Returns:
            pandas DataFrame
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        pdf = df.to_pandas()

        if set_index and set_index in pdf.columns:
            pdf = pdf.set_index(set_index)

        return pdf

    def validate(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame"],
    ) -> tuple[bool, list[str]]:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, warnings)
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        return self.validator.validate(df)


class PolarsFeatureEngine:
    """
    High-performance feature engineering using Polars expressions.

    Uses Polars' expression API for vectorized, parallelized computation
    of technical indicators and statistical features.
    """

    def __init__(self):
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required")

    def add_returns(
        self,
        df: "pl.DataFrame",
        periods: list[int] = [1, 5, 10, 20],
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add return features for multiple periods."""
        for period in periods:
            df = df.with_columns(
                (pl.col(price_col).pct_change(period))
                .alias(f"return_{period}")
            )
        return df

    def add_log_returns(
        self,
        df: "pl.DataFrame",
        periods: list[int] = [1, 5, 10, 20],
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add log return features for multiple periods."""
        for period in periods:
            df = df.with_columns(
                (pl.col(price_col).log() - pl.col(price_col).shift(period).log())
                .alias(f"log_return_{period}")
            )
        return df

    def add_volatility(
        self,
        df: "pl.DataFrame",
        windows: list[int] = [10, 20, 50],
        return_col: str = "return_1",
    ) -> "pl.DataFrame":
        """Add rolling volatility features."""
        for window in windows:
            df = df.with_columns(
                pl.col(return_col)
                .rolling_std(window_size=window)
                .alias(f"volatility_{window}")
            )
        return df

    def add_sma(
        self,
        df: "pl.DataFrame",
        periods: list[int] = [5, 10, 20, 50],
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add simple moving average features."""
        for period in periods:
            df = df.with_columns(
                pl.col(price_col)
                .rolling_mean(window_size=period)
                .alias(f"sma_{period}")
            )
        return df

    def add_ema(
        self,
        df: "pl.DataFrame",
        periods: list[int] = [5, 10, 20, 50],
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add exponential moving average features."""
        for period in periods:
            df = df.with_columns(
                pl.col(price_col)
                .ewm_mean(span=period)
                .alias(f"ema_{period}")
            )
        return df

    def add_rsi(
        self,
        df: "pl.DataFrame",
        periods: list[int] = [7, 14, 21],
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add RSI (Relative Strength Index) features."""
        # Calculate price changes
        delta = pl.col(price_col) - pl.col(price_col).shift(1)

        for period in periods:
            # Separate gains and losses
            gain = delta.clip(lower_bound=0)
            loss = (-delta).clip(lower_bound=0)

            # Calculate average gain/loss using EWM
            avg_gain = gain.ewm_mean(span=period)
            avg_loss = loss.ewm_mean(span=period)

            # Calculate RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))

            df = df.with_columns(rsi.alias(f"rsi_{period}"))

        return df

    def add_macd(
        self,
        df: "pl.DataFrame",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add MACD (Moving Average Convergence Divergence) features."""
        ema_fast = pl.col(price_col).ewm_mean(span=fast)
        ema_slow = pl.col(price_col).ewm_mean(span=slow)
        macd_line = ema_fast - ema_slow

        df = df.with_columns([
            macd_line.alias("macd_line"),
            macd_line.ewm_mean(span=signal).alias("macd_signal"),
        ])

        df = df.with_columns(
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
        )

        return df

    def add_bollinger_bands(
        self,
        df: "pl.DataFrame",
        window: int = 20,
        num_std: float = 2.0,
        price_col: str = "close",
    ) -> "pl.DataFrame":
        """Add Bollinger Band features."""
        rolling_mean = pl.col(price_col).rolling_mean(window_size=window)
        rolling_std = pl.col(price_col).rolling_std(window_size=window)

        df = df.with_columns([
            rolling_mean.alias(f"bb_middle_{window}"),
            (rolling_mean + num_std * rolling_std).alias(f"bb_upper_{window}"),
            (rolling_mean - num_std * rolling_std).alias(f"bb_lower_{window}"),
        ])

        # BB %B and width
        df = df.with_columns([
            ((pl.col(price_col) - pl.col(f"bb_lower_{window}")) /
             (pl.col(f"bb_upper_{window}") - pl.col(f"bb_lower_{window}") + 1e-10))
            .alias(f"bb_pct_{window}"),

            ((pl.col(f"bb_upper_{window}") - pl.col(f"bb_lower_{window}")) /
             (pl.col(f"bb_middle_{window}") + 1e-10))
            .alias(f"bb_width_{window}"),
        ])

        return df

    def add_atr(
        self,
        df: "pl.DataFrame",
        periods: list[int] = [7, 14, 21],
    ) -> "pl.DataFrame":
        """Add Average True Range features."""
        # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        prev_close = pl.col("close").shift(1)

        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        )

        df = df.with_columns(tr.alias("true_range"))

        for period in periods:
            df = df.with_columns(
                pl.col("true_range")
                .ewm_mean(span=period)
                .alias(f"atr_{period}")
            )

        return df

    def add_volume_features(
        self,
        df: "pl.DataFrame",
        windows: list[int] = [5, 10, 20],
    ) -> "pl.DataFrame":
        """Add volume-based features."""
        for window in windows:
            # Volume MA
            df = df.with_columns(
                pl.col("volume")
                .rolling_mean(window_size=window)
                .alias(f"volume_ma_{window}")
            )

            # Volume ratio
            df = df.with_columns(
                (pl.col("volume") / (pl.col(f"volume_ma_{window}") + 1))
                .alias(f"volume_ratio_{window}")
            )

        # VWAP (Volume Weighted Average Price)
        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
        df = df.with_columns(
            (typical_price * pl.col("volume"))
            .cum_sum()
            .truediv(pl.col("volume").cum_sum() + 1)
            .alias("vwap")
        )

        return df

    def add_all_features(
        self,
        df: "pl.DataFrame",
        include: list[str] = None,
    ) -> "pl.DataFrame":
        """
        Add all technical features.

        Args:
            df: Input DataFrame with OHLCV data
            include: List of feature types to include (None = all)
                Options: ['returns', 'volatility', 'sma', 'ema', 'rsi',
                         'macd', 'bollinger', 'atr', 'volume']

        Returns:
            DataFrame with added features
        """
        all_features = [
            'returns', 'volatility', 'sma', 'ema', 'rsi',
            'macd', 'bollinger', 'atr', 'volume'
        ]

        if include is None:
            include = all_features

        # Returns first (needed for volatility)
        if 'returns' in include:
            df = self.add_returns(df)
            df = self.add_log_returns(df)

        if 'volatility' in include and 'return_1' in df.columns:
            df = self.add_volatility(df)

        if 'sma' in include:
            df = self.add_sma(df)

        if 'ema' in include:
            df = self.add_ema(df)

        if 'rsi' in include:
            df = self.add_rsi(df)

        if 'macd' in include:
            df = self.add_macd(df)

        if 'bollinger' in include:
            df = self.add_bollinger_bands(df)

        if 'atr' in include:
            df = self.add_atr(df)

        if 'volume' in include and 'volume' in df.columns:
            df = self.add_volume_features(df)

        return df


# Cross-sectional features using group_by
class PolarsCrossSectionalFeatures:
    """
    Cross-sectional feature engineering for multi-asset portfolios.

    Computes features relative to the cross-section of assets
    at each timestamp (e.g., rankings, z-scores).
    """

    def __init__(self):
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required")

    def add_cross_sectional_rank(
        self,
        df: "pl.DataFrame",
        columns: list[str],
        group_col: str = "timestamp",
    ) -> "pl.DataFrame":
        """
        Add cross-sectional rank for specified columns.

        Rank is normalized to [0, 1] within each timestamp.
        """
        for col in columns:
            if col not in df.columns:
                continue

            df = df.with_columns(
                pl.col(col)
                .rank()
                .over(group_col)
                .truediv(pl.col(col).count().over(group_col))
                .alias(f"{col}_rank")
            )

        return df

    def add_cross_sectional_zscore(
        self,
        df: "pl.DataFrame",
        columns: list[str],
        group_col: str = "timestamp",
    ) -> "pl.DataFrame":
        """
        Add cross-sectional z-score for specified columns.

        Z-score standardizes to mean=0, std=1 within each timestamp.
        """
        for col in columns:
            if col not in df.columns:
                continue

            df = df.with_columns(
                ((pl.col(col) - pl.col(col).mean().over(group_col)) /
                 (pl.col(col).std().over(group_col) + 1e-10))
                .alias(f"{col}_zscore")
            )

        return df

    def add_sector_relative(
        self,
        df: "pl.DataFrame",
        columns: list[str],
        sector_col: str = "sector",
        group_col: str = "timestamp",
    ) -> "pl.DataFrame":
        """
        Add sector-relative features (value minus sector mean).

        Useful for identifying outperformers within sectors.
        """
        for col in columns:
            if col not in df.columns:
                continue

            # Sector mean at each timestamp
            df = df.with_columns(
                (pl.col(col) - pl.col(col).mean().over([group_col, sector_col]))
                .alias(f"{col}_sector_relative")
            )

        return df


# Convenience functions
def load_csv(path: Union[str, Path], **kwargs) -> "pl.DataFrame":
    """Quick function to load a CSV file."""
    loader = PolarsDataLoader()
    return loader.load_csv(path, lazy=False, **kwargs)


def load_directory(directory: Union[str, Path], **kwargs) -> "pl.DataFrame":
    """Quick function to load all files from directory."""
    loader = PolarsDataLoader()
    return loader.load_directory(directory, **kwargs)
