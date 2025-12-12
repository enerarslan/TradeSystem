"""
Institutional-Grade Data Loader
JPMorgan-Level Data Ingestion and Management

Features:
- Multi-source data loading (CSV, API, Database)
- Parallel data loading for 46+ symbols
- Data validation and quality checks
- Memory-efficient chunked loading
- Caching and incremental updates
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Generator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.logger import get_logger, get_performance_logger
from ..utils.helpers import (
    load_config, ensure_dir, timer, retry_with_backoff,
    validate_ohlcv, parallel_map, Result
)


logger = get_logger(__name__)
perf_logger = get_performance_logger()


@dataclass
class DataSpec:
    """Specification for data loading"""
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timeframe: str = "15min"
    source: str = "local_csv"
    columns: Optional[List[str]] = None


@dataclass
class LoadResult:
    """Result of data loading operation"""
    symbol: str
    success: bool
    data: Optional[pd.DataFrame] = None
    rows: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    error: Optional[str] = None
    load_time_ms: float = 0.0


class DataLoader:
    """
    High-performance data loader for single symbol.

    Supports multiple data sources:
    - Local CSV files
    - Parquet files
    - PostgreSQL/TimescaleDB
    - APIs (Alpaca, Polygon, Yahoo Finance)
    """

    # Standard OHLCV column mapping
    COLUMN_MAPPING = {
        'timestamp': ['timestamp', 'date', 'datetime', 'time', 'Date', 'Timestamp'],
        'open': ['open', 'Open', 'OPEN', 'o'],
        'high': ['high', 'High', 'HIGH', 'h'],
        'low': ['low', 'Low', 'LOW', 'l'],
        'close': ['close', 'Close', 'CLOSE', 'c', 'adj_close', 'Adj Close'],
        'volume': ['volume', 'Volume', 'VOLUME', 'v', 'vol']
    }

    def __init__(
        self,
        data_path: str = "data/raw",
        cache_path: str = "data/cache",
        enable_cache: bool = True
    ):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to raw data directory
            cache_path: Path for cached data
            enable_cache: Enable data caching
        """
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.enable_cache = enable_cache

        if enable_cache:
            ensure_dir(self.cache_path)

        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"DataLoader initialized. Data path: {self.data_path}")

    def load(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "15min",
        source: str = "auto"
    ) -> pd.DataFrame:
        """
        Load data for a single symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            timeframe: Data timeframe
            source: Data source ('auto', 'csv', 'parquet', 'api')

        Returns:
            DataFrame with OHLCV data
        """
        with perf_logger.measure_time(f"load_{symbol}"):
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._cache:
                df = self._cache[cache_key].copy()
                return self._filter_dates(df, start_date, end_date)

            # Determine source if auto
            if source == "auto":
                source = self._detect_source(symbol, timeframe)

            # Load based on source
            if source == "csv":
                df = self._load_csv(symbol, timeframe)
            elif source == "parquet":
                df = self._load_parquet(symbol, timeframe)
            elif source == "api":
                df = self._load_api(symbol, timeframe, start_date, end_date)
            else:
                raise ValueError(f"Unknown data source: {source}")

            # Standardize and validate
            df = self._standardize_columns(df)
            df = self._standardize_index(df)

            # Validate data
            is_valid, errors = validate_ohlcv(df)
            if not is_valid:
                logger.warning(f"Data validation warnings for {symbol}: {errors}")

            # Cache if enabled
            if self.enable_cache:
                self._cache[cache_key] = df.copy()

            return self._filter_dates(df, start_date, end_date)

    def _detect_source(self, symbol: str, timeframe: str) -> str:
        """Detect best available data source"""
        # Check for parquet first (faster)
        parquet_path = self.data_path / f"{symbol}_{timeframe}.parquet"
        if parquet_path.exists():
            return "parquet"

        # Check for clean/processed CSV first
        csv_clean_path = self.data_path / f"{symbol}_{timeframe}_clean.csv"
        if csv_clean_path.exists():
            return "csv"

        # Check for standard CSV
        csv_path = self.data_path / f"{symbol}_{timeframe}.csv"
        if csv_path.exists():
            return "csv"

        # Fall back to API
        return "api"

    def _load_csv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from CSV file"""
        # Try processed/clean file first, then standard format
        file_path = self.data_path / f"{symbol}_{timeframe}_clean.csv"

        if not file_path.exists():
            # Fall back to standard naming convention
            file_path = self.data_path / f"{symbol}_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.debug(f"Loading CSV: {file_path}")

        df = pd.read_csv(
            file_path,
            index_col=0,
            dtype={
                'open': np.float64,
                'high': np.float64,
                'low': np.float64,
                'close': np.float64,
                'volume': np.float64
            }
        )
        # Parse index as datetime with UTC conversion for timezone-aware timestamps
        df.index = pd.to_datetime(df.index, utc=True)

        return df

    def _load_parquet(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from Parquet file (faster than CSV)"""
        file_path = self.data_path / f"{symbol}_{timeframe}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        logger.debug(f"Loading Parquet: {file_path}")

        df = pd.read_parquet(file_path)

        return df

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _load_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data from API (Alpaca, Yahoo, etc.)"""
        # Implementation for API loading
        # This is a placeholder - actual implementation depends on API choice
        try:
            import yfinance as yf

            # Convert timeframe to yfinance format
            tf_map = {
                "1min": "1m",
                "5min": "5m",
                "15min": "15m",
                "30min": "30m",
                "1H": "1h",
                "4H": "4h",
                "1D": "1d"
            }
            yf_interval = tf_map.get(timeframe, "15m")

            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval
            )

            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            return df[['open', 'high', 'low', 'close', 'volume']]

        except ImportError:
            logger.warning("yfinance not installed, API loading unavailable")
            raise

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase"""
        column_mapping = {}

        for standard_name, variants in self.COLUMN_MAPPING.items():
            for variant in variants:
                if variant in df.columns:
                    column_mapping[variant] = standard_name
                    break

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df[required]

    def _standardize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame index to DatetimeIndex"""
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            df.index = pd.to_datetime(df.index)

        # Ensure timezone awareness (UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

        # Sort by index
        df = df.sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]

        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=df.index.tz)
            df = df[df.index >= start_date]

        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=df.index.tz)
            df = df[df.index <= end_date]

        return df

    def save_parquet(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save DataFrame to Parquet format for faster loading"""
        output_path = self.data_path / f"{symbol}_{timeframe}.parquet"

        # Convert to PyArrow table for better compression
        table = pa.Table.from_pandas(df)

        pq.write_table(
            table,
            output_path,
            compression='snappy',
            use_dictionary=True
        )

        logger.info(f"Saved {symbol} to Parquet: {output_path}")

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear data cache"""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


class MultiAssetLoader:
    """
    High-performance loader for multiple assets.

    Features:
    - Parallel loading with thread pool
    - Progress tracking
    - Error handling per symbol
    - Memory-efficient batch loading
    """

    def __init__(
        self,
        data_path: str = "data/raw",
        cache_path: str = "data/cache",
        max_workers: int = 8,
        enable_cache: bool = True
    ):
        """
        Initialize MultiAssetLoader.

        Args:
            data_path: Path to raw data directory
            cache_path: Path for cached data
            max_workers: Maximum parallel workers
            enable_cache: Enable data caching
        """
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.max_workers = max_workers
        self.enable_cache = enable_cache

        self.loader = DataLoader(
            data_path=str(data_path),
            cache_path=str(cache_path),
            enable_cache=enable_cache
        )

        logger.info(
            f"MultiAssetLoader initialized. "
            f"Workers: {max_workers}, Path: {data_path}"
        )

    def load_symbols(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "15min",
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols in parallel.

        Args:
            symbols: List of stock symbols
            start_date: Start date filter
            end_date: End date filter
            timeframe: Data timeframe
            show_progress: Show loading progress

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Loading {len(symbols)} symbols...")

        results: Dict[str, pd.DataFrame] = {}
        errors: Dict[str, str] = {}

        with perf_logger.measure_time("multi_asset_load"):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all loading tasks
                future_to_symbol = {
                    executor.submit(
                        self._load_single,
                        symbol,
                        start_date,
                        end_date,
                        timeframe
                    ): symbol
                    for symbol in symbols
                }

                # Collect results
                completed = 0
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1

                    if show_progress:
                        pct = completed / len(symbols) * 100
                        print(f"\rLoading: {completed}/{len(symbols)} ({pct:.1f}%)", end="")

                    try:
                        result = future.result()
                        if result.success:
                            results[symbol] = result.data
                        else:
                            errors[symbol] = result.error
                    except Exception as e:
                        errors[symbol] = str(e)

                if show_progress:
                    print()

        # Log summary
        logger.info(
            f"Loaded {len(results)} symbols successfully, "
            f"{len(errors)} failed"
        )

        if errors:
            logger.warning(f"Failed symbols: {list(errors.keys())}")

        return results

    def _load_single(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        timeframe: str
    ) -> LoadResult:
        """Load single symbol with error handling"""
        import time
        start = time.perf_counter()

        try:
            df = self.loader.load(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )

            return LoadResult(
                symbol=symbol,
                success=True,
                data=df,
                rows=len(df),
                start_date=df.index.min(),
                end_date=df.index.max(),
                load_time_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            return LoadResult(
                symbol=symbol,
                success=False,
                error=str(e),
                load_time_ms=(time.perf_counter() - start) * 1000
            )

    def load_universe(
        self,
        config_path: str = "config/symbols.yaml",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "15min"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all symbols defined in universe config.

        Args:
            config_path: Path to symbols configuration
            start_date: Start date filter
            end_date: End date filter
            timeframe: Data timeframe

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        config = load_config(config_path)

        # Extract all symbols from all sectors
        symbols = []
        for sector_data in config.get('sectors', {}).values():
            symbols.extend(sector_data.get('symbols', []))

        logger.info(f"Loading universe of {len(symbols)} symbols")

        return self.load_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

    def create_panel(
        self,
        data: Dict[str, pd.DataFrame],
        field: str = 'close',
        align: bool = True
    ) -> pd.DataFrame:
        """
        Create panel data (symbols as columns).

        Args:
            data: Dictionary of symbol DataFrames
            field: OHLCV field to extract
            align: Align all series to common index

        Returns:
            Panel DataFrame with symbols as columns
        """
        # Extract field from each symbol
        series_dict = {}
        for symbol, df in data.items():
            if field in df.columns:
                series_dict[symbol] = df[field]

        # Combine into panel
        panel = pd.DataFrame(series_dict)

        if align:
            # Forward fill missing values (limited)
            panel = panel.ffill(limit=3)

        return panel

    def get_statistics(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate statistics for loaded data.

        Args:
            data: Dictionary of symbol DataFrames

        Returns:
            DataFrame with statistics per symbol
        """
        stats = []

        for symbol, df in data.items():
            stat = {
                'symbol': symbol,
                'rows': len(df),
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'days': (df.index.max() - df.index.min()).days,
                'null_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)),
                'avg_volume': df['volume'].mean(),
                'avg_price': df['close'].mean(),
                'volatility': df['close'].pct_change().std() * np.sqrt(252 * 26)  # Annualized
            }
            stats.append(stat)

        return pd.DataFrame(stats).set_index('symbol')

    def convert_to_parquet(self, symbols: Optional[List[str]] = None) -> None:
        """Convert CSV files to Parquet for faster loading"""
        if symbols is None:
            # Find all CSV files
            csv_files = list(self.data_path.glob("*_15min.csv"))
            symbols = [f.stem.replace("_15min", "") for f in csv_files]

        for symbol in symbols:
            try:
                df = self.loader.load(symbol, source="csv")
                self.loader.save_parquet(df, symbol, "15min")
            except Exception as e:
                logger.error(f"Failed to convert {symbol} to parquet: {e}")


class StreamingDataLoader:
    """
    Memory-efficient streaming loader for large datasets.

    Uses chunked reading and generators to handle
    datasets larger than available memory.
    """

    def __init__(
        self,
        data_path: str = "data/raw",
        chunk_size: int = 10000
    ):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size

    def stream_csv(
        self,
        symbol: str,
        timeframe: str = "15min"
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream CSV file in chunks.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe

        Yields:
            DataFrame chunks
        """
        file_path = self.data_path / f"{symbol}_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=self.chunk_size,
            parse_dates=[0],
            index_col=0
        ):
            yield chunk

    def process_streaming(
        self,
        symbol: str,
        processor: callable,
        timeframe: str = "15min"
    ) -> Any:
        """
        Process data in streaming fashion.

        Args:
            symbol: Stock symbol
            processor: Function to process each chunk
            timeframe: Data timeframe

        Returns:
            Aggregated result from processor
        """
        results = []

        for chunk in self.stream_csv(symbol, timeframe):
            result = processor(chunk)
            results.append(result)

        return results


class Level2DataLoader:
    """
    Loader for Level 2 (Order Book) data.

    Supports various formats:
    - Multi-level order book snapshots
    - Best bid/ask data
    - Trade and quote (TAQ) data

    Expected file formats:
    - CSV with columns: timestamp, bid_price_1, bid_size_1, ..., ask_price_1, ask_size_1, ...
    - Parquet with same schema
    """

    # Standard column mapping for Level 2 data
    LEVEL2_COLUMN_MAPPING = {
        'timestamp': ['timestamp', 'time', 'datetime', 'date'],
        'best_bid': ['best_bid', 'bid', 'bid_price', 'bid_price_1'],
        'best_ask': ['best_ask', 'ask', 'ask_price', 'ask_price_1'],
        'bid_size': ['bid_size', 'bid_volume', 'bid_size_1', 'bid_qty'],
        'ask_size': ['ask_size', 'ask_volume', 'ask_size_1', 'ask_qty'],
    }

    def __init__(
        self,
        data_path: str = "data/level2",
        cache_path: str = "data/cache",
        n_levels: int = 5,
        enable_cache: bool = True
    ):
        """
        Initialize Level2DataLoader.

        Args:
            data_path: Path to Level 2 data directory
            cache_path: Path for cached data
            n_levels: Number of order book levels to load
            enable_cache: Enable data caching
        """
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.n_levels = n_levels
        self.enable_cache = enable_cache

        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"Level2DataLoader initialized. Path: {self.data_path}")

    def load(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = "auto"
    ) -> pd.DataFrame:
        """
        Load Level 2 data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            source: Data source ('auto', 'csv', 'parquet')

        Returns:
            DataFrame with Level 2 data
        """
        cache_key = f"{symbol}_l2"
        if cache_key in self._cache:
            df = self._cache[cache_key].copy()
            return self._filter_dates(df, start_date, end_date)

        # Detect source
        if source == "auto":
            source = self._detect_source(symbol)

        # Load data
        if source == "csv":
            df = self._load_csv(symbol)
        elif source == "parquet":
            df = self._load_parquet(symbol)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Standardize columns
        df = self._standardize_columns(df)

        # Cache
        if self.enable_cache:
            self._cache[cache_key] = df.copy()

        return self._filter_dates(df, start_date, end_date)

    def _detect_source(self, symbol: str) -> str:
        """Detect best available data source"""
        parquet_path = self.data_path / f"{symbol}_l2.parquet"
        if parquet_path.exists():
            return "parquet"

        csv_path = self.data_path / f"{symbol}_l2.csv"
        if csv_path.exists():
            return "csv"

        # Check alternative naming
        parquet_path = self.data_path / f"{symbol}_orderbook.parquet"
        if parquet_path.exists():
            return "parquet"

        csv_path = self.data_path / f"{symbol}_orderbook.csv"
        if csv_path.exists():
            return "csv"

        raise FileNotFoundError(f"No Level 2 data found for {symbol}")

    def _load_csv(self, symbol: str) -> pd.DataFrame:
        """Load Level 2 data from CSV"""
        # Try different file names
        for suffix in ['_l2.csv', '_orderbook.csv', '_quotes.csv']:
            file_path = self.data_path / f"{symbol}{suffix}"
            if file_path.exists():
                break
        else:
            raise FileNotFoundError(f"CSV file not found for {symbol}")

        logger.debug(f"Loading Level 2 CSV: {file_path}")

        df = pd.read_csv(
            file_path,
            parse_dates=[0],
            index_col=0
        )

        return df

    def _load_parquet(self, symbol: str) -> pd.DataFrame:
        """Load Level 2 data from Parquet"""
        for suffix in ['_l2.parquet', '_orderbook.parquet', '_quotes.parquet']:
            file_path = self.data_path / f"{symbol}{suffix}"
            if file_path.exists():
                break
        else:
            raise FileNotFoundError(f"Parquet file not found for {symbol}")

        logger.debug(f"Loading Level 2 Parquet: {file_path}")

        df = pd.read_parquet(file_path)

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        column_mapping = {}

        for standard_name, variants in self.LEVEL2_COLUMN_MAPPING.items():
            for variant in variants:
                if variant in df.columns:
                    column_mapping[variant] = standard_name
                    break

        df = df.rename(columns=column_mapping)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df

    def merge_with_ohlcv(
        self,
        level2_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        method: str = 'asof'
    ) -> pd.DataFrame:
        """
        Merge Level 2 data with OHLCV data.

        Args:
            level2_df: Level 2 DataFrame
            ohlcv_df: OHLCV DataFrame
            method: Merge method ('asof', 'nearest', 'intersection')

        Returns:
            Merged DataFrame
        """
        if method == 'asof':
            # Align Level 2 to OHLCV timestamps
            merged = pd.merge_asof(
                ohlcv_df.sort_index(),
                level2_df.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
        elif method == 'nearest':
            merged = pd.merge_asof(
                ohlcv_df.sort_index(),
                level2_df.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest'
            )
        elif method == 'intersection':
            common_idx = ohlcv_df.index.intersection(level2_df.index)
            merged = pd.concat([
                ohlcv_df.loc[common_idx],
                level2_df.loc[common_idx]
            ], axis=1)
        else:
            raise ValueError(f"Unknown merge method: {method}")

        return merged

    def aggregate_to_bars(
        self,
        level2_df: pd.DataFrame,
        freq: str = '15min'
    ) -> pd.DataFrame:
        """
        Aggregate tick-level Level 2 data to bars.

        Args:
            level2_df: Tick-level Level 2 data
            freq: Target frequency

        Returns:
            Aggregated Level 2 data
        """
        agg_dict = {}

        # Aggregate best bid/ask
        if 'best_bid' in level2_df.columns:
            agg_dict['best_bid'] = ['first', 'last', 'mean', 'max', 'min']
        if 'best_ask' in level2_df.columns:
            agg_dict['best_ask'] = ['first', 'last', 'mean', 'max', 'min']
        if 'bid_size' in level2_df.columns:
            agg_dict['bid_size'] = ['mean', 'sum', 'max']
        if 'ask_size' in level2_df.columns:
            agg_dict['ask_size'] = ['mean', 'sum', 'max']

        # Resample
        resampled = level2_df.resample(freq).agg(agg_dict)

        # Flatten column names
        resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]

        return resampled

    def create_snapshot(
        self,
        row: pd.Series,
        timestamp: pd.Timestamp
    ) -> 'OrderBookSnapshot':
        """
        Create OrderBookSnapshot from a data row.

        Args:
            row: Data row with Level 2 columns
            timestamp: Timestamp for snapshot

        Returns:
            OrderBookSnapshot object
        """
        from ..features.microstructure import OrderBookSnapshot, OrderBookLevel

        bids = []
        asks = []

        for i in range(1, self.n_levels + 1):
            bid_price_col = f'bid_price_{i}'
            bid_size_col = f'bid_size_{i}'
            ask_price_col = f'ask_price_{i}'
            ask_size_col = f'ask_size_{i}'

            if bid_price_col in row.index and pd.notna(row[bid_price_col]):
                bids.append(OrderBookLevel(
                    price=row[bid_price_col],
                    size=row.get(bid_size_col, 0)
                ))

            if ask_price_col in row.index and pd.notna(row[ask_price_col]):
                asks.append(OrderBookLevel(
                    price=row[ask_price_col],
                    size=row.get(ask_size_col, 0)
                ))

        # Fallback to simple bid/ask if no multi-level data
        if not bids and 'best_bid' in row.index:
            bids.append(OrderBookLevel(
                price=row['best_bid'],
                size=row.get('bid_size', 0)
            ))
        if not asks and 'best_ask' in row.index:
            asks.append(OrderBookLevel(
                price=row['best_ask'],
                size=row.get('ask_size', 0)
            ))

        return OrderBookSnapshot(
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )

    def generate_synthetic_level2(
        self,
        ohlcv_df: pd.DataFrame,
        spread_pct: float = 0.001,
        depth_base: float = 1000,
        n_levels: int = 5
    ) -> pd.DataFrame:
        """
        Generate synthetic Level 2 data from OHLCV for testing.

        Args:
            ohlcv_df: OHLCV DataFrame
            spread_pct: Base spread as percentage of price
            depth_base: Base depth at each level
            n_levels: Number of levels to generate

        Returns:
            Synthetic Level 2 DataFrame
        """
        level2_data = []

        for idx, row in ohlcv_df.iterrows():
            close = row['close']
            volume = row.get('volume', depth_base * n_levels * 2)

            # Base spread varies with volatility
            if 'high' in row and 'low' in row:
                volatility = (row['high'] - row['low']) / close
                spread = close * max(spread_pct, volatility * 0.5)
            else:
                spread = close * spread_pct

            mid = close
            half_spread = spread / 2

            record = {'timestamp': idx}

            for i in range(1, n_levels + 1):
                # Price gets worse by level
                level_offset = half_spread * (i - 1) * 0.5

                record[f'bid_price_{i}'] = mid - half_spread - level_offset
                record[f'ask_price_{i}'] = mid + half_spread + level_offset

                # Depth decreases by level (with some randomness)
                level_depth = depth_base * np.exp(-0.3 * (i - 1))
                record[f'bid_size_{i}'] = level_depth * (0.8 + 0.4 * np.random.random())
                record[f'ask_size_{i}'] = level_depth * (0.8 + 0.4 * np.random.random())

            level2_data.append(record)

        df = pd.DataFrame(level2_data)
        df.set_index('timestamp', inplace=True)

        return df
