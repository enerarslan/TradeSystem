"""
Data Loader Module
==================

High-performance data loading for OHLCV market data.
Supports CSV files, APIs, and databases.

Features:
- Polars-based for maximum performance
- Automatic caching with configurable TTL
- Parallel loading for multiple symbols
- Schema validation
- Memory-efficient streaming for large files

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import polars as pl

from config.settings import get_settings, get_logger, TimeFrame
from core.types import DataError, DataNotFoundError, DataValidationError

logger = get_logger(__name__)


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


# =============================================================================
# CACHE UTILITIES
# =============================================================================

class DataCache:
    """
    Simple file-based cache for DataFrames.
    
    Uses pickle for serialization and SHA256 for cache keys.
    """
    
    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, identifier: str) -> str:
        """Generate cache key from identifier."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.parquet"
    
    def get(self, identifier: str) -> pl.DataFrame | None:
        """
        Get cached DataFrame.
        
        Args:
            identifier: Cache identifier
        
        Returns:
            Cached DataFrame or None if not found/expired
        """
        key = self._get_cache_key(identifier)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check TTL
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            cache_path.unlink()
            return None
        
        try:
            return pl.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None
    
    def set(self, identifier: str, data: pl.DataFrame) -> None:
        """
        Cache a DataFrame.
        
        Args:
            identifier: Cache identifier
            data: DataFrame to cache
        """
        key = self._get_cache_key(identifier)
        cache_path = self._get_cache_path(key)
        
        try:
            data.write_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self) -> int:
        """
        Clear all cache files.
        
        Returns:
            Number of files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
            count += 1
        return count


# =============================================================================
# CSV LOADER
# =============================================================================

class CSVLoader:
    """
    High-performance CSV loader using Polars.
    
    Features:
    - Automatic type inference
    - Schema validation
    - Date parsing
    - Missing data handling
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize CSV loader.
        
        Args:
            storage_path: Path to CSV storage directory
            use_cache: Enable caching
            cache_ttl_hours: Cache TTL in hours
        """
        settings = get_settings()
        self.storage_path = storage_path or settings.data.storage_path
        self.use_cache = use_cache
        
        if use_cache:
            self.cache = DataCache(
                settings.data.cache_path,
                cache_ttl_hours,
            )
        else:
            self.cache = None
    
    def load(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> pl.DataFrame:
        """
        Load OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            timeframe: Data timeframe
        
        Returns:
            DataFrame with OHLCV data
        
        Raises:
            DataNotFoundError: If data file not found
            DataValidationError: If data fails validation
        """
        # Check cache first
        cache_id = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        if self.cache:
            cached = self.cache.get(cache_id)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol}")
                return cached
        
        # Find data file
        file_path = self._find_data_file(symbol, timeframe)
        if file_path is None:
            raise DataNotFoundError(f"No data file found for {symbol}")
        
        # Load data
        logger.info(f"Loading {symbol} from {file_path}")
        df = self._load_csv(file_path)
        
        # Apply date filters
        if start_date:
            df = df.filter(pl.col("timestamp") >= start_date)
        if end_date:
            df = df.filter(pl.col("timestamp") <= end_date)
        
        # Add symbol column
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        
        # Cache result
        if self.cache and len(df) > 0:
            self.cache.set(cache_id, df)
        
        return df
    
    def _find_data_file(
        self,
        symbol: str,
        timeframe: str,
    ) -> Path | None:
        """Find data file for symbol."""
        # Try common naming patterns
        patterns = [
            f"{symbol}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe}.csv",
            f"{symbol.lower()}_{timeframe}.csv",
            f"{symbol}.csv",
            f"{symbol.upper()}.csv",
        ]
        
        for pattern in patterns:
            file_path = self.storage_path / pattern
            if file_path.exists():
                return file_path
        
        return None
    
    def _load_csv(self, file_path: Path) -> pl.DataFrame:
        """Load and parse CSV file."""
        try:
            # Read with Polars (much faster than pandas)
            df = pl.read_csv(
                file_path,
                try_parse_dates=True,
                ignore_errors=True,
            )
            
            # Normalize column names
            df = df.rename({col: col.lower().strip() for col in df.columns})
            
            # Ensure timestamp column
            if "timestamp" not in df.columns:
                if "date" in df.columns:
                    df = df.rename({"date": "timestamp"})
                elif "datetime" in df.columns:
                    df = df.rename({"datetime": "timestamp"})
                else:
                    raise DataValidationError("No timestamp column found")
            
            # Parse timestamp if string
            if df["timestamp"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(
                        pl.Datetime,
                        format="%Y-%m-%d %H:%M:%S",
                        strict=False,
                    )
                )
            
            # Validate required columns
            missing = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing:
                raise DataValidationError(f"Missing columns: {missing}")
            
            # Cast to proper types
            df = df.with_columns([
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ])
            
            # Sort by timestamp
            df = df.sort("timestamp")
            
            return df
            
        except Exception as e:
            raise DataError(f"Failed to load {file_path}: {e}")
    
    def get_available_symbols(self) -> list[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of symbol strings
        """
        symbols = set()
        
        for file_path in self.storage_path.glob("*.csv"):
            # Extract symbol from filename
            name = file_path.stem
            # Remove timeframe suffix if present
            parts = name.split("_")
            if len(parts) >= 1:
                symbols.add(parts[0].upper())
        
        return sorted(symbols)
    
    def load_multiple(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
        max_workers: int = 4,
    ) -> dict[str, pl.DataFrame]:
        """
        Load data for multiple symbols in parallel.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date filter
            end_date: End date filter
            timeframe: Data timeframe
            max_workers: Maximum parallel workers
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results: dict[str, pl.DataFrame] = {}
        errors: list[str] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.load, symbol, start_date, end_date, timeframe
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Failed to load {symbol}: {e}")
                    errors.append(symbol)
        
        if errors:
            logger.warning(f"Failed to load {len(errors)} symbols: {errors}")
        
        return results


# =============================================================================
# GENERIC DATA LOADER
# =============================================================================

class DataLoader:
    """
    Unified data loader supporting multiple sources.
    
    Sources:
    - CSV files (local)
    - APIs (Alpaca, Alpha Vantage, etc.)
    - Databases (TimescaleDB, PostgreSQL)
    """
    
    def __init__(
        self,
        source: str = "csv",
        **kwargs: Any,
    ):
        """
        Initialize data loader.
        
        Args:
            source: Data source type ("csv", "alpaca", "database")
            **kwargs: Source-specific arguments
        """
        self.source = source
        self._loader = self._create_loader(source, **kwargs)
    
    def _create_loader(self, source: str, **kwargs: Any) -> Any:
        """Create source-specific loader."""
        if source == "csv":
            return CSVLoader(**kwargs)
        elif source == "alpaca":
            return AlpacaLoader(**kwargs)
        elif source == "database":
            return DatabaseLoader(**kwargs)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def load(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> pl.DataFrame:
        """Load data for a symbol."""
        return self._loader.load(symbol, start_date, end_date, timeframe)
    
    def load_multiple(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> dict[str, pl.DataFrame]:
        """Load data for multiple symbols."""
        return self._loader.load_multiple(
            symbols, start_date, end_date, timeframe
        )
    
    def get_available_symbols(self) -> list[str]:
        """Get available symbols."""
        return self._loader.get_available_symbols()


# =============================================================================
# ALPACA LOADER (Placeholder for Phase 5)
# =============================================================================

class AlpacaLoader:
    """
    Alpaca API data loader.
    
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
    """
    
    def __init__(self, **kwargs: Any):
        """Initialize Alpaca loader."""
        settings = get_settings()
        self.api_key = settings.alpaca.api_key
        self.secret_key = settings.alpaca.secret_key
        self.base_url = settings.alpaca.base_url
        self._client = None
    
    def _get_client(self) -> Any:
        """Get or create Alpaca client."""
        if self._client is None:
            try:
                from alpaca.data import StockHistoricalDataClient
                self._client = StockHistoricalDataClient(
                    self.api_key,
                    self.secret_key,
                )
            except ImportError:
                raise DataError("alpaca-py not installed")
            except Exception as e:
                raise DataError(f"Failed to create Alpaca client: {e}")
        return self._client
    
    def load(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> pl.DataFrame:
        """Load data from Alpaca API."""
        # Placeholder - will be implemented in Phase 5
        raise NotImplementedError("Alpaca loader will be implemented in Phase 5")
    
    def load_multiple(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> dict[str, pl.DataFrame]:
        """Load multiple symbols from Alpaca."""
        raise NotImplementedError("Alpaca loader will be implemented in Phase 5")
    
    def get_available_symbols(self) -> list[str]:
        """Get tradable symbols from Alpaca."""
        raise NotImplementedError("Alpaca loader will be implemented in Phase 5")


# =============================================================================
# DATABASE LOADER (Placeholder for Production)
# =============================================================================

class DatabaseLoader:
    """
    Database data loader for TimescaleDB/PostgreSQL.
    
    Optimized for time-series queries.
    """
    
    def __init__(self, **kwargs: Any):
        """Initialize database loader."""
        settings = get_settings()
        self.db_url = settings.database.url
        self._engine = None
    
    def load(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> pl.DataFrame:
        """Load data from database."""
        raise NotImplementedError("Database loader will be implemented in production")
    
    def load_multiple(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> dict[str, pl.DataFrame]:
        """Load multiple symbols from database."""
        raise NotImplementedError("Database loader will be implemented in production")
    
    def get_available_symbols(self) -> list[str]:
        """Get symbols from database."""
        raise NotImplementedError("Database loader will be implemented in production")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_csv_data(
    symbol: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    timeframe: str = "15min",
    storage_path: Path | None = None,
) -> pl.DataFrame:
    """
    Convenience function to load CSV data.
    
    Args:
        symbol: Trading symbol
        start_date: Start date filter
        end_date: End date filter
        timeframe: Data timeframe
        storage_path: Optional custom storage path
    
    Returns:
        DataFrame with OHLCV data
    """
    loader = CSVLoader(storage_path=storage_path)
    return loader.load(symbol, start_date, end_date, timeframe)


def load_all_symbols(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    timeframe: str = "15min",
    storage_path: Path | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Load data for all available symbols.
    
    Args:
        start_date: Start date filter
        end_date: End date filter
        timeframe: Data timeframe
        storage_path: Optional custom storage path
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    loader = CSVLoader(storage_path=storage_path)
    symbols = loader.get_available_symbols()
    return loader.load_multiple(symbols, start_date, end_date, timeframe)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DataCache",
    "CSVLoader",
    "AlpacaLoader",
    "DatabaseLoader",
    "DataLoader",
    "load_csv_data",
    "load_all_symbols",
    "OHLCV_SCHEMA",
    "REQUIRED_COLUMNS",
]