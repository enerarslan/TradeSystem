"""
Data Provider Module
====================

Unified data access interface for the trading platform.
Abstracts data source details and provides consistent API.

Features:
- Historical data retrieval
- Multi-symbol data loading
- Automatic caching
- Data preprocessing
- Symbol universe management

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from config.settings import get_settings, get_logger, TimeFrame
from core.types import DataNotFoundError
from core.events import MarketEvent, MultiAssetMarketEvent
from data.loader import CSVLoader, DataLoader
from data.processor import DataProcessor, DataValidator

logger = get_logger(__name__)


# =============================================================================
# HISTORICAL DATA PROVIDER
# =============================================================================

class HistoricalDataProvider:
    """
    Provider for historical OHLCV data.
    
    Handles data loading, validation, and preprocessing
    for backtesting and analysis.
    
    Example:
        provider = HistoricalDataProvider()
        data = provider.get_data("AAPL", start_date, end_date)
        
        # Or multiple symbols
        all_data = provider.get_multi_data(["AAPL", "GOOGL", "MSFT"])
    """
    
    def __init__(
        self,
        source: str = "csv",
        storage_path: Path | None = None,
        use_cache: bool = True,
        auto_process: bool = True,
        validate: bool = True,
    ):
        """
        Initialize provider.
        
        Args:
            source: Data source type
            storage_path: Path to data storage
            use_cache: Enable caching
            auto_process: Auto-clean data
            validate: Validate data on load
        """
        settings = get_settings()
        
        self.source = source
        self.storage_path = storage_path or settings.data.storage_path
        self.use_cache = use_cache
        self.auto_process = auto_process
        self.validate = validate
        
        # Initialize loader
        self._loader = CSVLoader(
            storage_path=self.storage_path,
            use_cache=use_cache,
        )
        
        # Initialize processor
        self._processor = DataProcessor() if auto_process else None
        self._validator = DataValidator() if validate else None
        
        # Data cache (in memory)
        self._data_cache: dict[str, pl.DataFrame] = {}
        
        # Symbol universe
        self._symbols: list[str] = []
    
    @property
    def symbols(self) -> list[str]:
        """Get available symbols."""
        if not self._symbols:
            self._symbols = self._loader.get_available_symbols()
        return self._symbols
    
    def get_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
        use_memory_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            use_memory_cache: Use in-memory cache
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        # Check memory cache
        if use_memory_cache and cache_key in self._data_cache:
            logger.debug(f"Memory cache hit for {symbol}")
            return self._data_cache[cache_key]
        
        # Load data
        df = self._loader.load(symbol, start_date, end_date, timeframe)
        
        # Validate
        if self._validator:
            result = self._validator.validate(df)
            if not result.is_valid:
                logger.warning(f"Validation issues for {symbol}: {result.errors}")
        
        # Process
        if self._processor:
            df = self._processor.process(df, symbol=symbol)
        
        # Cache
        if use_memory_cache:
            self._data_cache[cache_key] = df
        
        return df
    
    def get_multi_data(
        self,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> dict[str, pl.DataFrame]:
        """
        Get data for multiple symbols.
        
        Args:
            symbols: List of symbols (None = all available)
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.symbols
        
        result = {}
        errors = []
        
        for symbol in symbols:
            try:
                result[symbol] = self.get_data(
                    symbol, start_date, end_date, timeframe
                )
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                errors.append(symbol)
        
        if errors:
            logger.warning(f"Failed to load {len(errors)} symbols")
        
        logger.info(f"Loaded {len(result)} symbols")
        return result
    
    def get_combined_data(
        self,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> pl.DataFrame:
        """
        Get combined data for multiple symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
        
        Returns:
            Single DataFrame with all symbols (stacked)
        """
        multi_data = self.get_multi_data(symbols, start_date, end_date, timeframe)
        
        if not multi_data:
            raise DataNotFoundError("No data loaded")
        
        return pl.concat(list(multi_data.values()))
    
    def get_market_event(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> MarketEvent:
        """
        Get data as a MarketEvent.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
        
        Returns:
            MarketEvent with data
        """
        data = self.get_data(symbol, start_date, end_date, timeframe)
        
        return MarketEvent(
            symbol=symbol,
            data=data,
            timeframe=timeframe,
            is_realtime=False,
        )
    
    def get_multi_asset_event(
        self,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> MultiAssetMarketEvent:
        """
        Get multi-asset data as an event.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
        
        Returns:
            MultiAssetMarketEvent with data
        """
        data = self.get_multi_data(symbols, start_date, end_date, timeframe)
        
        return MultiAssetMarketEvent(
            data=data,
            timeframe=timeframe,
            is_realtime=False,
        )
    
    def get_latest_prices(
        self,
        symbols: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Get latest close prices for symbols.
        
        Args:
            symbols: List of symbols (None = all cached)
        
        Returns:
            Dictionary of symbol to price
        """
        prices = {}
        
        target_symbols = symbols or list(self._data_cache.keys())
        
        for key in target_symbols:
            if key in self._data_cache:
                df = self._data_cache[key]
                if len(df) > 0:
                    symbol = df["symbol"].first() if "symbol" in df.columns else key.split("_")[0]
                    prices[symbol] = df["close"].last()
        
        return prices
    
    def preload(
        self,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = "15min",
    ) -> None:
        """
        Preload data into memory cache.
        
        Args:
            symbols: Symbols to preload
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
        """
        logger.info("Preloading data into memory cache")
        self.get_multi_data(symbols, start_date, end_date, timeframe)
    
    def clear_cache(self) -> None:
        """Clear memory cache."""
        self._data_cache.clear()
        logger.info("Memory cache cleared")
    
    def get_date_range(
        self,
        symbol: str,
        timeframe: str = "15min",
    ) -> tuple[datetime, datetime]:
        """
        Get available date range for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
        
        Returns:
            Tuple of (start_date, end_date)
        """
        data = self.get_data(symbol, timeframe=timeframe)
        return (data["timestamp"].min(), data["timestamp"].max())
    
    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """
        Get information about a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dictionary with symbol information
        """
        try:
            data = self.get_data(symbol)
            return {
                "symbol": symbol,
                "rows": len(data),
                "start_date": data["timestamp"].min(),
                "end_date": data["timestamp"].max(),
                "latest_close": data["close"].last(),
                "avg_volume": data["volume"].mean(),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}


# =============================================================================
# DATA PROVIDER FACTORY
# =============================================================================

class DataProviderFactory:
    """
    Factory for creating data providers.
    
    Supports different data sources and configurations.
    """
    
    @staticmethod
    def create(
        source: str = "csv",
        **kwargs: Any,
    ) -> HistoricalDataProvider:
        """
        Create a data provider.
        
        Args:
            source: Data source type
            **kwargs: Additional arguments
        
        Returns:
            Configured data provider
        """
        return HistoricalDataProvider(source=source, **kwargs)
    
    @staticmethod
    def create_from_settings() -> HistoricalDataProvider:
        """
        Create provider from settings.
        
        Returns:
            Configured data provider
        """
        settings = get_settings()
        
        return HistoricalDataProvider(
            source="csv",
            storage_path=settings.data.storage_path,
            use_cache=settings.data.use_cache,
            auto_process=True,
            validate=True,
        )


# =============================================================================
# BAR ITERATOR
# =============================================================================

class BarIterator:
    """
    Iterator for streaming bars during backtesting.
    
    Yields one bar at a time for event-driven processing.
    
    Example:
        for bar_event in BarIterator(data, "AAPL"):
            strategy.on_bar(bar_event)
    """
    
    def __init__(
        self,
        data: pl.DataFrame,
        symbol: str,
        timeframe: str = "15min",
    ):
        """
        Initialize iterator.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Data timeframe
        """
        self.data = data.sort("timestamp")
        self.symbol = symbol
        self.timeframe = timeframe
        self._index = 0
        self._len = len(data)
    
    def __iter__(self) -> "BarIterator":
        return self
    
    def __next__(self) -> MarketEvent:
        if self._index >= self._len:
            raise StopIteration
        
        # Get current bar and all history up to this point
        current_idx = self._index
        self._index += 1
        
        # Include all bars up to current for indicator calculation
        history = self.data.slice(0, current_idx + 1)
        
        return MarketEvent(
            symbol=self.symbol,
            data=history,
            timeframe=self.timeframe,
            is_realtime=False,
        )
    
    def __len__(self) -> int:
        return self._len
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._index = 0


class MultiAssetBarIterator:
    """
    Iterator for streaming bars from multiple assets.
    
    Synchronizes bars across assets by timestamp.
    """
    
    def __init__(
        self,
        data: dict[str, pl.DataFrame],
        timeframe: str = "15min",
    ):
        """
        Initialize iterator.
        
        Args:
            data: Dictionary of symbol to DataFrame
            timeframe: Data timeframe
        """
        self.data = {k: v.sort("timestamp") for k, v in data.items()}
        self.timeframe = timeframe
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in self.data.values():
            all_timestamps.update(df["timestamp"].to_list())
        
        self.timestamps = sorted(all_timestamps)
        self._index = 0
        self._len = len(self.timestamps)
    
    def __iter__(self) -> "MultiAssetBarIterator":
        return self
    
    def __next__(self) -> MultiAssetMarketEvent:
        if self._index >= self._len:
            raise StopIteration
        
        current_ts = self.timestamps[self._index]
        self._index += 1
        
        # Get history for each symbol up to current timestamp
        event_data = {}
        for symbol, df in self.data.items():
            history = df.filter(pl.col("timestamp") <= current_ts)
            if len(history) > 0:
                event_data[symbol] = history
        
        return MultiAssetMarketEvent(
            data=event_data,
            timeframe=self.timeframe,
            is_realtime=False,
        )
    
    def __len__(self) -> int:
        return self._len
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._index = 0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HistoricalDataProvider",
    "DataProviderFactory",
    "BarIterator",
    "MultiAssetBarIterator",
]