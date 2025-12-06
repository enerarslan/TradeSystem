"""
Data Module
===========

Data loading, processing, and providing for the algorithmic trading platform.

Components:
- loader: CSV, API, and database data loading
- processor: Data cleaning, validation, and resampling
- provider: Unified data access interface

Author: Algo Trading Platform
License: MIT
"""

from data.loader import (
    DataLoader,
    CSVLoader,
    load_csv_data,
    load_all_symbols,
)
from data.processor import (
    DataProcessor,
    DataValidator,
    clean_ohlcv_data,
    resample_ohlcv,
    normalize_data,
)
from data.provider import (
    HistoricalDataProvider,
    DataProviderFactory,
)

__all__ = [
    # Loader
    "DataLoader",
    "CSVLoader",
    "load_csv_data",
    "load_all_symbols",
    # Processor
    "DataProcessor",
    "DataValidator",
    "clean_ohlcv_data",
    "resample_ohlcv",
    "normalize_data",
    # Provider
    "HistoricalDataProvider",
    "DataProviderFactory",
]