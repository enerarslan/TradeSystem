"""
Data layer for AlphaTrade system.

This module provides comprehensive data handling capabilities:
- Data loading from multiple formats (CSV, Parquet, Feather)
- High-performance loading with Polars
- Data validation and quality checks
- Data preprocessing and cleaning
- Data storage and caching
- TimescaleDB time-series database support
"""

from src.data.loaders.data_loader import (
    DataLoader,
    load_single_stock,
    load_all_stocks,
    get_available_symbols,
)
from src.data.validators.data_validator import (
    DataValidator,
    ValidationResult,
    validate_ohlcv,
)
from src.data.processors.data_processor import (
    DataProcessor,
    preprocess_ohlcv,
    create_train_test_split,
)
from src.data.storage import (
    DataCache,
    TIMESCALE_AVAILABLE,
)
from src.data.loaders import POLARS_AVAILABLE

# Optional imports
if POLARS_AVAILABLE:
    from src.data.loaders import PolarsDataLoader

if TIMESCALE_AVAILABLE:
    from src.data.storage import TimescaleClient, AsyncTimescaleClient

__all__ = [
    # Standard loaders
    "DataLoader",
    "load_single_stock",
    "load_all_stocks",
    "get_available_symbols",
    # Validation
    "DataValidator",
    "ValidationResult",
    "validate_ohlcv",
    # Processing
    "DataProcessor",
    "preprocess_ohlcv",
    "create_train_test_split",
    # Storage
    "DataCache",
    # Availability flags
    "POLARS_AVAILABLE",
    "TIMESCALE_AVAILABLE",
]
