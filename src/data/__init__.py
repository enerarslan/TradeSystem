"""
Data layer for AlphaTrade system.

This module provides comprehensive data handling capabilities:
- Data loading from multiple formats (CSV, Parquet, Feather)
- Data validation and quality checks
- Data preprocessing and cleaning
- Data storage and caching
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

__all__ = [
    "DataLoader",
    "load_single_stock",
    "load_all_stocks",
    "get_available_symbols",
    "DataValidator",
    "ValidationResult",
    "validate_ohlcv",
    "DataProcessor",
    "preprocess_ohlcv",
    "create_train_test_split",
]
