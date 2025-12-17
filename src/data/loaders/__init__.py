"""
Data loading modules.

This module provides:
- DataLoader: Standard pandas-based CSV/Parquet loader
- PolarsDataLoader: High-performance loader using Polars (optional)
"""

from src.data.loaders.data_loader import (
    DataLoader,
    load_single_stock,
    load_all_stocks,
    get_available_symbols,
)

# Polars loader (optional - requires polars)
try:
    from src.data.loaders.polars_loader import (
        PolarsDataLoader,
        PolarsFeatureEngine,
        PolarsCrossSectionalFeatures,
    )
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    PolarsDataLoader = None
    PolarsFeatureEngine = None
    PolarsCrossSectionalFeatures = None

__all__ = [
    "DataLoader",
    "load_single_stock",
    "load_all_stocks",
    "get_available_symbols",
    "PolarsDataLoader",
    "PolarsFeatureEngine",
    "PolarsCrossSectionalFeatures",
    "POLARS_AVAILABLE",
]
