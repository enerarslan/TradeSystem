"""
Data storage and caching modules.

This module provides:
- In-memory caching (DataCache)
- TimescaleDB time-series database client (optional)
"""

from src.data.storage.cache import DataCache, cache_data, load_cached_data

# TimescaleDB client (optional - requires psycopg2/asyncpg)
try:
    from src.data.storage.timescale_client import (
        TimescaleClient,
        AsyncTimescaleClient,
        ConnectionConfig,
    )
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False
    TimescaleClient = None
    AsyncTimescaleClient = None
    ConnectionConfig = None

__all__ = [
    "DataCache",
    "cache_data",
    "load_cached_data",
    "TimescaleClient",
    "AsyncTimescaleClient",
    "ConnectionConfig",
    "TIMESCALE_AVAILABLE",
]
