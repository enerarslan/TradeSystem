"""
TimescaleDB Data Loader
=======================

High-performance data loading from TimescaleDB for the trading system.
Replaces CSV-based loading with database queries.

Features:
- Async database operations
- Connection pooling
- Efficient time-series queries
- Automatic caching
- Batch inserts for real-time data

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
import os

import polars as pl
import numpy as np

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from config.settings import get_logger

logger = get_logger(__name__)


class TimescaleLoader:
    """
    High-performance data loader for TimescaleDB.

    Provides fast queries for:
    - Historical OHLCV data
    - Real-time bar updates
    - Feature vectors
    - Aggregated timeframes

    Example:
        loader = TimescaleLoader(database_url="postgresql://user:pass@localhost/market_data")
        await loader.connect()

        # Get last 6 months of 1-minute bars
        df = await loader.get_bars("AAPL", timeframe="1min", days=180)

        # Get multiple symbols
        data = await loader.get_multi_symbol_bars(
            ["AAPL", "MSFT", "GOOGL"],
            timeframe="1h",
            days=30
        )
    """

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int = 10,
        cache_ttl: int = 60,
    ):
        """
        Initialize TimescaleDB loader.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            cache_ttl: Cache time-to-live in seconds
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg required: pip install asyncpg")

        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://alphatrade:alphatrade_secret@localhost:5432/market_data"
        )
        self.pool_size = pool_size
        self.cache_ttl = cache_ttl

        self._pool: asyncpg.Pool | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect to TimescaleDB."""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=60,
            )
            self._connected = True
            logger.info("Connected to TimescaleDB")

        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from TimescaleDB."""
        if self._pool:
            await self._pool.close()
        self._connected = False
        logger.info("Disconnected from TimescaleDB")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1min",
        days: int = 30,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Get OHLCV bars for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1min, 5min, 1h, daily)
            days: Number of days to fetch (if start_date not specified)
            start_date: Start datetime
            end_date: End datetime

        Returns:
            Polars DataFrame with OHLCV data
        """
        # Check cache
        cache_key = f"bars:{symbol}:{timeframe}:{days}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if not self._pool:
            raise RuntimeError("Not connected to database")

        # Determine table based on timeframe
        if timeframe == "5min":
            table = "bars_5min"
        elif timeframe in ("1h", "1hour"):
            table = "bars_1h"
        elif timeframe == "daily":
            table = "bars_daily"
        else:
            table = "bars"

        # Build query
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=days))

        query = f"""
            SELECT time, open, high, low, close, volume, vwap
            FROM {table}
            WHERE symbol = $1
              AND time >= $2
              AND time <= $3
            ORDER BY time ASC
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_date, end_date)

        # Convert to Polars DataFrame
        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame({
            "timestamp": [r["time"] for r in rows],
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [int(r["volume"]) for r in rows],
        })

        if rows[0]["vwap"]:
            df = df.with_columns(pl.Series("vwap", [float(r["vwap"]) for r in rows]))

        # Cache result
        self._set_cache(cache_key, df)

        logger.debug(f"Loaded {len(df)} bars for {symbol} ({timeframe})")
        return df

    async def get_multi_symbol_bars(
        self,
        symbols: list[str],
        timeframe: str = "1min",
        days: int = 30,
    ) -> dict[str, pl.DataFrame]:
        """
        Get bars for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            days: Number of days

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        tasks = [
            self.get_bars(symbol, timeframe, days)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {symbol}: {result}")
                data[symbol] = pl.DataFrame()
            else:
                data[symbol] = result

        return data

    async def get_latest_bars(
        self,
        symbol: str,
        count: int = 100,
    ) -> pl.DataFrame:
        """
        Get the latest N bars for a symbol.

        Args:
            symbol: Trading symbol
            count: Number of bars to fetch

        Returns:
            Polars DataFrame with latest bars
        """
        if not self._pool:
            raise RuntimeError("Not connected to database")

        query = """
            SELECT time, open, high, low, close, volume
            FROM bars
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, count)

        if not rows:
            return pl.DataFrame()

        # Reverse to chronological order
        rows = list(reversed(rows))

        return pl.DataFrame({
            "timestamp": [r["time"] for r in rows],
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [int(r["volume"]) for r in rows],
        })

    async def get_latest_price(self, symbol: str) -> float | None:
        """
        Get the latest price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest close price or None
        """
        if not self._pool:
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT get_latest_price($1) AS price",
                symbol
            )

        return float(row["price"]) if row and row["price"] else None

    async def insert_bar(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        vwap: float | None = None,
    ) -> bool:
        """
        Insert a single bar.

        Args:
            symbol: Trading symbol
            timestamp: Bar timestamp
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            vwap: VWAP (optional)

        Returns:
            True if inserted successfully
        """
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bars (time, symbol, open, high, low, close, volume, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap
                    """,
                    timestamp, symbol, open_, high, low, close, volume, vwap
                )
            return True

        except Exception as e:
            logger.error(f"Failed to insert bar: {e}")
            return False

    async def insert_bars_batch(
        self,
        bars: list[dict[str, Any]],
    ) -> int:
        """
        Insert multiple bars in a batch.

        Args:
            bars: List of bar dictionaries

        Returns:
            Number of bars inserted
        """
        if not self._pool or not bars:
            return 0

        try:
            async with self._pool.acquire() as conn:
                # Prepare data for copy
                records = [
                    (
                        bar["timestamp"],
                        bar["symbol"],
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["volume"],
                        bar.get("vwap"),
                    )
                    for bar in bars
                ]

                await conn.executemany(
                    """
                    INSERT INTO bars (time, symbol, open, high, low, close, volume, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap
                    """,
                    records
                )

            logger.debug(f"Inserted {len(bars)} bars")
            return len(bars)

        except Exception as e:
            logger.error(f"Failed to insert bars batch: {e}")
            return 0

    async def get_feature_vector(
        self,
        symbol: str,
        timestamp: datetime | None = None,
    ) -> dict[str, float] | None:
        """
        Get pre-computed feature vector for a symbol.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp (default: latest)

        Returns:
            Dictionary of features
        """
        if not self._pool:
            return None

        if timestamp:
            query = """
                SELECT features
                FROM feature_vectors
                WHERE symbol = $1 AND time <= $2
                ORDER BY time DESC
                LIMIT 1
            """
            params = (symbol, timestamp)
        else:
            query = """
                SELECT features
                FROM feature_vectors
                WHERE symbol = $1
                ORDER BY time DESC
                LIMIT 1
            """
            params = (symbol,)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row and row["features"]:
            import json
            return json.loads(row["features"])

        return None

    async def insert_feature_vector(
        self,
        symbol: str,
        timestamp: datetime,
        features: dict[str, float],
    ) -> bool:
        """
        Insert a feature vector.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            features: Feature dictionary

        Returns:
            True if inserted successfully
        """
        if not self._pool:
            return False

        try:
            import json

            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO feature_vectors (time, symbol, features)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (time, symbol) DO UPDATE SET
                        features = EXCLUDED.features
                    """,
                    timestamp, symbol, json.dumps(features)
                )
            return True

        except Exception as e:
            logger.error(f"Failed to insert feature vector: {e}")
            return False

    def _get_from_cache(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (datetime.now().timestamp(), value)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_timescale_loader(
    database_url: str | None = None,
) -> TimescaleLoader:
    """
    Create and connect a TimescaleDB loader.

    Args:
        database_url: Database URL

    Returns:
        Connected TimescaleLoader instance
    """
    loader = TimescaleLoader(database_url=database_url)
    await loader.connect()
    return loader


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TimescaleLoader",
    "create_timescale_loader",
]
