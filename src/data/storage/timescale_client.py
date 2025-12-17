"""
TimescaleDB client for institutional-grade time-series data storage.

This module provides a high-performance interface to TimescaleDB for storing
and retrieving tick data, OHLCV bars, and other time-series financial data.

Based on JPMorgan-level requirements for:
- Sub-millisecond query performance
- Petabyte-scale data handling
- Automatic data compression and retention
- Point-in-time correct queries
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generator, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import execute_batch, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Supported timeframes for OHLCV data."""
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class Tick:
    """Single tick data point."""
    timestamp: datetime
    symbol: str
    price: float
    size: int
    exchange: Optional[str] = None
    conditions: Optional[str] = None


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "alphatrade_db"
    user: str = "alphatrade"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: int = 30
    command_timeout: int = 60
    ssl_mode: str = "prefer"

    def to_dsn(self) -> str:
        """Convert to connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to psycopg2 connection parameters."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "connect_timeout": self.connection_timeout,
        }


@dataclass
class QueryResult:
    """Result container for database queries."""
    data: pd.DataFrame
    query_time_ms: float
    rows_fetched: int
    cache_hit: bool = False


class RetryPolicy:
    """Retry policy for database operations."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


class TimescaleClient:
    """
    Synchronous TimescaleDB client for financial time-series data.

    Features:
    - Connection pooling for high throughput
    - Batch inserts using COPY for maximum performance
    - Automatic retry with exponential backoff
    - Query timeout handling
    - Transaction support

    Example:
        config = ConnectionConfig(
            host="localhost",
            database="alphatrade_db",
            user="trader",
            password="secret"
        )
        client = TimescaleClient(config)
        client.connect()

        # Insert ticks
        ticks = [Tick(datetime.now(), "AAPL", 150.0, 100)]
        client.insert_ticks(ticks)

        # Query OHLCV
        df = client.get_ohlcv("AAPL", start, end, "15min")
    """

    def __init__(
        self,
        config: ConnectionConfig,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for TimescaleClient. "
                "Install with: pip install psycopg2-binary"
            )

        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialized = False

    def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return

        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                **self.config.to_dict()
            )
            logger.info(
                f"Connected to TimescaleDB at {self.config.host}:{self.config.port}"
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    def disconnect(self) -> None:
        """Close all connections."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("Disconnected from TimescaleDB")

    @contextmanager
    def _get_connection(self) -> Generator:
        """Get a connection from the pool."""
        if self._pool is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def _execute_with_retry(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        last_error = None

        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                last_error = e
                if attempt < self.retry_policy.max_retries:
                    delay = self.retry_policy.get_delay(attempt)
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    import time
                    time.sleep(delay)
                    # Reconnect on connection errors
                    self.disconnect()
                    self.connect()

        raise last_error

    def initialize_schema(self) -> None:
        """Create required tables and hypertables."""
        schema_sql = """
        -- Enable TimescaleDB extension
        CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

        -- Tick data table
        CREATE TABLE IF NOT EXISTS tick_data (
            time TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            size BIGINT NOT NULL,
            exchange VARCHAR(10),
            conditions VARCHAR(50)
        );

        -- Convert to hypertable if not already
        SELECT create_hypertable('tick_data', 'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );

        -- OHLCV bars table
        CREATE TABLE IF NOT EXISTS ohlcv_bars (
            time TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            vwap DOUBLE PRECISION,
            trade_count INTEGER
        );

        SELECT create_hypertable('ohlcv_bars', 'time',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_tick_symbol_time
            ON tick_data (symbol, time DESC);
        CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_time
            ON ohlcv_bars (symbol, timeframe, time DESC);

        -- Symbol metadata table
        CREATE TABLE IF NOT EXISTS symbol_metadata (
            symbol VARCHAR(20) PRIMARY KEY,
            name VARCHAR(200),
            sector VARCHAR(100),
            industry VARCHAR(100),
            exchange VARCHAR(20),
            lot_size INTEGER DEFAULT 1,
            tick_size DOUBLE PRECISION DEFAULT 0.01,
            currency VARCHAR(10) DEFAULT 'USD',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Corporate actions table
        CREATE TABLE IF NOT EXISTS corporate_actions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            action_type VARCHAR(20) NOT NULL,
            ex_date DATE NOT NULL,
            record_date DATE,
            pay_date DATE,
            ratio DOUBLE PRECISION,
            amount DOUBLE PRECISION,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol_date
            ON corporate_actions (symbol, ex_date DESC);
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()

        self._initialized = True
        logger.info("TimescaleDB schema initialized")

    def setup_continuous_aggregates(self) -> None:
        """Create continuous aggregates for automatic OHLCV rollups."""
        # 15-minute continuous aggregate from ticks
        agg_15min = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_15min
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('15 minutes', time) AS bucket,
            symbol,
            first(price, time) AS open,
            max(price) AS high,
            min(price) AS low,
            last(price, time) AS close,
            sum(size) AS volume,
            sum(price * size) / NULLIF(sum(size), 0) AS vwap,
            count(*) AS trade_count
        FROM tick_data
        GROUP BY bucket, symbol
        WITH NO DATA;

        -- Refresh policy
        SELECT add_continuous_aggregate_policy('ohlcv_15min',
            start_offset => INTERVAL '1 day',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '15 minutes',
            if_not_exists => TRUE
        );
        """

        # 1-hour continuous aggregate
        agg_1h = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            symbol,
            first(price, time) AS open,
            max(price) AS high,
            min(price) AS low,
            last(price, time) AS close,
            sum(size) AS volume,
            sum(price * size) / NULLIF(sum(size), 0) AS vwap,
            count(*) AS trade_count
        FROM tick_data
        GROUP BY bucket, symbol
        WITH NO DATA;

        SELECT add_continuous_aggregate_policy('ohlcv_1h',
            start_offset => INTERVAL '7 days',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
        """

        # Daily continuous aggregate
        agg_daily = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', time) AS bucket,
            symbol,
            first(price, time) AS open,
            max(price) AS high,
            min(price) AS low,
            last(price, time) AS close,
            sum(size) AS volume,
            sum(price * size) / NULLIF(sum(size), 0) AS vwap,
            count(*) AS trade_count
        FROM tick_data
        GROUP BY bucket, symbol
        WITH NO DATA;

        SELECT add_continuous_aggregate_policy('ohlcv_daily',
            start_offset => INTERVAL '30 days',
            end_offset => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for agg_sql in [agg_15min, agg_1h, agg_daily]:
                    try:
                        cur.execute(agg_sql)
                    except psycopg2.Error as e:
                        logger.warning(f"Continuous aggregate setup warning: {e}")
            conn.commit()

        logger.info("Continuous aggregates configured")

    def setup_retention_policies(
        self,
        tick_retention_days: int = 30,
        bar_retention_years: int = 10,
    ) -> None:
        """Configure data retention policies."""
        retention_sql = f"""
        -- Tick data retention (raw ticks are expensive to store)
        SELECT add_retention_policy('tick_data',
            INTERVAL '{tick_retention_days} days',
            if_not_exists => TRUE
        );

        -- OHLCV bars retention (keep longer for analysis)
        SELECT add_retention_policy('ohlcv_bars',
            INTERVAL '{bar_retention_years} years',
            if_not_exists => TRUE
        );
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(retention_sql)
            conn.commit()

        logger.info(
            f"Retention policies set: ticks={tick_retention_days}d, "
            f"bars={bar_retention_years}y"
        )

    def setup_compression(self) -> None:
        """Enable compression for historical data."""
        compression_sql = """
        -- Enable compression on tick_data
        ALTER TABLE tick_data SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'time DESC'
        );

        SELECT add_compression_policy('tick_data',
            INTERVAL '7 days',
            if_not_exists => TRUE
        );

        -- Enable compression on ohlcv_bars
        ALTER TABLE ohlcv_bars SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, timeframe',
            timescaledb.compress_orderby = 'time DESC'
        );

        SELECT add_compression_policy('ohlcv_bars',
            INTERVAL '30 days',
            if_not_exists => TRUE
        );
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(compression_sql)
                except psycopg2.Error as e:
                    logger.warning(f"Compression setup warning: {e}")
            conn.commit()

        logger.info("Compression policies configured")

    def insert_ticks(
        self,
        ticks: Sequence[Tick],
        batch_size: int = 10000,
    ) -> int:
        """
        Batch insert tick data.

        Uses COPY for maximum performance.

        Args:
            ticks: Sequence of Tick objects
            batch_size: Rows per batch

        Returns:
            Number of rows inserted
        """
        if not ticks:
            return 0

        def _insert():
            total_inserted = 0

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare data
                    values = [
                        (
                            t.timestamp,
                            t.symbol,
                            t.price,
                            t.size,
                            t.exchange,
                            t.conditions
                        )
                        for t in ticks
                    ]

                    # Batch insert using execute_values (efficient)
                    for i in range(0, len(values), batch_size):
                        batch = values[i:i + batch_size]
                        execute_values(
                            cur,
                            """
                            INSERT INTO tick_data
                            (time, symbol, price, size, exchange, conditions)
                            VALUES %s
                            """,
                            batch,
                            page_size=batch_size
                        )
                        total_inserted += len(batch)

                conn.commit()

            return total_inserted

        return self._execute_with_retry(_insert)

    def insert_ohlcv(
        self,
        bars: Sequence[Bar],
        batch_size: int = 10000,
    ) -> int:
        """
        Batch insert OHLCV bars.

        Args:
            bars: Sequence of Bar objects
            batch_size: Rows per batch

        Returns:
            Number of rows inserted
        """
        if not bars:
            return 0

        def _insert():
            total_inserted = 0

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    values = [
                        (
                            b.timestamp,
                            b.symbol,
                            b.timeframe,
                            b.open,
                            b.high,
                            b.low,
                            b.close,
                            b.volume,
                            b.vwap,
                            b.trade_count
                        )
                        for b in bars
                    ]

                    for i in range(0, len(values), batch_size):
                        batch = values[i:i + batch_size]
                        execute_values(
                            cur,
                            """
                            INSERT INTO ohlcv_bars
                            (time, symbol, timeframe, open, high, low, close,
                             volume, vwap, trade_count)
                            VALUES %s
                            ON CONFLICT DO NOTHING
                            """,
                            batch,
                            page_size=batch_size
                        )
                        total_inserted += len(batch)

                conn.commit()

            return total_inserted

        return self._execute_with_retry(_insert)

    def insert_ohlcv_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """
        Insert OHLCV data from a pandas DataFrame.

        Expected columns: timestamp/time/index, open, high, low, close, volume

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            timeframe: Timeframe string (e.g., "15min")

        Returns:
            Number of rows inserted
        """
        # Standardize DataFrame
        df = df.copy()

        # Handle index as timestamp
        if df.index.name in ('timestamp', 'time', 'datetime'):
            df = df.reset_index()

        # Find timestamp column
        time_col = None
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            raise ValueError("DataFrame must have a timestamp column")

        # Convert to Bar objects
        bars = []
        for _, row in df.iterrows():
            bars.append(Bar(
                timestamp=pd.to_datetime(row[time_col]),
                symbol=symbol,
                timeframe=timeframe,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row.get('volume', 0)),
                vwap=float(row['vwap']) if 'vwap' in row and pd.notna(row['vwap']) else None,
                trade_count=int(row['trade_count']) if 'trade_count' in row and pd.notna(row['trade_count']) else None,
            ))

        return self.insert_ohlcv(bars)

    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "15min",
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV bars for a symbol.

        Args:
            symbol: Symbol to query
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)
            timeframe: Timeframe to query

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        query = """
        SELECT
            time as timestamp,
            open, high, low, close, volume, vwap, trade_count
        FROM ohlcv_bars
        WHERE symbol = %s
            AND timeframe = %s
            AND time >= %s
            AND time < %s
        ORDER BY time
        """

        def _query():
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, start, end),
                    parse_dates=['timestamp']
                )
            return df

        df = self._execute_with_retry(_query)

        if not df.empty:
            df = df.set_index('timestamp')

        return df

    def get_ohlcv_multi(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: str = "15min",
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV bars for multiple symbols.

        Returns a DataFrame with MultiIndex (symbol, timestamp).
        """
        query = """
        SELECT
            symbol,
            time as timestamp,
            open, high, low, close, volume
        FROM ohlcv_bars
        WHERE symbol = ANY(%s)
            AND timeframe = %s
            AND time >= %s
            AND time < %s
        ORDER BY symbol, time
        """

        def _query():
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(list(symbols), timeframe, start, end),
                    parse_dates=['timestamp']
                )
            return df

        df = self._execute_with_retry(_query)

        if not df.empty:
            df = df.set_index(['symbol', 'timestamp'])

        return df

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve tick data for a symbol.

        Args:
            symbol: Symbol to query
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)

        Returns:
            DataFrame with tick data
        """
        query = """
        SELECT
            time as timestamp,
            price, size, exchange, conditions
        FROM tick_data
        WHERE symbol = %s
            AND time >= %s
            AND time < %s
        ORDER BY time
        """

        def _query():
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, start, end),
                    parse_dates=['timestamp']
                )
            return df

        df = self._execute_with_retry(_query)

        if not df.empty:
            df = df.set_index('timestamp')

        return df

    def get_latest_bar(
        self,
        symbol: str,
        timeframe: str = "15min",
    ) -> Optional[Bar]:
        """Get the most recent bar for a symbol."""
        query = """
        SELECT
            time, symbol, timeframe,
            open, high, low, close, volume, vwap, trade_count
        FROM ohlcv_bars
        WHERE symbol = %s AND timeframe = %s
        ORDER BY time DESC
        LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (symbol, timeframe))
                row = cur.fetchone()

        if row is None:
            return None

        return Bar(
            timestamp=row[0],
            symbol=row[1],
            timeframe=row[2],
            open=row[3],
            high=row[4],
            low=row[5],
            close=row[6],
            volume=row[7],
            vwap=row[8],
            trade_count=row[9],
        )

    def get_symbols(self) -> list[str]:
        """Get list of all symbols in database."""
        query = "SELECT DISTINCT symbol FROM ohlcv_bars ORDER BY symbol"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [row[0] for row in cur.fetchall()]

    def get_date_range(
        self,
        symbol: str,
        timeframe: str = "15min",
    ) -> tuple[datetime, datetime]:
        """Get date range for a symbol."""
        query = """
        SELECT MIN(time), MAX(time)
        FROM ohlcv_bars
        WHERE symbol = %s AND timeframe = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (symbol, timeframe))
                row = cur.fetchone()

        return row[0], row[1]

    def execute_raw(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute raw SQL query and return DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def __enter__(self) -> "TimescaleClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class AsyncTimescaleClient:
    """
    Asynchronous TimescaleDB client for high-throughput operations.

    Uses asyncpg for non-blocking database operations.
    Ideal for real-time data ingestion and concurrent queries.
    """

    def __init__(
        self,
        config: ConnectionConfig,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for AsyncTimescaleClient. "
                "Install with: pip install asyncpg"
            )

        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize async connection pool."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            self.config.to_dsn(),
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            command_timeout=self.config.command_timeout,
        )
        logger.info(
            f"Async connected to TimescaleDB at {self.config.host}:{self.config.port}"
        )

    async def disconnect(self) -> None:
        """Close all connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Async disconnected from TimescaleDB")

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool."""
        if self._pool is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        async with self._pool.acquire() as conn:
            yield conn

    async def insert_ticks(
        self,
        ticks: Sequence[Tick],
        batch_size: int = 10000,
    ) -> int:
        """Async batch insert tick data."""
        if not ticks:
            return 0

        records = [
            (t.timestamp, t.symbol, t.price, t.size, t.exchange, t.conditions)
            for t in ticks
        ]

        total_inserted = 0

        async with self._get_connection() as conn:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                result = await conn.executemany(
                    """
                    INSERT INTO tick_data
                    (time, symbol, price, size, exchange, conditions)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    batch
                )
                total_inserted += len(batch)

        return total_inserted

    async def insert_ohlcv(
        self,
        bars: Sequence[Bar],
        batch_size: int = 10000,
    ) -> int:
        """Async batch insert OHLCV bars."""
        if not bars:
            return 0

        records = [
            (
                b.timestamp, b.symbol, b.timeframe,
                b.open, b.high, b.low, b.close,
                b.volume, b.vwap, b.trade_count
            )
            for b in bars
        ]

        total_inserted = 0

        async with self._get_connection() as conn:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                await conn.executemany(
                    """
                    INSERT INTO ohlcv_bars
                    (time, symbol, timeframe, open, high, low, close,
                     volume, vwap, trade_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT DO NOTHING
                    """,
                    batch
                )
                total_inserted += len(batch)

        return total_inserted

    async def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "15min",
    ) -> pd.DataFrame:
        """Async retrieve OHLCV bars."""
        query = """
        SELECT
            time as timestamp,
            open, high, low, close, volume, vwap, trade_count
        FROM ohlcv_bars
        WHERE symbol = $1
            AND timeframe = $2
            AND time >= $3
            AND time < $4
        ORDER BY time
        """

        async with self._get_connection() as conn:
            rows = await conn.fetch(query, symbol, timeframe, start, end)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'vwap', 'trade_count'
        ])
        df = df.set_index('timestamp')

        return df

    async def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Async retrieve tick data."""
        query = """
        SELECT
            time as timestamp,
            price, size, exchange, conditions
        FROM tick_data
        WHERE symbol = $1
            AND time >= $2
            AND time < $3
        ORDER BY time
        """

        async with self._get_connection() as conn:
            rows = await conn.fetch(query, symbol, start, end)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'timestamp', 'price', 'size', 'exchange', 'conditions'
        ])
        df = df.set_index('timestamp')

        return df

    async def __aenter__(self) -> "AsyncTimescaleClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


# Factory function
def create_timescale_client(
    host: str = "localhost",
    port: int = 5432,
    database: str = "alphatrade_db",
    user: str = "alphatrade",
    password: str = "",
    async_mode: bool = False,
) -> TimescaleClient | AsyncTimescaleClient:
    """
    Factory function to create TimescaleDB client.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        async_mode: If True, return AsyncTimescaleClient

    Returns:
        TimescaleClient or AsyncTimescaleClient instance
    """
    config = ConnectionConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )

    if async_mode:
        return AsyncTimescaleClient(config)
    return TimescaleClient(config)
