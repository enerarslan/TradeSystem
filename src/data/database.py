"""
Institutional-Grade Database Layer
JPMorgan-Level Data Storage and Retrieval

Features:
- PostgreSQL/TimescaleDB integration
- Redis caching layer
- InfluxDB for metrics
- Connection pooling
- Async operations support
"""

import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logger import get_logger, get_audit_logger
from ..utils.helpers import retry_with_backoff, Result


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class DBConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "alphatrade"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


class DatabaseManager:
    """
    PostgreSQL/TimescaleDB database manager.

    Features:
    - Connection pooling
    - OHLCV data storage
    - Trade history
    - Portfolio snapshots
    """

    def __init__(self, config: Optional[DBConfig] = None):
        """
        Initialize DatabaseManager.

        Args:
            config: Database configuration
        """
        self.config = config or DBConfig()
        self._engine = None
        self._pool = None
        self._initialized = False

        logger.info(f"DatabaseManager configured for {self.config.host}:{self.config.port}")

    def initialize(self) -> bool:
        """Initialize database connection and create tables"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool

            connection_string = (
                f"postgresql://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )

            self._engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True  # Check connection health
            )

            # Test connection
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")

            # Create tables
            self._create_tables()

            self._initialized = True
            logger.info("Database initialized successfully")
            return True

        except ImportError:
            logger.warning("SQLAlchemy not installed, database features unavailable")
            return False
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False

    def _create_tables(self) -> None:
        """Create required database tables"""
        create_statements = [
            # OHLCV data table (TimescaleDB hypertable)
            """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                PRIMARY KEY (timestamp, symbol)
            );
            """,

            # Trade history table
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR(50) PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                fill_price DOUBLE PRECISION,
                commission DOUBLE PRECISION,
                slippage DOUBLE PRECISION,
                strategy VARCHAR(50),
                signal_strength DOUBLE PRECISION,
                pnl DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,

            # Portfolio snapshots
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                snapshot_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                total_value DOUBLE PRECISION NOT NULL,
                cash DOUBLE PRECISION NOT NULL,
                positions_value DOUBLE PRECISION NOT NULL,
                daily_pnl DOUBLE PRECISION,
                total_pnl DOUBLE PRECISION,
                drawdown DOUBLE PRECISION,
                positions JSONB
            );
            """,

            # Signal history
            """
            CREATE TABLE IF NOT EXISTS signals (
                signal_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                strategy VARCHAR(50) NOT NULL,
                direction INTEGER NOT NULL,
                strength DOUBLE PRECISION,
                features JSONB,
                executed BOOLEAN DEFAULT FALSE
            );
            """,

            # Risk events
            """
            CREATE TABLE IF NOT EXISTS risk_events (
                event_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                description TEXT,
                details JSONB,
                resolved BOOLEAN DEFAULT FALSE
            );
            """,

            # Model performance tracking
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                metric_name VARCHAR(50) NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                metadata JSONB
            );
            """
        ]

        with self._engine.connect() as conn:
            for statement in create_statements:
                try:
                    conn.execute(statement)
                except Exception as e:
                    logger.warning(f"Table creation warning: {e}")

            conn.commit()

        logger.info("Database tables created/verified")

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")

        conn = self._engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def store_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        upsert: bool = True
    ) -> int:
        """
        Store OHLCV data to database.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            upsert: Update existing records

        Returns:
            Number of rows stored
        """
        if not self._initialized:
            logger.warning("Database not initialized, skipping store")
            return 0

        df = df.copy()
        df['symbol'] = symbol

        # Reset index to include timestamp as column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'timestamp'})

        rows_stored = 0

        with self.get_connection() as conn:
            if upsert:
                # Use PostgreSQL ON CONFLICT for upsert
                for _, row in df.iterrows():
                    conn.execute(
                        """
                        INSERT INTO ohlcv_data (timestamp, symbol, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp, symbol)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                        """,
                        (row['timestamp'], row['symbol'], row['open'],
                         row['high'], row['low'], row['close'], row['volume'])
                    )
                    rows_stored += 1
            else:
                # Bulk insert (faster but no upsert)
                df.to_sql(
                    'ohlcv_data',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                rows_stored = len(df)

            conn.commit()

        logger.debug(f"Stored {rows_stored} rows for {symbol}")
        return rows_stored

    def load_ohlcv(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            OHLCV DataFrame
        """
        if not self._initialized:
            logger.warning("Database not initialized")
            return pd.DataFrame()

        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data WHERE symbol = %s"
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            df = pd.read_sql(query, conn, params=params, parse_dates=['timestamp'])

        if not df.empty:
            df = df.set_index('timestamp')

        return df

    def store_trade(self, trade: Dict[str, Any]) -> bool:
        """Store trade record"""
        if not self._initialized:
            return False

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO trades
                (trade_id, timestamp, symbol, side, quantity, price,
                 fill_price, commission, slippage, strategy, signal_strength, pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (trade.get('trade_id'), trade.get('timestamp'), trade.get('symbol'),
                 trade.get('side'), trade.get('quantity'), trade.get('price'),
                 trade.get('fill_price'), trade.get('commission'), trade.get('slippage'),
                 trade.get('strategy'), trade.get('signal_strength'), trade.get('pnl'))
            )
            conn.commit()

        audit_logger.log_trade(trade)
        return True

    def store_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Store portfolio snapshot"""
        if not self._initialized:
            return False

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_snapshots
                (timestamp, total_value, cash, positions_value,
                 daily_pnl, total_pnl, drawdown, positions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (snapshot.get('timestamp'), snapshot.get('total_value'),
                 snapshot.get('cash'), snapshot.get('positions_value'),
                 snapshot.get('daily_pnl'), snapshot.get('total_pnl'),
                 snapshot.get('drawdown'), json.dumps(snapshot.get('positions', {})))
            )
            conn.commit()

        return True

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Get trade history"""
        if not self._initialized:
            return pd.DataFrame()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        if strategy:
            query += " AND strategy = %s"
            params.append(strategy)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params, parse_dates=['timestamp'])


class TimeSeriesDB:
    """
    TimescaleDB-specific optimizations for time series data.

    Features:
    - Hypertable creation
    - Compression policies
    - Continuous aggregates
    - Retention policies
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_hypertable(self, table_name: str, time_column: str = 'timestamp') -> bool:
        """Convert table to TimescaleDB hypertable"""
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    f"SELECT create_hypertable('{table_name}', '{time_column}', "
                    f"if_not_exists => TRUE, migrate_data => TRUE);"
                )
                conn.commit()
            logger.info(f"Created hypertable for {table_name}")
            return True
        except Exception as e:
            logger.warning(f"Hypertable creation failed (TimescaleDB may not be installed): {e}")
            return False

    def enable_compression(
        self,
        table_name: str,
        compress_after: str = '7 days'
    ) -> bool:
        """Enable compression on hypertable"""
        try:
            with self.db.get_connection() as conn:
                # Enable compression
                conn.execute(
                    f"ALTER TABLE {table_name} SET (timescaledb.compress);"
                )

                # Add compression policy
                conn.execute(
                    f"SELECT add_compression_policy('{table_name}', "
                    f"INTERVAL '{compress_after}', if_not_exists => TRUE);"
                )
                conn.commit()
            logger.info(f"Enabled compression for {table_name}")
            return True
        except Exception as e:
            logger.warning(f"Compression setup failed: {e}")
            return False

    def create_continuous_aggregate(
        self,
        name: str,
        source_table: str,
        interval: str = '1 hour'
    ) -> bool:
        """Create continuous aggregate for faster queries"""
        try:
            with self.db.get_connection() as conn:
                conn.execute(f"""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS {name}
                    WITH (timescaledb.continuous) AS
                    SELECT
                        time_bucket('{interval}', timestamp) AS bucket,
                        symbol,
                        FIRST(open, timestamp) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        LAST(close, timestamp) AS close,
                        SUM(volume) AS volume
                    FROM {source_table}
                    GROUP BY bucket, symbol
                    WITH NO DATA;
                """)
                conn.commit()
            logger.info(f"Created continuous aggregate {name}")
            return True
        except Exception as e:
            logger.warning(f"Continuous aggregate creation failed: {e}")
            return False


class RedisCache:
    """
    Redis caching layer for high-performance data access.

    Features:
    - Latest price caching
    - Feature vector caching
    - Signal caching
    - Session state management
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._client = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            import redis

            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            self._client.ping()

            self._initialized = True
            logger.info(f"Redis connected: {self.host}:{self.port}")
            return True

        except ImportError:
            logger.warning("redis package not installed")
            return False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return False

    def set_price(self, symbol: str, price_data: Dict[str, float], ttl: int = 60) -> bool:
        """Cache latest price data"""
        if not self._initialized:
            return False

        key = f"price:{symbol}"
        self._client.hset(key, mapping=price_data)
        self._client.expire(key, ttl)
        return True

    def get_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get cached price data"""
        if not self._initialized:
            return None

        key = f"price:{symbol}"
        data = self._client.hgetall(key)

        if data:
            return {k: float(v) for k, v in data.items()}
        return None

    def set_features(
        self,
        symbol: str,
        features: Dict[str, float],
        ttl: int = 300
    ) -> bool:
        """Cache computed features"""
        if not self._initialized:
            return False

        key = f"features:{symbol}"
        self._client.set(key, json.dumps(features), ex=ttl)
        return True

    def get_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get cached features"""
        if not self._initialized:
            return None

        key = f"features:{symbol}"
        data = self._client.get(key)

        if data:
            return json.loads(data)
        return None

    def publish_signal(self, channel: str, signal: Dict[str, Any]) -> int:
        """Publish trading signal to channel"""
        if not self._initialized:
            return 0

        return self._client.publish(channel, json.dumps(signal))

    def subscribe_signals(self, channel: str):
        """Subscribe to trading signals"""
        if not self._initialized:
            return None

        pubsub = self._client.pubsub()
        pubsub.subscribe(channel)
        return pubsub

    def set_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Cache position data"""
        if not self._initialized:
            return False

        key = f"position:{symbol}"
        self._client.set(key, json.dumps(position))
        return True

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached positions"""
        if not self._initialized:
            return {}

        positions = {}
        for key in self._client.scan_iter("position:*"):
            symbol = key.split(":")[1]
            data = self._client.get(key)
            if data:
                positions[symbol] = json.loads(data)

        return positions

    def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment counter (for rate limiting, etc.)"""
        if not self._initialized:
            return 0

        return self._client.incr(key, amount)

    def get_counter(self, key: str) -> int:
        """Get counter value"""
        if not self._initialized:
            return 0

        value = self._client.get(key)
        return int(value) if value else 0
