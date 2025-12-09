-- AlphaTrade TimescaleDB Initialization Script
-- Creates hypertables for time-series market data

-- Create MLflow database for experiment tracking
CREATE DATABASE mlflow;

-- Switch to market_data database
\c market_data;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ==========================================================================
-- MARKET DATA TABLES
-- ==========================================================================

-- OHLCV bars table
CREATE TABLE IF NOT EXISTS bars (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    vwap        DOUBLE PRECISION,
    trade_count INTEGER,
    timeframe   VARCHAR(10) DEFAULT '1min',
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('bars', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create index for fast symbol lookups
CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON bars (symbol, time DESC);

-- Quotes table
CREATE TABLE IF NOT EXISTS quotes (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    bid_price   DOUBLE PRECISION NOT NULL,
    bid_size    INTEGER NOT NULL,
    ask_price   DOUBLE PRECISION NOT NULL,
    ask_size    INTEGER NOT NULL,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('quotes', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes (symbol, time DESC);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    size        INTEGER NOT NULL,
    exchange    VARCHAR(10),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('trades', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ==========================================================================
-- FEATURE STORE TABLES
-- ==========================================================================

-- Pre-computed features table
CREATE TABLE IF NOT EXISTS features (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(10) NOT NULL,
    feature_name    VARCHAR(50) NOT NULL,
    feature_value   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol, feature_name)
);

SELECT create_hypertable('features', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_name ON features (symbol, feature_name, time DESC);

-- Feature vectors for ML inference (denormalized for fast reads)
CREATE TABLE IF NOT EXISTS feature_vectors (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    features    JSONB NOT NULL,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('feature_vectors', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ==========================================================================
-- TRADING TABLES
-- ==========================================================================

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id        VARCHAR(36) PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(10) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    order_type      VARCHAR(20) NOT NULL,
    limit_price     DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    status          VARCHAR(20) NOT NULL,
    filled_qty      DOUBLE PRECISION DEFAULT 0,
    avg_fill_price  DOUBLE PRECISION,
    strategy        VARCHAR(50),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_time ON orders (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);

-- Fills table
CREATE TABLE IF NOT EXISTS fills (
    fill_id         VARCHAR(36) PRIMARY KEY,
    order_id        VARCHAR(36) REFERENCES orders(order_id),
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(10) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION DEFAULT 0,
    pnl             DOUBLE PRECISION
);

SELECT create_hypertable('fills', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    symbol          VARCHAR(10) PRIMARY KEY,
    quantity        DOUBLE PRECISION NOT NULL,
    avg_entry_price DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    side            VARCHAR(10),
    opened_at       TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ==========================================================================
-- RISK & PERFORMANCE TABLES
-- ==========================================================================

-- Daily PnL tracking
CREATE TABLE IF NOT EXISTS daily_pnl (
    date            DATE PRIMARY KEY,
    starting_equity DOUBLE PRECISION NOT NULL,
    ending_equity   DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION NOT NULL,
    total_pnl       DOUBLE PRECISION NOT NULL,
    max_drawdown    DOUBLE PRECISION,
    trade_count     INTEGER DEFAULT 0,
    win_count       INTEGER DEFAULT 0,
    loss_count      INTEGER DEFAULT 0
);

-- Equity curve
CREATE TABLE IF NOT EXISTS equity_curve (
    time            TIMESTAMPTZ NOT NULL,
    equity          DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    margin_used     DOUBLE PRECISION DEFAULT 0,
    drawdown        DOUBLE PRECISION DEFAULT 0,
    PRIMARY KEY (time)
);

SELECT create_hypertable('equity_curve', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ==========================================================================
-- CONTINUOUS AGGREGATES (for fast historical queries)
-- ==========================================================================

-- 5-minute bars aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    sum(volume * close) / NULLIF(sum(volume), 0) AS vwap
FROM bars
GROUP BY time_bucket('5 minutes', time), symbol
WITH NO DATA;

-- Refresh policy for 5-min bars
SELECT add_continuous_aggregate_policy('bars_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- Hourly bars aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    sum(volume * close) / NULLIF(sum(volume), 0) AS vwap
FROM bars
GROUP BY time_bucket('1 hour', time), symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('bars_1h',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily bars aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    sum(volume * close) / NULLIF(sum(volume), 0) AS vwap
FROM bars
GROUP BY time_bucket('1 day', time), symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('bars_daily',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ==========================================================================
-- DATA RETENTION POLICIES
-- ==========================================================================

-- Keep raw 1-minute data for 30 days
SELECT add_retention_policy('bars', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep quotes for 7 days
SELECT add_retention_policy('quotes', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep trades for 7 days
SELECT add_retention_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep features for 90 days
SELECT add_retention_policy('features', INTERVAL '90 days', if_not_exists => TRUE);

-- ==========================================================================
-- HELPER FUNCTIONS
-- ==========================================================================

-- Function to get latest price for a symbol
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol VARCHAR)
RETURNS DOUBLE PRECISION AS $$
    SELECT close
    FROM bars
    WHERE symbol = p_symbol
    ORDER BY time DESC
    LIMIT 1;
$$ LANGUAGE SQL;

-- Function to get latest N bars for a symbol
CREATE OR REPLACE FUNCTION get_latest_bars(p_symbol VARCHAR, p_count INTEGER)
RETURNS TABLE (
    time TIMESTAMPTZ,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT
) AS $$
    SELECT time, open, high, low, close, volume
    FROM bars
    WHERE symbol = p_symbol
    ORDER BY time DESC
    LIMIT p_count;
$$ LANGUAGE SQL;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alphatrade;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alphatrade;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'AlphaTrade database initialized successfully';
END $$;
