-- AlphaTrade System - Database Initialization Script
-- PostgreSQL with TimescaleDB extension

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- OHLCV Data Tables
-- ==============================================================================

-- Raw OHLCV data (hypertable)
CREATE TABLE IF NOT EXISTS ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    vwap DOUBLE PRECISION,
    trade_count INTEGER,
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('ohlcv_data', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_time ON ohlcv_data (time DESC);

-- ==============================================================================
-- Trade Tables
-- ==============================================================================

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_order_id VARCHAR(50),
    broker_order_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    limit_price DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,
    time_in_force VARCHAR(10),
    status VARCHAR(20) NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price DOUBLE PRECISION,
    strategy_name VARCHAR(50),
    signal_strength DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders (created_at DESC);

-- Fills/Executions
CREATE TABLE IF NOT EXISTS fills (
    fill_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES orders(order_id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0,
    slippage_bps DOUBLE PRECISION DEFAULT 0,
    executed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fills_order ON fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills (symbol);
CREATE INDEX IF NOT EXISTS idx_fills_time ON fills (executed_at DESC);

-- ==============================================================================
-- Position Tables
-- ==============================================================================

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    position_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL UNIQUE,
    quantity INTEGER NOT NULL,
    avg_cost DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION,
    market_value DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION DEFAULT 0,
    sector VARCHAR(50),
    entry_date TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol);

-- Position history (hypertable)
CREATE TABLE IF NOT EXISTS position_history (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER,
    market_value DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('position_history', 'time', if_not_exists => TRUE);

-- ==============================================================================
-- Portfolio Tables
-- ==============================================================================

-- Portfolio snapshots (hypertable)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    total_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    drawdown DOUBLE PRECISION,
    num_positions INTEGER
);

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- ==============================================================================
-- Risk Tables
-- ==============================================================================

-- Risk metrics (hypertable)
CREATE TABLE IF NOT EXISTS risk_metrics (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    portfolio_value DOUBLE PRECISION,
    var_95 DOUBLE PRECISION,
    var_99 DOUBLE PRECISION,
    cvar_95 DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    current_drawdown DOUBLE PRECISION,
    gross_exposure DOUBLE PRECISION,
    net_exposure DOUBLE PRECISION,
    leverage DOUBLE PRECISION
);

SELECT create_hypertable('risk_metrics', 'time', if_not_exists => TRUE);

-- ==============================================================================
-- Signal Tables
-- ==============================================================================

-- Trading signals (hypertable)
CREATE TABLE IF NOT EXISTS signals (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    signal_type VARCHAR(20),
    strength DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    metadata JSONB,
    PRIMARY KEY (time, symbol, strategy_name)
);

SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, time DESC);

-- ==============================================================================
-- Feature Tables
-- ==============================================================================

-- Computed features (hypertable)
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, feature_name)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol, time DESC);

-- ==============================================================================
-- Model Tables
-- ==============================================================================

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50),
    description TEXT,
    hyperparameters JSONB,
    metrics JSONB,
    file_path VARCHAR(500),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (name, version)
);

-- Model predictions (hypertable)
CREATE TABLE IF NOT EXISTS model_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_id UUID REFERENCES models(model_id),
    prediction DOUBLE PRECISION,
    probability DOUBLE PRECISION,
    actual DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, model_id)
);

SELECT create_hypertable('model_predictions', 'time', if_not_exists => TRUE);

-- ==============================================================================
-- Audit Tables
-- ==============================================================================

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20),
    message TEXT,
    details JSONB,
    user_id VARCHAR(50),
    ip_address VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log (event_type);

-- ==============================================================================
-- Compression Policies (for older data)
-- ==============================================================================

-- Compress data older than 7 days
SELECT add_compression_policy('ohlcv_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('portfolio_snapshots', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('risk_metrics', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('signals', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('features', INTERVAL '3 days', if_not_exists => TRUE);

-- ==============================================================================
-- Retention Policies (optional - delete very old data)
-- ==============================================================================

-- Keep OHLCV data for 2 years
-- SELECT add_retention_policy('ohlcv_data', INTERVAL '2 years');

-- Keep features for 6 months
-- SELECT add_retention_policy('features', INTERVAL '6 months');

-- ==============================================================================
-- Views
-- ==============================================================================

-- Latest positions view
CREATE OR REPLACE VIEW v_latest_positions AS
SELECT
    p.*,
    o.close as latest_price
FROM positions p
LEFT JOIN LATERAL (
    SELECT close
    FROM ohlcv_data
    WHERE symbol = p.symbol
    ORDER BY time DESC
    LIMIT 1
) o ON true;

-- Daily P&L view
CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT
    time_bucket('1 day', time) as date,
    total_value,
    pnl,
    drawdown
FROM portfolio_snapshots
ORDER BY date DESC;

-- ==============================================================================
-- Functions
-- ==============================================================================

-- Function to update position from fills
CREATE OR REPLACE FUNCTION update_position_from_fill()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO positions (symbol, quantity, avg_cost, sector, entry_date)
    VALUES (NEW.symbol, NEW.quantity, NEW.price, 'Unknown', NOW())
    ON CONFLICT (symbol) DO UPDATE SET
        quantity = positions.quantity + NEW.quantity * CASE WHEN NEW.side = 'buy' THEN 1 ELSE -1 END,
        avg_cost = CASE
            WHEN NEW.side = 'buy' THEN
                (positions.quantity * positions.avg_cost + NEW.quantity * NEW.price) /
                (positions.quantity + NEW.quantity)
            ELSE positions.avg_cost
        END,
        updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for position updates
DROP TRIGGER IF EXISTS trg_update_position ON fills;
CREATE TRIGGER trg_update_position
    AFTER INSERT ON fills
    FOR EACH ROW
    EXECUTE FUNCTION update_position_from_fill();

-- ==============================================================================
-- Initial Data
-- ==============================================================================

-- Insert system configuration
INSERT INTO audit_log (event_type, severity, message, details)
VALUES ('SYSTEM', 'INFO', 'Database initialized', '{"version": "1.0.0"}');

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO trading_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_app;
