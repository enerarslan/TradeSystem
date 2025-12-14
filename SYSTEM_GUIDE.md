# AlphaTrade System - Complete Guide
## Institutional-Grade Algorithmic Trading System

This document is prepared for developers and AI agents who want to understand and run the AlphaTrade system from scratch.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Backtesting](#backtesting)
7. [Live Trading](#live-trading)
8. [Risk Management](#risk-management)
9. [Monitoring](#monitoring)
10. [Troubleshooting](#troubleshooting)
11. [AI Agent Integration](#ai-agent-integration)

---

## System Overview

AlphaTrade is an **institutional-grade** algorithmic trading system. Core features:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AlphaTrade System                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │   Data   │ → │ Features │ → │  Model   │ → │ Strategy │     │
│  │ Pipeline │   │  Engine  │   │ Training │   │  Engine  │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       ↓              ↓              ↓              ↓            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ Backtest │   │   Risk   │   │Execution │   │Monitoring│     │
│  │  Engine  │   │ Manager  │   │  Engine  │   │Dashboard │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description | Main Files |
|-----------|-------------|------------|
| **Data Pipeline** | Fetch data from Alpaca API, cleaning | `src/data/` |
| **Feature Engine** | AFML-based feature generation | `src/features/` |
| **Model Training** | CatBoost/XGBoost ensemble | `src/models/` |
| **Backtest Engine** | Realistic simulation | `src/backtest/` |
| **Risk Manager** | VaR, position limits | `src/risk/` |
| **Execution Engine** | Alpaca order management | `src/execution/` |
| **Strategy Engine** | Signal generation, decisions | `src/strategy/` |

### Technology Stack

- **Language**: Python 3.9+
- **Broker**: Alpaca Markets API
- **ML Models**: CatBoost, XGBoost, LightGBM
- **State Management**: Redis (optional, with in-memory fallback)
- **Async Framework**: asyncio, aiohttp
- **Feature Engineering**: AFML (Advances in Financial Machine Learning)

---

## Architecture

### Directory Structure

```
AlphaTrade_System/
├── config/                     # Configuration files
│   ├── symbols.yaml           # Trading symbols
│   ├── risk_params.yaml       # Risk parameters
│   └── triple_barrier_params.yaml
│
├── src/                       # Main source code
│   ├── data/                  # Data management
│   │   ├── alpaca_client.py  # Alpaca API client
│   │   ├── live_feed.py      # WebSocket live data
│   │   ├── live_validator.py # Data validation ✨NEW
│   │   └── websocket_hardened.py # Hardened WebSocket ✨NEW
│   │
│   ├── features/             # Feature engineering
│   │   ├── institutional.py  # AFML features (HMM, VPIN)
│   │   ├── microstructure.py # Market microstructure
│   │   └── point_in_time.py  # Look-ahead bias protected ✨NEW
│   │
│   ├── models/               # ML models
│   │   ├── ml_model.py       # XGBoost, CatBoost, LightGBM
│   │   ├── ensemble.py       # Model ensemble
│   │   ├── meta_labeling.py  # Meta-labeling
│   │   └── calibration.py    # Probability calibration ✨NEW
│   │
│   ├── backtest/             # Backtesting
│   │   ├── engine.py         # Main backtest engine
│   │   └── realistic_fills.py # Realistic fill model ✨NEW
│   │
│   ├── risk/                 # Risk management
│   │   ├── risk_manager.py   # Main risk manager
│   │   ├── position_sizer.py # Position sizing
│   │   ├── bayesian_kelly.py # Bayesian Kelly ✨NEW
│   │   └── correlation_breaker.py # Correlation circuit breaker ✨NEW
│   │
│   ├── execution/            # Order execution
│   │   ├── broker_api.py     # Alpaca broker API
│   │   ├── order_manager.py  # Order lifecycle
│   │   ├── protected_positions.py # Bracket orders ✨NEW
│   │   ├── reconciliation.py # State reconciliation ✨NEW
│   │   ├── rejection_handler.py # Rejection handling ✨NEW
│   │   └── impact_model.py   # Pre-trade impact ✨NEW
│   │
│   ├── strategy/             # Trading strategies
│   │   ├── base_strategy.py  # Base strategy class
│   │   └── ml_strategy.py    # ML-based strategy
│   │
│   ├── core/                 # Core systems
│   │   ├── graceful_degradation.py # Graceful degradation ✨NEW
│   │   ├── health_check.py   # Health checks ✨NEW
│   │   └── state_manager.py  # Redis state ✨NEW
│   │
│   ├── monitoring/           # Monitoring ✨NEW
│   │   └── execution_dashboard.py # Execution quality
│   │
│   ├── analytics/            # Analytics ✨NEW
│   │   └── attribution.py    # P&L attribution
│   │
│   ├── mlops/                # MLOps
│   │   ├── monitoring.py     # Drift detection
│   │   └── staleness.py      # Model staleness ✨NEW
│   │
│   └── utils/                # Utilities
│       ├── logger.py         # Logging
│       └── metrics.py        # Performance metrics
│
├── scripts/                   # Executable scripts
│   ├── calibrate_triple_barrier.py
│   ├── run_backtest.py
│   └── run_pre_training_validation.py
│
├── main.py                    # Main orchestrator
├── ARCHITECTURAL_REVIEW_REPORT.md  # Architecture review
└── SYSTEM_GUIDE.md            # This document
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Alpaca API ──→ Raw OHLCV ──→ Validation ──→ Feature Engine         │
│       │              │             │              │                  │
│       ▼              ▼             ▼              ▼                  │
│  Historical    Price/Volume   Quality Gate   AFML Features          │
│  + Real-time     Data          (staleness,    (VPIN, HMM,           │
│                                 spikes)        FracDiff)            │
│                                                    │                 │
│                                                    ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    ML MODEL PIPELINE                         │    │
│  │  Features → Calibrated Model → Probability → Signal         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                    │                 │
│                                                    ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    EXECUTION PIPELINE                        │    │
│  │  Signal → Risk Check → Impact Est. → Order → Broker         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Step 1: Requirements

```bash
# Python 3.9+ required
python --version

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

# Core dependencies:
# - alpaca-trade-api    (Broker API)
# - pandas, numpy       (Data processing)
# - catboost, xgboost   (ML models)
# - scikit-learn        (ML utilities)
# - aiohttp, websockets (Async networking)
# - redis               (State management - optional)
# - scipy               (Statistical functions)
```

### Step 3: Configuration

#### `config/symbols.yaml` - Trading Symbols
```yaml
symbols:
  - AAPL
  - MSFT
  - NVDA
  - GOOGL
  - AMZN
  - META
  - TSLA
```

#### `.env` - API Keys
```bash
# Alpaca API (for Paper Trading)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Redis (Optional - recommended for production)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Step 4: Alpaca Account Setup

1. Create account at [alpaca.markets](https://alpaca.markets)
2. Get paper trading API keys from dashboard
3. Add keys to `.env` file
4. Verify connection:

```python
from src.data.alpaca_client import AlpacaDataClient

client = AlpacaDataClient()
account = await client.get_account()
print(f"Account Status: {account.status}")
print(f"Buying Power: ${account.buying_power}")
```

---

## Data Pipeline

### Overview

```
Alpaca API → Raw Data → Validation → Features → Model Input
     ↓           ↓          ↓           ↓           ↓
  Historical   OHLCV     Quality     AFML      Predictions
    + Live    + Volume    Check    Features
```

### Step 1: Fetch Historical Data

```python
from src.data.alpaca_client import AlpacaDataClient

# Create client
client = AlpacaDataClient()

# Fetch data (last 2 years, 15-minute bars)
df = await client.get_historical_bars(
    symbol='AAPL',
    timeframe='15Min',
    start='2023-01-01',
    end='2024-12-01'
)

# Output: DataFrame with columns
# [timestamp, open, high, low, close, volume, vwap, trade_count]
print(df.head())
```

### Step 2: Data Validation

```python
from src.data.live_validator import LiveDataValidator

validator = LiveDataValidator(
    max_price_change_pct=0.10,  # 10% max change
    min_volume=1000,
    max_staleness_seconds=60
)

# Validate incoming data
result = validator.validate(df)

if not result.is_valid:
    print(f"Validation failed: {result.issues}")
    # Issues could be: price spike, stale data, volume anomaly
else:
    print("Data validation passed")
```

### Step 3: Feature Engineering

```python
from src.features.institutional import InstitutionalFeatureEngine

engine = InstitutionalFeatureEngine()

# Generate AFML features
features = engine.generate_all_features(df)

# Generated features include:
# - Fractional Differentiation (for stationarity)
# - VPIN (Volume-Synchronized Probability of Informed Trading)
# - Kyle's Lambda (market impact coefficient)
# - HMM Regime (Hidden Markov Model states)
# - Microstructure features (spread, depth, etc.)

print(f"Generated {len(features.columns)} features")
print(features.columns.tolist())
```

### Point-in-Time Feature Engine (No Look-Ahead Bias)

```python
from src.features.point_in_time import PointInTimeFeatureEngine

# This engine guarantees no future data leakage
pit_engine = PointInTimeFeatureEngine()

# Add custom features
pit_engine.add_feature(
    name='sma_20',
    calculation=lambda df: df['close'].rolling(20).mean(),
    lookback=20
)

pit_engine.add_feature(
    name='volatility',
    calculation=lambda df: df['close'].pct_change().rolling(20).std(),
    lookback=20
)

# Compute with automatic validation
features = pit_engine.compute_all(df, validate=True)
# Raises error if look-ahead bias detected
```

---

## Model Training

### Training Pipeline

```
Raw Data → Labels → Features → Train/Val Split → Model Training → Evaluation
              ↓
      Triple Barrier Method
      (profit target, stop loss, max holding period)
```

### Step 1: Triple Barrier Labeling

```bash
# Calibrate barrier parameters per symbol
python scripts/calibrate_triple_barrier.py --calibrate

# Output: config/triple_barrier_params.yaml
```

```python
# Programmatic usage
from src.models.meta_labeling import TripleBarrierLabeler

labeler = TripleBarrierLabeler(
    profit_taking=0.02,      # 2% take profit
    stop_loss=0.01,          # 1% stop loss
    max_holding_periods=20   # Max 20 bars holding
)

labels = labeler.fit_transform(df)
# Labels: 0 (loss), 1 (neutral), 2 (profit)

print(f"Label distribution: {labels.value_counts()}")
```

### Step 2: Model Training

```python
from src.models.ml_model import CatBoostModel
from src.models.ensemble import EnsembleModel
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, shuffle=False  # Time series - no shuffle!
)

# Single model training
model = CatBoostModel(
    task='classification',
    iterations=500,
    learning_rate=0.05,
    depth=6
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    early_stopping_rounds=50
)

# Ensemble model (recommended)
ensemble = EnsembleModel(
    models=['catboost', 'xgboost', 'lightgbm'],
    voting='soft'  # Probability averaging
)
ensemble.fit(X_train, y_train)

# Evaluate
predictions = ensemble.predict(X_val)
accuracy = (predictions == y_val).mean()
print(f"Validation Accuracy: {accuracy:.2%}")
```

### Step 3: Probability Calibration

ML models often output miscalibrated probabilities. A 70% prediction doesn't mean 70% actual win rate.

```python
from src.models.calibration import ProbabilityCalibrationManager

calibrator = ProbabilityCalibrationManager(
    auto_select=True  # Automatically select best method
)

# Fit on validation set
val_probs = model.predict_proba(X_val)[:, 2]  # Probability of class 2
calibrator.fit(val_probs, (y_val == 2).astype(int))

# Use in production
raw_prob = model.predict_proba(X_new)[:, 2]
calibrated_prob = calibrator.calibrate(raw_prob)

# Check calibration quality
metrics = calibrator.get_metrics()
print(f"ECE (Expected Calibration Error): {metrics.expected_calibration_error:.4f}")
print(f"Brier Score: {metrics.brier_score:.4f}")
```

### Available Calibration Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Isotonic | Non-parametric, monotonic | Large datasets |
| Platt | Sigmoid transformation | Small datasets |
| Temperature | Single parameter scaling | Neural networks |
| Beta | Flexible parametric | General purpose |

---

## Backtesting

### Running a Backtest

```bash
# Simple backtest
python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-01

# With detailed parameters
python scripts/run_backtest.py \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --initial-capital 100000 \
    --symbols AAPL,MSFT,NVDA \
    --strategy ml_momentum
```

### Programmatic Usage

```python
from src.backtest.engine import BacktestEngine
from src.backtest.realistic_fills import RealisticFillSimulator

# Realistic fill model (accounts for market impact)
fill_simulator = RealisticFillSimulator(
    slippage_model='sqrt_impact',  # Square-root market impact
    commission_per_share=0.005,
    min_fill_rate=0.85
)

# Initialize backtest engine
engine = BacktestEngine(
    initial_capital=100000,
    fill_simulator=fill_simulator
)

# Run backtest
results = engine.run(
    strategy=my_strategy,
    data=historical_data,
    start_date='2024-01-01',
    end_date='2024-06-01'
)

# Print results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Profit Factor: {results.profit_factor:.2f}")
```

### Realistic Fill Model

The system uses a square-root market impact model (Kyle's Lambda):

```python
# Impact = sigma * sqrt(Q/V) * coefficient
# Where:
#   sigma = volatility
#   Q = order quantity
#   V = average daily volume

from src.backtest.realistic_fills import RealisticFillSimulator

simulator = RealisticFillSimulator(
    permanent_impact_coef=0.1,   # Information leakage
    temporary_impact_coef=0.2,  # Liquidity consumption
    adverse_selection_coef=0.5  # Worse fills when model is right
)

fill = simulator.simulate_fill(
    symbol='AAPL',
    side='buy',
    quantity=1000,
    signal_price=175.00,
    current_bar=bar_data
)

print(f"Fill Price: ${fill.fill_price:.2f}")
print(f"Slippage: {fill.slippage_bps:.1f} bps")
print(f"Market Impact: {fill.market_impact_bps:.1f} bps")
```

### Key Backtest Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted return | > 1.5 |
| Sortino Ratio | Downside risk-adjusted | > 2.0 |
| Max Drawdown | Largest peak-to-trough | < 15% |
| Win Rate | Winning trade percentage | > 52% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |
| Calmar Ratio | Annual return / Max DD | > 1.0 |

---

## Live Trading

### Starting the System

```bash
# Paper trading mode (recommended for testing)
python main.py --mode paper

# Live trading (CAUTION - real money!)
python main.py --mode live
```

### Main Loop Structure

```python
# Simplified main.py structure

async def main():
    # 1. Initialize system components
    await initialize_system()

    # 2. Recover state after crash/restart
    recovered = await state_manager.recover_state()
    if recovered['previous_session']:
        logger.info(f"Recovered from session: {recovered['previous_session'].session_id}")

    # 3. Start new session
    await state_manager.start_session(session_id=generate_session_id())

    # 4. Main trading loop
    while market_is_open():
        try:
            # Get latest market data
            bars = await data_feed.get_latest_bars()

            # Validate data quality
            if not validator.validate(bars).is_valid:
                logger.warning("Data validation failed, skipping")
                continue

            # Generate features
            features = feature_engine.generate(bars)

            # Get model predictions
            predictions = model.predict(features)
            calibrated_probs = calibrator.calibrate(predictions)

            # Check risk limits
            risk_check = risk_manager.check_all_limits()
            if not risk_check.passed:
                logger.warning(f"Risk limit hit: {risk_check.reason}")
                continue

            # Generate trading signals
            signals = strategy.generate_signals(calibrated_probs)

            # Execute orders
            for signal in signals:
                # Pre-trade impact estimation
                impact = impact_model.estimate_impact(signal)

                if impact.should_execute:
                    await order_manager.submit(signal)

            # Update state
            await state_manager.heartbeat()

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await graceful_degradation.handle_error(e)

        # Wait for next bar
        await asyncio.sleep(bar_interval)

    # 5. End session
    await state_manager.end_session()
```

### Order Management with Protection

```python
from src.execution.protected_positions import ProtectedPositionManager

# Create position manager
position_manager = ProtectedPositionManager(broker_api)

# Open position with bracket order (broker-side SL/TP)
position, success = await position_manager.open_position_with_protection(
    symbol='AAPL',
    side='buy',
    quantity=100,
    stop_loss_pct=0.02,     # 2% stop loss
    take_profit_pct=0.04,   # 4% take profit
    use_bracket_order=True  # Server-side protection
)

if success:
    print(f"Position opened: {position.position_id}")
    print(f"Stop Loss Order: {position.stop_loss_order_id}")
    print(f"Take Profit Order: {position.take_profit_order_id}")
else:
    print(f"Failed to open position: {position.error}")
```

### State Reconciliation

Ensures local state matches broker state:

```python
from src.execution.reconciliation import ReconciliationEngine

reconciler = ReconciliationEngine(
    broker_api=broker_api,
    local_state=state_manager,
    reconcile_interval=30  # Every 30 seconds
)

# Manual reconciliation
report = await reconciler.reconcile()

if report.has_discrepancies:
    print(f"Found {len(report.discrepancies)} discrepancies:")
    for disc in report.discrepancies:
        print(f"  - {disc.type}: {disc.details}")

    # Auto-resolve
    await reconciler.resolve_discrepancies(report)
```

### Graceful Degradation

System continues operating with reduced functionality when components fail:

```python
from src.core.graceful_degradation import GracefulDegradationManager

degradation_manager = GracefulDegradationManager()

# Register components
degradation_manager.register_component(
    name='data_feed',
    component_type=ComponentType.DATA,
    health_check=data_feed.health_check,
    fallback_config=FallbackConfig(
        fallback_action=FallbackAction.USE_CACHED,
        max_cache_age_seconds=300
    )
)

# On failure, system automatically:
# 1. Uses cached data (up to 5 minutes old)
# 2. Reduces position sizes
# 3. Halts new positions if critical
```

---

## Risk Management

### Risk Layer Architecture

```
┌─────────────────────────────────────────────────┐
│              Risk Management Stack               │
├─────────────────────────────────────────────────┤
│  Layer 1: Pre-Trade Checks                      │
│  ├── Position size limits                       │
│  ├── Sector/symbol exposure limits              │
│  └── Pre-trade impact estimation                │
├─────────────────────────────────────────────────┤
│  Layer 2: Execution Protection                  │
│  ├── Bracket orders (SL/TP)                     │
│  ├── Order rejection handling                   │
│  └── Rate limiting                              │
├─────────────────────────────────────────────────┤
│  Layer 3: Portfolio Risk                        │
│  ├── VaR (Value at Risk) limits                 │
│  ├── Correlation circuit breaker                │
│  └── Drawdown limits                            │
├─────────────────────────────────────────────────┤
│  Layer 4: System Risk                           │
│  ├── Graceful degradation                       │
│  ├── Health checks                              │
│  └── State persistence (crash recovery)         │
└─────────────────────────────────────────────────┘
```

### Bayesian Kelly Position Sizing

Standard Kelly assumes known win rate. Bayesian Kelly accounts for uncertainty:

```python
from src.risk.bayesian_kelly import BayesianKellySizer, TradeOutcome

sizer = BayesianKellySizer(
    kelly_fraction=0.25,      # Use 25% of full Kelly
    max_position_pct=0.20,    # Max 20% of portfolio per position
    min_observations=20       # Min 20 trades before full sizing
)

# Record trade outcomes (Bayesian posterior updates)
sizer.record_outcome(TradeOutcome(
    symbol='AAPL',
    strategy='momentum',
    win=True,
    profit_pct=0.025
))

# Get position size
shares, kelly_result = sizer.get_position_size(
    symbol='AAPL',
    strategy='momentum',
    portfolio_value=100000,
    current_price=175.50,
    signal_strength=0.8
)

print(f"Recommended shares: {shares}")
print(f"Kelly fraction: {kelly_result.fractional_kelly:.2%}")
print(f"Win rate estimate: {kelly_result.win_rate_estimate.mean:.2%}")
print(f"Uncertainty penalty: {kelly_result.uncertainty_penalty:.2%}")

# Key insight: Higher uncertainty → smaller position
# As more trades are recorded, uncertainty decreases
```

### Correlation Circuit Breaker

Detects when correlations spike (crisis mode) and reduces exposure:

```python
from src.risk.correlation_breaker import CorrelationCircuitBreaker

# Initialize with baseline correlation from training period
circuit_breaker = CorrelationCircuitBreaker(
    baseline_correlation=baseline_corr_matrix,
    correlation_spike_threshold=0.25,  # 25% increase = warning
    crisis_threshold=0.40,             # 40% increase = crisis
    first_pc_threshold=0.55            # First PC > 55% = warning
)

# Check on each bar
triggered, alert = circuit_breaker.check(recent_returns)

if triggered:
    print(f"Circuit breaker triggered: {alert.state.value}")
    print(f"Action: {alert.action.value}")

    # Get position multiplier
    multiplier = circuit_breaker.get_position_multiplier()
    # Returns: 1.0 (normal), 0.5 (elevated), 0.25 (crisis), 0.0 (flatten)

    # Reduce all positions
    for position in active_positions:
        new_size = int(position.quantity * multiplier)
        if new_size < position.quantity:
            await reduce_position(position, new_size)
```

### Pre-Trade Impact Estimation

Estimate market impact before executing:

```python
from src.execution.impact_model import AlmgrenChrissModel

impact_model = AlmgrenChrissModel(
    permanent_impact_coef=0.1,
    temporary_impact_coef=0.2
)

# Estimate impact
estimate = impact_model.estimate_impact(
    quantity=1000,
    price=175.50,
    adv=5_000_000,      # Average daily volume
    volatility=0.02,    # Daily volatility (2%)
    spread_bps=3.0      # Bid-ask spread
)

print(f"Expected Impact: {estimate.total_bps:.1f} bps")
print(f"  Permanent: {estimate.permanent_bps:.1f} bps")
print(f"  Temporary: {estimate.temporary_bps:.1f} bps")
print(f"Optimal Horizon: {estimate.optimal_horizon_minutes:.0f} minutes")
print(f"Recommended Strategy: {estimate.recommended_strategy}")

# Should we execute?
decision = impact_model.should_execute(
    quantity=1000,
    price=175.50,
    adv=5_000_000,
    volatility=0.02,
    expected_alpha_bps=15.0,           # Expected return
    max_impact_to_alpha_ratio=0.5      # Don't trade if impact > 50% of alpha
)

if decision.should_execute:
    print(f"Execute: Net alpha = {decision.expected_net_alpha:.1f} bps")
else:
    print(f"Skip trade: {decision.reason}")
```

---

## Monitoring

### Execution Dashboard

Real-time execution quality monitoring:

```python
from src.monitoring.execution_dashboard import ExecutionMonitor

monitor = ExecutionMonitor(
    slippage_alert_threshold_bps=10.0,
    fill_rate_alert_threshold=0.8
)

# Start tracking an execution
execution = monitor.start_execution(
    order_id='ord_123',
    symbol='AAPL',
    side='buy',
    target_quantity=100,
    target_price=175.50,
    vwap_benchmark=175.40
)

# Record fills as they come in
monitor.record_fill(
    order_id='ord_123',
    fill_quantity=50,
    fill_price=175.55
)

monitor.record_fill(
    order_id='ord_123',
    fill_quantity=50,
    fill_price=175.52
)

# Generate quality report
report = monitor.generate_report(period_hours=24)

print(f"Executions: {report.total_executions}")
print(f"Avg Slippage: {report.avg_slippage_bps:.2f} bps")
print(f"Median Slippage: {report.median_slippage_bps:.2f} bps")
print(f"P95 Slippage: {report.p95_slippage_bps:.2f} bps")
print(f"Avg Fill Rate: {report.avg_fill_rate:.1%}")
print(f"Beat VWAP: {report.pct_beat_vwap:.1%}")
print(f"Total Slippage Cost: ${report.total_slippage_cost:,.2f}")
```

### P&L Attribution

Understand where profits and losses come from:

```python
from src.analytics.attribution import PnLAttribution, Trade, TradeSide, ExitReason

attribution = PnLAttribution()

# Record a completed trade
trade = Trade(
    trade_id='t_123',
    symbol='AAPL',
    strategy='momentum',
    side=TradeSide.LONG,
    entry_time=datetime(2024, 1, 15, 10, 30),
    entry_price=175.00,
    entry_quantity=100,
    exit_time=datetime(2024, 1, 15, 14, 30),
    exit_price=178.50,
    exit_quantity=100,
    exit_reason=ExitReason.TAKE_PROFIT,
    commission=1.00,
    entry_vwap=175.10,
    exit_vwap=178.40
)

# Get attribution breakdown
attr = attribution.add_trade(trade)

print("=== Trade Attribution ===")
print(f"Gross P&L: ${attr.gross_pnl:.2f}")
print(f"Net P&L: ${attr.net_pnl:.2f}")
print()
print("Alpha Sources:")
print(f"  Direction Alpha: ${attr.direction_alpha:.2f}")
print(f"  Timing Alpha: ${attr.timing_alpha:.2f}")
print(f"  Sizing Alpha: ${attr.sizing_alpha:.2f}")
print()
print("Costs:")
print(f"  Commission: ${attr.commission_cost:.2f}")
print(f"  Slippage: ${attr.slippage_cost:.2f}")
print(f"  Spread: ${attr.spread_cost:.2f}")
print(f"  Market Impact: ${attr.market_impact_cost:.2f}")

# Portfolio-level report
portfolio = attribution.get_portfolio_attribution(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30)
)

print(f"\n=== Portfolio Attribution ===")
print(f"Total Trades: {portfolio.total_trades}")
print(f"Net P&L: ${portfolio.net_pnl:,.2f}")
print(f"Win Rate: {portfolio.win_rate:.1%}")
print(f"Profit Factor: {portfolio.profit_factor:.2f}")
```

### Model Staleness Detection

Detect when model needs retraining:

```python
from src.mlops.staleness import ModelStalenessDetector

detector = ModelStalenessDetector(
    model_name='momentum_v1',
    model_trained_date=datetime(2024, 10, 1),
    max_age_days=30,
    min_accuracy_threshold=0.52
)

# Record prediction outcomes
detector.record_prediction(
    symbol='AAPL',
    prediction=2,       # Predicted class
    probability=0.68,   # Model confidence
    actual=2            # Actual outcome (from future)
)

# Check staleness (run periodically)
report = detector.check_staleness()

print(f"Model: {report.model_name}")
print(f"Staleness Level: {report.staleness_level.value}")
print(f"Is Stale: {report.is_stale}")

if report.issues:
    print("Issues:")
    for issue in report.issues:
        print(f"  - {issue}")

print(f"Recommendation: {report.recommendation}")
print(f"Retrain Urgency: {report.retrain_urgency}")

# Staleness levels:
# - FRESH: Model performing well
# - AGING: Early signs of degradation
# - STALE: Needs attention
# - CRITICAL: Should not be used
```

### Health Checks

For Kubernetes/container deployment:

```python
from src.core.health_check import HealthCheckRegistry, HealthCheckServer

registry = HealthCheckRegistry()

# Register component health checks
registry.register('broker_api', broker_api.health_check)
registry.register('data_feed', data_feed.health_check)
registry.register('redis', state_manager.health_check)
registry.register('model', model.health_check)

# Start HTTP server
server = HealthCheckServer(registry, port=8080)
await server.start()

# Available endpoints:
# GET /health       - Overall system health
# GET /health/live  - Liveness probe (is process running?)
# GET /health/ready - Readiness probe (can handle requests?)
```

---

## Troubleshooting

### Common Issues

#### 1. "Broker connection failed"

```python
# Check API keys are set
import os

print(f"API Key set: {bool(os.getenv('ALPACA_API_KEY'))}")
print(f"Secret Key set: {bool(os.getenv('ALPACA_SECRET_KEY'))}")

# Verify URL is correct
# Paper: https://paper-api.alpaca.markets
# Live: https://api.alpaca.markets

# Test connection
from src.data.alpaca_client import AlpacaDataClient
client = AlpacaDataClient()
try:
    account = await client.get_account()
    print(f"Connected! Status: {account.status}")
except Exception as e:
    print(f"Connection failed: {e}")
```

#### 2. "Look-ahead bias detected"

```python
# Solution: Use point-in-time feature engine
from src.features.point_in_time import PointInTimeFeatureEngine

engine = PointInTimeFeatureEngine()
# This engine automatically validates for future data leakage

# Common causes of look-ahead bias:
# - Using future data in feature calculation
# - Fitting models on entire dataset then testing on same data
# - Not respecting time ordering in train/test split
```

#### 3. "Redis connection failed"

```python
# System automatically falls back to in-memory storage
# Check logs for: "Using in-memory fallback"

from src.core.state_manager import RedisStateManager

manager = RedisStateManager()
await manager.initialize()

# If Redis unavailable, still works but:
# - No crash recovery
# - State lost on restart
# - Single instance only
```

#### 4. "Model accuracy degraded"

```python
# Check staleness detector
report = staleness_detector.check_staleness()

if report.retrain_urgency in ['high', 'critical']:
    # Options:
    # 1. Retrain model with recent data
    # 2. Reduce position sizes temporarily
    # 3. Switch to backup model

    # Check accuracy windows
    windows = staleness_detector.get_accuracy_windows()
    for w in windows:
        print(f"{w.window_name}: {w.accuracy:.1%}")
```

#### 5. "High correlation detected"

```python
# Check correlation circuit breaker
status = circuit_breaker.get_status()

print(f"State: {status['state']}")
print(f"Triggered: {status['is_triggered']}")
print(f"Position Multiplier: {status['position_multiplier']}")

if status['is_triggered']:
    # Correlation spike detected
    # System should automatically reduce positions
    # Check recommended action
    print(f"Action: {status['recommended_action']}")
```

### Log Levels

```python
import logging

# For debugging - verbose output
logging.getLogger('alphatrade').setLevel(logging.DEBUG)

# For production - important events only
logging.getLogger('alphatrade').setLevel(logging.INFO)

# Log locations:
# - logs/trading.log    (main trading logs)
# - logs/execution.log  (order execution)
# - logs/risk.log       (risk events)
```

### Quick Diagnostics

```python
# Run system diagnostics
async def run_diagnostics():
    results = {}

    # 1. Broker connection
    try:
        account = await broker_api.get_account()
        results['broker'] = f"OK - {account.status}"
    except Exception as e:
        results['broker'] = f"FAILED - {e}"

    # 2. Data feed
    try:
        bars = await data_feed.get_latest_bars(['AAPL'])
        results['data_feed'] = f"OK - {len(bars)} bars"
    except Exception as e:
        results['data_feed'] = f"FAILED - {e}"

    # 3. Model
    try:
        pred = model.predict(sample_features)
        results['model'] = "OK - predictions working"
    except Exception as e:
        results['model'] = f"FAILED - {e}"

    # 4. Redis
    try:
        pong = await state_manager._redis.ping()
        results['redis'] = "OK" if pong else "FAILED"
    except:
        results['redis'] = "FALLBACK - using in-memory"

    # Print results
    for component, status in results.items():
        symbol = "✓" if "OK" in status or "FALLBACK" in status else "✗"
        print(f"{symbol} {component}: {status}")

await run_diagnostics()
```

---

## AI Agent Integration

This system is optimized for AI agent usage.

### Key Files for Understanding the System

```
# 1. Architecture and design decisions
ARCHITECTURAL_REVIEW_REPORT.md  # Detailed architecture analysis

# 2. This guide
SYSTEM_GUIDE.md                 # Complete system documentation

# 3. Configuration
config/symbols.yaml             # Trading symbols
config/risk_params.yaml         # Risk limits

# 4. Main entry point
main.py                         # System orchestrator

# 5. Critical modules
src/execution/order_manager.py  # Order lifecycle management
src/risk/risk_manager.py        # Risk checks and limits
src/strategy/ml_strategy.py     # Signal generation
```

### Common Agent Commands

```bash
# Start system (paper trading)
python main.py --mode paper

# Run backtest
python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-01

# Train model
python scripts/train_model.py --config config/training.yaml

# Calibrate barriers
python scripts/calibrate_triple_barrier.py --calibrate

# Health check
curl http://localhost:8080/health

# Metrics (Prometheus format)
curl http://localhost:8081/metrics
```

### Agent Task Examples

```
┌────────────────────────────────────────────────────────────────┐
│ Task: "Run a backtest and analyze results"                     │
├────────────────────────────────────────────────────────────────┤
│ Steps:                                                         │
│ 1. python scripts/run_backtest.py --start 2024-01-01 \        │
│       --end 2024-06-01 --output results/                       │
│ 2. Read results/backtest_report.csv                            │
│ 3. Report: Sharpe, Max DD, Win Rate, Profit Factor             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Task: "Check if model needs retraining"                        │
├────────────────────────────────────────────────────────────────┤
│ Steps:                                                         │
│ 1. Load src/mlops/staleness.py                                 │
│ 2. Call detector.check_staleness()                             │
│ 3. Report: staleness_level, issues, recommendation             │
│ 4. If urgent, suggest retraining steps                         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Task: "Analyze execution quality"                              │
├────────────────────────────────────────────────────────────────┤
│ Steps:                                                         │
│ 1. Load src/monitoring/execution_dashboard.py                  │
│ 2. Call monitor.generate_report(period_hours=24)               │
│ 3. Report: avg_slippage, fill_rate, vwap_performance           │
│ 4. If slippage high, check impact_model estimates              │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Task: "Check system risk status"                               │
├────────────────────────────────────────────────────────────────┤
│ Steps:                                                         │
│ 1. Load src/risk/correlation_breaker.py                        │
│ 2. Call circuit_breaker.get_status()                           │
│ 3. Report: state, is_triggered, position_multiplier            │
│ 4. If triggered, list recommended actions                      │
└────────────────────────────────────────────────────────────────┘
```

### Module Quick Reference

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `src/data/alpaca_client.py` | Data fetching | `AlpacaDataClient` |
| `src/data/live_validator.py` | Data validation | `LiveDataValidator` |
| `src/features/institutional.py` | Feature engineering | `InstitutionalFeatureEngine` |
| `src/features/point_in_time.py` | Bias-free features | `PointInTimeFeatureEngine` |
| `src/models/ml_model.py` | ML models | `CatBoostModel`, `XGBoostModel` |
| `src/models/calibration.py` | Prob calibration | `ProbabilityCalibrationManager` |
| `src/backtest/engine.py` | Backtesting | `BacktestEngine` |
| `src/backtest/realistic_fills.py` | Fill simulation | `RealisticFillSimulator` |
| `src/risk/bayesian_kelly.py` | Position sizing | `BayesianKellySizer` |
| `src/risk/correlation_breaker.py` | Correlation monitor | `CorrelationCircuitBreaker` |
| `src/execution/protected_positions.py` | Bracket orders | `ProtectedPositionManager` |
| `src/execution/impact_model.py` | Impact estimation | `AlmgrenChrissModel` |
| `src/execution/reconciliation.py` | State sync | `ReconciliationEngine` |
| `src/core/state_manager.py` | Redis state | `RedisStateManager` |
| `src/core/graceful_degradation.py` | Fault tolerance | `GracefulDegradationManager` |
| `src/monitoring/execution_dashboard.py` | Exec quality | `ExecutionMonitor` |
| `src/analytics/attribution.py` | P&L breakdown | `PnLAttribution` |
| `src/mlops/staleness.py` | Model health | `ModelStalenessDetector` |

---

---

## Execution Order (Step-by-Step Pipeline)

This section provides the **exact order** of scripts to run after models are trained.

### Pre-requisites Checklist

Before proceeding, verify:
- [ ] Python environment activated
- [ ] `.env` file configured with Alpaca API keys
- [ ] Models trained (in `models/` directory)
- [ ] Historical data available (in `data/raw/` directory)

### PHASE 1: Data Validation (Run First)

```bash
# Step 1: Validate data quality
python scripts/data_quality_pipeline.py --symbols config/symbols.yaml

# This checks:
# - OHLC validity (high >= low, etc.)
# - Volume anomalies
# - Missing data gaps
# - Price spikes/outliers

# Expected output: data_quality_report.json in results/
```

### PHASE 2: Pre-Training Validation

```bash
# Step 2: Validate features and labels (if retraining needed)
python scripts/run_pre_training_validation.py

# This validates:
# - Feature stationarity
# - Label distribution
# - Look-ahead bias checks
# - Train/test data integrity
```

### PHASE 3: Backtesting (CRITICAL - Do This After Training)

```bash
# Step 3: Run backtest with InstitutionalBacktestEngine
python scripts/run_pipeline.py --stage backtest --force

# OR run directly:
python scripts/run_backtest.py \
    --start 2024-01-01 \
    --end 2024-11-01 \
    --initial-capital 1000000 \
    --output results/

# This runs backtest with:
# - InstitutionalBacktestEngine (microstructure simulation)
# - Realistic fill model (slippage, market impact)
# - Risk management integration
```

**Expected Output:**
```
results/
├── equity_curve.csv
├── trades.csv
├── backtest_report.json
└── metrics_summary.txt
```

### PHASE 4: Analyze Backtest Results

```bash
# Step 4: Review backtest metrics
python -c "
import json
with open('results/backtest_report.json', 'r') as f:
    report = json.load(f)
print('=== BACKTEST RESULTS ===')
print(f\"Sharpe Ratio: {report.get('sharpe_ratio', 'N/A')}\")
print(f\"Max Drawdown: {report.get('max_drawdown', 'N/A')}\")
print(f\"Win Rate: {report.get('win_rate', 'N/A')}\")
print(f\"Profit Factor: {report.get('profit_factor', 'N/A')}\")
print(f\"Total Trades: {report.get('total_trades', 'N/A')}\")
"
```

**Minimum Thresholds Before Proceeding:**
| Metric | Minimum | Ideal |
|--------|---------|-------|
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 20% | < 10% |
| Win Rate | > 48% | > 52% |
| Profit Factor | > 1.2 | > 1.5 |

### PHASE 5: Paper Trading (Recommended)

```bash
# Step 5: Start paper trading
python main.py --mode paper --config config/settings.yaml

# Run for minimum 2 weeks before live trading
# Monitor logs in logs/ directory
```

### PHASE 6: Live Trading (CAUTION)

```bash
# Step 6: Live trading (ONLY after successful paper trading)
python main.py --mode live --config config/settings.yaml

# CRITICAL SAFETY FEATURES NOW ACTIVE:
# - Emergency kill switch (await system.emergency_halt())
# - Reconciliation engine (broker state sync)
# - Protected positions (bracket orders)
# - Circuit breakers (1% intraday loss halt)
```

---

## Complete Pipeline Command Summary

```bash
# Stage 1: Data - Download historical data
python scripts/run_pipeline.py --stage data

# Stage 2: Features - Generate institutional features  
python scripts/run_pipeline.py --stage features

# Stage 3: Labels - Generate triple barrier labels
python scripts/run_pipeline.py --stage labels
# OR calibrate separately:
python scripts/calibrate_triple_barrier.py --calibrate

# Stage 4: Train - Train ML model with purged k-fold CV
python scripts/run_pipeline.py --stage train

# Stage 5: Calibrate - Calibrate probabilities for Kelly sizing
python scripts/run_pipeline.py --stage calibrate

# Stage 6: Backtest - Validate with realistic execution ← YOU ARE HERE
python scripts/run_pipeline.py --stage backtest --force

# Stage 7: Validate - Final validation
python scripts/run_pipeline.py --stage validate

# Stage 8: Paper Trading (optional)
python scripts/run_pipeline.py --stage paper

```

---

## Emergency Procedures

### Emergency Kill Switch

If something goes wrong during live trading:

```python
# Method 1: From running system
# The system has an emergency_halt() method that:
# - Cancels ALL pending orders
# - Closes ALL positions at market
# - Disables trading until manual reset

# Method 2: Via Python console
import asyncio
from main import AlphaTradeSystem

async def emergency_stop():
    system = AlphaTradeSystem()
    await system.initialize()
    results = await system.emergency_halt()
    print(f"Cancelled orders: {results['orders_cancelled']}")
    print(f"Closed positions: {results['positions_closed']}")

asyncio.run(emergency_stop())
```

### Manual Position Close

```python
# Close all positions manually via Alpaca
from src.execution.broker_api import AlpacaBroker

async def close_all():
    broker = AlpacaBroker()
    await broker.connect()
    positions = await broker.get_positions()
    for pos in positions:
        await broker.close_position(pos.symbol)
        print(f"Closed {pos.symbol}")

asyncio.run(close_all())
```

---

## Recent Updates (December 2024 Audit)

The following critical fixes were implemented:

### 1. Signal Handler Race Condition (FIXED)
- Added idempotency guard to prevent multiple concurrent shutdowns
- Location: `main.py:1263-1279`

### 2. Look-Ahead Bias in Ichimoku (FIXED)
- Chikou Span (`shift(-26)`) excluded by default
- Safe components only: `tenkan_sen`, `kijun_sen`
- Location: `src/features/technical.py:187-258`

### 3. Emergency Kill Switch (ADDED)
- New method: `system.emergency_halt()`
- Immediately cancels orders and closes positions
- Location: `main.py:1265-1355`

### 4. Thread Safety in OrderManager (FIXED)
- Added `threading.Lock()` for `_active_orders`
- Prevents race conditions with broker callbacks
- Location: `src/execution/order_manager.py:384-411`

### 5. Circuit Breaker Thresholds (UPDATED)
- Added: `intraday_loss_circuit_breaker: 0.01` (1%)
- Added: `intraday_loss_warning: 0.005` (0.5%)
- Location: `config/risk_params.yaml:186-188`

### 6. Execution Engine Tests (ADDED)
- New test file: `tests/test_execution_engine.py`
- Includes broker mocking and stress tests
- Run with: `pytest tests/test_execution_engine.py -v`

---

## Next Steps

1. **Run Backtest** - Execute backtest with trained models
2. **Review Metrics** - Ensure Sharpe > 1.0, Max DD < 20%
3. **Paper Trading** - Run for minimum 2 weeks
4. **Monitor & Iterate** - Track model staleness, tune parameters
5. **Live Trading** - Start with small positions (25% of intended)

---

*This document is for AlphaTrade System v4.1*
*Last updated: December 14, 2024*
*Audit Status: PASSED - System ready for backtesting*
