# AlphaTrade Platform

**JPMorgan-Level Institutional Algorithmic Trading System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Pre-trained Models](#pre-trained-models)
6. [Data](#data)
7. [Core Components](#core-components)
8. [JPMorgan Backtest System](#jpmorgan-backtest-system)
9. [Feature Engineering](#feature-engineering)
10. [Trading Strategies](#trading-strategies)
11. [Risk Management](#risk-management)
12. [Configuration](#configuration)
13. [Running the System](#running-the-system)
14. [API Reference](#api-reference)
15. [Development Guide](#development-guide)

---

## Overview

AlphaTrade is a production-grade, distributed algorithmic trading platform with institutional-level architecture. It integrates ML-powered strategies, advanced portfolio optimization, regime detection, and rigorous validation methodologies used by quantitative hedge funds.

### Key Capabilities

| Category | Features |
|----------|----------|
| **ML Models** | LightGBM, XGBoost classifiers with 97 features per symbol |
| **Portfolio Optimization** | Max Sharpe, Risk Parity, HRP, Kelly Criterion, Min Variance |
| **Regime Detection** | HMM-inspired classification (Bull/Bear × High/Low Vol, Sideways, Crisis) |
| **Execution** | TWAP, VWAP, Iceberg, liquidity-constrained execution with market impact |
| **Validation** | Deflated Sharpe Ratio, PBO, Walk-Forward with Purging/Embargo |
| **Risk Management** | Position limits, drawdown controls, VaR limits, kill switch |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Gateway (FastAPI :8000)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Data Ingestion │  │ Strategy       │  │ Risk Engine    │                 │
│  │ Service        │──│ Engine         │──│                │                 │
│  │ (Alpaca WS)    │  │ (ML Inference) │  │ (Limits/VaR)   │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│          │                  │                   │                            │
│          ▼                  ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Redis Message Bus (Pub/Sub)                       │    │
│  │              Channels: quotes, signals, orders, risk, fills          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│          │                  │                   │                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ OEMS           │  │ Feature Store  │  │ Watchdog       │                 │
│  │ (Execution)    │  │ (Real-time)    │  │ (Kill Switch)  │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Redis State    │  │ TimescaleDB    │  │ MLflow         │                 │
│  │ Store          │  │ (Time Series)  │  │ Registry       │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
AlphaTrade_System/
├── api/                          # FastAPI REST API
│   ├── __init__.py
│   └── main.py                   # API endpoints (/health, /status, /backtest, etc.)
│
├── backtesting/                  # Backtesting Engine
│   ├── __init__.py
│   ├── engine.py                 # Core backtest engine
│   ├── execution.py              # Simulated order execution
│   ├── institutional.py          # Institutional-grade features
│   ├── institutional_extentions.py
│   ├── liquidity_constraints.py  # Volume participation, market impact
│   ├── metrics.py                # Performance metrics calculation
│   ├── order_book_simulator.py   # Order book reconstruction
│   └── validation.py             # DSR, PBO, walk-forward validation
│
├── config/                       # Configuration
│   ├── __init__.py
│   ├── settings.py               # Environment & app settings (Pydantic)
│   └── symbols.py                # Symbol definitions (46 symbols)
│
├── core/                         # Core Types & Interfaces
│   ├── __init__.py
│   ├── enums.py                  # OrderSide, OrderType, TimeInForce, etc.
│   ├── events.py                 # Event classes for pub/sub
│   ├── interfaces.py             # Abstract base classes
│   └── types.py                  # NamedTuples, TypedDicts
│
├── data/                         # Data Loading & Processing
│   ├── __init__.py
│   ├── alternative_data.py       # Macro, sentiment, options data
│   ├── bar_aggregation.py        # Dollar bars, volume bars
│   ├── loader.py                 # CSV data loader (Polars-based)
│   ├── processor.py              # Data cleaning, normalization
│   ├── provider.py               # Real-time data provider interface
│   ├── tick_data.py              # Tick data handling
│   └── timescale_loader.py       # TimescaleDB integration
│
├── execution/                    # Order Execution
│   ├── __init__.py
│   ├── algorithms.py             # TWAP, VWAP, Iceberg, Pegged
│   ├── broker.py                 # Broker interface (Alpaca)
│   └── smart_routing.py          # Smart order routing
│
├── features/                     # Feature Engineering
│   ├── __init__.py
│   ├── advanced.py               # Microstructure features (VPIN, Kyle's Lambda)
│   ├── cross_asset.py            # Cross-asset correlations
│   ├── feature_selection.py      # MDA, SFI feature importance
│   ├── feature_store.py          # Real-time feature store
│   ├── pipeline.py               # Feature pipeline orchestration
│   ├── statistical.py            # Statistical features
│   └── technical.py              # Technical indicators (167 features)
│
├── infrastructure/               # Infrastructure Layer
│   ├── __init__.py
│   ├── async_pool.py             # Async worker pool
│   ├── heartbeat.py              # Service heartbeat
│   ├── message_bus.py            # Redis pub/sub wrapper
│   ├── service_registry.py       # Service discovery
│   └── state_store.py            # Redis state management
│
├── models/                       # ML Models
│   ├── __init__.py
│   ├── artifacts/                # Trained model files (see below)
│   ├── base.py                   # Base model interface
│   ├── classifiers.py            # LightGBM, XGBoost, RandomForest
│   ├── deep.py                   # Neural network models
│   ├── mlflow_registry.py        # MLflow integration
│   ├── model_manager.py          # Model loading/versioning
│   ├── reinforcement.py          # RL models (PPO, A2C)
│   ├── stacking.py               # Ensemble stacking
│   └── training.py               # Training pipeline
│
├── portfolio/                    # Portfolio Optimization
│   ├── __init__.py
│   └── optimizer.py              # MVO, Risk Parity, HRP, Kelly
│
├── risk/                         # Risk Management
│   ├── __init__.py
│   └── manager.py                # Position limits, VaR, drawdown
│
├── scripts/                      # Utility Scripts
│   ├── __init__.py
│   ├── paper_trade.py            # Paper trading runner
│   ├── run_jpmorgan_backtest.py  # MAIN: Institutional backtest (2200+ lines)
│   ├── train_all_symbols.py      # Batch model training
│   └── validate_backtest.py      # Backtest validation suite
│
├── services/                     # Microservices
│   ├── __init__.py
│   ├── base_service.py           # Base service class
│   ├── data_ingestion.py         # Market data streaming
│   ├── oems.py                   # Order Execution Management
│   ├── risk_engine.py            # Real-time risk monitoring
│   ├── strategy_engine.py        # Signal generation
│   └── watchdog.py               # Health monitoring, kill switch
│
├── strategies/                   # Trading Strategies
│   ├── __init__.py
│   ├── alpha_ml_v2.py            # ML-based alpha strategy
│   ├── base.py                   # Strategy interface
│   ├── ml_strategy.py            # Generic ML strategy wrapper
│   ├── momentum.py               # Momentum strategies
│   └── statistical.py            # Statistical arbitrage
│
├── tests/                        # Test Suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_backtesting.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_strategies.py
│
├── data/storage/                 # Historical Data (46 CSV files)
├── results/                      # Backtest results output
├── docker/                       # Docker configuration
├── main.py                       # Monolithic entry point
├── main_distributed.py           # Distributed system entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.11+
- Redis 7.0+ (for distributed mode)
- PostgreSQL 15+ with TimescaleDB (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AlphaTrade_System.git
cd AlphaTrade_System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `polars` | High-performance DataFrames (faster than pandas) |
| `lightgbm`, `xgboost` | Gradient boosting models |
| `scikit-learn` | ML utilities, preprocessing |
| `scipy` | Optimization, statistics |
| `numpy` | Numerical computing |
| `fastapi` | REST API |
| `redis` | Message bus, state store |
| `structlog` | Structured logging |
| `alpaca-py` | Broker integration |

---

## Pre-trained Models

### Location
```
models/artifacts/
├── AAPL/
│   ├── metadata.json           # Model metadata & feature names
│   ├── AAPL_lightgbm_v1.pkl    # LightGBM classifier
│   └── AAPL_xgboost_v1.pkl     # XGBoost classifier
├── MSFT/
│   └── ...
└── [39 symbol directories]
```

### Available Symbols (39 trained models)
```
AAPL, ABBV, ADBE, AMZN, AVGO, AXP, BA, BAC, CAT, COST, CRM, CSCO, CVX, DIS,
GOOGL, GS, HD, HON, IBM, INTC, JNJ, JPM, KO, LLY, MA, MCD, META, MRK, MSFT,
NKE, NVDA, PEP, PG, TSLA, UNH, V, VZ, WMT, XOM
```

### Model Metadata Structure

```json
{
  "symbol": "AAPL",
  "models": {
    "xgboost_v1": {
      "model_id": "3d01f64c",
      "model_type": "xgboost",
      "version": "v1",
      "trained_at": "2025-12-07T16:52:17",
      "training_samples": 57607,
      "feature_count": 97,
      "feature_names": ["rsi_14", "macd_signal", ...],
      "train_accuracy": 0.647,
      "test_accuracy": 0.614,
      "test_auc": 0.574,
      "model_path": "models/artifacts/AAPL/AAPL_xgboost_v1.pkl",
      "is_active": true
    }
  },
  "active_model": "xgboost_v1"
}
```

### Loading Models

```python
import pickle
from pathlib import Path

# Load model with embedded feature names
model_path = Path("models/artifacts/AAPL/AAPL_xgboost_v1.pkl")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

# model_data contains:
# - "model": The trained classifier
# - "feature_names": List of 97 feature names in correct order
model = model_data["model"]
features = model_data["feature_names"]
```

---

## Data

### Location
```
data/storage/
├── AAPL_15min.csv    # 72,261 bars (~11 years of 15-min data)
├── MSFT_15min.csv    # 70,866 bars
├── GOOGL_15min.csv   # 62,488 bars
└── [46 CSV files total]
```

### Data Format

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Bar timestamp (UTC) |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | int | Bar volume |

### Loading Data

```python
from data.loader import CSVLoader
import polars as pl

# Using project loader
loader = CSVLoader()
df = loader.load("AAPL")  # Returns Polars DataFrame

# Direct load
df = pl.read_csv("data/storage/AAPL_15min.csv", try_parse_dates=True)
```

---

## Core Components

### 1. Feature Pipeline (`features/pipeline.py`)

Generates 167 technical features across 12 categories:

| Category | Features | Examples |
|----------|----------|----------|
| **Momentum** | 8 | RSI, ROC, Williams %R, TSI |
| **Trend** | 12 | MACD, ADX, Supertrend, Aroon |
| **Volatility** | 6 | ATR, Bollinger Width, Historical Vol |
| **Volume** | 5 | MFI, CMF, Force Index, EOM |
| **Price Action** | 10 | Returns, cumulative returns |
| **Statistical** | 15 | Skewness, Kurtosis, Rolling stats |
| **Microstructure** | 6 | VPIN, Kyle's Lambda (simulated) |

```python
from features.pipeline import FeaturePipeline, create_default_config

pipeline = FeaturePipeline(create_default_config())
features_df = pipeline.generate(ohlcv_df, symbol="AAPL")
```

### 2. Regime Detection (`scripts/run_jpmorgan_backtest.py`)

HMM-inspired market regime classification:

| Regime | Characteristics | Position Sizing | Stop Loss |
|--------|-----------------|-----------------|-----------|
| `BULL_LOW_VOL` | Uptrend, low volatility | 120% base | 1.0x |
| `BULL_HIGH_VOL` | Uptrend, high volatility | 80% base | 0.8x |
| `BEAR_LOW_VOL` | Downtrend, low volatility | 70% base | 0.7x |
| `BEAR_HIGH_VOL` | Downtrend, high volatility | 50% base | 0.6x |
| `SIDEWAYS` | Range-bound | 90% base | 0.9x |
| `CRISIS` | Extreme volatility | 25% base | 0.5x |

```python
from scripts.run_jpmorgan_backtest import RegimeDetector, RegimeType

detector = RegimeDetector(lookback=60)
regime = detector.detect(prices, volumes, timestamp)
params = detector.get_regime_params(regime, base_position_pct=0.08)
```

### 3. Portfolio Optimization (`scripts/run_jpmorgan_backtest.py`)

```python
from scripts.run_jpmorgan_backtest import PortfolioOptimizer, OptimizationMethod

optimizer = PortfolioOptimizer()

# Available methods
weights = optimizer.optimize(
    returns,
    current_weights,
    method=OptimizationMethod.MAX_SHARPE  # or RISK_PARITY, HRP, KELLY, MIN_VARIANCE
)
```

### 4. Liquidity Constraints (`backtesting/liquidity_constraints.py`)

```python
from backtesting.liquidity_constraints import LiquidityConstrainedExecutor, LiquidityConfig

config = LiquidityConfig(
    max_participation_rate=0.10,   # Max 10% of bar volume
    max_position_adv_pct=0.10,     # Max 10% of ADV
    enable_market_impact=True,     # Almgren-Chriss model
)

executor = LiquidityConstrainedExecutor(config)
result = executor.execute_order(
    order_id="order_123",
    symbol="AAPL",
    side="buy",
    quantity=1000,
    price=150.0,
    bar_volume=50000,
    bar_timestamp=datetime.now()
)
```

---

## JPMorgan Backtest System

### Main Script
```
scripts/run_jpmorgan_backtest.py  (~2,200 lines)
```

### Features Implemented

| Phase | Component | Status |
|-------|-----------|--------|
| **Phase 1** | Tick/Quote data, Dollar bars, Order flow | Simulated |
| **Phase 2** | Order book, Queue simulation, Market impact | Implemented |
| **Phase 3** | Alternative data, Feature selection, Regime detection | Implemented |
| **Phase 4** | Portfolio optimization (5 methods) | Implemented |
| **Phase 5** | DSR, PBO, Walk-forward validation | Implemented |

### Configuration

```python
@dataclass
class JPMorganBacktestConfig:
    # Capital
    initial_capital: float = 10_000_000  # $10M

    # Position Sizing
    max_position_pct: float = 0.08       # Max 8% per position
    max_portfolio_positions: int = 15    # Max 15 concurrent positions
    min_confidence: float = 0.55         # Min model confidence for entry

    # Execution
    max_participation_rate: float = 0.10  # Max 10% of bar volume
    max_position_adv_pct: float = 0.10    # Max 10% of ADV

    # Transaction Costs (institutional rates)
    commission_bps: float = 0.5          # 0.5 bps
    spread_bps: float = 1.0              # 1 bps half-spread
    market_impact_bps: float = 2.0       # 2 bps

    # Risk Management
    max_drawdown_pct: float = 0.20       # Kill at 20% drawdown
    daily_var_limit: float = 0.03        # 3% daily VaR
    position_stop_loss: float = 0.05     # 5% stop loss
    position_take_profit: float = 0.12   # 12% take profit
    min_holding_bars: int = 2            # Min 30 min hold (15-min bars)

    # Portfolio Optimization
    optimization_method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    enable_regime_detection: bool = True
```

### Running Backtests

```bash
# Single symbol
python scripts/run_jpmorgan_backtest.py --symbols AAPL

# Multiple symbols
python scripts/run_jpmorgan_backtest.py --symbols AAPL MSFT GOOGL JPM GS

# With specific capital
python scripts/run_jpmorgan_backtest.py --symbols AAPL --capital 5000000

# Different optimization method
python scripts/run_jpmorgan_backtest.py --symbols AAPL --optimization risk_parity

# All available symbols
python scripts/run_jpmorgan_backtest.py --all-symbols

# Core symbols (tech + finance)
python scripts/run_jpmorgan_backtest.py --core-symbols
```

### Output

```
======================================================================
BACKTEST RESULTS
======================================================================

Performance:
  Total Return: 158.96%
  Annual Return: 158.96%
  Annual Volatility: 506.42%
  Sharpe Ratio: 10.468
  Sortino Ratio: 0.525
  Calmar Ratio: 6.555
  Max Drawdown: 24.25%

Trading:
  Total Trades: 150
  Win Rate: 27.27%
  Profit Factor: 1.23
  Total Costs: $120,597.74

Regime Analysis:
  sideways: 18 bars
  bull_low_vol: 114 bars

Validation:
  Deflated Sharpe: 6.030
  Significant: True
  PBO: 50.0%
  Likely Overfit: False
======================================================================
```

---

## Feature Engineering

### 97 Model Features (in order)

The models expect features in this exact order:

```python
FEATURE_NAMES = [
    # Momentum (8)
    "rsi_14", "macd_signal", "macd_histogram", "stoch_d", "williams_r_14",
    "roc_12", "cci_20", "ultimate_oscillator",

    # TSI (2)
    "tsi", "tsi_signal",

    # Directional (3)
    "plus_di", "minus_di", "adx",

    # Trend (5)
    "supertrend", "supertrend_direction", "aroon_up", "aroon_down", "aroon_oscillator",

    # Other (2)
    "psar_direction", "bb_width",

    # Volatility (2)
    "atr_14", "hv_20",

    # Volume (4)
    "mfi_14", "ad_line", "cmf_20", "force_index_13", "eom_14",

    # Returns (4)
    "return_1", "cum_return_60", "cum_return_120", "cum_return_252",

    # Rolling Stats (12)
    "rolling_range_pct_5", "rolling_std_10", "rolling_range_pct_10",
    "rolling_std_60", "rolling_range_pct_60", "rolling_std_120",
    "rolling_range_pct_120", "rolling_std_252", "rolling_max_252",
    "rolling_range_pct_252", "rolling_skewness_60", "rolling_kurtosis_60",

    # Momentum & Price Position (8)
    "momentum_5", "price_position_5", "momentum_10", "price_position_10",
    "momentum_20", "price_position_20", "price_position_60"
]
```

---

## Trading Strategies

### Alpha ML Strategy (`strategies/alpha_ml_v2.py`)

```python
from strategies.alpha_ml_v2 import AlphaMLStrategy

strategy = AlphaMLStrategy(
    symbols=["AAPL", "MSFT"],
    model_type="xgboost",
    min_confidence=0.55
)

signals = strategy.generate_signals(features_dict)
```

### Strategy Interface

```python
from strategies.base import Strategy

class CustomStrategy(Strategy):
    def generate_signals(self, data: dict) -> dict[str, Signal]:
        # Return dict of symbol -> Signal(direction, confidence, price)
        pass

    def on_bar(self, bar: Bar) -> None:
        # Called on each new bar
        pass
```

---

## Risk Management

### Position Limits

```python
from risk.manager import RiskManager

manager = RiskManager(
    max_position_pct=0.08,      # 8% max per position
    max_portfolio_value=1.0,    # 100% invested max
    max_drawdown=0.20,          # 20% drawdown limit
    daily_var_limit=0.03        # 3% daily VaR
)

approved = manager.validate_order(order, portfolio_state)
```

### Kill Switch

The watchdog service monitors system health and triggers emergency shutdown:

```python
# Triggered automatically when:
# - Drawdown exceeds limit
# - Daily loss exceeds limit
# - System errors accumulate
# - Manual trigger via API

POST /killswitch
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# Trading Mode
TRADING_MODE=backtest  # backtest, paper, live

# Broker (Alpaca)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Infrastructure
REDIS_URL=redis://localhost:6379
TIMESCALE_URL=postgresql://user:pass@localhost:5432/trading

# Risk Limits
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=50000
MAX_DRAWDOWN_PCT=0.20

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## Running the System

### Backtest Mode

```bash
# Interactive
python main.py

# Command line
python main.py backtest --symbol AAPL MSFT --strategy alpha_ml

# JPMorgan-level institutional backtest
python scripts/run_jpmorgan_backtest.py --symbols AAPL MSFT GOOGL
```

### Distributed Mode (requires Redis)

```bash
# Start all services
python main_distributed.py dev

# Start specific service
python main_distributed.py start --service strategy_engine

# Check status
python main_distributed.py status
```

### Docker Mode

```bash
cd docker
docker compose up -d
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status (positions, PnL) |
| `/services` | GET | List registered services |
| `/positions` | GET | Current positions |
| `/risk` | GET | Risk state (drawdown, VaR) |
| `/killswitch` | POST | Emergency shutdown |
| `/backtest` | POST | Run backtest |
| `/models` | GET | List ML models |

### Example Request

```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "strategy": "alpha_ml",
    "start_date": "2024-01-01",
    "end_date": "2024-12-01",
    "initial_capital": 1000000
  }'
```

---

## Development Guide

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_strategies.py -v
```

### Code Style

```bash
# Lint
ruff check .

# Format
black .
isort .

# Type check
mypy .
```

### Adding a New Symbol

1. Add CSV data to `data/storage/{SYMBOL}_15min.csv`
2. Train models:
   ```bash
   python scripts/train_all_symbols.py --symbols NEW_SYMBOL
   ```
3. Models saved to `models/artifacts/{SYMBOL}/`

### Adding a New Strategy

1. Create `strategies/my_strategy.py`
2. Inherit from `Strategy` base class
3. Implement `generate_signals()` method
4. Register in `strategies/__init__.py`

---

## Important Notes for AI Agents

### Key Files to Understand

1. **`scripts/run_jpmorgan_backtest.py`** - Main institutional backtest (2200 lines)
   - Contains `JPMorganBacktestConfig`, `RegimeDetector`, `PortfolioOptimizer`
   - Main entry point for running backtests

2. **`features/pipeline.py`** - Feature generation pipeline
   - Generates 167 features from OHLCV data

3. **`models/artifacts/{SYMBOL}/metadata.json`** - Model metadata
   - Contains `feature_names` list (97 features) in correct order

4. **`config/settings.py`** - Application settings via Pydantic

### Common Patterns

```python
# Loading data
from data.loader import CSVLoader
loader = CSVLoader()
df = loader.load("AAPL")

# Loading model
import pickle
with open("models/artifacts/AAPL/AAPL_xgboost_v1.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
features = model_data["feature_names"]

# Generating features
from features.pipeline import FeaturePipeline, create_default_config
pipeline = FeaturePipeline(create_default_config())
features_df = pipeline.generate(df, "AAPL")
```

### Current Limitations

- Win rate is ~27% (low but profitable due to high reward/risk ratio)
- Models have test AUC ~0.57 (marginal predictive power)
- Stop losses may not trigger fast enough during gaps
- Some alternative data features are simulated (not real)

---

## License

MIT License - See [LICENSE](LICENSE)

## Disclaimer

**This software is for educational purposes only. Use at your own risk.**

Trading financial instruments carries significant risk of loss. Past performance is not indicative of future results. Always paper trade first and never risk money you cannot afford to lose.
