# AlphaTrade System

An **institutional-grade algorithmic trading platform** designed for systematic, multi-strategy trading. Built with production-quality code following **JPMorgan-level** design standards.

**Version 2.0.0** - Complete institutional upgrade with circuit breakers, real-time monitoring, audit trail, and compliance systems.

---

## Quick Start (5 Minutes)

### Step 1: Clone & Setup Environment

```powershell
# Clone repository
git clone <repository-url>
cd AlphaTrade_System

# Create virtual environment (Python 3.11-3.13)
py -3.13 -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate

# Activate (Unix/MacOS)
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data

Place your OHLCV data files in `data/raw/` folder:
```
data/raw/
├── AAPL_15min.csv
├── GOOGL_15min.csv
├── MSFT_15min.csv
└── ...
```

CSV format:
```csv
datetime,open,high,low,close,volume
2023-01-01 09:30:00,150.00,151.50,149.50,151.00,1000000
```

### Step 3: Run Backtest

```powershell
python main.py --mode backtest
```

---

## Complete Step-by-Step Guide (Zero to Backtest)

For someone starting from scratch, follow these steps in order:

### Phase 1: Environment Setup

```powershell
# 1. Open PowerShell in the project folder

# 2. Create virtual environment
py -3.13 -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import numpy; import pandas; import sklearn; print('All OK!')"
```

### Phase 2: Data Preparation

```powershell
# 1. Create data directories (if not exist)
mkdir data\raw -Force
mkdir data\processed -Force
mkdir data\cache -Force

# 2. Place your OHLCV CSV files in data/raw/
# File naming: SYMBOL_15min.csv (e.g., AAPL_15min.csv)

# 3. Verify data
python -c "from src.data import DataLoader; dl = DataLoader('data/raw'); print(f'Found {len(dl.symbols)} symbols')"
```

### Phase 3: Configuration Check

```powershell
# 1. Review main config
cat config/trading_config.yaml

# 2. Review risk limits
cat config/risk_limits.yaml

# 3. Review ML config
cat config/ml_config.yaml
```

### Phase 4: Feature Generation

```python
# Run in Python interpreter or script
from src.data import DataLoader
from src.features import FeaturePipeline

# Load data
loader = DataLoader('data/raw')
data = {s: loader.load_symbol(s) for s in loader.symbols[:5]}

# Generate features
pipeline = FeaturePipeline()
for symbol, df in data.items():
    features = pipeline.fit_transform(df)
    print(f"{symbol}: {len(features.columns)} features generated")
```

### Phase 5: Run Backtest

```powershell
# Simple backtest with momentum strategy
python main.py --mode backtest --strategy momentum

# Full pipeline (train + backtest)
python main.py

# With specific config
python main.py --config config/institutional_defaults.yaml
```

### Phase 6: Review Results

```powershell
# Results are saved in results/ folder
# - results/backtest_YYYYMMDD_HHMMSS.html (interactive report)
# - results/metrics.json (performance metrics)
# - logs/alphatrade_YYYYMMDD.log (detailed logs)
```

---

## File Execution Order (For Developers)

If you want to understand or modify the system, here's the complete execution flow:

### 1. Configuration Loading
```
config/__init__.py          → Load all YAML configs
config/settings.py          → Pydantic settings validation
```

### 2. Data Loading
```
src/data/loaders/data_loader.py     → Load OHLCV data
src/data/validators/data_validator.py → Validate data quality
src/data/processors/data_processor.py → Clean and process
src/data/pit/universe_manager.py     → Survivorship bias handling
```

### 3. Feature Engineering
```
src/features/technical/indicators.py  → Technical indicators
src/features/pipeline.py              → Feature pipeline (fit/transform)
src/features/processor.py             → Feature scaling/normalization
```

### 4. Strategy Signal Generation
```
src/strategies/momentum.py            → Momentum signals
src/strategies/mean_reversion.py      → Mean reversion signals
src/strategies/ml_based/ml_alpha.py   → ML-based signals
```

### 5. Risk Management (Pre-Trade)
```
src/risk/pretrade_compliance.py       → Pre-trade checks
src/risk/circuit_breakers.py          → Circuit breaker checks
src/risk/position_sizing.py           → Position sizing
```

### 6. Backtesting
```
src/backtesting/engine.py             → Vectorized backtest
src/backtesting/event_engine.py       → Event-driven backtest
src/backtesting/market_impact.py      → Market impact model
src/backtesting/metrics.py            → Performance metrics
```

### 7. Risk Monitoring (Post-Trade)
```
src/risk/realtime_monitor.py          → Real-time monitoring
src/risk/drawdown.py                  → Drawdown control
src/risk/var_models.py                → VaR calculations
```

### 8. Reporting
```
src/reporting/html_report.py          → HTML reports
src/reporting/metrics_report.py       → Metrics summary
```

---

## What's New in v2.0.0

### Critical Bug Fixes
- **Cash Validation**: Proper cash checks before trades (no negative cash)
- **Data Leakage Prevention**: Proper fit/transform separation with warnings
- **Market Impact**: Connected to actual ADV data (not hardcoded)
- **Risk Parity Convergence**: Added convergence logging

### New Institutional Modules

#### Circuit Breakers (`src/risk/circuit_breakers.py`)
- Market-wide circuit breakers (7%, 13%, 20% drops)
- Portfolio-specific rapid loss detection
- Volatility spike detection
- Automatic trading halts

#### Real-Time Risk Monitor (`src/risk/realtime_monitor.py`)
- Continuous P&L monitoring
- VaR limit tracking
- Exposure monitoring
- Alert generation

#### Pre-Trade Compliance (`src/risk/pretrade_compliance.py`)
- Order size limits
- Position concentration checks
- Sector exposure limits
- ADV liquidity checks
- Restricted securities list

#### Audit Trail (`src/compliance/audit_trail.py`)
- Immutable event logging
- Hash chain verification
- Regulatory compliance (MiFID II, SEC)
- CSV export for reporting

#### Health Monitoring (`src/infrastructure/health_monitor.py`)
- Component health tracking
- Resource utilization monitoring
- Alert callbacks

#### Failover/Recovery (`src/infrastructure/failover.py`)
- State checkpointing
- Automatic recovery
- Manual intervention support

### Environment-Specific Configs
- `config/development.yaml` - Fast iteration settings
- `config/staging.yaml` - Pre-production testing
- `config/production.yaml` - Production settings

---

## Key Features

### Trading Strategies
- **Multi-Strategy Framework**: Momentum, mean reversion, volatility breakout, ML-based
- **Signal Generation**: Composite signals with confidence scoring
- **Walk-Forward Training**: Time-series cross-validation with purging

### Machine Learning Pipeline
- **Model Factory**: LightGBM, XGBoost, CatBoost, Random Forest
- **Hyperparameter Optimization**: Optuna integration
- **Deep Learning**: LSTM with attention (PyTorch Lightning)
- **Experiment Tracking**: MLflow integration
- **Combinatorial Purged CV (CPCV)**: Institutional-grade validation

### Feature Engineering
- **Technical Indicators**: 50+ indicators
- **Fractional Differentiation**: Memory-preserving stationarity
- **Microstructure Features**: OFI, VPIN, Kyle's lambda
- **GARCH Volatility**: Multiple GARCH variants

### Risk Management
- **Position Sizing**: Kelly, volatility targeting, risk parity
- **VaR/CVaR**: Historical, parametric, Monte Carlo (99% confidence)
- **Drawdown Controls**: Automatic scaling with state persistence
- **Circuit Breakers**: Active monitoring and halts
- **Pre-Trade Compliance**: Full order validation

### Portfolio Optimization
- **Mean-Variance**: Classic Markowitz
- **Risk Parity**: Equal risk contribution
- **Black-Litterman**: View-based optimization

---

## Project Structure

```
AlphaTrade_System/
├── config/                          # Configuration
│   ├── settings.py                  # Pydantic settings
│   ├── base.yaml                    # Base configuration
│   ├── development.yaml             # Dev environment
│   ├── staging.yaml                 # Staging environment
│   ├── production.yaml              # Production environment
│   ├── risk_limits.yaml             # Risk limits
│   └── ml_config.yaml               # ML configuration
├── data/
│   ├── raw/                         # Raw OHLCV data
│   ├── processed/                   # Processed data
│   └── cache/                       # Cache
├── src/
│   ├── data/                        # Data layer
│   │   ├── loaders/                 # Data loaders
│   │   ├── validators/              # Data validation
│   │   ├── processors/              # Data processing
│   │   ├── pit/                     # Point-in-Time data
│   │   └── storage/                 # Storage backends
│   ├── features/                    # Feature engineering
│   │   ├── technical/               # Technical indicators
│   │   ├── regime/                  # Regime detection
│   │   ├── microstructure/          # Microstructure features
│   │   └── pipeline.py              # Feature pipeline
│   ├── strategies/                  # Trading strategies
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   └── ml_based/
│   ├── risk/                        # Risk management
│   │   ├── circuit_breakers.py      # Circuit breakers
│   │   ├── realtime_monitor.py      # Real-time monitoring
│   │   ├── pretrade_compliance.py   # Pre-trade checks
│   │   ├── position_sizing.py       # Position sizing
│   │   ├── drawdown.py              # Drawdown control
│   │   └── var_models.py            # VaR calculations
│   ├── compliance/                  # Compliance
│   │   └── audit_trail.py           # Audit trail
│   ├── infrastructure/              # Infrastructure
│   │   ├── health_monitor.py        # Health monitoring
│   │   └── failover.py              # Failover/recovery
│   ├── backtesting/                 # Backtesting
│   │   ├── engine.py                # Vectorized engine
│   │   ├── event_engine.py          # Event-driven engine
│   │   └── market_impact.py         # Market impact
│   ├── portfolio/                   # Portfolio management
│   └── training/                    # ML training
├── tests/                           # Test suite
│   ├── unit/
│   │   ├── test_risk.py
│   │   ├── test_circuit_breakers.py
│   │   ├── test_compliance.py
│   │   └── test_drawdown_persistence.py
│   └── integration/
├── logs/                            # Log files
├── results/                         # Backtest results
├── state/                           # State persistence
├── main.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project config
└── setup_env.bat                    # Environment setup
```

---

## Configuration

### Risk Limits (`config/risk_limits.yaml`)
```yaml
position_sizing:
  method: "volatility_target"
  target_volatility: 0.15
  max_leverage: 2.0

drawdown:
  max_drawdown_pct: 0.20
  warning_threshold: 0.10

exposure:
  max_gross_exposure: 1.5
  max_single_position: 0.20

circuit_breakers:
  market_drop:
    - threshold: -0.07
      action: "pause_15min"
    - threshold: -0.20
      action: "halt_day"
```

### Environment Variables (`.env`)
```bash
# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000

# FRED API (for macro features)
FRED_API_KEY=your_api_key_here

# Database (optional)
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432
```

---

## Testing

```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_circuit_breakers.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Performance Metrics

### Statistical Metrics
- **Deflated Sharpe Ratio (DSR)**: Adjusts for multiple testing
- **Probabilistic Sharpe Ratio (PSR)**: Statistical significance

### Risk Metrics
- VaR (95%, 99%), CVaR
- Max Drawdown, Volatility
- Sharpe, Sortino, Calmar Ratios

### Trading Metrics
- Win Rate, Profit Factor
- Average Win/Loss
- Trade Count, Turnover

---

## Troubleshooting

### "Module not found" errors
```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate

# Reinstall dependencies
pip install -r requirements.txt
```

### VS Code doesn't activate venv
1. Close VS Code completely
2. Delete `.vscode/settings.json` if corrupted
3. Reopen VS Code
4. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
5. Select `.venv` Python

### Data loading errors
```powershell
# Check data format
python -c "import pandas as pd; df = pd.read_csv('data/raw/AAPL_15min.csv'); print(df.head())"
```

---

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Black, F. and Litterman, R. (1992). *Global Portfolio Optimization*
- Cartea, A. et al. (2015). *Algorithmic and High-Frequency Trading*

---

## License

MIT License - See LICENSE file for details.
