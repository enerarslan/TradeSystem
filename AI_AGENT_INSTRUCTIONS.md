# AI AGENT TASK: AlphaTrade System Integration & Orchestration
## Priority: CRITICAL | Complexity: HIGH | Estimated Time: 4-6 hours

---

# CONTEXT

You are working on AlphaTrade, an institutional-grade algorithmic trading system. The system has ~35,000 lines of code across 55+ Python files. Recently, critical modules were added:

**NEW MODULES (not integrated into main.py):**
- `src/execution/protected_positions.py` - Bracket orders with server-side stop loss
- `src/execution/reconciliation.py` - State sync between local and broker
- `src/features/point_in_time.py` - Look-ahead bias prevention
- `src/core/graceful_degradation.py` - Fault tolerance
- `src/core/state_manager.py` - Redis state persistence
- `src/risk/bayesian_kelly.py` - Uncertainty-aware position sizing
- `src/risk/correlation_breaker.py` - Crisis detection
- `src/models/calibration.py` - Probability calibration
- `src/monitoring/execution_dashboard.py` - Execution quality metrics
- `src/mlops/staleness.py` - Model health monitoring
- `src/backtest/realistic_fills.py` - Conservative fill simulation
- `src/execution/impact_model.py` - Pre-trade impact estimation

There are also other modules that were created. You need to check them also separately.

**PROBLEM:** The main.py orchestrator does NOT use these new modules. The user has not trained the model or run backtests yet. They don't know which files to run in what order.

---

# TASK 1: UPDATE main.py TO INTEGRATE ALL NEW MODULES

Update `main.py` to include:

1. **Import all new modules** at the top of the file

2. **In the `__init__` method, add new component references:**
   - `self.protected_position_manager` - For bracket orders
   - `self.reconciliation_engine` - For state sync
   - `self.graceful_degradation` - For fault tolerance
   - `self.state_manager` - For Redis persistence
   - `self.bayesian_kelly` - For position sizing
   - `self.correlation_breaker` - For crisis detection
   - `self.calibration_manager` - For probability calibration
   - `self.execution_monitor` - For execution quality
   - `self.staleness_detector` - For model health

3. **In the `initialize` method:**
   - Initialize GracefulDegradationManager first (catches failures)
   - Initialize RedisStateManager for crash recovery
   - Initialize ReconciliationEngine with broker
   - Initialize ProtectedPositionManager with broker
   - Initialize CorrelationCircuitBreaker with baseline correlations
   - Initialize BayesianKellySizer with priors
   - Initialize ProbabilityCalibrationManager
   - Initialize ExecutionMonitor for metrics
   - Initialize ModelStalenessDetector
   - Register all components with GracefulDegradationManager for health checks

4. **In the trading loop:**
   - Before each trade cycle: check `graceful_degradation.is_trading_allowed()`
   - Before each trade cycle: check `correlation_breaker.check()` for crisis
   - Replace old position sizing with `bayesian_kelly.get_position_size()`
   - Replace direct broker orders with `protected_position_manager.open_position_with_protection()`
   - After signals: calibrate probabilities with `calibration_manager.calibrate()`
   - After each fill: record outcome with `bayesian_kelly.record_outcome()`
   - After each fill: record with `staleness_detector.record_prediction()`
   - Every 30 seconds: trigger `reconciliation_engine.reconcile()`

5. **In shutdown:**
   - Stop reconciliation engine
   - Stop graceful degradation monitoring
   - Save state to Redis
   - Generate execution quality report

6. **Add health check endpoint** (if HTTP server exists):
   - Return status from all registered components
   - Return current degradation level
   - Return correlation breaker state

---

# TASK 2: CREATE MASTER PIPELINE SCRIPT

Create `scripts/run_pipeline.py` that orchestrates the entire workflow:

## Pipeline Stages (in order):

### Stage 1: Data Preparation
```
Purpose: Download and preprocess historical data
Input: Symbol list, date range from config
Output: results/data/combined_data.pkl
Dependencies: None
Command: Calls MultiAssetLoader and DataPreprocessor
```

### Stage 2: Feature Engineering  
```
Purpose: Generate all features (institutional + point-in-time)
Input: results/data/combined_data.pkl
Output: results/features/combined_features.pkl
Dependencies: Stage 1
Command: Calls InstitutionalFeatureEngineer and PointInTimeFeaturePipeline
Important: Use expanding window mode for HMM, not full-series fit
```

### Stage 3: Label Generation
```
Purpose: Create triple barrier labels for supervised learning
Input: results/data/combined_data.pkl
Output: results/labels/labels.pkl
Dependencies: Stage 1
Command: Calls TripleBarrierLabeler with calibrated parameters
Note: Run calibrate_triple_barrier.py first if params don't exist
```

### Stage 4: Model Training
```
Purpose: Train CatBoost model with purged k-fold CV
Input: results/features/combined_features.pkl, results/labels/labels.pkl
Output: models/catboost_model.pkl, models/training_metrics.json
Dependencies: Stage 2, Stage 3
Command: Calls ModelTrainer with WalkForwardValidator
Important: Use 80/20 train/val split, NO SHUFFLE (time series!)
```

### Stage 5: Probability Calibration
```
Purpose: Calibrate model probabilities for Kelly sizing
Input: models/catboost_model.pkl, validation data
Output: models/calibration_model.pkl
Dependencies: Stage 4
Command: Calls ProbabilityCalibrationManager.fit()
```

### Stage 6: Backtest
```
Purpose: Validate strategy with realistic execution
Input: All previous outputs
Output: results/backtest/backtest_report.json, results/backtest/equity_curve.csv
Dependencies: Stages 1-5
Command: Calls InstitutionalBacktestEngine with RealisticFillSimulator
Important: Use conservative fill model, not optimistic one
```

### Stage 7: Validation
```
Purpose: Verify all components work together
Input: All models and configs
Output: results/validation_report.json
Dependencies: Stages 1-6
Command: Run system health checks, verify metrics are reasonable
Checks:
- Sharpe ratio > 0.5
- Max drawdown < 25%
- Win rate > 45%
- Model accuracy > 52%
- Calibration ECE < 0.1
```

### Stage 8: Paper Trading (optional)
```
Purpose: Live paper trading for validation
Input: All previous outputs
Output: Real-time trading, logs in results/paper/
Dependencies: Stages 1-7, Alpaca paper account
Command: Runs main.py --mode paper
```

## Pipeline Script Features:
- Accept command line args: `--stage`, `--from-stage`, `--symbols`, `--start`, `--end`
- Save checkpoint after each stage
- Resume from last successful stage if interrupted
- Print clear progress with timing
- Generate final summary report

---

# TASK 3: CREATE Makefile FOR EASY COMMANDS

Create a `Makefile` with these targets:

```makefile
# Data & Features
make data          # Download and preprocess data
make features      # Generate all features
make labels        # Create triple barrier labels

# Training
make train         # Train model with CV
make calibrate     # Calibrate probabilities
make train-all     # labels + train + calibrate

# Backtesting
make backtest      # Run backtest with realistic fills
make analyze       # Generate backtest analysis report

# Full Pipeline
make pipeline      # Run entire pipeline (data → backtest)
make pipeline-fast # Skip data if exists, retrain model

# Trading
make paper         # Start paper trading
make live          # Start live trading (requires confirmation)

# Utilities
make validate      # Validate all components
make health        # Check system health
make clean         # Remove all generated files
make test          # Run unit tests

# Docker
make docker-build  # Build Docker image
make docker-up     # Start all services (Redis, Grafana, etc.)
make docker-down   # Stop all services
make docker-logs   # View logs
```

---

# TASK 4: UPDATE docker-compose.yaml

Ensure docker-compose.yaml properly orchestrates:

1. **alphatrade** - Main trading application
   - Depends on: redis, postgres
   - Mounts: config/, models/, results/
   - Environment: API keys from .env

2. **redis** - State persistence
   - Persistence enabled (AOF)
   - Health check configured

3. **postgres/timescaledb** - Market data storage
   - Init script for schema
   - Health check configured

4. **grafana** - Monitoring dashboards
   - Pre-configured dashboards for:
     - Execution quality (slippage, fill rates)
     - Portfolio P&L
     - Model performance
     - System health

5. **prometheus** - Metrics collection
   - Scrape config for alphatrade metrics endpoint

6. **jupyter** (optional) - Research notebook
   - For ad-hoc analysis

Add a `docker-compose.override.yaml` for development settings.

---

# TASK 5: CREATE SINGLE-COMMAND SETUP SCRIPT

Create `scripts/setup.sh`:

```bash
#!/bin/bash
# AlphaTrade One-Command Setup

echo "🚀 AlphaTrade Setup Starting..."

# 1. Check prerequisites
check_prerequisites()  # Python 3.10+, pip, docker

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create necessary directories
mkdir -p results/{data,features,models,backtest,logs}
mkdir -p models
mkdir -p logs

# 5. Copy example configs if not exist
cp -n config/settings.example.yaml config/settings.yaml
cp -n config/symbols.example.yaml config/symbols.yaml
cp -n .env.example .env

# 6. Prompt for API keys
read -p "Enter Alpaca API Key: " ALPACA_KEY
read -p "Enter Alpaca API Secret: " ALPACA_SECRET
# Save to .env

# 7. Start Docker services
docker-compose up -d redis grafana prometheus

# 8. Run validation
python scripts/validate_setup.py

echo "✅ Setup complete! Run 'make pipeline' to start."
```

---

# TASK 6: CREATE DOCUMENTATION

Create `docs/QUICKSTART.md`:

## Quick Start Guide

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Alpaca account (paper trading)

### One-Command Setup
```bash
./scripts/setup.sh
```

### Run Full Pipeline
```bash
make pipeline
```

### Start Paper Trading
```bash
make paper
```

### Monitor System
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Logs: `tail -f logs/trading.log`

### File Structure After Pipeline
```
results/
├── data/
│   └── combined_data.pkl          # Historical OHLCV data
├── features/
│   └── combined_features.pkl      # Engineered features
├── labels/
│   └── labels.pkl                 # Triple barrier labels
├── models/
│   ├── catboost_model.pkl         # Trained model
│   └── calibration_model.pkl      # Probability calibrator
└── backtest/
    ├── backtest_report.json       # Performance metrics
    └── equity_curve.csv           # Daily equity
```

---

# TASK 7: VALIDATION CHECKLIST

After completing all tasks, verify:

## Code Integration
- [ ] main.py imports all new modules
- [ ] main.py initializes all new components
- [ ] Trading loop uses ProtectedPositionManager
- [ ] Trading loop checks CorrelationCircuitBreaker
- [ ] Position sizing uses BayesianKelly
- [ ] ReconciliationEngine runs every 30 seconds
- [ ] GracefulDegradation handles component failures

## Pipeline
- [ ] `make pipeline` runs end-to-end without errors
- [ ] All output files are generated
- [ ] Backtest produces valid metrics (Sharpe > 0)
- [ ] Model accuracy > 50%

## Docker
- [ ] `docker-compose up` starts all services
- [ ] Redis is accessible from main app
- [ ] Grafana dashboards load correctly
- [ ] Prometheus scrapes metrics

## Documentation
- [ ] QUICKSTART.md is accurate
- [ ] All make commands work
- [ ] setup.sh completes successfully

---

# SUCCESS CRITERIA

The task is complete when a new user can:

1. Clone the repository
2. Run `./scripts/setup.sh`
3. Run `make pipeline`
4. See backtest results in `results/backtest/`
5. Run `make paper` to start paper trading
6. Monitor via Grafana at localhost:3000

All with ZERO manual intervention or Python knowledge required.

---

# PRIORITY ORDER

Execute tasks in this order:
1. **TASK 1** - Update main.py (CRITICAL - nothing works without this)
2. **TASK 2** - Create pipeline script (HIGH - enables training/backtesting)
3. **TASK 3** - Create Makefile (HIGH - simplifies commands)
4. **TASK 5** - Create setup script (MEDIUM - onboarding)
5. **TASK 4** - Update Docker (MEDIUM - production deployment)
6. **TASK 6** - Documentation (LOW - can be done last)
7. **TASK 7** - Validation (FINAL - verify everything works)

---

# NOTES FOR AI AGENT

- Do NOT break existing functionality while integrating
- Preserve backward compatibility with existing config files
- Add sensible defaults for all new components
- Include error handling with clear messages
- Log all important events for debugging
- Test each stage independently before moving to next
- The user's Alpaca credentials should come from environment variables
- All file paths should be relative to project root
- Use async/await consistently throughout
