# AlphaTrade System - Quick Start Guide

Institutional-grade algorithmic trading system with ML-powered signal generation.

## Prerequisites

- **Python 3.10+** - Required for all core functionality
- **Docker & Docker Compose** - Optional, for Redis/Grafana/Prometheus
- **Alpaca Account** - Required for paper/live trading (free tier available)

## One-Command Setup

```bash
# Clone and setup
git clone <repository-url>
cd AlphaTrade_System
./scripts/setup.sh
```

The setup script will:
1. Check prerequisites
2. Create virtual environment
3. Install dependencies
4. Create directory structure
5. Configure API keys
6. Optionally start Docker services

## Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw,cache,processed,holdout}
mkdir -p results/{features,labels,backtest,paper}
mkdir -p models logs

# Copy example configs
cp config/settings.example.yaml config/settings.yaml
cp config/symbols.example.yaml config/symbols.yaml
cp .env.example .env

# Edit .env with your Alpaca API keys
```

## Run Full Pipeline

The pipeline handles everything from data to backtest:

```bash
make pipeline
```

This runs 7 stages:
1. **Data** - Download historical OHLCV data
2. **Features** - Generate institutional features (FracDiff, VPIN, HMM Regime)
3. **Labels** - Create triple barrier labels with meta-labeling
4. **Train** - Train CatBoost ensemble with purged K-fold CV
5. **Calibrate** - Calibrate probabilities for Kelly sizing
6. **Backtest** - Validate with realistic fills and slippage
7. **Validate** - Verify all components work together

### Skip Existing Data

If you already have data downloaded:

```bash
make pipeline-fast
```

### Run Individual Stages

```bash
make data       # Download data
make features   # Generate features
make labels     # Create labels
make train      # Train model
make calibrate  # Calibrate probabilities
make backtest   # Run backtest
make validate   # Validate system
```

## Start Paper Trading

Once the pipeline completes:

```bash
make paper
```

Or directly with Python:

```bash
python main.py --mode paper
```

### Required Environment Variables

```bash
export ALPACA_API_KEY=your_api_key
export ALPACA_API_SECRET=your_api_secret
```

Or add to `.env` file.

## Monitor System

### Grafana Dashboards

If Docker is running:

- **URL**: http://localhost:3000
- **Default login**: admin / admin123

Available dashboards:
- Execution Quality (slippage, fill rates)
- Portfolio P&L
- Model Performance
- System Health

### Prometheus Metrics

- **URL**: http://localhost:9090

### Log Files

```bash
tail -f logs/trading.log        # Main log
tail -f logs/audit.log          # Audit trail
```

### Health Check

```bash
make health
```

## File Structure After Pipeline

```
AlphaTrade_System/
├── data/
│   ├── raw/                    # Downloaded OHLCV data
│   ├── cache/                  # Cached data
│   ├── processed/
│   │   └── combined_data.pkl   # Preprocessed data
│   └── holdout/                # Holdout data for validation
├── results/
│   ├── features/
│   │   └── combined_features.pkl   # Engineered features
│   ├── labels/
│   │   └── labels.pkl              # Triple barrier labels
│   ├── backtest/
│   │   ├── backtest_report.json    # Performance metrics
│   │   ├── equity_curve.csv        # Daily equity
│   │   └── trades.csv              # Trade log
│   └── paper/                      # Paper trading logs
├── models/
│   ├── model.pkl               # Trained CatBoost model
│   ├── features.txt            # Feature names
│   ├── metrics.yaml            # Training metrics
│   └── calibration_model.pkl   # Probability calibrator
├── config/
│   ├── settings.yaml           # Main settings
│   ├── symbols.yaml            # Trading universe
│   └── risk_params.yaml        # Risk parameters
└── logs/
    ├── trading.log             # Main log
    └── audit.log               # Audit trail
```

## Configuration

### Main Settings (`config/settings.yaml`)

```yaml
trading:
  initial_capital: 1000000
  max_positions: 20

broker:
  name: alpaca
  # API keys from environment

logging:
  level: INFO
  log_dir: logs
```

### Trading Universe (`config/symbols.yaml`)

```yaml
sectors:
  technology:
    symbols: [AAPL, MSFT, GOOGL, META, NVDA]
  financials:
    symbols: [JPM, BAC, GS, MS]
  # ... more sectors
```

### Risk Parameters (`config/risk_params.yaml`)

```yaml
position_limits:
  max_position_pct: 0.10    # Max 10% per position
  max_sector_pct: 0.30      # Max 30% per sector

drawdown:
  max_drawdown: 0.15        # Max 15% drawdown
  daily_loss_limit: 0.03    # Max 3% daily loss

volatility:
  target_annual: 0.15       # Target 15% annual vol
```

## Available Make Commands

```bash
make help          # Show all commands

# Data & Features
make data          # Download data
make features      # Generate features
make labels        # Create labels

# Training
make train         # Train model
make calibrate     # Calibrate probabilities
make train-all     # Full training pipeline

# Backtesting
make backtest      # Run backtest
make validate      # Validate system

# Trading
make paper         # Paper trading
make live          # Live trading (requires confirmation)

# Utilities
make health        # Check system health
make test          # Run tests
make clean         # Remove generated files

# Docker
make docker-up     # Start services
make docker-down   # Stop services
make docker-logs   # View logs
```

## Integrated Components

The system includes these institutional-grade components:

| Component | Purpose |
|-----------|---------|
| **ProtectedPositionManager** | Server-side stop loss with bracket orders |
| **ReconciliationEngine** | State sync between local and broker |
| **GracefulDegradationManager** | Fault tolerance and fallback handling |
| **RedisStateManager** | Crash recovery with persistent state |
| **BayesianKellySizer** | Uncertainty-aware position sizing |
| **CorrelationCircuitBreaker** | Crisis detection and exposure reduction |
| **ProbabilityCalibrator** | Model probability calibration for Kelly |
| **ExecutionMetricsCollector** | Execution quality monitoring |
| **ModelStalenessDetector** | Model health and accuracy monitoring |
| **AlmgrenChrissModel** | Pre-trade market impact estimation |
| **RealisticFillSimulator** | Conservative fill simulation in backtest |

## Troubleshooting

### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Docker Issues

```bash
# Check Docker status
docker ps

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### No Data Downloaded

```bash
# Check Alpaca API credentials
echo $ALPACA_API_KEY

# Download data manually
make data
```

### Model Training Fails

```bash
# Check for sufficient data
ls -la data/processed/

# Run with verbose logging
python scripts/train_models.py --symbol AAPL
```

### Paper Trading Won't Start

```bash
# Check API credentials
python -c "import os; print(os.environ.get('ALPACA_API_KEY', 'NOT SET'))"

# Check model exists
make check-model

# Validate system
make validate
```

## Next Steps

1. **Customize Strategy**: Edit `config/symbols.yaml` to adjust trading universe
2. **Tune Risk**: Modify `config/risk_params.yaml` for your risk tolerance
3. **Monitor Performance**: Use Grafana dashboards for real-time monitoring
4. **Iterate**: Retrain model periodically with `make train`

## Support

For issues and questions:
- Check logs in `logs/` directory
- Run `make validate` to verify system health
- Refer to inline documentation in source code
