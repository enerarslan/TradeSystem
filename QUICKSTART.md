# AlphaTrade System - Quick Start Guide

## System Requirements

- Python 3.11 or 3.12
- 16GB RAM minimum (32GB recommended for ML training)
- Windows 10/11, macOS, or Linux

---

## Step 1: Installation

### 1.1 Create Virtual Environment

```bash
# Navigate to project directory
cd AlphaTrade_System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
# Install all required dependencies
pip install -e .

# For development (testing, linting):
pip install -e ".[dev]"
```

### 1.3 Verify Installation

```bash
python -c "from src import DataLoader, BacktestEngine; print('Installation successful!')"
```

---

## Step 2: Prepare Your Data

### 2.1 Data Directory Structure

Place your market data in the `data/raw/` directory:

```
data/
  raw/
    AAPL.csv
    MSFT.csv
    GOOGL.csv
    ...
```

### 2.2 Data Format

Each CSV file must have these columns:
- `date` or `Date` - Date/datetime index
- `open` or `Open` - Opening price
- `high` or `High` - High price
- `low` or `Low` - Low price
- `close` or `Close` - Closing price
- `volume` or `Volume` - Trading volume

Example CSV format:
```csv
date,open,high,low,close,volume
2023-01-03,130.28,130.90,124.17,125.07,112117500
2023-01-04,126.89,128.66,125.08,126.36,89113600
...
```

### 2.3 Sample Data (for Testing)

If you don't have data, create sample data:

```python
python scripts/generate_sample_data.py
```

Or download from your data provider (Alpha Vantage, Yahoo Finance, etc.)

---

## Step 3: Configuration (Optional)

### 3.1 Default Configuration

The system uses sensible defaults. No configuration needed for basic usage.

### 3.2 Custom Configuration

Create/edit `config/trading_config.yaml`:

```yaml
data:
  path: "data/raw"
  min_bars: 100

backtest:
  initial_capital: 1000000
  commission_pct: 0.001
  slippage_pct: 0.0005

strategies:
  momentum:
    enabled: true
    lookback_periods: [5, 10, 20]
    top_n_long: 5
  mean_reversion:
    enabled: true
    lookback_period: 20
    entry_zscore: 2.0
  volatility_breakout:
    enabled: true
    atr_period: 14
    atr_multiplier: 2.0

training:
  model_type: lightgbm
  cv_splits: 5
  optuna_trials: 50

risk:
  max_position_pct: 0.05
  max_drawdown_pct: 0.15
```

---

## Step 4: Run the System

### 4.1 Basic Backtest (Start Here!)

Run all strategies with default settings:

```bash
python main.py
```

This will:
1. Load data from `data/raw/`
2. Generate technical features
3. Run Momentum, Mean Reversion, and Volatility Breakout strategies
4. Run Ensemble strategy
5. Generate reports in `reports/`

### 4.2 Single Strategy

```bash
# Run momentum strategy only
python main.py --strategy momentum

# Run mean reversion strategy only
python main.py --strategy mean_reversion

# Run volatility breakout strategy only
python main.py --strategy volatility_breakout

# Run ensemble strategy
python main.py --strategy ensemble
```

### 4.3 Event-Driven Backtest

For tick-by-tick simulation:

```bash
python main.py --engine event-driven
```

### 4.4 Custom Capital

```bash
python main.py --capital 500000
```

### 4.5 Specific Symbols

```bash
python main.py --symbols AAPL MSFT GOOGL
```

---

## Step 5: ML Model Training

### 5.1 Train ML Model

```bash
# Train with LightGBM (default)
python main.py --mode train

# Train with XGBoost
python main.py --mode train --model xgboost

# Train with CatBoost
python main.py --mode train --model catboost
```

### 5.2 Train with Hyperparameter Optimization

```bash
python main.py --mode train --model lightgbm --optimize
```

This runs Optuna to find optimal hyperparameters.

---

## Step 6: View Results

### 6.1 Output Files

After running, check these directories:

```
reports/
  tear_sheet_YYYYMMDD_HHMMSS.html     # Interactive performance dashboard
  backtest_report_YYYYMMDD_HHMMSS.html # Detailed report
  strategy_comparison_YYYYMMDD_HHMMSS.csv # Summary comparison

results/
  backtest_YYYYMMDD_HHMMSS.pkl        # Raw results (for reloading)

logs/
  alphatrade_YYYY-MM-DD.log           # Detailed logs
```

### 6.2 View Tear Sheet

Open the HTML file in your browser:

```bash
# Windows
start reports/tear_sheet_*.html

# macOS
open reports/tear_sheet_*.html

# Linux
xdg-open reports/tear_sheet_*.html
```

### 6.3 Regenerate Reports

From saved results:

```bash
python main.py --mode report --input results/backtest_20231201_120000.pkl
```

---

## Step 7: Advanced Usage

### 7.1 MLflow Experiment Tracking

Start MLflow UI:

```bash
mlflow ui
```

Open http://localhost:5000 in browser.

### 7.2 Feature Store

```python
from src.features import FeatureStore

store = FeatureStore(base_path="data/features")
store.register_feature("returns_5d", compute_fn, version="1.0")
features = store.get_features(["returns_5d", "volatility_20d"], as_of_date)
```

### 7.3 Macroeconomic Features

Set FRED API key:

```bash
# Windows
set FRED_API_KEY=your_api_key_here

# Linux/macOS
export FRED_API_KEY=your_api_key_here
```

```python
from src.features import MacroFeatureGenerator

macro = MacroFeatureGenerator(api_key="your_key")
macro_features = macro.get_all_features(start_date, end_date)
```

### 7.4 Deep Learning Models

```python
from src.training.deep_learning import LSTMModel, TemporalFusionTransformer

model = LSTMModel(input_size=50, hidden_size=128)
trainer.train(model, X_train, y_train)
```

---

## Execution Order Summary

For first-time users, follow this order:

1. **Install**: `pip install -e .`
2. **Add Data**: Place CSV files in `data/raw/`
3. **Run Basic**: `python main.py`
4. **View Results**: Open `reports/tear_sheet_*.html`
5. **Customize**: Edit `config/trading_config.yaml`
6. **Train ML**: `python main.py --mode train`
7. **Optimize**: `python main.py --mode train --optimize`

---

## Troubleshooting

### Import Errors

```bash
# Reinstall package
pip install -e . --force-reinstall
```

### No Data Found

Ensure CSV files are in `data/raw/` with correct format.

### Out of Memory

- Reduce number of symbols
- Use fewer features
- Increase system swap space

### GPU Not Found (Deep Learning)

```bash
# Check PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Performance

- Use Polars for data loading (automatic)
- Enable Numba JIT compilation (automatic)
- Reduce lookback periods

---

## Command Reference

```bash
# All options
python main.py --help

# Common commands
python main.py                              # Default backtest
python main.py --strategy momentum          # Single strategy
python main.py --engine event-driven        # Event-driven
python main.py --mode train --model lightgbm # Train ML
python main.py --mode train --optimize      # Train + optimize
python main.py --mode report --input file.pkl # Regenerate report
python main.py --symbols AAPL MSFT          # Specific symbols
python main.py --capital 500000             # Custom capital
python main.py --log-level DEBUG            # Verbose logging
```

---

## Next Steps

1. **Review the tear sheet** - Understand your strategy performance
2. **Adjust parameters** - Tune strategy parameters in config
3. **Add more data** - Include more symbols or longer history
4. **Train ML models** - Use machine learning for signal generation
5. **Run optimization** - Find optimal hyperparameters with Optuna
6. **Analyze risk** - Review drawdowns and risk metrics
7. **Paper trade** - Test strategies in real-time simulation

---

## Support

- Check `logs/` for detailed error messages
- Review docstrings in source code
- Run with `--log-level DEBUG` for verbose output
