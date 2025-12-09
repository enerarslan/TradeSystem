# 🚀 AlphaTrade Platform - Complete Guide

## Your Trading System at a Glance

You have built a **JPMorgan-level algorithmic trading platform** with:
- ✅ 46 supported stock symbols
- ✅ 6 ML model types (LightGBM, XGBoost, CatBoost, LSTM, Transformer, TCN)
- ✅ Triple Barrier Method for proper target labeling
- ✅ Meta-labeling for bet sizing
- ✅ Institutional-grade backtesting with transaction costs
- ✅ Alpaca integration for live trading
- ✅ TWAP/VWAP/Iceberg execution algorithms

---

## 🎯 CRITICAL: Where to Start

### Step 1: Check Your Data
```bash
ls data/storage/*.csv
```
Your CSV files should be named: `{SYMBOL}_15min.csv`

**Required columns:** `timestamp, open, high, low, close, volume`

### Step 2: Train Models (MOST IMPORTANT)
```bash
# Train all 46 symbols
python scripts/train_all_symbols.py

# Or train specific symbols
python scripts/train_all_symbols.py --symbols AAPL GOOGL MSFT

# With hyperparameter optimization
python scripts/train_all_symbols.py --optimize --n-trials 50
```

### Step 3: Run Backtest
```bash
# Full institutional backtest
python scripts/run_institutional_backtest.py

# Specific symbols
python scripts/run_institutional_backtest.py --symbols AAPL GOOGL
```

### Step 4: Paper Trading (After Backtest Success)
```bash
python main.py paper
```

---

## 📁 Project Architecture

```
your-project/
├── main.py                          # 🎯 MAIN ENTRY POINT
├── config/
│   ├── settings.py                  # Environment config
│   └── symbols.py                   # 46 supported symbols
├── data/
│   ├── storage/                     # 📊 PUT YOUR CSV FILES HERE
│   ├── loader.py                    # Data loading
│   └── processor.py                 # Data cleaning
├── features/
│   ├── technical.py                 # 50+ technical indicators
│   ├── statistical.py               # Statistical features
│   ├── advanced.py                  # ⭐ Triple Barrier, Meta-labeling
│   └── pipeline.py                  # Feature orchestration
├── models/
│   ├── classifiers.py               # LightGBM, XGBoost, CatBoost
│   ├── deep.py                      # LSTM, Transformer, TCN
│   ├── training.py                  # Optuna optimization
│   ├── model_manager.py             # Model registry
│   └── artifacts/                   # 📦 TRAINED MODELS SAVED HERE
│       └── {SYMBOL}/
│           ├── {SYMBOL}_lightgbm_v1.pkl
│           └── {SYMBOL}_xgboost_v1.pkl
├── strategies/
│   ├── alpha_ml_v2.py               # ⭐⭐⭐ THE MAIN ML STRATEGY
│   ├── momentum.py                  # MACD, RSI, Breakout
│   └── statistical.py               # Pairs, Cointegration
├── backtesting/
│   └── institutional.py             # JPMorgan-level backtesting
├── execution/
│   ├── broker.py                    # Alpaca integration
│   ├── algorithms.py                # TWAP, VWAP, Iceberg
│   └── live_engine.py               # Live trading engine
├── api/
│   └── main.py                      # FastAPI server
└── scripts/
    ├── train_all_symbols.py         # ⭐ TRAIN ALL MODELS
    └── run_institutional_backtest.py # ⭐ RUN BACKTESTS
```

---

## 🔥 The Most Important Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `main.py` | Main entry point | Always - Interactive menu |
| `scripts/train_all_symbols.py` | Train ML models | First step - Train your models |
| `scripts/run_institutional_backtest.py` | Run backtests | After training - Validate strategy |
| `strategies/alpha_ml_v2.py` | THE ML strategy | This generates trading signals |
| `execution/broker.py` | Alpaca integration | For live/paper trading |

---

## ⚙️ Environment Setup (.env file)

Create a `.env` file in your project root:

```bash
# Trading Mode
TRADING_MODE=backtest
DEBUG=true
LOG_LEVEL=INFO

# Alpaca Credentials (get from alpaca.markets)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER_TRADING=true

# Data Paths
DATA_STORAGE_PATH=data/storage
DATA_PROCESSED_PATH=data/processed
DATA_CACHE_PATH=data/cache

# Backtest Settings
BACKTEST_INITIAL_CAPITAL=100000
BACKTEST_COMMISSION_PCT=0.001
BACKTEST_SLIPPAGE_PCT=0.0005
```

---

## 🤖 ML Models Explained

### Primary Models (Use These)

| Model | When to Use | Strengths |
|-------|-------------|-----------|
| **LightGBM** | Default choice | Fast, handles categorical features |
| **XGBoost** | Ensemble member | Robust, regularization |

### Secondary Models (Optional)

| Model | When to Use | Strengths |
|-------|-------------|-----------|
| CatBoost | Alternative | Native categorical handling |
| LSTM | Sequential patterns | Captures time dependencies |
| Transformer | Complex patterns | Attention mechanism |
| TCN | Time series | Temporal convolutions |

---

## 📈 Strategy: AlphaML V2

Your main strategy (`strategies/alpha_ml_v2.py`) uses:

1. **Triple Barrier Method** - Proper target labeling
   - Take profit: 2x ATR
   - Stop loss: 1x ATR
   - Max holding: 20 bars

2. **Ensemble Prediction** - LightGBM + XGBoost

3. **Meta-Labeling** - Bet sizing based on confidence

4. **Regime Detection** - Adapts to market conditions

---

## 📊 Complete Command Reference

### Training Commands
```bash
# Train all symbols
python scripts/train_all_symbols.py

# Train specific symbols
python scripts/train_all_symbols.py --symbols AAPL GOOGL MSFT TSLA

# Train with Optuna optimization
python scripts/train_all_symbols.py --optimize --n-trials 100

# Train only core liquid symbols
python scripts/train_all_symbols.py --core-only
```

### Backtesting Commands
```bash
# Standard backtest
python scripts/run_institutional_backtest.py

# Specific symbols
python scripts/run_institutional_backtest.py --symbols AAPL GOOGL

# All symbols with $1M capital
python scripts/run_institutional_backtest.py --all-symbols --capital 1000000

# Custom timeframe
python scripts/run_institutional_backtest.py --timeframe 1hour
```

### Trading Commands
```bash
# Interactive menu
python main.py

# Paper trading
python main.py paper

# Start API server
python main.py api
```

---

## ⚠️ Live Trading Checklist

Before going live, ensure:

- [ ] Models trained with 1+ years of data
- [ ] Backtest Sharpe ratio > 1.0
- [ ] Maximum drawdown < 20%
- [ ] Paper traded for 1+ month
- [ ] Paper results match backtest
- [ ] Alpaca live credentials configured
- [ ] Risk limits properly set
- [ ] Kill switches tested

---

## 🔧 Troubleshooting

### "No data files found"
```bash
# Check data location
ls data/storage/

# Files should be named like:
# AAPL_15min.csv, GOOGL_15min.csv, etc.
```

### "Model not found"
```bash
# Train models first
python scripts/train_all_symbols.py

# Check if models exist
ls models/artifacts/AAPL/
```

### "Alpaca connection failed"
```bash
# Check .env file has correct credentials
# Make sure you're using paper API URL first
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## 🎯 Your Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   1. PREPARE DATA                                           │
│   └── Place CSV files in data/storage/                      │
│                                                             │
│   2. TRAIN MODELS                                           │
│   └── python scripts/train_all_symbols.py                   │
│                                                             │
│   3. BACKTEST                                               │
│   └── python scripts/run_institutional_backtest.py          │
│                                                             │
│   4. PAPER TRADE                                            │
│   └── python main.py paper                                  │
│                                                             │
│   5. LIVE TRADE (after extensive testing)                   │
│   └── Change ALPACA_BASE_URL to live API                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📞 Quick Help

- **Main entry:** `python main.py`
- **Train models:** `python scripts/train_all_symbols.py`
- **Run backtest:** `python scripts/run_institutional_backtest.py`
- **Paper trade:** `python main.py paper`
- **API server:** `python main.py api`

---

*This is a JPMorgan-level trading system. Use responsibly. Never trade with money you can't afford to lose.*
