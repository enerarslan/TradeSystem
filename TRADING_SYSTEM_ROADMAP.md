# AlphaTrade System - AI Agent Roadmap
## JPMorgan-Level Institutional Trading Platform

**Version:** 3.3 - AFML INSTITUTIONAL GRADE + PRE-TRAINING VALIDATION PIPELINE
**Status:** ✅ All Components Built + Full AFML Implementation + Pre-Training Data Quality Pipeline
**Total Files:** 52+ Python files + configs + deployment
**Lines of Code:** ~30,000+
**Last Updated:** December 2024

---

## 🎯 System Overview

A complete institutional-grade algorithmic trading system capable of:
- Live trading 46 US stocks using 15-min OHLCV data
- ML-based signal generation (XGBoost, LightGBM, CatBoost, Neural Networks)
- Professional risk management (VaR, position limits, circuit breakers)
- Algorithmic execution (TWAP, VWAP, POV, Adaptive)
- Real-time monitoring and reporting

### Version 3.3 Pre-Training Validation Pipeline (NEW):
- **Data Quality Pipeline** - Trading hours filtering, volume anomaly detection, OHLC validation
- **Triple Barrier Calibration** - Per-symbol ATR-based barriers, VIX regime adjustment
- **Label Quality Validation** - Class distribution (25-40%), autocorrelation (<0.1) checks
- **Embargo Verification** - Calculates max lookback (200 bars), ensures ≥5% embargo
- **Holdout Data Reservation** - 3-month temporal holdout, 6-symbol holdout, stress periods
- **Feature Optimization** - Correlation clustering, reduction to 60-80 features
- **Regime Awareness Features** - VIX regime, trend regime, volatility regime
- **Symbol Parameter Calculator** - Real spread, volume, beta calculation per symbol

### Version 3.2 Enhanced Training Pipeline:
- **Automatic Symbol Loading** - Extracts all 46 symbols from sectors in symbols.yaml
- **Trading Hours Filter** - Filters to US regular hours (9:30-16:00 ET), removes pre/post market noise
- **Cross-Asset Features Integration** - Correlations, sector momentum, beta, factor exposures connected
- **Regime Detection Features** - HMM regime, volatility regime, trend regime integrated into training
- **Enhanced Sample Weights** - Combines uniqueness weights + time decay (configurable alpha)
- **Dynamic Embargo Calculation** - Automatically sized based on maximum feature lookback periods
- **Cross-Sectional Features** - Return/volume/momentum ranks, sector-relative metrics, z-scores
- **Symbol-Specific Transaction Costs** - Uses spread_bps from symbols.yaml for realistic backtests

### Version 3.1 AFML Institutional Features:
- **Information-Driven Bars** - Volume/Dollar/Tick bars for better statistical properties
- **Triple Barrier Method** - Path-dependent labeling with dynamic volatility barriers
- **Meta-Labeling Framework** - Two-stage approach separating direction from bet sizing
- **CUSUM Event Sampling** - Adaptive event detection for structural breaks
- **Sample Weight Calculation** - Handles overlapping labels with uniqueness weights
- **Clustered Feature Importance** - Hierarchical clustering for robust feature selection
- **Probabilistic Sharpe Ratio (PSR)** - Accounts for non-normality in returns
- **Deflated Sharpe Ratio (DSR)** - Adjusts for multiple testing / p-hacking
- **PurgedKFoldCV with 5% Embargo** - Eliminates serial correlation leakage
- **Feature Neutralization** - Market beta removal for alpha isolation
- **Winsorization Policy** - Preserves tail event information

### Version 3.0 Features (Previous):
- **Fractional Differentiation (FFD)** for stationary yet memory-preserving features
- **Hierarchical Risk Parity (HRP)** for robust portfolio optimization
- **Dynamic Transaction Cost Analysis** with Almgren-Chriss market impact
- **Numba JIT Acceleration** for 10-100x performance gains
- **Async Event-Driven Pipeline** for real-time trading
- **MLflow Experiment Tracking** for reproducible ML
- **DVC Data Versioning** for data pipeline management
- **SHAP Model Explainability** for regulatory compliance

---

## 🛠️ Tech Stack (Implemented)

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data Processing | pandas, numpy, scipy |
| ML/AI | XGBoost, LightGBM, CatBoost, PyTorch |
| Deep Learning | LSTM, Transformer, Attention |
| Technical Analysis | TA-Lib, pandas-ta |
| Broker API | Alpaca, Interactive Brokers |
| Database | PostgreSQL + TimescaleDB |
| Cache | Redis |
| Monitoring | Grafana, Prometheus |
| Deployment | Docker, docker-compose |
| **Performance** | **Numba JIT, asyncio, multiprocessing** |
| **MLOps** | **MLflow, DVC** |
| **Explainability** | **SHAP** |
| **Async** | **asyncio, aiohttp, websockets** |

---

## 📁 Project Structure (Complete)

```
alphatrade/
├── config/
│   ├── settings.yaml          ✅ Global configuration
│   ├── symbols.yaml           ✅ 46-stock universe
│   └── risk_params.yaml       ✅ Risk parameters
├── data/
│   ├── raw/                   ✅ 46 CSV files (15-min OHLCV)
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          ✅ Multi-asset data loader
│   │   ├── preprocessor.py    ✅ Data cleaning + Information-Driven Bars + Trading Hours Filter (v3.2)
│   │   ├── labeling.py        ✅ NEW: Triple Barrier + Meta-Labeling (v3.1)
│   │   ├── database.py        ✅ PostgreSQL/TimescaleDB/Redis
│   │   └── live_feed.py       ✅ WebSocket real-time feed
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py       ✅ 100+ technical indicators
│   │   ├── builder.py         ✅ Feature pipeline (200+ features)
│   │   ├── microstructure.py  ✅ Market microstructure
│   │   ├── cross_asset.py     ✅ Cross-asset analysis (integrated v3.2)
│   │   ├── regime.py          ✅ HMM regime detection (integrated v3.2)
│   │   └── alternative.py     ✅ Alternative data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py      ✅ Model registry & versioning
│   │   ├── ml_model.py        ✅ XGBoost/LightGBM/CatBoost/RF + MetaLabeling
│   │   ├── ensemble.py        ✅ Voting/Stacking/Blending
│   │   ├── deep_learning.py   ✅ LSTM/Transformer/Attention
│   │   ├── training.py        ✅ Walk-forward + PurgedKFoldCV 5% embargo + Clustered Feature Importance (v3.1)
│   │   └── explainability.py  ✅ SHAP-based model explanations
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base_strategy.py   ✅ Strategy framework
│   │   ├── momentum.py        ✅ Momentum & trend following
│   │   ├── mean_reversion.py  ✅ Mean reversion & pairs
│   │   └── ml_strategy.py     ✅ ML-based strategies
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizer.py  ✅ Kelly/Volatility/Risk Parity
│   │   ├── risk_manager.py    ✅ VaR/CVaR/Circuit breakers
│   │   └── portfolio.py       ✅ Portfolio optimization + HRP
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── broker_api.py      ✅ Alpaca & IBKR integration
│   │   ├── order_manager.py   ✅ Order lifecycle management
│   │   ├── executor.py        ✅ TWAP/VWAP/POV/Adaptive
│   │   └── async_pipeline.py  ✅ Async event-driven pipeline
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py          ✅ Event-driven + Dynamic TCA + Symbol-Specific Costs (v3.2)
│   │   └── metrics.py         ✅ Performance attribution + PSR/DSR (v3.1)
│   ├── mlops/                  ✅ NEW - MLOps Module
│   │   ├── __init__.py
│   │   ├── experiment_tracking.py ✅ MLflow integration
│   │   └── dvc_config.py      ✅ Data version control
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          ✅ Institutional logging
│       ├── helpers.py         ✅ Utility functions
│       └── numba_accelerators.py ✅ JIT-compiled functions
├── scripts/
│   ├── init_db.sql                      ✅ Database schema
│   ├── train_models.py                  ✅ AFML Institutional Training Pipeline + Full Feature Integration (v3.2)
│   ├── data_quality_pipeline.py         ✅ NEW v3.3: Trading hours, volume anomalies, OHLC validation
│   ├── calibrate_triple_barrier.py      ✅ NEW v3.3: Per-symbol ATR-based barrier calibration
│   ├── setup_validation.py              ✅ NEW v3.3: Embargo verification, holdout data setup
│   ├── optimize_features.py             ✅ NEW v3.3: Feature correlation/clustering, regime features
│   ├── update_symbol_params.py          ✅ NEW v3.3: Calculate spread, volume, beta per symbol
│   └── run_pre_training_validation.py   ✅ NEW v3.3: Master validation script (runs all checks)
├── monitoring/
│   └── prometheus/
│       └── prometheus.yml     ✅ Metrics config
├── models/                    📁 Trained models (generated)
├── results/                   📁 Backtest results (generated)
├── logs/                      📁 Log files (generated)
├── notebooks/                 📁 Research notebooks
├── main.py                    ✅ Main orchestrator
├── Dockerfile                 ✅ Production container
├── Dockerfile.jupyter         ✅ Research environment
├── docker-compose.yaml        ✅ Full stack deployment
├── requirements.txt           ✅ Production dependencies
├── requirements-research.txt  ✅ Research dependencies
├── setup.py                   ✅ Package setup
├── .env.example              ✅ Environment template
└── .gitignore                ✅ Git ignore rules
```

---

## 🚀 QUICK START GUIDE

### Step 1: Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env file with your credentials:
#    - ALPACA_API_KEY
#    - ALPACA_API_SECRET
#    - POSTGRES_PASSWORD
```

### Step 3: Verify Data

```bash
# Check that CSV files are in data/raw/
ls data/raw/
# Should show: AAPL_15min.csv, MSFT_15min.csv, etc.
```

### Step 4: Run Backtest (First Test)

```bash
python main.py --mode backtest
```

### Step 5: Train ML Models (Optional)

```bash
python scripts/train_models.py
```

### Step 6: Run Paper Trading

```bash
python main.py --mode paper
```

### Step 7: Docker Deployment (Production)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f alphatrade

# Stop services
docker-compose down
```

---

## 📋 EXECUTION ORDER FOR AI AGENT

### Phase 1: Initial Setup & Verification
```
Order | File/Command | Purpose
------|--------------|--------
1     | pip install -r requirements.txt | Install all dependencies
2     | Verify data/raw/*.csv exists | Check 46 CSV files present
3     | python -c "from src.utils.logger import get_logger" | Test imports
```

### Phase 2: Data Pipeline Test
```
Order | File/Command | Purpose
------|--------------|--------
4     | python -c "from src.data.loader import DataLoader; dl = DataLoader(); print(dl.load('AAPL').head())" | Test data loading
5     | python -c "from src.data.preprocessor import DataPreprocessor; dp = DataPreprocessor()" | Test preprocessor
```

### Phase 3: Feature Engineering Test
```
Order | File/Command | Purpose
------|--------------|--------
6     | python -c "from src.features.technical import TechnicalIndicators; ti = TechnicalIndicators()" | Test indicators
7     | python -c "from src.features.builder import FeatureBuilder; fb = FeatureBuilder()" | Test feature builder
```

### Phase 4: Run Backtest
```
Order | File/Command | Purpose
------|--------------|--------
8     | python main.py --mode backtest | Full backtest run
```

### Phase 5: Train Models (Optional but Recommended)
```
Order | File/Command | Purpose
------|--------------|--------
9     | python scripts/train_models.py | Train XGBoost, LightGBM, CatBoost
10    | Check models/ directory | Verify model files created
```

### Phase 6: Paper Trading
```
Order | File/Command | Purpose
------|--------------|--------
11    | Set ALPACA_API_KEY in .env | Configure broker
12    | Set ALPACA_API_SECRET in .env | Configure broker
13    | python main.py --mode paper | Start paper trading
```

### Phase 7: Production Deployment (Docker)
```
Order | File/Command | Purpose
------|--------------|--------
14    | docker-compose up -d postgres redis | Start databases
15    | docker-compose up -d alphatrade | Start trading system
16    | docker-compose up -d grafana | Start monitoring
17    | Access http://localhost:3000 | View Grafana dashboard
```

---

## 🔧 COMPONENT DETAILS

### Data Layer (`src/data/`)

| File | Features |
|------|----------|
| `loader.py` | Multi-asset parallel loading, CSV/Parquet/API support |
| `preprocessor.py` | Gap filling, Winsorization (not drop), **Information-Driven Bars (Volume/Dollar/Tick)** |
| `labeling.py` | **NEW v3.1:** Triple Barrier Method, Meta-Labeling, CUSUM Filter, Sample Weights |
| `database.py` | TimescaleDB hypertables, Redis caching |
| `live_feed.py` | Alpaca/Polygon WebSocket, bar aggregation |

### Feature Engineering (`src/features/`)

| File | Features |
|------|----------|
| `technical.py` | 100+ indicators: SMA, EMA, RSI, MACD, Bollinger, Ichimoku, ATR, etc. |
| `builder.py` | 200+ total features, automatic feature selection, **Triple Barrier Method**, **Fractional Differentiation** |
| `microstructure.py` | Kyle's Lambda, VPIN, Amihud illiquidity, Roll spread, **Level 2 Order Book Features** |
| `cross_asset.py` | Rolling correlations, beta, sector momentum |
| `regime.py` | HMM-based regime detection (bull/bear/sideways) |
| `alternative.py` | Sentiment, economic indicators, options-derived |

### ML Models (`src/models/`)

| File | Features |
|------|----------|
| `ml_model.py` | XGBoost, LightGBM, CatBoost, RandomForest with GPU, **MetaLabelingModel** |
| `ensemble.py` | VotingEnsemble, StackingEnsemble, BlendingEnsemble |
| `deep_learning.py` | Bidirectional LSTM, Transformer with attention |
| `training.py` | Walk-forward, Optuna, **PurgedKFoldCV (5% embargo)**, **Combinatorial Purged CV**, **Clustered Feature Importance (MDI/MDA)** |
| `explainability.py` | **SHAP-based explanations**, feature importance, waterfall/force plots |

### Strategy Framework (`src/strategy/`)

| File | Features |
|------|----------|
| `momentum.py` | Multi-timeframe momentum, breakout detection |
| `mean_reversion.py` | Z-score mean reversion, pairs trading with cointegration |
| `ml_strategy.py` | ML signal generation, confidence thresholds |

### Risk Management (`src/risk/`)

| File | Features |
|------|----------|
| `position_sizer.py` | Kelly Criterion, Volatility-based, Risk Parity, Optimal-F |
| `risk_manager.py` | VaR (95%, 99%), CVaR, circuit breakers, pre-trade checks |
| `portfolio.py` | MVO, Black-Litterman, Maximum Diversification, **Hierarchical Risk Parity (HRP)** |

### Execution (`src/execution/`)

| File | Features |
|------|----------|
| `broker_api.py` | Alpaca REST + WebSocket, IBKR TWS API |
| `order_manager.py` | Order lifecycle, smart order routing |
| `executor.py` | TWAP, VWAP, POV, Adaptive execution algorithms |
| `async_pipeline.py` | **Async event-driven pipeline**, priority queues, parallel workers |

### Backtesting (`src/backtest/`)

| File | Features |
|------|----------|
| `engine.py` | Event-driven + vectorized, realistic fills, slippage, **Dynamic Transaction Cost Analysis (Almgren-Chriss)** |
| `metrics.py` | Sharpe, Sortino, Calmar, Max DD, attribution, **Probabilistic SR (PSR)**, **Deflated SR (DSR)**, **Minimum Track Record Length** |

### MLOps (`src/mlops/`) - NEW

| File | Features |
|------|----------|
| `experiment_tracking.py` | **MLflow integration**, experiment management, model registry, artifact logging |
| `dvc_config.py` | **DVC data versioning**, pipeline management, remote storage, data lineage |

### Performance Utils (`src/utils/`)

| File | Features |
|------|----------|
| `numba_accelerators.py` | **Numba JIT-compiled** indicators (EMA, RSI, ATR, MACD), FFD, triple barrier, rolling stats |
| `logger.py` | Institutional-grade logging with rotation |
| `helpers.py` | Utility functions |

---

## ⚠️ IMPORTANT NOTES

### Before Running Live:
1. ✅ Test thoroughly with backtest mode
2. ✅ Run paper trading for at least 1 week
3. ✅ Verify all risk limits are correctly set
4. ✅ Check broker API credentials
5. ✅ Monitor logs for errors

### Risk Defaults (config/risk_params.yaml):
- Max position: 10% of portfolio
- Max sector: 30% of portfolio
- Max drawdown: 15%
- Daily loss limit: 3%
- Circuit breaker: 3% intraday loss

### Required API Keys:
- Alpaca API Key & Secret (for paper/live trading)
- Optional: Polygon API Key (for additional data)

---

## 📊 Monitoring URLs (After Docker Deploy)

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | - |
| Jupyter | http://localhost:8888 | Token in .env |
| PostgreSQL | localhost:5432 | trading / (see .env) |
| Redis | localhost:6379 | - |

---

## 🎓 Next Steps for Enhancement

1. **Add More Strategies**: Implement sector rotation, factor investing
2. **Enhance ML**: Add reinforcement learning, online learning
3. **Options Trading**: Extend to options strategies
4. **Multi-Asset**: Add crypto, forex support
5. **Cloud Deployment**: AWS/GCP with auto-scaling

---

## 📈 VERSION 3.0 IMPLEMENTATION DETAILS

### Phase 1: Advanced Data Science (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Fractional Differentiation | ✅ | Fixed-window FFD with ADF test for stationarity |
| Level 2 Microstructure | ✅ | Order book imbalance, depth, spread analysis |
| Feature Neutralization | ✅ | Cross-sectional neutralization, sector-relative features |
| Robust Outlier Handling | ✅ | Winsorization, MAD-based detection |

### Phase 2: Institutional Labeling (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Triple Barrier Method | ✅ | Volatility-based barriers, asymmetric targets (AFML Ch.3) |
| Meta-Labeling Framework | ✅ | Secondary model filtering, Kelly criterion bet sizing |
| Purged K-Fold CV | ✅ | Embargo periods, combinatorial CV, group-aware purging |

### Phase 3: Portfolio & Risk Management (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Hierarchical Risk Parity | ✅ | Quasi-diagonalization, recursive bisection, rolling HRP |
| Dynamic Transaction Costs | ✅ | Almgren-Chriss market impact, optimal execution scheduling |

### Phase 4: Infrastructure & Performance (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Numba JIT Acceleration | ✅ | 10-100x speedup for indicators, FFD, triple barrier |
| Async Trading Pipeline | ✅ | Event-driven, priority queues, parallel feature computation |

### Phase 5: MLOps & Explainability (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| MLflow Integration | ✅ | Experiment tracking, model registry, artifact management |
| DVC Data Versioning | ✅ | Data pipelines, remote storage, reproducibility |
| SHAP Explainability | ✅ | Feature importance, waterfall plots, regime explanations |

### Phase 6: AFML Institutional Methodology v3.1 (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Information-Driven Bars | ✅ | Volume/Dollar/Tick bars - better statistical properties than time bars |
| Triple Barrier Method | ✅ | Path-dependent labeling with dynamic volatility barriers (AFML Ch.3) |
| Meta-Labeling | ✅ | Two-stage approach: primary model (direction) + secondary model (bet sizing) |
| CUSUM Event Filter | ✅ | Adaptive event sampling for structural break detection |
| Sample Weight Calculation | ✅ | Uniqueness weights + time decay for overlapping labels |
| Clustered Feature Importance | ✅ | Hierarchical clustering + MDI/MDA at cluster level (AFML Ch.8) |
| Probabilistic Sharpe Ratio | ✅ | Accounts for skewness/kurtosis in returns distribution |
| Deflated Sharpe Ratio | ✅ | Adjusts for multiple testing / p-hacking bias |
| PurgedKFoldCV 5% Embargo | ✅ | Minimum 5% embargo to eliminate serial correlation leakage |
| Feature Neutralization | ✅ | Market beta removal, microstructure feature downweighting |
| Winsorization Policy | ✅ | Default outlier handling preserves tail event information |

### Phase 7: Enhanced Training Pipeline v3.2 (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Automatic Symbol Loading | ✅ | Extracts all 46 symbols from sectors in symbols.yaml automatically |
| Trading Hours Filter | ✅ | `TradingHoursFilter` class filters to US regular hours (9:30-16:00 ET) |
| Cross-Asset Features | ✅ | `CrossAssetFeatures` integrated: correlations, sector momentum, beta, factor exposures |
| Regime Detection | ✅ | `RegimeDetector` integrated: HMM, volatility regime, trend regime features |
| Combined Sample Weights | ✅ | `combine_weights()`: uniqueness + time decay with configurable alpha |
| Dynamic Embargo | ✅ | `calculate_dynamic_embargo()`: sized based on max feature lookback (200+ periods) |
| Cross-Sectional Features | ✅ | `add_cross_sectional_features()`: ranks, z-scores, sector-relative metrics |
| Symbol-Specific Transaction Costs | ✅ | `BacktestConfig.symbol_spread_bps`: uses spread_bps from symbols.yaml |

### Phase 8: Pre-Training Validation Pipeline v3.3 (COMPLETED ✅)

| Component | Status | Description |
|-----------|--------|-------------|
| Data Quality Pipeline | ✅ | `data_quality_pipeline.py`: Trading hours filtering, OHLC validation, volume anomaly detection |
| Triple Barrier Calibration | ✅ | `calibrate_triple_barrier.py`: Per-symbol ATR-based barriers, VIX adjustment, optimal holding |
| Label Quality Validation | ✅ | Class distribution check (25-40%), autocorrelation < 0.1, barrier touch analysis |
| Embargo Verification | ✅ | `setup_validation.py`: Max lookback calculation (200 bars), minimum 5% embargo |
| Holdout Data Setup | ✅ | 3-month temporal holdout, 6-symbol holdout (2/sector), stress period marking |
| Feature Optimization | ✅ | `optimize_features.py`: Correlation clustering, redundancy removal, 60-80 optimal features |
| Regime Features | ✅ | VIX regime (low/normal/high/extreme), trend regime, volatility regime |
| Symbol Parameters | ✅ | `update_symbol_params.py`: Real spread, daily volume, beta to SPY calculation |
| Master Validation | ✅ | `run_pre_training_validation.py`: Runs all checks, generates pass/fail report |

---

## 🔬 ADVANCED FEATURES USAGE

### Triple Barrier Method (v3.1)
```python
from src.data.labeling import TripleBarrierLabeler, TripleBarrierConfig

config = TripleBarrierConfig(
    pt_sl_ratio=(1.0, 1.0),      # Symmetric barriers
    volatility_lookback=20,      # EWM volatility window
    max_holding_period=10,       # Max bars to hold
    min_return=0.001             # Minimum return threshold
)

labeler = TripleBarrierLabeler(config)
events = labeler.get_events_with_ohlcv(
    prices=df,                   # OHLCV DataFrame
    pt_sl=(1.0, 1.0)            # Profit/StopLoss multipliers
)
# events contains: label, bin_label, t1, ret, touch_type
```

### Meta-Labeling with Bet Sizing (v3.1)
```python
from src.data.labeling import MetaLabeler, MetaLabelingConfig

config = MetaLabelingConfig(
    primary_threshold=0.5,
    use_probability=True
)
meta_labeler = MetaLabeler(config)

# Get side from primary model
side = meta_labeler.get_primary_side(primary_predictions)

# Create meta-labels
meta_events = meta_labeler.create_meta_labels(triple_barrier_events, side)

# Prepare training data for secondary model
X_meta, y_meta = meta_labeler.get_meta_training_data(features, meta_events)
```

### Information-Driven Bars (v3.1)
```python
from src.data.preprocessor import convert_time_bars_to_information_bars

# Convert 15-min bars to dollar bars
dollar_bars = convert_time_bars_to_information_bars(
    time_bars=df,
    bar_type="dollar",           # "volume", "dollar", or "tick"
    target_bars_per_day=50       # Auto-estimate threshold
)
# Returns IID-normal distributed returns with lower serial correlation
```

### Purged Cross-Validation with 5% Embargo (v3.1)
```python
from src.models.training import CrossValidationTrainer

cv_trainer = CrossValidationTrainer(
    cv_method='purged_kfold',
    n_splits=5,
    purge_gap=0,
    embargo_pct=0.05             # Minimum 5% per AFML recommendations
)
results = cv_trainer.cross_validate(model, X, y, t1=events['t1'])
```

### Clustered Feature Importance (v3.1)
```python
from src.models.training import feature_importance_with_clustering

result = feature_importance_with_clustering(
    model=fitted_model,
    X=features,
    y=labels,
    n_clusters=None,             # Auto-determine via silhouette
    method='mda',                # 'mda' or 'mdi'
    n_iterations=10
)
# Returns: cluster_importance, feature_importance, clusters, selected_features
```

### Probabilistic & Deflated Sharpe Ratio (v3.1)
```python
from src.backtest.metrics import SharpeRatioStatistics

sr_stats = SharpeRatioStatistics(periods_per_year=252)
report = sr_stats.generate_sharpe_report(
    returns=strategy_returns,
    n_trials=10,                 # Number of backtests run
    sr_benchmark=0.0,
    confidence=0.95
)
# report contains: sharpe_ratio, probabilistic_sr, deflated_sr,
# minimum_track_record, confidence_interval, interpretation
```

### Hierarchical Risk Parity
```python
from src.risk.portfolio import HierarchicalRiskParity

hrp = HierarchicalRiskParity()
weights = hrp.optimize(returns_df)
# Or rolling optimization
rolling_weights = hrp.rolling_optimize(returns_df, window=252)
```

### Async Trading Pipeline
```python
from src.execution.async_pipeline import PipelineBuilder

pipeline = (PipelineBuilder()
    .with_data_source('alpaca', symbols=['AAPL', 'MSFT'])
    .with_feature_builder(feature_builder)
    .with_model(ml_model)
    .with_risk_manager(risk_manager)
    .with_broker(alpaca_broker)
    .build())

await pipeline.start()
```

### MLflow Experiment Tracking
```python
from src.mlops.experiment_tracking import MLflowTracker

tracker = MLflowTracker(experiment_name='strategy_v3')
with tracker.start_run(run_name='lgbm_triple_barrier'):
    tracker.log_params(model_params)
    tracker.log_metrics(backtest_results)
    tracker.log_model(model, 'lightgbm')
```

### SHAP Explainability
```python
from src.models.explainability import TradingExplainer

explainer = TradingExplainer(model, X_train, feature_names)
explainer.generate_report(X_test, output_dir='reports/shap')
regime_analysis = explainer.explain_by_regime(X_test, regimes)
```

---

## 🆕 VERSION 3.2 ENHANCED FEATURES USAGE

### Trading Hours Filter (v3.2)
```python
from src.data.preprocessor import TradingHoursFilter

# Filter to regular US trading hours only
filter = TradingHoursFilter(
    market_open="09:30",
    market_close="16:00",
    timezone="America/New_York",
    include_extended_hours=False  # Set True for pre/post market
)

# Single DataFrame
df_filtered = filter.filter(df)

# Multiple symbols
filtered_data = filter.filter_multi({"AAPL": df_aapl, "MSFT": df_msft})

# Check if timestamp is in trading hours
is_open = filter.is_trading_hours(datetime.now())
```

### Cross-Sectional Features (v3.2)
```python
from scripts.train_models import add_cross_sectional_features

# Add cross-sectional ranking features
symbol_data = {
    "AAPL": df_aapl,
    "MSFT": df_msft,
    "GOOGL": df_googl
}

sector_mapping = {
    "Technology": ["AAPL", "MSFT", "GOOGL"]
}

enhanced_data = add_cross_sectional_features(symbol_data, sector_mapping)
# Each DataFrame now contains:
#   cs_return_rank, cs_return_zscore, cs_volume_rank,
#   cs_volatility_pctl, cs_momentum_rank, cs_sector_rank,
#   cs_vs_sector_return, cs_vs_sector_momentum,
#   cs_distance_from_max, cs_distance_from_min
```

### Dynamic Embargo Calculation (v3.2)
```python
from scripts.train_models import calculate_dynamic_embargo

# Automatically size embargo based on feature lookbacks
embargo_pct = calculate_dynamic_embargo(
    feature_columns=X.columns.tolist(),
    data_frequency_minutes=15,    # 15-minute bars
    min_embargo_pct=0.05          # AFML minimum 5%
)
# Returns e.g., 6.4% if max lookback is 200 periods
```

### Combined Sample Weights (v3.2)
```python
from src.data.labeling import get_sample_weights, get_time_decay_weights, combine_weights

# Uniqueness weights for overlapping labels
uniqueness = get_sample_weights(events, close_prices)

# Time decay weights (recent samples get more weight)
time_decay = get_time_decay_weights(events, c=0.5)

# Combine with 50/50 weighting
combined = combine_weights(uniqueness, time_decay, alpha=0.5)
```

### Symbol-Specific Transaction Costs (v3.2)
```python
from src.backtest.engine import BacktestConfig, load_symbol_spreads_from_config
import yaml

# Load symbol spreads from config
with open('config/symbols.yaml') as f:
    symbols_config = yaml.safe_load(f)

spreads = load_symbol_spreads_from_config(symbols_config)
# Returns: {"AAPL": 1.5, "MSFT": 1.2, "SPY": 0.5, ...}

# Create config with symbol-specific costs
config = BacktestConfig(
    slippage_bps=5,              # Default fallback
    symbol_spread_bps=spreads   # Symbol-specific spreads
)

# Or use the factory function
from src.backtest.engine import create_backtest_config_from_yaml
config = create_backtest_config_from_yaml(settings_config, symbols_config)
```

### Full Training Pipeline (v3.2)
```python
from scripts.train_models import prepare_data

# Load symbols.yaml for sector mapping
symbols_config = load_config("config/symbols.yaml")

# Prepare data with all v3.2 enhancements
features_df, labels, weights, events = prepare_data(
    symbols=["AAPL", "MSFT", "GOOGL"],
    use_information_bars=True,
    use_triple_barrier=True,
    neutralize=True,
    filter_trading_hours=True,      # v3.2: Remove pre/post market
    include_extended_hours=False,
    use_cross_asset_features=True,  # v3.2: Correlations, sector momentum
    use_regime_features=True,       # v3.2: HMM regime, volatility regime
    use_time_decay_weights=True,    # v3.2: Combined sample weights
    time_decay_factor=0.5,
    symbols_config=symbols_config   # v3.2: For sector mapping
)
# Returns enhanced features including cross-sectional ranks,
# regime features, and properly weighted samples
```

---

---

## 🆕 VERSION 3.3 PRE-TRAINING VALIDATION - STEP BY STEP EXECUTION GUIDE

Bu bölüm, model eğitimi öncesi TÜM veri doğrulama adımlarını sırasıyla açıklar.
**ÖNEMLİ:** Bu adımları sırasıyla çalıştırın. Her adım bir öncekine bağlıdır.

---

### 📋 ÖZET: Çalıştırma Sırası

| Adım | Komut | Süre | Çıktı |
|------|-------|------|-------|
| 1 | `python scripts/data_quality_pipeline.py --analyze` | ~2 dk | Konsol raporu |
| 2 | `python scripts/data_quality_pipeline.py --process` | ~5 dk | `data/processed/*.csv` |
| 3 | `python scripts/calibrate_triple_barrier.py --calibrate` | ~10 dk | `config/triple_barrier_params.yaml` |
| 4 | `python scripts/setup_validation.py --verify-embargo` | ~1 dk | Embargo yüzdesi |
| 5 | `python scripts/setup_validation.py --setup-holdout` | ~2 dk | `config/holdout_manifest.yaml` |
| 6 | `python scripts/optimize_features.py --reduce` | ~2 dk | `config/optimal_features.yaml` |
| 7 | `python scripts/optimize_features.py --add-regime` | ~1 dk | Regime feature raporu |
| 8 | `python scripts/update_symbol_params.py --calculate` | ~3 dk | Konsol raporu |
| 9 | `python scripts/update_symbol_params.py --update` | ~1 dk | `config/symbols.yaml` güncellenir |
| 10 | `python scripts/run_pre_training_validation.py --all` | ~5 dk | `pre_training_validation_report.json` |

---

### ▶️ ADIM 1: Data Quality Analysis (Analiz - Değişiklik Yapmaz)

**Amaç:** 46 sembolün veri kalitesini kontrol et (trading hours, OHLC, volume anomalies)

```bash
python scripts/data_quality_pipeline.py --analyze
```

**Beklenen Çıktı:**
- Her sembol için kalite skoru (0-100)
- Extended hours yüzdesi (hedef: <1%)
- OHLC violation sayısı (hedef: 0)
- Volume anomaly sayısı

**Başarı Kriteri:** Ortalama skor > 80

---

### ▶️ ADIM 2: Data Quality Processing (Temiz Veri Üretimi)

**Amaç:** Verileri temizle ve `data/processed/` klasörüne kaydet

```bash
python scripts/data_quality_pipeline.py --process
```

**Beklenen Çıktı:**
- `data/processed/{SYMBOL}_15min_clean.csv` dosyaları (46 adet)
- Sadece regular trading hours (09:30-16:00 ET)
- OHLC anomalileri düzeltilmiş
- Volume spike'ları işaretlenmiş

**Doğrulama:**
```bash
python scripts/data_quality_pipeline.py --validate
```

---

### ▶️ ADIM 3: Triple Barrier Calibration (Label Üretimi)

**Amaç:** Her sembol için optimal barrier parametreleri hesapla

```bash
python scripts/calibrate_triple_barrier.py --calibrate
```

**Beklenen Çıktı:**
- `config/triple_barrier_params.yaml` dosyası
- Her sembol için: pt_multiplier, sl_multiplier, max_holding_bars
- ATR-bazlı barrier genişlikleri

**Tek Sembol Analizi (opsiyonel):**
```bash
python scripts/calibrate_triple_barrier.py --analyze AAPL
python scripts/calibrate_triple_barrier.py --analyze MSFT
python scripts/calibrate_triple_barrier.py --analyze NVDA
```

**Label Kalite Doğrulama:**
```bash
python scripts/calibrate_triple_barrier.py --validate
```

**Başarı Kriterleri:**
- Her class %25-40 arasında
- Label autocorrelation < 0.1
- Vertical touch < %50

---

### ▶️ ADIM 4: Embargo Verification

**Amaç:** PurgedKFoldCV için minimum %5 embargo hesapla

```bash
python scripts/setup_validation.py --verify-embargo
```

**Beklenen Çıktı:**
- Max feature lookback: ~200 bars
- Calculated embargo: ~11% (200/1800 günlük bar)
- PASS/FAIL durumu

**Başarı Kriteri:** Embargo >= 5%

---

### ▶️ ADIM 5: Holdout Data Setup (KRİTİK - Sadece 1 Kez Çalıştır!)

**Amaç:** Test verisi ayır (temporal + symbol-based)

```bash
python scripts/setup_validation.py --setup-holdout
```

**Beklenen Çıktı:**
- `config/holdout_manifest.yaml` dosyası
- 3 aylık temporal holdout (son 3 ay)
- 6 sembol holdout (her sektörden 2)
- Stress period işaretleme (COVID, Fed meetings)

**UYARI:** Bu komutu sadece 1 KEZ çalıştırın! Tekrar çalıştırırsanız holdout değişir.

**Doğrulama:**
```bash
python scripts/setup_validation.py --verify-holdout
```

---

### ▶️ ADIM 6: Feature Optimization (Redundancy Removal)

**Amaç:** 68 teknik özelliği ~36 optimal özelliğe indir

```bash
python scripts/optimize_features.py --reduce
```

**Beklenen Çıktı:**
- `config/optimal_features.yaml` dosyası
- Original: 68 features
- Final: ~36 features (47% reduction)
- Correlation clustering raporu

**Başarı Kriteri:** Final feature count: 35-60

---

### ▶️ ADIM 7: Regime Feature Generation

**Amaç:** VIX, Trend ve Volatility regime özellikleri ekle

```bash
python scripts/optimize_features.py --add-regime --symbol AAPL
```

**Beklenen Çıktı:**
- 24 regime feature eklendi
- VIX regime dağılımı (Low/Normal/High/Extreme)
- Trend regime dağılımı (Strong Bear → Strong Bull)
- Volatility regime dağılımı

**Toplam Feature Sayısı:** 36 teknik + 24 regime = **60 feature**

---

### ▶️ ADIM 8: Symbol Parameters Calculation

**Amaç:** Her sembol için spread, volume, beta hesapla

```bash
python scripts/update_symbol_params.py --calculate
```

**Beklenen Çıktı:**
- 46 sembol için parametre tablosu
- Ortalama spread (bps)
- Günlük ortalama volume
- Beta to SPY

---

### ▶️ ADIM 9: Update symbols.yaml

**Amaç:** Hesaplanan parametreleri config dosyasına yaz

```bash
python scripts/update_symbol_params.py --update
```

**Beklenen Çıktı:**
- `config/symbols.yaml` güncellendi
- Her sembol için: spread_bps, avg_daily_volume, beta

**Doğrulama:**
```bash
python scripts/update_symbol_params.py --validate
```

---

### ▶️ ADIM 10: Final Validation (Tüm Kontroller)

**Amaç:** Tüm adımların başarılı olduğunu doğrula

```bash
python scripts/run_pre_training_validation.py --all
```

**Beklenen Çıktı:**
- `pre_training_validation_report.json` dosyası
- Her kontrol için PASS/FAIL durumu
- Özet rapor

---

### ✅ Başarı Kriterleri Tablosu

| Kriter | Hedef | Kontrol Komutu |
|--------|-------|----------------|
| Regular Hours Only | extended_hours < 1% | `--analyze` |
| OHLC Valid | violations = 0 | `--analyze` |
| Label Balance | Her class %25-40 | `--validate` (calibrate) |
| Label Autocorr | < 0.1 | `--validate` (calibrate) |
| Embargo | >= 5% | `--verify-embargo` |
| Holdout Reserved | Manifest exists | `--verify-holdout` |
| Technical Features | 35-60 | `--reduce` |
| Regime Features | 24 | `--add-regime` |
| Total Features | 55-85 | Manual check |
| Final Validation | All PASS | `--all` |

---

### 🚀 HIZLI BAŞLANGIÇ (Copy-Paste)

Tüm adımları sırasıyla çalıştırmak için:

```bash
# Adım 1-2: Data Quality
python scripts/data_quality_pipeline.py --analyze
python scripts/data_quality_pipeline.py --process

# Adım 3: Triple Barrier
python scripts/calibrate_triple_barrier.py --calibrate

# Adım 4-5: Embargo & Holdout
python scripts/setup_validation.py --verify-embargo
python scripts/setup_validation.py --setup-holdout

# Adım 6-7: Features
python scripts/optimize_features.py --reduce
python scripts/optimize_features.py --add-regime

# Adım 8-9: Symbol Parameters
python scripts/update_symbol_params.py --calculate
python scripts/update_symbol_params.py --update

# Adım 10: Final Check
python scripts/run_pre_training_validation.py --all
```

---

### ⚠️ Troubleshooting

**Hata: "Data file not found"**
- `data/raw/` klasöründe `{SYMBOL}_15min.csv` dosyaları olmalı
- Dosya adı formatı: `AAPL_15min.csv`

**Hata: "ModuleNotFoundError"**
- `pip install -r requirements.txt` çalıştırın
- Eksik modül: `pip install {module_name}`

**Hata: "Label imbalance"**
- Triple barrier parametrelerini ayarlayın
- `pt_sl_ratio` değerini değiştirin

**Hata: "Too few features"**
- `optimize_features.py` içinde `cluster_threshold` değerini artırın (0.92 → 0.95)

---

*Document Version: 3.3*
*Implementation Status: COMPLETE + FULL AFML INSTITUTIONAL METHODOLOGY + PRE-TRAINING VALIDATION PIPELINE*
*Ready for: Pre-Training Validation → Backtest → Paper Trading → Live Trading*
*v3.1 Features: Triple Barrier, Meta-Labeling, Information Bars, Clustered Importance, PSR/DSR, PurgedKFoldCV 5% Embargo*
*v3.2 Features: Trading Hours Filter, Cross-Asset Integration, Regime Features, Cross-Sectional Ranks, Dynamic Embargo, Combined Weights, Symbol-Specific Costs*
*v3.3 Features: Data Quality Pipeline, Barrier Calibration, Label Validation, Holdout Setup, Feature Optimization, Regime Awareness, Symbol Parameters*
