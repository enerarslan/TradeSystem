# AlphaTrade System

An **institutional-grade algorithmic trading platform** designed for systematic, multi-strategy trading. Built with production-quality code following **JPMorgan-level** design standards.

**Version 2.0.0** - Complete institutional upgrade based on comprehensive architectural audit.

## What's New in v2.0.0

### Critical Bug Fixes
- **Data Leakage Prevention**: Proper fit/transform separation in feature pipeline
- **Target Variable Leakage Fix**: Embargo period implementation for ML strategies
- **Purge Gap Calculation**: Dynamic calculation (horizon + lookback + buffer = 215)
- **Infinite Liquidity Fix**: Order book execution with partial fills enabled by default

### New Institutional Modules
- **Point-in-Time (PIT) Data Infrastructure**: Survivorship bias handling, as-of queries
- **Regime Detection**: HMM, GARCH volatility, correlation regimes, structural breaks
- **Microstructure Features**: OFI, VPIN, Kyle's Lambda, Roll Spread, Amihud
- **Black-Litterman Portfolio Optimization**: View-based portfolio construction
- **L2/L3 Order Book Support**: Full order book reconstruction and features
- **Hyperparameter Management**: YAML-based versioned hyperparameter storage

### Enhanced Metrics
- **Deflated Sharpe Ratio (DSR)**: Primary optimization metric with multiple testing correction
- **Probabilistic Sharpe Ratio (PSR)**: Statistical significance of Sharpe
- **Minimum Track Record Length (MinTRL)**: Required history for statistical validity

## Key Features

### Trading Strategies
- **Multi-Strategy Framework**: Momentum, mean reversion, volatility breakout, ML-based, and ensemble strategies
- **Signal Generation**: Composite signals with confidence scoring and attribution
- **Walk-Forward Training**: Time-series cross-validation with purging and embargo

### Machine Learning Pipeline
- **Model Factory**: LightGBM, XGBoost, CatBoost, Random Forest support
- **Hyperparameter Optimization**: Optuna integration with TPE sampler and pruning
- **Deep Learning**: LSTM with attention, Temporal Fusion Transformer (PyTorch Lightning)
- **Experiment Tracking**: MLflow integration for reproducibility
- **Model Registry**: Staging workflow (None -> Staging -> Production -> Archived)
- **Combinatorial Purged CV (CPCV)**: Mandatory default for institutional-grade validation

### Feature Engineering
- **Technical Indicators**: 50+ indicators (trend, momentum, volatility, volume)
- **Fractional Differentiation**: Memory-preserving stationarity transformations
- **Statistical Arbitrage**: Cointegration testing, Ornstein-Uhlenbeck estimation
- **Macroeconomic Features**: FRED integration (GDP, CPI, yields, spreads)
- **Feature Store**: Point-in-time feature serving with versioning
- **Microstructure Features**: Order flow imbalance, VPIN, Kyle's lambda
- **GARCH Volatility Models**: GARCH(1,1), EGARCH, GJR-GARCH

### Data Infrastructure
- **Point-in-Time Queries**: Historical data as it was known at any point
- **Survivorship Bias Handling**: UniverseManager tracks delistings, M&A
- **Corporate Action Adjustment**: Splits, dividends, spinoffs
- **L2/L3 Order Book**: Full depth data support with reconstruction

### Backtesting
- **Vectorized Engine**: High-performance numpy-based backtesting
- **Event-Driven Engine**: Microsecond-precision event simulation
- **Market Impact Models**: Almgren-Chriss optimal execution
- **Realistic Execution**: Partial fills, order rejection, latency simulation
- **Monte Carlo Analysis**: Block bootstrap, Probabilistic Sharpe Ratio

### Risk Management
- **Position Sizing**: Kelly, volatility targeting, risk parity
- **VaR/CVaR**: Historical, parametric, and Monte Carlo methods (99% confidence)
- **Drawdown Controls**: Automatic position scaling based on drawdown
- **Correlation Monitoring**: Rolling correlation and regime detection
- **ADV-Based Limits**: Maximum participation rate (2% of ADV)

### Portfolio Optimization
- **Mean-Variance**: Classic Markowitz optimization
- **Risk Parity**: Equal risk contribution
- **Hierarchical Risk Parity**: Clustering-based allocation
- **Black-Litterman**: View-based optimization with ML integration

### Regime Detection
- **HMM Regime Detector**: Hidden Markov Model for market state classification
- **Volatility Regimes**: GARCH-based volatility regime detection
- **Correlation Regimes**: Dynamic correlation breakdown detection
- **Structural Breaks**: CUSUM and Bai-Perron tests

### Reporting
- **Interactive Dashboards**: Plotly-based visualizations
- **Tear Sheets**: Comprehensive performance reports
- **Monthly Heatmaps**: Returns calendar view
- **Risk Attribution**: Factor exposure analysis

## Installation

### Prerequisites
- Python 3.11+
- Git

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd AlphaTrade_System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Install core dependencies
pip install -e .
```

### Optional Dependencies

```bash
# Deep learning (PyTorch)
pip install -e ".[deep]"

# MLflow experiment tracking
pip install -e ".[mlflow]"

# TimescaleDB support
pip install -e ".[timescale]"

# FRED macroeconomic data
pip install -e ".[macro]"

# Institutional features (GARCH, HMM, etc.)
pip install -e ".[institutional]"

# Development tools
pip install -e ".[dev]"

# Full installation (all features)
pip install -e ".[all]"
```

## Project Structure

```
AlphaTrade_System/
├── config/                          # Configuration files
│   ├── settings.py                  # Pydantic settings
│   ├── ml_config.yaml               # ML pipeline configuration
│   ├── trading_config.yaml          # Trading parameters
│   ├── strategy_params.yaml         # Strategy configuration
│   ├── risk_limits.yaml             # Risk management limits
│   ├── institutional_defaults.yaml  # Institutional-grade defaults
│   └── hyperparameters/             # Versioned hyperparameters
│       ├── lightgbm_defaults.yaml
│       └── xgboost_defaults.yaml
├── data/
│   ├── raw/                         # Raw OHLCV data
│   ├── processed/                   # Processed data
│   ├── features/                    # Feature store cache
│   └── cache/                       # General cache
├── src/
│   ├── data/                        # Data layer
│   │   ├── loaders/                 # Data loaders
│   │   ├── pit/                     # Point-in-Time infrastructure
│   │   │   ├── universe_manager.py  # Survivorship bias handling
│   │   │   ├── pit_loader.py        # PIT data loader
│   │   │   ├── corporate_actions.py # Split/dividend adjustment
│   │   │   └── as_of_query.py       # As-of timestamp queries
│   │   ├── orderbook/               # L2/L3 order book support
│   │   │   ├── order_book.py        # Order book data structures
│   │   │   ├── book_builder.py      # Book reconstruction
│   │   │   └── book_features.py     # Order book features
│   │   └── storage/                 # Storage backends
│   ├── features/                    # Feature engineering
│   │   ├── technical/               # Technical indicators
│   │   ├── regime/                  # Regime detection
│   │   │   ├── hmm_regime.py        # HMM regime detector
│   │   │   ├── volatility_regime.py # GARCH volatility regimes
│   │   │   ├── correlation_regime.py # Correlation breakdown
│   │   │   └── structural_breaks.py # CUSUM, Bai-Perron tests
│   │   ├── microstructure/          # Microstructure features
│   │   │   ├── order_flow_imbalance.py # OFI
│   │   │   ├── vpin.py              # Volume-Synchronized PIN
│   │   │   ├── kyle_lambda.py       # Kyle's lambda
│   │   │   └── roll_spread.py       # Roll spread, Amihud
│   │   ├── fractional_diff.py       # Fractional differentiation
│   │   ├── cointegration.py         # Statistical arbitrage
│   │   ├── macro_features.py        # FRED macroeconomic
│   │   └── pipeline.py              # Feature pipeline (fit/transform)
│   ├── training/                    # ML training module
│   │   ├── validation.py            # Purged CV, CPCV, Walk-Forward
│   │   ├── optimization.py          # Optuna optimization
│   │   ├── hyperparameters/         # Hyperparameter management
│   │   │   └── manager.py           # HyperparameterManager
│   │   └── deep_learning/           # Deep learning models
│   ├── validation/                  # Data validation
│   │   └── leakage_detector.py      # Feature leakage detection
│   ├── strategies/                  # Trading strategies
│   │   ├── ml_based/                # ML strategies
│   │   │   └── ml_alpha.py          # ML Alpha with embargo
│   │   └── ensemble.py              # Ensemble strategy
│   ├── risk/                        # Risk management
│   ├── portfolio/                   # Portfolio management
│   │   ├── optimizer.py             # MVO, Risk Parity, HRP
│   │   └── black_litterman.py       # Black-Litterman optimizer
│   ├── backtesting/                 # Backtesting engine
│   │   ├── engine.py                # Vectorized backtest
│   │   ├── event_engine.py          # Event-driven (order book exec)
│   │   ├── market_impact.py         # Almgren-Chriss model
│   │   └── metrics.py               # DSR, PSR, MinTRL
│   └── utils/                       # Utilities
├── tests/                           # Test suite
│   ├── unit/
│   │   ├── test_feature_leakage.py  # Leakage detection tests
│   │   ├── test_purged_cv.py        # Cross-validation tests
│   │   └── test_market_impact.py    # Market impact tests
│   └── integration/
├── main.py                          # Main entry point
├── run.py                           # Unified launcher
└── pyproject.toml                   # Project configuration
```

## Quick Start

### First Time Setup
```bash
python run.py setup
```

### Running a Full Pipeline
```bash
python run.py                    # Full pipeline (train + backtest)
python run.py backtest           # Backtest only
python run.py train              # Train ML model
python run.py train --deep       # Train deep learning
```

### Running with Institutional Defaults
```bash
python main.py --config config/institutional_defaults.yaml
```

### Backtest with Event-Driven Engine (Order Book Execution)
```bash
python main.py --mode backtest --engine event-driven
```

### ML Training with CPCV
```python
from src.training import Trainer, ModelFactory
from src.training.validation import CombinatorialPurgedKFoldCV

# Create model
model = ModelFactory.create("lightgbm")

# Setup CPCV (institutional standard)
cv = CombinatorialPurgedKFoldCV(
    n_splits=6,
    n_test_splits=2,
    purge_gap=215,  # horizon(5) + lookback(200) + buffer(10)
    embargo_pct=0.02,
)

# Train
trainer = Trainer(model=model, cv=cv)
result = trainer.train(X_train, y_train)
```

### Black-Litterman Portfolio Optimization
```python
from src.portfolio import (
    BlackLittermanOptimizer,
    create_absolute_view,
    create_relative_view,
)

optimizer = BlackLittermanOptimizer(
    risk_free_rate=0.02,
    tau=0.05,
)

# Add views
optimizer.add_view(create_absolute_view(
    asset="AAPL",
    expected_return=0.15,
    confidence=0.7,
))

optimizer.add_view(create_relative_view(
    long_asset="GOOGL",
    short_asset="META",
    expected_outperformance=0.03,
    confidence=0.5,
))

# Optimize
result = optimizer.optimize(returns)
print(result.optimal_weights)
```

### Regime Detection
```python
from src.features.regime import (
    HMMRegimeDetector,
    VolatilityRegimeDetector,
)

# HMM regime detection
hmm = HMMRegimeDetector(n_regimes=3)
regimes = hmm.fit_predict(returns)

# GARCH volatility regimes
vol_detector = VolatilityRegimeDetector()
vol_regimes = vol_detector.detect(returns)
```

### Microstructure Features
```python
from src.features.microstructure import (
    OrderFlowImbalance,
    VPIN,
    KyleLambda,
)

# Order Flow Imbalance
ofi = OrderFlowImbalance()
ofi_values = ofi.calculate(prices, volumes, trades)

# VPIN
vpin = VPIN(bucket_size=1000)
vpin_values = vpin.calculate(trades)

# Kyle's Lambda
kyle = KyleLambda()
lambda_values = kyle.estimate(returns, volumes)
```

## Institutional Configuration

### Key Settings (config/institutional_defaults.yaml)

```yaml
data:
  point_in_time: true
  survivorship_bias_correction: true
  corporate_action_adjustment: true
  min_adv_filter: 1_000_000

features:
  max_lookback_periods: 200
  microstructure_enabled: true
  garch_volatility: true
  leakage_check: strict

cross_validation:
  type: combinatorial_purged_kfold
  n_splits: 6
  n_test_splits: 2
  purge_gap: auto  # Calculated: horizon + lookback + buffer
  embargo_pct: auto
  min_train_samples: 1000

optimization:
  primary_metric: deflated_sharpe_ratio
  multiple_testing_correction: true
  min_track_record_months: 12

backtesting:
  execution_simulator: order_book
  partial_fills: true
  max_participation_rate: 0.02
  latency_ms: 50
  rejection_rate: 0.02

risk:
  var_confidence: 0.99
  max_position_adv_pct: 5.0
  max_sector_exposure: 0.25
  drawdown_flatten_threshold: 0.15
```

## Performance Metrics

### Statistical Metrics
- **Deflated Sharpe Ratio (DSR)**: Adjusts for multiple testing bias
- **Probabilistic Sharpe Ratio (PSR)**: Probability of true Sharpe > benchmark
- **Minimum Track Record Length**: Required history for statistical significance

### Return Metrics
- Total Return, CAGR, MTD/QTD/YTD
- Best/Worst Day, Month, Year

### Risk-Adjusted Metrics
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Omega Ratio, Information Ratio

### Risk Metrics
- Volatility, Max Drawdown, Avg Drawdown
- VaR (95%, 99%), CVaR (Expected Shortfall)
- Skewness, Kurtosis

### Trading Metrics
- Win Rate, Profit Factor
- Average Win/Loss, Max Consecutive Wins/Losses

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run leakage detection tests
pytest tests/unit/test_feature_leakage.py -v

# Run purged CV tests
pytest tests/unit/test_purged_cv.py -v

# Run market impact tests
pytest tests/unit/test_market_impact.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Audit Compliance

This system has been audited against institutional standards. All critical issues identified in `audit_report.md` have been addressed:

| Issue | Status |
|-------|--------|
| Data leakage in feature pipeline | FIXED |
| Target variable construction leakage | FIXED |
| Purged CV not enforced in optimization | FIXED |
| Infinite liquidity assumption | FIXED |
| No survivorship bias handling | FIXED |
| Missing PIT data infrastructure | IMPLEMENTED |
| Missing regime detection | IMPLEMENTED |
| Missing microstructure features | IMPLEMENTED |
| Missing test coverage | IMPLEMENTED |
| Magic numbers in code | EXTERNALIZED |

## Environment Variables

```bash
# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=alphatrade

# FRED API (for macro features)
FRED_API_KEY=your_api_key_here

# TimescaleDB (optional)
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432
TIMESCALE_DB=alphatrade
TIMESCALE_USER=postgres
TIMESCALE_PASSWORD=password
```

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Black, F. and Litterman, R. (1992). *Global Portfolio Optimization*
- Cartea, A. et al. (2015). *Algorithmic and High-Frequency Trading*
- Lehalle, C.A. and Laruelle, S. (2018). *Market Microstructure in Practice*

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request
