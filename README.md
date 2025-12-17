# AlphaTrade System

An institutional-grade algorithmic trading platform designed for systematic, multi-strategy trading. Built with production-quality code following JPMorgan-level design standards.

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

### Feature Engineering
- **Technical Indicators**: 50+ indicators (trend, momentum, volatility, volume)
- **Fractional Differentiation**: Memory-preserving stationarity transformations
- **Statistical Arbitrage**: Cointegration testing, Ornstein-Uhlenbeck estimation
- **Macroeconomic Features**: FRED integration (GDP, CPI, yields, spreads)
- **Feature Store**: Point-in-time feature serving with versioning

### Backtesting
- **Vectorized Engine**: High-performance numpy-based backtesting
- **Event-Driven Engine**: Microsecond-precision event simulation
- **Market Impact Models**: Almgren-Chriss optimal execution
- **Monte Carlo Analysis**: Block bootstrap, Probabilistic Sharpe Ratio

### Risk Management
- **Position Sizing**: Kelly, volatility targeting, risk parity
- **VaR/CVaR**: Historical, parametric, and Monte Carlo methods
- **Drawdown Controls**: Automatic position scaling based on drawdown
- **Correlation Monitoring**: Rolling correlation and regime detection

### Portfolio Optimization
- **Mean-Variance**: Classic Markowitz optimization
- **Risk Parity**: Equal risk contribution
- **Hierarchical Risk Parity**: Clustering-based allocation
- **Black-Litterman**: View-based optimization

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
│   └── logging_config.yaml          # Logging configuration
├── data/
│   ├── raw/                         # Raw OHLCV data
│   ├── processed/                   # Processed data
│   ├── features/                    # Feature store cache
│   └── cache/                       # General cache
├── scripts/
│   └── migrate_csv_to_timescale.py  # Data migration script
├── src/
│   ├── data/                        # Data layer
│   │   ├── loaders/                 # Data loaders
│   │   │   ├── data_loader.py       # CSV/Parquet loader
│   │   │   └── polars_loader.py     # High-performance Polars loader
│   │   ├── validators/              # Data validation
│   │   ├── processors/              # Data processing
│   │   └── storage/                 # Storage backends
│   │       ├── cache.py             # In-memory caching
│   │       └── timescale_client.py  # TimescaleDB client
│   ├── features/                    # Feature engineering
│   │   ├── technical/               # Technical indicators
│   │   │   └── indicators.py        # 50+ indicators
│   │   ├── fractional_diff.py       # Fractional differentiation
│   │   ├── cointegration.py         # Statistical arbitrage features
│   │   ├── macro_features.py        # FRED macroeconomic features
│   │   ├── feature_store.py         # Feature store implementation
│   │   └── pipeline.py              # Feature pipeline
│   ├── training/                    # ML training module
│   │   ├── experiment_tracker.py    # MLflow integration
│   │   ├── model_registry.py        # Model lifecycle management
│   │   ├── model_factory.py         # Standardized model creation
│   │   ├── trainer.py               # Training orchestrator
│   │   ├── validation.py            # Purged K-Fold CV
│   │   ├── optimization.py          # Optuna hyperparameter optimization
│   │   └── deep_learning/           # Deep learning models
│   │       ├── losses.py            # Financial loss functions
│   │       ├── lstm.py              # LSTM with attention
│   │       └── transformer.py       # Temporal Fusion Transformer
│   ├── strategies/                  # Trading strategies
│   │   ├── base.py                  # Base strategy class
│   │   ├── momentum/                # Momentum strategies
│   │   ├── mean_reversion/          # Mean reversion strategies
│   │   ├── ml_based/                # ML strategies
│   │   ├── multi_factor/            # Multi-factor strategies
│   │   └── ensemble.py              # Ensemble strategy
│   ├── risk/                        # Risk management
│   │   ├── position_sizing.py       # Position sizing algorithms
│   │   ├── var_models.py            # VaR/CVaR models
│   │   ├── drawdown.py              # Drawdown controls
│   │   └── correlation.py           # Correlation analysis
│   ├── portfolio/                   # Portfolio management
│   │   ├── optimizer.py             # Portfolio optimization
│   │   ├── rebalancer.py            # Rebalancing logic
│   │   └── allocation.py            # Asset allocation
│   ├── backtesting/                 # Backtesting engine
│   │   ├── engine.py                # Vectorized backtest engine
│   │   ├── event_engine.py          # Event-driven backtest engine
│   │   ├── events/                  # Event system
│   │   │   ├── base.py              # Base event classes
│   │   │   ├── market.py            # Market events (tick, bar, order book)
│   │   │   ├── signal.py            # Signal events
│   │   │   ├── order.py             # Order events
│   │   │   ├── fill.py              # Fill events
│   │   │   └── queue.py             # Event queue and dispatcher
│   │   ├── market_impact.py         # Almgren-Chriss model
│   │   ├── monte_carlo.py           # Monte Carlo analysis
│   │   ├── metrics.py               # Performance metrics
│   │   ├── analysis.py              # Result analysis
│   │   └── reports/                 # Report generation
│   │       ├── report_generator.py  # HTML reports
│   │       └── dashboard.py         # Interactive dashboards
│   ├── execution/                   # Order execution
│   │   ├── order_manager.py         # Order management
│   │   ├── slippage.py              # Slippage models
│   │   └── transaction_cost.py      # Transaction cost models
│   └── utils/                       # Utilities
│       ├── logger.py                # Logging utilities
│       ├── decorators.py            # Common decorators
│       └── helpers.py               # Helper functions
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
├── main.py                          # Main entry point
└── pyproject.toml                   # Project configuration
```

## Quick Start

### Running a Backtest (Vectorized)

```python
from src.data.loaders.data_loader import DataLoader
from src.strategies.momentum.multi_factor_momentum import MultiFactorMomentumStrategy
from src.backtesting.engine import BacktestEngine

# Load data
loader = DataLoader(data_path="data/raw")
data = loader.load_all()

# Create strategy
strategy = MultiFactorMomentumStrategy(params={
    "lookback_periods": [5, 10, 20],
    "top_n_long": 5,
})

# Run backtest
engine = BacktestEngine(
    initial_capital=1_000_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
)
result = engine.run(strategy, data)

# Print metrics
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

### Event-Driven Backtesting

```python
from src.backtesting import EventDrivenEngine, EventEngineConfig
from src.backtesting.events import BarEvent

# Configure engine
config = EventEngineConfig(
    initial_capital=1_000_000,
    slippage_bps=1.0,
    commission_per_share=0.005,
)

engine = EventDrivenEngine(config)

# Define strategy callback
def on_bar(event: BarEvent):
    # Your strategy logic here
    if event.close > event.open:
        # Generate buy signal
        pass

engine.register_strategy(on_bar=on_bar)

# Run backtest
result = engine.run(data)
```

### ML Training Pipeline

```python
from src.training import Trainer, ModelFactory, OptunaOptimizer
from src.training.validation import PurgedKFoldCV

# Create model
model = ModelFactory.create("lightgbm")

# Setup cross-validation
cv = PurgedKFoldCV(n_splits=5, purge_gap=5, embargo_pct=0.01)

# Optuna optimization
optimizer = OptunaOptimizer(
    model_type="lightgbm",
    cv=cv,
    metric="sharpe_ratio",
    n_trials=100,
)
best_params = optimizer.optimize(X_train, y_train)

# Train with best params
trainer = Trainer(model=model, cv=cv)
result = trainer.train(X_train, y_train)
```

### Feature Engineering

```python
from src.features import (
    TechnicalIndicators,
    FractionalDiffTransformer,
    MacroFeatureGenerator,
    FeatureStore,
)

# Technical indicators
indicators = TechnicalIndicators(data)
features = indicators.add_all()

# Fractional differentiation
frac_diff = FractionalDiffTransformer(d=0.4)
stationary_prices = frac_diff.fit_transform(prices)

# Macro features
macro_gen = MacroFeatureGenerator()
macro_features = macro_gen.get_all_features()

# Feature store
store = FeatureStore(storage_path="data/features")
store.materialize(data, feature_names=["returns_1d", "volatility_20d"])
features = store.get_historical_features(entity_df, ["returns_1d"])
```

### Generate Performance Report

```python
from src.backtesting.reports import create_tear_sheet, PerformanceDashboard

# Generate tear sheet
html = create_tear_sheet(
    returns=result.returns,
    benchmark_returns=benchmark_returns,
    strategy_name="Multi-Factor Momentum",
    output_path="reports/tear_sheet.html",
)

# Interactive dashboard
dashboard = PerformanceDashboard(
    returns=result.returns,
    benchmark_returns=benchmark_returns,
)
dashboard.generate_tear_sheet(output_path="reports/dashboard.html")
```

## Configuration

### ML Configuration (config/ml_config.yaml)

```yaml
random_seed: 42
periods_per_year: 252

feature_selection:
  method: importance
  top_k: 50
  min_importance: 0.001

cross_validation:
  n_splits: 5
  purge_gap: 5
  embargo_pct: 0.01

optimization:
  n_trials: 100
  timeout_seconds: 3600
  metric: sharpe_ratio

models:
  lightgbm:
    objective: regression
    n_estimators: 1000
    learning_rate: 0.05
```

### Risk Configuration (config/risk_limits.yaml)

```yaml
position_limits:
  max_position_pct: 0.05
  max_sector_pct: 0.25
  max_leverage: 1.0

drawdown_controls:
  warning_threshold: 0.05
  reduction_threshold: 0.08
  flatten_threshold: 0.15

var_limits:
  confidence_level: 0.95
  max_var_pct: 0.02
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

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

## Performance Metrics

The system calculates comprehensive performance metrics:

**Return Metrics:**
- Total Return, CAGR, MTD/QTD/YTD
- Best/Worst Day, Month, Year

**Risk-Adjusted Metrics:**
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Omega Ratio, Information Ratio

**Risk Metrics:**
- Volatility, Max Drawdown, Avg Drawdown
- VaR (95%, 99%), CVaR (Expected Shortfall)
- Skewness, Kurtosis

**Statistical Tests:**
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Minimum Track Record Length

**Trading Metrics:**
- Win Rate, Profit Factor
- Average Win/Loss, Max Consecutive Wins/Losses

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request
