# AlphaTrade System

An institutional-grade algorithmic trading platform designed for systematic, multi-strategy trading. Built with production-quality code following JPMorgan-level design standards.

## Features

- **Multi-Strategy Framework**: Support for momentum, mean reversion, volatility breakout, ML-based, and ensemble strategies
- **Advanced Risk Management**: VaR/CVaR calculations, drawdown controls, position sizing algorithms
- **Portfolio Optimization**: Mean-variance, risk parity, HRP, and Black-Litterman optimization
- **Comprehensive Backtesting**: Event-driven and vectorized backtesting with transaction cost modeling
- **50+ Technical Indicators**: Full suite of trend, momentum, volatility, and volume indicators
- **Walk-Forward Analysis**: Time-series cross-validation with purging and embargo

## Installation

### Prerequisites

- Python 3.11+
- Git

### Setup

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

# Install dependencies
pip install -e .
```

### Dependencies

Core dependencies include:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- lightgbm >= 4.0.0
- xgboost >= 2.0.0
- scipy >= 1.11.0
- plotly >= 5.17.0
- pydantic >= 2.0.0
- loguru >= 0.7.0
- PyYAML >= 6.0

See `pyproject.toml` for complete list.

## Project Structure

```
AlphaTrade_System/
├── config/                     # Configuration files
│   ├── settings.py            # Pydantic settings
│   ├── trading_config.yaml    # Trading parameters
│   ├── strategy_params.yaml   # Strategy configuration
│   ├── risk_limits.yaml       # Risk management limits
│   ├── feature_params.yaml    # Feature engineering config
│   └── logging_config.yaml    # Logging configuration
├── data/
│   ├── raw/                   # Raw OHLCV data (CSV)
│   ├── processed/             # Processed data
│   └── features/              # Cached features
├── src/
│   ├── data/                  # Data layer
│   │   ├── loaders/          # Data loaders
│   │   ├── validators/       # Data validation
│   │   ├── processors/       # Data processing
│   │   └── storage/          # Data caching
│   ├── features/             # Feature engineering
│   │   ├── technical/        # Technical indicators
│   │   └── pipeline.py       # Feature pipeline
│   ├── strategies/           # Trading strategies
│   │   ├── base.py          # Base strategy class
│   │   ├── momentum/        # Momentum strategies
│   │   ├── mean_reversion/  # Mean reversion strategies
│   │   ├── ml_based/        # ML strategies
│   │   ├── multi_factor/    # Multi-factor strategies
│   │   └── ensemble.py      # Ensemble strategy
│   ├── risk/                 # Risk management
│   │   ├── position_sizing.py
│   │   ├── var_models.py
│   │   ├── drawdown.py
│   │   └── correlation.py
│   ├── portfolio/            # Portfolio management
│   │   ├── optimizer.py
│   │   ├── rebalancer.py
│   │   └── allocation.py
│   ├── backtesting/          # Backtesting engine
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   ├── analysis.py
│   │   └── reports/
│   ├── execution/            # Order execution
│   │   ├── order_manager.py
│   │   ├── slippage.py
│   │   └── transaction_cost.py
│   └── utils/                # Utilities
│       ├── logger.py
│       ├── decorators.py
│       └── helpers.py
├── tests/                    # Test suite
│   ├── unit/
│   └── integration/
├── main.py                   # Main entry point
└── pyproject.toml           # Project configuration
```

## Quick Start

### Running a Backtest

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
    initial_capital=1000000,
    commission_pct=0.001,
    slippage_pct=0.0005,
)
result = engine.run(strategy, data)

# Print metrics
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

### Using the CLI

```bash
# Run full backtest with default settings
python main.py

# Run with specific strategy
python main.py --strategy momentum

# Generate report only
python main.py --report-only
```

## Strategies

### Multi-Factor Momentum

Combines multiple momentum factors across different lookback periods:
- Price momentum (5, 10, 20, 60 day)
- Volume-adjusted momentum
- Volatility-adjusted returns

```yaml
# config/strategy_params.yaml
momentum:
  lookback_periods: [5, 10, 20, 60]
  top_n_long: 5
  weighting_method: "decay"
```

### Mean Reversion

Z-score based mean reversion with Bollinger Band confirmation:
- Entry on extreme z-scores
- Exit on mean reversion
- Optional trend filter

```yaml
mean_reversion:
  lookback_period: 20
  entry_zscore: 2.0
  exit_zscore: 0.5
```

### Volatility Breakout

ATR-based channel breakout strategy:
- Dynamic channels based on ATR
- Volume confirmation
- Trailing stops

```yaml
volatility_breakout:
  atr_period: 14
  atr_multiplier: 2.0
  volume_surge_threshold: 1.5
```

### ML Alpha

Machine learning based signal generation:
- LightGBM/XGBoost models
- Walk-forward training
- Feature importance analysis

```yaml
ml_alpha:
  model_type: "lightgbm"
  prediction_horizon: 4
  walk_forward: true
```

### Ensemble

Combines signals from multiple strategies:
- Weighted average
- Majority voting
- Dynamic weight adjustment

```yaml
ensemble:
  strategies: [momentum, mean_reversion, volatility_breakout]
  combination_method: "weighted_average"
  dynamic_weights: true
```

## Risk Management

### Position Sizing

Multiple position sizing algorithms:
- **Fixed Fraction**: Constant percentage per trade
- **Kelly Criterion**: Optimal sizing based on edge
- **Volatility Target**: Scale positions to target volatility
- **Risk Parity**: Equal risk contribution

### VaR Limits

Value at Risk controls:
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- CVaR (Expected Shortfall)

### Drawdown Controls

Automatic position reduction:
- Warning at 10% drawdown
- 50% reduction at 8% drawdown
- Full flatten at 15% drawdown

## Configuration

All parameters are configurable via YAML files in `config/`:

- `trading_config.yaml`: General trading parameters
- `strategy_params.yaml`: Strategy-specific parameters
- `risk_limits.yaml`: Risk management limits
- `feature_params.yaml`: Feature engineering settings

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Data Format

Input data should be in CSV format with the following columns:
- `timestamp`: DateTime index
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

Example:
```csv
timestamp,open,high,low,close,volume
2023-01-03 09:30:00,150.25,150.50,150.10,150.35,125000
2023-01-03 09:45:00,150.35,150.75,150.30,150.65,98000
```

## Performance Metrics

The system calculates comprehensive performance metrics:

**Return Metrics:**
- Total Return
- Annualized Return
- Monthly Returns

**Risk-Adjusted Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio

**Risk Metrics:**
- Volatility
- Max Drawdown
- VaR/CVaR

**Trading Metrics:**
- Win Rate
- Profit Factor
- Average Win/Loss
- Number of Trades

## API Reference

### DataLoader

```python
class DataLoader:
    def __init__(self, data_path: str, cache_path: str = None)
    def load(self, symbol: str) -> pd.DataFrame
    def load_all(self) -> Dict[str, pd.DataFrame]
    def get_symbols(self) -> List[str]
```

### BaseStrategy

```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data, features) -> pd.DataFrame
    def calculate_positions(self, signals, prices, capital) -> pd.DataFrame
```

### BacktestEngine

```python
class BacktestEngine:
    def __init__(self, initial_capital, commission_pct, slippage_pct)
    def run(self, strategy, data) -> BacktestResult
```

### PortfolioOptimizer

```python
class PortfolioOptimizer:
    def optimize(self, method: str) -> np.ndarray
    def min_variance(self) -> np.ndarray
    def max_sharpe(self) -> np.ndarray
    def risk_parity(self) -> np.ndarray
```

## License

Proprietary - All rights reserved.

## Contributing

Internal use only. Contact the quantitative research team for contributions.
