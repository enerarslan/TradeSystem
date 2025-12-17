"""
AlphaTrade - Institutional-Grade Algorithmic Trading System

This package provides a comprehensive suite of tools for:
- Data loading and preprocessing (Pandas, Polars, TimescaleDB)
- Feature engineering with 50+ technical indicators
- Advanced feature transformations (fractional diff, cointegration)
- Macroeconomic data integration (FRED)
- Feature store for ML pipelines
- Multiple trading strategy implementations
- ML training pipeline (LightGBM, XGBoost, CatBoost)
- Deep learning models (LSTM, Transformer)
- Hyperparameter optimization (Optuna)
- Experiment tracking (MLflow)
- Risk management and position sizing
- Portfolio optimization (Mean-Variance, Risk Parity, HRP)
- High-performance backtesting (Vectorized, Event-Driven)
- Market impact modeling (Almgren-Chriss)
- Monte Carlo analysis and statistical tests
- Performance analytics and reporting
"""

__version__ = "2.0.0"
__author__ = "AlphaTrade Team"

from typing import Final

# Package metadata
PACKAGE_NAME: Final[str] = "alphatrade"
VERSION: Final[str] = __version__

# Convenience imports for common components
from src.data import DataLoader, DataValidator, DataProcessor
from src.features import TechnicalIndicators, FeaturePipeline
from src.strategies import BaseStrategy
from src.backtesting import BacktestEngine, BacktestResult, EventDrivenEngine
from src.risk import PositionSizer, VaRCalculator, DrawdownController
from src.portfolio import PortfolioOptimizer

__all__ = [
    # Metadata
    "PACKAGE_NAME",
    "VERSION",
    "__version__",
    # Data
    "DataLoader",
    "DataValidator",
    "DataProcessor",
    # Features
    "TechnicalIndicators",
    "FeaturePipeline",
    # Strategies
    "BaseStrategy",
    # Backtesting
    "BacktestEngine",
    "BacktestResult",
    "EventDrivenEngine",
    # Risk
    "PositionSizer",
    "VaRCalculator",
    "DrawdownController",
    # Portfolio
    "PortfolioOptimizer",
]
