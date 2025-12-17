"""
AlphaTrade - Institutional-Grade Algorithmic Trading System v2.0.0

JPMorgan-Level Trading Platform providing:

DATA LAYER:
- High-performance data loading (Pandas, Polars)
- TimescaleDB time-series database integration
- Data validation and quality checks
- Caching and storage management

FEATURE ENGINEERING:
- 50+ technical indicators
- Fractional differentiation (memory-preserving stationarity)
- Cointegration and Ornstein-Uhlenbeck features
- Macroeconomic features (FRED API integration)
- Point-in-time feature store

ML TRAINING:
- Model factory (LightGBM, XGBoost, CatBoost, Random Forest)
- Purged K-Fold cross-validation
- Walk-forward validation
- Optuna hyperparameter optimization
- MLflow experiment tracking
- Model registry and versioning

DEEP LEARNING:
- LSTM with attention mechanism
- Temporal Fusion Transformer
- Custom financial loss functions (Sharpe, Sortino, MaxDD)
- PyTorch Lightning integration

BACKTESTING:
- Vectorized backtest engine
- Event-driven backtest engine (microsecond precision)
- Market impact modeling (Almgren-Chriss)
- Monte Carlo analysis
- Statistical significance tests (PSR, DSR)

RISK MANAGEMENT:
- Position sizing (Kelly, Volatility-scaled)
- VaR/CVaR calculation
- Drawdown control
- Portfolio optimization (MVO, Risk Parity, HRP)

All components are REQUIRED and fully integrated.
"""

__version__ = "2.0.0"
__author__ = "AlphaTrade Team"

from typing import Final

# Package metadata
PACKAGE_NAME: Final[str] = "alphatrade"
VERSION: Final[str] = __version__

# =============================================================================
# DATA LAYER - REQUIRED
# =============================================================================
from src.data import (
    DataLoader,
    DataValidator,
    DataProcessor,
    load_single_stock,
    load_all_stocks,
    get_available_symbols,
    ValidationResult,
    validate_ohlcv,
    preprocess_ohlcv,
    create_train_test_split,
    DataCache,
    POLARS_AVAILABLE,
    TIMESCALE_AVAILABLE,
)

# =============================================================================
# FEATURE ENGINEERING - REQUIRED
# =============================================================================
from src.features import (
    # Technical indicators
    TechnicalIndicators,
    FeaturePipeline,
    FeatureProcessor,
    create_feature_matrix,
    # Fractional differentiation
    frac_diff_ffd,
    find_min_d,
    test_stationarity_adf,
    FractionalDiffTransformer,
    # Statistical arbitrage
    CointegrationAnalyzer,
    OrnsteinUhlenbeckEstimator,
    # Macro features
    FREDClient,
    MacroIndicator,
    MacroFeatureGenerator,
    EconomicRegimeDetector,
    align_macro_to_price_data,
    # Feature store
    FeatureStore,
    FeatureDefinition,
    FeatureView,
    FeatureBuilder,
    create_standard_features,
)

# =============================================================================
# TRAINING - REQUIRED
# =============================================================================
from src.training import (
    # Experiment tracking
    ExperimentTracker,
    # Model factory and registry
    ModelFactory,
    ModelRegistry,
    # Training
    Trainer,
    TrainingResult,
    # Validation
    PurgedKFoldCV,
    CombinatorialPurgedKFoldCV,
    WalkForwardValidator,
    # Optimization
    OptunaOptimizer,
    MultiObjectiveOptimizer,
    # Deep Learning
    LSTMPredictor,
    AttentionLSTM,
    TemporalFusionTransformer,
    SharpeLoss,
    SortinoLoss,
    MaxDrawdownLoss,
    CombinedFinancialLoss,
)

# =============================================================================
# STRATEGIES - REQUIRED
# =============================================================================
from src.strategies import BaseStrategy

# =============================================================================
# BACKTESTING - REQUIRED
# =============================================================================
from src.backtesting import (
    # Vectorized
    BacktestEngine,
    BacktestResult,
    run_backtest,
    VectorizedBacktest,
    TransactionCostModel,
    # Event-driven
    EventDrivenEngine,
    EventEngineConfig,
    EventEngineResult,
    Position,
    Portfolio,
    ExecutionSimulator,
    SimpleExecutionSimulator,
    OrderBookExecutionSimulator,
    run_event_backtest,
    # Metrics
    PerformanceMetrics,
    calculate_all_metrics,
    BacktestAnalyzer,
    analyze_trades,
    analyze_positions,
    # Market impact
    AlmgrenChrissModel,
    DynamicSpreadModel,
    LatencySimulator,
    # Monte Carlo
    MonteCarloAnalyzer,
    StatisticalTests,
)

# =============================================================================
# RISK MANAGEMENT - REQUIRED
# =============================================================================
from src.risk import (
    PositionSizer,
    VaRCalculator,
    DrawdownController,
)

# =============================================================================
# PORTFOLIO - REQUIRED
# =============================================================================
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
    "load_single_stock",
    "load_all_stocks",
    "get_available_symbols",
    "ValidationResult",
    "validate_ohlcv",
    "preprocess_ohlcv",
    "create_train_test_split",
    "DataCache",
    "POLARS_AVAILABLE",
    "TIMESCALE_AVAILABLE",
    # Features - Technical
    "TechnicalIndicators",
    "FeaturePipeline",
    "FeatureProcessor",
    "create_feature_matrix",
    # Features - Fractional Diff
    "frac_diff_ffd",
    "find_min_d",
    "test_stationarity_adf",
    "FractionalDiffTransformer",
    # Features - Statistical Arbitrage
    "CointegrationAnalyzer",
    "OrnsteinUhlenbeckEstimator",
    # Features - Macro
    "FREDClient",
    "MacroIndicator",
    "MacroFeatureGenerator",
    "EconomicRegimeDetector",
    "align_macro_to_price_data",
    # Features - Store
    "FeatureStore",
    "FeatureDefinition",
    "FeatureView",
    "FeatureBuilder",
    "create_standard_features",
    # Training
    "ExperimentTracker",
    "ModelFactory",
    "ModelRegistry",
    "Trainer",
    "TrainingResult",
    "PurgedKFoldCV",
    "CombinatorialPurgedKFoldCV",
    "WalkForwardValidator",
    "OptunaOptimizer",
    "MultiObjectiveOptimizer",
    # Deep Learning
    "LSTMPredictor",
    "AttentionLSTM",
    "TemporalFusionTransformer",
    "SharpeLoss",
    "SortinoLoss",
    "MaxDrawdownLoss",
    "CombinedFinancialLoss",
    # Strategies
    "BaseStrategy",
    # Backtesting - Vectorized
    "BacktestEngine",
    "BacktestResult",
    "run_backtest",
    "VectorizedBacktest",
    "TransactionCostModel",
    # Backtesting - Event-driven
    "EventDrivenEngine",
    "EventEngineConfig",
    "EventEngineResult",
    "Position",
    "Portfolio",
    "ExecutionSimulator",
    "SimpleExecutionSimulator",
    "OrderBookExecutionSimulator",
    "run_event_backtest",
    # Backtesting - Metrics
    "PerformanceMetrics",
    "calculate_all_metrics",
    "BacktestAnalyzer",
    "analyze_trades",
    "analyze_positions",
    # Backtesting - Market Impact
    "AlmgrenChrissModel",
    "DynamicSpreadModel",
    "LatencySimulator",
    # Backtesting - Monte Carlo
    "MonteCarloAnalyzer",
    "StatisticalTests",
    # Risk
    "PositionSizer",
    "VaRCalculator",
    "DrawdownController",
    # Portfolio
    "PortfolioOptimizer",
]
