"""
Strategies Module
=================

Comprehensive trading strategy library for the algorithmic trading platform.
Provides momentum, statistical arbitrage, and ML-based strategies.

Strategy Categories:
- Base: Abstract strategy interface, combiners
- Momentum: Trend following, breakout, mean reversion, MACD, RSI
- Statistical: Pairs trading, cointegration, Kalman filter, OU process
- ML: Classifiers, regressors, ensemble, neural networks

Architecture:
- StrategyRegistry: Dynamic strategy registration and discovery
- StrategyFactory: Type-safe strategy instantiation
- BaseStrategy: Abstract base with lifecycle management

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Type, TypeVar

from config.settings import get_logger

# =============================================================================
# BASE STRATEGY IMPORTS
# =============================================================================

from strategies.base import (
    # Enums
    StrategyState,
    SignalAction,
    # Configuration
    StrategyConfig,
    StrategyMetrics,
    # Base class
    BaseStrategy,
    # Combiner
    StrategyCombiner,
)

# =============================================================================
# MOMENTUM STRATEGY IMPORTS
# =============================================================================

from strategies.momentum import (
    # Configurations
    TrendFollowingConfig,
    BreakoutConfig,
    MeanReversionConfig,
    DualMomentumConfig,
    RSIDivergenceConfig,
    MACDStrategyConfig,
    # Strategies
    TrendFollowingStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    DualMomentumStrategy,
    RSIDivergenceStrategy,
    MACDStrategy,
)

# =============================================================================
# STATISTICAL STRATEGY IMPORTS
# =============================================================================

from strategies.statistical import (
    # Utilities
    calculate_zscore,
    calculate_spread,
    calculate_hedge_ratio_ols,
    calculate_half_life,
    adf_test,
    engle_granger_test,
    # Configurations
    PairsTradingConfig,
    CointegrationConfig,
    KalmanFilterConfig,
    OrnsteinUhlenbeckConfig,
    # Strategies
    PairsTradingStrategy,
    CointegrationStrategy,
    KalmanFilterStrategy,
    OrnsteinUhlenbeckStrategy,
)

# =============================================================================
# ML STRATEGY IMPORTS
# =============================================================================

from strategies.ml_strategy import (
    # Types
    MLModel,
    PredictionType,
    # Configurations
    MLStrategyConfig,
    MLClassifierConfig,
    MLRegressorConfig,
    EnsembleMLConfig,
    NeuralNetConfig,
    # Strategies
    BaseMLStrategy,
    MLClassifierStrategy,
    MLRegressorStrategy,
    EnsembleMLStrategy,
    NeuralNetStrategy,
    # Factory
    create_ml_strategy,
)


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseStrategy)


# =============================================================================
# STRATEGY CATEGORY ENUM
# =============================================================================

class StrategyCategory(str, Enum):
    """Strategy categories for classification."""
    MOMENTUM = "momentum"
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    STATISTICAL = "statistical"
    PAIRS = "pairs"
    ML_CLASSIFIER = "ml_classifier"
    ML_REGRESSOR = "ml_regressor"
    ML_ENSEMBLE = "ml_ensemble"
    ML_NEURAL = "ml_neural"
    CUSTOM = "custom"


# =============================================================================
# STRATEGY METADATA
# =============================================================================

@dataclass(frozen=True)
class StrategyMetadata:
    """
    Metadata describing a strategy.
    
    Attributes:
        name: Human-readable strategy name
        description: Strategy description
        category: Strategy category
        config_class: Configuration class type
        strategy_class: Strategy class type
        author: Strategy author
        version: Strategy version
        requires_multiple_symbols: Whether strategy needs multiple symbols
        min_history_bars: Minimum bars of history required
        supported_timeframes: List of supported timeframes
    """
    name: str
    description: str
    category: StrategyCategory
    config_class: Type[StrategyConfig]
    strategy_class: Type[BaseStrategy]
    author: str = "Algo Trading Platform"
    version: str = "1.0.0"
    requires_multiple_symbols: bool = False
    min_history_bars: int = 100
    supported_timeframes: tuple[str, ...] = ("15min", "1hour", "4hour", "1day")


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

class StrategyRegistry:
    """
    Central registry for strategy discovery and instantiation.
    
    Provides:
    - Strategy registration
    - Strategy discovery by name/category
    - Factory method for creating strategies
    - Metadata access
    
    Example:
        # Register a custom strategy
        StrategyRegistry.register(
            "my_strategy",
            MyStrategy,
            MyStrategyConfig,
            StrategyCategory.CUSTOM
        )
        
        # Create strategy instance
        strategy = StrategyRegistry.create("trend_following", config_dict)
        
        # Get all momentum strategies
        momentum = StrategyRegistry.get_by_category(StrategyCategory.MOMENTUM)
    """
    
    _strategies: dict[str, StrategyMetadata] = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initialize registry with built-in strategies."""
        if cls._initialized:
            return
        
        # Register momentum strategies
        cls._register_builtin(
            key="trend_following",
            name="Trend Following",
            description="MA crossover trend following with ADX filter",
            category=StrategyCategory.TREND,
            config_class=TrendFollowingConfig,
            strategy_class=TrendFollowingStrategy,
            min_history=200,
        )
        
        cls._register_builtin(
            key="breakout",
            name="Breakout",
            description="Channel breakout with volume confirmation",
            category=StrategyCategory.BREAKOUT,
            config_class=BreakoutConfig,
            strategy_class=BreakoutStrategy,
            min_history=60,
        )
        
        cls._register_builtin(
            key="mean_reversion",
            name="Mean Reversion",
            description="Bollinger Bands + RSI mean reversion",
            category=StrategyCategory.MEAN_REVERSION,
            config_class=MeanReversionConfig,
            strategy_class=MeanReversionStrategy,
            min_history=50,
        )
        
        cls._register_builtin(
            key="dual_momentum",
            name="Dual Momentum",
            description="Absolute and relative momentum combination",
            category=StrategyCategory.MOMENTUM,
            config_class=DualMomentumConfig,
            strategy_class=DualMomentumStrategy,
            min_history=252,
            requires_multiple=True,
        )
        
        cls._register_builtin(
            key="rsi_divergence",
            name="RSI Divergence",
            description="RSI price divergence detection",
            category=StrategyCategory.MOMENTUM,
            config_class=RSIDivergenceConfig,
            strategy_class=RSIDivergenceStrategy,
            min_history=50,
        )
        
        cls._register_builtin(
            key="macd",
            name="MACD Crossover",
            description="MACD/signal line crossover strategy",
            category=StrategyCategory.MOMENTUM,
            config_class=MACDStrategyConfig,
            strategy_class=MACDStrategy,
            min_history=60,
        )
        
        # Register statistical strategies
        cls._register_builtin(
            key="pairs_trading",
            name="Pairs Trading",
            description="Classic z-score pairs trading",
            category=StrategyCategory.PAIRS,
            config_class=PairsTradingConfig,
            strategy_class=PairsTradingStrategy,
            min_history=60,
            requires_multiple=True,
        )
        
        cls._register_builtin(
            key="cointegration",
            name="Cointegration",
            description="Multi-pair cointegration trading",
            category=StrategyCategory.STATISTICAL,
            config_class=CointegrationConfig,
            strategy_class=CointegrationStrategy,
            min_history=120,
            requires_multiple=True,
        )
        
        cls._register_builtin(
            key="kalman_filter",
            name="Kalman Filter",
            description="Dynamic hedge ratio with Kalman filter",
            category=StrategyCategory.STATISTICAL,
            config_class=KalmanFilterConfig,
            strategy_class=KalmanFilterStrategy,
            min_history=60,
            requires_multiple=True,
        )
        
        cls._register_builtin(
            key="ornstein_uhlenbeck",
            name="Ornstein-Uhlenbeck",
            description="OU process mean reversion",
            category=StrategyCategory.STATISTICAL,
            config_class=OrnsteinUhlenbeckConfig,
            strategy_class=OrnsteinUhlenbeckStrategy,
            min_history=100,
            requires_multiple=True,
        )
        
        # Register ML strategies
        cls._register_builtin(
            key="ml_classifier",
            name="ML Classifier",
            description="Classification-based trading signals",
            category=StrategyCategory.ML_CLASSIFIER,
            config_class=MLClassifierConfig,
            strategy_class=MLClassifierStrategy,
            min_history=100,
        )
        
        cls._register_builtin(
            key="ml_regressor",
            name="ML Regressor",
            description="Regression-based return prediction",
            category=StrategyCategory.ML_REGRESSOR,
            config_class=MLRegressorConfig,
            strategy_class=MLRegressorStrategy,
            min_history=100,
        )
        
        cls._register_builtin(
            key="ml_ensemble",
            name="ML Ensemble",
            description="Ensemble of multiple ML models",
            category=StrategyCategory.ML_ENSEMBLE,
            config_class=EnsembleMLConfig,
            strategy_class=EnsembleMLStrategy,
            min_history=100,
        )
        
        cls._register_builtin(
            key="ml_neural",
            name="Neural Network",
            description="Deep learning strategy (LSTM/Transformer)",
            category=StrategyCategory.ML_NEURAL,
            config_class=NeuralNetConfig,
            strategy_class=NeuralNetStrategy,
            min_history=200,
        )
        
        cls._initialized = True
        logger.info(f"StrategyRegistry initialized with {len(cls._strategies)} strategies")
    
    @classmethod
    def _register_builtin(
        cls,
        key: str,
        name: str,
        description: str,
        category: StrategyCategory,
        config_class: Type[StrategyConfig],
        strategy_class: Type[BaseStrategy],
        min_history: int = 100,
        requires_multiple: bool = False,
    ) -> None:
        """Register a built-in strategy."""
        cls._strategies[key] = StrategyMetadata(
            name=name,
            description=description,
            category=category,
            config_class=config_class,
            strategy_class=strategy_class,
            requires_multiple_symbols=requires_multiple,
            min_history_bars=min_history,
        )
    
    @classmethod
    def register(
        cls,
        key: str,
        strategy_class: Type[BaseStrategy],
        config_class: Type[StrategyConfig],
        category: StrategyCategory = StrategyCategory.CUSTOM,
        name: str | None = None,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Register a custom strategy.
        
        Args:
            key: Unique strategy identifier
            strategy_class: Strategy class type
            config_class: Configuration class type
            category: Strategy category
            name: Human-readable name
            description: Strategy description
            **kwargs: Additional metadata fields
        """
        cls._ensure_initialized()
        
        if key in cls._strategies:
            logger.warning(f"Overwriting existing strategy: {key}")
        
        cls._strategies[key] = StrategyMetadata(
            name=name or key.replace("_", " ").title(),
            description=description,
            category=category,
            config_class=config_class,
            strategy_class=strategy_class,
            **kwargs,
        )
        logger.info(f"Registered strategy: {key}")
    
    @classmethod
    def unregister(cls, key: str) -> bool:
        """
        Unregister a strategy.
        
        Args:
            key: Strategy identifier
        
        Returns:
            True if strategy was removed
        """
        cls._ensure_initialized()
        
        if key in cls._strategies:
            del cls._strategies[key]
            logger.info(f"Unregistered strategy: {key}")
            return True
        return False
    
    @classmethod
    def get(cls, key: str) -> StrategyMetadata | None:
        """
        Get strategy metadata.
        
        Args:
            key: Strategy identifier
        
        Returns:
            StrategyMetadata or None
        """
        cls._ensure_initialized()
        return cls._strategies.get(key)
    
    @classmethod
    def create(
        cls,
        key: str,
        config: dict[str, Any] | StrategyConfig | None = None,
        **kwargs: Any,
    ) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            key: Strategy identifier
            config: Configuration dict or instance
            **kwargs: Additional arguments passed to strategy
        
        Returns:
            Strategy instance
        
        Raises:
            ValueError: If strategy not found
        """
        cls._ensure_initialized()
        
        metadata = cls._strategies.get(key)
        if metadata is None:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: {key}. Available: {available}"
            )
        
        # Create config if dict provided
        if isinstance(config, dict):
            config_instance = metadata.config_class(**config)
        elif config is None:
            config_instance = metadata.config_class()
        else:
            config_instance = config
        
        # Create strategy
        strategy = metadata.strategy_class(config_instance, **kwargs)
        logger.debug(f"Created strategy: {key}")
        
        return strategy
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """
        Get list of all registered strategy keys.
        
        Returns:
            List of strategy identifiers
        """
        cls._ensure_initialized()
        return list(cls._strategies.keys())
    
    @classmethod
    def list_metadata(cls) -> dict[str, StrategyMetadata]:
        """
        Get all strategy metadata.
        
        Returns:
            Dictionary of key to metadata
        """
        cls._ensure_initialized()
        return cls._strategies.copy()
    
    @classmethod
    def get_by_category(
        cls,
        category: StrategyCategory,
    ) -> dict[str, StrategyMetadata]:
        """
        Get strategies by category.
        
        Args:
            category: Strategy category
        
        Returns:
            Dictionary of matching strategies
        """
        cls._ensure_initialized()
        return {
            key: meta
            for key, meta in cls._strategies.items()
            if meta.category == category
        }
    
    @classmethod
    def get_multi_symbol_strategies(cls) -> dict[str, StrategyMetadata]:
        """
        Get strategies requiring multiple symbols.
        
        Returns:
            Dictionary of multi-symbol strategies
        """
        cls._ensure_initialized()
        return {
            key: meta
            for key, meta in cls._strategies.items()
            if meta.requires_multiple_symbols
        }
    
    @classmethod
    def get_single_symbol_strategies(cls) -> dict[str, StrategyMetadata]:
        """
        Get strategies for single symbols.
        
        Returns:
            Dictionary of single-symbol strategies
        """
        cls._ensure_initialized()
        return {
            key: meta
            for key, meta in cls._strategies.items()
            if not meta.requires_multiple_symbols
        }


# =============================================================================
# STRATEGY FACTORY (CONVENIENCE)
# =============================================================================

class StrategyFactory:
    """
    Factory for creating strategies with validation.
    
    Provides convenience methods and validation
    for common strategy creation patterns.
    
    Example:
        # Create trend following strategy
        strategy = StrategyFactory.create_momentum_strategy(
            "trend_following",
            symbols=["AAPL", "GOOGL"],
            ma_fast_period=10,
            ma_slow_period=50,
        )
        
        # Create ML strategy with model
        strategy = StrategyFactory.create_ml_strategy(
            "classifier",
            model=trained_model,
            symbols=["AAPL"],
        )
    """
    
    @staticmethod
    def create_momentum_strategy(
        strategy_type: str,
        symbols: list[str],
        **config_params: Any,
    ) -> BaseStrategy:
        """
        Create a momentum-based strategy.
        
        Args:
            strategy_type: Type of momentum strategy
            symbols: Symbols to trade
            **config_params: Strategy-specific parameters
        
        Returns:
            Configured strategy instance
        """
        config_params["symbols"] = symbols
        return StrategyRegistry.create(strategy_type, config_params)
    
    @staticmethod
    def create_statistical_strategy(
        strategy_type: str,
        symbols: list[str],
        **config_params: Any,
    ) -> BaseStrategy:
        """
        Create a statistical arbitrage strategy.
        
        Args:
            strategy_type: Type of statistical strategy
            symbols: Symbols to trade (pairs)
            **config_params: Strategy-specific parameters
        
        Returns:
            Configured strategy instance
        """
        if len(symbols) < 2:
            raise ValueError("Statistical strategies require at least 2 symbols")
        
        config_params["symbols"] = symbols
        return StrategyRegistry.create(strategy_type, config_params)
    
    @staticmethod
    def create_ml_strategy_with_model(
        strategy_type: str,
        model: Any,
        symbols: list[str],
        **config_params: Any,
    ) -> BaseStrategy:
        """
        Create an ML strategy with a trained model.
        
        Args:
            strategy_type: Type of ML strategy
            model: Trained ML model
            symbols: Symbols to trade
            **config_params: Strategy-specific parameters
        
        Returns:
            Configured ML strategy instance
        """
        config_params["symbols"] = symbols
        return StrategyRegistry.create(strategy_type, config_params, model=model)
    
    @staticmethod
    def create_ensemble(
        strategies: list[tuple[str, dict[str, Any]]],
        combination_method: str = "voting",
        weights: list[float] | None = None,
    ) -> StrategyCombiner:
        """
        Create an ensemble of strategies.
        
        Args:
            strategies: List of (strategy_key, config) tuples
            combination_method: How to combine signals
            weights: Strategy weights (for weighted method)
        
        Returns:
            StrategyCombiner instance
        """
        strategy_instances = [
            StrategyRegistry.create(key, config)
            for key, config in strategies
        ]
        
        return StrategyCombiner(
            strategies=strategy_instances,
            method=combination_method,
            weights=weights,
        )
    
    @staticmethod
    def create_from_config(
        config: dict[str, Any],
    ) -> BaseStrategy:
        """
        Create strategy from a configuration dictionary.
        
        Expected format:
        {
            "strategy_type": "trend_following",
            "symbols": ["AAPL"],
            "parameters": {
                "ma_fast_period": 10,
                ...
            }
        }
        
        Args:
            config: Strategy configuration dictionary
        
        Returns:
            Configured strategy instance
        """
        strategy_type = config.get("strategy_type")
        if not strategy_type:
            raise ValueError("Configuration must include 'strategy_type'")
        
        symbols = config.get("symbols", [])
        parameters = config.get("parameters", {})
        parameters["symbols"] = symbols
        
        return StrategyRegistry.create(strategy_type, parameters)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_strategy(
    strategy_type: str,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseStrategy:
    """
    Convenience function to create a strategy.
    
    Args:
        strategy_type: Strategy identifier
        config: Configuration parameters
        **kwargs: Additional arguments
    
    Returns:
        Strategy instance
    """
    return StrategyRegistry.create(strategy_type, config, **kwargs)


def list_strategies() -> list[str]:
    """
    List all available strategies.
    
    Returns:
        List of strategy identifiers
    """
    return StrategyRegistry.list_strategies()


def get_strategy_info(strategy_type: str) -> dict[str, Any]:
    """
    Get information about a strategy.
    
    Args:
        strategy_type: Strategy identifier
    
    Returns:
        Dictionary with strategy information
    """
    metadata = StrategyRegistry.get(strategy_type)
    if metadata is None:
        return {}
    
    return {
        "name": metadata.name,
        "description": metadata.description,
        "category": metadata.category.value,
        "requires_multiple_symbols": metadata.requires_multiple_symbols,
        "min_history_bars": metadata.min_history_bars,
        "supported_timeframes": metadata.supported_timeframes,
        "config_class": metadata.config_class.__name__,
        "strategy_class": metadata.strategy_class.__name__,
    }


def get_strategies_by_category(category: str) -> list[str]:
    """
    Get strategies by category name.
    
    Args:
        category: Category name (e.g., "momentum", "statistical")
    
    Returns:
        List of strategy identifiers
    """
    try:
        cat_enum = StrategyCategory(category)
    except ValueError:
        return []
    
    return list(StrategyRegistry.get_by_category(cat_enum).keys())


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _initialize_module() -> None:
    """Initialize module on import."""
    StrategyRegistry._ensure_initialized()


_initialize_module()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Base ===
    "StrategyState",
    "SignalAction",
    "StrategyConfig",
    "StrategyMetrics",
    "BaseStrategy",
    "StrategyCombiner",
    
    # === Momentum Configs ===
    "TrendFollowingConfig",
    "BreakoutConfig",
    "MeanReversionConfig",
    "DualMomentumConfig",
    "RSIDivergenceConfig",
    "MACDStrategyConfig",
    
    # === Momentum Strategies ===
    "TrendFollowingStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "DualMomentumStrategy",
    "RSIDivergenceStrategy",
    "MACDStrategy",
    
    # === Statistical Utilities ===
    "calculate_zscore",
    "calculate_spread",
    "calculate_hedge_ratio_ols",
    "calculate_half_life",
    "adf_test",
    "engle_granger_test",
    
    # === Statistical Configs ===
    "PairsTradingConfig",
    "CointegrationConfig",
    "KalmanFilterConfig",
    "OrnsteinUhlenbeckConfig",
    
    # === Statistical Strategies ===
    "PairsTradingStrategy",
    "CointegrationStrategy",
    "KalmanFilterStrategy",
    "OrnsteinUhlenbeckStrategy",
    
    # === ML Types ===
    "MLModel",
    "PredictionType",
    
    # === ML Configs ===
    "MLStrategyConfig",
    "MLClassifierConfig",
    "MLRegressorConfig",
    "EnsembleMLConfig",
    "NeuralNetConfig",
    
    # === ML Strategies ===
    "BaseMLStrategy",
    "MLClassifierStrategy",
    "MLRegressorStrategy",
    "EnsembleMLStrategy",
    "NeuralNetStrategy",
    "create_ml_strategy",
    
    # === Registry & Factory ===
    "StrategyCategory",
    "StrategyMetadata",
    "StrategyRegistry",
    "StrategyFactory",
    
    # === Convenience Functions ===
    "create_strategy",
    "list_strategies",
    "get_strategy_info",
    "get_strategies_by_category",
]