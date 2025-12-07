"""
Strategies Module
=================

Trading strategies for the algorithmic trading platform.

Modules:
- base: Base strategy class and utilities
- momentum: Momentum-based strategies (MACD, RSI, Breakout)
- statistical: Statistical arbitrage strategies (Pairs, Cointegration, Kalman)
- ml_strategy: Basic ML strategy implementations
- alpha_ml: Production ML strategy (V1 - deprecated)
- alpha_ml_v2: JPMorgan-level ML strategy (V2 - recommended)

Author: Algo Trading Platform
License: MIT
"""

from typing import Any, TypeVar, Type

from config.settings import get_logger

# =============================================================================
# BASE STRATEGY
# =============================================================================

from strategies.base import (
    # Enums
    StrategyState,
    SignalAction,
    # Config
    StrategyConfig,
    # Strategy
    BaseStrategy,
    # Metrics
    StrategyMetrics,
)

# =============================================================================
# MOMENTUM STRATEGIES
# =============================================================================

from strategies.momentum import (
    # Configs
    MACDStrategyConfig,
    RSIMomentumConfig,
    BreakoutStrategyConfig,
    DualMomentumConfig,
    # Strategies
    MACDStrategy,
    RSIMomentumStrategy,
    BreakoutStrategy,
    DualMomentumStrategy,
)

# =============================================================================
# STATISTICAL STRATEGIES
# =============================================================================

from strategies.statistical import (
    # Configs
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
# ML STRATEGY (BASIC)
# =============================================================================

from strategies.ml_strategy import (
    # Types
    MLModel,
    PredictionType,
    # Configs
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

# =============================================================================
# ALPHA ML V1 (ORIGINAL - DEPRECATED)
# =============================================================================

from strategies.alpha_ml import (
    # Enums
    MarketRegime,
    ModelType,
    SignalStrengthLevel,
    # Config
    AlphaMLConfig,
    # Strategy
    AlphaMLStrategy,
    # Factory
    create_alpha_ml_strategy,
)

# =============================================================================
# ALPHA ML V2 (NEW - RECOMMENDED)
# =============================================================================

from strategies.alpha_ml_v2 import (
    # Enums (reuses MarketRegime, ModelType from v1)
    PredictionMode,
    # Config
    AlphaMLConfigV2,
    # Strategy
    AlphaMLStrategyV2,
    # Factory
    create_alpha_ml_strategy_v2,
)


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseStrategy)


# =============================================================================
# STRATEGY CATEGORY ENUM
# =============================================================================

from enum import Enum


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
    ML_ALPHA = "ml_alpha"
    ML_ALPHA_V2 = "ml_alpha_v2"
    CUSTOM = "custom"


# =============================================================================
# STRATEGY METADATA
# =============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyMetadata:
    """Metadata describing a strategy."""
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
    """Central registry for strategy discovery and instantiation."""
    
    _strategies: dict[str, StrategyMetadata] = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure built-in strategies are registered."""
        if cls._initialized:
            return
        
        cls._register_builtin_strategies()
    
    @classmethod
    def _register_builtin_strategies(cls) -> None:
        """Register all built-in strategies."""
        # Momentum strategies
        cls._register_builtin(
            key="macd",
            name="MACD Crossover",
            description="MACD line and signal crossover strategy",
            category=StrategyCategory.MOMENTUM,
            config_class=MACDStrategyConfig,
            strategy_class=MACDStrategy,
        )
        
        cls._register_builtin(
            key="rsi_momentum",
            name="RSI Momentum",
            description="RSI overbought/oversold strategy",
            category=StrategyCategory.MOMENTUM,
            config_class=RSIMomentumConfig,
            strategy_class=RSIMomentumStrategy,
        )
        
        cls._register_builtin(
            key="breakout",
            name="Breakout",
            description="Price breakout with volume confirmation",
            category=StrategyCategory.BREAKOUT,
            config_class=BreakoutStrategyConfig,
            strategy_class=BreakoutStrategy,
        )
        
        # Statistical strategies
        cls._register_builtin(
            key="pairs_trading",
            name="Pairs Trading",
            description="Z-score based pairs trading",
            category=StrategyCategory.PAIRS,
            config_class=PairsTradingConfig,
            strategy_class=PairsTradingStrategy,
            requires_multiple=True,
        )
        
        cls._register_builtin(
            key="cointegration",
            name="Cointegration",
            description="Cointegration-based pairs trading",
            category=StrategyCategory.STATISTICAL,
            config_class=CointegrationConfig,
            strategy_class=CointegrationStrategy,
            requires_multiple=True,
        )
        
        cls._register_builtin(
            key="kalman_filter",
            name="Kalman Filter",
            description="Dynamic hedge ratio with Kalman filter",
            category=StrategyCategory.STATISTICAL,
            config_class=KalmanFilterConfig,
            strategy_class=KalmanFilterStrategy,
            requires_multiple=True,
        )
        
        # ML strategies
        cls._register_builtin(
            key="ml_classifier",
            name="ML Classifier",
            description="Classification-based ML strategy",
            category=StrategyCategory.ML_CLASSIFIER,
            config_class=MLClassifierConfig,
            strategy_class=MLClassifierStrategy,
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
        
        # Alpha ML V1 (deprecated but still available)
        cls._register_builtin(
            key="alpha_ml",
            name="Alpha ML (V1)",
            description="Production ML strategy (deprecated - use V2)",
            category=StrategyCategory.ML_ALPHA,
            config_class=AlphaMLConfig,
            strategy_class=AlphaMLStrategy,
            min_history=200,
        )
        
        # Alpha ML V2 (recommended)
        cls._register_builtin(
            key="alpha_ml_v2",
            name="Alpha ML V2",
            description="JPMorgan-level ML strategy with Triple Barrier",
            category=StrategyCategory.ML_ALPHA_V2,
            config_class=AlphaMLConfigV2,
            strategy_class=AlphaMLStrategyV2,
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
        """Register a custom strategy."""
        cls._ensure_initialized()
        
        cls._strategies[key] = StrategyMetadata(
            name=name or key.replace("_", " ").title(),
            description=description,
            category=category,
            config_class=config_class,
            strategy_class=strategy_class,
            **kwargs,
        )
        logger.info(f"Registered custom strategy: {key}")
    
    @classmethod
    def get(cls, key: str) -> StrategyMetadata | None:
        """Get strategy metadata by key."""
        cls._ensure_initialized()
        return cls._strategies.get(key)
    
    @classmethod
    def create(
        cls,
        key: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseStrategy:
        """Create a strategy instance by key."""
        cls._ensure_initialized()
        
        metadata = cls._strategies.get(key)
        if not metadata:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: {key}. Available: {available}")
        
        config_obj = metadata.config_class(**(config or {}))
        return metadata.strategy_class(config_obj, **kwargs)
    
    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered strategies."""
        cls._ensure_initialized()
        return list(cls._strategies.keys())
    
    @classmethod
    def list_by_category(cls, category: StrategyCategory) -> list[str]:
        """List strategies by category."""
        cls._ensure_initialized()
        return [
            key for key, meta in cls._strategies.items()
            if meta.category == category
        ]
    
    @classmethod
    def get_ml_strategies(cls) -> list[str]:
        """Get all ML-based strategies."""
        cls._ensure_initialized()
        ml_categories = {
            StrategyCategory.ML_CLASSIFIER,
            StrategyCategory.ML_REGRESSOR,
            StrategyCategory.ML_ENSEMBLE,
            StrategyCategory.ML_NEURAL,
            StrategyCategory.ML_ALPHA,
            StrategyCategory.ML_ALPHA_V2,
        }
        return [
            key for key, meta in cls._strategies.items()
            if meta.category in ml_categories
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_strategy(
    strategy_type: str,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseStrategy:
    """
    Factory function to create any strategy.
    
    Args:
        strategy_type: Strategy key (e.g., "macd", "alpha_ml_v2")
        config: Strategy configuration dictionary
        **kwargs: Additional parameters
    
    Returns:
        Configured strategy instance
    
    Example:
        strategy = create_strategy("alpha_ml_v2", {
            "use_lightgbm": True,
            "use_xgboost": True,
            "min_confidence": 0.55,
        })
    """
    return StrategyRegistry.create(strategy_type, config, **kwargs)


def list_available_strategies() -> dict[str, str]:
    """
    List all available strategies with descriptions.
    
    Returns:
        Dictionary of strategy_key -> description
    """
    StrategyRegistry._ensure_initialized()
    return {
        key: meta.description
        for key, meta in StrategyRegistry._strategies.items()
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Base ===
    "StrategyState",
    "SignalAction",
    "StrategyConfig",
    "BaseStrategy",
    "StrategyMetrics",
    
    # === Momentum - Configs ===
    "MACDStrategyConfig",
    "RSIMomentumConfig",
    "BreakoutStrategyConfig",
    "DualMomentumConfig",
    # === Momentum - Strategies ===
    "MACDStrategy",
    "RSIMomentumStrategy",
    "BreakoutStrategy",
    "DualMomentumStrategy",
    
    # === Statistical - Configs ===
    "PairsTradingConfig",
    "CointegrationConfig",
    "KalmanFilterConfig",
    "OrnsteinUhlenbeckConfig",
    # === Statistical - Strategies ===
    "PairsTradingStrategy",
    "CointegrationStrategy",
    "KalmanFilterStrategy",
    "OrnsteinUhlenbeckStrategy",
    
    # === ML Strategy (Basic) - Types ===
    "MLModel",
    "PredictionType",
    # === ML Strategy (Basic) - Configs ===
    "MLStrategyConfig",
    "MLClassifierConfig",
    "MLRegressorConfig",
    "EnsembleMLConfig",
    "NeuralNetConfig",
    # === ML Strategy (Basic) - Strategies ===
    "BaseMLStrategy",
    "MLClassifierStrategy",
    "MLRegressorStrategy",
    "EnsembleMLStrategy",
    "NeuralNetStrategy",
    # === ML Strategy (Basic) - Factory ===
    "create_ml_strategy",
    
    # === Alpha ML V1 (Deprecated) ===
    "MarketRegime",
    "ModelType",
    "SignalStrengthLevel",
    "AlphaMLConfig",
    "AlphaMLStrategy",
    "create_alpha_ml_strategy",
    
    # === Alpha ML V2 (Recommended) ===
    "PredictionMode",
    "AlphaMLConfigV2",
    "AlphaMLStrategyV2",
    "create_alpha_ml_strategy_v2",
    
    # === Registry ===
    "StrategyCategory",
    "StrategyMetadata",
    "StrategyRegistry",
    
    # === Convenience Functions ===
    "create_strategy",
    "list_available_strategies",
]