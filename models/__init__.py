"""
Models Module
=============

Machine learning models for the algorithmic trading platform.

Modules:
- base: Base model class and utilities
- classifiers: Gradient boosting and ensemble classifiers
- deep: Deep learning models (LSTM, Transformer, TCN)
- reinforcement: Reinforcement learning agents
- training: Training pipeline with Optuna optimization
- model_manager: Centralized model registry for all 46 symbols

Author: Algo Trading Platform
License: MIT
"""

# =============================================================================
# BASE MODEL COMPONENTS
# =============================================================================

from models.base import (
    # Enums
    ModelType,
    ModelState,
    ValidationMethod,
    # Config
    ModelConfig,
    # Metrics
    ClassificationMetrics,
    RegressionMetrics,
    ModelMetrics,
    # Base class
    BaseModel,
    # Registry
    ModelRegistry,
)

# =============================================================================
# CLASSIFIERS
# =============================================================================

from models.classifiers import (
    # Configs
    LightGBMClassifierConfig,
    XGBoostClassifierConfig,
    CatBoostClassifierConfig,
    RandomForestClassifierConfig,
    ExtraTreesClassifierConfig,
    StackingClassifierConfig,
    VotingClassifierConfig,
    # Models
    LightGBMClassifier,
    XGBoostClassifier,
    CatBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
    # Factory
    create_classifier,
)

# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

from models.deep import (
    # Configs
    DeepLearningConfig,
    LSTMConfig,
    TransformerConfig,
    TCNConfig,
    # Models
    DeepLearningModel,
    LSTMModel,
    TransformerModel,
    TCNModel,
    # Factory
    create_deep_model,
)

# =============================================================================
# REINFORCEMENT LEARNING MODELS
# =============================================================================

from models.reinforcement import (
    # Configs
    RLConfig,
    DQNConfig,
    PPOConfig,
    # Agents
    DQNAgent,
    PPOAgent,
    # Factory
    create_rl_agent,
    # Utilities
    ReplayBuffer,
    TradingEnvironment,
)

# =============================================================================
# TRAINING PIPELINE
# =============================================================================

from models.training import (
    # Enums
    OptimizationDirection,
    SamplerType,
    PrunerType,
    # Config
    OptimizationConfig,
    TrainingConfig,
    # Cross-validation
    PurgedKFold,
    CombinatorialPurgedKFold,
    # Optimizer
    HyperparameterOptimizer,
    # Pipeline
    TrainingPipeline,
    # Functions
    quick_train,
    auto_ml,
    # Param spaces
    PARAM_SPACES,
)

# =============================================================================
# MODEL MANAGER (NEW - for 46 symbol support)
# =============================================================================

from models.model_manager import (
    # Data classes
    ModelMetadata,
    SymbolModelRegistry,
    
    # Main manager
    ModelManager,
    
    # Convenience functions
    get_model_manager,
    save_model,
    load_model,
    list_models,
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === Base - Enums ===
    "ModelType",
    "ModelState",
    "ValidationMethod",
    # === Base - Config ===
    "ModelConfig",
    # === Base - Metrics ===
    "ClassificationMetrics",
    "RegressionMetrics",
    "ModelMetrics",
    # === Base - Classes ===
    "BaseModel",
    "ModelRegistry",
    
    # === Classifiers - Configs ===
    "LightGBMClassifierConfig",
    "XGBoostClassifierConfig",
    "CatBoostClassifierConfig",
    "RandomForestClassifierConfig",
    "ExtraTreesClassifierConfig",
    "StackingClassifierConfig",
    "VotingClassifierConfig",
    # === Classifiers - Models ===
    "LightGBMClassifier",
    "XGBoostClassifier",
    "CatBoostClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "StackingClassifier",
    "VotingClassifier",
    # === Classifiers - Factory ===
    "create_classifier",
    
    # === Deep Learning - Configs ===
    "DeepLearningConfig",
    "LSTMConfig",
    "TransformerConfig",
    "TCNConfig",
    # === Deep Learning - Models ===
    "DeepLearningModel",
    "LSTMModel",
    "TransformerModel",
    "TCNModel",
    # === Deep Learning - Factory ===
    "create_deep_model",
    
    # === Reinforcement Learning - Configs ===
    "RLConfig",
    "DQNConfig",
    "PPOConfig",
    # === Reinforcement Learning - Agents ===
    "DQNAgent",
    "PPOAgent",
    # === Reinforcement Learning - Factory ===
    "create_rl_agent",
    # === Reinforcement Learning - Utilities ===
    "ReplayBuffer",
    "TradingEnvironment",
    
    # === Training - Enums ===
    "OptimizationDirection",
    "SamplerType",
    "PrunerType",
    # === Training - Config ===
    "OptimizationConfig",
    "TrainingConfig",
    # === Training - Cross-validation ===
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    # === Training - Optimizer ===
    "HyperparameterOptimizer",
    # === Training - Pipeline ===
    "TrainingPipeline",
    # === Training - Functions ===
    "quick_train",
    "auto_ml",
    # === Training - Param Spaces ===
    "PARAM_SPACES",
    
    # === Model Manager - Data Classes ===
    "ModelMetadata",
    "SymbolModelRegistry",
    # === Model Manager - Main Class ===
    "ModelManager",
    # === Model Manager - Functions ===
    "get_model_manager",
    "save_model",
    "load_model",
    "list_models",
]