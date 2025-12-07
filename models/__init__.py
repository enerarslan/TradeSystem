"""
Models Module
=============

Production-grade machine learning models for algorithmic trading.
Implements JPMorgan-level ML standards with proper validation.

Components:
- base: Model base classes and registry
- classifiers: Gradient boosting, ensemble classifiers
- deep: LSTM, Transformer, TCN neural networks
- reinforcement: DQN, PPO reinforcement learning
- training: Training pipeline with Optuna optimization

Model Categories:
- Classifiers: LightGBM, XGBoost, CatBoost, RandomForest
- Ensembles: Stacking, Voting
- Deep Learning: LSTM, Transformer, TCN
- Reinforcement Learning: DQN, PPO

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

# =============================================================================
# BASE MODULE
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
# CLASSIFIER MODELS
# =============================================================================

from models.classifiers import (
    # LightGBM
    LightGBMClassifierConfig,
    LightGBMClassifier,
    # XGBoost
    XGBoostClassifierConfig,
    XGBoostClassifier,
    # CatBoost
    CatBoostClassifierConfig,
    CatBoostClassifier,
    # Random Forest
    RandomForestClassifierConfig,
    RandomForestClassifier,
    # Extra Trees
    ExtraTreesClassifierConfig,
    ExtraTreesClassifier,
    # Stacking
    StackingClassifierConfig,
    StackingClassifier,
    # Voting
    VotingClassifierConfig,
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
]