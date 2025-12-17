"""
Training module for institutional-grade ML model development.

This module provides:
- Experiment tracking with MLflow
- Model registry and versioning
- Model factory for standardized model creation
- Training orchestration with callbacks
- Purged cross-validation for time-series
- Walk-forward validation
- Hyperparameter optimization with Optuna
- Deep learning model implementations
- Custom financial loss functions

Designed to meet JPMorgan-level requirements for:
- Reproducible experiments
- Model governance and audit trails
- Statistical rigor in validation
- Production-ready model deployment
"""

from src.training.experiment_tracker import ExperimentTracker
from src.training.model_factory import ModelFactory
from src.training.model_registry import ModelRegistry
from src.training.trainer import Trainer, TrainingResult
from src.training.validation import (
    PurgedKFoldCV,
    CombinatorialPurgedKFoldCV,
    WalkForwardValidator,
)
from src.training.optimization import (
    OptunaOptimizer,
    MultiObjectiveOptimizer,
)

__all__ = [
    # Experiment tracking
    "ExperimentTracker",
    # Model factory and registry
    "ModelFactory",
    "ModelRegistry",
    # Training
    "Trainer",
    "TrainingResult",
    # Validation
    "PurgedKFoldCV",
    "CombinatorialPurgedKFoldCV",
    "WalkForwardValidator",
    # Optimization
    "OptunaOptimizer",
    "MultiObjectiveOptimizer",
]
