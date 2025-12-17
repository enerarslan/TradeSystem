"""
Training module for institutional-grade ML model development.

This module provides:
- Experiment tracking with MLflow
- Model registry and versioning
- Purged cross-validation for time-series
- Hyperparameter optimization with Optuna
- Deep learning model implementations
- Custom financial loss functions

Designed to meet JPMorgan-level requirements for:
- Reproducible experiments
- Model governance and audit trails
- Statistical rigor in validation
- Production-ready model deployment
"""

from .experiment_tracker import ExperimentTracker
from .model_factory import ModelFactory
from .model_registry import ModelRegistry

__all__ = [
    "ExperimentTracker",
    "ModelFactory",
    "ModelRegistry",
]
