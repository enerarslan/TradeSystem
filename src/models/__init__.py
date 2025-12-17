"""
Models module for AlphaTrade system.

This module provides backward compatibility imports from the
new training module. For new code, import directly from:
- src.training.model_factory
- src.training.trainer
- src.training.deep_learning

Contains ML/Statistical models for:
- Alpha generation (via training.model_factory)
- Risk modeling (via risk.var_models)
- Execution optimization (via execution module)
- Model ensembles (via training.model_factory)
"""

# Re-export from training module for backward compatibility
from src.training import (
    ModelFactory,
    Trainer,
    TrainingResult,
    ExperimentTracker,
    ModelRegistry,
)

__all__ = [
    "ModelFactory",
    "Trainer",
    "TrainingResult",
    "ExperimentTracker",
    "ModelRegistry",
]
