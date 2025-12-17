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
- Preprocessing pipelines with data leakage prevention
- Probability calibration for reliable predictions
- Time-series data loading for deep learning

Designed to meet JPMorgan-level requirements for:
- Reproducible experiments
- Model governance and audit trails
- Statistical rigor in validation
- Production-ready model deployment
- Data lineage tracking
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
    WalkForwardOptimizer,
    WalkForwardOptimizationResult,
    AdaptiveOptimizer,
    OptimizationResult,
    optimize_model,
    walk_forward_optimize,
)

# Pipeline - Preprocessing with data leakage prevention
from src.training.pipeline import (
    ModelPipelineFactory,
    PipelineConfig,
    ScalerType,
    ImputerType,
    FinancialPreprocessor,
    Winsorizer,
    create_financial_pipeline,
)

# Calibration - Probability calibration for reliable predictions
from src.training.calibration import (
    ProbabilityCalibrator,
    CalibratedModel,
    CalibrationMethod,
    CalibrationResult,
    calibrate_model_predictions,
    plot_calibration_curve,
)

# Deep Learning - REQUIRED
from src.training.deep_learning import (
    LSTMPredictor,
    AttentionLSTM,
    TemporalFusionTransformer,
    SharpeLoss,
    SortinoLoss,
    MaxDrawdownLoss,
    CombinedFinancialLoss,
)

# Deep Learning Data Loading
from src.training.deep_learning.dataset import (
    TimeSeriesDataset,
    TimeSeriesDataModule,
    MultiHorizonDataset,
    create_dataloaders,
    create_cv_dataloaders,
    prepare_data_for_dl,
)

# Drift Detection
from src.training.drift_detection import (
    DriftDetector,
    DriftResult,
    DriftThresholds,
    DriftSeverity,
    DriftType,
    DriftRecommendation,
)

# Training Pipeline Orchestrator
from src.training.training_pipeline import (
    TrainingPipeline,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
    ValidationResult,
    EvaluationResult,
)

# Checkpointing - Fault-tolerant training with resumption
from src.training.checkpointing import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetadata,
    TrainingState,
    ResumableTrainer,
    create_checkpoint_callback,
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
    "WalkForwardOptimizer",
    "WalkForwardOptimizationResult",
    "AdaptiveOptimizer",
    "OptimizationResult",
    "optimize_model",
    "walk_forward_optimize",
    # Pipeline - Preprocessing
    "ModelPipelineFactory",
    "PipelineConfig",
    "ScalerType",
    "ImputerType",
    "FinancialPreprocessor",
    "Winsorizer",
    "create_financial_pipeline",
    # Calibration
    "ProbabilityCalibrator",
    "CalibratedModel",
    "CalibrationMethod",
    "CalibrationResult",
    "calibrate_model_predictions",
    "plot_calibration_curve",
    # Deep Learning - Models
    "LSTMPredictor",
    "AttentionLSTM",
    "TemporalFusionTransformer",
    "SharpeLoss",
    "SortinoLoss",
    "MaxDrawdownLoss",
    "CombinedFinancialLoss",
    # Deep Learning - Data Loading
    "TimeSeriesDataset",
    "TimeSeriesDataModule",
    "MultiHorizonDataset",
    "create_dataloaders",
    "create_cv_dataloaders",
    "prepare_data_for_dl",
    # Drift Detection
    "DriftDetector",
    "DriftResult",
    "DriftThresholds",
    "DriftSeverity",
    "DriftType",
    "DriftRecommendation",
    # Training Pipeline
    "TrainingPipeline",
    "PipelineResult",
    "PipelineStage",
    "PipelineStatus",
    "ValidationResult",
    "EvaluationResult",
    # Checkpointing
    "CheckpointManager",
    "CheckpointConfig",
    "CheckpointMetadata",
    "TrainingState",
    "ResumableTrainer",
    "create_checkpoint_callback",
]
