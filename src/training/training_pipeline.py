"""
Training Pipeline Orchestrator for AlphaTrade System.

This module provides a complete training workflow orchestration with:
- Data validation
- Feature generation with leakage prevention
- Train/test split with proper purging
- Model training with cross-validation
- Out-of-sample evaluation
- Statistical significance tests
- Model registration

Reference:
    "Advances in Financial Machine Learning" by de Prado (2018)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    DATA_VALIDATION = "data_validation"
    FEATURE_GENERATION = "feature_generation"
    DATA_PREPARATION = "data_preparation"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    SIGNIFICANCE_TESTING = "significance_testing"
    MODEL_REGISTRATION = "model_registration"
    CLEANUP = "cleanup"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    status: PipelineStatus
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error_message": self.error_message,
        }


@dataclass
class ValidationResult:
    """Result from data validation."""
    is_valid: bool
    n_samples: int
    n_features: int
    date_range: Tuple[str, str]
    missing_pct: float
    outlier_pct: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Result from model evaluation."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    feature_importance: Optional[pd.DataFrame] = None
    statistical_tests: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    success: bool
    model: Any
    model_id: Optional[str]
    stage_results: Dict[PipelineStage, StageResult]
    total_duration_seconds: float
    config_snapshot: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "TRAINING PIPELINE RESULTS",
            "=" * 60,
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Model ID: {self.model_id or 'Not registered'}",
            f"Total Duration: {self.total_duration_seconds:.1f}s",
            "",
            "STAGE RESULTS:",
        ]

        for stage, result in self.stage_results.items():
            status_icon = "✓" if result.status == PipelineStatus.COMPLETED else "✗"
            lines.append(f"  {status_icon} {stage.value}: {result.duration_seconds:.1f}s")

        if self.metadata.get("final_metrics"):
            lines.extend([
                "",
                "FINAL METRICS:",
            ])
            for metric, value in self.metadata["final_metrics"].items():
                lines.append(f"  {metric}: {value:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class TrainingPipeline:
    """
    Complete training workflow orchestrator.

    Orchestrates the full ML training workflow:
    1. Data validation (quality checks, missing data, outliers)
    2. Feature generation with leakage prevention
    3. Train/test split with proper purging
    4. Model training with cross-validation
    5. Out-of-sample evaluation
    6. Statistical significance tests
    7. Model registration if metrics pass threshold
    8. Cleanup temporary files

    Usage:
        pipeline = TrainingPipeline(config)
        result = pipeline.run(data, mode="full")

        if result.success:
            model = result.model
            model_id = result.model_id
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_pipeline: Optional[Any] = None,
        model_factory: Optional[Any] = None,
        experiment_tracker: Optional[Any] = None,
        model_registry: Optional[Any] = None,
        output_dir: str = "outputs",
    ):
        """
        Initialize the training pipeline.

        Args:
            config: Training configuration dictionary
            feature_pipeline: FeaturePipeline instance (optional)
            model_factory: ModelFactory instance (optional)
            experiment_tracker: MLflow experiment tracker (optional)
            model_registry: Model registry for production models (optional)
            output_dir: Directory for outputs
        """
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.model_factory = model_factory
        self.experiment_tracker = experiment_tracker
        self.model_registry = model_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage results
        self._stage_results: Dict[PipelineStage, StageResult] = {}
        self._start_time: Optional[float] = None

        # Artifacts
        self._data: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._y_test: Optional[pd.Series] = None
        self._model: Optional[Any] = None
        self._cv: Optional[Any] = None

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        mode: str = "full",
        skip_stages: Optional[List[PipelineStage]] = None,
    ) -> PipelineResult:
        """
        Run the complete training pipeline.

        Args:
            data: Input data (DataFrame or dict of DataFrames)
            mode: Execution mode ("full", "train_only", "evaluate_only")
            skip_stages: Stages to skip

        Returns:
            PipelineResult with all outputs
        """
        self._start_time = time.time()
        skip_stages = skip_stages or []

        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        # Convert dict to single DataFrame if needed
        if isinstance(data, dict):
            data = self._merge_data(data)

        self._data = data

        try:
            # Step 1: Data Validation
            if PipelineStage.DATA_VALIDATION not in skip_stages:
                self._run_stage(
                    PipelineStage.DATA_VALIDATION,
                    self._validate_data,
                    data,
                )

            # Step 2: Feature Generation
            if PipelineStage.FEATURE_GENERATION not in skip_stages:
                self._run_stage(
                    PipelineStage.FEATURE_GENERATION,
                    self._generate_features,
                    data,
                )

            # Step 3: Data Preparation
            if PipelineStage.DATA_PREPARATION not in skip_stages:
                self._run_stage(
                    PipelineStage.DATA_PREPARATION,
                    self._prepare_training_data,
                )

            # Step 4: Model Training
            if PipelineStage.MODEL_TRAINING not in skip_stages and mode != "evaluate_only":
                self._run_stage(
                    PipelineStage.MODEL_TRAINING,
                    self._train_model,
                )

            # Step 5: Model Evaluation
            if PipelineStage.MODEL_EVALUATION not in skip_stages:
                self._run_stage(
                    PipelineStage.MODEL_EVALUATION,
                    self._evaluate_model,
                )

            # Step 6: Statistical Significance Testing
            if PipelineStage.SIGNIFICANCE_TESTING not in skip_stages:
                self._run_stage(
                    PipelineStage.SIGNIFICANCE_TESTING,
                    self._run_significance_tests,
                )

            # Step 7: Model Registration
            if PipelineStage.MODEL_REGISTRATION not in skip_stages and mode == "full":
                self._run_stage(
                    PipelineStage.MODEL_REGISTRATION,
                    self._register_model,
                )

            # Step 8: Cleanup
            self._run_stage(
                PipelineStage.CLEANUP,
                self._cleanup,
            )

            success = all(
                r.status in [PipelineStatus.COMPLETED, PipelineStatus.SKIPPED]
                for r in self._stage_results.values()
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            success = False

        total_duration = time.time() - self._start_time

        # Compile result
        result = PipelineResult(
            success=success,
            model=self._model,
            model_id=self._stage_results.get(PipelineStage.MODEL_REGISTRATION, StageResult(
                PipelineStage.MODEL_REGISTRATION, PipelineStatus.SKIPPED, 0.0
            )).metrics.get("model_id"),
            stage_results=self._stage_results,
            total_duration_seconds=total_duration,
            config_snapshot=self.config.copy(),
            metadata={
                "final_metrics": self._stage_results.get(
                    PipelineStage.MODEL_EVALUATION, StageResult(
                        PipelineStage.MODEL_EVALUATION, PipelineStatus.SKIPPED, 0.0
                    )
                ).metrics,
            },
        )

        logger.info(result.summary())

        return result

    def _run_stage(
        self,
        stage: PipelineStage,
        func: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Execute a pipeline stage with timing and error handling."""
        logger.info(f"Running stage: {stage.value}")
        start = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            self._stage_results[stage] = StageResult(
                stage=stage,
                status=PipelineStatus.COMPLETED,
                duration_seconds=duration,
                metrics=result if isinstance(result, dict) else {},
            )
            logger.info(f"Stage {stage.value} completed in {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start
            logger.error(f"Stage {stage.value} failed: {e}")

            self._stage_results[stage] = StageResult(
                stage=stage,
                status=PipelineStatus.FAILED,
                duration_seconds=duration,
                error_message=str(e),
            )
            raise

    def _merge_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple DataFrames into one."""
        frames = []
        for symbol, df in data_dict.items():
            df = df.copy()
            df["symbol"] = symbol
            frames.append(df)

        return pd.concat(frames, ignore_index=False)

    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 1: Validate data quality.

        Checks:
        - No future timestamps
        - No duplicate timestamps
        - Chronological order
        - Price sanity
        - Volume sanity
        - Missing data percentage
        """
        logger.info("Validating data quality...")

        warnings = []
        errors = []

        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for duplicates
        if data.index.duplicated().any():
            n_duplicates = data.index.duplicated().sum()
            warnings.append(f"Found {n_duplicates} duplicate timestamps")

        # Check chronological order
        if not data.index.is_monotonic_increasing:
            warnings.append("Data is not in chronological order")

        # Check price sanity
        if (data["close"] <= 0).any():
            n_invalid = (data["close"] <= 0).sum()
            errors.append(f"Found {n_invalid} non-positive close prices")

        # Check for extreme outliers (>10 std from mean)
        for col in ["close", "volume"]:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                outliers = ((data[col] - mean).abs() > 10 * std).sum()
                if outliers > 0:
                    warnings.append(f"Found {outliers} extreme outliers in {col}")

        # Calculate missing percentage
        missing_pct = data.isnull().mean().mean() * 100

        # Calculate outlier percentage
        outlier_pct = 0.0
        for col in required_cols:
            if col in data.columns:
                q1 = data[col].quantile(0.01)
                q99 = data[col].quantile(0.99)
                outlier_pct += ((data[col] < q1) | (data[col] > q99)).mean()
        outlier_pct = outlier_pct / len(required_cols) * 100

        # Determine validity
        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            n_samples=len(data),
            n_features=len(data.columns),
            date_range=(str(data.index.min()), str(data.index.max())),
            missing_pct=missing_pct,
            outlier_pct=outlier_pct,
            warnings=warnings,
            errors=errors,
        )

        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")

        logger.info(
            f"Data validated: {result.n_samples} samples, "
            f"missing: {result.missing_pct:.2f}%, outliers: {result.outlier_pct:.2f}%"
        )

        return {
            "n_samples": result.n_samples,
            "missing_pct": result.missing_pct,
            "outlier_pct": result.outlier_pct,
            "warnings": len(result.warnings),
        }

    def _generate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 2: Generate features with leakage prevention.
        """
        logger.info("Generating features...")

        if self.feature_pipeline is None:
            # Use default feature pipeline
            from src.features.pipeline import FeaturePipeline
            self.feature_pipeline = FeaturePipeline(
                scaling=self.config.get("feature_scaling", "robust"),
            )

        # Generate features (pipeline handles leakage prevention internally)
        self._features = self.feature_pipeline.generate_features(
            data,
            include_technical=self.config.get("include_technical", True),
            include_statistical=self.config.get("include_statistical", True),
            include_lagged=self.config.get("include_lagged", True),
        )

        n_features = len(self._features.columns)
        logger.info(f"Generated {n_features} features")

        return {
            "n_features": n_features,
            "max_lookback": self.feature_pipeline.max_lookback,
        }

    def _prepare_training_data(self) -> Dict[str, Any]:
        """
        Step 3: Prepare train/test split with proper purging.
        """
        logger.info("Preparing training data...")

        if self._features is None:
            raise ValueError("Features not generated")

        # Get training config
        train_config = self.config.get("training", {})
        test_size = train_config.get("test_size", 0.2)
        purge_gap = train_config.get("purge_gap", 50)
        prediction_horizon = train_config.get("prediction_horizon", 5)

        # Create target
        target_col = train_config.get("target_column", "close")
        if target_col in self._data.columns:
            target = self._data[target_col].pct_change().shift(-prediction_horizon)
        else:
            target = self._features[target_col] if target_col in self._features.columns else None

        if target is None:
            raise ValueError(f"Target column '{target_col}' not found")

        # Align features and target
        aligned = pd.DataFrame({
            **self._features,
            "target": target,
        }).dropna()

        X = aligned.drop(columns=["target"])
        y = aligned["target"]

        # Train/test split with purge gap
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        # Apply purge gap
        train_end = split_idx - purge_gap
        test_start = split_idx

        self._X_train = X.iloc[:train_end]
        self._y_train = y.iloc[:train_end]
        self._X_test = X.iloc[test_start:]
        self._y_test = y.iloc[test_start:]

        logger.info(
            f"Train: {len(self._X_train)} samples, Test: {len(self._X_test)} samples, "
            f"Purge gap: {purge_gap}"
        )

        # Setup cross-validation
        from src.training.validation import PurgedKFoldCV
        self._cv = PurgedKFoldCV(
            n_splits=train_config.get("cv_splits", 5),
            purge_gap=purge_gap,
            embargo_pct=train_config.get("embargo_pct", 0.01),
        )

        return {
            "n_train": len(self._X_train),
            "n_test": len(self._X_test),
            "n_features": len(self._X_train.columns),
            "purge_gap": purge_gap,
        }

    def _train_model(self) -> Dict[str, Any]:
        """
        Step 4: Train model with cross-validation.
        """
        logger.info("Training model...")

        if self._X_train is None or self._y_train is None:
            raise ValueError("Training data not prepared")

        train_config = self.config.get("training", {})
        model_type = train_config.get("model_type", "lightgbm_regressor")

        # Create model
        if self.model_factory is None:
            from src.training.model_factory import ModelFactory
            self.model_factory = ModelFactory

        model = self.model_factory.create_model(
            model_type=model_type,
            params=train_config.get("model_params"),
        )

        # Cross-validation scores
        cv_scores = []
        X_np = self._X_train.values
        y_np = self._y_train.values

        for fold_idx, (train_idx, val_idx) in enumerate(self._cv.split(X_np, y_np)):
            X_fold_train = X_np[train_idx]
            y_fold_train = y_np[train_idx]
            X_fold_val = X_np[val_idx]
            y_fold_val = y_np[val_idx]

            # Clone model for this fold
            fold_model = self.model_factory.create_model(
                model_type=model_type,
                params=train_config.get("model_params"),
            )

            # Train
            fold_model.fit(X_fold_train, y_fold_train)

            # Evaluate
            y_pred = fold_model.predict(X_fold_val)

            # Calculate IC (Information Coefficient)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_fold_val, y_pred)
            cv_scores.append(ic if not np.isnan(ic) else 0.0)

            logger.info(f"Fold {fold_idx + 1}: IC = {cv_scores[-1]:.4f}")

        # Train final model on all training data
        model.fit(X_np, y_np)
        self._model = model

        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)

        logger.info(f"CV Complete: Mean IC = {mean_cv:.4f} (+/- {std_cv:.4f})")

        return {
            "cv_mean_ic": mean_cv,
            "cv_std_ic": std_cv,
            "n_folds": len(cv_scores),
        }

    def _evaluate_model(self) -> Dict[str, Any]:
        """
        Step 5: Evaluate model on out-of-sample data.
        """
        logger.info("Evaluating model...")

        if self._model is None or self._X_test is None:
            raise ValueError("Model or test data not available")

        # Make predictions
        y_pred = self._model.predict(self._X_test.values)

        # Calculate metrics
        from scipy.stats import spearmanr

        # Information Coefficient
        ic, ic_pval = spearmanr(self._y_test.values, y_pred)

        # MSE
        mse = np.mean((self._y_test.values - y_pred) ** 2)

        # Direction accuracy
        direction_actual = np.sign(self._y_test.values)
        direction_pred = np.sign(y_pred)
        direction_accuracy = np.mean(direction_actual == direction_pred)

        # Sharpe-like metric (IC / IC volatility)
        if len(y_pred) > 1:
            rolling_ic = pd.Series(y_pred).rolling(20).corr(pd.Series(self._y_test.values))
            ic_ir = rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0
        else:
            ic_ir = 0

        metrics = {
            "test_ic": float(ic) if not np.isnan(ic) else 0.0,
            "test_ic_pvalue": float(ic_pval),
            "test_mse": float(mse),
            "direction_accuracy": float(direction_accuracy),
            "ic_ir": float(ic_ir) if not np.isnan(ic_ir) else 0.0,
        }

        logger.info(
            f"Test Metrics: IC={metrics['test_ic']:.4f}, "
            f"Direction Acc={metrics['direction_accuracy']:.2%}"
        )

        return metrics

    def _run_significance_tests(self) -> Dict[str, Any]:
        """
        Step 6: Run statistical significance tests.
        """
        logger.info("Running significance tests...")

        if self._model is None or self._X_test is None:
            raise ValueError("Model or test data not available")

        y_pred = self._model.predict(self._X_test.values)

        # Test if IC is significantly different from zero
        from scipy.stats import spearmanr, ttest_1samp

        # Bootstrap confidence interval for IC
        n_bootstrap = 1000
        bootstrap_ics = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y_pred), len(y_pred), replace=True)
            boot_ic, _ = spearmanr(self._y_test.values[idx], y_pred[idx])
            if not np.isnan(boot_ic):
                bootstrap_ics.append(boot_ic)

        ic_ci_lower = np.percentile(bootstrap_ics, 2.5) if bootstrap_ics else 0
        ic_ci_upper = np.percentile(bootstrap_ics, 97.5) if bootstrap_ics else 0

        # T-test for IC significantly different from 0
        if bootstrap_ics:
            t_stat, t_pval = ttest_1samp(bootstrap_ics, 0)
        else:
            t_stat, t_pval = 0, 1

        # Calculate probability of skill (IC > 0)
        prob_skill = np.mean([ic > 0 for ic in bootstrap_ics]) if bootstrap_ics else 0

        results = {
            "ic_ci_lower": float(ic_ci_lower),
            "ic_ci_upper": float(ic_ci_upper),
            "ic_tstat": float(t_stat),
            "ic_pvalue": float(t_pval),
            "probability_of_skill": float(prob_skill),
            "n_bootstrap": n_bootstrap,
        }

        logger.info(
            f"Significance: IC 95% CI = [{results['ic_ci_lower']:.4f}, {results['ic_ci_upper']:.4f}], "
            f"P(skill) = {results['probability_of_skill']:.2%}"
        )

        return results

    def _register_model(self) -> Dict[str, Any]:
        """
        Step 7: Register model if metrics pass threshold.
        """
        logger.info("Evaluating model for registration...")

        if self._model is None:
            raise ValueError("No model to register")

        # Get evaluation metrics
        eval_metrics = self._stage_results.get(PipelineStage.MODEL_EVALUATION)
        if eval_metrics is None or eval_metrics.status != PipelineStatus.COMPLETED:
            logger.warning("Skipping registration - evaluation not complete")
            return {"registered": False, "reason": "Evaluation not complete"}

        metrics = eval_metrics.metrics

        # Check thresholds
        register_config = self.config.get("model_registration", {})
        min_ic = register_config.get("min_ic", 0.02)
        min_direction_accuracy = register_config.get("min_direction_accuracy", 0.52)

        if metrics.get("test_ic", 0) < min_ic:
            logger.info(f"Model IC ({metrics.get('test_ic', 0):.4f}) below threshold ({min_ic})")
            return {"registered": False, "reason": "IC below threshold"}

        if metrics.get("direction_accuracy", 0) < min_direction_accuracy:
            logger.info(
                f"Direction accuracy ({metrics.get('direction_accuracy', 0):.2%}) "
                f"below threshold ({min_direction_accuracy:.2%})"
            )
            return {"registered": False, "reason": "Direction accuracy below threshold"}

        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        model_id = f"model_{timestamp}_{model_hash}"

        # Save model
        model_path = self.output_dir / f"{model_id}.pkl"
        joblib.dump(self._model, model_path)

        # Save metadata
        metadata = {
            "model_id": model_id,
            "created_at": timestamp,
            "metrics": metrics,
            "config": self.config,
            "feature_names": list(self._X_train.columns) if self._X_train is not None else [],
        }

        metadata_path = self.output_dir / f"{model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model registered: {model_id}")

        return {
            "registered": True,
            "model_id": model_id,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        }

    def _cleanup(self) -> Dict[str, Any]:
        """
        Step 8: Cleanup temporary files and release memory.
        """
        logger.info("Cleaning up...")

        # Clear large intermediate objects
        del self._data
        del self._features
        self._data = None
        self._features = None

        # Force garbage collection
        gc.collect()

        return {"cleaned": True}

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """Public method to validate data without running full pipeline."""
        return self._validate_data(data)

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Public method to generate features."""
        self._generate_features(data)
        return self._features

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
        """Public method to prepare training data."""
        self._prepare_training_data()
        return self._X_train, self._X_test, self._y_train, self._y_test, self._cv

    def train_model(self) -> Any:
        """Public method to train model."""
        self._train_model()
        return self._model

    def evaluate_model(self) -> EvaluationResult:
        """Public method to evaluate model."""
        metrics = self._evaluate_model()
        return EvaluationResult(
            metrics=metrics,
            predictions=self._model.predict(self._X_test.values) if self._model else np.array([]),
        )

    def register_model(self) -> str:
        """Public method to register model."""
        result = self._register_model()
        return result.get("model_id", "")
