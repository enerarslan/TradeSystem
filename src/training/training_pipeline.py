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

# Optional GPU detection
CUDA_AVAILABLE = False
MPS_AVAILABLE = False
LIGHTGBM_GPU_AVAILABLE = False

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    pass

# Check LightGBM GPU support independently (doesn't need PyTorch)
try:
    import lightgbm as lgb
    import numpy as np
    # Quick test to check GPU support
    test_data = lgb.Dataset(np.array([[1,2],[3,4]], dtype=np.float32), label=np.array([0,1], dtype=np.float32))
    test_params = {'device': 'gpu', 'verbose': -1, 'num_iterations': 1}
    try:
        lgb.train(test_params, test_data, num_boost_round=1)
        LIGHTGBM_GPU_AVAILABLE = True
    except lgb.basic.LightGBMError:
        LIGHTGBM_GPU_AVAILABLE = False
except ImportError:
    pass

# Optional MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Optional TimescaleDB
try:
    import psycopg2
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False

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
            status_icon = "[OK]" if result.status == PipelineStatus.COMPLETED else "[FAIL]"
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
        use_gpu: bool = True,
        use_mlflow: bool = True,
        use_timescaledb: bool = False,
        timescale_config: Optional[Dict[str, Any]] = None,
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
            use_gpu: Enable GPU acceleration for supported models
            use_mlflow: Enable MLflow experiment tracking
            use_timescaledb: Enable TimescaleDB for data storage
            timescale_config: TimescaleDB connection config
        """
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.model_factory = model_factory
        self.experiment_tracker = experiment_tracker
        self.model_registry = model_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # GPU Configuration - check both PyTorch CUDA and LightGBM GPU
        self.use_gpu = use_gpu and (CUDA_AVAILABLE or MPS_AVAILABLE or LIGHTGBM_GPU_AVAILABLE)
        self.device = self._detect_device() if self.use_gpu else "cpu"
        self.lightgbm_gpu = LIGHTGBM_GPU_AVAILABLE  # Track LightGBM GPU separately

        # MLflow Configuration
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self._mlflow_run_id: Optional[str] = None

        # TimescaleDB Configuration
        self.use_timescaledb = use_timescaledb and TIMESCALE_AVAILABLE
        self.timescale_config = timescale_config or {}
        self._db_connection = None

        # Log capabilities
        logger.info(f"Pipeline initialized:")
        gpu_status = "DISABLED"
        if self.use_gpu:
            if self.device == "cuda":
                gpu_status = f"ENABLED (CUDA)"
            elif self.device == "mps":
                gpu_status = f"ENABLED (MPS)"
            elif self.lightgbm_gpu:
                gpu_status = f"ENABLED (LightGBM GPU)"
        logger.info(f"  GPU: {gpu_status}")
        logger.info(f"  MLflow: {'ENABLED' if self.use_mlflow else 'DISABLED'}")
        logger.info(f"  TimescaleDB: {'ENABLED' if self.use_timescaledb else 'DISABLED'}")

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

    def _detect_device(self) -> str:
        """Detect best available device for training."""
        if CUDA_AVAILABLE:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {gpu_name}")
            return "cuda"
        elif MPS_AVAILABLE:
            logger.info("Apple MPS (Metal) detected")
            return "mps"
        elif LIGHTGBM_GPU_AVAILABLE:
            logger.info("LightGBM GPU support detected (no PyTorch CUDA)")
            return "lightgbm_gpu"
        return "cpu"

    def _init_mlflow(self, run_name: str) -> None:
        """Initialize MLflow tracking."""
        if not self.use_mlflow:
            return

        try:
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("alphatrade_pipeline")

            run = mlflow.start_run(run_name=run_name)
            self._mlflow_run_id = run.info.run_id

            # Log config as params
            flat_config = self._flatten_dict(self.config)
            for key, value in flat_config.items():
                try:
                    mlflow.log_param(key[:250], str(value)[:250])  # MLflow limits
                except Exception:
                    pass

            logger.info(f"MLflow run started: {self._mlflow_run_id}")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")
            self.use_mlflow = False

    def _log_mlflow_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if not self.use_mlflow or not self._mlflow_run_id:
            return

        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"MLflow metric logging failed: {e}")

    def _end_mlflow(self, success: bool) -> None:
        """End MLflow run."""
        if not self.use_mlflow or not self._mlflow_run_id:
            return

        try:
            mlflow.log_param("pipeline_success", success)
            mlflow.end_run()
            logger.info(f"MLflow run ended: {self._mlflow_run_id}")
        except Exception as e:
            logger.warning(f"MLflow end failed: {e}")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _init_timescaledb(self) -> None:
        """Initialize TimescaleDB connection."""
        if not self.use_timescaledb:
            return

        try:
            self._db_connection = psycopg2.connect(
                host=self.timescale_config.get("host", "localhost"),
                port=self.timescale_config.get("port", 5432),
                database=self.timescale_config.get("database", "alphatrade"),
                user=self.timescale_config.get("user", "postgres"),
                password=self.timescale_config.get("password", ""),
            )
            logger.info("TimescaleDB connection established")
        except Exception as e:
            logger.warning(f"TimescaleDB connection failed: {e}")
            self.use_timescaledb = False
            self._db_connection = None

    def _save_to_timescaledb(self, table: str, data: pd.DataFrame) -> None:
        """Save data to TimescaleDB."""
        if not self.use_timescaledb or self._db_connection is None:
            return

        try:
            from io import StringIO
            buffer = StringIO()
            data.to_csv(buffer, index=True, header=False)
            buffer.seek(0)

            cursor = self._db_connection.cursor()
            cursor.copy_from(buffer, table, sep=',')
            self._db_connection.commit()
            logger.info(f"Saved {len(data)} rows to TimescaleDB table: {table}")
        except Exception as e:
            logger.warning(f"TimescaleDB save failed: {e}")

    def _save_features_to_db(self, features: pd.DataFrame, symbol: str = "UNKNOWN") -> None:
        """Save features to TimescaleDB features_store table."""
        if not self.use_timescaledb or self._db_connection is None:
            return

        try:
            cursor = self._db_connection.cursor()

            # Melt features DataFrame for storage
            for col in features.columns:
                for idx, value in features[col].items():
                    if pd.notna(value):
                        cursor.execute(
                            """
                            INSERT INTO features_store (time, symbol, feature_name, feature_value)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (time, symbol, feature_name) DO UPDATE
                            SET feature_value = EXCLUDED.feature_value
                            """,
                            (idx, symbol, col, float(value))
                        )

            self._db_connection.commit()
            logger.info(f"Saved {len(features)} rows x {len(features.columns)} features to TimescaleDB")
        except Exception as e:
            logger.warning(f"TimescaleDB features save failed: {e}")

    def _save_predictions_to_db(self, predictions: np.ndarray, model_id: str) -> None:
        """Save model predictions to TimescaleDB."""
        if not self.use_timescaledb or self._db_connection is None:
            return

        try:
            cursor = self._db_connection.cursor()

            # Get test data indices for timestamps
            if self._X_test is not None and hasattr(self._X_test, 'index'):
                for i, (idx, pred) in enumerate(zip(self._X_test.index, predictions)):
                    # Determine symbol from data if available
                    symbol = "UNKNOWN"
                    if self._data is not None and "symbol" in self._data.columns:
                        try:
                            symbol = self._data.loc[idx, "symbol"]
                        except Exception:
                            pass

                    cursor.execute(
                        """
                        INSERT INTO model_predictions (time, symbol, model_id, prediction, horizon)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol, model_id) DO UPDATE
                        SET prediction = EXCLUDED.prediction
                        """,
                        (idx, symbol, model_id, float(pred), self.config.get("training", {}).get("prediction_horizon", 5))
                    )

            self._db_connection.commit()
            logger.info(f"Saved {len(predictions)} predictions to TimescaleDB")
        except Exception as e:
            logger.warning(f"TimescaleDB predictions save failed: {e}")

    def _save_metrics_to_db(self, metrics: Dict[str, float], metric_type: str, entity_id: str) -> None:
        """Save performance metrics to TimescaleDB."""
        if not self.use_timescaledb or self._db_connection is None:
            return

        try:
            cursor = self._db_connection.cursor()

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    cursor.execute(
                        """
                        INSERT INTO performance_metrics (time, metric_type, metric_name, metric_value, entity_id)
                        VALUES (NOW(), %s, %s, %s, %s)
                        """,
                        (metric_type, metric_name, float(value), entity_id)
                    )

            self._db_connection.commit()
            logger.info(f"Saved {len(metrics)} metrics to TimescaleDB")
        except Exception as e:
            logger.warning(f"TimescaleDB metrics save failed: {e}")

    def _register_model_to_db(self, model_id: str, metrics: Dict[str, Any]) -> None:
        """Register model in TimescaleDB model_registry."""
        if not self.use_timescaledb or self._db_connection is None:
            return

        try:
            import json as json_module
            cursor = self._db_connection.cursor()

            cursor.execute(
                """
                INSERT INTO model_registry (
                    model_id, model_type, status,
                    cv_mean_ic, cv_std_ic, test_ic, direction_accuracy,
                    config, feature_names, mlflow_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    updated_at = NOW(),
                    cv_mean_ic = EXCLUDED.cv_mean_ic,
                    test_ic = EXCLUDED.test_ic
                """,
                (
                    model_id,
                    self.config.get("training", {}).get("model_type", "unknown"),
                    "active",
                    metrics.get("cv_mean_ic"),
                    metrics.get("cv_std_ic"),
                    metrics.get("test_ic"),
                    metrics.get("direction_accuracy"),
                    json_module.dumps(self.config),
                    json_module.dumps(list(self._X_train.columns) if self._X_train is not None else []),
                    self._mlflow_run_id
                )
            )

            self._db_connection.commit()
            logger.info(f"Model {model_id} registered in TimescaleDB")
        except Exception as e:
            logger.warning(f"TimescaleDB model registration failed: {e}")

    def _close_timescaledb(self) -> None:
        """Close TimescaleDB connection."""
        if self._db_connection:
            try:
                self._db_connection.close()
                logger.info("TimescaleDB connection closed")
            except Exception:
                pass

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
        logger.info(f"  Device: {self.device}")
        logger.info(f"  MLflow: {'ON' if self.use_mlflow else 'OFF'}")
        logger.info(f"  TimescaleDB: {'ON' if self.use_timescaledb else 'OFF'}")
        logger.info("=" * 60)

        # Initialize integrations
        run_name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_mlflow(run_name)
        self._init_timescaledb()

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
                "device": self.device,
                "mlflow_run_id": self._mlflow_run_id,
            },
        )

        # Log final metrics to MLflow
        final_metrics = result.metadata.get("final_metrics", {})
        if final_metrics:
            self._log_mlflow_metrics(final_metrics)
            self._log_mlflow_metrics({"total_duration_seconds": total_duration})

        # End MLflow run
        self._end_mlflow(success)

        # Close TimescaleDB connection
        self._close_timescaledb()

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
        prediction_horizon = train_config.get("prediction_horizon", 5)

        # Handle purge_gap - can be "auto" string or int
        purge_gap_setting = train_config.get("purge_gap", 50)
        if purge_gap_setting == "auto" or not isinstance(purge_gap_setting, (int, float)):
            # Calculate auto purge gap based on prediction horizon
            # For financial data, purge gap should be roughly 2x prediction_horizon
            # to ensure no look-ahead bias, but not too large to waste data
            # Use prediction_horizon * 2 + small buffer, NOT max_feature_lookback
            # max_feature_lookback is for feature calculation, not purging
            purge_gap = prediction_horizon * 2 + 5  # Reasonable purge: 2x horizon + buffer
            logger.info(f"Auto-calculated purge_gap: {purge_gap} (prediction_horizon={prediction_horizon})")
        else:
            purge_gap = int(purge_gap_setting)

        # Create target
        target_col = train_config.get("target_column", "close")
        if target_col in self._data.columns:
            target = self._data[target_col].pct_change().shift(-prediction_horizon)
        else:
            target = self._features[target_col] if target_col in self._features.columns else None

        if target is None:
            raise ValueError(f"Target column '{target_col}' not found")

        # Clean features before alignment
        features_clean = self._features.copy()

        # Drop columns that are entirely NaN
        all_nan_cols = features_clean.columns[features_clean.isna().all()].tolist()
        if all_nan_cols:
            logger.warning(f"Dropping {len(all_nan_cols)} columns that are all NaN: {all_nan_cols}")
            features_clean = features_clean.drop(columns=all_nan_cols)

        # Forward fill then backward fill remaining NaN values
        features_clean = features_clean.ffill().bfill()

        # Log remaining NaN info
        remaining_nan = features_clean.isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"Still have {remaining_nan} NaN values after ffill/bfill")
            # Fill any remaining with 0
            features_clean = features_clean.fillna(0)

        # Align features and target
        aligned = pd.DataFrame({
            **features_clean,
            "target": target,
        })

        # Only drop rows where target is NaN (not features, since we cleaned those)
        aligned = aligned.dropna(subset=["target"])

        logger.info(f"After alignment: {len(aligned)} samples, {len(features_clean.columns)} features")

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
        Step 4: Train model with cross-validation and GPU support.
        """
        logger.info(f"Training model on device: {self.device}...")

        if self._X_train is None or self._y_train is None:
            raise ValueError("Training data not prepared")

        train_config = self.config.get("training", {})
        model_type = train_config.get("model_type", "lightgbm_regressor")

        # Get base model params
        model_params = train_config.get("model_params") or {}

        # Add GPU parameters if available
        gpu_params = self._get_gpu_params(model_type)
        if gpu_params:
            model_params = {**model_params, **gpu_params}
            logger.info(f"GPU params applied: {gpu_params}")

        # Create model
        if self.model_factory is None:
            from src.training.model_factory import ModelFactory
            self.model_factory = ModelFactory

        model = self.model_factory.create_model(
            model_type=model_type,
            params=model_params,
        )

        # Cross-validation scores
        cv_scores = []
        X_np = self._X_train.values
        y_np = self._y_train.values

        from tqdm import tqdm
        n_splits = self._cv.n_splits if hasattr(self._cv, 'n_splits') else 5

        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(
            self._cv.split(X_np, y_np),
            total=n_splits,
            desc="CV Folds"
        )):
            X_fold_train = X_np[train_idx]
            y_fold_train = y_np[train_idx]
            X_fold_val = X_np[val_idx]
            y_fold_val = y_np[val_idx]

            # Clone model for this fold with GPU params
            fold_model = self.model_factory.create_model(
                model_type=model_type,
                params=model_params,
            )

            # Train
            fold_model.fit(X_fold_train, y_fold_train)

            # Evaluate
            y_pred = fold_model.predict(X_fold_val)

            # Calculate IC (Information Coefficient)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_fold_val, y_pred)
            cv_scores.append(ic if not np.isnan(ic) else 0.0)

            # Log to MLflow
            self._log_mlflow_metrics({"fold_ic": cv_scores[-1]}, step=fold_idx)

            logger.info(f"Fold {fold_idx + 1}: IC = {cv_scores[-1]:.4f}")

        # Train final model on all training data
        logger.info("Training final model on all data...")
        model.fit(X_np, y_np)
        self._model = model

        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)

        logger.info(f"CV Complete: Mean IC = {mean_cv:.4f} (+/- {std_cv:.4f})")

        # Log CV results to MLflow
        self._log_mlflow_metrics({
            "cv_mean_ic": mean_cv,
            "cv_std_ic": std_cv,
            "n_folds": len(cv_scores),
        })

        return {
            "cv_mean_ic": mean_cv,
            "cv_std_ic": std_cv,
            "n_folds": len(cv_scores),
            "device": self.device,
        }

    def _get_gpu_params(self, model_type: str) -> Dict[str, Any]:
        """Get GPU-specific parameters for model type."""
        if not self.use_gpu or self.device == "cpu":
            return {}

        model_type_lower = model_type.lower()

        if "lightgbm" in model_type_lower:
            # LightGBM GPU parameters - only use device=gpu, no platform/device ids
            # which can cause issues with empty string parsing
            if self.lightgbm_gpu or self.device in ("cuda", "lightgbm_gpu"):
                return {"device": "gpu"}
            return {}
        elif "xgboost" in model_type_lower:
            # XGBoost GPU parameters - only works with CUDA
            if self.device == "cuda":
                return {
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "gpu_id": 0,
                }
            return {}
        elif "catboost" in model_type_lower:
            # CatBoost GPU parameters
            if self.device == "cuda" or self.lightgbm_gpu:
                return {
                    "task_type": "GPU",
                    "devices": "0",
                }
            return {}

        return {}

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

        # Register to TimescaleDB
        cv_metrics = self._stage_results.get(PipelineStage.MODEL_TRAINING)
        all_metrics = {**metrics}
        if cv_metrics and cv_metrics.metrics:
            all_metrics.update(cv_metrics.metrics)
        self._register_model_to_db(model_id, all_metrics)

        # Save predictions to TimescaleDB
        if self._model is not None and self._X_test is not None:
            y_pred = self._model.predict(self._X_test.values)
            self._save_predictions_to_db(y_pred, model_id)

        # Save metrics to TimescaleDB
        self._save_metrics_to_db(metrics, "model", model_id)

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
