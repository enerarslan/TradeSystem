"""
MLflow Model Registry Integration
=================================

Automated model lifecycle management using MLflow.
Replaces manual file naming with a proper model registry.

Features:
- Experiment tracking
- Model versioning
- Automatic model promotion
- A/B testing support
- Model lineage tracking

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import tempfile

import numpy as np

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config.settings import get_logger

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """
    Model version information.

    Attributes:
        name: Model name
        version: Version number
        stage: Model stage (None, Staging, Production, Archived)
        run_id: MLflow run ID
        metrics: Training/validation metrics
        params: Model parameters
        tags: Additional tags
        created_at: Creation timestamp
    """
    name: str
    version: int
    stage: str = "None"
    run_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class MLflowRegistry:
    """
    MLflow Model Registry for AlphaTrade.

    Manages the complete model lifecycle:
    1. Training: Log experiments, metrics, artifacts
    2. Registration: Register models with versions
    3. Staging: Promote models through stages
    4. Production: Serve production models
    5. Archival: Archive old models

    Example:
        registry = MLflowRegistry(tracking_uri="http://localhost:5000")

        # Log training run
        with registry.start_run(experiment="AAPL_classifier") as run:
            model = train_model(X, y)
            registry.log_params({"learning_rate": 0.01})
            registry.log_metrics({"accuracy": 0.85, "f1": 0.82})
            registry.log_model(model, "model")

        # Register model
        registry.register_model(run.info.run_id, "AAPL_classifier")

        # Promote to production
        registry.promote_model("AAPL_classifier", version=1, stage="Production")

        # Load production model
        model = registry.load_production_model("AAPL_classifier")
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """
        Initialize MLflow registry.

        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: Model registry URI (defaults to tracking_uri)
            artifact_location: Default artifact storage location
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow package required: pip install mlflow")

        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
        self.registry_uri = registry_uri or self.tracking_uri
        self.artifact_location = artifact_location

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create client
        self._client = MlflowClient(tracking_uri=self.tracking_uri)

        self._current_run = None
        self._experiment_id = None

        logger.info(f"MLflow registry initialized: {self.tracking_uri}")

    def create_experiment(
        self,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Create or get existing experiment.

        Args:
            name: Experiment name
            tags: Experiment tags

        Returns:
            Experiment ID
        """
        try:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment:
                return experiment.experiment_id

            experiment_id = mlflow.create_experiment(
                name=name,
                artifact_location=self.artifact_location,
                tags=tags or {},
            )

            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id

        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise

    def start_run(
        self,
        experiment: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """
        Start an MLflow run for logging.

        Args:
            experiment: Experiment name
            run_name: Optional run name
            tags: Run tags

        Returns:
            Context manager for the run
        """
        experiment_id = self.create_experiment(experiment)
        self._experiment_id = experiment_id

        return mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        mlflow.log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact file."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Any = None,
        input_example: Any = None,
        registered_model_name: str | None = None,
    ) -> None:
        """
        Log a model to MLflow.

        Supports multiple model flavors:
        - sklearn
        - lightgbm
        - xgboost
        - tensorflow/keras
        - pytorch

        Args:
            model: The model object
            artifact_path: Path within the run's artifact directory
            signature: Model signature
            input_example: Example input for signature inference
            registered_model_name: If provided, also register the model
        """
        # Determine model flavor
        model_type = type(model).__module__.split(".")[0]

        try:
            if model_type == "lightgbm" or "lightgbm" in str(type(model)):
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            elif model_type == "xgboost" or "xgboost" in str(type(model)):
                mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            elif model_type == "sklearn" or "sklearn" in str(type(model)):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            elif model_type == "tensorflow" or model_type == "keras":
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            elif model_type == "torch":
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            else:
                # Fallback to pickle
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )

            logger.info(f"Logged model to {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise

    def register_model(
        self,
        run_id: str,
        name: str,
        artifact_path: str = "model",
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """
        Register a model from a run.

        Args:
            run_id: MLflow run ID
            name: Model name for registry
            artifact_path: Path to model artifact
            tags: Model tags

        Returns:
            ModelVersion info
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"

            result = mlflow.register_model(model_uri, name)

            # Add tags
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(
                        name=name,
                        version=result.version,
                        key=key,
                        value=value,
                    )

            logger.info(f"Registered model: {name} v{result.version}")

            return ModelVersion(
                name=name,
                version=int(result.version),
                run_id=run_id,
                tags=tags or {},
            )

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def promote_model(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """
        Promote a model version to a stage.

        Args:
            name: Model name
            version: Version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in that stage
        """
        try:
            self._client.transition_model_version_stage(
                name=name,
                version=str(version),
                stage=stage,
                archive_existing_versions=archive_existing,
            )

            logger.info(f"Promoted {name} v{version} to {stage}")

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    def get_latest_version(
        self,
        name: str,
        stage: str | None = None,
    ) -> ModelVersion | None:
        """
        Get the latest version of a model.

        Args:
            name: Model name
            stage: Filter by stage (None = any)

        Returns:
            ModelVersion or None
        """
        try:
            if stage:
                versions = self._client.get_latest_versions(name, stages=[stage])
            else:
                versions = self._client.get_latest_versions(name)

            if not versions:
                return None

            mv = versions[0]

            return ModelVersion(
                name=mv.name,
                version=int(mv.version),
                stage=mv.current_stage,
                run_id=mv.run_id,
                created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000),
            )

        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None

    def load_model(
        self,
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            name: Model name
            version: Specific version (optional)
            stage: Stage to load from (optional)

        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{name}/{version}"
            elif stage:
                model_uri = f"models:/{name}/{stage}"
            else:
                model_uri = f"models:/{name}/latest"

            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(f"Loaded model: {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_production_model(self, name: str) -> Any:
        """
        Load the production model.

        Args:
            name: Model name

        Returns:
            Production model
        """
        return self.load_model(name, stage="Production")

    def load_staging_model(self, name: str) -> Any:
        """
        Load the staging model.

        Args:
            name: Model name

        Returns:
            Staging model
        """
        return self.load_model(name, stage="Staging")

    def list_models(self) -> list[str]:
        """List all registered model names."""
        try:
            models = self._client.search_registered_models()
            return [m.name for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_versions(self, name: str) -> list[ModelVersion]:
        """
        Get all versions of a model.

        Args:
            name: Model name

        Returns:
            List of ModelVersion
        """
        try:
            versions = self._client.search_model_versions(f"name='{name}'")

            return [
                ModelVersion(
                    name=v.name,
                    version=int(v.version),
                    stage=v.current_stage,
                    run_id=v.run_id,
                    created_at=datetime.fromtimestamp(v.creation_timestamp / 1000),
                )
                for v in versions
            ]

        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def delete_model_version(self, name: str, version: int) -> bool:
        """
        Delete a model version.

        Args:
            name: Model name
            version: Version to delete

        Returns:
            True if deleted
        """
        try:
            self._client.delete_model_version(name, str(version))
            logger.info(f"Deleted {name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False

    def set_model_tag(
        self,
        name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        """Set a tag on a model version."""
        self._client.set_model_version_tag(name, str(version), key, value)

    def get_run_metrics(self, run_id: str) -> dict[str, float]:
        """Get metrics from a run."""
        try:
            run = self._client.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Failed to get run metrics: {e}")
            return {}

    def compare_models(
        self,
        name: str,
        version_a: int,
        version_b: int,
    ) -> dict[str, Any]:
        """
        Compare two model versions.

        Args:
            name: Model name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison dictionary
        """
        versions = {v.version: v for v in self.get_model_versions(name)}

        va = versions.get(version_a)
        vb = versions.get(version_b)

        if not va or not vb:
            return {"error": "Version not found"}

        metrics_a = self.get_run_metrics(va.run_id)
        metrics_b = self.get_run_metrics(vb.run_id)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics": {},
        }

        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        for metric in all_metrics:
            val_a = metrics_a.get(metric, 0)
            val_b = metrics_b.get(metric, 0)
            diff = val_b - val_a
            diff_pct = (diff / val_a * 100) if val_a != 0 else 0

            comparison["metrics"][metric] = {
                "version_a": val_a,
                "version_b": val_b,
                "difference": diff,
                "difference_pct": diff_pct,
            }

        return comparison


# =============================================================================
# MODEL MANAGER WITH MLFLOW
# =============================================================================

class MLModelManager:
    """
    High-level model management for trading system.

    Integrates MLflow with trading-specific workflows:
    - Symbol-specific model naming
    - Automatic model selection
    - Fallback handling
    - Performance tracking

    Example:
        manager = MLModelManager()

        # Train and register
        model = manager.train_and_register(
            symbol="AAPL",
            model_type="lightgbm",
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
        )

        # Load for inference
        model = manager.get_model("AAPL", model_type="lightgbm")
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_prefix: str = "alphatrade",
    ):
        """
        Initialize model manager.

        Args:
            tracking_uri: MLflow tracking URI
            model_prefix: Prefix for model names
        """
        self.registry = MLflowRegistry(tracking_uri=tracking_uri)
        self.model_prefix = model_prefix
        self._model_cache: dict[str, Any] = {}

    def _make_model_name(self, symbol: str, model_type: str) -> str:
        """Create standardized model name."""
        return f"{self.model_prefix}_{symbol}_{model_type}"

    def train_and_register(
        self,
        symbol: str,
        model_type: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        auto_promote: bool = False,
    ) -> ModelVersion:
        """
        Train (if needed), log, and register a model.

        Args:
            symbol: Trading symbol
            model_type: Model type (lightgbm, xgboost, etc.)
            model: Model object (trained or to be trained)
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Model parameters to log
            metrics: Pre-computed metrics to log
            auto_promote: Automatically promote to Production if better

        Returns:
            ModelVersion info
        """
        model_name = self._make_model_name(symbol, model_type)
        experiment_name = f"{symbol}_training"

        with self.registry.start_run(experiment_name, run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            if params:
                self.registry.log_params(params)

            self.registry.log_params({
                "symbol": symbol,
                "model_type": model_type,
                "n_train_samples": len(X_train),
                "n_features": X_train.shape[1],
            })

            # Log metrics
            if metrics:
                self.registry.log_metrics(metrics)
            elif X_val is not None and y_val is not None:
                # Calculate validation metrics
                try:
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import accuracy_score, f1_score

                    val_metrics = {
                        "val_accuracy": accuracy_score(y_val, y_pred),
                        "val_f1": f1_score(y_val, y_pred, average="weighted"),
                    }
                    self.registry.log_metrics(val_metrics)
                except Exception as e:
                    logger.warning(f"Could not compute validation metrics: {e}")

            # Log model with signature
            try:
                signature = infer_signature(X_train, model.predict(X_train[:10]))
            except Exception:
                signature = None

            self.registry.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_train[:5],
            )

            run_id = run.info.run_id

        # Register model
        version = self.registry.register_model(
            run_id=run_id,
            name=model_name,
            tags={
                "symbol": symbol,
                "model_type": model_type,
                "trained_at": datetime.now().isoformat(),
            },
        )

        # Auto promote if requested
        if auto_promote:
            self._auto_promote_if_better(model_name, version.version)

        return version

    def _auto_promote_if_better(self, model_name: str, new_version: int) -> None:
        """Promote new version if it outperforms production."""
        try:
            prod_version = self.registry.get_latest_version(model_name, stage="Production")

            if not prod_version:
                # No production model, promote new one
                self.registry.promote_model(model_name, new_version, "Production")
                return

            # Compare metrics
            comparison = self.registry.compare_models(
                model_name,
                prod_version.version,
                new_version,
            )

            # Simple heuristic: if val_accuracy or val_f1 improved, promote
            metrics = comparison.get("metrics", {})
            accuracy_diff = metrics.get("val_accuracy", {}).get("difference_pct", 0)
            f1_diff = metrics.get("val_f1", {}).get("difference_pct", 0)

            if accuracy_diff > 0 or f1_diff > 0:
                logger.info(f"New model is better, promoting v{new_version}")
                self.registry.promote_model(model_name, new_version, "Production")
            else:
                logger.info(f"Production model is still better, keeping v{prod_version.version}")
                # Stage the new version
                self.registry.promote_model(model_name, new_version, "Staging")

        except Exception as e:
            logger.error(f"Auto-promote failed: {e}")

    def get_model(
        self,
        symbol: str,
        model_type: str,
        stage: str = "Production",
        use_cache: bool = True,
    ) -> Any | None:
        """
        Get a model for inference.

        Args:
            symbol: Trading symbol
            model_type: Model type
            stage: Model stage to load
            use_cache: Use cached model if available

        Returns:
            Model or None if not found
        """
        model_name = self._make_model_name(symbol, model_type)
        cache_key = f"{model_name}:{stage}"

        # Check cache
        if use_cache and cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            model = self.registry.load_model(model_name, stage=stage)

            # Cache the model
            if use_cache:
                self._model_cache[cache_key] = model

            return model

        except Exception as e:
            logger.warning(f"Could not load {model_name} ({stage}): {e}")

            # Try fallback to latest version
            try:
                model = self.registry.load_model(model_name)
                if use_cache:
                    self._model_cache[cache_key] = model
                return model
            except Exception:
                return None

    def clear_cache(self) -> None:
        """Clear model cache."""
        self._model_cache.clear()

    def list_symbol_models(self, symbol: str) -> list[str]:
        """List all models for a symbol."""
        all_models = self.registry.list_models()
        prefix = f"{self.model_prefix}_{symbol}_"
        return [m for m in all_models if m.startswith(prefix)]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ModelVersion",
    "MLflowRegistry",
    "MLModelManager",
]
