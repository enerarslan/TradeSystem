"""
MLflow-based experiment tracking for institutional-grade ML operations.

This module provides comprehensive experiment tracking capabilities including:
- Automatic parameter logging
- Metric tracking with versioning
- Model artifact storage
- Experiment comparison
- Model lineage tracking

Designed for JPMorgan-level requirements:
- Full audit trail of all experiments
- Reproducibility guarantees
- Model governance compliance
- Automated documentation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    tracking_uri: str = "mlruns"  # Local storage by default
    artifact_location: Optional[str] = None
    experiment_name: str = "alphatrade"
    tags: Dict[str, str] = field(default_factory=dict)
    auto_log_sklearn: bool = True
    auto_log_xgboost: bool = True
    auto_log_lightgbm: bool = True


@dataclass
class RunMetrics:
    """Container for run metrics."""
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        metrics = {}
        for key, value in self.__dict__.items():
            if key == 'custom_metrics':
                metrics.update(value)
            elif value is not None:
                metrics[key] = value
        return metrics


class ExperimentTracker:
    """
    MLflow-based experiment tracker for financial ML models.

    Provides comprehensive tracking of:
    - Model hyperparameters
    - Training and validation metrics
    - Feature importance
    - Model artifacts
    - Custom visualizations

    Example:
        tracker = ExperimentTracker(experiment_name="momentum_strategy")

        with tracker.start_run(run_name="lgb_v1") as run:
            tracker.log_params({"n_estimators": 100, "max_depth": 6})
            tracker.log_metrics({"sharpe_ratio": 1.5, "max_drawdown": 0.15})
            tracker.log_model(model, "lightgbm_model")
            tracker.log_feature_importance(model, feature_names)

        # Compare runs
        comparison = tracker.compare_runs(metric="sharpe_ratio")
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is required for ExperimentTracker. "
                "Install with: pip install mlflow"
            )

        self.config = config or ExperimentConfig()

        # Override with explicit args
        if experiment_name:
            self.config.experiment_name = experiment_name
        if tracking_uri:
            self.config.tracking_uri = tracking_uri

        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Create or get experiment
        self._experiment = mlflow.set_experiment(self.config.experiment_name)
        self._client = MlflowClient()
        self._active_run = None

        # Enable autologging if configured
        self._setup_autolog()

        logger.info(
            f"ExperimentTracker initialized: experiment='{self.config.experiment_name}', "
            f"tracking_uri='{self.config.tracking_uri}'"
        )

    def _setup_autolog(self) -> None:
        """Set up autologging for supported frameworks."""
        if self.config.auto_log_sklearn:
            try:
                mlflow.sklearn.autolog(log_models=False)
            except Exception:
                pass

        if self.config.auto_log_xgboost:
            try:
                mlflow.xgboost.autolog(log_models=False)
            except Exception:
                pass

        if self.config.auto_log_lightgbm:
            try:
                mlflow.lightgbm.autolog(log_models=False)
            except Exception:
                pass

    @property
    def experiment_id(self) -> str:
        """Get current experiment ID."""
        return self._experiment.experiment_id

    @property
    def active_run_id(self) -> Optional[str]:
        """Get active run ID if any."""
        if self._active_run:
            return self._active_run.info.run_id
        return None

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Additional tags for the run
            nested: If True, allow nested runs

        Yields:
            MLflow Run object
        """
        # Merge tags
        all_tags = {**self.config.tags}
        if tags:
            all_tags.update(tags)

        # Add standard tags
        all_tags["timestamp"] = datetime.now().isoformat()
        all_tags["experiment"] = self.config.experiment_name

        try:
            self._active_run = mlflow.start_run(
                run_name=run_name,
                tags=all_tags,
                nested=nested,
            )
            logger.info(f"Started run: {run_name or self._active_run.info.run_id}")
            yield self._active_run

        finally:
            mlflow.end_run()
            self._active_run = None
            logger.info("Run completed")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the active run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self._active_run:
            logger.warning("No active run. Call start_run() first.")
            return

        # Handle nested dicts and convert non-string values
        flat_params = self._flatten_dict(params)

        # MLflow has a 500 char limit for param values
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            mlflow.log_param(key, str_value)

    def log_metrics(
        self,
        metrics: Union[Dict[str, float], RunMetrics],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to the active run.

        Args:
            metrics: Dictionary of metric names and values, or RunMetrics object
            step: Optional step number for time-series metrics
        """
        if not self._active_run:
            logger.warning("No active run. Call start_run() first.")
            return

        if isinstance(metrics, RunMetrics):
            metrics = metrics.to_dict()

        for key, value in metrics.items():
            if value is not None and not np.isnan(value):
                mlflow.log_metric(key, float(value), step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict] = None,
        signature: Optional[Any] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None,
    ) -> str:
        """
        Log a model artifact.

        Args:
            model: The model object to log
            artifact_path: Path within run's artifact directory
            conda_env: Conda environment specification
            signature: Model signature (input/output schema)
            input_example: Example input for documentation
            registered_model_name: If provided, register model with this name

        Returns:
            URI of the logged model
        """
        if not self._active_run:
            logger.warning("No active run. Call start_run() first.")
            return ""

        # Infer signature if not provided
        if signature is None and input_example is not None:
            try:
                predictions = model.predict(input_example)
                signature = infer_signature(input_example, predictions)
            except Exception as e:
                logger.warning(f"Could not infer model signature: {e}")

        # Detect model type and use appropriate logging
        model_info = self._log_model_by_type(
            model=model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        return model_info.model_uri if model_info else ""

    def _log_model_by_type(
        self,
        model: Any,
        artifact_path: str,
        **kwargs,
    ):
        """Log model using appropriate MLflow flavor."""
        model_type = type(model).__name__

        try:
            # LightGBM
            if "LGBMClassifier" in model_type or "LGBMRegressor" in model_type:
                return mlflow.lightgbm.log_model(model, artifact_path, **kwargs)

            # XGBoost
            elif "XGB" in model_type:
                return mlflow.xgboost.log_model(model, artifact_path, **kwargs)

            # CatBoost
            elif "CatBoost" in model_type:
                return mlflow.catboost.log_model(model, artifact_path, **kwargs)

            # Sklearn
            elif hasattr(model, 'fit') and hasattr(model, 'predict'):
                return mlflow.sklearn.log_model(model, artifact_path, **kwargs)

            # PyTorch
            elif "torch" in str(type(model).__module__):
                return mlflow.pytorch.log_model(model, artifact_path, **kwargs)

            # Generic fallback
            else:
                return mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model,
                    **kwargs
                )

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return None

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to local file
            artifact_path: Directory within run's artifacts
        """
        if not self._active_run:
            logger.warning("No active run. Call start_run() first.")
            return

        mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(
        self,
        local_dir: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Directory within run's artifacts
        """
        if not self._active_run:
            logger.warning("No active run. Call start_run() first.")
            return

        mlflow.log_artifacts(str(local_dir), artifact_path)

    def log_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str = "gain",
        top_n: int = 50,
    ) -> None:
        """
        Log feature importance as artifact and metrics.

        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            importance_type: Type of importance (gain, weight, cover)
            top_n: Number of top features to log as metrics
        """
        if not self._active_run:
            return

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'get_score'):
            # XGBoost
            score = model.get_score(importance_type=importance_type)
            importance = [score.get(f, 0) for f in feature_names]
        else:
            logger.warning("Model does not have feature importance")
            return

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Log top features as metrics
        for i, row in importance_df.head(top_n).iterrows():
            mlflow.log_metric(
                f"feature_importance_{row['feature'][:50]}",
                row['importance']
            )

        # Save as CSV artifact
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            importance_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "feature_importance")
            os.unlink(f.name)

        # Create visualization if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            self._plot_feature_importance(importance_df.head(30))

    def _plot_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """Create and log feature importance plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            importance_df['feature'].values[::-1],
            importance_df['importance'].values[::-1]
        )
        ax.set_xlabel('Importance')
        ax.set_title('Top Feature Importance')
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(
            suffix='.png', delete=False
        ) as f:
            fig.savefig(f.name, dpi=150)
            mlflow.log_artifact(f.name, "plots")
            os.unlink(f.name)

        plt.close(fig)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Log confusion matrix as artifact."""
        if not MATPLOTLIB_AVAILABLE:
            return

        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        with tempfile.NamedTemporaryFile(
            suffix='.png', delete=False
        ) as f:
            fig.savefig(f.name, dpi=150)
            mlflow.log_artifact(f.name, "plots")
            os.unlink(f.name)

        plt.close(fig)

    def log_equity_curve(
        self,
        equity: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> None:
        """Log equity curve plot as artifact."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(equity.index, equity.values, label='Strategy', linewidth=2)

        if benchmark is not None:
            ax.plot(
                benchmark.index, benchmark.values,
                label='Benchmark', linewidth=1, alpha=0.7
            )

        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        with tempfile.NamedTemporaryFile(
            suffix='.png', delete=False
        ) as f:
            fig.savefig(f.name, dpi=150)
            mlflow.log_artifact(f.name, "plots")
            os.unlink(f.name)

        plt.close(fig)

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_file: str,
    ) -> None:
        """Log a dictionary as JSON artifact."""
        if not self._active_run:
            return

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(dictionary, f, indent=2, default=str)
            mlflow.log_artifact(f.name, artifact_file)
            os.unlink(f.name)

    def log_dataframe(
        self,
        df: pd.DataFrame,
        artifact_path: str,
        file_format: str = "csv",
    ) -> None:
        """Log a DataFrame as artifact."""
        if not self._active_run:
            return

        suffix = f".{file_format}"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix=suffix, delete=False
        ) as f:
            if file_format == "csv":
                df.to_csv(f.name)
            elif file_format == "parquet":
                df.to_parquet(f.name)
            else:
                df.to_csv(f.name)

            mlflow.log_artifact(f.name, artifact_path)
            os.unlink(f.name)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active run."""
        if self._active_run:
            mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags on the active run."""
        if self._active_run:
            mlflow.set_tags(tags)

    def log_data_lineage(
        self,
        data_hash: str,
        data_path: Optional[str] = None,
        data_version: Optional[str] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log data lineage information for reproducibility.

        This is CRITICAL for JPMorgan compliance - every model must
        be traceable to the exact data used to train it.

        Args:
            data_hash: Hash/checksum of the training data
            data_path: Path to data source (if applicable)
            data_version: Version of the data (e.g., date, commit)
            n_samples: Number of training samples
            n_features: Number of features
            feature_names: List of feature names
            data_stats: Additional data statistics
        """
        if not self._active_run:
            logger.warning("No active run. Call start_run() first.")
            return

        # Log as tags (always visible in MLflow UI)
        mlflow.set_tag("data.hash", data_hash)

        if data_path:
            mlflow.set_tag("data.path", str(data_path))
        if data_version:
            mlflow.set_tag("data.version", data_version)

        # Log as params
        if n_samples:
            mlflow.log_param("data.n_samples", n_samples)
        if n_features:
            mlflow.log_param("data.n_features", n_features)

        # Log git commit if available
        git_commit = self._get_git_commit()
        if git_commit:
            mlflow.set_tag("git.commit", git_commit)
            mlflow.set_tag("git.branch", self._get_git_branch() or "unknown")

        # Save detailed lineage as artifact
        lineage_info = {
            "data_hash": data_hash,
            "data_path": data_path,
            "data_version": data_version,
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_names": feature_names[:100] if feature_names else None,  # Limit size
            "git_commit": git_commit,
            "git_branch": self._get_git_branch(),
            "timestamp": datetime.now().isoformat(),
            "data_stats": data_stats,
        }

        self.log_dict(lineage_info, "data_lineage")
        logger.info(f"Data lineage logged: hash={data_hash}, git={git_commit}")

    def log_training_result(
        self,
        result: Any,  # TrainingResult
    ) -> None:
        """
        Log a complete TrainingResult object.

        Convenience method that logs all result components.

        Args:
            result: TrainingResult from trainer
        """
        if not self._active_run:
            return

        # Log params
        self.log_params(result.params)

        # Log metrics
        all_metrics = {**result.train_metrics, **result.validation_metrics}
        self.log_metrics(all_metrics)

        # Log data lineage
        if hasattr(result, 'data_hash') and result.data_hash:
            self.log_data_lineage(
                data_hash=result.data_hash,
                n_samples=result.n_train_samples,
                n_features=result.n_features,
            )

        # Log CV scores if available
        if result.cv_scores:
            for metric, scores in result.cv_scores.items():
                for i, score in enumerate(scores):
                    mlflow.log_metric(f"cv_{metric}", score, step=i)

        # Log feature importance stability if available
        if hasattr(result, 'cv_feature_importances') and result.cv_feature_importances:
            stable_importance = result.get_stable_feature_importance()
            if stable_importance is not None:
                self.log_dataframe(
                    stable_importance,
                    "feature_importance_stability",
                    file_format="csv"
                )

        # Log model
        if result.model is not None:
            self.log_model(result.model, "model")

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]  # Short hash
        except Exception:
            pass
        return None

    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_git_status(self) -> Dict[str, Any]:
        """Get comprehensive git status."""
        status = {
            "commit": self._get_git_commit(),
            "branch": self._get_git_branch(),
            "is_dirty": False,
        }

        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                status["is_dirty"] = len(result.stdout.strip()) > 0
        except Exception:
            pass

        return status

    def get_run(self, run_id: str) -> Any:
        """Get a run by ID."""
        return self._client.get_run(run_id)

    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Search runs in the experiment.

        Args:
            filter_string: MLflow filter string (e.g., "metrics.sharpe_ratio > 1.0")
            max_results: Maximum number of results
            order_by: List of columns to order by (e.g., ["metrics.sharpe_ratio DESC"])

        Returns:
            DataFrame with run information
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
        )
        return runs

    def compare_runs(
        self,
        metric: str,
        top_n: int = 10,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Compare runs by a specific metric.

        Args:
            metric: Metric name to compare
            top_n: Number of top runs to return
            ascending: If True, sort ascending (lower is better)

        Returns:
            DataFrame with top runs
        """
        order = "ASC" if ascending else "DESC"
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=top_n,
        )
        return runs

    def get_best_run(
        self,
        metric: str,
        ascending: bool = False,
    ) -> Optional[Any]:
        """
        Get the best run by a specific metric.

        Args:
            metric: Metric name to optimize
            ascending: If True, lower is better

        Returns:
            Best run object or None
        """
        runs = self.compare_runs(metric, top_n=1, ascending=ascending)

        if runs.empty:
            return None

        best_run_id = runs.iloc[0]['run_id']
        return self.get_run(best_run_id)

    def delete_run(self, run_id: str) -> None:
        """Delete a run."""
        self._client.delete_run(run_id)

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '.',
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# Convenience function for quick experiment tracking
def track_experiment(
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Optional[Any] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convenience function for quick experiment tracking.

    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run
        params: Parameters to log
        metrics: Metrics to log
        model: Optional model to log
        artifacts: Optional dict of artifact_path: local_path
        tags: Optional tags

    Returns:
        Run ID
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)

    with tracker.start_run(run_name=run_name, tags=tags) as run:
        tracker.log_params(params)
        tracker.log_metrics(metrics)

        if model is not None:
            tracker.log_model(model, "model")

        if artifacts:
            for artifact_path, local_path in artifacts.items():
                tracker.log_artifact(local_path, artifact_path)

        return run.info.run_id
