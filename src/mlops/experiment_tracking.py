"""
MLflow Experiment Tracking Integration
Enterprise-Grade ML Experiment Management

Features:
- Automatic experiment and run tracking
- Model versioning and registry
- Hyperparameter logging
- Metric tracking with plots
- Artifact storage (models, data, plots)
- Model comparison and analysis

Usage:
    tracker = MLflowTracker("trading_experiments")
    with tracker.start_run("model_training"):
        tracker.log_params(hyperparams)
        tracker.log_metrics(metrics)
        tracker.log_model(model, "best_model")
"""

import os
import json
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import warnings

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    experiment_name: str = "trading_system"
    tracking_uri: str = "mlruns"  # Local directory or MLflow server URI
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    # Auto-logging settings
    auto_log_params: bool = True
    auto_log_metrics: bool = True
    auto_log_models: bool = True

    # Storage settings
    log_system_metrics: bool = True
    log_git_info: bool = True


@dataclass
class RunMetadata:
    """Metadata for an experiment run"""
    run_id: str
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "RUNNING"
    tags: Dict[str, str] = field(default_factory=dict)


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.

    Provides simplified API for:
    - Experiment management
    - Run tracking
    - Metric/param logging
    - Model versioning
    """

    def __init__(
        self,
        experiment_name: str = "trading_system",
        tracking_uri: Optional[str] = None,
        config: Optional[ExperimentConfig] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI
            config: Full experiment configuration
        """
        self.config = config or ExperimentConfig(experiment_name=experiment_name)

        if tracking_uri:
            self.config.tracking_uri = tracking_uri

        self._client: Optional['MlflowClient'] = None
        self._experiment_id: Optional[str] = None
        self._active_run: Optional['mlflow.ActiveRun'] = None
        self._run_metadata: Optional[RunMetadata] = None

        self._initialize()

    def _initialize(self):
        """Initialize MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Using fallback logging.")
            return

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)

            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags
                )
            else:
                self._experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.config.experiment_name)

            self._client = MlflowClient()

            logger.info(
                f"MLflow initialized: experiment='{self.config.experiment_name}', "
                f"tracking_uri='{self.config.tracking_uri}'"
            )

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
            self._client = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> 'MLflowTracker':
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Additional tags
            nested: Whether this is a nested run

        Returns:
            Self for context manager usage
        """
        if not MLFLOW_AVAILABLE or self._client is None:
            return self

        try:
            run_tags = {**(self.config.tags), **(tags or {})}

            if self.config.log_git_info:
                run_tags.update(self._get_git_info())

            self._active_run = mlflow.start_run(
                run_name=run_name,
                experiment_id=self._experiment_id,
                tags=run_tags,
                nested=nested
            )

            self._run_metadata = RunMetadata(
                run_id=self._active_run.info.run_id,
                experiment_id=self._experiment_id,
                start_time=datetime.now(),
                tags=run_tags
            )

            # Log system metrics
            if self.config.log_system_metrics:
                self._log_system_metrics()

            logger.info(f"MLflow run started: {self._run_metadata.run_id}")

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")

        return self

    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            if self._run_metadata:
                self._run_metadata.end_time = datetime.now()
                self._run_metadata.status = status

            mlflow.end_run(status=status)
            logger.info(f"MLflow run ended: {self._run_metadata.run_id}")

        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

        self._active_run = None

    def __enter__(self) -> 'MLflowTracker':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameters
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            logger.debug(f"Params (not logged): {params}")
            return

        try:
            # Flatten nested dicts and convert to strings
            flat_params = self._flatten_dict(params)

            # MLflow has a 500 param limit per batch
            batch_size = 100
            param_items = list(flat_params.items())

            for i in range(0, len(param_items), batch_size):
                batch = dict(param_items[i:i + batch_size])
                mlflow.log_params(batch)

            logger.debug(f"Logged {len(flat_params)} parameters")

        except Exception as e:
            logger.error(f"Failed to log params: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number for time series metrics
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            logger.debug(f"Metrics (not logged): {metrics}")
            return

        try:
            # Filter out non-numeric values
            numeric_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not np.isnan(v)
            }

            mlflow.log_metrics(numeric_metrics, step=step)
            logger.debug(f"Logged {len(numeric_metrics)} metrics")

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None
    ):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        self.log_metrics({key: value}, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log a model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path in artifact store
            registered_name: Optional name to register in model registry
            signature: Model signature
            input_example: Example input for signature inference
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            logger.debug(f"Model (not logged): {artifact_path}")
            return

        try:
            # Determine model type and log appropriately
            model_type = type(model).__name__

            if hasattr(model, 'get_booster'):
                # LightGBM
                import mlflow.lightgbm
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_name
                )

            elif hasattr(model, 'get_params') and 'xgboost' in str(type(model)):
                # XGBoost
                import mlflow.xgboost
                mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_name
                )

            elif hasattr(model, '_estimator_type'):
                # Sklearn-compatible model
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_name,
                    signature=signature,
                    input_example=input_example
                )

            else:
                # Generic pickle save
                self.log_artifact_pickle(model, f"{artifact_path}.pkl")

            logger.info(f"Model logged: {artifact_path} (type: {model_type})")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to local file
            artifact_path: Optional path in artifact store
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Artifact logged: {local_path}")

        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_artifact_pickle(self, obj: Any, filename: str):
        """
        Log a Python object as a pickled artifact.

        Args:
            obj: Object to pickle
            filename: Artifact filename
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                pickle.dump(obj, f)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path=os.path.dirname(filename))
            os.unlink(temp_path)

            logger.debug(f"Pickled artifact logged: {filename}")

        except Exception as e:
            logger.error(f"Failed to log pickled artifact: {e}")

    def log_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = 'parquet'
    ):
        """
        Log a DataFrame as an artifact.

        Args:
            df: DataFrame to log
            filename: Artifact filename
            format: File format ('parquet', 'csv', 'json')
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, filename)

                if format == 'parquet':
                    df.to_parquet(filepath)
                elif format == 'csv':
                    df.to_csv(filepath)
                elif format == 'json':
                    df.to_json(filepath)
                else:
                    raise ValueError(f"Unknown format: {format}")

                mlflow.log_artifact(filepath)

            logger.debug(f"DataFrame logged: {filename}")

        except Exception as e:
            logger.error(f"Failed to log DataFrame: {e}")

    def log_figure(self, figure: Any, filename: str):
        """
        Log a matplotlib figure as an artifact.

        Args:
            figure: Matplotlib figure
            filename: Artifact filename
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, filename)
                figure.savefig(filepath, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(filepath)

            logger.debug(f"Figure logged: {filename}")

        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def log_backtest_result(self, result: Any, prefix: str = "backtest"):
        """
        Log a backtest result with all metrics and artifacts.

        Args:
            result: BacktestResult object
            prefix: Metric prefix
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            # Log performance metrics
            metrics = {
                f"{prefix}_total_return": result.total_return,
                f"{prefix}_annualized_return": result.annualized_return,
                f"{prefix}_volatility": result.volatility,
                f"{prefix}_sharpe_ratio": result.sharpe_ratio,
                f"{prefix}_sortino_ratio": result.sortino_ratio,
                f"{prefix}_calmar_ratio": result.calmar_ratio,
                f"{prefix}_max_drawdown": result.max_drawdown,
                f"{prefix}_total_trades": float(result.total_trades),
                f"{prefix}_win_rate": result.win_rate,
                f"{prefix}_profit_factor": result.profit_factor,
                f"{prefix}_avg_win": result.avg_win,
                f"{prefix}_avg_loss": result.avg_loss,
                f"{prefix}_total_costs": result.total_costs,
            }
            self.log_metrics(metrics)

            # Log equity curve
            if result.equity_curve is not None:
                self.log_dataframe(
                    result.equity_curve.to_frame('equity'),
                    f"{prefix}_equity_curve.parquet"
                )

            # Log trades
            if result.trades:
                trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
                self.log_dataframe(trades_df, f"{prefix}_trades.parquet")

            logger.info(f"Backtest result logged: {prefix}")

        except Exception as e:
            logger.error(f"Failed to log backtest result: {e}")

    def log_cv_results(
        self,
        cv_results: Dict[str, Any],
        prefix: str = "cv"
    ):
        """
        Log cross-validation results.

        Args:
            cv_results: CV results dictionary
            prefix: Metric prefix
        """
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            # Log summary metrics
            summary_metrics = {
                f"{prefix}_mean_score": cv_results.get('mean_score', 0),
                f"{prefix}_std_score": cv_results.get('std_score', 0),
                f"{prefix}_min_score": cv_results.get('min_score', 0),
                f"{prefix}_max_score": cv_results.get('max_score', 0),
                f"{prefix}_n_splits": cv_results.get('n_splits', 0),
            }
            self.log_metrics(summary_metrics)

            # Log per-fold results
            fold_results = cv_results.get('fold_results', [])
            if fold_results:
                for i, fold in enumerate(fold_results):
                    fold_metrics = {
                        f"{prefix}_fold{i}_{k}": v
                        for k, v in fold.items()
                        if isinstance(v, (int, float))
                    }
                    self.log_metrics(fold_metrics)

            logger.info(f"CV results logged: {prefix}")

        except Exception as e:
            logger.error(f"Failed to log CV results: {e}")

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags on the current run."""
        if not MLFLOW_AVAILABLE or self._active_run is None:
            return

        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Register a model in the model registry.

        Args:
            model_uri: URI of logged model
            name: Model name in registry
            tags: Optional tags

        Returns:
            Model version or None
        """
        if not MLFLOW_AVAILABLE or self._client is None:
            return None

        try:
            result = mlflow.register_model(model_uri, name)
            version = result.version

            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(name, version, key, value)

            logger.info(f"Model registered: {name} v{version}")
            return version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def get_best_run(
        self,
        metric: str = "sharpe_ratio",
        ascending: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run from the experiment.

        Args:
            metric: Metric to optimize
            ascending: Sort direction

        Returns:
            Best run info or None
        """
        if not MLFLOW_AVAILABLE or self._client is None:
            return None

        try:
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )

            if len(runs) > 0:
                return runs.iloc[0].to_dict()
            return None

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare

        Returns:
            DataFrame with comparison
        """
        if not MLFLOW_AVAILABLE or self._client is None:
            return pd.DataFrame()

        try:
            runs_data = []

            for run_id in run_ids:
                run = self._client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'start_time': run.info.start_time,
                    **run.data.params,
                    **run.data.metrics
                }

                if metrics:
                    run_data = {
                        k: v for k, v in run_data.items()
                        if k in metrics or k in ['run_id', 'start_time']
                    }

                runs_data.append(run_data)

            return pd.DataFrame(runs_data)

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return pd.DataFrame()

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, str]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                # Convert to string (MLflow params are strings)
                items.append((new_key, str(v)))

        return dict(items)

    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        info = {}

        try:
            import subprocess

            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            info['git_commit'] = result.stdout.strip()[:8]

            # Get branch name
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            info['git_branch'] = result.stdout.strip()

            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            info['git_dirty'] = 'true' if result.stdout.strip() else 'false'

        except Exception:
            pass

        return info

    def _log_system_metrics(self):
        """Log system information."""
        try:
            import platform
            import sys

            self.set_tags({
                'python_version': sys.version.split()[0],
                'platform': platform.system(),
                'platform_version': platform.version(),
                'hostname': platform.node()
            })

            # Try to log GPU info
            try:
                import torch
                if torch.cuda.is_available():
                    self.set_tags({
                        'cuda_available': 'true',
                        'cuda_device': torch.cuda.get_device_name(0),
                        'cuda_version': torch.version.cuda
                    })
            except ImportError:
                pass

        except Exception:
            pass


class ExperimentManager:
    """
    High-level experiment management.

    Provides utilities for:
    - Hyperparameter search tracking
    - Model comparison
    - Experiment organization
    """

    def __init__(self, tracking_uri: str = "mlruns"):
        """
        Initialize experiment manager.

        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            self._client = MlflowClient()
        else:
            self._client = None

    def create_experiment(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a new experiment."""
        if not MLFLOW_AVAILABLE:
            return name

        experiment = mlflow.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id

        return mlflow.create_experiment(name, tags=tags)

    def list_experiments(self) -> pd.DataFrame:
        """List all experiments."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return pd.DataFrame()

        experiments = self._client.search_experiments()
        return pd.DataFrame([
            {
                'experiment_id': e.experiment_id,
                'name': e.name,
                'artifact_location': e.artifact_location,
                'lifecycle_stage': e.lifecycle_stage,
                'tags': e.tags
            }
            for e in experiments
        ])

    def search_runs(
        self,
        experiment_name: str,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Search runs in an experiment.

        Args:
            experiment_name: Experiment name
            filter_string: MLflow filter string
            order_by: Order by clauses
            max_results: Maximum results

        Returns:
            DataFrame of runs
        """
        if not MLFLOW_AVAILABLE:
            return pd.DataFrame()

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return pd.DataFrame()

        return mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )

    def get_run_artifacts(self, run_id: str) -> List[str]:
        """Get list of artifacts for a run."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return []

        artifacts = self._client.list_artifacts(run_id)
        return [a.path for a in artifacts]

    def load_model(
        self,
        run_id: str,
        artifact_path: str = "model"
    ) -> Any:
        """
        Load a model from a run.

        Args:
            run_id: Run ID
            artifact_path: Model artifact path

        Returns:
            Loaded model
        """
        if not MLFLOW_AVAILABLE:
            return None

        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.pyfunc.load_model(model_uri)

    def delete_run(self, run_id: str):
        """Delete a run."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return

        self._client.delete_run(run_id)

    def restore_run(self, run_id: str):
        """Restore a deleted run."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return

        self._client.restore_run(run_id)


def is_mlflow_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE
