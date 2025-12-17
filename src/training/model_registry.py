"""
MLflow Model Registry for model lifecycle management.

This module provides institutional-grade model governance including:
- Model versioning and staging
- Approval workflows
- Model comparison and promotion
- Production deployment tracking

Designed for JPMorgan-level requirements:
- Full audit trail of model deployments
- Staging workflow (None → Staging → Production → Archived)
- Model lineage tracking
- Rollback capabilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class RegisteredModel:
    """Information about a registered model."""
    name: str
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    description: Optional[str]
    latest_versions: List[Dict[str, Any]]
    tags: Dict[str, str]


@dataclass
class ModelVersionInfo:
    """Information about a specific model version."""
    name: str
    version: int
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    current_stage: str
    source: str
    run_id: str
    status: str
    description: Optional[str]
    tags: Dict[str, str]


@dataclass
class ModelComparison:
    """Comparison results between model versions."""
    model_name: str
    versions: List[int]
    metrics: pd.DataFrame
    best_version: int
    best_metric_value: float
    comparison_metric: str


class ModelRegistry:
    """
    MLflow Model Registry for production model management.

    Provides staging workflow:
    1. None - Initial state after registration
    2. Staging - Under testing/validation
    3. Production - Serving predictions
    4. Archived - Retired from use

    Example:
        registry = ModelRegistry()

        # Register a model from a run
        version = registry.register_model(
            run_id="abc123",
            model_name="momentum_strategy_v1"
        )

        # Promote to staging for testing
        registry.transition_stage(
            model_name="momentum_strategy_v1",
            version=1,
            stage=ModelStage.STAGING
        )

        # After validation, promote to production
        registry.transition_stage(
            model_name="momentum_strategy_v1",
            version=1,
            stage=ModelStage.PRODUCTION
        )

        # Load production model for inference
        model = registry.load_production_model("momentum_strategy_v1")
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is required for ModelRegistry. "
                "Install with: pip install mlflow"
            )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._client = MlflowClient()
        logger.info("ModelRegistry initialized")

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersionInfo:
        """
        Register a model from an MLflow run.

        Args:
            run_id: ID of the run containing the model
            model_name: Name for the registered model
            artifact_path: Path to model artifact within run
            description: Optional description
            tags: Optional tags for the model version

        Returns:
            ModelVersionInfo with registration details
        """
        # Build model URI
        model_uri = f"runs:/{run_id}/{artifact_path}"

        # Register the model
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

            # Update description if provided
            if description:
                self._client.update_model_version(
                    name=model_name,
                    version=result.version,
                    description=description,
                )

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(
                        name=model_name,
                        version=result.version,
                        key=key,
                        value=value,
                    )

            logger.info(
                f"Registered model '{model_name}' version {result.version} "
                f"from run {run_id}"
            )

            return self._version_to_info(result)

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def create_registered_model(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> RegisteredModel:
        """
        Create a new registered model (without any versions).

        Args:
            name: Model name
            description: Optional description
            tags: Optional tags

        Returns:
            RegisteredModel info
        """
        try:
            registered = self._client.create_registered_model(
                name=name,
                description=description,
                tags=tags,
            )

            return RegisteredModel(
                name=registered.name,
                creation_timestamp=datetime.fromtimestamp(
                    registered.creation_timestamp / 1000
                ),
                last_updated_timestamp=datetime.fromtimestamp(
                    registered.last_updated_timestamp / 1000
                ),
                description=registered.description,
                latest_versions=[],
                tags=dict(registered.tags) if registered.tags else {},
            )

        except Exception as e:
            logger.error(f"Failed to create registered model: {e}")
            raise

    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: ModelStage,
        archive_existing: bool = True,
    ) -> ModelVersionInfo:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Name of the registered model
            version: Version number to transition
            stage: Target stage
            archive_existing: If True, archive current model in target stage

        Returns:
            Updated ModelVersionInfo
        """
        try:
            result = self._client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=stage.value,
                archive_existing_versions=archive_existing,
            )

            logger.info(
                f"Transitioned '{model_name}' v{version} to stage '{stage.value}'"
            )

            return self._version_to_info(result)

        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise

    def get_model_version(
        self,
        model_name: str,
        version: int,
    ) -> ModelVersionInfo:
        """Get information about a specific model version."""
        result = self._client.get_model_version(
            name=model_name,
            version=str(version),
        )
        return self._version_to_info(result)

    def get_latest_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None,
    ) -> List[ModelVersionInfo]:
        """
        Get latest model versions, optionally filtered by stage.

        Args:
            model_name: Name of the registered model
            stages: List of stages to filter by (None = all stages)

        Returns:
            List of ModelVersionInfo for latest versions
        """
        versions = self._client.get_latest_versions(
            name=model_name,
            stages=stages,
        )
        return [self._version_to_info(v) for v in versions]

    def get_production_version(
        self,
        model_name: str,
    ) -> Optional[ModelVersionInfo]:
        """Get the current production version of a model."""
        versions = self.get_latest_versions(
            model_name,
            stages=[ModelStage.PRODUCTION.value]
        )
        return versions[0] if versions else None

    def get_staging_version(
        self,
        model_name: str,
    ) -> Optional[ModelVersionInfo]:
        """Get the current staging version of a model."""
        versions = self.get_latest_versions(
            model_name,
            stages=[ModelStage.STAGING.value]
        )
        return versions[0] if versions else None

    def load_model(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[ModelStage] = None,
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the registered model
            version: Specific version to load (takes precedence)
            stage: Stage to load from (if version not specified)

        Returns:
            Loaded model object
        """
        if version is not None:
            model_uri = f"models:/{model_name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{model_name}/{stage.value}"
        else:
            # Default to production
            model_uri = f"models:/{model_name}/{ModelStage.PRODUCTION.value}"

        try:
            # Try to load with appropriate flavor
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_production_model(self, model_name: str) -> Any:
        """Load the production version of a model."""
        return self.load_model(model_name, stage=ModelStage.PRODUCTION)

    def load_staging_model(self, model_name: str) -> Any:
        """Load the staging version of a model."""
        return self.load_model(model_name, stage=ModelStage.STAGING)

    def compare_versions(
        self,
        model_name: str,
        versions: List[int],
        metric: str = "sharpe_ratio",
        higher_is_better: bool = True,
    ) -> ModelComparison:
        """
        Compare multiple model versions by their metrics.

        Args:
            model_name: Name of the registered model
            versions: List of versions to compare
            metric: Metric to compare by
            higher_is_better: If True, higher metric values are better

        Returns:
            ModelComparison with comparison results
        """
        comparison_data = []

        for version in versions:
            version_info = self.get_model_version(model_name, version)

            # Get run metrics
            run = mlflow.get_run(version_info.run_id)
            metrics = run.data.metrics
            params = run.data.params

            comparison_data.append({
                'version': version,
                'stage': version_info.current_stage,
                'run_id': version_info.run_id,
                metric: metrics.get(metric),
                **{k: v for k, v in metrics.items() if k != metric},
            })

        df = pd.DataFrame(comparison_data)

        # Find best version
        if higher_is_better:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()

        best_version = df.loc[best_idx, 'version']
        best_value = df.loc[best_idx, metric]

        return ModelComparison(
            model_name=model_name,
            versions=versions,
            metrics=df,
            best_version=int(best_version),
            best_metric_value=float(best_value),
            comparison_metric=metric,
        )

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        archive_existing: bool = True,
    ) -> ModelVersionInfo:
        """
        Promote a model version directly to production.

        Args:
            model_name: Name of the registered model
            version: Version to promote
            archive_existing: If True, archive current production model

        Returns:
            Updated ModelVersionInfo
        """
        return self.transition_stage(
            model_name=model_name,
            version=version,
            stage=ModelStage.PRODUCTION,
            archive_existing=archive_existing,
        )

    def archive_version(
        self,
        model_name: str,
        version: int,
    ) -> ModelVersionInfo:
        """Archive a model version."""
        return self.transition_stage(
            model_name=model_name,
            version=version,
            stage=ModelStage.ARCHIVED,
            archive_existing=False,
        )

    def delete_version(
        self,
        model_name: str,
        version: int,
    ) -> None:
        """Delete a model version (must be in 'None' or 'Archived' stage)."""
        self._client.delete_model_version(
            name=model_name,
            version=str(version),
        )
        logger.info(f"Deleted '{model_name}' version {version}")

    def delete_model(
        self,
        model_name: str,
    ) -> None:
        """Delete a registered model and all its versions."""
        self._client.delete_registered_model(name=model_name)
        logger.info(f"Deleted registered model '{model_name}'")

    def list_registered_models(self) -> List[RegisteredModel]:
        """List all registered models."""
        models = self._client.search_registered_models()
        return [
            RegisteredModel(
                name=m.name,
                creation_timestamp=datetime.fromtimestamp(
                    m.creation_timestamp / 1000
                ),
                last_updated_timestamp=datetime.fromtimestamp(
                    m.last_updated_timestamp / 1000
                ),
                description=m.description,
                latest_versions=[
                    self._version_to_dict(v) for v in (m.latest_versions or [])
                ],
                tags=dict(m.tags) if m.tags else {},
            )
            for m in models
        ]

    def search_model_versions(
        self,
        filter_string: str = "",
        max_results: int = 100,
    ) -> List[ModelVersionInfo]:
        """
        Search model versions with filter.

        Args:
            filter_string: MLflow filter (e.g., "name='my_model'")
            max_results: Maximum results to return

        Returns:
            List of matching ModelVersionInfo
        """
        versions = self._client.search_model_versions(
            filter_string=filter_string,
            max_results=max_results,
        )
        return [self._version_to_info(v) for v in versions]

    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        """Set a tag on a model version."""
        self._client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key=key,
            value=value,
        )

    def update_model_description(
        self,
        model_name: str,
        description: str,
    ) -> None:
        """Update the description of a registered model."""
        self._client.update_registered_model(
            name=model_name,
            description=description,
        )

    def update_version_description(
        self,
        model_name: str,
        version: int,
        description: str,
    ) -> None:
        """Update the description of a model version."""
        self._client.update_model_version(
            name=model_name,
            version=str(version),
            description=description,
        )

    def _version_to_info(self, version: ModelVersion) -> ModelVersionInfo:
        """Convert MLflow ModelVersion to ModelVersionInfo."""
        return ModelVersionInfo(
            name=version.name,
            version=int(version.version),
            creation_timestamp=datetime.fromtimestamp(
                version.creation_timestamp / 1000
            ),
            last_updated_timestamp=datetime.fromtimestamp(
                version.last_updated_timestamp / 1000
            ),
            current_stage=version.current_stage,
            source=version.source,
            run_id=version.run_id,
            status=version.status,
            description=version.description,
            tags=dict(version.tags) if version.tags else {},
        )

    def _version_to_dict(self, version: ModelVersion) -> Dict[str, Any]:
        """Convert MLflow ModelVersion to dictionary."""
        return {
            'version': int(version.version),
            'stage': version.current_stage,
            'run_id': version.run_id,
            'status': version.status,
        }


# Convenience functions
def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
) -> ModelVersionInfo:
    """Quick function to register a model."""
    registry = ModelRegistry(tracking_uri=tracking_uri)
    return registry.register_model(
        run_id=run_id,
        model_name=model_name,
        artifact_path=artifact_path,
    )


def load_production_model(
    model_name: str,
    tracking_uri: Optional[str] = None,
) -> Any:
    """Quick function to load production model."""
    registry = ModelRegistry(tracking_uri=tracking_uri)
    return registry.load_production_model(model_name)
