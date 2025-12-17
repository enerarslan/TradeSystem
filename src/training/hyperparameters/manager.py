"""
Hyperparameter Manager for AlphaTrade System.

This module provides centralized hyperparameter management with:
1. YAML-based storage with version control
2. Regime-aware parameter selection
3. Search space definitions for Optuna/Hyperopt
4. Validation against constraints

Reference:
    "Advances in Financial Machine Learning" by Lopez de Prado (2018)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

# Default path for hyperparameter configs
DEFAULT_CONFIG_PATH = Path("config/hyperparameters")


@dataclass
class HyperparameterSet:
    """Container for a set of hyperparameters with metadata."""

    model_type: str                          # e.g., "lightgbm", "xgboost"
    environment: str                         # e.g., "production", "research"
    parameters: Dict[str, Any]               # The actual hyperparameters
    version: str = "1.0"                     # Version string
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""                    # Optional description
    regime: Optional[str] = None             # Optional regime identifier
    hash: Optional[str] = None               # Hash for version control

    def __post_init__(self):
        """Calculate hash after initialization."""
        if self.hash is None:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate hash of parameters for version control."""
        param_str = str(sorted(self.parameters.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type,
            "environment": self.environment,
            "parameters": self.parameters,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "regime": self.regime,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterSet":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            model_type=data["model_type"],
            environment=data["environment"],
            parameters=data["parameters"],
            version=data.get("version", "1.0"),
            created_at=created_at,
            description=data.get("description", ""),
            regime=data.get("regime"),
            hash=data.get("hash"),
        )


class HyperparameterManager:
    """
    Centralized hyperparameter management.

    Provides:
    - Loading/saving from YAML configs
    - Regime-aware parameter selection
    - Search space definitions
    - Constraint validation

    Example usage:
        manager = HyperparameterManager()

        # Get production LightGBM params
        params = manager.get_params("lightgbm", environment="production")

        # Get regime-specific params
        params = manager.get_params(
            "lightgbm",
            environment="production",
            regime="high_volatility"
        )

        # Get search space for optimization
        search_space = manager.get_search_space("lightgbm")
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize hyperparameter manager.

        Args:
            config_path: Path to hyperparameter config directory
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH

        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)

        # Cache loaded configs
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Track parameter history
        self._history: List[HyperparameterSet] = []

    def get_params(
        self,
        model_type: str,
        environment: str = "production",
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get hyperparameters for a model.

        Args:
            model_type: Type of model ("lightgbm", "xgboost", etc.)
            environment: Environment ("production", "research")
            regime: Optional market regime

        Returns:
            Dictionary of hyperparameters
        """
        config = self._load_config(model_type)

        # Start with base environment params
        if environment not in config:
            raise ValueError(f"Unknown environment: {environment}")

        params = config[environment].copy()

        # Override with regime-specific params if specified
        if regime and "regimes" in config and regime in config["regimes"]:
            regime_params = config["regimes"][regime]
            params.update(regime_params)
            logger.info(f"Applied {regime} regime params for {model_type}")

        return params

    def get_search_space(
        self,
        model_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get hyperparameter search space for optimization.

        Args:
            model_type: Type of model

        Returns:
            Dictionary defining search space
        """
        config = self._load_config(model_type)

        if "search_space" not in config:
            raise ValueError(f"No search space defined for {model_type}")

        return config["search_space"]

    def get_optuna_search_space(
        self,
        model_type: str,
    ) -> Dict[str, Any]:
        """
        Get search space formatted for Optuna.

        Args:
            model_type: Type of model

        Returns:
            Dictionary for use with Optuna trial.suggest_*
        """
        search_space = self.get_search_space(model_type)

        # Convert to Optuna format
        optuna_space = {}
        for param, spec in search_space.items():
            optuna_space[param] = {
                "type": spec["type"],
                "low": spec.get("low"),
                "high": spec.get("high"),
                "step": spec.get("step"),
            }

        return optuna_space

    def validate_params(
        self,
        model_type: str,
        params: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate hyperparameters against constraints.

        Args:
            model_type: Type of model
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        config = self._load_config(model_type)
        messages = []
        is_valid = True

        if "constraints" not in config:
            return True, []

        constraints = config["constraints"]

        # Check depth
        if "max_depth_warning" in constraints:
            if params.get("max_depth", 0) > constraints["max_depth_warning"]:
                messages.append(
                    f"WARNING: max_depth ({params['max_depth']}) exceeds "
                    f"recommended maximum ({constraints['max_depth_warning']})"
                )

        # Check regularization
        if "min_regularization_warning" in constraints:
            min_reg = constraints["min_regularization_warning"]
            reg_alpha = params.get("reg_alpha", 0)
            reg_lambda = params.get("reg_lambda", 0)

            if reg_alpha < min_reg and reg_lambda < min_reg:
                messages.append(
                    f"WARNING: Both reg_alpha ({reg_alpha}) and reg_lambda "
                    f"({reg_lambda}) are below recommended minimum ({min_reg})"
                )

        # Check num_leaves for LightGBM
        if "max_leaves_warning" in constraints:
            if params.get("num_leaves", 0) > constraints["max_leaves_warning"]:
                messages.append(
                    f"WARNING: num_leaves ({params['num_leaves']}) exceeds "
                    f"recommended maximum ({constraints['max_leaves_warning']})"
                )

        return is_valid, messages

    def save_params(
        self,
        params: HyperparameterSet,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save hyperparameter set to file.

        Args:
            params: HyperparameterSet to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{params.model_type}_{params.environment}_{timestamp}.yaml"

        filepath = self.config_path / "saved" / filename

        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(params.to_dict(), f, default_flow_style=False)

        logger.info(f"Saved hyperparameters to {filepath}")

        return filepath

    def load_saved_params(
        self,
        filename: str,
    ) -> HyperparameterSet:
        """
        Load saved hyperparameter set.

        Args:
            filename: Filename to load

        Returns:
            HyperparameterSet
        """
        filepath = self.config_path / "saved" / filename

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        return HyperparameterSet.from_dict(data)

    def record_training_params(
        self,
        model_type: str,
        params: Dict[str, Any],
        environment: str = "production",
        regime: Optional[str] = None,
        description: str = "",
    ) -> HyperparameterSet:
        """
        Record hyperparameters used for training (for audit trail).

        Args:
            model_type: Type of model
            params: Parameters used
            environment: Environment
            regime: Market regime if applicable
            description: Optional description

        Returns:
            HyperparameterSet record
        """
        param_set = HyperparameterSet(
            model_type=model_type,
            environment=environment,
            parameters=params,
            regime=regime,
            description=description,
        )

        self._history.append(param_set)
        logger.info(
            f"Recorded {model_type} params: hash={param_set.hash}, "
            f"regime={regime}"
        )

        return param_set

    def get_history(self) -> List[HyperparameterSet]:
        """Get history of recorded parameter sets."""
        return self._history.copy()

    def _load_config(self, model_type: str) -> Dict[str, Any]:
        """Load and cache config for model type."""
        if model_type in self._cache:
            return self._cache[model_type]

        filename = f"{model_type}_defaults.yaml"
        filepath = self.config_path / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"No config found for {model_type} at {filepath}"
            )

        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        self._cache[model_type] = config

        return config

    def clear_cache(self) -> None:
        """Clear the config cache."""
        self._cache.clear()


def load_hyperparameters(
    model_type: str,
    environment: str = "production",
    regime: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load hyperparameters.

    Args:
        model_type: Type of model
        environment: Environment
        regime: Optional market regime
        config_path: Optional config path

    Returns:
        Dictionary of hyperparameters
    """
    manager = HyperparameterManager(config_path=config_path)
    return manager.get_params(model_type, environment, regime)


def save_hyperparameters(
    model_type: str,
    params: Dict[str, Any],
    environment: str = "production",
    regime: Optional[str] = None,
    description: str = "",
    config_path: Optional[Path] = None,
) -> Path:
    """
    Convenience function to save hyperparameters.

    Args:
        model_type: Type of model
        params: Parameters to save
        environment: Environment
        regime: Optional regime
        description: Optional description
        config_path: Optional config path

    Returns:
        Path to saved file
    """
    manager = HyperparameterManager(config_path=config_path)

    param_set = HyperparameterSet(
        model_type=model_type,
        environment=environment,
        parameters=params,
        regime=regime,
        description=description,
    )

    return manager.save_params(param_set)


class OptunaSuggestor:
    """
    Helper class to suggest hyperparameters in Optuna trials.

    Example usage:
        def objective(trial):
            suggestor = OptunaSuggestor("lightgbm")
            params = suggestor.suggest_all(trial)

            model = LGBMClassifier(**params)
            # ... train and evaluate
    """

    def __init__(
        self,
        model_type: str,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize suggestor.

        Args:
            model_type: Type of model
            config_path: Optional config path
        """
        self.manager = HyperparameterManager(config_path=config_path)
        self.model_type = model_type
        self.search_space = self.manager.get_search_space(model_type)

    def suggest_all(self, trial: Any) -> Dict[str, Any]:
        """
        Suggest all hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for name, spec in self.search_space.items():
            params[name] = self._suggest_param(trial, name, spec)

        return params

    def _suggest_param(
        self,
        trial: Any,
        name: str,
        spec: Dict[str, Any],
    ) -> Any:
        """Suggest a single parameter."""
        param_type = spec["type"]

        if param_type == "int":
            step = spec.get("step", 1)
            return trial.suggest_int(
                name, spec["low"], spec["high"], step=step
            )

        elif param_type == "float":
            return trial.suggest_float(
                name, spec["low"], spec["high"]
            )

        elif param_type == "loguniform":
            return trial.suggest_float(
                name, spec["low"], spec["high"], log=True
            )

        elif param_type == "categorical":
            return trial.suggest_categorical(
                name, spec["choices"]
            )

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
