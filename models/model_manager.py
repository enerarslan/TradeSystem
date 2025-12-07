"""
Model Manager Module
====================

Centralized model registry and management for the algorithmic trading platform.
Handles model saving, loading, versioning, and tracking for all 46 symbols.

Features:
- Standardized model naming convention
- Per-symbol model directories
- Model versioning and metadata
- Model registry with performance tracking
- Automatic model discovery
- Model comparison utilities

Naming Convention:
    models/artifacts/{SYMBOL}/{SYMBOL}_{model_type}_{version}.pkl
    models/artifacts/{SYMBOL}/metadata.json

Example Structure:
    models/artifacts/
    ├── AAPL/
    │   ├── AAPL_lightgbm_v1.pkl
    │   ├── AAPL_xgboost_v1.pkl
    │   ├── AAPL_ensemble_v1.pkl
    │   └── metadata.json
    ├── GOOGL/
    │   └── ...
    └── model_registry.json

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from config.settings import get_logger, get_settings
from config.symbols import (
    ALL_SYMBOLS,
    get_model_filename,
    get_model_directory,
    parse_model_filename,
    validate_symbol,
)

logger = get_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    symbol: str
    model_type: str
    version: str
    
    # Training info
    trained_at: str  # ISO format
    training_samples: int = 0
    feature_count: int = 0
    feature_names: list[str] = field(default_factory=list)
    
    # Performance metrics
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_f1: float = 0.0
    test_f1: float = 0.0
    train_auc: float = 0.0
    test_auc: float = 0.0
    
    # Walk-forward results
    walk_forward_accuracy: float = 0.0
    walk_forward_sharpe: float = 0.0
    
    # Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    
    # File paths
    model_path: str = ""
    scaler_path: str = ""
    
    # Status
    is_active: bool = True
    is_production: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SymbolModelRegistry:
    """Registry of models for a single symbol."""
    symbol: str
    models: dict[str, ModelMetadata] = field(default_factory=dict)
    active_model: str | None = None
    production_model: str | None = None
    last_updated: str = ""
    
    def add_model(self, metadata: ModelMetadata) -> None:
        """Add a model to the registry."""
        key = f"{metadata.model_type}_{metadata.version}"
        self.models[key] = metadata
        self.last_updated = datetime.now().isoformat()
    
    def get_model(self, model_type: str, version: str = "v1") -> ModelMetadata | None:
        """Get a specific model."""
        key = f"{model_type}_{version}"
        return self.models.get(key)
    
    def get_best_model(self, metric: str = "test_accuracy") -> ModelMetadata | None:
        """Get the best performing model."""
        if not self.models:
            return None
        
        best = None
        best_score = -float('inf')
        
        for model in self.models.values():
            score = getattr(model, metric, 0)
            if score > best_score:
                best_score = score
                best = model
        
        return best
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "active_model": self.active_model,
            "production_model": self.production_model,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolModelRegistry":
        """Create from dictionary."""
        models = {
            k: ModelMetadata.from_dict(v) 
            for k, v in data.get("models", {}).items()
        }
        return cls(
            symbol=data["symbol"],
            models=models,
            active_model=data.get("active_model"),
            production_model=data.get("production_model"),
            last_updated=data.get("last_updated", ""),
        )


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """
    Centralized model management for all symbols.
    
    Provides:
    - Model saving with standardized naming
    - Model loading with automatic discovery
    - Registry management
    - Performance tracking
    - Model versioning
    
    Example:
        manager = ModelManager()
        
        # Save a model
        manager.save_model(
            model=trained_model,
            symbol="AAPL",
            model_type="lightgbm",
            metrics={"accuracy": 0.58, "f1": 0.55},
        )
        
        # Load a model
        model = manager.load_model("AAPL", "lightgbm")
        
        # Get best model for symbol
        best = manager.get_best_model("AAPL")
    """
    
    def __init__(
        self,
        base_dir: Path | str | None = None,
    ):
        """
        Initialize model manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        settings = get_settings()
        self.base_dir = Path(base_dir) if base_dir else Path(settings.ml.models_path)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry for all symbols
        self._registry: dict[str, SymbolModelRegistry] = {}
        
        # Registry file path
        self._registry_path = self.base_dir / "model_registry.json"
        
        # Load existing registry
        self._load_registry()
    
    # =========================================================================
    # REGISTRY MANAGEMENT
    # =========================================================================
    
    def _load_registry(self) -> None:
        """Load the model registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r") as f:
                    data = json.load(f)
                
                for symbol, symbol_data in data.get("symbols", {}).items():
                    self._registry[symbol] = SymbolModelRegistry.from_dict(symbol_data)
                
                logger.info(f"Loaded model registry with {len(self._registry)} symbols")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                self._registry = {}
    
    def _save_registry(self) -> None:
        """Save the model registry to disk."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "total_models": sum(len(r.models) for r in self._registry.values()),
                "symbols": {s: r.to_dict() for s, r in self._registry.items()},
            }
            
            with open(self._registry_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved model registry to {self._registry_path}")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _get_or_create_symbol_registry(self, symbol: str) -> SymbolModelRegistry:
        """Get or create registry for a symbol."""
        symbol = symbol.upper()
        if symbol not in self._registry:
            self._registry[symbol] = SymbolModelRegistry(symbol=symbol)
        return self._registry[symbol]
    
    # =========================================================================
    # MODEL SAVING
    # =========================================================================
    
    def save_model(
        self,
        model: Any,
        symbol: str,
        model_type: str,
        version: str = "v1",
        metrics: dict[str, float] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        training_samples: int = 0,
        set_active: bool = True,
        set_production: bool = False,
    ) -> Path:
        """
        Save a model with standardized naming.
        
        Args:
            model: Trained model object
            symbol: Trading symbol
            model_type: Type of model (lightgbm, xgboost, ensemble, etc.)
            version: Version string (default: v1)
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            feature_names: Feature names used
            training_samples: Number of training samples
            set_active: Set as active model
            set_production: Set as production model
        
        Returns:
            Path to saved model file
        """
        symbol = symbol.upper()
        metrics = metrics or {}
        hyperparameters = hyperparameters or {}
        feature_names = feature_names or []
        
        # Validate symbol
        if not validate_symbol(symbol):
            logger.warning(f"Unknown symbol: {symbol}")
        
        # Get symbol directory
        symbol_dir = get_model_directory(self.base_dir, symbol)
        
        # Generate filename
        filename = get_model_filename(symbol, model_type, version)
        model_path = symbol_dir / filename
        
        # Save model
        # Check if model has its own save method
        if hasattr(model, 'save'):
            model.save(model_path)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved model to {model_path}")
        
        # Create metadata
        model_id = str(uuid4())[:8]
        metadata = ModelMetadata(
            model_id=model_id,
            symbol=symbol,
            model_type=model_type,
            version=version,
            trained_at=datetime.now().isoformat(),
            training_samples=training_samples,
            feature_count=len(feature_names),
            feature_names=feature_names[:50],  # Store top 50 features
            train_accuracy=metrics.get("train_accuracy", 0),
            test_accuracy=metrics.get("test_accuracy", metrics.get("accuracy", 0)),
            train_f1=metrics.get("train_f1", 0),
            test_f1=metrics.get("test_f1", metrics.get("f1_macro", 0)),
            train_auc=metrics.get("train_auc", 0),
            test_auc=metrics.get("test_auc", metrics.get("roc_auc", 0)),
            walk_forward_accuracy=metrics.get("walk_forward_accuracy", 0),
            walk_forward_sharpe=metrics.get("walk_forward_sharpe", 0),
            hyperparameters=hyperparameters,
            model_path=str(model_path),
            is_active=set_active,
            is_production=set_production,
        )
        
        # Update registry
        symbol_registry = self._get_or_create_symbol_registry(symbol)
        symbol_registry.add_model(metadata)
        
        if set_active:
            symbol_registry.active_model = f"{model_type}_{version}"
        if set_production:
            symbol_registry.production_model = f"{model_type}_{version}"
        
        # Save symbol metadata
        self._save_symbol_metadata(symbol, symbol_registry)
        
        # Save global registry
        self._save_registry()
        
        return model_path
    
    def _save_symbol_metadata(
        self,
        symbol: str,
        registry: SymbolModelRegistry,
    ) -> None:
        """Save metadata file for a symbol."""
        symbol_dir = get_model_directory(self.base_dir, symbol)
        metadata_path = symbol_dir / "metadata.json"
        
        with open(metadata_path, "w") as f:
            json.dump(registry.to_dict(), f, indent=2)
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def load_model(
        self,
        symbol: str,
        model_type: str,
        version: str = "v1",
    ) -> Any:
        """
        Load a model from disk.
        
        Args:
            symbol: Trading symbol
            model_type: Type of model
            version: Version string
        
        Returns:
            Loaded model object
        """
        symbol = symbol.upper()
        symbol_dir = get_model_directory(self.base_dir, symbol, create=False)
        
        if not symbol_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")
        
        filename = get_model_filename(symbol, model_type, version)
        model_path = symbol_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return self._load_model_file(model_path)
    
    def load_active_model(self, symbol: str) -> Any:
        """
        Load the active model for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Loaded model object
        """
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            raise ValueError(f"No models registered for {symbol}")
        
        registry = self._registry[symbol]
        
        if not registry.active_model:
            raise ValueError(f"No active model set for {symbol}")
        
        metadata = registry.models.get(registry.active_model)
        if not metadata:
            raise ValueError(f"Active model metadata not found for {symbol}")
        
        return self._load_model_file(Path(metadata.model_path))
    
    def load_production_model(self, symbol: str) -> Any:
        """
        Load the production model for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Loaded model object
        """
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            raise ValueError(f"No models registered for {symbol}")
        
        registry = self._registry[symbol]
        
        if not registry.production_model:
            raise ValueError(f"No production model set for {symbol}")
        
        metadata = registry.models.get(registry.production_model)
        if not metadata:
            raise ValueError(f"Production model metadata not found for {symbol}")
        
        return self._load_model_file(Path(metadata.model_path))
    
    def _load_model_file(self, path: Path) -> Any:
        """Load model from file with format detection."""
        # Try to detect model type from saved data
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Check if it's a wrapped model with class info
        if isinstance(data, dict) and "class_name" in data:
            class_name = data.get("class_name", "")
            
            # Import the correct class
            if "LightGBM" in class_name:
                from models.classifiers import LightGBMClassifier
                return LightGBMClassifier.load(path)
            elif "XGBoost" in class_name:
                from models.classifiers import XGBoostClassifier
                return XGBoostClassifier.load(path)
            elif "CatBoost" in class_name:
                from models.classifiers import CatBoostClassifier
                return CatBoostClassifier.load(path)
            elif "RandomForest" in class_name:
                from models.classifiers import RandomForestClassifier
                return RandomForestClassifier.load(path)
            elif "Stacking" in class_name:
                from models.classifiers import StackingClassifier
                return StackingClassifier.load(path)
            elif "LSTM" in class_name:
                from models.deep import LSTMModel
                return LSTMModel.load(path)
            elif "Transformer" in class_name:
                from models.deep import TransformerModel
                return TransformerModel.load(path)
        
        # Return raw loaded data
        return data
    
    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================
    
    def discover_models(self) -> dict[str, list[str]]:
        """
        Discover all models in the artifacts directory.
        
        Returns:
            Dictionary mapping symbol to list of model files
        """
        discovered = {}
        
        for symbol_dir in self.base_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.isupper():
                symbol = symbol_dir.name
                models = []
                
                for model_file in symbol_dir.glob("*.pkl"):
                    models.append(model_file.name)
                
                if models:
                    discovered[symbol] = models
        
        return discovered
    
    def sync_registry(self) -> int:
        """
        Sync registry with actual model files on disk.
        
        Returns:
            Number of models synced
        """
        discovered = self.discover_models()
        synced = 0
        
        for symbol, model_files in discovered.items():
            symbol_dir = get_model_directory(self.base_dir, symbol, create=False)
            
            for filename in model_files:
                parsed = parse_model_filename(filename)
                if not parsed:
                    continue
                
                # Check if already in registry
                registry = self._get_or_create_symbol_registry(symbol)
                key = f"{parsed['model_type']}_{parsed['version']}"
                
                if key not in registry.models:
                    # Add to registry
                    metadata = ModelMetadata(
                        model_id=str(uuid4())[:8],
                        symbol=symbol,
                        model_type=parsed["model_type"],
                        version=parsed["version"],
                        trained_at=datetime.now().isoformat(),
                        model_path=str(symbol_dir / filename),
                    )
                    registry.add_model(metadata)
                    synced += 1
                    logger.info(f"Synced: {symbol}/{filename}")
        
        if synced > 0:
            self._save_registry()
        
        return synced
    
    # =========================================================================
    # QUERYING
    # =========================================================================
    
    def get_model_metadata(
        self,
        symbol: str,
        model_type: str,
        version: str = "v1",
    ) -> ModelMetadata | None:
        """Get metadata for a specific model."""
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            return None
        
        return self._registry[symbol].get_model(model_type, version)
    
    def get_best_model(
        self,
        symbol: str,
        metric: str = "test_accuracy",
    ) -> ModelMetadata | None:
        """Get the best performing model for a symbol."""
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            return None
        
        return self._registry[symbol].get_best_model(metric)
    
    def get_all_models_for_symbol(self, symbol: str) -> list[ModelMetadata]:
        """Get all models for a symbol."""
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            return []
        
        return list(self._registry[symbol].models.values())
    
    def get_symbols_with_models(self) -> list[str]:
        """Get list of symbols that have trained models."""
        return sorted(self._registry.keys())
    
    def get_symbols_without_models(self) -> list[str]:
        """Get list of symbols without trained models."""
        with_models = set(self._registry.keys())
        return [s for s in ALL_SYMBOLS if s not in with_models]
    
    def get_model_summary(self) -> dict[str, Any]:
        """Get summary of all models."""
        return {
            "total_symbols": len(self._registry),
            "total_models": sum(len(r.models) for r in self._registry.values()),
            "symbols_with_models": sorted(self._registry.keys()),
            "symbols_without_models": self.get_symbols_without_models(),
            "by_symbol": {
                symbol: {
                    "model_count": len(registry.models),
                    "active": registry.active_model,
                    "production": registry.production_model,
                    "models": list(registry.models.keys()),
                }
                for symbol, registry in self._registry.items()
            },
        }
    
    def get_performance_leaderboard(
        self,
        metric: str = "test_accuracy",
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top performing models across all symbols."""
        all_models = []
        
        for symbol, registry in self._registry.items():
            for key, metadata in registry.models.items():
                score = getattr(metadata, metric, 0)
                all_models.append({
                    "symbol": symbol,
                    "model_key": key,
                    "model_type": metadata.model_type,
                    "version": metadata.version,
                    metric: score,
                    "trained_at": metadata.trained_at,
                })
        
        # Sort by metric
        all_models.sort(key=lambda x: x[metric], reverse=True)
        
        return all_models[:top_n]
    
    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================
    
    def set_active_model(
        self,
        symbol: str,
        model_type: str,
        version: str = "v1",
    ) -> bool:
        """Set the active model for a symbol."""
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            return False
        
        key = f"{model_type}_{version}"
        if key not in self._registry[symbol].models:
            return False
        
        self._registry[symbol].active_model = key
        self._save_registry()
        
        return True
    
    def set_production_model(
        self,
        symbol: str,
        model_type: str,
        version: str = "v1",
    ) -> bool:
        """Set the production model for a symbol."""
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            return False
        
        key = f"{model_type}_{version}"
        if key not in self._registry[symbol].models:
            return False
        
        self._registry[symbol].production_model = key
        self._save_registry()
        
        return True
    
    def delete_model(
        self,
        symbol: str,
        model_type: str,
        version: str = "v1",
        delete_file: bool = True,
    ) -> bool:
        """Delete a model from registry and optionally disk."""
        symbol = symbol.upper()
        
        if symbol not in self._registry:
            return False
        
        key = f"{model_type}_{version}"
        if key not in self._registry[symbol].models:
            return False
        
        metadata = self._registry[symbol].models[key]
        
        # Delete file if requested
        if delete_file and metadata.model_path:
            model_path = Path(metadata.model_path)
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted model file: {model_path}")
        
        # Remove from registry
        del self._registry[symbol].models[key]
        
        # Update active/production if needed
        if self._registry[symbol].active_model == key:
            self._registry[symbol].active_model = None
        if self._registry[symbol].production_model == key:
            self._registry[symbol].production_model = None
        
        self._save_registry()
        
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


def save_model(
    model: Any,
    symbol: str,
    model_type: str,
    **kwargs: Any,
) -> Path:
    """Convenience function to save a model."""
    return get_model_manager().save_model(model, symbol, model_type, **kwargs)


def load_model(
    symbol: str,
    model_type: str,
    version: str = "v1",
) -> Any:
    """Convenience function to load a model."""
    return get_model_manager().load_model(symbol, model_type, version)


def list_models(symbol: str | None = None) -> dict[str, Any]:
    """List available models."""
    manager = get_model_manager()
    
    if symbol:
        return {
            "symbol": symbol,
            "models": [m.to_dict() for m in manager.get_all_models_for_symbol(symbol)]
        }
    else:
        return manager.get_model_summary()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "ModelMetadata",
    "SymbolModelRegistry",
    "ModelManager",
    # Functions
    "get_model_manager",
    "save_model",
    "load_model",
    "list_models",
]