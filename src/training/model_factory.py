"""
Model Factory for standardized model creation and configuration.

This module provides a factory pattern for creating ML models with:
- Consistent interface across model types
- Default hyperparameters per model type
- Optuna parameter space definitions
- GPU/device configuration
- Serialization support

Designed for JPMorgan-level requirements:
- Standardized model interfaces
- Configuration-driven model creation
- Support for ensemble methods
- Production-ready serialization
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from sklearn.ensemble import (
        StackingClassifier, StackingRegressor,
        VotingClassifier, VotingRegressor
    )
    SKLEARN_ENSEMBLE_AVAILABLE = True
except ImportError:
    SKLEARN_ENSEMBLE_AVAILABLE = False


logger = logging.getLogger(__name__)


# ==============================================================================
# PyTorch Neural Network Architectures (JPMorgan Institutional-Level)
# ==============================================================================

if PYTORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """
        LSTM model for time series prediction.

        Architecture follows institutional best practices:
        - Bidirectional LSTM for capturing past and future context
        - Layer normalization for training stability
        - Dropout for regularization
        - Residual connections for gradient flow
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = True,
            output_size: int = 1,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            lstm_output_size = hidden_size * (2 if bidirectional else 1)

            self.layer_norm = nn.LayerNorm(lstm_output_size)
            self.dropout = nn.Dropout(dropout)

            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take last timestep output
            out = lstm_out[:, -1, :]
            out = self.layer_norm(out)
            out = self.dropout(out)
            return self.fc(out)


    class GRUModel(nn.Module):
        """
        GRU model for time series prediction.

        GRU is computationally more efficient than LSTM
        while maintaining comparable performance.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = True,
            output_size: int = 1,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            gru_output_size = hidden_size * (2 if bidirectional else 1)

            self.layer_norm = nn.LayerNorm(gru_output_size)
            self.dropout = nn.Dropout(dropout)

            self.fc = nn.Sequential(
                nn.Linear(gru_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gru_out, _ = self.gru(x)
            out = gru_out[:, -1, :]
            out = self.layer_norm(out)
            out = self.dropout(out)
            return self.fc(out)


    class PositionalEncoding(nn.Module):
        """Positional encoding for Transformer models."""

        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class TransformerModel(nn.Module):
        """
        Transformer model for time series prediction.

        Based on "Attention Is All You Need" (Vaswani et al., 2017)
        adapted for financial time series.

        Features:
        - Multi-head self-attention
        - Positional encoding
        - Layer normalization
        - Feedforward networks
        """

        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            output_size: int = 1,
            max_seq_len: int = 500,
        ):
            super().__init__()

            self.input_projection = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for better training stability
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            self.output_layer = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, input_size)
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            # Use CLS-like approach: take mean of sequence
            x = x.mean(dim=1)
            return self.output_layer(x)


    class NeuralNetWrapper:
        """
        Sklearn-compatible wrapper for PyTorch neural networks.

        Provides fit/predict interface compatible with sklearn,
        enabling use in pipelines and cross-validation.
        """

        def __init__(
            self,
            model_class: Type[nn.Module],
            model_params: Dict[str, Any],
            task_type: str = "classification",
            learning_rate: float = 1e-3,
            batch_size: int = 64,
            epochs: int = 100,
            early_stopping_patience: int = 10,
            device: Optional[str] = None,
            sequence_length: int = 20,
            random_state: int = 42,
        ):
            self.model_class = model_class
            self.model_params = model_params
            self.task_type = task_type
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.early_stopping_patience = early_stopping_patience
            self.sequence_length = sequence_length
            self.random_state = random_state

            if device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = device

            self.model_ = None
            self.scaler_ = None
            self.classes_ = None

        def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
            """Prepare sequential data for RNN/Transformer models."""
            if len(X.shape) == 2:
                # Create sequences from 2D data
                n_samples = len(X) - self.sequence_length + 1
                n_features = X.shape[1]
                sequences = np.zeros((n_samples, self.sequence_length, n_features))
                for i in range(n_samples):
                    sequences[i] = X[i:i + self.sequence_length]
                return sequences
            return X

        def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetWrapper":
            """Fit the neural network model."""
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            # Prepare data
            X_seq = self._prepare_sequences(X)
            y_aligned = y[self.sequence_length - 1:] if len(X.shape) == 2 else y

            # Store classes for classification
            if self.task_type == "classification":
                self.classes_ = np.unique(y_aligned)

            # Initialize model
            input_size = X_seq.shape[-1]
            output_size = len(self.classes_) if self.task_type == "classification" else 1

            self.model_ = self.model_class(
                input_size=input_size,
                output_size=output_size,
                **self.model_params
            ).to(self.device)

            # Loss and optimizer
            if self.task_type == "classification":
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()

            optimizer = torch.optim.AdamW(
                self.model_.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            if self.task_type == "classification":
                y_tensor = torch.LongTensor(y_aligned).to(self.device)
            else:
                y_tensor = torch.FloatTensor(y_aligned).unsqueeze(1).to(self.device)

            # Training loop with early stopping
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size
            )

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.epochs):
                # Training
                self.model_.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model_(batch_X)
                    loss = criterion(outputs, batch_y.squeeze() if self.task_type == "classification" else batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                self.model_.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model_(batch_X)
                        loss = criterion(outputs, batch_y.squeeze() if self.task_type == "classification" else batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions."""
            self.model_.eval()
            X_seq = self._prepare_sequences(X)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)

            with torch.no_grad():
                outputs = self.model_(X_tensor)
                if self.task_type == "classification":
                    predictions = outputs.argmax(dim=1).cpu().numpy()
                else:
                    predictions = outputs.squeeze().cpu().numpy()

            return predictions

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities (classification only)."""
            if self.task_type != "classification":
                raise ValueError("predict_proba only available for classification")

            self.model_.eval()
            X_seq = self._prepare_sequences(X)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)

            with torch.no_grad():
                outputs = self.model_(X_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

            return probabilities

        @property
        def feature_importances_(self) -> np.ndarray:
            """Return placeholder feature importances (not available for neural nets)."""
            # Neural networks don't have native feature importance
            # Use SHAP or integrated gradients for explanation
            return np.array([])


class ModelType(str, Enum):
    """Supported model types."""
    # Gradient Boosting Models
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    XGBOOST_REGRESSOR = "xgboost_regressor"
    CATBOOST_CLASSIFIER = "catboost_classifier"
    CATBOOST_REGRESSOR = "catboost_regressor"
    # Traditional ML Models
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    LOGISTIC_REGRESSION = "logistic_regression"
    # Neural Network Models (JPMorgan Institutional-Level)
    LSTM_CLASSIFIER = "lstm_classifier"
    LSTM_REGRESSOR = "lstm_regressor"
    GRU_CLASSIFIER = "gru_classifier"
    GRU_REGRESSOR = "gru_regressor"
    TRANSFORMER_CLASSIFIER = "transformer_classifier"
    TRANSFORMER_REGRESSOR = "transformer_regressor"
    # Ensemble Models
    STACKING_CLASSIFIER = "stacking_classifier"
    STACKING_REGRESSOR = "stacking_regressor"
    VOTING_CLASSIFIER = "voting_classifier"
    VOTING_REGRESSOR = "voting_regressor"


class TaskType(str, Enum):
    """ML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ParamSpace:
    """Parameter space definition for hyperparameter optimization."""
    name: str
    param_type: str  # int, float, float_log, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    log: bool = False

    def sample(self, trial: "optuna.Trial") -> Any:
        """Sample a value from this parameter space."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for hyperparameter sampling")

        if self.param_type == "int":
            return trial.suggest_int(
                self.name, int(self.low), int(self.high),
                step=int(self.step) if self.step else 1
            )
        elif self.param_type == "float":
            return trial.suggest_float(
                self.name, self.low, self.high,
                step=self.step, log=False
            )
        elif self.param_type == "float_log":
            return trial.suggest_float(
                self.name, self.low, self.high, log=True
            )
        elif self.param_type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise ValueError(f"Unknown param type: {self.param_type}")


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_type: ModelType
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)
    param_spaces: List[ParamSpace] = field(default_factory=list)
    random_state: int = 42
    n_jobs: int = -1
    gpu_enabled: bool = False


# Default parameters for each model type
DEFAULT_PARAMS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.LIGHTGBM_CLASSIFIER: {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
    },
    ModelType.LIGHTGBM_REGRESSOR: {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
    },
    ModelType.XGBOOST_CLASSIFIER: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 1,
        "gamma": 0,
        "tree_method": "hist",
        "verbosity": 0,
    },
    ModelType.XGBOOST_REGRESSOR: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 1,
        "gamma": 0,
        "tree_method": "hist",
        "verbosity": 0,
    },
    ModelType.CATBOOST_CLASSIFIER: {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3,
        "verbose": False,
    },
    ModelType.CATBOOST_REGRESSOR: {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3,
        "verbose": False,
    },
    ModelType.RANDOM_FOREST_CLASSIFIER: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
    },
    ModelType.RANDOM_FOREST_REGRESSOR: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
    },
    ModelType.RIDGE: {
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "auto",
    },
    ModelType.LASSO: {
        "alpha": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
    },
    ModelType.ELASTIC_NET: {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "max_iter": 1000,
    },
    ModelType.LOGISTIC_REGRESSION: {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    # Neural Network Default Parameters
    ModelType.LSTM_CLASSIFIER: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 10,
        "sequence_length": 20,
    },
    ModelType.LSTM_REGRESSOR: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 10,
        "sequence_length": 20,
    },
    ModelType.GRU_CLASSIFIER: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 10,
        "sequence_length": 20,
    },
    ModelType.GRU_REGRESSOR: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 10,
        "sequence_length": 20,
    },
    ModelType.TRANSFORMER_CLASSIFIER: {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 15,
        "sequence_length": 50,
    },
    ModelType.TRANSFORMER_REGRESSOR: {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 15,
        "sequence_length": 50,
    },
}

# Default parameter spaces for Optuna optimization
DEFAULT_PARAM_SPACES: Dict[ModelType, List[ParamSpace]] = {
    ModelType.LIGHTGBM_CLASSIFIER: [
        ParamSpace("n_estimators", "int", 50, 500),
        ParamSpace("max_depth", "int", 3, 12),
        ParamSpace("num_leaves", "int", 20, 100),
        ParamSpace("learning_rate", "float_log", 0.01, 0.3),
        ParamSpace("subsample", "float", 0.6, 1.0),
        ParamSpace("colsample_bytree", "float", 0.6, 1.0),
        ParamSpace("reg_alpha", "float_log", 1e-8, 10.0),
        ParamSpace("reg_lambda", "float_log", 1e-8, 10.0),
        ParamSpace("min_child_samples", "int", 10, 100),
    ],
    ModelType.XGBOOST_CLASSIFIER: [
        ParamSpace("n_estimators", "int", 50, 500),
        ParamSpace("max_depth", "int", 3, 12),
        ParamSpace("learning_rate", "float_log", 0.01, 0.3),
        ParamSpace("subsample", "float", 0.6, 1.0),
        ParamSpace("colsample_bytree", "float", 0.6, 1.0),
        ParamSpace("reg_alpha", "float_log", 1e-8, 10.0),
        ParamSpace("reg_lambda", "float_log", 1e-8, 10.0),
        ParamSpace("min_child_weight", "int", 1, 20),
        ParamSpace("gamma", "float", 0, 5),
    ],
    ModelType.CATBOOST_CLASSIFIER: [
        ParamSpace("iterations", "int", 50, 500),
        ParamSpace("depth", "int", 3, 10),
        ParamSpace("learning_rate", "float_log", 0.01, 0.3),
        ParamSpace("l2_leaf_reg", "float_log", 1.0, 10.0),
        ParamSpace("bagging_temperature", "float", 0, 3),
    ],
    ModelType.RANDOM_FOREST_CLASSIFIER: [
        ParamSpace("n_estimators", "int", 50, 300),
        ParamSpace("max_depth", "int", 5, 30),
        ParamSpace("min_samples_split", "int", 2, 20),
        ParamSpace("min_samples_leaf", "int", 1, 10),
        ParamSpace("max_features", "categorical", choices=["sqrt", "log2", 0.5, 0.7]),
    ],
}

# Copy classifier spaces to regressors
DEFAULT_PARAM_SPACES[ModelType.LIGHTGBM_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.LIGHTGBM_CLASSIFIER]
DEFAULT_PARAM_SPACES[ModelType.XGBOOST_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.XGBOOST_CLASSIFIER]
DEFAULT_PARAM_SPACES[ModelType.CATBOOST_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.CATBOOST_CLASSIFIER]
DEFAULT_PARAM_SPACES[ModelType.RANDOM_FOREST_REGRESSOR] = DEFAULT_PARAM_SPACES[ModelType.RANDOM_FOREST_CLASSIFIER]

# Neural Network parameter spaces
_LSTM_PARAM_SPACES = [
    ParamSpace("hidden_size", "categorical", choices=[64, 128, 256]),
    ParamSpace("num_layers", "int", 1, 4),
    ParamSpace("dropout", "float", 0.1, 0.5),
    ParamSpace("learning_rate", "float_log", 1e-5, 1e-2),
    ParamSpace("batch_size", "categorical", choices=[32, 64, 128]),
    ParamSpace("sequence_length", "categorical", choices=[10, 20, 50]),
]

_TRANSFORMER_PARAM_SPACES = [
    ParamSpace("d_model", "categorical", choices=[64, 128, 256]),
    ParamSpace("nhead", "categorical", choices=[4, 8, 16]),
    ParamSpace("num_layers", "int", 2, 6),
    ParamSpace("dim_feedforward", "categorical", choices=[256, 512, 1024]),
    ParamSpace("dropout", "float", 0.05, 0.3),
    ParamSpace("learning_rate", "float_log", 1e-5, 1e-3),
    ParamSpace("sequence_length", "categorical", choices=[20, 50, 100]),
]

DEFAULT_PARAM_SPACES[ModelType.LSTM_CLASSIFIER] = _LSTM_PARAM_SPACES
DEFAULT_PARAM_SPACES[ModelType.LSTM_REGRESSOR] = _LSTM_PARAM_SPACES
DEFAULT_PARAM_SPACES[ModelType.GRU_CLASSIFIER] = _LSTM_PARAM_SPACES
DEFAULT_PARAM_SPACES[ModelType.GRU_REGRESSOR] = _LSTM_PARAM_SPACES
DEFAULT_PARAM_SPACES[ModelType.TRANSFORMER_CLASSIFIER] = _TRANSFORMER_PARAM_SPACES
DEFAULT_PARAM_SPACES[ModelType.TRANSFORMER_REGRESSOR] = _TRANSFORMER_PARAM_SPACES


class ModelFactory:
    """
    Factory for creating and configuring ML models.

    Provides standardized model creation with:
    - Consistent hyperparameter handling
    - GPU detection and configuration
    - Optuna parameter space generation
    - Model serialization/deserialization

    Example:
        # Create model with default params
        model = ModelFactory.create_model(ModelType.LIGHTGBM_CLASSIFIER)

        # Create model with custom params
        model = ModelFactory.create_model(
            ModelType.XGBOOST_CLASSIFIER,
            params={"n_estimators": 200, "max_depth": 8}
        )

        # Get Optuna parameter space
        space = ModelFactory.get_param_space(ModelType.LIGHTGBM_CLASSIFIER)

        # Create model from Optuna trial
        model = ModelFactory.create_model_from_trial(
            ModelType.LIGHTGBM_CLASSIFIER,
            trial
        )
    """

    @staticmethod
    def create_model(
        model_type: Union[ModelType, str],
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        gpu_enabled: bool = False,
    ) -> Any:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            params: Model hyperparameters (merged with defaults)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            gpu_enabled: Whether to enable GPU acceleration

        Returns:
            Instantiated model object
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        # Get default params and merge with provided
        default_params = DEFAULT_PARAMS.get(model_type, {}).copy()
        if params:
            default_params.update(params)

        # Add random state and n_jobs where applicable
        if "random_state" not in default_params:
            default_params["random_state"] = random_state
        if "n_jobs" not in default_params and "n_jobs" in str(model_type):
            default_params["n_jobs"] = n_jobs

        # Create model based on type
        return ModelFactory._create_model_instance(
            model_type, default_params, gpu_enabled
        )

    @staticmethod
    def _create_model_instance(
        model_type: ModelType,
        params: Dict[str, Any],
        gpu_enabled: bool,
    ) -> Any:
        """Create the actual model instance."""

        # LightGBM models
        if model_type == ModelType.LIGHTGBM_CLASSIFIER:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            if gpu_enabled:
                params["device"] = "gpu"
            return lgb.LGBMClassifier(**params)

        elif model_type == ModelType.LIGHTGBM_REGRESSOR:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            if gpu_enabled:
                params["device"] = "gpu"
            return lgb.LGBMRegressor(**params)

        # XGBoost models
        elif model_type == ModelType.XGBOOST_CLASSIFIER:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            if gpu_enabled:
                params["tree_method"] = "gpu_hist"
                params["predictor"] = "gpu_predictor"
            return xgb.XGBClassifier(**params)

        elif model_type == ModelType.XGBOOST_REGRESSOR:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            if gpu_enabled:
                params["tree_method"] = "gpu_hist"
                params["predictor"] = "gpu_predictor"
            return xgb.XGBRegressor(**params)

        # CatBoost models
        elif model_type == ModelType.CATBOOST_CLASSIFIER:
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            if gpu_enabled:
                params["task_type"] = "GPU"
            return cb.CatBoostClassifier(**params)

        elif model_type == ModelType.CATBOOST_REGRESSOR:
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            if gpu_enabled:
                params["task_type"] = "GPU"
            return cb.CatBoostRegressor(**params)

        # Sklearn models
        elif model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return RandomForestClassifier(**params)

        elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return RandomForestRegressor(**params)

        elif model_type == ModelType.RIDGE:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return Ridge(**params)

        elif model_type == ModelType.LASSO:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return Lasso(**params)

        elif model_type == ModelType.ELASTIC_NET:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return ElasticNet(**params)

        elif model_type == ModelType.LOGISTIC_REGRESSION:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return LogisticRegression(**params)

        # Neural Network Models
        elif model_type in [ModelType.LSTM_CLASSIFIER, ModelType.LSTM_REGRESSOR]:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available for LSTM models")
            task_type = "classification" if "classifier" in model_type.value else "regression"
            model_params = {
                "hidden_size": params.get("hidden_size", 128),
                "num_layers": params.get("num_layers", 2),
                "dropout": params.get("dropout", 0.2),
                "bidirectional": params.get("bidirectional", True),
            }
            return NeuralNetWrapper(
                model_class=LSTMModel,
                model_params=model_params,
                task_type=task_type,
                learning_rate=params.get("learning_rate", 1e-3),
                batch_size=params.get("batch_size", 64),
                epochs=params.get("epochs", 100),
                early_stopping_patience=params.get("early_stopping_patience", 10),
                sequence_length=params.get("sequence_length", 20),
                random_state=params.get("random_state", 42),
            )

        elif model_type in [ModelType.GRU_CLASSIFIER, ModelType.GRU_REGRESSOR]:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available for GRU models")
            task_type = "classification" if "classifier" in model_type.value else "regression"
            model_params = {
                "hidden_size": params.get("hidden_size", 128),
                "num_layers": params.get("num_layers", 2),
                "dropout": params.get("dropout", 0.2),
                "bidirectional": params.get("bidirectional", True),
            }
            return NeuralNetWrapper(
                model_class=GRUModel,
                model_params=model_params,
                task_type=task_type,
                learning_rate=params.get("learning_rate", 1e-3),
                batch_size=params.get("batch_size", 64),
                epochs=params.get("epochs", 100),
                early_stopping_patience=params.get("early_stopping_patience", 10),
                sequence_length=params.get("sequence_length", 20),
                random_state=params.get("random_state", 42),
            )

        elif model_type in [ModelType.TRANSFORMER_CLASSIFIER, ModelType.TRANSFORMER_REGRESSOR]:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available for Transformer models")
            task_type = "classification" if "classifier" in model_type.value else "regression"
            model_params = {
                "d_model": params.get("d_model", 128),
                "nhead": params.get("nhead", 8),
                "num_layers": params.get("num_layers", 4),
                "dim_feedforward": params.get("dim_feedforward", 512),
                "dropout": params.get("dropout", 0.1),
            }
            return NeuralNetWrapper(
                model_class=TransformerModel,
                model_params=model_params,
                task_type=task_type,
                learning_rate=params.get("learning_rate", 1e-4),
                batch_size=params.get("batch_size", 32),
                epochs=params.get("epochs", 100),
                early_stopping_patience=params.get("early_stopping_patience", 15),
                sequence_length=params.get("sequence_length", 50),
                random_state=params.get("random_state", 42),
            )

        # Ensemble Models (require base_estimators to be passed in params)
        elif model_type == ModelType.STACKING_CLASSIFIER:
            if not SKLEARN_ENSEMBLE_AVAILABLE:
                raise ImportError("sklearn ensemble not available")
            base_estimators = params.pop("estimators", None)
            final_estimator = params.pop("final_estimator", None)
            if base_estimators is None:
                raise ValueError("Stacking requires 'estimators' parameter")
            return StackingClassifier(
                estimators=base_estimators,
                final_estimator=final_estimator,
                **params
            )

        elif model_type == ModelType.STACKING_REGRESSOR:
            if not SKLEARN_ENSEMBLE_AVAILABLE:
                raise ImportError("sklearn ensemble not available")
            base_estimators = params.pop("estimators", None)
            final_estimator = params.pop("final_estimator", None)
            if base_estimators is None:
                raise ValueError("Stacking requires 'estimators' parameter")
            return StackingRegressor(
                estimators=base_estimators,
                final_estimator=final_estimator,
                **params
            )

        elif model_type == ModelType.VOTING_CLASSIFIER:
            if not SKLEARN_ENSEMBLE_AVAILABLE:
                raise ImportError("sklearn ensemble not available")
            base_estimators = params.pop("estimators", None)
            if base_estimators is None:
                raise ValueError("Voting requires 'estimators' parameter")
            return VotingClassifier(
                estimators=base_estimators,
                voting=params.pop("voting", "soft"),
                **params
            )

        elif model_type == ModelType.VOTING_REGRESSOR:
            if not SKLEARN_ENSEMBLE_AVAILABLE:
                raise ImportError("sklearn ensemble not available")
            base_estimators = params.pop("estimators", None)
            if base_estimators is None:
                raise ValueError("Voting requires 'estimators' parameter")
            return VotingRegressor(
                estimators=base_estimators,
                **params
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_default_params(model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """
        Get default hyperparameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of default parameters
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return DEFAULT_PARAMS.get(model_type, {}).copy()

    @staticmethod
    def get_param_space(
        model_type: Union[ModelType, str],
    ) -> List[ParamSpace]:
        """
        Get parameter space for hyperparameter optimization.

        Args:
            model_type: Type of model

        Returns:
            List of ParamSpace definitions
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return DEFAULT_PARAM_SPACES.get(model_type, [])

    @staticmethod
    def create_model_from_trial(
        model_type: Union[ModelType, str],
        trial: "optuna.Trial",
        fixed_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        gpu_enabled: bool = False,
    ) -> Any:
        """
        Create a model with parameters sampled from Optuna trial.

        Args:
            model_type: Type of model
            trial: Optuna trial object
            fixed_params: Parameters that should not be optimized
            random_state: Random seed
            gpu_enabled: Enable GPU acceleration

        Returns:
            Model with sampled parameters
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for trial-based model creation")

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        # Get param space
        param_spaces = ModelFactory.get_param_space(model_type)

        # Sample parameters
        sampled_params = {}
        for space in param_spaces:
            sampled_params[space.name] = space.sample(trial)

        # Merge with fixed params
        if fixed_params:
            sampled_params.update(fixed_params)

        return ModelFactory.create_model(
            model_type,
            params=sampled_params,
            random_state=random_state,
            gpu_enabled=gpu_enabled,
        )

    @staticmethod
    def get_task_type(model_type: Union[ModelType, str]) -> TaskType:
        """
        Get the task type (classification/regression) for a model type.

        Args:
            model_type: Type of model

        Returns:
            TaskType enum value
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        classification_types = [
            ModelType.LIGHTGBM_CLASSIFIER,
            ModelType.XGBOOST_CLASSIFIER,
            ModelType.CATBOOST_CLASSIFIER,
            ModelType.RANDOM_FOREST_CLASSIFIER,
            ModelType.LOGISTIC_REGRESSION,
            ModelType.LSTM_CLASSIFIER,
            ModelType.GRU_CLASSIFIER,
            ModelType.TRANSFORMER_CLASSIFIER,
            ModelType.STACKING_CLASSIFIER,
            ModelType.VOTING_CLASSIFIER,
        ]

        if model_type in classification_types:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    @staticmethod
    def save_model(
        model: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a model to disk.

        Args:
            model: Model to save
            path: Path to save to
            metadata: Optional metadata to save alongside model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine save method based on model type
        model_type = type(model).__name__

        if "LGBM" in model_type:
            model.booster_.save_model(str(path.with_suffix(".txt")))
        elif "XGB" in model_type:
            model.save_model(str(path.with_suffix(".json")))
        elif "CatBoost" in model_type:
            model.save_model(str(path.with_suffix(".cbm")))
        else:
            # Default: use joblib
            joblib.dump(model, str(path.with_suffix(".joblib")))

        # Save metadata if provided
        if metadata:
            import json
            meta_path = path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    @staticmethod
    def load_model(
        path: Union[str, Path],
        model_type: Optional[Union[ModelType, str]] = None,
    ) -> Any:
        """
        Load a model from disk.

        Args:
            path: Path to load from
            model_type: Type of model (optional, inferred from extension if not provided)

        Returns:
            Loaded model
        """
        path = Path(path)

        if not path.exists():
            # Try common extensions
            for ext in [".joblib", ".txt", ".json", ".cbm"]:
                if path.with_suffix(ext).exists():
                    path = path.with_suffix(ext)
                    break

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".txt" and LIGHTGBM_AVAILABLE:
            # LightGBM model
            return lgb.Booster(model_file=str(path))

        elif suffix == ".json" and XGBOOST_AVAILABLE:
            # XGBoost model
            model = xgb.Booster()
            model.load_model(str(path))
            return model

        elif suffix == ".cbm" and CATBOOST_AVAILABLE:
            # CatBoost model
            return cb.CatBoost().load_model(str(path))

        else:
            # Default: joblib
            return joblib.load(str(path))

    @staticmethod
    def get_feature_importance(
        model: Any,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Extract feature importance from a model.

        Args:
            model: Trained model
            feature_names: Feature names (optional)
            importance_type: Type of importance (gain, weight, cover)

        Returns:
            DataFrame with feature importance
        """
        model_type = type(model).__name__

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "get_score"):
            # XGBoost Booster
            score = model.get_score(importance_type=importance_type)
            if feature_names:
                importance = [score.get(f, 0) for f in feature_names]
            else:
                feature_names = list(score.keys())
                importance = list(score.values())
        elif hasattr(model, "feature_importance"):
            # LightGBM Booster
            importance = model.feature_importance(importance_type=importance_type)
        else:
            raise ValueError(f"Cannot extract feature importance from {model_type}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        # Normalize
        df["importance_normalized"] = df["importance"] / df["importance"].sum()

        return df

    @staticmethod
    def detect_gpu() -> bool:
        """Detect if GPU is available for training."""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass

        # Try XGBoost GPU detection
        if XGBOOST_AVAILABLE:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5
                )
                return result.returncode == 0
            except Exception:
                pass

        return False

    @staticmethod
    def list_available_models() -> List[ModelType]:
        """List all available model types based on installed packages."""
        available = []

        if LIGHTGBM_AVAILABLE:
            available.extend([
                ModelType.LIGHTGBM_CLASSIFIER,
                ModelType.LIGHTGBM_REGRESSOR,
            ])

        if XGBOOST_AVAILABLE:
            available.extend([
                ModelType.XGBOOST_CLASSIFIER,
                ModelType.XGBOOST_REGRESSOR,
            ])

        if CATBOOST_AVAILABLE:
            available.extend([
                ModelType.CATBOOST_CLASSIFIER,
                ModelType.CATBOOST_REGRESSOR,
            ])

        if SKLEARN_AVAILABLE:
            available.extend([
                ModelType.RANDOM_FOREST_CLASSIFIER,
                ModelType.RANDOM_FOREST_REGRESSOR,
                ModelType.RIDGE,
                ModelType.LASSO,
                ModelType.ELASTIC_NET,
                ModelType.LOGISTIC_REGRESSION,
            ])

        if PYTORCH_AVAILABLE:
            available.extend([
                ModelType.LSTM_CLASSIFIER,
                ModelType.LSTM_REGRESSOR,
                ModelType.GRU_CLASSIFIER,
                ModelType.GRU_REGRESSOR,
                ModelType.TRANSFORMER_CLASSIFIER,
                ModelType.TRANSFORMER_REGRESSOR,
            ])

        if SKLEARN_ENSEMBLE_AVAILABLE:
            available.extend([
                ModelType.STACKING_CLASSIFIER,
                ModelType.STACKING_REGRESSOR,
                ModelType.VOTING_CLASSIFIER,
                ModelType.VOTING_REGRESSOR,
            ])

        return available


# ==============================================================================
# Ensemble Factory Methods (JPMorgan Institutional-Level)
# ==============================================================================

class EnsembleFactory:
    """
    Factory for creating ensemble models with predefined configurations.

    Provides institutional-level ensemble strategies:
    - Gradient Boosting ensemble (LightGBM + XGBoost + CatBoost)
    - Diverse ensemble (mix of algorithms)
    - Stacked ensemble with meta-learner
    - Blended predictions

    Example:
        # Create a voting ensemble with default GB models
        model = EnsembleFactory.create_gb_voting_ensemble(task_type="classification")

        # Create a stacking ensemble with diverse base models
        model = EnsembleFactory.create_diverse_stacking_ensemble(
            task_type="classification",
            final_estimator="logistic_regression"
        )
    """

    @staticmethod
    def create_gb_voting_ensemble(
        task_type: str = "classification",
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> Any:
        """
        Create a voting ensemble using gradient boosting models.

        Combines LightGBM, XGBoost, and CatBoost for robust predictions.

        Args:
            task_type: "classification" or "regression"
            voting: "soft" or "hard" (classification only)
            weights: Optional weights for each model
            random_state: Random seed
            n_jobs: Number of parallel jobs

        Returns:
            VotingClassifier or VotingRegressor
        """
        if task_type == "classification":
            base_models = []

            if LIGHTGBM_AVAILABLE:
                base_models.append((
                    "lgb",
                    ModelFactory.create_model(
                        ModelType.LIGHTGBM_CLASSIFIER,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if XGBOOST_AVAILABLE:
                base_models.append((
                    "xgb",
                    ModelFactory.create_model(
                        ModelType.XGBOOST_CLASSIFIER,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if CATBOOST_AVAILABLE:
                base_models.append((
                    "cb",
                    ModelFactory.create_model(
                        ModelType.CATBOOST_CLASSIFIER,
                        random_state=random_state,
                    )
                ))

            if not base_models:
                raise ImportError("No gradient boosting libraries available")

            return VotingClassifier(
                estimators=base_models,
                voting=voting,
                weights=weights,
                n_jobs=n_jobs,
            )

        else:  # regression
            base_models = []

            if LIGHTGBM_AVAILABLE:
                base_models.append((
                    "lgb",
                    ModelFactory.create_model(
                        ModelType.LIGHTGBM_REGRESSOR,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if XGBOOST_AVAILABLE:
                base_models.append((
                    "xgb",
                    ModelFactory.create_model(
                        ModelType.XGBOOST_REGRESSOR,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if CATBOOST_AVAILABLE:
                base_models.append((
                    "cb",
                    ModelFactory.create_model(
                        ModelType.CATBOOST_REGRESSOR,
                        random_state=random_state,
                    )
                ))

            if not base_models:
                raise ImportError("No gradient boosting libraries available")

            return VotingRegressor(
                estimators=base_models,
                weights=weights,
                n_jobs=n_jobs,
            )

    @staticmethod
    def create_diverse_stacking_ensemble(
        task_type: str = "classification",
        final_estimator: str = "logistic_regression",
        include_neural: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        cv: int = 5,
    ) -> Any:
        """
        Create a stacking ensemble with diverse model types.

        Uses a variety of algorithms (tree-based, linear, optionally neural)
        to capture different patterns in the data.

        Args:
            task_type: "classification" or "regression"
            final_estimator: Type of meta-learner
            include_neural: Include neural network models if available
            random_state: Random seed
            n_jobs: Number of parallel jobs
            cv: Cross-validation folds for stacking

        Returns:
            StackingClassifier or StackingRegressor
        """
        if task_type == "classification":
            base_models = []

            # Gradient boosting models
            if LIGHTGBM_AVAILABLE:
                base_models.append((
                    "lgb",
                    ModelFactory.create_model(
                        ModelType.LIGHTGBM_CLASSIFIER,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if XGBOOST_AVAILABLE:
                base_models.append((
                    "xgb",
                    ModelFactory.create_model(
                        ModelType.XGBOOST_CLASSIFIER,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            # Random Forest
            if SKLEARN_AVAILABLE:
                base_models.append((
                    "rf",
                    ModelFactory.create_model(
                        ModelType.RANDOM_FOREST_CLASSIFIER,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            # Neural network (optional)
            if include_neural and PYTORCH_AVAILABLE:
                base_models.append((
                    "lstm",
                    ModelFactory.create_model(
                        ModelType.LSTM_CLASSIFIER,
                        random_state=random_state,
                    )
                ))

            # Create meta-learner
            if final_estimator == "logistic_regression":
                meta = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=random_state
                )
            elif final_estimator == "lightgbm" and LIGHTGBM_AVAILABLE:
                meta = ModelFactory.create_model(
                    ModelType.LIGHTGBM_CLASSIFIER,
                    params={"n_estimators": 50, "max_depth": 4},
                    random_state=random_state,
                )
            else:
                meta = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=random_state
                )

            return StackingClassifier(
                estimators=base_models,
                final_estimator=meta,
                cv=cv,
                n_jobs=n_jobs,
                passthrough=False,
            )

        else:  # regression
            base_models = []

            if LIGHTGBM_AVAILABLE:
                base_models.append((
                    "lgb",
                    ModelFactory.create_model(
                        ModelType.LIGHTGBM_REGRESSOR,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if XGBOOST_AVAILABLE:
                base_models.append((
                    "xgb",
                    ModelFactory.create_model(
                        ModelType.XGBOOST_REGRESSOR,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if SKLEARN_AVAILABLE:
                base_models.append((
                    "rf",
                    ModelFactory.create_model(
                        ModelType.RANDOM_FOREST_REGRESSOR,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                ))

            if include_neural and PYTORCH_AVAILABLE:
                base_models.append((
                    "lstm",
                    ModelFactory.create_model(
                        ModelType.LSTM_REGRESSOR,
                        random_state=random_state,
                    )
                ))

            # Meta-learner
            if final_estimator == "ridge":
                meta = Ridge(alpha=1.0, random_state=random_state)
            elif final_estimator == "lightgbm" and LIGHTGBM_AVAILABLE:
                meta = ModelFactory.create_model(
                    ModelType.LIGHTGBM_REGRESSOR,
                    params={"n_estimators": 50, "max_depth": 4},
                    random_state=random_state,
                )
            else:
                meta = Ridge(alpha=1.0, random_state=random_state)

            return StackingRegressor(
                estimators=base_models,
                final_estimator=meta,
                cv=cv,
                n_jobs=n_jobs,
                passthrough=False,
            )

    @staticmethod
    def create_weighted_blend(
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Create a weighted blend of model predictions.

        Args:
            predictions: List of prediction arrays (same shape)
            weights: Weights for each model (default: equal weights)

        Returns:
            Blended predictions
        """
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)

        if len(weights) != len(predictions):
            raise ValueError("Number of weights must match number of predictions")

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        blended = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            blended += pred * weight

        return blended

    @staticmethod
    def optimize_blend_weights(
        predictions: List[np.ndarray],
        y_true: np.ndarray,
        metric: str = "accuracy",
    ) -> Tuple[List[float], float]:
        """
        Optimize blend weights using scipy optimization.

        Args:
            predictions: List of prediction arrays
            y_true: True labels
            metric: Metric to optimize ("accuracy", "rmse", "auc")

        Returns:
            Tuple of (optimal_weights, best_score)
        """
        from scipy.optimize import minimize
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

        n_models = len(predictions)

        def objective(weights: np.ndarray) -> float:
            weights = np.abs(weights) / np.abs(weights).sum()
            blended = EnsembleFactory.create_weighted_blend(predictions, list(weights))

            if metric == "accuracy":
                blended_labels = (blended > 0.5).astype(int)
                return -accuracy_score(y_true, blended_labels)
            elif metric == "auc":
                return -roc_auc_score(y_true, blended)
            elif metric == "rmse":
                return np.sqrt(mean_squared_error(y_true, blended))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Initial weights (equal)
        x0 = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0, 1)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )

        optimal_weights = list(np.abs(result.x) / np.abs(result.x).sum())
        best_score = -result.fun if metric in ["accuracy", "auc"] else result.fun

        return optimal_weights, best_score

    @staticmethod
    def create_custom_ensemble(
        model_configs: List[Dict[str, Any]],
        ensemble_type: str = "voting",
        task_type: str = "classification",
        final_estimator: Optional[Any] = None,
        voting: str = "soft",
        n_jobs: int = -1,
    ) -> Any:
        """
        Create a custom ensemble from model configurations.

        Args:
            model_configs: List of dicts with keys:
                - "name": Estimator name
                - "model_type": ModelType enum value
                - "params": Optional dict of parameters
            ensemble_type: "voting" or "stacking"
            task_type: "classification" or "regression"
            final_estimator: Meta-learner for stacking (optional)
            voting: Voting method for VotingClassifier
            n_jobs: Number of parallel jobs

        Returns:
            Ensemble model

        Example:
            configs = [
                {"name": "lgb", "model_type": ModelType.LIGHTGBM_CLASSIFIER,
                 "params": {"n_estimators": 200}},
                {"name": "xgb", "model_type": ModelType.XGBOOST_CLASSIFIER,
                 "params": {"max_depth": 8}},
            ]
            model = EnsembleFactory.create_custom_ensemble(
                configs,
                ensemble_type="voting",
                task_type="classification"
            )
        """
        estimators = []

        for config in model_configs:
            name = config["name"]
            model_type = config["model_type"]
            params = config.get("params", {})

            model = ModelFactory.create_model(model_type, params=params)
            estimators.append((name, model))

        if ensemble_type == "voting":
            if task_type == "classification":
                return VotingClassifier(
                    estimators=estimators,
                    voting=voting,
                    n_jobs=n_jobs,
                )
            else:
                return VotingRegressor(
                    estimators=estimators,
                    n_jobs=n_jobs,
                )

        elif ensemble_type == "stacking":
            if final_estimator is None:
                if task_type == "classification":
                    final_estimator = LogisticRegression(max_iter=1000)
                else:
                    final_estimator = Ridge()

            if task_type == "classification":
                return StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    n_jobs=n_jobs,
                )
            else:
                return StackingRegressor(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    n_jobs=n_jobs,
                )

        else:
            raise ValueError(f"Unknown ensemble_type: {ensemble_type}")
