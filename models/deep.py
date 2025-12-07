"""
Deep Learning Models Module
===========================

Production-grade deep learning models for financial time series prediction.
Implements JPMorgan-level architectures with proper regularization.

Models:
- LSTMModel: Long Short-Term Memory networks
- TransformerModel: Attention-based transformer
- TCNModel: Temporal Convolutional Network
- AttentionLSTM: LSTM with self-attention

Features:
- Time series aware architecture
- Proper sequence handling
- GPU acceleration (if available)
- Early stopping and checkpointing
- Learning rate scheduling

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
import math

import numpy as np
from numpy.typing import NDArray

from config.settings import get_logger
from models.base import (
    BaseModel,
    ModelConfig,
    ModelType,
    ModelRegistry,
)

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeepLearningConfig(ModelConfig):
    """Base configuration for deep learning models."""
    model_type: ModelType = ModelType.DEEP_LEARNING
    
    # Input configuration
    sequence_length: int = 60  # Lookback window
    n_features: int = 0  # Set during training
    prediction_type: str = "classification"  # classification, regression
    num_classes: int = 3  # For classification
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1
    
    # Device
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_path: str | None = None


@dataclass
class LSTMConfig(DeepLearningConfig):
    """Configuration for LSTM model."""
    name: str = "LSTMModel"
    
    # Architecture
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    
    # Additional layers
    use_attention: bool = True
    attention_heads: int = 4
    fc_hidden: int = 64


@dataclass
class TransformerConfig(DeepLearningConfig):
    """Configuration for Transformer model."""
    name: str = "TransformerModel"
    
    # Architecture
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    d_ff: int = 512
    
    # Positional encoding
    max_seq_length: int = 512
    use_learned_pos: bool = False


@dataclass
class TCNConfig(DeepLearningConfig):
    """Configuration for TCN model."""
    name: str = "TCNModel"
    
    # Architecture
    num_channels: list[int] = field(default_factory=lambda: [64, 128, 128, 64])
    kernel_size: int = 3
    dilation_base: int = 2


# =============================================================================
# PYTORCH MODELS
# =============================================================================

def get_device(device_str: str) -> "torch.device":
    """Get the appropriate device."""
    import torch
    
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class LSTMNetwork:
    """
    LSTM Network with optional attention.
    
    Architecture:
        Input -> LSTM layers -> (Attention) -> FC layers -> Output
    """
    
    def __init__(self, config: LSTMConfig):
        import torch
        import torch.nn as nn
        
        self.config = config
        
        class LSTMModule(nn.Module):
            def __init__(self, cfg: LSTMConfig):
                super().__init__()
                self.cfg = cfg
                
                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=cfg.n_features,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0,
                    bidirectional=cfg.bidirectional,
                )
                
                lstm_out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
                
                # Attention layer (optional)
                self.use_attention = cfg.use_attention
                if cfg.use_attention:
                    self.attention = nn.MultiheadAttention(
                        embed_dim=lstm_out_size,
                        num_heads=cfg.attention_heads,
                        dropout=cfg.dropout,
                        batch_first=True,
                    )
                    self.attention_norm = nn.LayerNorm(lstm_out_size)
                
                # Output layers
                self.fc = nn.Sequential(
                    nn.Linear(lstm_out_size, cfg.fc_hidden),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.fc_hidden, cfg.num_classes if cfg.prediction_type == "classification" else 1),
                )
                
                self._init_weights()
            
            def _init_weights(self):
                for name, param in self.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            
            def forward(self, x):
                # LSTM forward
                lstm_out, (h_n, c_n) = self.lstm(x)
                
                if self.use_attention:
                    # Self-attention on LSTM output
                    attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    lstm_out = self.attention_norm(lstm_out + attn_out)
                
                # Use last timestep
                out = lstm_out[:, -1, :]
                
                # Fully connected
                out = self.fc(out)
                
                return out
        
        self.model = LSTMModule(config)
    
    def get_model(self):
        return self.model


class TransformerNetwork:
    """
    Transformer Network for time series.
    
    Architecture:
        Input -> Positional Encoding -> Transformer Encoder -> Pooling -> FC -> Output
    """
    
    def __init__(self, config: TransformerConfig):
        import torch
        import torch.nn as nn
        
        self.config = config
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(1, max_len, d_model)
                pe[0, :, 0::2] = torch.sin(position * div_term)
                pe[0, :, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)
        
        class TransformerModule(nn.Module):
            def __init__(self, cfg: TransformerConfig):
                super().__init__()
                self.cfg = cfg
                
                # Input projection
                self.input_projection = nn.Linear(cfg.n_features, cfg.d_model)
                
                # Positional encoding
                self.pos_encoder = PositionalEncoding(
                    cfg.d_model,
                    cfg.max_seq_length,
                    cfg.dropout,
                )
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_ff,
                    dropout=cfg.dropout,
                    batch_first=True,
                    norm_first=True,  # Pre-LN for stability
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=cfg.n_encoder_layers,
                )
                
                # Output head
                self.output_head = nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model // 2, cfg.num_classes if cfg.prediction_type == "classification" else 1),
                )
                
                self._init_weights()
            
            def _init_weights(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            
            def forward(self, x, mask=None):
                # Project input
                x = self.input_projection(x)
                
                # Add positional encoding
                x = self.pos_encoder(x)
                
                # Transformer encoder
                x = self.transformer_encoder(x, mask=mask)
                
                # Global average pooling or use [CLS] token position
                x = x.mean(dim=1)
                
                # Output
                return self.output_head(x)
        
        self.model = TransformerModule(config)
    
    def get_model(self):
        return self.model


class TCNNetwork:
    """
    Temporal Convolutional Network.
    
    Architecture:
        Input -> [Dilated Conv -> Norm -> ReLU -> Dropout] * N -> FC -> Output
    """
    
    def __init__(self, config: TCNConfig):
        import torch
        import torch.nn as nn
        
        self.config = config
        
        class CausalConv1d(nn.Module):
            """Causal convolution with proper padding."""
            def __init__(self, in_channels, out_channels, kernel_size, dilation):
                super().__init__()
                self.padding = (kernel_size - 1) * dilation
                self.conv = nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=self.padding, dilation=dilation
                )
            
            def forward(self, x):
                out = self.conv(x)
                return out[:, :, :-self.padding] if self.padding > 0 else out
        
        class TemporalBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
                super().__init__()
                
                self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                
                self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
                self.bn2 = nn.BatchNorm1d(out_channels)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)
                
                # Residual connection
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.relu = nn.ReLU()
            
            def forward(self, x):
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu1(out)
                out = self.dropout1(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu2(out)
                out = self.dropout2(out)
                
                res = x if self.downsample is None else self.downsample(x)
                return self.relu(out + res)
        
        class TCNModule(nn.Module):
            def __init__(self, cfg: TCNConfig):
                super().__init__()
                self.cfg = cfg
                
                layers = []
                num_channels = cfg.num_channels
                
                for i in range(len(num_channels)):
                    in_channels = cfg.n_features if i == 0 else num_channels[i - 1]
                    out_channels = num_channels[i]
                    dilation = cfg.dilation_base ** i
                    
                    layers.append(TemporalBlock(
                        in_channels, out_channels,
                        cfg.kernel_size, dilation, cfg.dropout
                    ))
                
                self.network = nn.Sequential(*layers)
                
                # Output head
                self.output_head = nn.Sequential(
                    nn.Linear(num_channels[-1], num_channels[-1] // 2),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(num_channels[-1] // 2, cfg.num_classes if cfg.prediction_type == "classification" else 1),
                )
            
            def forward(self, x):
                # x shape: (batch, seq_len, features)
                # TCN expects: (batch, features, seq_len)
                x = x.transpose(1, 2)
                
                # Apply TCN
                x = self.network(x)
                
                # Use last timestep
                x = x[:, :, -1]
                
                # Output
                return self.output_head(x)
        
        self.model = TCNModule(config)
    
    def get_model(self):
        return self.model


# =============================================================================
# DEEP LEARNING MODEL WRAPPER
# =============================================================================

class DeepLearningModel(BaseModel[DeepLearningConfig]):
    """
    Base wrapper for PyTorch deep learning models.
    
    Handles:
    - Training loop with early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Checkpointing
    """
    
    def __init__(
        self,
        config: DeepLearningConfig,
        network_builder: Callable[[DeepLearningConfig], Any] | None = None,
    ):
        super().__init__(config)
        self._network_builder = network_builder
        self._network = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None  # For mixed precision
        self._best_val_loss = float("inf")
        
        # Set device
        self._device = get_device(config.device)
        logger.info(f"Using device: {self._device}")
    
    def _default_config(self) -> DeepLearningConfig:
        return DeepLearningConfig()
    
    def _build_model(self) -> Any:
        """Build the neural network."""
        import torch
        
        if self._network_builder is not None:
            self.config.n_features = self._n_features
            network = self._network_builder(self.config)
            self._network = network.get_model().to(self._device)
        else:
            raise ValueError("No network builder provided")
        
        # Initialize optimizer
        self._optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr_min,
            )
        elif self.config.lr_scheduler == "plateau":
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=self.config.lr_min,
            )
        
        # Mixed precision scaler
        if self.config.mixed_precision and self._device.type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()
        
        return self._network
    
    def _fit_impl(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Train the deep learning model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Reshape data for sequences
        X_seq = self._prepare_sequences(X)
        y_seq = y[self.config.sequence_length - 1:]
        
        if X_val is not None and y_val is not None:
            X_val_seq = self._prepare_sequences(X_val)
            y_val_seq = y_val[self.config.sequence_length - 1:]
        else:
            # Split training data
            split_idx = int(len(X_seq) * 0.9)
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            X_seq = X_seq[:split_idx]
            y_seq = y_seq[:split_idx]
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.LongTensor(y_seq) if self.config.prediction_type == "classification" else torch.FloatTensor(y_seq),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.LongTensor(y_val_seq) if self.config.prediction_type == "classification" else torch.FloatTensor(y_val_seq),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        # Loss function
        if self.config.prediction_type == "classification":
            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        self._network.train()
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(train_loader, criterion)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            if self.config.lr_scheduler == "plateau":
                self._scheduler.step(val_loss)
            else:
                self._scheduler.step()
            
            # Early stopping
            if val_loss < self._best_val_loss - self.config.min_delta:
                self._best_val_loss = val_loss
                patience_counter = 0
                self._best_iteration = epoch
                
                # Save best model
                if self.config.save_best_only and self.config.checkpoint_path:
                    torch.save(self._network.state_dict(), self.config.checkpoint_path)
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}"
                )
            
            self._train_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.config.save_best_only and self.config.checkpoint_path:
            self._network.load_state_dict(torch.load(self.config.checkpoint_path))
    
    def _train_epoch(
        self,
        train_loader,
        criterion,
    ) -> float:
        """Train for one epoch."""
        import torch
        
        self._network.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self._device)
            batch_y = batch_y.to(self._device)
            
            self._optimizer.zero_grad()
            
            # Mixed precision training
            if self._scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self._network(batch_x)
                    if self.config.prediction_type == "regression":
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                
                self._scaler.scale(loss).backward()
                
                # Gradient clipping
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                outputs = self._network(batch_x)
                if self.config.prediction_type == "regression":
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                self._optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(
        self,
        val_loader,
        criterion,
    ) -> tuple[float, float]:
        """Validate for one epoch."""
        import torch
        
        self._network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                
                outputs = self._network(batch_x)
                if self.config.prediction_type == "regression":
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                if self.config.prediction_type == "classification":
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        
        return val_loss, val_acc
    
    def _predict_impl(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict with the deep learning model."""
        import torch
        
        self._network.eval()
        
        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        
        # Predict in batches
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_seq), self.config.batch_size):
                batch = torch.FloatTensor(X_seq[i:i + self.config.batch_size]).to(self._device)
                outputs = self._network(batch)
                
                if self.config.prediction_type == "classification":
                    _, preds = outputs.max(1)
                    predictions.extend(preds.cpu().numpy())
                else:
                    predictions.extend(outputs.squeeze().cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Get class probabilities."""
        import torch
        import torch.nn.functional as F
        
        if self.config.prediction_type != "classification":
            return None
        
        self._network.eval()
        X_seq = self._prepare_sequences(X)
        
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X_seq), self.config.batch_size):
                batch = torch.FloatTensor(X_seq[i:i + self.config.batch_size]).to(self._device)
                outputs = self._network(batch)
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def _prepare_sequences(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepare sequential data for time series models."""
        n_samples = len(X) - self.config.sequence_length + 1
        n_features = X.shape[1]
        
        sequences = np.zeros((n_samples, self.config.sequence_length, n_features))
        
        for i in range(n_samples):
            sequences[i] = X[i:i + self.config.sequence_length]
        
        return sequences


# =============================================================================
# LSTM MODEL
# =============================================================================

class LSTMModel(DeepLearningModel):
    """
    LSTM Model for trading signal prediction.
    
    Optimal for:
    - Sequential patterns
    - Long-range dependencies
    - Market regime detection
    
    Example:
        config = LSTMConfig(
            sequence_length=60,
            hidden_size=128,
            num_layers=2,
        )
        model = LSTMModel(config)
        model.fit(X_train, y_train)
    """
    
    def __init__(self, config: LSTMConfig | None = None):
        config = config or LSTMConfig()
        super().__init__(config, lambda cfg: LSTMNetwork(cfg))
        self.config: LSTMConfig = config
    
    def _default_config(self) -> LSTMConfig:
        return LSTMConfig()


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

class TransformerModel(DeepLearningModel):
    """
    Transformer Model for trading signal prediction.
    
    Optimal for:
    - Attention to important events
    - Multi-scale patterns
    - Parallel processing
    
    Example:
        config = TransformerConfig(
            sequence_length=60,
            d_model=128,
            n_heads=8,
        )
        model = TransformerModel(config)
        model.fit(X_train, y_train)
    """
    
    def __init__(self, config: TransformerConfig | None = None):
        config = config or TransformerConfig()
        super().__init__(config, lambda cfg: TransformerNetwork(cfg))
        self.config: TransformerConfig = config
    
    def _default_config(self) -> TransformerConfig:
        return TransformerConfig()


# =============================================================================
# TCN MODEL
# =============================================================================

class TCNModel(DeepLearningModel):
    """
    Temporal Convolutional Network for trading signal prediction.
    
    Optimal for:
    - Local patterns
    - Fast inference
    - Efficient training
    
    Example:
        config = TCNConfig(
            sequence_length=60,
            num_channels=[64, 128, 128],
        )
        model = TCNModel(config)
        model.fit(X_train, y_train)
    """
    
    def __init__(self, config: TCNConfig | None = None):
        config = config or TCNConfig()
        super().__init__(config, lambda cfg: TCNNetwork(cfg))
        self.config: TCNConfig = config
    
    def _default_config(self) -> TCNConfig:
        return TCNConfig()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_deep_model(
    model_type: str,
    **kwargs: Any,
) -> DeepLearningModel:
    """
    Factory function to create deep learning models.
    
    Args:
        model_type: Type of model (lstm, transformer, tcn)
        **kwargs: Model configuration parameters
    
    Returns:
        Configured deep learning model
    """
    model_map = {
        "lstm": (LSTMModel, LSTMConfig),
        "transformer": (TransformerModel, TransformerConfig),
        "tcn": (TCNModel, TCNConfig),
    }
    
    model_type = model_type.lower()
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    model_class, config_class = model_map[model_type]
    config = config_class(**kwargs)
    return model_class(config=config)


# =============================================================================
# REGISTER MODELS
# =============================================================================

ModelRegistry.register("lstm", LSTMModel, LSTMConfig)
ModelRegistry.register("transformer", TransformerModel, TransformerConfig)
ModelRegistry.register("tcn", TCNModel, TCNConfig)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configs
    "DeepLearningConfig",
    "LSTMConfig",
    "TransformerConfig",
    "TCNConfig",
    # Networks
    "LSTMNetwork",
    "TransformerNetwork",
    "TCNNetwork",
    # Models
    "DeepLearningModel",
    "LSTMModel",
    "TransformerModel",
    "TCNModel",
    # Factory
    "create_deep_model",
]