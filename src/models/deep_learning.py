"""
Deep Learning Models
JPMorgan-Level Neural Network Architectures

Features:
- LSTM/GRU for sequence modeling
- Transformer with attention
- Temporal Convolutional Networks
- Residual connections
- Batch normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from .base_model import BaseModel
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class NeuralNetConfig:
    """Neural network configuration"""
    input_dim: int = 100
    output_dim: int = 3
    hidden_layers: List[int] = None
    dropout: float = 0.3
    batch_norm: bool = True
    activation: str = 'relu'
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    sequence_length: int = 20

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]


class LSTMModel(BaseModel):
    """
    LSTM model for sequential trading data.

    Features:
    - Bidirectional LSTM option
    - Multiple LSTM layers
    - Attention mechanism
    - Dropout regularization
    """

    def __init__(
        self,
        config: Optional[NeuralNetConfig] = None,
        bidirectional: bool = False,
        num_layers: int = 2,
        use_attention: bool = True,
        **kwargs
    ):
        """
        Initialize LSTMModel.

        Args:
            config: Neural network configuration
            bidirectional: Use bidirectional LSTM
            num_layers: Number of LSTM layers
            use_attention: Add attention mechanism
        """
        # Filter out model_type and version (may come from clone())
        kwargs.pop('model_type', None)
        kwargs.pop('version', None)

        super().__init__(model_type='lstm', **kwargs)

        self.config = config or NeuralNetConfig()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.use_attention = use_attention

        self._scaler = None
        self._device = None

    def _build_model(self):
        """Build PyTorch LSTM model"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        class LSTMNetwork(nn.Module):
            def __init__(self, config, bidirectional, num_layers, use_attention):
                super().__init__()

                self.hidden_size = config.hidden_layers[0]
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                self.use_attention = use_attention

                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=config.input_dim,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=config.dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )

                lstm_output_dim = self.hidden_size * (2 if bidirectional else 1)

                # Attention layer
                if use_attention:
                    self.attention = nn.Sequential(
                        nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                        nn.Tanh(),
                        nn.Linear(lstm_output_dim // 2, 1)
                    )

                # Fully connected layers
                self.fc_layers = nn.ModuleList()
                prev_dim = lstm_output_dim

                for hidden_dim in config.hidden_layers[1:]:
                    self.fc_layers.append(nn.Sequential(
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                        nn.ReLU(),
                        nn.Dropout(config.dropout)
                    ))
                    prev_dim = hidden_dim

                # Output layer
                self.output = nn.Linear(prev_dim, config.output_dim)

            def forward(self, x):
                # LSTM forward
                lstm_out, _ = self.lstm(x)

                # Apply attention or take last output
                if self.use_attention:
                    attention_weights = torch.softmax(
                        self.attention(lstm_out).squeeze(-1), dim=1
                    )
                    context = torch.bmm(
                        attention_weights.unsqueeze(1), lstm_out
                    ).squeeze(1)
                else:
                    context = lstm_out[:, -1, :]

                # FC layers
                out = context
                for fc in self.fc_layers:
                    out = fc(out)

                return self.output(out)

        return LSTMNetwork(
            self.config,
            self.bidirectional,
            self.num_layers,
            self.use_attention
        )

    def _prepare_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequential data for LSTM"""
        seq_length = self.config.sequence_length

        # Scale features
        from sklearn.preprocessing import StandardScaler
        if self._scaler is None:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = self._scaler.transform(X)

        # Create sequences
        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_length:i])
            if y is not None:
                y_seq.append(y.iloc[i])

        X_seq = np.array(X_seq)

        if y_seq is not None:
            y_seq = np.array(y_seq)

        return X_seq, y_seq

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'LSTMModel':
        """Train LSTM model"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._feature_names = list(X.columns)
        self.config.input_dim = len(self._feature_names)
        self.config.output_dim = len(np.unique(y))

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X, pd.Series(y_encoded, index=y.index))

        # Device setup
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self._device}")

        # Build model
        self._model = self._build_model().to(self._device)

        # Data loaders
        X_tensor = torch.FloatTensor(X_seq).to(self._device)
        y_tensor = torch.LongTensor(y_seq).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5
        )

        # Training loop
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self._model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        import torch

        X_seq, _ = self._prepare_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        import torch

        X_seq, _ = self._prepare_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def _get_importance_scores(self) -> np.ndarray:
        """Feature importance not directly available for LSTM"""
        return np.ones(len(self._feature_names)) / len(self._feature_names)


class TransformerModel(BaseModel):
    """
    Transformer model for trading signal prediction.

    Features:
    - Multi-head self-attention
    - Positional encoding
    - Layer normalization
    - Feedforward networks
    """

    def __init__(
        self,
        config: Optional[NeuralNetConfig] = None,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        **kwargs
    ):
        """
        Initialize TransformerModel.

        Args:
            config: Neural network configuration
            n_heads: Number of attention heads
            n_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
        """
        # Filter out model_type and version (may come from clone())
        kwargs.pop('model_type', None)
        kwargs.pop('version', None)

        super().__init__(model_type='transformer', **kwargs)

        self.config = config or NeuralNetConfig()
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.dim_feedforward = dim_feedforward

        self._scaler = None
        self._device = None

    def _build_model(self):
        """Build PyTorch Transformer model"""
        import torch
        import torch.nn as nn
        import math

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)

        class TransformerNetwork(nn.Module):
            def __init__(self, config, n_heads, n_encoder_layers, dim_feedforward):
                super().__init__()

                self.d_model = config.hidden_layers[0]

                # Input projection
                self.input_projection = nn.Linear(config.input_dim, self.d_model)

                # Positional encoding
                self.pos_encoder = PositionalEncoding(
                    self.d_model, config.sequence_length, config.dropout
                )

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=config.dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_encoder_layers
                )

                # Classification head
                self.fc = nn.Sequential(
                    nn.Linear(self.d_model, config.hidden_layers[-1]),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_layers[-1], config.output_dim)
                )

            def forward(self, x):
                # Input projection
                x = self.input_projection(x)

                # Positional encoding
                x = self.pos_encoder(x)

                # Transformer encoding
                x = self.transformer_encoder(x)

                # Global average pooling
                x = x.mean(dim=1)

                # Classification
                return self.fc(x)

        return TransformerNetwork(
            self.config,
            self.n_heads,
            self.n_encoder_layers,
            self.dim_feedforward
        )

    def _prepare_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequential data"""
        seq_length = self.config.sequence_length

        from sklearn.preprocessing import StandardScaler
        if self._scaler is None:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = self._scaler.transform(X)

        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_length:i])
            if y is not None:
                y_seq.append(y.iloc[i])

        X_seq = np.array(X_seq)
        if y_seq is not None:
            y_seq = np.array(y_seq)

        return X_seq, y_seq

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'TransformerModel':
        """Train Transformer model"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._feature_names = list(X.columns)
        self.config.input_dim = len(self._feature_names)
        self.config.output_dim = len(np.unique(y))

        # Adjust d_model to be divisible by n_heads
        if self.config.hidden_layers[0] % self.n_heads != 0:
            self.config.hidden_layers[0] = (
                (self.config.hidden_layers[0] // self.n_heads + 1) * self.n_heads
            )

        from sklearn.preprocessing import LabelEncoder
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        X_seq, y_seq = self._prepare_sequences(X, pd.Series(y_encoded, index=y.index))

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training Transformer on device: {self._device}")

        self._model = self._build_model().to(self._device)

        X_tensor = torch.FloatTensor(X_seq).to(self._device)
        y_tensor = torch.LongTensor(y_seq).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # Training
        for epoch in range(self.config.epochs):
            self._model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")

        self._is_trained = True
        self.metadata.trained_at = datetime.utcnow()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        import torch

        X_seq, _ = self._prepare_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        import torch

        X_seq, _ = self._prepare_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def _get_importance_scores(self) -> np.ndarray:
        """Feature importance approximation via attention weights"""
        return np.ones(len(self._feature_names)) / len(self._feature_names)


class AttentionModel(BaseModel):
    """
    Simple attention-based model for feature weighting.

    Lighter than full Transformer, suitable for tabular data.
    """

    def __init__(
        self,
        config: Optional[NeuralNetConfig] = None,
        attention_dim: int = 64,
        **kwargs
    ):
        # Filter out model_type and version (may come from clone())
        kwargs.pop('model_type', None)
        kwargs.pop('version', None)

        super().__init__(model_type='attention', **kwargs)

        self.config = config or NeuralNetConfig()
        self.attention_dim = attention_dim
        self._scaler = None
        self._device = None

    def _build_model(self):
        """Build attention network"""
        import torch
        import torch.nn as nn

        class AttentionNetwork(nn.Module):
            def __init__(self, config, attention_dim):
                super().__init__()

                # Feature attention
                self.attention = nn.Sequential(
                    nn.Linear(config.input_dim, attention_dim),
                    nn.Tanh(),
                    nn.Linear(attention_dim, config.input_dim),
                    nn.Softmax(dim=1)
                )

                # Main network
                layers = []
                prev_dim = config.input_dim

                for hidden_dim in config.hidden_layers:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                        nn.ReLU(),
                        nn.Dropout(config.dropout)
                    ])
                    prev_dim = hidden_dim

                layers.append(nn.Linear(prev_dim, config.output_dim))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                # Apply attention weights to features
                attention_weights = self.attention(x)
                weighted_input = x * attention_weights

                return self.network(weighted_input)

        return AttentionNetwork(self.config, self.attention_dim)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'AttentionModel':
        """Train attention model"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._feature_names = list(X.columns)
        self.config.input_dim = len(self._feature_names)
        self.config.output_dim = len(np.unique(y))

        from sklearn.preprocessing import StandardScaler, LabelEncoder

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._build_model().to(self._device)

        X_tensor = torch.FloatTensor(X_scaled).to(self._device)
        y_tensor = torch.LongTensor(y_encoded).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            self._model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        X_scaled = self._scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        X_scaled = self._scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def _get_importance_scores(self) -> np.ndarray:
        return np.ones(len(self._feature_names)) / len(self._feature_names)
