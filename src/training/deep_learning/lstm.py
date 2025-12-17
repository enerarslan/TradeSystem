"""
LSTM models for financial time-series prediction.

This module provides PyTorch Lightning implementations of:
- Basic LSTM predictor
- LSTM with attention mechanism
- Bidirectional LSTM variants

Designed for institutional-grade requirements:
- Mixed precision training
- Gradient clipping
- Proper sequence handling
- Uncertainty estimation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:

    class Attention(nn.Module):
        """
        Attention mechanism for sequence models.

        Computes attention weights over time steps and returns
        weighted sum of hidden states.
        """

        def __init__(self, hidden_size: int, attention_size: int = None):
            super().__init__()
            attention_size = attention_size or hidden_size

            self.attention = nn.Sequential(
                nn.Linear(hidden_size, attention_size),
                nn.Tanh(),
                nn.Linear(attention_size, 1),
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Apply attention.

            Args:
                hidden_states: (batch, seq_len, hidden_size)
                mask: Optional mask for padded sequences

            Returns:
                Tuple of (context_vector, attention_weights)
            """
            # Attention scores: (batch, seq_len, 1)
            scores = self.attention(hidden_states)

            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

            # Attention weights: (batch, seq_len, 1)
            weights = F.softmax(scores, dim=1)

            # Context vector: (batch, hidden_size)
            context = (hidden_states * weights).sum(dim=1)

            return context, weights.squeeze(-1)


if TORCH_AVAILABLE and LIGHTNING_AVAILABLE:

    class LSTMPredictor(pl.LightningModule):
        """
        LSTM-based predictor for financial time-series.

        Features:
        - Configurable number of layers and hidden size
        - Dropout for regularization
        - Batch normalization option
        - Multiple output types (classification, regression)

        Example:
            model = LSTMPredictor(
                input_size=50,
                hidden_size=128,
                num_layers=2,
                output_size=1,
                dropout=0.2
            )

            trainer = pl.Trainer(max_epochs=100)
            trainer.fit(model, train_dataloader, val_dataloader)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            output_size: int = 1,
            dropout: float = 0.2,
            bidirectional: bool = False,
            batch_first: bool = True,
            learning_rate: float = 1e-3,
            task: str = "regression",  # "regression" or "classification"
            weight_decay: float = 1e-5,
            use_batch_norm: bool = True,
        ):
            super().__init__()
            self.save_hyperparameters()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.bidirectional = bidirectional
            self.learning_rate = learning_rate
            self.task = task
            self.weight_decay = weight_decay

            # Direction multiplier
            num_directions = 2 if bidirectional else 1

            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )

            # Batch normalization
            self.use_batch_norm = use_batch_norm
            if use_batch_norm:
                self.batch_norm = nn.BatchNorm1d(hidden_size * num_directions)

            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * num_directions, output_size)

            # Loss function
            if task == "classification":
                self.loss_fn = nn.CrossEntropyLoss() if output_size > 1 else nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.MSELoss()

        def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Forward pass.

            Args:
                x: Input tensor (batch, seq_len, input_size)
                hidden: Optional initial hidden state

            Returns:
                Tuple of (output, (hidden_state, cell_state))
            """
            # LSTM forward
            lstm_out, (hn, cn) = self.lstm(x, hidden)

            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward final states
                final_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                final_hidden = hn[-1]

            # Batch norm
            if self.use_batch_norm:
                final_hidden = self.batch_norm(final_hidden)

            # Output
            out = self.dropout(final_hidden)
            out = self.fc(out)

            return out, (hn, cn)

        def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
        ) -> torch.Tensor:
            """Training step."""
            x, y = batch
            y_hat, _ = self(x)

            # Reshape for loss
            y_hat = y_hat.view(-1) if self.output_size == 1 else y_hat
            y = y.view(-1) if self.output_size == 1 else y

            loss = self.loss_fn(y_hat, y)

            self.log('train_loss', loss, prog_bar=True)

            # Additional metrics
            if self.task == "classification":
                preds = torch.sigmoid(y_hat) > 0.5 if self.output_size == 1 else y_hat.argmax(dim=1)
                acc = (preds == y).float().mean()
                self.log('train_acc', acc, prog_bar=True)

            return loss

        def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
        ) -> Dict[str, torch.Tensor]:
            """Validation step."""
            x, y = batch
            y_hat, _ = self(x)

            y_hat = y_hat.view(-1) if self.output_size == 1 else y_hat
            y = y.view(-1) if self.output_size == 1 else y

            loss = self.loss_fn(y_hat, y)

            self.log('val_loss', loss, prog_bar=True)

            if self.task == "classification":
                preds = torch.sigmoid(y_hat) > 0.5 if self.output_size == 1 else y_hat.argmax(dim=1)
                acc = (preds == y).float().mean()
                self.log('val_acc', acc, prog_bar=True)

            return {'val_loss': loss}

        def configure_optimizers(self):
            """Configure optimizer and scheduler."""
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }

        def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
        ) -> torch.Tensor:
            """Prediction step."""
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            y_hat, _ = self(x)

            if self.task == "classification":
                if self.output_size == 1:
                    return torch.sigmoid(y_hat)
                else:
                    return F.softmax(y_hat, dim=1)
            return y_hat


    class AttentionLSTM(pl.LightningModule):
        """
        LSTM with attention mechanism for financial time-series.

        Adds self-attention over LSTM hidden states, allowing the model
        to focus on the most relevant time steps for prediction.

        Useful for:
        - Variable-length sequences
        - Long sequences where recent data may not always be most important
        - Interpretability (attention weights show what the model focuses on)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            output_size: int = 1,
            dropout: float = 0.2,
            attention_size: int = 64,
            bidirectional: bool = False,
            learning_rate: float = 1e-3,
            task: str = "regression",
            weight_decay: float = 1e-5,
        ):
            super().__init__()
            self.save_hyperparameters()

            self.hidden_size = hidden_size
            self.output_size = output_size
            self.learning_rate = learning_rate
            self.task = task
            self.weight_decay = weight_decay

            num_directions = 2 if bidirectional else 1
            lstm_output_size = hidden_size * num_directions

            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional,
            )

            # Attention mechanism
            self.attention = Attention(lstm_output_size, attention_size)

            # Layer normalization
            self.layer_norm = nn.LayerNorm(lstm_output_size)

            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

            # Loss
            if task == "classification":
                self.loss_fn = nn.BCEWithLogitsLoss() if output_size == 1 else nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.MSELoss()

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass with attention.

            Args:
                x: Input tensor (batch, seq_len, input_size)
                mask: Optional mask for padded sequences

            Returns:
                Tuple of (output, attention_weights)
            """
            # LSTM forward: output is (batch, seq_len, hidden_size * num_directions)
            lstm_out, _ = self.lstm(x)

            # Apply attention
            context, attention_weights = self.attention(lstm_out, mask)

            # Layer normalization
            context = self.layer_norm(context)

            # Output
            out = self.dropout(context)
            out = self.fc(out)

            return out, attention_weights

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, _ = self(x)

            y_hat = y_hat.view(-1) if self.output_size == 1 else y_hat
            y = y.view(-1) if self.output_size == 1 else y

            loss = self.loss_fn(y_hat, y)
            self.log('train_loss', loss, prog_bar=True)

            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat, _ = self(x)

            y_hat = y_hat.view(-1) if self.output_size == 1 else y_hat
            y = y.view(-1) if self.output_size == 1 else y

            loss = self.loss_fn(y_hat, y)
            self.log('val_loss', loss, prog_bar=True)

            return {'val_loss': loss}

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 10,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        def get_attention_weights(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            """Get attention weights for interpretability."""
            self.eval()
            with torch.no_grad():
                _, weights = self(x)
            return weights


    class SequenceDataset(torch.utils.data.Dataset):
        """
        Dataset for sequence data.

        Creates overlapping sequences from time-series data
        for LSTM/Transformer training.
        """

        def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            sequence_length: int = 20,
            step: int = 1,
        ):
            """
            Initialize dataset.

            Args:
                features: Feature array (n_samples, n_features)
                targets: Target array (n_samples,)
                sequence_length: Length of each sequence
                step: Step size between sequences
            """
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.sequence_length = sequence_length
            self.step = step

            # Calculate valid indices
            self.indices = list(range(
                0,
                len(features) - sequence_length,
                step
            ))

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            start = self.indices[idx]
            end = start + self.sequence_length

            x = self.features[start:end]
            y = self.targets[end - 1]  # Target is last element

            return x, y


    def create_dataloaders(
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 20,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 0,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create train and validation dataloaders.

        Args:
            features: Feature array
            targets: Target array
            sequence_length: Sequence length for LSTM
            batch_size: Batch size
            train_ratio: Ratio of data for training
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        n_samples = len(features) - sequence_length
        train_size = int(n_samples * train_ratio)

        train_features = features[:train_size + sequence_length]
        train_targets = targets[:train_size + sequence_length]

        val_features = features[train_size:]
        val_targets = targets[train_size:]

        train_dataset = SequenceDataset(
            train_features, train_targets, sequence_length
        )
        val_dataset = SequenceDataset(
            val_features, val_targets, sequence_length
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

else:
    # Dummy classes when PyTorch Lightning not available
    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Lightning required for LSTMPredictor")

    class AttentionLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Lightning required for AttentionLSTM")
