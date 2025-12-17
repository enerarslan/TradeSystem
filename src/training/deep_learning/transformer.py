"""
Temporal Fusion Transformer for financial time-series prediction.

This module implements a simplified version of the Temporal Fusion Transformer (TFT)
architecture, optimized for financial forecasting.

Reference:
    Lim, B., et al. (2021). Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting. International Journal of Forecasting.

Key features:
- Variable selection networks for feature importance
- Interpretable multi-head attention
- Static and temporal covariate handling
- Quantile outputs for uncertainty estimation
"""

from __future__ import annotations

import logging
import math
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
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding for transformer models.

        Adds positional information to input embeddings using sine and cosine
        functions of different frequencies.
        """

        def __init__(
            self,
            d_model: int,
            max_len: int = 5000,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)


    class GatedResidualNetwork(nn.Module):
        """
        Gated Residual Network (GRN) - core building block of TFT.

        Provides non-linear processing with skip connections and gating.

        Architecture:
        1. Dense layer with ELU activation
        2. Dense layer for gating
        3. GLU activation with residual connection
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
            context_size: Optional[int] = None,
        ):
            super().__init__()

            self.input_size = input_size
            self.output_size = output_size
            self.context_size = context_size

            # Input projection if sizes differ
            if input_size != output_size:
                self.skip_layer = nn.Linear(input_size, output_size)
            else:
                self.skip_layer = None

            # Main layers
            self.fc1 = nn.Linear(input_size, hidden_size)

            if context_size is not None:
                self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
            else:
                self.context_projection = None

            self.fc2 = nn.Linear(hidden_size, output_size * 2)  # For GLU

            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_size)

        def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor
                context: Optional context tensor

            Returns:
                Output tensor with same shape as input (or output_size)
            """
            # Skip connection
            if self.skip_layer is not None:
                skip = self.skip_layer(x)
            else:
                skip = x

            # Main path
            hidden = self.fc1(x)

            # Add context if provided
            if context is not None and self.context_projection is not None:
                context_proj = self.context_projection(context)
                # Expand context to match hidden shape if needed
                if context_proj.dim() < hidden.dim():
                    context_proj = context_proj.unsqueeze(1).expand_as(hidden)
                hidden = hidden + context_proj

            hidden = F.elu(hidden)
            hidden = self.dropout(hidden)

            # GLU (Gated Linear Unit)
            hidden = self.fc2(hidden)
            hidden, gate = hidden.chunk(2, dim=-1)
            hidden = hidden * torch.sigmoid(gate)

            # Residual connection and layer norm
            output = self.layer_norm(hidden + skip)

            return output


    class VariableSelectionNetwork(nn.Module):
        """
        Variable Selection Network for automatic feature selection.

        Learns which features are most important for the prediction task.
        Provides interpretability through variable weights.
        """

        def __init__(
            self,
            input_size: int,
            num_inputs: int,
            hidden_size: int,
            dropout: float = 0.1,
            context_size: Optional[int] = None,
        ):
            super().__init__()

            self.num_inputs = num_inputs
            self.hidden_size = hidden_size

            # Flattened input processors
            self.input_grns = nn.ModuleList([
                GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_inputs)
            ])

            # Variable selection GRN
            self.selection_grn = GatedResidualNetwork(
                input_size=hidden_size * num_inputs,
                hidden_size=hidden_size,
                output_size=num_inputs,
                dropout=dropout,
                context_size=context_size,
            )

            self.softmax = nn.Softmax(dim=-1)

        def forward(
            self,
            inputs: List[torch.Tensor],
            context: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                inputs: List of input tensors, each (batch, seq_len, input_size)
                context: Optional context tensor

            Returns:
                Tuple of (selected_features, selection_weights)
            """
            # Process each input
            processed = []
            for i, (x, grn) in enumerate(zip(inputs, self.input_grns)):
                processed.append(grn(x))

            # Stack: (batch, seq_len, num_inputs, hidden_size)
            stacked = torch.stack(processed, dim=-2)

            # Flatten for selection: (batch, seq_len, num_inputs * hidden_size)
            batch_size = stacked.size(0)
            seq_len = stacked.size(1)
            flattened = stacked.view(batch_size, seq_len, -1)

            # Get selection weights
            selection_weights = self.selection_grn(flattened, context)
            selection_weights = self.softmax(selection_weights)

            # Weight and combine: (batch, seq_len, hidden_size)
            selection_weights_expanded = selection_weights.unsqueeze(-1)
            selected = (stacked * selection_weights_expanded).sum(dim=-2)

            return selected, selection_weights


    class InterpretableMultiHeadAttention(nn.Module):
        """
        Multi-head attention with interpretable attention weights.

        Similar to standard multi-head attention but designed to provide
        more interpretable attention patterns for financial time-series.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1,
        ):
            super().__init__()

            assert d_model % n_heads == 0

            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.out_linear = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                query: Query tensor (batch, seq_len, d_model)
                key: Key tensor (batch, seq_len, d_model)
                value: Value tensor (batch, seq_len, d_model)
                mask: Optional attention mask

            Returns:
                Tuple of (output, attention_weights)
            """
            batch_size = query.size(0)

            # Linear projections
            q = self.q_linear(query)
            k = self.k_linear(key)
            v = self.v_linear(value)

            # Reshape for multi-head: (batch, n_heads, seq_len, head_dim)
            q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # Attention weights
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply attention
            context = torch.matmul(attention_weights, v)

            # Reshape back
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, -1, self.d_model)

            # Output projection
            output = self.out_linear(context)

            # Average attention weights across heads for interpretability
            avg_attention = attention_weights.mean(dim=1)

            return output, avg_attention


if TORCH_AVAILABLE and LIGHTNING_AVAILABLE:

    class TemporalFusionTransformer(pl.LightningModule):
        """
        Temporal Fusion Transformer for financial time-series forecasting.

        A simplified implementation focusing on:
        - Temporal feature processing with self-attention
        - Multi-horizon forecasting
        - Quantile outputs for uncertainty estimation
        - Interpretable attention weights

        Example:
            model = TemporalFusionTransformer(
                input_size=50,
                hidden_size=128,
                num_attention_heads=4,
                num_layers=2,
                output_size=1
            )

            trainer = pl.Trainer(max_epochs=100, accelerator='auto')
            trainer.fit(model, train_dataloader, val_dataloader)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_attention_heads: int = 4,
            num_encoder_layers: int = 2,
            dropout: float = 0.1,
            output_size: int = 1,
            prediction_horizon: int = 1,
            quantiles: Optional[List[float]] = None,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            max_seq_len: int = 512,
        ):
            super().__init__()
            self.save_hyperparameters()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.prediction_horizon = prediction_horizon
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

            # Quantiles for uncertainty estimation
            self.quantiles = quantiles or [0.1, 0.5, 0.9]
            self.num_quantiles = len(self.quantiles)

            # Input projection
            self.input_projection = nn.Linear(input_size, hidden_size)

            # Positional encoding
            self.positional_encoding = PositionalEncoding(
                hidden_size, max_seq_len, dropout
            )

            # Temporal processing with GRN
            self.temporal_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )

            # Self-attention layers
            self.attention_layers = nn.ModuleList([
                InterpretableMultiHeadAttention(
                    d_model=hidden_size,
                    n_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ])

            # Post-attention GRNs
            self.post_attention_grns = nn.ModuleList([
                GatedResidualNetwork(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ])

            # Layer norms
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size)
                for _ in range(num_encoder_layers)
            ])

            # Output layers
            self.output_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )

            # Quantile outputs
            if quantiles:
                self.output_layer = nn.Linear(
                    hidden_size, output_size * self.num_quantiles
                )
            else:
                self.output_layer = nn.Linear(hidden_size, output_size)

            # Store attention weights for interpretability
            self.attention_weights = None

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor (batch, seq_len, input_size)
                mask: Optional attention mask

            Returns:
                Output predictions (batch, output_size) or
                (batch, output_size * num_quantiles) if using quantiles
            """
            # Input projection
            x = self.input_projection(x)

            # Add positional encoding
            x = self.positional_encoding(x)

            # Temporal GRN
            x = self.temporal_grn(x)

            # Self-attention layers with residual connections
            attention_weights_list = []
            for attention, grn, layer_norm in zip(
                self.attention_layers,
                self.post_attention_grns,
                self.layer_norms,
            ):
                # Self-attention
                attended, attn_weights = attention(x, x, x, mask)
                attention_weights_list.append(attn_weights)

                # Add & norm
                x = layer_norm(x + attended)

                # GRN
                x = grn(x)

            # Store attention for interpretability
            self.attention_weights = torch.stack(attention_weights_list, dim=1)

            # Use last timestep for prediction
            x = x[:, -1, :]

            # Output GRN
            x = self.output_grn(x)

            # Final output
            output = self.output_layer(x)

            return output

        def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
        ) -> torch.Tensor:
            """Training step."""
            x, y = batch
            y_hat = self(x)

            if self.quantiles:
                loss = self._quantile_loss(y_hat, y)
            else:
                loss = F.mse_loss(y_hat.view(-1), y.view(-1))

            self.log('train_loss', loss, prog_bar=True)
            return loss

        def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
        ) -> Dict[str, torch.Tensor]:
            """Validation step."""
            x, y = batch
            y_hat = self(x)

            if self.quantiles:
                loss = self._quantile_loss(y_hat, y)
            else:
                loss = F.mse_loss(y_hat.view(-1), y.view(-1))

            self.log('val_loss', loss, prog_bar=True)

            return {'val_loss': loss}

        def _quantile_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            """
            Quantile loss (pinball loss).

            Used when outputting multiple quantiles for uncertainty estimation.
            """
            targets = targets.view(-1)

            # Reshape predictions: (batch, output_size * num_quantiles)
            # -> (batch * output_size, num_quantiles)
            predictions = predictions.view(-1, self.num_quantiles)

            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, i]
                losses.append(
                    torch.max((q - 1) * errors, q * errors).mean()
                )

            return sum(losses) / len(losses)

        def configure_optimizers(self):
            """Configure optimizer and scheduler."""
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
                anneal_strategy='cos',
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        def get_attention_weights(self) -> Optional[torch.Tensor]:
            """
            Get stored attention weights for interpretability.

            Returns:
                Attention weights tensor or None if not computed
            """
            return self.attention_weights

        def predict_with_uncertainty(
            self,
            x: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """
            Make predictions with uncertainty estimates.

            Returns dictionary with point prediction and confidence intervals.
            """
            self.eval()
            with torch.no_grad():
                output = self(x)

            if not self.quantiles:
                return {'prediction': output}

            output = output.view(-1, self.output_size, self.num_quantiles)

            result = {
                'prediction': output[:, :, self.num_quantiles // 2],  # Median
            }

            for i, q in enumerate(self.quantiles):
                result[f'q{int(q*100)}'] = output[:, :, i]

            return result

else:
    # Dummy class when PyTorch Lightning not available
    class TemporalFusionTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch and PyTorch Lightning required for TemporalFusionTransformer. "
                "Install with: pip install torch pytorch-lightning"
            )
