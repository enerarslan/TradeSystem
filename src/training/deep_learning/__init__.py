"""
Deep Learning models for financial time-series prediction.

This module provides PyTorch Lightning-based implementations of:
- LSTM with attention
- Temporal Fusion Transformer
- Custom financial loss functions

Designed for institutional-grade requirements:
- GPU/MPS acceleration
- Mixed precision training
- Robust validation
- Production deployment
"""

from .lstm import LSTMPredictor, AttentionLSTM
from .transformer import TemporalFusionTransformer
from .losses import (
    SharpeLoss,
    SortinoLoss,
    MaxDrawdownLoss,
    CombinedFinancialLoss,
)

__all__ = [
    "LSTMPredictor",
    "AttentionLSTM",
    "TemporalFusionTransformer",
    "SharpeLoss",
    "SortinoLoss",
    "MaxDrawdownLoss",
    "CombinedFinancialLoss",
]
