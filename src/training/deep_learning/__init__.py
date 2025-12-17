"""
Deep Learning models for financial time-series prediction.

This module provides PyTorch Lightning-based implementations of:
- LSTM with attention
- Temporal Fusion Transformer
- Custom financial loss functions
- Time-series data loading utilities

Designed for institutional-grade requirements:
- GPU/MPS acceleration
- Mixed precision training
- Robust validation
- Production deployment
- Memory-efficient data loading
"""

from .lstm import LSTMPredictor, AttentionLSTM
from .transformer import TemporalFusionTransformer
from .losses import (
    SharpeLoss,
    SortinoLoss,
    MaxDrawdownLoss,
    CombinedFinancialLoss,
)
from .dataset import (
    TimeSeriesDataset,
    TimeSeriesDataModule,
    MultiHorizonDataset,
    create_dataloaders,
    create_cv_dataloaders,
    prepare_data_for_dl,
)

__all__ = [
    # Models
    "LSTMPredictor",
    "AttentionLSTM",
    "TemporalFusionTransformer",
    # Losses
    "SharpeLoss",
    "SortinoLoss",
    "MaxDrawdownLoss",
    "CombinedFinancialLoss",
    # Data Loading
    "TimeSeriesDataset",
    "TimeSeriesDataModule",
    "MultiHorizonDataset",
    "create_dataloaders",
    "create_cv_dataloaders",
    "prepare_data_for_dl",
]
