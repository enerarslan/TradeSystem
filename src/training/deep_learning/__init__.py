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

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = []

# Models - require PyTorch
if TORCH_AVAILABLE:
    try:
        from .lstm import LSTMPredictor, AttentionLSTM
        from .transformer import TemporalFusionTransformer
        __all__.extend([
            "LSTMPredictor",
            "AttentionLSTM",
            "TemporalFusionTransformer",
        ])
    except ImportError as e:
        import logging
        logging.warning(f"Could not import deep learning models: {e}")

# Losses - require PyTorch
if TORCH_AVAILABLE:
    try:
        from .losses import (
            SharpeLoss,
            SortinoLoss,
            MaxDrawdownLoss,
            CombinedFinancialLoss,
        )
        __all__.extend([
            "SharpeLoss",
            "SortinoLoss",
            "MaxDrawdownLoss",
            "CombinedFinancialLoss",
        ])
    except ImportError as e:
        import logging
        logging.warning(f"Could not import loss functions: {e}")

# Data Loading - require PyTorch
if TORCH_AVAILABLE:
    try:
        from .dataset import (
            TimeSeriesDataset,
            TimeSeriesDataModule,
            MultiHorizonDataset,
            create_dataloaders,
            create_cv_dataloaders,
            prepare_data_for_dl,
        )
        __all__.extend([
            "TimeSeriesDataset",
            "TimeSeriesDataModule",
            "MultiHorizonDataset",
            "create_dataloaders",
            "create_cv_dataloaders",
            "prepare_data_for_dl",
        ])
    except ImportError as e:
        import logging
        logging.warning(f"Could not import dataset utilities: {e}")
