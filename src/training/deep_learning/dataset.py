"""
PyTorch Dataset and DataLoader utilities for financial time-series.

This module provides data loading infrastructure for deep learning models:
- TimeSeriesDataset for sliding window sequences
- SequenceDataModule for PyTorch Lightning
- Multi-horizon prediction support
- Memory-efficient data loading
- Train/Validation/Test splitting with temporal ordering

Designed for JPMorgan-level requirements:
- No data leakage in temporal splits
- Memory-efficient handling of large datasets
- Support for [Batch, Sequence_Len, Features] tensors
- Integration with LSTM, Transformer models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Deep learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_HORIZON = "multi_horizon"


@dataclass
class DatasetConfig:
    """Configuration for time-series dataset."""
    window_size: int = 60
    prediction_horizon: int = 1
    step_size: int = 1
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    task_type: TaskType = TaskType.REGRESSION
    normalize_target: bool = False
    dtype: str = "float32"


if TORCH_AVAILABLE:

    class TimeSeriesDataset(Dataset):
        """
        PyTorch Dataset for time-series with sliding window.

        Creates sequences of shape [window_size, num_features] from
        flat tabular data for LSTM/Transformer training.

        This dataset handles:
        - Rolling window creation
        - Proper temporal alignment (no future data leakage)
        - Multi-horizon targets
        - Memory-efficient indexing

        Example:
            dataset = TimeSeriesDataset(
                features=X,  # (n_samples, n_features)
                targets=y,   # (n_samples,)
                window_size=60,
                prediction_horizon=1
            )

            # Returns x: [60, n_features], y: scalar
            x, y = dataset[0]

            # Use with DataLoader
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
        """

        def __init__(
            self,
            features: Union[np.ndarray, pd.DataFrame],
            targets: Union[np.ndarray, pd.Series],
            window_size: int = 60,
            prediction_horizon: int = 1,
            step_size: int = 1,
            dtype: torch.dtype = torch.float32,
            transform: Optional[Callable] = None,
        ):
            """
            Initialize TimeSeriesDataset.

            Args:
                features: Feature array (n_samples, n_features)
                targets: Target array (n_samples,) or (n_samples, n_horizons)
                window_size: Number of time steps in each sequence
                prediction_horizon: How many steps ahead to predict
                step_size: Step between consecutive sequences (default 1)
                dtype: PyTorch data type
                transform: Optional transform to apply to features
            """
            # Convert to numpy if needed
            if isinstance(features, pd.DataFrame):
                self.feature_names = list(features.columns)
                features = features.values
            else:
                self.feature_names = None

            if isinstance(targets, pd.Series):
                targets = targets.values

            # Validate shapes
            if len(features) != len(targets):
                raise ValueError(
                    f"Features and targets must have same length. "
                    f"Got {len(features)} and {len(targets)}"
                )

            # Store parameters
            self.window_size = window_size
            self.prediction_horizon = prediction_horizon
            self.step_size = step_size
            self.dtype = dtype
            self.transform = transform

            # Convert to tensors
            self.features = torch.tensor(features, dtype=dtype)
            self.targets = torch.tensor(targets, dtype=dtype)

            # Calculate valid indices
            # We need window_size steps for input + prediction_horizon for target
            max_start_idx = len(features) - window_size - prediction_horizon + 1
            self.valid_indices = list(range(0, max_start_idx, step_size))

            if len(self.valid_indices) == 0:
                raise ValueError(
                    f"Not enough data for window_size={window_size} and "
                    f"prediction_horizon={prediction_horizon}. Need at least "
                    f"{window_size + prediction_horizon} samples."
                )

            logger.info(
                f"Created TimeSeriesDataset: {len(self.valid_indices)} sequences, "
                f"window_size={window_size}, horizon={prediction_horizon}"
            )

        def __len__(self) -> int:
            """Return number of sequences."""
            return len(self.valid_indices)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get a sequence and its target.

            Args:
                idx: Sequence index

            Returns:
                Tuple of (x, y) where:
                - x: [window_size, n_features]
                - y: scalar or [n_horizons]
            """
            start_idx = self.valid_indices[idx]
            end_idx = start_idx + self.window_size

            # Feature window: [window_size, n_features]
            x = self.features[start_idx:end_idx]

            # Target: value at end of window + prediction_horizon
            target_idx = end_idx + self.prediction_horizon - 1
            y = self.targets[target_idx]

            # Apply transform if provided
            if self.transform is not None:
                x = self.transform(x)

            return x, y

        def get_feature_dim(self) -> int:
            """Return number of features."""
            return self.features.shape[1]

        def get_window_size(self) -> int:
            """Return window size."""
            return self.window_size


    class MultiHorizonDataset(Dataset):
        """
        Dataset for multi-horizon prediction.

        Returns multiple future targets for each input sequence,
        useful for models that predict multiple time steps ahead.

        Example:
            dataset = MultiHorizonDataset(
                features=X,
                targets=y,
                window_size=60,
                horizons=[1, 5, 10, 20]  # Predict 1, 5, 10, 20 steps ahead
            )

            x, y = dataset[0]
            # x: [60, n_features]
            # y: [4] (targets for each horizon)
        """

        def __init__(
            self,
            features: Union[np.ndarray, pd.DataFrame],
            targets: Union[np.ndarray, pd.Series],
            window_size: int = 60,
            horizons: List[int] = None,
            step_size: int = 1,
            dtype: torch.dtype = torch.float32,
        ):
            if horizons is None:
                horizons = [1, 5, 10]

            if isinstance(features, pd.DataFrame):
                features = features.values
            if isinstance(targets, pd.Series):
                targets = targets.values

            self.window_size = window_size
            self.horizons = sorted(horizons)
            self.max_horizon = max(horizons)
            self.step_size = step_size
            self.dtype = dtype

            self.features = torch.tensor(features, dtype=dtype)
            self.targets = torch.tensor(targets, dtype=dtype)

            # Calculate valid indices
            max_start_idx = len(features) - window_size - self.max_horizon
            self.valid_indices = list(range(0, max_start_idx, step_size))

            logger.info(
                f"Created MultiHorizonDataset: {len(self.valid_indices)} sequences, "
                f"horizons={self.horizons}"
            )

        def __len__(self) -> int:
            return len(self.valid_indices)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            start_idx = self.valid_indices[idx]
            end_idx = start_idx + self.window_size

            x = self.features[start_idx:end_idx]

            # Get targets for all horizons
            y = torch.stack([
                self.targets[end_idx + h - 1]
                for h in self.horizons
            ])

            return x, y


    class TemporalBatchSampler(Sampler):
        """
        Batch sampler that preserves temporal ordering.

        Important for time-series to avoid shuffling across time,
        while still allowing batch-level randomization.
        """

        def __init__(
            self,
            data_source: Dataset,
            batch_size: int,
            shuffle_batches: bool = False,
            drop_last: bool = False,
        ):
            self.data_source = data_source
            self.batch_size = batch_size
            self.shuffle_batches = shuffle_batches
            self.drop_last = drop_last

        def __iter__(self):
            # Create batches in temporal order
            indices = list(range(len(self.data_source)))

            batches = [
                indices[i:i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]

            if self.drop_last and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]

            if self.shuffle_batches:
                np.random.shuffle(batches)

            for batch in batches:
                yield batch

        def __len__(self):
            n = len(self.data_source)
            if self.drop_last:
                return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size


    def create_dataloaders(
        features: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        window_size: int = 60,
        prediction_horizon: int = 1,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.

        Splits data temporally (no shuffling to prevent leakage).

        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array
            window_size: Sequence length
            prediction_horizon: Steps ahead to predict
            batch_size: Batch size
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            num_workers: Number of data loading workers
            pin_memory: Pin memory for GPU transfer
            shuffle_train: Whether to shuffle training batches

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.Series):
            targets = targets.values

        n_samples = len(features)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Split temporally
        train_features = features[:train_end]
        train_targets = targets[:train_end]

        val_features = features[train_end:val_end]
        val_targets = targets[train_end:val_end]

        test_features = features[val_end:]
        test_targets = targets[val_end:]

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_features, train_targets,
            window_size=window_size,
            prediction_horizon=prediction_horizon
        )

        val_dataset = TimeSeriesDataset(
            val_features, val_targets,
            window_size=window_size,
            prediction_horizon=prediction_horizon
        )

        test_dataset = TimeSeriesDataset(
            test_features, test_targets,
            window_size=window_size,
            prediction_horizon=prediction_horizon
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,  # Usually False for time-series
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info(
            f"Created DataLoaders: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)} sequences"
        )

        return train_loader, val_loader, test_loader


    def create_cv_dataloaders(
        features: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        window_size: int = 60,
        prediction_horizon: int = 1,
        batch_size: int = 32,
        n_splits: int = 5,
        gap: int = 0,
        num_workers: int = 0,
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create time-series cross-validation DataLoaders.

        Uses expanding window CV where training data grows
        over time (no future data leakage).

        Args:
            features: Feature array
            targets: Target array
            window_size: Sequence length
            prediction_horizon: Steps ahead to predict
            batch_size: Batch size
            n_splits: Number of CV folds
            gap: Gap between train and validation
            num_workers: Number of workers

        Returns:
            List of (train_loader, val_loader) tuples for each fold
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.Series):
            targets = targets.values

        n_samples = len(features)
        fold_size = n_samples // (n_splits + 1)

        dataloaders = []

        for fold in range(n_splits):
            # Expanding window: train on increasing data
            train_end = (fold + 1) * fold_size
            val_start = train_end + gap
            val_end = val_start + fold_size

            if val_end > n_samples:
                break

            train_features = features[:train_end]
            train_targets = targets[:train_end]

            val_features = features[val_start:val_end]
            val_targets = targets[val_start:val_end]

            train_dataset = TimeSeriesDataset(
                train_features, train_targets,
                window_size=window_size,
                prediction_horizon=prediction_horizon
            )

            val_dataset = TimeSeriesDataset(
                val_features, val_targets,
                window_size=window_size,
                prediction_horizon=prediction_horizon
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            dataloaders.append((train_loader, val_loader))
            logger.info(f"Fold {fold + 1}: train={len(train_dataset)}, val={len(val_dataset)}")

        return dataloaders


if TORCH_AVAILABLE and LIGHTNING_AVAILABLE:

    class TimeSeriesDataModule(pl.LightningDataModule):
        """
        PyTorch Lightning DataModule for time-series data.

        Encapsulates data loading logic for Lightning training:
        - Automatic train/val/test splitting
        - Configurable DataLoaders
        - Memory-efficient setup

        Example:
            dm = TimeSeriesDataModule(
                features=X,
                targets=y,
                window_size=60,
                batch_size=32
            )

            trainer = pl.Trainer(max_epochs=100)
            trainer.fit(model, dm)
        """

        def __init__(
            self,
            features: Union[np.ndarray, pd.DataFrame],
            targets: Union[np.ndarray, pd.Series],
            window_size: int = 60,
            prediction_horizon: int = 1,
            batch_size: int = 32,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
            super().__init__()

            self.features = features
            self.targets = targets
            self.window_size = window_size
            self.prediction_horizon = prediction_horizon
            self.batch_size = batch_size
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.num_workers = num_workers
            self.pin_memory = pin_memory

            # Will be set in setup()
            self.train_dataset: Optional[TimeSeriesDataset] = None
            self.val_dataset: Optional[TimeSeriesDataset] = None
            self.test_dataset: Optional[TimeSeriesDataset] = None

        def setup(self, stage: Optional[str] = None):
            """Set up datasets for each stage."""
            if isinstance(self.features, pd.DataFrame):
                features = self.features.values
            else:
                features = self.features

            if isinstance(self.targets, pd.Series):
                targets = self.targets.values
            else:
                targets = self.targets

            n_samples = len(features)
            train_end = int(n_samples * self.train_ratio)
            val_end = int(n_samples * (self.train_ratio + self.val_ratio))

            if stage == "fit" or stage is None:
                self.train_dataset = TimeSeriesDataset(
                    features[:train_end],
                    targets[:train_end],
                    window_size=self.window_size,
                    prediction_horizon=self.prediction_horizon,
                )

                self.val_dataset = TimeSeriesDataset(
                    features[train_end:val_end],
                    targets[train_end:val_end],
                    window_size=self.window_size,
                    prediction_horizon=self.prediction_horizon,
                )

            if stage == "test" or stage is None:
                self.test_dataset = TimeSeriesDataset(
                    features[val_end:],
                    targets[val_end:],
                    window_size=self.window_size,
                    prediction_horizon=self.prediction_horizon,
                )

        def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Keep temporal order
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        def get_feature_dim(self) -> int:
            """Get number of input features."""
            if isinstance(self.features, pd.DataFrame):
                return self.features.shape[1]
            return self.features.shape[1]

        def get_window_size(self) -> int:
            """Get window size."""
            return self.window_size


else:
    # Fallback when PyTorch not available

    class TimeSeriesDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required for TimeSeriesDataset. "
                "Install with: pip install torch"
            )

    class TimeSeriesDataModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch and PyTorch Lightning required for TimeSeriesDataModule. "
                "Install with: pip install torch pytorch-lightning"
            )

    def create_dataloaders(*args, **kwargs):
        raise ImportError("PyTorch required. Install with: pip install torch")

    def create_cv_dataloaders(*args, **kwargs):
        raise ImportError("PyTorch required. Install with: pip install torch")


def prepare_data_for_dl(
    features: Union[np.ndarray, pd.DataFrame],
    targets: Union[np.ndarray, pd.Series],
    window_size: int = 60,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    High-level function to prepare all data for deep learning.

    Returns everything needed for training:
    - DataLoaders
    - Dimension information
    - Configuration

    Args:
        features: Feature data
        targets: Target data
        window_size: Sequence length
        prediction_horizon: Prediction horizon
        batch_size: Batch size
        train_ratio: Training data ratio
        val_ratio: Validation data ratio

    Returns:
        Dictionary with dataloaders and metadata
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    train_loader, val_loader, test_loader = create_dataloaders(
        features=features,
        targets=targets,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    n_features = features.shape[1] if hasattr(features, 'shape') else len(features[0])

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'n_features': n_features,
        'window_size': window_size,
        'prediction_horizon': prediction_horizon,
        'batch_size': batch_size,
        'n_train_samples': len(train_loader.dataset),
        'n_val_samples': len(val_loader.dataset),
        'n_test_samples': len(test_loader.dataset),
    }
