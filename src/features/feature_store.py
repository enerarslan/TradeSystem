"""
Feature Store implementation for AlphaTrade system.

This module provides a lightweight feature store for:
- Feature definition and registration
- Point-in-time feature serving
- Feature versioning
- Offline/Online feature serving simulation
- Feature lineage tracking

Designed for institutional requirements:
- Reproducible feature pipelines
- Time-travel for backtesting
- Feature consistency between training and serving
- Audit trail for model governance

Note: This is a lightweight implementation. For production at scale,
consider integrating with Feast, Tecton, or similar feature platforms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of features."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TIMESTAMP = "timestamp"


class DataSource(str, Enum):
    """Feature data sources."""
    OHLCV = "ohlcv"
    TICK = "tick"
    ORDER_BOOK = "order_book"
    FUNDAMENTAL = "fundamental"
    ALTERNATIVE = "alternative"
    MACRO = "macro"
    DERIVED = "derived"


class AggregationType(str, Enum):
    """Time-window aggregation types."""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"
    COUNT = "count"
    MEDIAN = "median"


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature.

    Attributes:
        name: Unique feature name
        description: Human-readable description
        feature_type: Type of feature (continuous, categorical, etc.)
        data_source: Source of the raw data
        transform_fn: Function to compute the feature
        dependencies: Names of features this depends on
        lookback_periods: Historical data required
        tags: Metadata tags
        version: Feature version
        created_at: Creation timestamp
    """
    name: str
    description: str = ""
    feature_type: FeatureType = FeatureType.CONTINUOUS
    data_source: DataSource = DataSource.DERIVED
    transform_fn: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    lookback_periods: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def feature_id(self) -> str:
        """Unique feature identifier (name + version)."""
        return f"{self.name}:v{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "data_source": self.data_source.value,
            "dependencies": self.dependencies,
            "lookback_periods": self.lookback_periods,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FeatureView:
    """
    A collection of related features.

    Attributes:
        name: View name
        features: List of feature definitions
        entity_columns: Columns that identify entities (e.g., symbol, date)
        ttl: Time-to-live for cached features
        online: Whether to serve online
        tags: Metadata tags
    """
    name: str
    features: List[FeatureDefinition] = field(default_factory=list)
    entity_columns: List[str] = field(default_factory=lambda: ["symbol", "timestamp"])
    ttl: Optional[timedelta] = None
    online: bool = False
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]

    def add_feature(self, feature: FeatureDefinition) -> None:
        """Add a feature to the view."""
        if feature.name in self.feature_names:
            raise ValueError(f"Feature '{feature.name}' already exists in view")
        self.features.append(feature)


@dataclass
class FeatureValue:
    """
    A single feature value with metadata.

    Attributes:
        name: Feature name
        value: Feature value
        timestamp: As-of timestamp
        entity_key: Entity identifier
        version: Feature version
    """
    name: str
    value: Any
    timestamp: pd.Timestamp
    entity_key: Dict[str, Any]
    version: str = "1.0.0"


class FeatureStore:
    """
    Lightweight feature store for quantitative trading.

    Provides:
    - Feature registration and management
    - Point-in-time feature retrieval
    - Feature materialization and caching
    - Time-travel for backtesting

    Example:
        store = FeatureStore(storage_path="features/")

        # Register features
        store.register_feature(FeatureDefinition(
            name="returns_1d",
            transform_fn=lambda df: df['close'].pct_change(),
        ))

        # Materialize features
        store.materialize(data, start_date, end_date)

        # Get features for training
        features = store.get_historical_features(
            entity_df=training_dates,
            feature_names=["returns_1d", "volume_ma_20"],
        )

        # Point-in-time lookup
        value = store.get_feature_value(
            feature_name="returns_1d",
            entity_key={"symbol": "AAPL"},
            timestamp=pd.Timestamp("2024-01-15"),
        )
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize feature store.

        Args:
            storage_path: Path for feature storage
            cache_enabled: Enable in-memory caching
        """
        self.storage_path = Path(storage_path) if storage_path else Path("data/features")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}

        # Feature registry
        self._features: Dict[str, FeatureDefinition] = {}
        self._views: Dict[str, FeatureView] = {}

        # Materialized feature data
        self._materialized: Dict[str, pd.DataFrame] = {}

        # Load existing registry
        self._load_registry()

        logger.info(f"FeatureStore initialized at {self.storage_path}")

    def register_feature(
        self,
        feature: FeatureDefinition,
        overwrite: bool = False,
    ) -> None:
        """
        Register a feature definition.

        Args:
            feature: Feature definition to register
            overwrite: Overwrite if exists
        """
        if feature.name in self._features and not overwrite:
            raise ValueError(
                f"Feature '{feature.name}' already registered. "
                "Use overwrite=True to replace."
            )

        self._features[feature.name] = feature
        self._save_registry()

        logger.info(f"Registered feature: {feature.name} (v{feature.version})")

    def register_view(
        self,
        view: FeatureView,
        overwrite: bool = False,
    ) -> None:
        """
        Register a feature view.

        Args:
            view: Feature view to register
            overwrite: Overwrite if exists
        """
        if view.name in self._views and not overwrite:
            raise ValueError(
                f"View '{view.name}' already registered. "
                "Use overwrite=True to replace."
            )

        # Register all features in the view
        for feature in view.features:
            if feature.name not in self._features:
                self.register_feature(feature, overwrite=False)

        self._views[view.name] = view
        self._save_registry()

        logger.info(f"Registered view: {view.name} with {len(view.features)} features")

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self._features.get(name)

    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._features.keys())

    def list_views(self) -> List[str]:
        """List all registered view names."""
        return list(self._views.keys())

    def materialize(
        self,
        data: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        feature_names: Optional[List[str]] = None,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Materialize features from raw data.

        Args:
            data: Raw OHLCV or other source data
            start_date: Start date for materialization
            end_date: End date for materialization
            feature_names: Features to materialize (None = all)
            symbol: Symbol identifier

        Returns:
            DataFrame with materialized features
        """
        if feature_names is None:
            feature_names = self.list_features()

        # Filter by date if provided
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        results = pd.DataFrame(index=data.index)

        # Sort features by dependency order
        ordered_features = self._topological_sort(feature_names)

        for feature_name in ordered_features:
            if feature_name not in self._features:
                logger.warning(f"Feature '{feature_name}' not registered, skipping")
                continue

            feature = self._features[feature_name]

            try:
                if feature.transform_fn is not None:
                    # Apply transform function
                    # First, ensure dependencies are computed
                    dep_data = data.copy()
                    for dep in feature.dependencies:
                        if dep in results.columns:
                            dep_data[dep] = results[dep]

                    feature_values = feature.transform_fn(dep_data)

                    if isinstance(feature_values, pd.Series):
                        results[feature_name] = feature_values
                    elif isinstance(feature_values, pd.DataFrame):
                        for col in feature_values.columns:
                            results[f"{feature_name}_{col}"] = feature_values[col]
                    else:
                        results[feature_name] = feature_values

            except Exception as e:
                logger.error(f"Error computing feature '{feature_name}': {e}")
                continue

        # Store in materialized cache
        cache_key = f"{symbol or 'default'}_{start_date}_{end_date}"
        self._materialized[cache_key] = results

        # Persist to storage
        if symbol:
            feature_file = self.storage_path / f"{symbol}_features.parquet"
            results.to_parquet(feature_file)

        logger.info(f"Materialized {len(results.columns)} features for {len(results)} rows")

        return results

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: List[str],
        ttl: Optional[timedelta] = None,
    ) -> pd.DataFrame:
        """
        Get historical features for entity DataFrame.

        This is the main interface for training data retrieval.
        Performs point-in-time joins to avoid data leakage.

        Args:
            entity_df: DataFrame with entity keys and timestamps
            feature_names: Features to retrieve
            ttl: Time-to-live for feature values

        Returns:
            DataFrame with features joined to entity_df
        """
        result = entity_df.copy()

        # Determine entity columns
        timestamp_col = None
        for col in ["timestamp", "date", "time", "datetime"]:
            if col in entity_df.columns:
                timestamp_col = col
                break

        if timestamp_col is None and isinstance(entity_df.index, pd.DatetimeIndex):
            timestamp_col = "index"
            entity_df = entity_df.reset_index()
            entity_df.columns = ["timestamp"] + list(entity_df.columns[1:])
            timestamp_col = "timestamp"

        # Get symbol column if exists
        symbol_col = None
        for col in ["symbol", "ticker", "asset"]:
            if col in entity_df.columns:
                symbol_col = col
                break

        # For each row, get point-in-time feature values
        feature_data = []

        for idx, row in entity_df.iterrows():
            ts = row[timestamp_col] if timestamp_col else idx
            symbol = row[symbol_col] if symbol_col else "default"

            row_features = {}
            for feature_name in feature_names:
                value = self.get_feature_value(
                    feature_name=feature_name,
                    entity_key={"symbol": symbol},
                    timestamp=ts,
                    ttl=ttl,
                )
                row_features[feature_name] = value.value if value else np.nan

            feature_data.append(row_features)

        feature_df = pd.DataFrame(feature_data, index=entity_df.index)
        result = pd.concat([result, feature_df], axis=1)

        return result

    def get_feature_value(
        self,
        feature_name: str,
        entity_key: Dict[str, Any],
        timestamp: pd.Timestamp,
        ttl: Optional[timedelta] = None,
    ) -> Optional[FeatureValue]:
        """
        Get a single feature value at a specific point in time.

        Args:
            feature_name: Name of the feature
            entity_key: Entity identifier (e.g., {"symbol": "AAPL"})
            timestamp: As-of timestamp
            ttl: Maximum age of feature value

        Returns:
            FeatureValue if found, None otherwise
        """
        symbol = entity_key.get("symbol", "default")

        # Check materialized cache
        for cache_key, df in self._materialized.items():
            if symbol in cache_key or cache_key.startswith("default"):
                if feature_name in df.columns:
                    # Point-in-time lookup
                    mask = df.index <= timestamp
                    if mask.any():
                        valid_data = df.loc[mask, feature_name]
                        latest_ts = valid_data.index[-1]

                        # Check TTL
                        if ttl and (timestamp - latest_ts) > ttl:
                            return None

                        return FeatureValue(
                            name=feature_name,
                            value=valid_data.iloc[-1],
                            timestamp=latest_ts,
                            entity_key=entity_key,
                            version=self._features.get(feature_name, FeatureDefinition(name="")).version,
                        )

        # Try to load from storage
        feature_file = self.storage_path / f"{symbol}_features.parquet"
        if feature_file.exists():
            try:
                df = pd.read_parquet(feature_file)
                if feature_name in df.columns:
                    mask = df.index <= timestamp
                    if mask.any():
                        valid_data = df.loc[mask, feature_name]
                        return FeatureValue(
                            name=feature_name,
                            value=valid_data.iloc[-1],
                            timestamp=valid_data.index[-1],
                            entity_key=entity_key,
                        )
            except Exception as e:
                logger.error(f"Error loading features from storage: {e}")

        return None

    def get_online_features(
        self,
        feature_names: List[str],
        entity_key: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get latest feature values for online serving.

        Args:
            feature_names: Features to retrieve
            entity_key: Entity identifier

        Returns:
            Dictionary of feature name to value
        """
        result = {}
        current_time = pd.Timestamp.now()

        for feature_name in feature_names:
            value = self.get_feature_value(
                feature_name=feature_name,
                entity_key=entity_key,
                timestamp=current_time,
            )
            result[feature_name] = value.value if value else None

        return result

    def _topological_sort(self, feature_names: List[str]) -> List[str]:
        """Sort features by dependency order."""
        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        for name in feature_names:
            feature = self._features.get(name)
            if feature:
                graph[name] = set(feature.dependencies)
            else:
                graph[name] = set()

        # Kahn's algorithm
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for dep in graph[node]:
                if dep in in_degree:
                    in_degree[node] += 1

        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for other in graph:
                if node in graph[other]:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        return result

    def _save_registry(self) -> None:
        """Save feature registry to disk."""
        registry = {
            "features": {
                name: feature.to_dict()
                for name, feature in self._features.items()
            },
            "views": {
                name: {
                    "name": view.name,
                    "feature_names": view.feature_names,
                    "entity_columns": view.entity_columns,
                    "tags": view.tags,
                }
                for name, view in self._views.items()
            },
        }

        registry_file = self.storage_path / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def _load_registry(self) -> None:
        """Load feature registry from disk."""
        registry_file = self.storage_path / "registry.json"
        if not registry_file.exists():
            return

        try:
            with open(registry_file) as f:
                registry = json.load(f)

            # Load features
            for name, feature_dict in registry.get("features", {}).items():
                feature = FeatureDefinition(
                    name=feature_dict["name"],
                    description=feature_dict.get("description", ""),
                    feature_type=FeatureType(feature_dict.get("feature_type", "continuous")),
                    data_source=DataSource(feature_dict.get("data_source", "derived")),
                    dependencies=feature_dict.get("dependencies", []),
                    lookback_periods=feature_dict.get("lookback_periods", 0),
                    tags=feature_dict.get("tags", {}),
                    version=feature_dict.get("version", "1.0.0"),
                )
                self._features[name] = feature

            logger.info(f"Loaded {len(self._features)} features from registry")

        except Exception as e:
            logger.error(f"Error loading registry: {e}")


class FeatureBuilder:
    """
    Builder class for creating feature definitions.

    Provides fluent API for feature definition.

    Example:
        feature = (FeatureBuilder("returns_20d")
            .description("20-day returns")
            .type(FeatureType.CONTINUOUS)
            .source(DataSource.OHLCV)
            .transform(lambda df: df['close'].pct_change(20))
            .lookback(20)
            .tag("category", "momentum")
            .build())
    """

    def __init__(self, name: str):
        """Initialize builder with feature name."""
        self._name = name
        self._description = ""
        self._type = FeatureType.CONTINUOUS
        self._source = DataSource.DERIVED
        self._transform = None
        self._dependencies: List[str] = []
        self._lookback = 0
        self._tags: Dict[str, str] = {}
        self._version = "1.0.0"

    def description(self, desc: str) -> "FeatureBuilder":
        """Set feature description."""
        self._description = desc
        return self

    def type(self, feature_type: FeatureType) -> "FeatureBuilder":
        """Set feature type."""
        self._type = feature_type
        return self

    def source(self, data_source: DataSource) -> "FeatureBuilder":
        """Set data source."""
        self._source = data_source
        return self

    def transform(self, fn: Callable) -> "FeatureBuilder":
        """Set transform function."""
        self._transform = fn
        return self

    def depends_on(self, *features: str) -> "FeatureBuilder":
        """Add feature dependencies."""
        self._dependencies.extend(features)
        return self

    def lookback(self, periods: int) -> "FeatureBuilder":
        """Set lookback periods."""
        self._lookback = periods
        return self

    def tag(self, key: str, value: str) -> "FeatureBuilder":
        """Add a tag."""
        self._tags[key] = value
        return self

    def version(self, ver: str) -> "FeatureBuilder":
        """Set version."""
        self._version = ver
        return self

    def build(self) -> FeatureDefinition:
        """Build the feature definition."""
        return FeatureDefinition(
            name=self._name,
            description=self._description,
            feature_type=self._type,
            data_source=self._source,
            transform_fn=self._transform,
            dependencies=self._dependencies,
            lookback_periods=self._lookback,
            tags=self._tags,
            version=self._version,
        )


# Pre-defined feature transforms
def create_standard_features() -> List[FeatureDefinition]:
    """
    Create standard feature definitions.

    Returns a list of commonly used features for trading.
    """
    features = []

    # Returns
    for period in [1, 5, 10, 20, 60]:
        features.append(
            FeatureBuilder(f"returns_{period}d")
            .description(f"{period}-day returns")
            .source(DataSource.OHLCV)
            .transform(lambda df, p=period: df["close"].pct_change(p))
            .lookback(period)
            .tag("category", "momentum")
            .build()
        )

    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        features.append(
            FeatureBuilder(f"sma_{period}")
            .description(f"{period}-day simple moving average")
            .source(DataSource.OHLCV)
            .transform(lambda df, p=period: df["close"].rolling(p).mean())
            .lookback(period)
            .tag("category", "trend")
            .build()
        )

    # Volatility
    for period in [5, 10, 20, 60]:
        features.append(
            FeatureBuilder(f"volatility_{period}d")
            .description(f"{period}-day realized volatility")
            .source(DataSource.OHLCV)
            .transform(
                lambda df, p=period: df["close"].pct_change().rolling(p).std() * np.sqrt(252)
            )
            .lookback(period)
            .tag("category", "volatility")
            .build()
        )

    # Volume features
    features.append(
        FeatureBuilder("volume_ma_20")
        .description("20-day volume moving average")
        .source(DataSource.OHLCV)
        .transform(lambda df: df["volume"].rolling(20).mean())
        .lookback(20)
        .tag("category", "volume")
        .build()
    )

    features.append(
        FeatureBuilder("volume_ratio")
        .description("Volume relative to 20-day average")
        .source(DataSource.OHLCV)
        .transform(lambda df: df["volume"] / df["volume"].rolling(20).mean())
        .lookback(20)
        .depends_on("volume_ma_20")
        .tag("category", "volume")
        .build()
    )

    # RSI
    def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    features.append(
        FeatureBuilder("rsi_14")
        .description("14-period RSI")
        .source(DataSource.OHLCV)
        .transform(calc_rsi)
        .lookback(14)
        .tag("category", "momentum")
        .build()
    )

    # Bollinger Band position
    def bb_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (df["close"] - lower) / (upper - lower)

    features.append(
        FeatureBuilder("bb_position")
        .description("Position within Bollinger Bands (0-1)")
        .source(DataSource.OHLCV)
        .transform(bb_position)
        .lookback(20)
        .tag("category", "mean_reversion")
        .build()
    )

    return features


def initialize_feature_store(
    storage_path: Optional[Union[str, Path]] = None,
    with_standard_features: bool = True,
) -> FeatureStore:
    """
    Initialize feature store with standard features.

    Args:
        storage_path: Path for feature storage
        with_standard_features: Include standard features

    Returns:
        Configured FeatureStore
    """
    store = FeatureStore(storage_path=storage_path)

    if with_standard_features:
        for feature in create_standard_features():
            try:
                store.register_feature(feature, overwrite=False)
            except ValueError:
                pass  # Feature already exists

    return store
