"""
Real-Time Feature Store
=======================

Feature store implementation for consistent feature serving between
training (offline) and inference (online).

Solves the "Feature Mismatch Error" by ensuring:
1. Same features used in training and inference
2. Real-time feature computation
3. Point-in-time correct features
4. Feature versioning

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable
from pathlib import Path

import numpy as np
import polars as pl

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_logger
from features.technical import TechnicalIndicators, IndicatorConfig
from features.statistical import StatisticalFeatures, StatisticalConfig

logger = get_logger(__name__)


@dataclass
class FeatureDefinition:
    """
    Definition of a feature for the feature store.

    Attributes:
        name: Feature name
        version: Feature version
        function: Computation function
        dependencies: Required input columns
        lookback: Required lookback period
        description: Feature description
    """
    name: str
    version: str = "1.0"
    function: Callable[[pl.DataFrame], pl.Series] | None = None
    dependencies: list[str] = field(default_factory=lambda: ["close"])
    lookback: int = 20
    description: str = ""

    @property
    def key(self) -> str:
        """Get unique feature key."""
        return f"{self.name}:v{self.version}"


@dataclass
class FeatureSet:
    """
    Collection of features for a model.

    Attributes:
        name: Feature set name
        version: Feature set version
        features: List of feature definitions
        created_at: Creation timestamp
    """
    name: str
    version: str = "1.0"
    features: list[FeatureDefinition] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]

    @property
    def max_lookback(self) -> int:
        """Get maximum lookback period."""
        return max(f.lookback for f in self.features) if self.features else 0


class FeatureStore:
    """
    Real-time feature store for ML trading models.

    Provides:
    1. Online feature serving (low latency)
    2. Offline feature computation (batch)
    3. Feature versioning
    4. Point-in-time correctness

    Architecture:
    - Redis: Online feature store (hot data)
    - TimescaleDB: Offline feature store (historical)
    - Feature Registry: Feature definitions and metadata

    Example:
        store = FeatureStore(redis_url="redis://localhost:6379")
        await store.connect()

        # Register feature set
        store.register_feature_set(my_features)

        # Get features for inference
        features = await store.get_online_features("AAPL", feature_set="ml_model_v1")

        # Compute and store features
        await store.compute_and_store("AAPL", bar_data)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "features",
        ttl: int = 3600,  # 1 hour default TTL
    ):
        """
        Initialize feature store.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for feature keys
            ttl: Time-to-live for features in seconds
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required: pip install redis")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl = ttl

        self._client: aioredis.Redis | None = None
        self._connected = False

        # Feature registry
        self._feature_sets: dict[str, FeatureSet] = {}
        self._feature_definitions: dict[str, FeatureDefinition] = {}

        # Feature calculators
        self._technical = TechnicalIndicators()
        self._statistical = StatisticalFeatures()

        # Register default features
        self._register_default_features()

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            await self._client.ping()
            self._connected = True
            logger.info("Feature store connected")

        except Exception as e:
            logger.error(f"Failed to connect to feature store: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
        self._connected = False
        logger.info("Feature store disconnected")

    def _register_default_features(self) -> None:
        """Register default technical and statistical features."""
        # Technical features
        technical_features = [
            FeatureDefinition(name="rsi_14", lookback=14, description="RSI with 14 periods"),
            FeatureDefinition(name="macd", lookback=26, description="MACD line"),
            FeatureDefinition(name="macd_signal", lookback=35, description="MACD signal line"),
            FeatureDefinition(name="macd_hist", lookback=35, description="MACD histogram"),
            FeatureDefinition(name="bb_upper", lookback=20, description="Bollinger upper band"),
            FeatureDefinition(name="bb_middle", lookback=20, description="Bollinger middle band"),
            FeatureDefinition(name="bb_lower", lookback=20, description="Bollinger lower band"),
            FeatureDefinition(name="bb_width", lookback=20, description="Bollinger band width"),
            FeatureDefinition(name="atr_14", lookback=14, dependencies=["high", "low", "close"], description="ATR"),
            FeatureDefinition(name="adx_14", lookback=28, dependencies=["high", "low", "close"], description="ADX"),
            FeatureDefinition(name="cci_20", lookback=20, dependencies=["high", "low", "close"], description="CCI"),
            FeatureDefinition(name="williams_r", lookback=14, dependencies=["high", "low", "close"], description="Williams %R"),
            FeatureDefinition(name="stoch_k", lookback=14, dependencies=["high", "low", "close"], description="Stochastic %K"),
            FeatureDefinition(name="stoch_d", lookback=17, dependencies=["high", "low", "close"], description="Stochastic %D"),
            FeatureDefinition(name="obv", lookback=1, dependencies=["close", "volume"], description="OBV"),
            FeatureDefinition(name="vwap", lookback=1, dependencies=["high", "low", "close", "volume"], description="VWAP"),
        ]

        # Statistical features
        statistical_features = [
            FeatureDefinition(name="return_1", lookback=1, description="1-bar return"),
            FeatureDefinition(name="return_5", lookback=5, description="5-bar return"),
            FeatureDefinition(name="return_10", lookback=10, description="10-bar return"),
            FeatureDefinition(name="return_20", lookback=20, description="20-bar return"),
            FeatureDefinition(name="volatility_20", lookback=20, description="20-bar volatility"),
            FeatureDefinition(name="volatility_60", lookback=60, description="60-bar volatility"),
            FeatureDefinition(name="momentum_10", lookback=10, description="10-bar momentum"),
            FeatureDefinition(name="momentum_20", lookback=20, description="20-bar momentum"),
            FeatureDefinition(name="skewness_20", lookback=20, description="20-bar skewness"),
            FeatureDefinition(name="kurtosis_20", lookback=20, description="20-bar kurtosis"),
            FeatureDefinition(name="zscore_20", lookback=20, description="20-bar z-score"),
            FeatureDefinition(name="percentile_20", lookback=20, description="20-bar percentile rank"),
        ]

        # Register all features
        for feature in technical_features + statistical_features:
            self._feature_definitions[feature.name] = feature

        # Create default feature set
        default_set = FeatureSet(
            name="default",
            version="1.0",
            features=technical_features + statistical_features,
        )
        self._feature_sets["default"] = default_set

    def register_feature_set(self, feature_set: FeatureSet) -> None:
        """
        Register a feature set.

        Args:
            feature_set: Feature set to register
        """
        self._feature_sets[feature_set.name] = feature_set

        for feature in feature_set.features:
            self._feature_definitions[feature.name] = feature

        logger.info(f"Registered feature set: {feature_set.name} with {len(feature_set.features)} features")

    def get_feature_set(self, name: str) -> FeatureSet | None:
        """Get a registered feature set."""
        return self._feature_sets.get(name)

    async def get_online_features(
        self,
        symbol: str,
        feature_set: str = "default",
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """
        Get features from online store (Redis).

        Args:
            symbol: Trading symbol
            feature_set: Feature set name
            timestamp: Timestamp (default: latest)

        Returns:
            Dictionary of feature values
        """
        if not self._client:
            return {}

        fs = self._feature_sets.get(feature_set)
        if not fs:
            logger.warning(f"Feature set not found: {feature_set}")
            return {}

        key = f"{self.key_prefix}:{symbol}:latest"

        # Get all features from Redis hash
        raw_features = await self._client.hgetall(key)

        if not raw_features:
            return {}

        # Filter to requested feature set
        features = {}
        for feature_def in fs.features:
            if feature_def.name in raw_features:
                try:
                    features[feature_def.name] = float(raw_features[feature_def.name])
                except (ValueError, TypeError):
                    features[feature_def.name] = 0.0

        return features

    async def get_online_feature_vector(
        self,
        symbol: str,
        feature_set: str = "default",
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get feature vector as numpy array (for ML inference).

        Args:
            symbol: Trading symbol
            feature_set: Feature set name

        Returns:
            Tuple of (feature_array, feature_names)
        """
        features = await self.get_online_features(symbol, feature_set)

        fs = self._feature_sets.get(feature_set, self._feature_sets["default"])
        feature_names = fs.feature_names

        # Create vector in correct order
        vector = np.array([
            features.get(name, 0.0)
            for name in feature_names
        ])

        return vector, feature_names

    async def store_features(
        self,
        symbol: str,
        features: dict[str, float],
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Store features in online store.

        Args:
            symbol: Trading symbol
            features: Feature dictionary
            timestamp: Feature timestamp

        Returns:
            True if stored successfully
        """
        if not self._client:
            return False

        try:
            timestamp = timestamp or datetime.now()
            key = f"{self.key_prefix}:{symbol}:latest"

            # Store as hash with TTL
            await self._client.hset(key, mapping={
                **{k: str(v) for k, v in features.items()},
                "_timestamp": timestamp.isoformat(),
            })
            await self._client.expire(key, self.ttl)

            return True

        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False

    async def compute_and_store(
        self,
        symbol: str,
        data: pl.DataFrame,
        feature_set: str = "default",
    ) -> dict[str, float]:
        """
        Compute features from data and store in online store.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
            feature_set: Feature set to compute

        Returns:
            Computed features
        """
        features = self.compute_features(data, feature_set)

        # Get latest values
        latest_features = {k: v[-1] for k, v in features.items() if len(v) > 0}

        # Store in Redis
        await self.store_features(symbol, latest_features)

        return latest_features

    def compute_features(
        self,
        data: pl.DataFrame,
        feature_set: str = "default",
    ) -> dict[str, np.ndarray]:
        """
        Compute features from OHLCV data (batch mode).

        Args:
            data: OHLCV DataFrame
            feature_set: Feature set name

        Returns:
            Dictionary of feature arrays
        """
        fs = self._feature_sets.get(feature_set, self._feature_sets["default"])

        # Add all indicators using existing modules
        df = self._technical.add_all_indicators(data)
        df = self._statistical.add_all_features(df)

        # Extract requested features
        features = {}
        for feature_def in fs.features:
            col_name = feature_def.name

            # Try to find matching column
            matching_cols = [c for c in df.columns if col_name in c.lower()]

            if matching_cols:
                features[feature_def.name] = df[matching_cols[0]].to_numpy()
            elif feature_def.function:
                # Compute using custom function
                try:
                    features[feature_def.name] = feature_def.function(df).to_numpy()
                except Exception as e:
                    logger.warning(f"Failed to compute {feature_def.name}: {e}")
                    features[feature_def.name] = np.zeros(len(df))
            else:
                # Feature not found
                features[feature_def.name] = np.zeros(len(df))

        return features

    def compute_features_realtime(
        self,
        bars: list[dict[str, float]],
        feature_set: str = "default",
    ) -> dict[str, float]:
        """
        Compute features from recent bars (real-time mode).

        Optimized for low-latency inference.

        Args:
            bars: List of OHLCV bar dictionaries
            feature_set: Feature set name

        Returns:
            Latest feature values
        """
        if not bars:
            return {}

        # Convert to arrays
        closes = np.array([b["close"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])
        volumes = np.array([b.get("volume", 0) for b in bars])

        features = {}

        # Returns
        if len(closes) >= 2:
            features["return_1"] = (closes[-1] / closes[-2] - 1)
        if len(closes) >= 6:
            features["return_5"] = (closes[-1] / closes[-6] - 1)
        if len(closes) >= 11:
            features["return_10"] = (closes[-1] / closes[-11] - 1)
        if len(closes) >= 21:
            features["return_20"] = (closes[-1] / closes[-21] - 1)

        # RSI
        if len(closes) >= 15:
            gains = np.maximum(np.diff(closes[-15:]), 0)
            losses = np.abs(np.minimum(np.diff(closes[-15:]), 0))
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features["rsi_14"] = 100 - (100 / (1 + rs))
            else:
                features["rsi_14"] = 100

        # MACD
        if len(closes) >= 26:
            ema_12 = self._ema(closes, 12)
            ema_26 = self._ema(closes, 26)
            features["macd"] = ema_12 - ema_26

        # Volatility
        if len(closes) >= 21:
            returns = np.diff(closes[-21:]) / closes[-21:-1]
            features["volatility_20"] = np.std(returns) * np.sqrt(252)

        # Momentum
        if len(closes) >= 11:
            features["momentum_10"] = (closes[-1] / closes[-11] - 1) * 100
        if len(closes) >= 21:
            features["momentum_20"] = (closes[-1] / closes[-21] - 1) * 100

        # Z-score
        if len(closes) >= 20:
            mean_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            if std_20 > 0:
                features["zscore_20"] = (closes[-1] - mean_20) / std_20

        # Bollinger Bands
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            features["bb_upper"] = sma_20 + 2 * std_20
            features["bb_middle"] = sma_20
            features["bb_lower"] = sma_20 - 2 * std_20
            features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / sma_20

        # ATR
        if len(closes) >= 15 and len(highs) >= 15:
            tr = np.maximum(
                highs[-14:] - lows[-14:],
                np.maximum(
                    np.abs(highs[-14:] - closes[-15:-1]),
                    np.abs(lows[-14:] - closes[-15:-1])
                )
            )
            features["atr_14"] = np.mean(tr)

        return features

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1]

        multiplier = 2 / (period + 1)
        ema = data[0]

        for price in data[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def get_feature_schema(
        self,
        feature_set: str = "default",
    ) -> dict[str, Any]:
        """
        Get schema for a feature set.

        Returns:
            Dictionary with feature metadata
        """
        fs = self._feature_sets.get(feature_set)
        if not fs:
            return {}

        return {
            "name": fs.name,
            "version": fs.version,
            "features": [
                {
                    "name": f.name,
                    "version": f.version,
                    "lookback": f.lookback,
                    "dependencies": f.dependencies,
                    "description": f.description,
                }
                for f in fs.features
            ],
            "max_lookback": fs.max_lookback,
            "created_at": fs.created_at.isoformat(),
        }

    def validate_features(
        self,
        features: dict[str, float],
        feature_set: str = "default",
    ) -> tuple[bool, list[str]]:
        """
        Validate that features match expected schema.

        Args:
            features: Feature dictionary
            feature_set: Feature set name

        Returns:
            Tuple of (is_valid, missing_features)
        """
        fs = self._feature_sets.get(feature_set)
        if not fs:
            return False, ["Feature set not found"]

        expected = set(f.name for f in fs.features)
        provided = set(features.keys())

        missing = expected - provided
        extra = provided - expected

        is_valid = len(missing) == 0

        issues = []
        if missing:
            issues.append(f"Missing: {list(missing)}")
        if extra:
            issues.append(f"Extra: {list(extra)}")

        return is_valid, issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FeatureDefinition",
    "FeatureSet",
    "FeatureStore",
]
