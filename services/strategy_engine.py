"""
Strategy Engine Service
=======================

Listens to market data events, runs ML inference, and generates
trading signals. Publishes signals to the message bus for risk
validation.

Responsibilities:
- Subscribe to market data
- Feature calculation
- ML model inference
- Signal generation
- Publishing to risk engine

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from pathlib import Path
import os

import numpy as np
import polars as pl

from config.settings import get_logger
from infrastructure.message_bus import (
    Message,
    MessageType,
    Channel,
    MessagePriority,
)
from infrastructure.service_registry import ServiceType
from services.base_service import BaseService, ServiceConfig

logger = get_logger(__name__)


@dataclass
class StrategyConfig(ServiceConfig):
    """Configuration for strategy engine service."""
    name: str = "strategy_engine"
    service_type: ServiceType = ServiceType.STRATEGY_ENGINE
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    model_path: str = "models/artifacts"
    min_bars_required: int = 50  # Minimum bars for feature calculation
    signal_threshold: float = 0.6  # Minimum confidence for signals
    max_signals_per_minute: int = 10  # Rate limiting
    feature_window: int = 20  # Rolling window for features


class StrategyService(BaseService):
    """
    Strategy Engine Service for ML-based signal generation.

    Listens to market data, calculates features, runs inference,
    and publishes trading signals for risk validation.

    Example:
        config = StrategyConfig(
            symbols=["AAPL", "MSFT"],
            model_path="models/artifacts"
        )

        service = StrategyService(config)
        await service.run_forever()
    """

    def __init__(self, config: StrategyConfig | None = None):
        """Initialize strategy engine."""
        config = config or StrategyConfig()
        super().__init__(config)

        self.config: StrategyConfig = config

        # Data storage
        self._bar_data: dict[str, list[dict]] = {s: [] for s in config.symbols}
        self._feature_cache: dict[str, np.ndarray] = {}

        # Models
        self._models: dict[str, Any] = {}

        # Statistics
        self._signal_count = 0
        self._inference_count = 0
        self._signals_this_minute = 0
        self._last_minute = 0

    async def _on_start(self) -> None:
        """Start strategy engine."""
        logger.info(f"Starting strategy engine for {len(self.config.symbols)} symbols")

        # Load ML models
        await self._load_models()

        # Subscribe to market data
        await self.subscribe(Channel.MARKET_DATA, self._handle_market_data)

        # Start feature calculation loop
        self.add_background_task(self._feature_calculation_loop())

        logger.info("Strategy engine started")

    async def _on_stop(self) -> None:
        """Stop strategy engine."""
        logger.info(
            f"Strategy engine stopped. "
            f"Signals: {self._signal_count}, Inferences: {self._inference_count}"
        )

    async def _load_models(self) -> None:
        """Load ML models for each symbol."""
        model_path = Path(self.config.model_path)

        for symbol in self.config.symbols:
            try:
                # Try to load model
                model_file = model_path / f"{symbol}_lightgbm_v1.pkl"

                if model_file.exists():
                    import pickle
                    with open(model_file, "rb") as f:
                        model_data = pickle.load(f)
                        # Handle both direct model and wrapped model
                        if isinstance(model_data, dict) and "model" in model_data:
                            self._models[symbol] = model_data["model"]
                        else:
                            self._models[symbol] = model_data
                    logger.info(f"Loaded model for {symbol}")
                else:
                    logger.warning(f"No model found for {symbol}, using mock predictions")
                    self._models[symbol] = None

            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")
                self._models[symbol] = None

    async def _handle_market_data(self, message: Message) -> None:
        """Handle incoming market data."""
        try:
            payload = message.payload
            symbol = payload.get("symbol")

            if symbol not in self.config.symbols:
                return

            if payload.get("data_type") != "bar":
                return

            # Store bar data
            bar = {
                "timestamp": payload["timestamp"],
                "open": payload["open"],
                "high": payload["high"],
                "low": payload["low"],
                "close": payload["close"],
                "volume": payload["volume"],
            }

            self._bar_data[symbol].append(bar)

            # Keep only recent bars
            max_bars = self.config.min_bars_required * 2
            if len(self._bar_data[symbol]) > max_bars:
                self._bar_data[symbol] = self._bar_data[symbol][-max_bars:]

            # Check if we have enough data
            if len(self._bar_data[symbol]) >= self.config.min_bars_required:
                await self._generate_signal(symbol)

        except Exception as e:
            logger.error(f"Error handling market data: {e}")

    async def _generate_signal(self, symbol: str) -> None:
        """Generate trading signal for a symbol."""
        try:
            # Rate limiting
            current_minute = int(time.time() / 60)
            if current_minute != self._last_minute:
                self._signals_this_minute = 0
                self._last_minute = current_minute

            if self._signals_this_minute >= self.config.max_signals_per_minute:
                return

            # Calculate features
            features = self._calculate_features(symbol)
            if features is None:
                return

            # Run inference
            prediction, confidence = await self._run_inference(symbol, features)

            if confidence < self.config.signal_threshold:
                return

            # Get latest price
            latest_bar = self._bar_data[symbol][-1]
            price = latest_bar["close"]

            # Create signal message
            signal = Message(
                type=MessageType.SIGNAL,
                channel=Channel.SIGNALS,
                payload={
                    "symbol": symbol,
                    "direction": int(prediction),  # 1=long, -1=short
                    "strength": float(confidence),
                    "price": float(price),
                    "strategy": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "features": {
                        "rsi": float(features.get("rsi", 50)),
                        "macd": float(features.get("macd", 0)),
                        "momentum": float(features.get("momentum", 0)),
                    },
                },
                priority=MessagePriority.HIGH,
                source=self.name,
            )

            await self.publish(signal)

            self._signal_count += 1
            self._signals_this_minute += 1

            logger.info(
                f"Signal generated: {symbol} direction={prediction} "
                f"confidence={confidence:.2%} price={price:.2f}"
            )

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")

    def _calculate_features(self, symbol: str) -> dict[str, float] | None:
        """Calculate features for a symbol."""
        try:
            bars = self._bar_data[symbol]
            if len(bars) < self.config.min_bars_required:
                return None

            # Convert to numpy arrays
            closes = np.array([b["close"] for b in bars])
            highs = np.array([b["high"] for b in bars])
            lows = np.array([b["low"] for b in bars])
            volumes = np.array([b["volume"] for b in bars])

            # Calculate basic features
            features = {}

            # Returns
            returns = np.diff(closes) / closes[:-1]
            features["return_1"] = returns[-1] if len(returns) > 0 else 0

            # RSI
            gains = np.maximum(np.diff(closes), 0)
            losses = np.abs(np.minimum(np.diff(closes), 0))

            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features["rsi"] = 100 - (100 / (1 + rs))
            else:
                features["rsi"] = 100

            # MACD
            ema_12 = self._ema(closes, 12)
            ema_26 = self._ema(closes, 26)
            features["macd"] = ema_12 - ema_26

            # Momentum
            if len(closes) >= 10:
                features["momentum"] = (closes[-1] / closes[-10] - 1) * 100
            else:
                features["momentum"] = 0

            # Volatility
            features["volatility"] = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0

            # Volume ratio
            features["volume_ratio"] = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1

            # Price position in range
            high_20 = np.max(highs[-20:])
            low_20 = np.min(lows[-20:])
            if high_20 != low_20:
                features["price_position"] = (closes[-1] - low_20) / (high_20 - low_20)
            else:
                features["price_position"] = 0.5

            # SMA crossover
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
            features["sma_cross"] = 1 if sma_20 > sma_50 else -1

            self._feature_cache[symbol] = features
            return features

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1]

        multiplier = 2 / (period + 1)
        ema = data[0]

        for price in data[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    async def _run_inference(
        self,
        symbol: str,
        features: dict[str, float],
    ) -> tuple[int, float]:
        """
        Run ML model inference.

        Returns:
            Tuple of (direction, confidence)
            direction: 1 for long, -1 for short, 0 for neutral
            confidence: 0.0 to 1.0
        """
        self._inference_count += 1

        model = self._models.get(symbol)

        if model is None:
            # Mock prediction based on features
            return self._mock_prediction(features)

        try:
            # Prepare feature vector
            feature_names = sorted(features.keys())
            X = np.array([[features[f] for f in feature_names]])

            # Get prediction
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                if len(proba) == 2:
                    # Binary classification
                    confidence = max(proba)
                    direction = 1 if proba[1] > 0.5 else -1
                else:
                    # Multi-class
                    direction = np.argmax(proba) - 1  # Assuming -1, 0, 1
                    confidence = proba[np.argmax(proba)]
            else:
                pred = model.predict(X)[0]
                direction = int(np.sign(pred))
                confidence = min(abs(pred), 1.0)

            return direction, confidence

        except Exception as e:
            logger.error(f"Inference error for {symbol}: {e}")
            return self._mock_prediction(features)

    def _mock_prediction(self, features: dict[str, float]) -> tuple[int, float]:
        """Generate mock prediction based on features."""
        # Simple rule-based mock
        rsi = features.get("rsi", 50)
        macd = features.get("macd", 0)
        momentum = features.get("momentum", 0)

        score = 0

        # RSI
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1

        # MACD
        if macd > 0:
            score += 0.5
        else:
            score -= 0.5

        # Momentum
        if momentum > 0:
            score += 0.5
        else:
            score -= 0.5

        direction = 1 if score > 0 else -1 if score < 0 else 0
        confidence = min(abs(score) / 2, 1.0)

        return direction, confidence

    async def _feature_calculation_loop(self) -> None:
        """Periodic feature calculation for all symbols."""
        while self._running:
            try:
                for symbol in self.config.symbols:
                    if len(self._bar_data[symbol]) >= self.config.min_bars_required:
                        self._calculate_features(symbol)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Feature calculation error: {e}")
                await asyncio.sleep(1)

    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        status = super().get_status()
        status.update({
            "symbols": self.config.symbols,
            "signal_count": self._signal_count,
            "inference_count": self._inference_count,
            "models_loaded": list(k for k, v in self._models.items() if v is not None),
            "bars_stored": {s: len(bars) for s, bars in self._bar_data.items()},
        })
        return status


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Run strategy engine service."""
    config = StrategyConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        model_path=os.getenv("MODEL_PATH", "models/artifacts"),
    )

    service = StrategyService(config)
    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
