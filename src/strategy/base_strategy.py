"""
Base Strategy Framework
JPMorgan-Level Strategy Architecture

Features:
- Abstract strategy interface
- Signal generation
- Position sizing interface
- Risk integration
- Performance tracking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from ..utils.logger import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class SignalType(Enum):
    """Trading signal types"""
    LONG = 1
    SHORT = -1
    FLAT = 0
    LONG_ENTRY = 2
    SHORT_ENTRY = -2
    LONG_EXIT = 3
    SHORT_EXIT = -3


@dataclass
class Signal:
    """Trading signal with metadata"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0 to 1
    timestamp: datetime
    price: float
    strategy_name: str
    confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        """Get signal direction (-1, 0, 1)"""
        if self.signal_type in [SignalType.LONG, SignalType.LONG_ENTRY]:
            return 1
        elif self.signal_type in [SignalType.SHORT, SignalType.SHORT_ENTRY]:
            return -1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.name,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'strategy_name': self.strategy_name,
            'confidence': self.confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str = "base_strategy"
    symbols: List[str] = field(default_factory=list)
    lookback_period: int = 20
    min_signal_strength: float = 0.5
    max_positions: int = 10
    position_size_method: str = "equal"  # equal, volatility, kelly
    rebalance_frequency: str = "daily"
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    use_take_profit: bool = True
    take_profit_pct: float = 0.04


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Provides common interface for:
    - Signal generation
    - Position sizing
    - Risk management integration
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize BaseStrategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()
        self.name = self.config.name

        self._signals: List[Signal] = []
        self._positions: Dict[str, float] = {}
        self._performance_history: List[Dict[str, Any]] = []

        logger.info(f"Strategy initialized: {self.name}")

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """
        Generate trading signals for all symbols.

        Args:
            data: Dictionary of OHLCV DataFrames by symbol

        Returns:
            Dictionary of signals by symbol
        """
        pass

    @abstractmethod
    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """
        Calculate signal type and strength for a symbol.

        Args:
            df: OHLCV DataFrame with indicators
            symbol: Stock symbol

        Returns:
            Tuple of (SignalType, strength)
        """
        pass

    def get_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on signal and config.

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_price: Current asset price
            volatility: Asset volatility (for vol-based sizing)

        Returns:
            Number of shares/contracts
        """
        if self.config.position_size_method == "equal":
            # Equal weight
            max_position_value = portfolio_value / self.config.max_positions
            shares = max_position_value / current_price

        elif self.config.position_size_method == "volatility":
            # Inverse volatility weighting
            if volatility is None or volatility == 0:
                volatility = 0.02  # Default 2% daily vol

            target_risk = portfolio_value * 0.01  # 1% portfolio risk
            shares = target_risk / (current_price * volatility)

        elif self.config.position_size_method == "kelly":
            # Kelly criterion (simplified)
            win_rate = signal.confidence
            win_loss_ratio = self.config.take_profit_pct / self.config.stop_loss_pct

            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%

            position_value = portfolio_value * kelly_fraction
            shares = position_value / current_price

        else:
            # Default to equal weight
            max_position_value = portfolio_value / self.config.max_positions
            shares = max_position_value / current_price

        # Apply signal strength scaling
        shares *= signal.strength

        return int(shares)

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate if signal meets strategy requirements.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        # Check minimum strength
        if signal.strength < self.config.min_signal_strength:
            return False

        # Check if symbol is in universe
        if self.config.symbols and signal.symbol not in self.config.symbols:
            return False

        # Check confidence threshold
        if signal.confidence < 0.5:
            return False

        return True

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            volatility: Asset volatility for ATR-based stops

        Returns:
            Stop loss price
        """
        if volatility:
            # ATR-based stop
            stop_distance = volatility * 2  # 2x volatility
        else:
            # Fixed percentage stop
            stop_distance = entry_price * self.config.stop_loss_pct

        if direction == 1:  # Long
            return entry_price - stop_distance
        else:  # Short
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int,
        stop_loss: Optional[float] = None
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            stop_loss: Stop loss price for R:R calculation

        Returns:
            Take profit price
        """
        if stop_loss:
            # Risk:Reward based
            risk = abs(entry_price - stop_loss)
            reward = risk * 2  # 2:1 reward:risk
        else:
            # Fixed percentage
            reward = entry_price * self.config.take_profit_pct

        if direction == 1:  # Long
            return entry_price + reward
        else:  # Short
            return entry_price - reward

    def update_positions(self, positions: Dict[str, float]) -> None:
        """Update current positions"""
        self._positions = positions.copy()

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self._positions.copy()

    def record_signal(self, signal: Signal) -> None:
        """Record signal for tracking"""
        self._signals.append(signal)

        audit_logger.log_system_event(
            "SIGNAL_GENERATED",
            signal.to_dict()
        )

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Signal]:
        """Get signal history"""
        signals = self._signals

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return signals[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        if not self._signals:
            return {}

        signals_df = pd.DataFrame([s.to_dict() for s in self._signals])

        return {
            'total_signals': len(self._signals),
            'long_signals': len([s for s in self._signals if s.direction == 1]),
            'short_signals': len([s for s in self._signals if s.direction == -1]),
            'avg_strength': signals_df['strength'].mean(),
            'avg_confidence': signals_df['confidence'].mean()
        }

    def reset(self) -> None:
        """Reset strategy state"""
        self._signals.clear()
        self._positions.clear()
        self._performance_history.clear()


class MultiStrategy:
    """
    Combines multiple strategies with weighted signals.

    Features:
    - Strategy weighting
    - Signal aggregation
    - Conflict resolution
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize MultiStrategy.

        Args:
            strategies: List of strategies
            weights: Weight for each strategy
        """
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """
        Generate aggregated signals from all strategies.

        Args:
            data: Dictionary of OHLCV DataFrames

        Returns:
            Aggregated signals by symbol
        """
        all_signals: Dict[str, List[Tuple[Signal, float]]] = {}

        # Collect signals from all strategies
        for strategy, weight in zip(self.strategies, self.weights):
            strategy_signals = strategy.generate_signals(data)

            for symbol, signal in strategy_signals.items():
                if symbol not in all_signals:
                    all_signals[symbol] = []
                all_signals[symbol].append((signal, weight))

        # Aggregate signals
        aggregated = {}

        for symbol, signal_list in all_signals.items():
            aggregated[symbol] = self._aggregate_signals(symbol, signal_list)

        return aggregated

    def _aggregate_signals(
        self,
        symbol: str,
        signals: List[Tuple[Signal, float]]
    ) -> Signal:
        """Aggregate multiple signals into one"""
        if not signals:
            return None

        # Weighted average of directions and strengths
        weighted_direction = sum(
            s.direction * s.strength * w for s, w in signals
        )
        weighted_strength = sum(
            s.strength * w for s, w in signals
        )
        weighted_confidence = sum(
            s.confidence * w for s, w in signals
        )

        # Determine final signal type
        if weighted_direction > 0.3:
            signal_type = SignalType.LONG
        elif weighted_direction < -0.3:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT

        # Use latest timestamp and price
        latest_signal = max(signals, key=lambda x: x[0].timestamp)[0]

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=min(abs(weighted_strength), 1.0),
            timestamp=latest_signal.timestamp,
            price=latest_signal.price,
            strategy_name="multi_strategy",
            confidence=weighted_confidence,
            stop_loss=latest_signal.stop_loss,
            take_profit=latest_signal.take_profit,
            metadata={
                'component_strategies': [s.strategy_name for s, _ in signals],
                'weighted_direction': weighted_direction
            }
        )
