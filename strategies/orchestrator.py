"""
STRATEGY ORCHESTRATOR
Enterprise Multi-Strategy Management System
JPMorgan Quantitative Trading Division Style

This module manages multiple trading strategies:
- Strategy lifecycle management (add, remove, enable, disable)
- Signal aggregation and voting
- Portfolio allocation across strategies
- Performance tracking per strategy
- Dynamic weight adjustment
- Risk-adjusted signal fusion

Usage:
    orchestrator = StrategyOrchestrator(portfolio_manager, risk_manager)
    orchestrator.add_strategy(MomentumStrategy(), weight=0.4)
    orchestrator.add_strategy(MeanReversionStrategy(), weight=0.3)
    orchestrator.add_strategy(PairsTradingStrategy(), weight=0.3)
    
    signal = await orchestrator.process_tick(tick)
"""

import asyncio
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

from strategies.base import BaseStrategy
from data.models import MarketTick, TradeSignal, Side, AlphaSignal


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class AggregationMethod(Enum):
    """Signal aggregation methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS = "unanimous"
    HIGHEST_CONFIDENCE = "highest_confidence"
    RISK_ADJUSTED = "risk_adjusted"


class StrategyStatus(Enum):
    """Strategy status"""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    COOLDOWN = "cooldown"
    ERROR = "error"


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    weight: float = 1.0
    max_allocation_pct: float = 100.0  # Max % of portfolio
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)  # Empty = all symbols
    cooldown_minutes: int = 0  # Cooldown after signal
    max_daily_signals: int = 100
    min_confidence: float = 0.5


@dataclass
class StrategyState:
    """Runtime state of a strategy"""
    status: StrategyStatus = StrategyStatus.ACTIVE
    signals_today: int = 0
    last_signal_time: Optional[datetime] = None
    winning_signals: int = 0
    losing_signals: int = 0
    total_pnl: float = 0.0
    cooldown_until: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    name: str
    total_signals: int = 0
    profitable_signals: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_confidence: float = 0.0
    contribution_pct: float = 0.0  # Contribution to portfolio P&L


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple strategies"""
    symbol: str
    side: Side
    confidence: float
    quantity: float
    timestamp: datetime
    
    # Aggregation details
    contributing_strategies: List[str] = field(default_factory=list)
    strategy_signals: Dict[str, Tuple[Side, float]] = field(default_factory=dict)
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    agreement_score: float = 0.0  # How much strategies agree
    
    def to_trade_signal(self) -> TradeSignal:
        """Convert to TradeSignal"""
        return TradeSignal(
            symbol=self.symbol,
            side=self.side,
            price=0.0,  # Will be set by executor
            quantity=self.quantity,
            strategy_name=f"Orchestrator[{','.join(self.contributing_strategies)}]",
            timestamp=self.timestamp,
            confidence=self.confidence,
            metadata={
                'aggregation_method': self.aggregation_method.value,
                'agreement_score': self.agreement_score,
                'strategies': self.strategy_signals
            }
        )


# ============================================================================
# STRATEGY WRAPPER
# ============================================================================

class StrategyWrapper:
    """Wrapper around a strategy with state management"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: StrategyConfig
    ):
        self.strategy = strategy
        self.config = config
        self.state = StrategyState()
        self.performance = StrategyPerformance(name=config.name)
        
        # Signal history
        self.signal_history: deque = deque(maxlen=1000)
        self.returns_history: deque = deque(maxlen=1000)
        
        # Current allocation
        self.current_allocation: float = 0.0
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def is_active(self) -> bool:
        """Check if strategy is active and can generate signals"""
        if not self.config.enabled:
            return False
        if self.state.status != StrategyStatus.ACTIVE:
            return False
        if self.state.cooldown_until and datetime.now() < self.state.cooldown_until:
            return False
        if self.state.signals_today >= self.config.max_daily_signals:
            return False
        return True
    
    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if strategy is allowed to trade symbol"""
        if not self.config.symbols:  # Empty = all symbols
            return True
        return symbol.upper() in [s.upper() for s in self.config.symbols]
    
    async def generate_signal(self, tick: MarketTick) -> Optional[TradeSignal]:
        """Generate signal from wrapped strategy"""
        if not self.is_active:
            return None
        
        if not self.can_trade_symbol(tick.symbol):
            return None
        
        try:
            signal = await self.strategy.on_tick(tick)
            
            if signal:
                # Check minimum confidence
                if hasattr(signal, 'confidence') and signal.confidence < self.config.min_confidence:
                    return None
                
                # Update state
                self.state.signals_today += 1
                self.state.last_signal_time = datetime.now()
                
                # Apply cooldown if configured
                if self.config.cooldown_minutes > 0:
                    self.state.cooldown_until = datetime.now() + timedelta(
                        minutes=self.config.cooldown_minutes
                    )
                
                # Record signal
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'price': signal.price,
                    'confidence': getattr(signal, 'confidence', 0.5)
                })
                
                self.performance.total_signals += 1
            
            return signal
            
        except Exception as e:
            self.state.error_count += 1
            self.state.last_error = str(e)
            if self.state.error_count >= 5:
                self.state.status = StrategyStatus.ERROR
            return None
    
    def record_trade_result(self, pnl: float, pnl_pct: float):
        """Record trade result for performance tracking"""
        self.returns_history.append(pnl_pct)
        self.state.total_pnl += pnl
        
        if pnl > 0:
            self.state.winning_signals += 1
            self.performance.profitable_signals += 1
        else:
            self.state.losing_signals += 1
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if self.performance.total_signals > 0:
            self.performance.win_rate = (
                self.performance.profitable_signals / self.performance.total_signals
            )
        
        if len(self.returns_history) > 0:
            returns = np.array(self.returns_history)
            self.performance.avg_return = float(np.mean(returns))
            
            if len(returns) > 1 and np.std(returns) > 0:
                self.performance.sharpe_ratio = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )
    
    def reset_daily_counters(self):
        """Reset daily counters"""
        self.state.signals_today = 0
        self.state.cooldown_until = None
    
    def pause(self):
        """Pause strategy"""
        self.state.status = StrategyStatus.PAUSED
    
    def resume(self):
        """Resume strategy"""
        self.state.status = StrategyStatus.ACTIVE
        self.state.error_count = 0


# ============================================================================
# SIGNAL AGGREGATOR
# ============================================================================

class SignalAggregator:
    """Aggregates signals from multiple strategies"""
    
    def __init__(self, method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE):
        self.method = method
    
    def aggregate(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]],
        tick: MarketTick
    ) -> Optional[AggregatedSignal]:
        """
        Aggregate signals from multiple strategies.
        
        Args:
            signals: List of (wrapper, signal) tuples
            tick: Current market tick
        
        Returns:
            Aggregated signal or None
        """
        if not signals:
            return None
        
        if self.method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(signals, tick)
        elif self.method == AggregationMethod.MAJORITY_VOTE:
            return self._majority_vote(signals, tick)
        elif self.method == AggregationMethod.UNANIMOUS:
            return self._unanimous(signals, tick)
        elif self.method == AggregationMethod.HIGHEST_CONFIDENCE:
            return self._highest_confidence(signals, tick)
        elif self.method == AggregationMethod.RISK_ADJUSTED:
            return self._risk_adjusted(signals, tick)
        
        return None
    
    def _weighted_average(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]],
        tick: MarketTick
    ) -> Optional[AggregatedSignal]:
        """Weighted average of signal directions"""
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        weighted_quantity = 0.0
        
        strategy_signals = {}
        contributing = []
        
        for wrapper, signal in signals:
            weight = wrapper.config.weight
            confidence = getattr(signal, 'confidence', 0.5)
            
            # Direction: +1 for BUY, -1 for SELL, 0 for HOLD
            if signal.side == Side.BUY:
                direction = 1.0
            elif signal.side == Side.SELL:
                direction = -1.0
            else:
                direction = 0.0
            
            weighted_direction += direction * weight * confidence
            weighted_confidence += confidence * weight
            weighted_quantity += signal.quantity * weight
            total_weight += weight
            
            strategy_signals[wrapper.name] = (signal.side, confidence)
            contributing.append(wrapper.name)
        
        if total_weight == 0:
            return None
        
        # Normalize
        avg_direction = weighted_direction / total_weight
        avg_confidence = weighted_confidence / total_weight
        avg_quantity = weighted_quantity / total_weight
        
        # Determine final side
        if avg_direction > 0.2:
            final_side = Side.BUY
        elif avg_direction < -0.2:
            final_side = Side.SELL
        else:
            final_side = Side.HOLD
        
        # Calculate agreement score
        agreement = self._calculate_agreement(signals)
        
        return AggregatedSignal(
            symbol=tick.symbol,
            side=final_side,
            confidence=avg_confidence * agreement,
            quantity=avg_quantity,
            timestamp=datetime.now(),
            contributing_strategies=contributing,
            strategy_signals=strategy_signals,
            aggregation_method=self.method,
            agreement_score=agreement
        )
    
    def _majority_vote(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]],
        tick: MarketTick
    ) -> Optional[AggregatedSignal]:
        """Majority voting - most common direction wins"""
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = {'BUY': [], 'SELL': [], 'HOLD': []}
        quantities = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        strategy_signals = {}
        contributing = []
        
        for wrapper, signal in signals:
            votes[signal.side.value] += wrapper.config.weight
            confidence = getattr(signal, 'confidence', 0.5)
            confidences[signal.side.value].append(confidence)
            quantities[signal.side.value].append(signal.quantity)
            strategy_signals[wrapper.name] = (signal.side, confidence)
            contributing.append(wrapper.name)
        
        # Find winner
        winner = max(votes, key=votes.get)
        final_side = Side(winner)
        
        if final_side == Side.HOLD or not confidences[winner]:
            return None
        
        avg_confidence = np.mean(confidences[winner])
        avg_quantity = np.mean(quantities[winner])
        agreement = votes[winner] / sum(votes.values()) if sum(votes.values()) > 0 else 0
        
        return AggregatedSignal(
            symbol=tick.symbol,
            side=final_side,
            confidence=avg_confidence * agreement,
            quantity=avg_quantity,
            timestamp=datetime.now(),
            contributing_strategies=contributing,
            strategy_signals=strategy_signals,
            aggregation_method=self.method,
            agreement_score=agreement
        )
    
    def _unanimous(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]],
        tick: MarketTick
    ) -> Optional[AggregatedSignal]:
        """All strategies must agree"""
        if len(signals) < 2:
            return None
        
        sides = set(sig.side for _, sig in signals)
        
        # Must all be same non-HOLD side
        if len(sides) != 1 or Side.HOLD in sides:
            return None
        
        final_side = list(sides)[0]
        
        strategy_signals = {}
        contributing = []
        confidences = []
        quantities = []
        
        for wrapper, signal in signals:
            confidence = getattr(signal, 'confidence', 0.5)
            strategy_signals[wrapper.name] = (signal.side, confidence)
            contributing.append(wrapper.name)
            confidences.append(confidence)
            quantities.append(signal.quantity)
        
        return AggregatedSignal(
            symbol=tick.symbol,
            side=final_side,
            confidence=np.mean(confidences),
            quantity=np.mean(quantities),
            timestamp=datetime.now(),
            contributing_strategies=contributing,
            strategy_signals=strategy_signals,
            aggregation_method=self.method,
            agreement_score=1.0  # Unanimous
        )
    
    def _highest_confidence(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]],
        tick: MarketTick
    ) -> Optional[AggregatedSignal]:
        """Take signal with highest confidence"""
        best_signal = None
        best_wrapper = None
        best_confidence = 0.0
        
        strategy_signals = {}
        contributing = []
        
        for wrapper, signal in signals:
            confidence = getattr(signal, 'confidence', 0.5)
            strategy_signals[wrapper.name] = (signal.side, confidence)
            contributing.append(wrapper.name)
            
            if confidence > best_confidence and signal.side != Side.HOLD:
                best_confidence = confidence
                best_signal = signal
                best_wrapper = wrapper
        
        if best_signal is None:
            return None
        
        agreement = self._calculate_agreement(signals)
        
        return AggregatedSignal(
            symbol=tick.symbol,
            side=best_signal.side,
            confidence=best_confidence * agreement,
            quantity=best_signal.quantity,
            timestamp=datetime.now(),
            contributing_strategies=[best_wrapper.name],
            strategy_signals=strategy_signals,
            aggregation_method=self.method,
            agreement_score=agreement
        )
    
    def _risk_adjusted(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]],
        tick: MarketTick
    ) -> Optional[AggregatedSignal]:
        """Weight by Sharpe ratio / performance"""
        total_score = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        weighted_quantity = 0.0
        
        strategy_signals = {}
        contributing = []
        
        for wrapper, signal in signals:
            # Use Sharpe ratio as weight (with minimum)
            sharpe = max(0.1, wrapper.performance.sharpe_ratio + 1)  # Shift to positive
            score = wrapper.config.weight * sharpe
            
            confidence = getattr(signal, 'confidence', 0.5)
            
            if signal.side == Side.BUY:
                direction = 1.0
            elif signal.side == Side.SELL:
                direction = -1.0
            else:
                direction = 0.0
            
            weighted_direction += direction * score * confidence
            weighted_confidence += confidence * score
            weighted_quantity += signal.quantity * score
            total_score += score
            
            strategy_signals[wrapper.name] = (signal.side, confidence)
            contributing.append(wrapper.name)
        
        if total_score == 0:
            return None
        
        avg_direction = weighted_direction / total_score
        avg_confidence = weighted_confidence / total_score
        avg_quantity = weighted_quantity / total_score
        
        if avg_direction > 0.2:
            final_side = Side.BUY
        elif avg_direction < -0.2:
            final_side = Side.SELL
        else:
            final_side = Side.HOLD
        
        agreement = self._calculate_agreement(signals)
        
        return AggregatedSignal(
            symbol=tick.symbol,
            side=final_side,
            confidence=avg_confidence * agreement,
            quantity=avg_quantity,
            timestamp=datetime.now(),
            contributing_strategies=contributing,
            strategy_signals=strategy_signals,
            aggregation_method=self.method,
            agreement_score=agreement
        )
    
    def _calculate_agreement(
        self,
        signals: List[Tuple[StrategyWrapper, TradeSignal]]
    ) -> float:
        """Calculate agreement score (0-1)"""
        if len(signals) <= 1:
            return 1.0
        
        buy_count = sum(1 for _, s in signals if s.side == Side.BUY)
        sell_count = sum(1 for _, s in signals if s.side == Side.SELL)
        
        max_agreement = max(buy_count, sell_count)
        total = len(signals)
        
        return max_agreement / total if total > 0 else 0.0


# ============================================================================
# STRATEGY ORCHESTRATOR
# ============================================================================

class StrategyOrchestrator:
    """
    Manages multiple trading strategies.
    
    Features:
    - Strategy lifecycle management
    - Signal aggregation
    - Dynamic weight adjustment
    - Performance tracking
    """
    
    def __init__(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        min_agreement: float = 0.5,
        min_confidence: float = 0.5,
        max_strategies: int = 10
    ):
        """
        Args:
            aggregation_method: How to aggregate signals
            min_agreement: Minimum agreement score to act
            min_confidence: Minimum confidence to act
            max_strategies: Maximum number of strategies
        """
        self.strategies: Dict[str, StrategyWrapper] = {}
        self.aggregator = SignalAggregator(method=aggregation_method)
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self.max_strategies = max_strategies
        
        # State
        self.is_running = False
        self.last_daily_reset = datetime.now().date()
        
        # History
        self.aggregated_signals: deque = deque(maxlen=1000)
        self.tick_count = 0
        
        # Logging
        self.logger = logging.getLogger('StrategyOrchestrator')
    
    def add_strategy(
        self,
        strategy: BaseStrategy,
        weight: float = 1.0,
        max_allocation_pct: float = 100.0,
        symbols: Optional[List[str]] = None,
        cooldown_minutes: int = 0,
        max_daily_signals: int = 100,
        min_confidence: float = 0.5,
        enabled: bool = True
    ) -> bool:
        """
        Add a strategy to the orchestrator.
        
        Args:
            strategy: Strategy instance
            weight: Strategy weight for signal aggregation
            max_allocation_pct: Max portfolio allocation %
            symbols: List of symbols to trade (empty = all)
            cooldown_minutes: Minutes to wait between signals
            max_daily_signals: Max signals per day
            min_confidence: Minimum signal confidence
            enabled: Is strategy active?
        
        Returns:
            True if added successfully
        """
        if len(self.strategies) >= self.max_strategies:
            self.logger.warning(f"Max strategies ({self.max_strategies}) reached")
            return False
        
        name = strategy.name
        if name in self.strategies:
            self.logger.warning(f"Strategy {name} already exists")
            return False
        
        config = StrategyConfig(
            name=name,
            weight=weight,
            max_allocation_pct=max_allocation_pct,
            enabled=enabled,
            symbols=symbols or [],
            cooldown_minutes=cooldown_minutes,
            max_daily_signals=max_daily_signals,
            min_confidence=min_confidence
        )
        
        wrapper = StrategyWrapper(strategy, config)
        self.strategies[name] = wrapper
        
        self.logger.info(f"✅ Added strategy: {name} (weight={weight})")
        return True
    
    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy"""
        if name in self.strategies:
            del self.strategies[name]
            self.logger.info(f"🗑️ Removed strategy: {name}")
            return True
        return False
    
    def enable_strategy(self, name: str) -> bool:
        """Enable a strategy"""
        if name in self.strategies:
            self.strategies[name].config.enabled = True
            self.strategies[name].resume()
            return True
        return False
    
    def disable_strategy(self, name: str) -> bool:
        """Disable a strategy"""
        if name in self.strategies:
            self.strategies[name].config.enabled = False
            self.strategies[name].pause()
            return True
        return False
    
    def set_weight(self, name: str, weight: float) -> bool:
        """Update strategy weight"""
        if name in self.strategies:
            self.strategies[name].config.weight = weight
            return True
        return False
    
    async def process_tick(self, tick: MarketTick) -> Optional[TradeSignal]:
        """
        Process a market tick through all strategies.
        
        Args:
            tick: Market data tick
        
        Returns:
            Aggregated TradeSignal or None
        """
        self.tick_count += 1
        
        # Daily reset check
        self._check_daily_reset()
        
        # Collect signals from all active strategies
        signals: List[Tuple[StrategyWrapper, TradeSignal]] = []
        
        for wrapper in self.strategies.values():
            if not wrapper.is_active:
                continue
            
            signal = await wrapper.generate_signal(tick)
            if signal and signal.side != Side.HOLD:
                signals.append((wrapper, signal))
        
        if not signals:
            return None
        
        # Aggregate signals
        aggregated = self.aggregator.aggregate(signals, tick)
        
        if aggregated is None:
            return None
        
        # Check thresholds
        if aggregated.agreement_score < self.min_agreement:
            return None
        
        if aggregated.confidence < self.min_confidence:
            return None
        
        if aggregated.side == Side.HOLD:
            return None
        
        # Record
        self.aggregated_signals.append(aggregated)
        
        # Convert to TradeSignal
        trade_signal = aggregated.to_trade_signal()
        trade_signal.price = tick.price
        
        return trade_signal
    
    def record_trade_result(
        self,
        strategy_names: List[str],
        pnl: float,
        pnl_pct: float
    ):
        """Record trade result for contributing strategies"""
        # Split P&L among contributing strategies
        if not strategy_names:
            return
        
        pnl_per_strategy = pnl / len(strategy_names)
        pnl_pct_per_strategy = pnl_pct / len(strategy_names)
        
        for name in strategy_names:
            if name in self.strategies:
                self.strategies[name].record_trade_result(
                    pnl_per_strategy,
                    pnl_pct_per_strategy
                )
    
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get statistics for all strategies"""
        stats = {}
        for name, wrapper in self.strategies.items():
            stats[name] = {
                'status': wrapper.state.status.value,
                'enabled': wrapper.config.enabled,
                'weight': wrapper.config.weight,
                'signals_today': wrapper.state.signals_today,
                'total_signals': wrapper.performance.total_signals,
                'win_rate': wrapper.performance.win_rate,
                'sharpe_ratio': wrapper.performance.sharpe_ratio,
                'total_pnl': wrapper.state.total_pnl,
                'is_active': wrapper.is_active
            }
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        total_signals = sum(w.performance.total_signals for w in self.strategies.values())
        total_pnl = sum(w.state.total_pnl for w in self.strategies.values())
        
        active_count = sum(1 for w in self.strategies.values() if w.is_active)
        
        # Calculate weighted average metrics
        total_weight = sum(w.config.weight for w in self.strategies.values() if w.is_active)
        
        if total_weight > 0:
            weighted_win_rate = sum(
                w.performance.win_rate * w.config.weight
                for w in self.strategies.values() if w.is_active
            ) / total_weight
            
            weighted_sharpe = sum(
                w.performance.sharpe_ratio * w.config.weight
                for w in self.strategies.values() if w.is_active
            ) / total_weight
        else:
            weighted_win_rate = 0.0
            weighted_sharpe = 0.0
        
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': active_count,
            'total_signals': total_signals,
            'total_pnl': total_pnl,
            'weighted_win_rate': weighted_win_rate,
            'weighted_sharpe': weighted_sharpe,
            'tick_count': self.tick_count,
            'aggregated_signals_count': len(self.aggregated_signals)
        }
    
    def _check_daily_reset(self):
        """Check and perform daily reset if needed"""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            for wrapper in self.strategies.values():
                wrapper.reset_daily_counters()
            self.last_daily_reset = today
            self.logger.info("🔄 Daily counters reset")
    
    def normalize_weights(self):
        """Normalize strategy weights to sum to 1"""
        total = sum(w.config.weight for w in self.strategies.values())
        if total > 0:
            for wrapper in self.strategies.values():
                wrapper.config.weight /= total
    
    def auto_adjust_weights(self, lookback_days: int = 30):
        """
        Auto-adjust weights based on recent performance.
        Better performing strategies get higher weights.
        """
        performances = []
        for wrapper in self.strategies.values():
            # Use Sharpe ratio as performance metric
            perf = max(0.1, wrapper.performance.sharpe_ratio + 1)
            performances.append((wrapper, perf))
        
        if not performances:
            return
        
        total_perf = sum(p for _, p in performances)
        
        for wrapper, perf in performances:
            wrapper.config.weight = perf / total_perf
        
        self.logger.info("📊 Weights auto-adjusted based on performance")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_orchestrator(
    aggregation_method: str = "weighted_average",
    min_agreement: float = 0.5,
    min_confidence: float = 0.5
) -> StrategyOrchestrator:
    """Create a strategy orchestrator"""
    method = AggregationMethod(aggregation_method)
    return StrategyOrchestrator(
        aggregation_method=method,
        min_agreement=min_agreement,
        min_confidence=min_confidence
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StrategyOrchestrator',
    'StrategyWrapper',
    'StrategyConfig',
    'StrategyState',
    'StrategyPerformance',
    'SignalAggregator',
    'AggregationMethod',
    'StrategyStatus',
    'AggregatedSignal',
    'create_orchestrator'
]