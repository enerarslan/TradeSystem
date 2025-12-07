#!/usr/bin/env python3
"""
Risk Management Integration Module
==================================

JPMorgan-level risk management integration for AlphaML Strategy.
Implements institutional-grade position sizing, stop-loss, and portfolio risk controls.

Key Components:
1. Position Sizing - Multiple methods (volatility, Kelly, risk parity)
2. Stop-Loss/Take-Profit - Dynamic based on ATR and volatility
3. Portfolio Risk - Correlation limits, concentration limits
4. Drawdown Protection - Circuit breakers, equity curve trading

This module bridges the gap between the strategy signals and actual order execution,
ensuring all trades comply with risk parameters.

Usage:
    from phase1.risk_integration import (
        RiskIntegrator,
        apply_risk_management,
    )
    
    integrator = RiskIntegrator(config)
    
    # Apply risk management to a signal
    sized_signal = integrator.process_signal(signal, portfolio)
    
    # Or batch process
    sized_signals = integrator.process_multiple(signals, portfolio)

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging
from core.types import Order, Trade, Position, PortfolioState, Signal, SignalStrength
from core.events import SignalEvent, RiskEvent, EventPriority
from risk.manager import (
    RiskManager as BaseRiskManager,
    RiskConfig,
    RiskMetrics,
    RiskLevel,
    PositionSizingMethod,
    calculate_position_size,
    kelly_criterion,
    optimal_f,
    calculate_var,
    calculate_cvar,
)

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class StopLossType(str, Enum):
    """Stop-loss types."""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    CHANDELIER = "chandelier"


class TakeProfitType(str, Enum):
    """Take-profit types."""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    RISK_REWARD = "risk_reward"  # Multiple of stop-loss distance
    TRAILING = "trailing"


class RiskAction(str, Enum):
    """Actions to take based on risk assessment."""
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE_SIZE = "reduce_size"
    DELAY = "delay"
    CLOSE_POSITION = "close_position"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskIntegrationConfig:
    """Configuration for risk integration."""
    # Position sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY
    max_position_size: float = 0.10  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% risk per trade
    
    # Stop-loss
    use_stop_loss: bool = True
    stop_loss_type: StopLossType = StopLossType.ATR_BASED
    stop_loss_pct: float = 0.02  # 2% fixed
    stop_loss_atr_mult: float = 2.0  # 2x ATR
    
    # Take-profit
    use_take_profit: bool = True
    take_profit_type: TakeProfitType = TakeProfitType.RISK_REWARD
    take_profit_pct: float = 0.04  # 4% fixed
    take_profit_rr_mult: float = 2.0  # 2:1 risk/reward
    
    # Trailing stops
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5%
    trailing_stop_atr_mult: float = 1.5
    trailing_activation_pct: float = 0.02  # Activate after 2% profit
    
    # Portfolio limits
    max_positions: int = 5
    max_sector_exposure: float = 0.30  # 30% in single sector
    max_correlation: float = 0.70  # Max correlation between positions
    max_correlated_positions: int = 3
    
    # Drawdown limits
    max_drawdown: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    weekly_loss_limit: float = 0.05  # 5% weekly loss limit
    
    # Circuit breakers
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 0.10  # 10% drawdown triggers
    circuit_breaker_cooldown_hours: int = 24
    
    # Volatility adjustments
    reduce_in_high_vol: bool = True
    high_vol_threshold: float = 1.5  # 1.5x normal vol
    high_vol_size_reduction: float = 0.5  # Reduce size by 50%
    
    # Kelly criterion settings
    kelly_fraction: float = 0.25  # Use quarter Kelly
    kelly_lookback_trades: int = 100
    
    # VaR settings
    var_confidence: float = 0.95
    var_lookback_days: int = 252
    max_var_exposure: float = 0.05  # 5% portfolio VaR limit


@dataclass
class RiskAssessment:
    """Result of risk assessment for a signal."""
    signal_id: str
    symbol: str
    
    # Decision
    action: RiskAction
    reason: str = ""
    
    # Position sizing
    original_size: float = 0.0
    adjusted_size: float = 0.0
    size_reduction_reason: str = ""
    
    # Stop/Take-profit levels
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trailing_stop_price: float = 0.0
    
    # Risk metrics
    position_risk_pct: float = 0.0  # % of portfolio at risk
    potential_loss: float = 0.0
    potential_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Portfolio impact
    new_position_count: int = 0
    new_portfolio_exposure: float = 0.0
    sector_exposure: float = 0.0
    correlation_score: float = 0.0
    
    # Market conditions
    current_volatility: float = 0.0
    volatility_regime: str = "normal"
    
    # Drawdown status
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    
    # Warnings
    warnings: list[str] = field(default_factory=list)
    
    def is_approved(self) -> bool:
        """Check if trade is approved."""
        return self.action == RiskAction.APPROVE
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PositionRiskProfile:
    """Risk profile for an active position."""
    symbol: str
    entry_price: float
    current_price: float
    shares: int
    
    # Stop-loss levels
    initial_stop: float = 0.0
    current_stop: float = 0.0  # May have trailed
    take_profit: float = 0.0
    
    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Risk
    risk_at_stop: float = 0.0
    risk_pct: float = 0.0
    
    # Trailing stop status
    trailing_activated: bool = False
    highest_price: float = 0.0
    
    # Time in trade
    bars_held: int = 0
    entry_timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# RISK INTEGRATOR
# =============================================================================

class RiskIntegrator:
    """
    Comprehensive risk management integration.
    
    Bridges strategy signals with execution, ensuring all trades
    comply with risk parameters and portfolio constraints.
    
    Example:
        integrator = RiskIntegrator(config)
        
        # Process a single signal
        assessment = integrator.process_signal(signal, portfolio, market_data)
        
        if assessment.is_approved():
            execute_order(
                signal,
                size=assessment.adjusted_size,
                stop_loss=assessment.stop_loss_price,
                take_profit=assessment.take_profit_price,
            )
        else:
            print(f"Signal rejected: {assessment.reason}")
    """
    
    def __init__(self, config: RiskIntegrationConfig | None = None):
        """Initialize risk integrator."""
        self.config = config or RiskIntegrationConfig()
        
        # Internal state
        self._position_profiles: dict[str, PositionRiskProfile] = {}
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_until: datetime | None = None
        
        # Tracking
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._session_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        
        # Trade history for Kelly
        self._trade_history: list[dict[str, Any]] = []
        
        # Correlation matrix (would be updated periodically)
        self._correlation_matrix: dict[tuple[str, str], float] = {}
        
        logger.info("RiskIntegrator initialized")
    
    def process_signal(
        self,
        signal: SignalEvent,
        portfolio: PortfolioState,
        market_data: dict[str, Any] | None = None,
    ) -> RiskAssessment:
        """
        Process a signal through risk management.
        
        Args:
            signal: Trading signal from strategy
            portfolio: Current portfolio state
            market_data: Optional market data (for volatility, ATR, etc.)
        
        Returns:
            RiskAssessment with decision and adjusted parameters
        """
        symbol = signal.symbol
        
        # Initialize assessment
        assessment = RiskAssessment(
            signal_id=str(uuid4())[:8],
            symbol=symbol,
            entry_price=signal.price,
            original_size=signal.target_size if hasattr(signal, 'target_size') else 0,
        )
        
        # Extract market data
        volatility = market_data.get("volatility", 0.02) if market_data else 0.02
        atr = market_data.get("atr", signal.price * 0.02) if market_data else signal.price * 0.02
        
        assessment.current_volatility = volatility
        
        # Check circuit breaker
        if self._check_circuit_breaker():
            assessment.action = RiskAction.REJECT
            assessment.reason = "Circuit breaker active"
            return assessment
        
        # Check portfolio limits
        limit_check = self._check_portfolio_limits(signal, portfolio)
        if not limit_check[0]:
            assessment.action = RiskAction.REJECT
            assessment.reason = limit_check[1]
            return assessment
        
        # Check drawdown limits
        drawdown_check = self._check_drawdown_limits(portfolio)
        if not drawdown_check[0]:
            assessment.action = RiskAction.REJECT
            assessment.reason = drawdown_check[1]
            return assessment
        
        # Calculate position size
        position_size = self._calculate_position_size(
            signal, portfolio, volatility, atr
        )
        
        # Apply volatility adjustment
        if self.config.reduce_in_high_vol:
            position_size = self._adjust_for_volatility(position_size, volatility)
            if position_size < assessment.original_size:
                assessment.size_reduction_reason = "High volatility reduction"
        
        assessment.adjusted_size = position_size
        
        # Calculate stop-loss
        stop_loss = self._calculate_stop_loss(signal.price, atr, volatility)
        assessment.stop_loss_price = stop_loss
        
        # Calculate take-profit
        take_profit = self._calculate_take_profit(
            signal.price, stop_loss, atr
        )
        assessment.take_profit_price = take_profit
        
        # Calculate risk metrics
        self._calculate_risk_metrics(assessment, portfolio, position_size)
        
        # Final approval
        if position_size <= 0:
            assessment.action = RiskAction.REJECT
            assessment.reason = "Position size calculated as zero"
        elif assessment.position_risk_pct > self.config.max_portfolio_risk * 1.5:
            assessment.action = RiskAction.REDUCE_SIZE
            assessment.adjusted_size *= 0.5
            assessment.reason = "Position risk exceeds limit"
        else:
            assessment.action = RiskAction.APPROVE
            assessment.reason = "All risk checks passed"
        
        # Add warnings
        self._add_warnings(assessment, portfolio)
        
        return assessment
    
    def process_multiple(
        self,
        signals: list[SignalEvent],
        portfolio: PortfolioState,
        market_data: dict[str, dict[str, Any]] | None = None,
    ) -> list[RiskAssessment]:
        """
        Process multiple signals with portfolio-level constraints.
        
        Args:
            signals: List of signals
            portfolio: Portfolio state
            market_data: Market data per symbol
        
        Returns:
            List of RiskAssessments
        """
        # Sort signals by confidence/priority
        sorted_signals = sorted(
            signals,
            key=lambda s: getattr(s, 'confidence', 0.5),
            reverse=True,
        )
        
        assessments = []
        approved_count = 0
        remaining_capacity = self.config.max_positions - len(portfolio.positions)
        
        for signal in sorted_signals:
            symbol_data = market_data.get(signal.symbol, {}) if market_data else {}
            assessment = self.process_signal(signal, portfolio, symbol_data)
            
            # Check portfolio-level capacity
            if assessment.is_approved():
                if approved_count >= remaining_capacity:
                    assessment.action = RiskAction.REJECT
                    assessment.reason = "Portfolio position limit reached"
                else:
                    approved_count += 1
            
            assessments.append(assessment)
        
        return assessments
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        portfolio: PortfolioState,
    ) -> tuple[bool, str, float | None]:
        """
        Update position tracking and check for exit signals.
        
        Args:
            symbol: Position symbol
            current_price: Current market price
            portfolio: Portfolio state
        
        Returns:
            Tuple of (should_exit, reason, exit_price)
        """
        if symbol not in self._position_profiles:
            return False, "", None
        
        profile = self._position_profiles[symbol]
        
        # Update P&L
        profile.current_price = current_price
        profile.unrealized_pnl = (current_price - profile.entry_price) * profile.shares
        profile.unrealized_pnl_pct = (current_price - profile.entry_price) / profile.entry_price
        
        # Update highest price for trailing stop
        if current_price > profile.highest_price:
            profile.highest_price = current_price
        
        # Check stop-loss
        if current_price <= profile.current_stop:
            return True, "Stop-loss triggered", profile.current_stop
        
        # Check take-profit
        if current_price >= profile.take_profit:
            return True, "Take-profit triggered", profile.take_profit
        
        # Update trailing stop
        if self.config.use_trailing_stop:
            new_trailing = self._update_trailing_stop(profile, current_price)
            if new_trailing > profile.current_stop:
                profile.current_stop = new_trailing
        
        profile.bars_held += 1
        
        return False, "", None
    
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Register a new position for tracking."""
        self._position_profiles[symbol] = PositionRiskProfile(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            shares=shares,
            initial_stop=stop_loss,
            current_stop=stop_loss,
            take_profit=take_profit,
            highest_price=entry_price,
            entry_timestamp=datetime.now(),
        )
    
    def close_position(self, symbol: str, exit_price: float, pnl: float) -> None:
        """Close a position and record result."""
        if symbol in self._position_profiles:
            profile = self._position_profiles.pop(symbol)
            
            # Record for Kelly calculation
            self._trade_history.append({
                "symbol": symbol,
                "entry_price": profile.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl / (profile.entry_price * profile.shares) if profile.shares > 0 else 0,
                "bars_held": profile.bars_held,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Update daily P&L
            self._daily_pnl += pnl
    
    def new_session(self, equity: float) -> None:
        """Start a new trading session."""
        self._session_start_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        self._daily_pnl = 0.0
        
        # Check for circuit breaker reset
        if self._circuit_breaker_until and datetime.now() > self._circuit_breaker_until:
            self._circuit_breaker_active = False
            self._circuit_breaker_until = None
            logger.info("Circuit breaker reset")
    
    def get_portfolio_risk_summary(
        self,
        portfolio: PortfolioState,
    ) -> dict[str, Any]:
        """Get comprehensive portfolio risk summary."""
        total_exposure = sum(
            abs(p.market_value) for p in portfolio.positions.values()
        ) if portfolio.positions else 0
        
        current_drawdown = self._calculate_drawdown(portfolio.equity)
        
        return {
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / portfolio.equity if portfolio.equity > 0 else 0,
            "position_count": len(portfolio.positions),
            "max_positions": self.config.max_positions,
            "current_drawdown": current_drawdown,
            "max_drawdown_limit": self.config.max_drawdown,
            "daily_pnl": self._daily_pnl,
            "daily_limit": self.config.daily_loss_limit * self._session_start_equity,
            "circuit_breaker_active": self._circuit_breaker_active,
            "positions": {
                symbol: {
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "current_stop": p.current_stop,
                    "take_profit": p.take_profit,
                    "bars_held": p.bars_held,
                }
                for symbol, p in self._position_profiles.items()
            },
        }
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is active."""
        if not self.config.circuit_breaker_enabled:
            return False
        
        if self._circuit_breaker_active:
            if self._circuit_breaker_until and datetime.now() < self._circuit_breaker_until:
                return True
            else:
                self._circuit_breaker_active = False
                self._circuit_breaker_until = None
        
        return False
    
    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        self._circuit_breaker_active = True
        self._circuit_breaker_until = datetime.now() + timedelta(
            hours=self.config.circuit_breaker_cooldown_hours
        )
        logger.warning(f"Circuit breaker triggered: {reason}")
    
    def _check_portfolio_limits(
        self,
        signal: SignalEvent,
        portfolio: PortfolioState,
    ) -> tuple[bool, str]:
        """Check portfolio-level limits."""
        # Check position count
        if len(portfolio.positions) >= self.config.max_positions:
            if signal.symbol not in portfolio.positions:
                return False, f"Maximum positions ({self.config.max_positions}) reached"
        
        # Check available cash
        min_trade_value = signal.price * 10  # At least 10 shares
        if portfolio.cash < min_trade_value:
            return False, "Insufficient cash for minimum trade"
        
        # Check correlation with existing positions
        if portfolio.positions:
            max_corr = self._get_max_correlation(signal.symbol, list(portfolio.positions.keys()))
            if max_corr > self.config.max_correlation:
                # Count correlated positions
                correlated_count = sum(
                    1 for sym in portfolio.positions
                    if self._get_correlation(signal.symbol, sym) > self.config.max_correlation
                )
                if correlated_count >= self.config.max_correlated_positions:
                    return False, f"Too many correlated positions ({correlated_count})"
        
        return True, ""
    
    def _check_drawdown_limits(
        self,
        portfolio: PortfolioState,
    ) -> tuple[bool, str]:
        """Check drawdown limits."""
        current_dd = self._calculate_drawdown(portfolio.equity)
        
        # Check max drawdown
        if current_dd > self.config.max_drawdown:
            self._trigger_circuit_breaker("Max drawdown exceeded")
            return False, f"Maximum drawdown ({self.config.max_drawdown:.1%}) exceeded"
        
        # Check circuit breaker threshold
        if current_dd > self.config.circuit_breaker_threshold:
            self._trigger_circuit_breaker("Drawdown threshold exceeded")
            return False, f"Drawdown threshold ({self.config.circuit_breaker_threshold:.1%}) exceeded"
        
        # Check daily loss
        if self._daily_pnl < -self.config.daily_loss_limit * self._session_start_equity:
            return False, f"Daily loss limit ({self.config.daily_loss_limit:.1%}) exceeded"
        
        return True, ""
    
    def _calculate_drawdown(self, equity: float) -> float:
        """Calculate current drawdown."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - equity) / self._peak_equity
    
    def _calculate_position_size(
        self,
        signal: SignalEvent,
        portfolio: PortfolioState,
        volatility: float,
        atr: float,
    ) -> float:
        """Calculate position size based on method."""
        price = signal.price
        equity = portfolio.equity
        
        # Get win rate and win/loss ratio for Kelly
        win_rate, win_loss_ratio = self._get_trade_statistics()
        
        # Calculate using configured method
        base_size = calculate_position_size(
            equity=equity,
            price=price,
            method=self.config.position_sizing_method,
            risk_per_trade=self.config.max_portfolio_risk,
            stop_loss_pct=self.config.stop_loss_pct,
            volatility=volatility,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            atr=atr,
        )
        
        # Apply maximum position size limit
        max_shares = (equity * self.config.max_position_size) / price
        base_size = min(base_size, max_shares)
        
        # Apply signal confidence scaling
        confidence = getattr(signal, 'confidence', 0.5)
        if confidence < 0.6:
            base_size *= confidence / 0.6  # Scale down for low confidence
        
        # Ensure we have enough cash
        max_from_cash = portfolio.cash * 0.95 / price  # 95% of cash
        base_size = min(base_size, max_from_cash)
        
        return max(0, base_size)
    
    def _adjust_for_volatility(
        self,
        position_size: float,
        volatility: float,
    ) -> float:
        """Adjust position size for volatility."""
        # Assume normal volatility is 2%
        normal_vol = 0.02
        vol_ratio = volatility / normal_vol
        
        if vol_ratio > self.config.high_vol_threshold:
            # Reduce position size
            reduction = 1 - (self.config.high_vol_size_reduction * (vol_ratio - 1))
            reduction = max(0.25, reduction)  # Don't reduce more than 75%
            return position_size * reduction
        
        return position_size
    
    def _calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        volatility: float,
    ) -> float:
        """Calculate stop-loss price."""
        if self.config.stop_loss_type == StopLossType.FIXED_PERCENT:
            return entry_price * (1 - self.config.stop_loss_pct)
        
        elif self.config.stop_loss_type == StopLossType.ATR_BASED:
            return entry_price - (atr * self.config.stop_loss_atr_mult)
        
        elif self.config.stop_loss_type == StopLossType.VOLATILITY_BASED:
            # Use 2 standard deviations
            return entry_price * (1 - 2 * volatility)
        
        elif self.config.stop_loss_type == StopLossType.CHANDELIER:
            # Chandelier exit: highest high - ATR multiple
            return entry_price - (atr * 3)
        
        else:
            return entry_price * (1 - self.config.stop_loss_pct)
    
    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        atr: float,
    ) -> float:
        """Calculate take-profit price."""
        if self.config.take_profit_type == TakeProfitType.FIXED_PERCENT:
            return entry_price * (1 + self.config.take_profit_pct)
        
        elif self.config.take_profit_type == TakeProfitType.ATR_BASED:
            return entry_price + (atr * self.config.stop_loss_atr_mult * 2)
        
        elif self.config.take_profit_type == TakeProfitType.RISK_REWARD:
            risk = entry_price - stop_loss
            reward = risk * self.config.take_profit_rr_mult
            return entry_price + reward
        
        else:
            return entry_price * (1 + self.config.take_profit_pct)
    
    def _update_trailing_stop(
        self,
        profile: PositionRiskProfile,
        current_price: float,
    ) -> float:
        """Update trailing stop level."""
        # Check if trailing should be activated
        profit_pct = (current_price - profile.entry_price) / profile.entry_price
        
        if profit_pct < self.config.trailing_activation_pct:
            return profile.current_stop
        
        profile.trailing_activated = True
        
        # Calculate new trailing stop
        new_stop = current_price * (1 - self.config.trailing_stop_pct)
        
        return max(new_stop, profile.current_stop)
    
    def _calculate_risk_metrics(
        self,
        assessment: RiskAssessment,
        portfolio: PortfolioState,
        position_size: float,
    ) -> None:
        """Calculate risk metrics for assessment."""
        entry_price = assessment.entry_price
        stop_loss = assessment.stop_loss_price
        take_profit = assessment.take_profit_price
        
        # Potential loss
        loss_per_share = entry_price - stop_loss
        assessment.potential_loss = loss_per_share * position_size
        
        # Potential profit
        profit_per_share = take_profit - entry_price
        assessment.potential_profit = profit_per_share * position_size
        
        # Risk/reward ratio
        if loss_per_share > 0:
            assessment.risk_reward_ratio = profit_per_share / loss_per_share
        
        # Position risk as % of portfolio
        assessment.position_risk_pct = assessment.potential_loss / portfolio.equity if portfolio.equity > 0 else 0
        
        # New position count
        assessment.new_position_count = len(portfolio.positions) + 1
        
        # New portfolio exposure
        current_exposure = sum(
            abs(p.market_value) for p in portfolio.positions.values()
        ) if portfolio.positions else 0
        new_exposure = current_exposure + (entry_price * position_size)
        assessment.new_portfolio_exposure = new_exposure / portfolio.equity if portfolio.equity > 0 else 0
    
    def _add_warnings(
        self,
        assessment: RiskAssessment,
        portfolio: PortfolioState,
    ) -> None:
        """Add warnings to assessment."""
        # Position size warning
        if assessment.adjusted_size < assessment.original_size * 0.5:
            assessment.warnings.append(
                f"Position size reduced by >50% ({assessment.original_size:.0f} -> {assessment.adjusted_size:.0f})"
            )
        
        # Drawdown warning
        current_dd = self._calculate_drawdown(portfolio.equity)
        if current_dd > self.config.max_drawdown * 0.7:
            assessment.warnings.append(
                f"Approaching max drawdown ({current_dd:.1%} / {self.config.max_drawdown:.1%})"
            )
        
        # Daily loss warning
        if self._daily_pnl < -self.config.daily_loss_limit * self._session_start_equity * 0.7:
            assessment.warnings.append("Approaching daily loss limit")
        
        # Position count warning
        if assessment.new_position_count >= self.config.max_positions:
            assessment.warnings.append("At maximum position capacity")
        
        # Volatility warning
        if assessment.volatility_regime == "high":
            assessment.warnings.append("High volatility regime - reduced position size")
    
    def _get_trade_statistics(self) -> tuple[float, float]:
        """Get win rate and win/loss ratio from trade history."""
        if len(self._trade_history) < 10:
            return 0.5, 1.5  # Default values
        
        recent = self._trade_history[-self.config.kelly_lookback_trades:]
        
        wins = [t for t in recent if t["pnl"] > 0]
        losses = [t for t in recent if t["pnl"] < 0]
        
        win_rate = len(wins) / len(recent) if recent else 0.5
        
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
        
        return win_rate, win_loss_ratio
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        key = tuple(sorted([symbol1, symbol2]))
        return self._correlation_matrix.get(key, 0.0)
    
    def _get_max_correlation(self, symbol: str, other_symbols: list[str]) -> float:
        """Get maximum correlation with other symbols."""
        if not other_symbols:
            return 0.0
        
        correlations = [
            self._get_correlation(symbol, other)
            for other in other_symbols
        ]
        
        return max(correlations) if correlations else 0.0
    
    def update_correlations(
        self,
        returns_data: dict[str, NDArray[np.float64]],
    ) -> None:
        """Update correlation matrix from returns data."""
        symbols = list(returns_data.keys())
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                ret1 = returns_data[sym1]
                ret2 = returns_data[sym2]
                
                # Align lengths
                min_len = min(len(ret1), len(ret2))
                if min_len < 20:
                    continue
                
                corr = np.corrcoef(ret1[-min_len:], ret2[-min_len:])[0, 1]
                
                key = tuple(sorted([sym1, sym2]))
                self._correlation_matrix[key] = corr


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_risk_management(
    signal: SignalEvent,
    portfolio: PortfolioState,
    config: RiskIntegrationConfig | None = None,
    market_data: dict[str, Any] | None = None,
) -> RiskAssessment:
    """
    Convenience function to apply risk management to a signal.
    
    Args:
        signal: Trading signal
        portfolio: Portfolio state
        config: Risk configuration
        market_data: Market data (volatility, ATR, etc.)
    
    Returns:
        RiskAssessment
    """
    integrator = RiskIntegrator(config)
    return integrator.process_signal(signal, portfolio, market_data)


def create_risk_integrator(
    max_position_size: float = 0.10,
    max_drawdown: float = 0.15,
    use_trailing_stop: bool = True,
) -> RiskIntegrator:
    """
    Create a risk integrator with common settings.
    
    Args:
        max_position_size: Max position as % of portfolio
        max_drawdown: Max drawdown limit
        use_trailing_stop: Enable trailing stops
    
    Returns:
        Configured RiskIntegrator
    """
    config = RiskIntegrationConfig(
        max_position_size=max_position_size,
        max_drawdown=max_drawdown,
        use_trailing_stop=use_trailing_stop,
    )
    return RiskIntegrator(config)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo of risk integration."""
    print("Risk Integration Module")
    print("="*60)
    
    # Create mock signal and portfolio
    from dataclasses import dataclass
    
    @dataclass
    class MockSignal:
        symbol: str = "AAPL"
        price: float = 150.0
        direction: int = 1
        confidence: float = 0.65
        target_size: float = 100
    
    @dataclass
    class MockPortfolio:
        equity: float = 100000.0
        cash: float = 50000.0
        positions: dict = field(default_factory=dict)
    
    signal = MockSignal()
    portfolio = MockPortfolio()
    
    # Process through risk management
    integrator = RiskIntegrator()
    integrator.new_session(portfolio.equity)
    
    assessment = integrator.process_signal(
        signal, portfolio,
        market_data={"volatility": 0.025, "atr": 3.0}
    )
    
    print(f"\nSignal: {assessment.symbol}")
    print(f"Action: {assessment.action.value}")
    print(f"Reason: {assessment.reason}")
    print(f"Original Size: {assessment.original_size}")
    print(f"Adjusted Size: {assessment.adjusted_size:.0f}")
    print(f"Entry Price: ${assessment.entry_price:.2f}")
    print(f"Stop Loss: ${assessment.stop_loss_price:.2f}")
    print(f"Take Profit: ${assessment.take_profit_price:.2f}")
    print(f"Risk/Reward: {assessment.risk_reward_ratio:.2f}")
    print(f"Position Risk: {assessment.position_risk_pct:.2%}")
    
    if assessment.warnings:
        print(f"\nWarnings:")
        for w in assessment.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    configure_logging(get_settings())
    main()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StopLossType",
    "TakeProfitType",
    "RiskAction",
    # Configuration
    "RiskIntegrationConfig",
    "RiskAssessment",
    "PositionRiskProfile",
    # Main class
    "RiskIntegrator",
    # Convenience functions
    "apply_risk_management",
    "create_risk_integrator",
]