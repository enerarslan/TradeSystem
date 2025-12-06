"""
Risk Manager Module
===================

Comprehensive risk management for the algorithmic trading platform.
Implements position sizing, VaR calculations, risk limits, and circuit breakers.

Features:
- Multiple position sizing methods (fixed, percent, Kelly, volatility-based)
- Value at Risk (VaR) calculations (historical, parametric, Cornish-Fisher)
- Conditional VaR (CVaR/Expected Shortfall)
- Portfolio risk monitoring
- Drawdown limits and circuit breakers
- Correlation-based position limits
- Real-time risk metrics

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from config.settings import get_logger, RiskConstants
from core.events import SignalEvent, RiskEvent, EventPriority
from core.types import (
    Order,
    Position,
    PortfolioState,
    RiskError,
    RiskLimitExceededError,
    DrawdownLimitError,
)

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class RiskLevel(str, Enum):
    """Risk alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED = "fixed"                    # Fixed dollar amount
    PERCENT = "percent"                # Percentage of equity
    KELLY = "kelly"                    # Kelly criterion
    VOLATILITY = "volatility"          # Volatility-adjusted
    RISK_PARITY = "risk_parity"        # Equal risk contribution
    OPTIMAL_F = "optimal_f"            # Optimal f method
    ATR = "atr"                        # ATR-based sizing


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """
    Risk management configuration.
    
    Attributes:
        max_position_size: Maximum position size as portfolio fraction
        max_portfolio_risk: Maximum portfolio risk per trade
        max_drawdown: Maximum allowed drawdown before circuit breaker
        daily_loss_limit: Maximum daily loss before stopping trading
        weekly_loss_limit: Maximum weekly loss limit
        max_positions: Maximum number of concurrent positions
        max_correlated_positions: Maximum positions with high correlation
        correlation_threshold: Threshold for high correlation
        var_confidence: VaR confidence level
        var_lookback: VaR lookback period in days
        position_sizing_method: Default position sizing method
        use_stop_loss: Require stop-loss on all positions
        default_stop_loss_pct: Default stop-loss percentage
        use_take_profit: Use take-profit orders
        default_take_profit_pct: Default take-profit percentage
        circuit_breaker_enabled: Enable circuit breakers
        circuit_breaker_cooldown: Cooldown period after circuit breaker
    """
    # Position limits
    max_position_size: float = 0.10
    max_portfolio_risk: float = 0.02
    max_positions: int = 20
    max_correlated_positions: int = 5
    correlation_threshold: float = 0.7
    
    # Loss limits
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.02
    weekly_loss_limit: float = 0.05
    
    # VaR settings
    var_confidence: float = 0.95
    var_lookback: int = 252
    
    # Position sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY
    
    # Stop-loss / Take-profit
    use_stop_loss: bool = True
    default_stop_loss_pct: float = 0.02
    use_take_profit: bool = True
    default_take_profit_pct: float = 0.04
    
    # Circuit breakers
    circuit_breaker_enabled: bool = True
    circuit_breaker_cooldown: timedelta = timedelta(hours=24)


@dataclass
class RiskMetrics:
    """
    Current risk metrics snapshot.
    
    Attributes:
        timestamp: Metrics calculation time
        portfolio_value: Current portfolio value
        cash: Available cash
        positions_value: Total value of positions
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss (session)
        daily_pnl: Today's P&L
        weekly_pnl: This week's P&L
        current_drawdown: Current drawdown from peak
        max_drawdown: Maximum drawdown observed
        var_95: Value at Risk (95%)
        var_99: Value at Risk (99%)
        cvar_95: Conditional VaR (95%)
        position_count: Number of open positions
        gross_exposure: Total absolute exposure
        net_exposure: Net long/short exposure
        leverage: Current leverage ratio
        correlation_risk: Portfolio correlation risk
        largest_position: Largest position as % of portfolio
    """
    timestamp: datetime = field(default_factory=datetime.now)
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    position_count: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    correlation_risk: float = 0.0
    largest_position: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "position_count": self.position_count,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "correlation_risk": self.correlation_risk,
            "largest_position": self.largest_position,
        }


# =============================================================================
# VAR CALCULATIONS
# =============================================================================

def calculate_var(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Array of historical returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: Calculation method ("historical", "parametric", "cornish_fisher")
    
    Returns:
        VaR value (positive number representing potential loss)
    """
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 10:
        return 0.0
    
    alpha = 1 - confidence
    
    if method == "historical":
        var = -np.percentile(returns, alpha * 100)
    
    elif method == "parametric":
        mu = np.mean(returns)
        sigma = np.std(returns)
        z = scipy_stats.norm.ppf(alpha)
        var = -(mu + z * sigma)
    
    elif method == "cornish_fisher":
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = scipy_stats.skew(returns)
        kurt = scipy_stats.kurtosis(returns)
        z = scipy_stats.norm.ppf(alpha)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * (kurt - 3) / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = -(mu + z_cf * sigma)
    
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    
    return max(0, var)


def calculate_cvar(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).
    
    CVaR is the expected loss given that loss exceeds VaR.
    
    Args:
        returns: Array of historical returns
        confidence: Confidence level
    
    Returns:
        CVaR value (positive number)
    """
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 10:
        return 0.0
    
    var = calculate_var(returns, confidence, "historical")
    
    # Expected shortfall: mean of returns worse than VaR
    tail_returns = returns[returns < -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


# =============================================================================
# POSITION SIZING
# =============================================================================

def calculate_position_size(
    equity: float,
    price: float,
    method: PositionSizingMethod = PositionSizingMethod.PERCENT,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.02,
    volatility: float | None = None,
    win_rate: float | None = None,
    win_loss_ratio: float | None = None,
    atr: float | None = None,
) -> float:
    """
    Calculate position size using various methods.
    
    Args:
        equity: Current portfolio equity
        price: Current asset price
        method: Position sizing method
        risk_per_trade: Risk per trade as fraction of equity
        stop_loss_pct: Stop-loss percentage
        volatility: Asset volatility (for volatility method)
        win_rate: Historical win rate (for Kelly)
        win_loss_ratio: Average win/loss ratio (for Kelly)
        atr: Average True Range (for ATR method)
    
    Returns:
        Number of shares to trade
    """
    if price <= 0 or equity <= 0:
        return 0.0
    
    if method == PositionSizingMethod.FIXED:
        # Fixed dollar amount
        position_value = equity * risk_per_trade * 10  # 10x risk for position
        shares = position_value / price
    
    elif method == PositionSizingMethod.PERCENT:
        # Fixed percentage of equity
        position_value = equity * 0.10  # 10% of portfolio
        shares = position_value / price
    
    elif method == PositionSizingMethod.KELLY:
        # Kelly criterion
        if win_rate is None or win_loss_ratio is None:
            # Default conservative Kelly
            kelly_fraction = 0.25
        else:
            # Kelly formula: f* = (p * b - q) / b
            # where p = win rate, q = 1-p, b = win/loss ratio
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        position_value = equity * kelly_fraction
        shares = position_value / price
    
    elif method == PositionSizingMethod.VOLATILITY:
        # Volatility-adjusted position sizing
        if volatility is None or volatility <= 0:
            volatility = 0.02  # Default 2% daily volatility
        
        # Target volatility contribution
        target_vol = risk_per_trade
        position_value = equity * (target_vol / volatility)
        shares = position_value / price
    
    elif method == PositionSizingMethod.ATR:
        # ATR-based position sizing
        if atr is None or atr <= 0:
            atr = price * 0.02  # Default 2% of price
        
        # Risk amount in dollars
        risk_amount = equity * risk_per_trade
        # Position size so that ATR move = risk amount
        shares = risk_amount / atr
    
    elif method == PositionSizingMethod.RISK_PARITY:
        # Equal risk contribution (simplified)
        if volatility is None or volatility <= 0:
            volatility = 0.02
        
        # Target equal volatility contribution
        target_risk = risk_per_trade / 10  # Assume 10 positions
        position_value = equity * (target_risk / volatility)
        shares = position_value / price
    
    elif method == PositionSizingMethod.OPTIMAL_F:
        # Optimal f method
        if win_rate is None or win_loss_ratio is None:
            opt_f = 0.10
        else:
            opt_f = optimal_f(win_rate, win_loss_ratio)
        
        position_value = equity * opt_f
        shares = position_value / price
    
    else:
        # Default to percent
        position_value = equity * 0.10
        shares = position_value / price
    
    return max(0, shares)


def kelly_criterion(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly criterion position size.
    
    Args:
        win_rate: Probability of winning
        win_loss_ratio: Average win / average loss
        fraction: Kelly fraction (0.25 = quarter Kelly)
    
    Returns:
        Optimal position size as fraction of bankroll
    """
    if win_loss_ratio <= 0:
        return 0.0
    
    # Kelly formula: f* = (p * b - q) / b
    q = 1 - win_rate
    kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
    
    # Apply fraction (quarter Kelly is safer)
    kelly = kelly * fraction
    
    return max(0, min(kelly, 0.25))  # Cap at 25%


def optimal_f(
    win_rate: float,
    win_loss_ratio: float,
) -> float:
    """
    Calculate optimal f for position sizing.
    
    Based on Ralph Vince's optimal f concept.
    
    Args:
        win_rate: Probability of winning
        win_loss_ratio: Average win / average loss
    
    Returns:
        Optimal f value
    """
    # Simplified optimal f calculation
    # Full implementation would use historical trade data
    if win_loss_ratio <= 0:
        return 0.0
    
    # TWR optimization (simplified)
    best_f = 0.0
    best_twr = 0.0
    
    for f in np.arange(0.01, 0.50, 0.01):
        # Simulate TWR (Terminal Wealth Relative)
        win_factor = 1 + f * win_loss_ratio
        loss_factor = 1 - f
        
        # Geometric mean
        twr = (win_factor ** win_rate) * (loss_factor ** (1 - win_rate))
        
        if twr > best_twr:
            best_twr = twr
            best_f = f
    
    return best_f


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Comprehensive risk management system.
    
    Implements the RiskManager interface from core/interfaces.py.
    
    Features:
        - Order validation against risk limits
        - Position sizing calculations
        - Portfolio risk monitoring
        - VaR and CVaR calculations
        - Drawdown tracking
        - Circuit breakers
        - Correlation risk monitoring
    
    Example:
        config = RiskConfig(max_drawdown=0.15)
        risk_mgr = RiskManager(config)
        
        # Validate order
        is_valid, reason = risk_mgr.validate_order(order, portfolio)
        
        # Calculate position size
        size = risk_mgr.calculate_position_size(symbol, signal, portfolio)
        
        # Check portfolio risk
        breaches = risk_mgr.check_portfolio_risk(portfolio)
    """
    
    def __init__(self, config: RiskConfig | None = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()
        
        # State tracking
        self._peak_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._session_start_equity: float = 0.0
        self._week_start_equity: float = 0.0
        self._last_reset_date: datetime | None = None
        self._week_start_date: datetime | None = None
        
        # Circuit breaker state
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_triggered_at: datetime | None = None
        
        # History
        self._returns_history: list[float] = []
        self._equity_history: list[float] = []
        self._risk_events: list[RiskEvent] = []
        
        logger.info(f"RiskManager initialized with config: max_dd={config.max_drawdown if config else 0.15}")
    
    def validate_order(
        self,
        order: Order,
        portfolio: PortfolioState,
    ) -> tuple[bool, str]:
        """
        Validate an order against risk limits.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check circuit breaker
        if self._circuit_breaker_active:
            if self._check_circuit_breaker_cooldown():
                return False, "Circuit breaker active - trading suspended"
        
        # Check maximum positions
        if len(portfolio.positions) >= self.config.max_positions:
            if order.side.lower() == "buy":
                return False, f"Maximum positions ({self.config.max_positions}) reached"
        
        # Check position size limit
        order_value = order.quantity * (order.limit_price or order.stop_price or 0)
        if order_value > 0:
            position_pct = order_value / portfolio.equity
            if position_pct > self.config.max_position_size:
                return False, f"Position size {position_pct:.1%} exceeds limit {self.config.max_position_size:.1%}"
        
        # Check available cash for buys
        if order.side.lower() == "buy":
            estimated_cost = order.quantity * (order.limit_price or order.stop_price or 0)
            estimated_cost *= (1 + 0.001)  # Add buffer for slippage
            
            if estimated_cost > portfolio.cash:
                return False, f"Insufficient cash: need ${estimated_cost:,.2f}, have ${portfolio.cash:,.2f}"
        
        # Check daily loss limit
        if self._daily_pnl < -self.config.daily_loss_limit * self._session_start_equity:
            return False, f"Daily loss limit ({self.config.daily_loss_limit:.1%}) reached"
        
        # Check drawdown limit
        current_dd = self._calculate_drawdown(portfolio.equity)
        if current_dd > self.config.max_drawdown:
            self._trigger_circuit_breaker("max_drawdown")
            return False, f"Maximum drawdown ({self.config.max_drawdown:.1%}) exceeded"
        
        return True, ""
    
    def calculate_position_size(
        self,
        symbol: str,
        signal: SignalEvent,
        portfolio: PortfolioState,
    ) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            portfolio: Current portfolio state
        
        Returns:
            Recommended position size (number of shares)
        """
        if portfolio.equity <= 0 or signal.price <= 0:
            return 0.0
        
        # Get volatility from signal metadata
        volatility = signal.metadata.get("volatility")
        atr = signal.metadata.get("atr")
        
        # Calculate base position size
        shares = calculate_position_size(
            equity=portfolio.equity,
            price=signal.price,
            method=self.config.position_sizing_method,
            risk_per_trade=self.config.max_portfolio_risk,
            stop_loss_pct=self.config.default_stop_loss_pct,
            volatility=volatility,
            atr=atr,
        )
        
        # Apply signal strength multiplier
        shares *= signal.strength
        
        # Check against position limits
        max_value = portfolio.equity * self.config.max_position_size
        max_shares = max_value / signal.price
        shares = min(shares, max_shares)
        
        # Check against available cash
        if signal.direction > 0:  # Long
            max_from_cash = portfolio.cash / signal.price
            shares = min(shares, max_from_cash * 0.95)  # 95% of available
        
        return max(0, shares)
    
    def check_portfolio_risk(
        self,
        portfolio: PortfolioState,
    ) -> list[tuple[str, str, float]]:
        """
        Check portfolio for risk limit breaches.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            List of (risk_type, level, value) tuples
        """
        breaches = []
        
        # Update equity tracking
        self._update_equity_tracking(portfolio.equity)
        
        # Check drawdown
        current_dd = self._calculate_drawdown(portfolio.equity)
        if current_dd > self.config.max_drawdown:
            breaches.append(("drawdown", "critical", current_dd))
        elif current_dd > self.config.max_drawdown * 0.8:
            breaches.append(("drawdown", "warning", current_dd))
        
        # Check daily loss
        daily_loss_pct = self._daily_pnl / self._session_start_equity if self._session_start_equity > 0 else 0
        if daily_loss_pct < -self.config.daily_loss_limit:
            breaches.append(("daily_loss", "critical", abs(daily_loss_pct)))
        elif daily_loss_pct < -self.config.daily_loss_limit * 0.8:
            breaches.append(("daily_loss", "warning", abs(daily_loss_pct)))
        
        # Check position concentration
        if portfolio.positions:
            max_pos_pct = max(
                abs(p.market_value) / portfolio.equity 
                for p in portfolio.positions.values()
            ) if portfolio.equity > 0 else 0
            
            if max_pos_pct > self.config.max_position_size:
                breaches.append(("position_concentration", "warning", max_pos_pct))
        
        # Check number of positions
        if len(portfolio.positions) >= self.config.max_positions:
            breaches.append(("position_count", "warning", len(portfolio.positions)))
        
        # Log breaches
        for risk_type, level, value in breaches:
            self._create_risk_event(risk_type, level, value, portfolio)
        
        return breaches
    
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
        
        Returns:
            VaR value
        """
        return calculate_var(returns, confidence, "historical")
    
    def get_risk_metrics(
        self,
        portfolio: PortfolioState,
    ) -> dict[str, float]:
        """
        Calculate current risk metrics.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            Dictionary of risk metrics
        """
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio.equity,
            cash=portfolio.cash,
            positions_value=portfolio.equity - portfolio.cash,
            unrealized_pnl=sum(p.unrealized_pnl for p in portfolio.positions.values()) if portfolio.positions else 0,
            daily_pnl=self._daily_pnl,
            weekly_pnl=self._weekly_pnl,
            current_drawdown=self._calculate_drawdown(portfolio.equity),
            max_drawdown=self._peak_equity - min(self._equity_history) if self._equity_history else 0,
            position_count=len(portfolio.positions),
        )
        
        # Calculate VaR if we have enough history
        if len(self._returns_history) >= 20:
            returns = np.array(self._returns_history)
            metrics.var_95 = calculate_var(returns, 0.95)
            metrics.var_99 = calculate_var(returns, 0.99)
            metrics.cvar_95 = calculate_cvar(returns, 0.95)
        
        # Calculate exposure
        if portfolio.positions:
            long_exposure = sum(
                p.market_value for p in portfolio.positions.values() 
                if p.quantity > 0
            )
            short_exposure = sum(
                abs(p.market_value) for p in portfolio.positions.values() 
                if p.quantity < 0
            )
            metrics.gross_exposure = long_exposure + short_exposure
            metrics.net_exposure = long_exposure - short_exposure
            metrics.leverage = metrics.gross_exposure / portfolio.equity if portfolio.equity > 0 else 0
            
            # Largest position
            if portfolio.equity > 0:
                metrics.largest_position = max(
                    abs(p.market_value) / portfolio.equity 
                    for p in portfolio.positions.values()
                )
        
        return metrics.to_dict()
    
    def get_full_metrics(self, portfolio: PortfolioState) -> RiskMetrics:
        """Get full RiskMetrics object."""
        metrics_dict = self.get_risk_metrics(portfolio)
        return RiskMetrics(**{k: v for k, v in metrics_dict.items() if k != "timestamp"})
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _update_equity_tracking(self, equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        now = datetime.now()
        
        # Initialize on first call
        if self._last_reset_date is None:
            self._last_reset_date = now.date()
            self._session_start_equity = equity
            self._peak_equity = equity
        
        # Reset daily tracking
        if now.date() != self._last_reset_date:
            self._daily_pnl = 0.0
            self._session_start_equity = equity
            self._last_reset_date = now.date()
        
        # Reset weekly tracking
        if self._week_start_date is None or (now - self._week_start_date).days >= 7:
            self._weekly_pnl = 0.0
            self._week_start_equity = equity
            self._week_start_date = now
        
        # Update peak equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        # Update P&L
        if self._session_start_equity > 0:
            self._daily_pnl = equity - self._session_start_equity
        if self._week_start_equity > 0:
            self._weekly_pnl = equity - self._week_start_equity
        
        # Store history
        self._equity_history.append(equity)
        if len(self._equity_history) > 1:
            ret = (equity - self._equity_history[-2]) / self._equity_history[-2]
            self._returns_history.append(ret)
        
        # Trim history to lookback period
        max_history = self.config.var_lookback * 2
        if len(self._equity_history) > max_history:
            self._equity_history = self._equity_history[-max_history:]
            self._returns_history = self._returns_history[-max_history:]
    
    def _calculate_drawdown(self, equity: float) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - equity) / self._peak_equity
    
    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker."""
        if not self.config.circuit_breaker_enabled:
            return
        
        self._circuit_breaker_active = True
        self._circuit_breaker_triggered_at = datetime.now()
        
        logger.critical(f"Circuit breaker triggered: {reason}")
    
    def _check_circuit_breaker_cooldown(self) -> bool:
        """Check if circuit breaker is still active."""
        if not self._circuit_breaker_active:
            return False
        
        if self._circuit_breaker_triggered_at is None:
            return False
        
        elapsed = datetime.now() - self._circuit_breaker_triggered_at
        if elapsed >= self.config.circuit_breaker_cooldown:
            self._circuit_breaker_active = False
            logger.info("Circuit breaker cooldown complete - trading resumed")
            return False
        
        return True
    
    def _create_risk_event(
        self,
        risk_type: str,
        level: str,
        value: float,
        portfolio: PortfolioState,
    ) -> RiskEvent:
        """Create and store a risk event."""
        event = RiskEvent(
            risk_type=risk_type,
            level=level,
            current_value=value,
            limit_value=self._get_limit_for_type(risk_type),
            message=f"{risk_type} at {value:.2%}",
            action_required=self._get_action_for_level(level),
            affected_positions=list(portfolio.positions.keys()) if portfolio.positions else [],
            priority=EventPriority.CRITICAL if level == "critical" else EventPriority.HIGH,
        )
        
        self._risk_events.append(event)
        return event
    
    def _get_limit_for_type(self, risk_type: str) -> float:
        """Get the limit value for a risk type."""
        limits = {
            "drawdown": self.config.max_drawdown,
            "daily_loss": self.config.daily_loss_limit,
            "weekly_loss": self.config.weekly_loss_limit,
            "position_concentration": self.config.max_position_size,
            "position_count": self.config.max_positions,
        }
        return limits.get(risk_type, 0.0)
    
    def _get_action_for_level(self, level: str) -> str:
        """Get recommended action for risk level."""
        actions = {
            "info": "Monitor",
            "warning": "Review positions and reduce exposure if necessary",
            "critical": "Immediate position reduction required",
            "emergency": "Close all positions immediately",
        }
        return actions.get(level, "Review")
    
    def reset(self) -> None:
        """Reset risk manager state."""
        self._peak_equity = 0.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._session_start_equity = 0.0
        self._week_start_equity = 0.0
        self._last_reset_date = None
        self._week_start_date = None
        self._circuit_breaker_active = False
        self._circuit_breaker_triggered_at = None
        self._returns_history.clear()
        self._equity_history.clear()
        self._risk_events.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RiskLevel",
    "PositionSizingMethod",
    # Classes
    "RiskConfig",
    "RiskMetrics",
    "RiskManager",
    # Functions
    "calculate_var",
    "calculate_cvar",
    "calculate_position_size",
    "kelly_criterion",
    "optimal_f",
]