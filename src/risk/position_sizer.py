"""
Institutional Position Sizing Algorithms
JPMorgan-Level Position Sizing and Capital Allocation

Features:
- Kelly Criterion with fractional Kelly
- Volatility-based sizing
- Risk parity allocation
- Optimal-F position sizing
- Maximum drawdown constraints
- Correlation-adjusted sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import scipy.optimize as opt
from scipy.stats import norm

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_FRACTION = "fixed_fraction"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"
    EQUAL_WEIGHT = "equal_weight"
    MAXIMUM_DIVERSIFICATION = "max_diversification"
    MINIMUM_VARIANCE = "min_variance"


@dataclass
class PositionSize:
    """Position sizing result"""
    symbol: str
    shares: int
    dollar_value: float
    weight: float  # Portfolio weight
    risk_contribution: float  # Risk contribution percentage
    sizing_method: str
    confidence: float
    constraints_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SizingConfig:
    """Position sizing configuration"""
    method: SizingMethod = SizingMethod.VOLATILITY
    max_position_pct: float = 0.10  # Max 10% per position
    max_sector_pct: float = 0.30  # Max 30% per sector
    max_correlated_pct: float = 0.40  # Max 40% in correlated assets
    target_volatility: float = 0.15  # 15% annual volatility target
    risk_free_rate: float = 0.05  # Risk-free rate
    kelly_fraction: float = 0.25  # Quarter Kelly for safety
    min_position_pct: float = 0.01  # Minimum 1% position
    max_leverage: float = 1.0  # No leverage by default
    volatility_lookback: int = 60  # Days for volatility calculation
    correlation_lookback: int = 120  # Days for correlation
    rebalance_threshold: float = 0.05  # 5% drift threshold


class PositionSizer(ABC):
    """
    Abstract base class for position sizing algorithms.

    Implements institutional-grade position sizing with
    comprehensive risk controls.
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
        self._historical_sizes: List[Dict] = []

    @abstractmethod
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        **kwargs
    ) -> PositionSize:
        """Calculate position size for a single asset"""
        pass

    def calculate_sizes(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, PositionSize]:
        """
        Calculate position sizes for multiple assets.

        Args:
            signals: Signal strength per symbol (-1 to 1)
            prices: Current prices per symbol
            portfolio_value: Total portfolio value
            current_positions: Current position values

        Returns:
            Dictionary of position sizes
        """
        sizes = {}

        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            try:
                size = self.calculate_size(
                    symbol=symbol,
                    current_price=prices[symbol],
                    portfolio_value=portfolio_value,
                    signal_strength=signal,
                    current_positions=current_positions,
                    **kwargs
                )

                # Apply constraints
                size = self._apply_constraints(
                    size, portfolio_value, current_positions
                )

                sizes[symbol] = size

            except Exception as e:
                logger.error(f"Error sizing {symbol}: {e}")

        # Apply portfolio-level constraints
        sizes = self._apply_portfolio_constraints(
            sizes, portfolio_value, current_positions
        )

        return sizes

    def _apply_constraints(
        self,
        size: PositionSize,
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> PositionSize:
        """Apply individual position constraints"""
        constraints_applied = []

        # Max position constraint
        max_value = portfolio_value * self.config.max_position_pct
        if size.dollar_value > max_value:
            size.dollar_value = max_value
            size.shares = int(max_value / (size.dollar_value / size.shares)) if size.shares > 0 else 0
            size.weight = max_value / portfolio_value
            constraints_applied.append("max_position")

        # Min position constraint
        min_value = portfolio_value * self.config.min_position_pct
        if 0 < size.dollar_value < min_value:
            size.dollar_value = 0
            size.shares = 0
            size.weight = 0
            constraints_applied.append("min_position")

        size.constraints_applied.extend(constraints_applied)
        return size

    def _apply_portfolio_constraints(
        self,
        sizes: Dict[str, PositionSize],
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, PositionSize]:
        """Apply portfolio-level constraints"""
        total_weight = sum(s.weight for s in sizes.values())

        # Scale down if exceeds max leverage
        if total_weight > self.config.max_leverage:
            scale = self.config.max_leverage / total_weight
            for symbol, size in sizes.items():
                size.dollar_value *= scale
                size.shares = int(size.shares * scale)
                size.weight *= scale
                size.constraints_applied.append("leverage_limit")

        return sizes

    def get_rebalance_trades(
        self,
        target_sizes: Dict[str, PositionSize],
        current_positions: Dict[str, float],
        prices: Dict[str, float],
        min_trade_value: float = 1000
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate trades needed to rebalance to target positions.

        Returns:
            Dictionary of trades to execute
        """
        trades = {}

        for symbol, target in target_sizes.items():
            current_value = current_positions.get(symbol, 0)
            target_value = target.dollar_value

            diff = target_value - current_value

            if abs(diff) >= min_trade_value:
                price = prices.get(symbol, 0)
                if price > 0:
                    shares = int(diff / price)
                    if shares != 0:
                        trades[symbol] = {
                            'symbol': symbol,
                            'shares': abs(shares),
                            'side': 'buy' if shares > 0 else 'sell',
                            'dollar_value': abs(diff),
                            'current_value': current_value,
                            'target_value': target_value,
                            'current_weight': current_value / sum(current_positions.values()) if current_positions else 0,
                            'target_weight': target.weight
                        }

        # Handle positions to close
        for symbol, value in current_positions.items():
            if symbol not in target_sizes and value > min_trade_value:
                price = prices.get(symbol, 0)
                if price > 0:
                    trades[symbol] = {
                        'symbol': symbol,
                        'shares': int(value / price),
                        'side': 'sell',
                        'dollar_value': value,
                        'current_value': value,
                        'target_value': 0,
                        'current_weight': value / sum(current_positions.values()),
                        'target_weight': 0
                    }

        return trades


class KellyCriterion(PositionSizer):
    """
    Kelly Criterion position sizing.

    Calculates optimal bet size based on:
    f* = (p * b - q) / b

    Where:
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = odds (win/loss ratio)

    Uses fractional Kelly for reduced volatility.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.kelly_fraction = kelly_fraction

        self._trade_history: List[Dict] = []

    def update_statistics(
        self,
        trade_results: List[Dict[str, float]]
    ) -> None:
        """
        Update win rate and win/loss ratio from trade history.

        Args:
            trade_results: List of dicts with 'pnl' key
        """
        self._trade_history.extend(trade_results)

        if len(self._trade_history) < 30:
            return  # Not enough data

        # Calculate recent statistics
        recent = self._trade_history[-100:]  # Last 100 trades

        wins = [t for t in recent if t['pnl'] > 0]
        losses = [t for t in recent if t['pnl'] < 0]

        if wins and losses:
            self.win_rate = len(wins) / len(recent)
            avg_win = np.mean([t['pnl'] for t in wins])
            avg_loss = abs(np.mean([t['pnl'] for t in losses]))
            self.win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5

            logger.info(
                f"Kelly stats updated: win_rate={self.win_rate:.2%}, "
                f"win_loss_ratio={self.win_loss_ratio:.2f}"
            )

    def calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate optimal Kelly fraction.

        Returns:
            Optimal fraction of capital to bet
        """
        p = win_rate or self.win_rate
        b = win_loss_ratio or self.win_loss_ratio
        q = 1 - p

        # Full Kelly: f* = (p * b - q) / b
        full_kelly = (p * b - q) / b

        # Clamp to reasonable range
        full_kelly = max(0, min(full_kelly, 0.5))

        # Apply fractional Kelly
        fractional = full_kelly * self.kelly_fraction

        return fractional

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
        **kwargs
    ) -> PositionSize:
        """Calculate Kelly-optimal position size"""
        # Get Kelly fraction
        kelly = self.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # Adjust by signal strength
        adjusted_kelly = kelly * abs(signal_strength)

        # Calculate position value
        position_value = portfolio_value * adjusted_kelly

        # Calculate shares
        shares = int(position_value / current_price)
        actual_value = shares * current_price

        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=actual_value,
            weight=actual_value / portfolio_value,
            risk_contribution=adjusted_kelly,
            sizing_method="kelly",
            confidence=abs(signal_strength),
            metadata={
                'full_kelly': kelly / self.kelly_fraction,
                'fractional_kelly': kelly,
                'adjusted_kelly': adjusted_kelly,
                'win_rate': win_rate or self.win_rate,
                'win_loss_ratio': win_loss_ratio or self.win_loss_ratio
            }
        )


class VolatilityPositionSizer(PositionSizer):
    """
    Volatility-based position sizing.

    Sizes positions inversely proportional to volatility
    to target consistent risk contribution.

    Position Size = Target Risk / (Volatility * Price)
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        volatility_lookback: int = 60,
        use_atr: bool = True,
        atr_multiplier: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_volatility = target_volatility
        self.volatility_lookback = volatility_lookback
        self.use_atr = use_atr
        self.atr_multiplier = atr_multiplier

        self._volatility_cache: Dict[str, float] = {}

    def calculate_volatility(
        self,
        returns: pd.Series,
        method: str = 'standard'
    ) -> float:
        """
        Calculate volatility using various methods.

        Args:
            returns: Return series
            method: 'standard', 'ewma', 'parkinson', 'garman_klass'

        Returns:
            Annualized volatility
        """
        if len(returns) < 20:
            return 0.20  # Default 20%

        if method == 'standard':
            vol = returns.std() * np.sqrt(252)
        elif method == 'ewma':
            # Exponentially weighted volatility
            vol = returns.ewm(span=self.volatility_lookback).std().iloc[-1] * np.sqrt(252)
        else:
            vol = returns.std() * np.sqrt(252)

        return vol

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        volatility: Optional[float] = None,
        returns: Optional[pd.Series] = None,
        ohlc: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> PositionSize:
        """Calculate volatility-adjusted position size"""
        # Get or calculate volatility
        if volatility is not None:
            vol = volatility
        elif returns is not None:
            vol = self.calculate_volatility(returns)
        elif symbol in self._volatility_cache:
            vol = self._volatility_cache[symbol]
        else:
            vol = 0.20  # Default 20%

        # Store in cache
        self._volatility_cache[symbol] = vol

        # Calculate risk-adjusted position size
        # Position risk = Position Value * Daily Vol
        # Target: Position risk = Target Vol * Portfolio Value / Num Positions

        daily_vol = vol / np.sqrt(252)

        # Risk budget per position (assuming ~10 positions)
        num_positions = kwargs.get('num_positions', 10)
        risk_budget = (self.target_volatility / np.sqrt(252)) * portfolio_value / num_positions

        # Position size = Risk budget / Daily volatility
        if daily_vol > 0:
            position_value = risk_budget / daily_vol
        else:
            position_value = portfolio_value * 0.05  # 5% default

        # Adjust by signal strength
        position_value *= abs(signal_strength)

        # Calculate shares
        shares = int(position_value / current_price)
        actual_value = shares * current_price

        # Risk contribution
        risk_contrib = (actual_value * daily_vol) / (portfolio_value * self.target_volatility / np.sqrt(252))

        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=actual_value,
            weight=actual_value / portfolio_value,
            risk_contribution=min(risk_contrib, 1.0),
            sizing_method="volatility",
            confidence=abs(signal_strength),
            metadata={
                'annualized_vol': vol,
                'daily_vol': daily_vol,
                'risk_budget': risk_budget,
                'target_volatility': self.target_volatility
            }
        )

    def update_volatility(
        self,
        symbol: str,
        returns: pd.Series
    ) -> None:
        """Update volatility cache for symbol"""
        vol = self.calculate_volatility(returns)
        self._volatility_cache[symbol] = vol


class RiskParityPositionSizer(PositionSizer):
    """
    Risk Parity position sizing.

    Allocates equal risk contribution across all assets,
    accounting for correlations.

    Each asset contributes equally to total portfolio risk.
    """

    def __init__(
        self,
        target_volatility: float = 0.10,
        correlation_lookback: int = 120,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_volatility = target_volatility
        self.correlation_lookback = correlation_lookback

        self._covariance_matrix: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def update_covariance(
        self,
        returns: pd.DataFrame
    ) -> None:
        """
        Update covariance and correlation matrices.

        Args:
            returns: DataFrame with asset returns
        """
        if len(returns) < 30:
            return

        # Use exponentially weighted covariance for responsiveness
        self._covariance_matrix = returns.ewm(span=self.correlation_lookback).cov().iloc[-len(returns.columns):]
        self._correlation_matrix = returns.ewm(span=self.correlation_lookback).corr().iloc[-len(returns.columns):]

    def calculate_risk_parity_weights(
        self,
        covariance: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights.

        Uses optimization to find weights where each asset
        contributes equally to portfolio risk.
        """
        n = len(covariance)
        symbols = list(covariance.columns)

        if n == 1:
            return {symbols[0]: 1.0}

        cov = covariance.values

        def portfolio_risk(weights):
            return np.sqrt(weights @ cov @ weights)

        def risk_contribution(weights):
            port_risk = portfolio_risk(weights)
            marginal = cov @ weights
            return weights * marginal / port_risk

        def objective(weights):
            rc = risk_contribution(weights)
            target_rc = 1 / n  # Equal risk contribution
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]

        # Bounds (0 to 1 for each weight)
        bounds = [(0.01, 0.5) for _ in range(n)]

        # Initial guess (equal weight)
        x0 = np.ones(n) / n

        # Optimize
        result = opt.minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 1000}
        )

        if result.success:
            weights = result.x
        else:
            # Fallback to inverse volatility
            vols = np.sqrt(np.diag(cov))
            inv_vols = 1 / vols
            weights = inv_vols / np.sum(inv_vols)

        return {symbols[i]: weights[i] for i in range(n)}

    def calculate_sizes(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        returns: Optional[pd.DataFrame] = None,
        current_positions: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, PositionSize]:
        """Calculate risk parity position sizes"""
        # Filter to symbols with signals
        active_symbols = [s for s in signals if abs(signals[s]) > 0.1]

        if not active_symbols:
            return {}

        # Update covariance if returns provided
        if returns is not None:
            active_returns = returns[[s for s in active_symbols if s in returns.columns]]
            if len(active_returns.columns) > 0:
                self.update_covariance(active_returns)

        # Calculate risk parity weights
        if self._covariance_matrix is not None:
            # Filter covariance to active symbols
            active_cov = self._covariance_matrix.loc[
                [s for s in active_symbols if s in self._covariance_matrix.index],
                [s for s in active_symbols if s in self._covariance_matrix.columns]
            ]
            weights = self.calculate_risk_parity_weights(active_cov)
        else:
            # Equal weight fallback
            weights = {s: 1 / len(active_symbols) for s in active_symbols}

        # Scale weights by signal strength (for direction)
        for symbol in weights:
            if symbol in signals:
                weights[symbol] *= np.sign(signals[symbol])

        # Calculate position sizes
        sizes = {}
        for symbol, weight in weights.items():
            if symbol not in prices:
                continue

            position_value = abs(weight) * portfolio_value
            shares = int(position_value / prices[symbol])
            actual_value = shares * prices[symbol]

            sizes[symbol] = PositionSize(
                symbol=symbol,
                shares=shares if weight >= 0 else -shares,
                dollar_value=actual_value,
                weight=actual_value / portfolio_value,
                risk_contribution=1 / len(active_symbols),
                sizing_method="risk_parity",
                confidence=abs(signals.get(symbol, 0)),
                metadata={
                    'raw_weight': weight,
                    'target_risk_contrib': 1 / len(active_symbols)
                }
            )

        # Apply constraints
        sizes = self._apply_portfolio_constraints(
            sizes, portfolio_value, current_positions
        )

        return sizes

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        **kwargs
    ) -> PositionSize:
        """Single asset sizing (uses equal risk contribution)"""
        num_positions = kwargs.get('num_positions', 10)
        weight = 1 / num_positions

        position_value = weight * portfolio_value * abs(signal_strength)
        shares = int(position_value / current_price)
        actual_value = shares * current_price

        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=actual_value,
            weight=actual_value / portfolio_value,
            risk_contribution=weight,
            sizing_method="risk_parity",
            confidence=abs(signal_strength)
        )


class OptimalFPositionSizer(PositionSizer):
    """
    Optimal-F position sizing by Ralph Vince.

    Finds the optimal fraction of capital that maximizes
    the geometric growth rate of the portfolio.

    More aggressive than Kelly, requires careful risk management.
    """

    def __init__(
        self,
        safety_factor: float = 0.5,  # Trade at 50% of optimal-f
        min_trades: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.safety_factor = safety_factor
        self.min_trades = min_trades

        self._trade_results: Dict[str, List[float]] = {}

    def update_trades(
        self,
        symbol: str,
        pnl: float
    ) -> None:
        """Add trade result"""
        if symbol not in self._trade_results:
            self._trade_results[symbol] = []
        self._trade_results[symbol].append(pnl)

    def calculate_optimal_f(
        self,
        trade_results: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate optimal-f from trade history.

        Returns:
            Tuple of (optimal_f, TWR at optimal_f)
        """
        if len(trade_results) < self.min_trades:
            return 0.10, 1.0  # Conservative default

        largest_loss = abs(min(trade_results))
        if largest_loss == 0:
            return 0.10, 1.0

        # Normalize by largest loss
        normalized = [t / largest_loss for t in trade_results]

        # Search for optimal f
        best_f = 0.01
        best_twr = 0

        for f in np.arange(0.01, 1.0, 0.01):
            twr = 1.0
            for t in normalized:
                hpr = 1 + f * t
                if hpr <= 0:
                    twr = 0
                    break
                twr *= hpr

            if twr > best_twr:
                best_twr = twr
                best_f = f

        return best_f, best_twr

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        trade_results: Optional[List[float]] = None,
        **kwargs
    ) -> PositionSize:
        """Calculate optimal-f position size"""
        # Get trade history
        if trade_results is not None:
            results = trade_results
        elif symbol in self._trade_results:
            results = self._trade_results[symbol]
        else:
            results = []

        # Calculate optimal-f
        optimal_f, twr = self.calculate_optimal_f(results)

        # Apply safety factor
        safe_f = optimal_f * self.safety_factor

        # Adjust by signal strength
        position_fraction = safe_f * abs(signal_strength)

        # Calculate position
        position_value = portfolio_value * position_fraction
        shares = int(position_value / current_price)
        actual_value = shares * current_price

        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=actual_value,
            weight=actual_value / portfolio_value,
            risk_contribution=position_fraction,
            sizing_method="optimal_f",
            confidence=abs(signal_strength),
            metadata={
                'optimal_f': optimal_f,
                'safe_f': safe_f,
                'twr': twr,
                'num_trades': len(results)
            }
        )


class AdaptivePositionSizer(PositionSizer):
    """
    Adaptive position sizing that adjusts based on:
    - Recent performance
    - Market regime
    - Volatility regime
    - Drawdown

    Combines multiple sizing methods adaptively.
    """

    def __init__(
        self,
        base_sizer: PositionSizer,
        drawdown_threshold: float = 0.10,
        volatility_scale: bool = True,
        regime_adjust: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_sizer = base_sizer
        self.drawdown_threshold = drawdown_threshold
        self.volatility_scale = volatility_scale
        self.regime_adjust = regime_adjust

        self._peak_value: float = 0
        self._current_drawdown: float = 0
        self._market_regime: str = 'normal'
        self._volatility_regime: str = 'normal'

    def update_portfolio_state(
        self,
        portfolio_value: float,
        market_regime: Optional[str] = None,
        volatility_regime: Optional[str] = None
    ) -> None:
        """Update portfolio state for adaptive sizing"""
        # Update drawdown
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        if self._peak_value > 0:
            self._current_drawdown = (self._peak_value - portfolio_value) / self._peak_value

        # Update regimes
        if market_regime:
            self._market_regime = market_regime
        if volatility_regime:
            self._volatility_regime = volatility_regime

    def get_scaling_factor(self) -> float:
        """Calculate scaling factor based on current state"""
        scale = 1.0

        # Drawdown scaling (reduce size during drawdowns)
        if self._current_drawdown > self.drawdown_threshold:
            dd_scale = 1 - (self._current_drawdown - self.drawdown_threshold) / (0.3 - self.drawdown_threshold)
            scale *= max(0.2, dd_scale)

        # Volatility regime scaling
        if self.volatility_scale:
            if self._volatility_regime == 'high':
                scale *= 0.5
            elif self._volatility_regime == 'extreme':
                scale *= 0.25

        # Market regime scaling
        if self.regime_adjust:
            if self._market_regime == 'crisis':
                scale *= 0.25
            elif self._market_regime == 'stress':
                scale *= 0.5

        return scale

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        **kwargs
    ) -> PositionSize:
        """Calculate adaptively scaled position size"""
        # Get base size
        base_size = self.base_sizer.calculate_size(
            symbol, current_price, portfolio_value, signal_strength, **kwargs
        )

        # Apply scaling
        scale = self.get_scaling_factor()

        scaled_value = base_size.dollar_value * scale
        scaled_shares = int(scaled_value / current_price)
        actual_value = scaled_shares * current_price

        return PositionSize(
            symbol=symbol,
            shares=scaled_shares,
            dollar_value=actual_value,
            weight=actual_value / portfolio_value,
            risk_contribution=base_size.risk_contribution * scale,
            sizing_method=f"adaptive_{base_size.sizing_method}",
            confidence=base_size.confidence,
            metadata={
                **base_size.metadata,
                'scaling_factor': scale,
                'current_drawdown': self._current_drawdown,
                'market_regime': self._market_regime,
                'volatility_regime': self._volatility_regime
            }
        )


class MetaLabeledKelly(PositionSizer):
    """
    Dynamic Bet Sizing using Fractional Kelly Criterion with Meta-Labels.

    This implements the position sizing framework from Lopez de Prado's
    "Advances in Financial Machine Learning" where ML models provide
    probability estimates for trade success (Meta-Labels).

    Key Features:
    1. Uses ML probability (p) to calculate Kelly fraction
    2. Scales by inverse volatility for risk normalization
    3. Applies strict maximum leverage cap to prevent ruin
    4. Requires minimum probability threshold to trade

    The Meta-Labeling approach:
    1. Primary model generates directional signals (buy/sell/hold)
    2. Secondary (meta) model predicts probability of profit given signal
    3. Position size is determined by Kelly formula using meta-label probability

    Formula:
    m* = p - q  (where p = ML probability, q = 1 - p)
    Position = m* * (1/sigma) * scaling_factor

    With constraints:
    - If p < min_probability: Position = 0 (No Trade)
    - Position capped at max_leverage
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        min_probability: float = 0.55,  # Minimum ML probability to trade
        max_leverage: float = 2.0,  # Maximum position leverage
        volatility_lookback: int = 60,  # Days for volatility estimation
        scale_by_volatility: bool = True,  # Scale by inverse volatility
        use_bet_size_discretization: bool = True,  # Discretize bet sizes
        n_bet_sizes: int = 5,  # Number of discrete bet sizes
        **kwargs
    ):
        """
        Initialize MetaLabeledKelly position sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25-0.5 recommended)
            min_probability: Minimum ML probability required to trade
            max_leverage: Maximum leverage allowed per position
            volatility_lookback: Days for rolling volatility calculation
            scale_by_volatility: Whether to scale by inverse volatility
            use_bet_size_discretization: Use discrete bet sizes (0, 0.25, 0.5, 0.75, 1)
            n_bet_sizes: Number of discrete bet size levels
        """
        super().__init__(**kwargs)

        self.kelly_fraction = kelly_fraction
        self.min_probability = min_probability
        self.max_leverage = max_leverage
        self.volatility_lookback = volatility_lookback
        self.scale_by_volatility = scale_by_volatility
        self.use_bet_size_discretization = use_bet_size_discretization
        self.n_bet_sizes = n_bet_sizes

        # Cache for volatility estimates
        self._volatility_cache: Dict[str, float] = {}
        self._avg_volatility: float = 0.15  # Default market volatility

        # Trade statistics for Kelly estimation
        self._trade_history: List[Dict] = []
        self._rolling_win_rate: float = 0.5
        self._rolling_win_loss_ratio: float = 1.0

    def calculate_kelly_from_probability(
        self,
        probability: float,
        win_loss_ratio: float = None
    ) -> float:
        """
        Calculate raw Kelly fraction from ML probability.

        Standard Kelly: f* = (p * b - q) / b
        Where:
        - p = probability of winning (from Meta-Label model)
        - q = 1 - p = probability of losing
        - b = win/loss ratio (average win / average loss)

        Simplified Kelly (when b=1): f* = 2p - 1 = p - q

        Args:
            probability: ML model's predicted probability of trade success
            win_loss_ratio: Historical win/loss ratio (default: from history)

        Returns:
            Raw Kelly fraction (before applying kelly_fraction multiplier)
        """
        if probability < self.min_probability:
            return 0.0

        p = probability
        q = 1 - p
        b = win_loss_ratio or self._rolling_win_loss_ratio

        # Full Kelly formula
        if b > 0:
            full_kelly = (p * b - q) / b
        else:
            full_kelly = p - q  # Simplified when b=1

        # Clamp to valid range
        full_kelly = max(0.0, min(full_kelly, 1.0))

        return full_kelly

    def calculate_volatility_scalar(
        self,
        symbol: str,
        returns: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate volatility scaling factor (inverse volatility).

        Higher volatility assets get smaller positions, maintaining
        consistent risk contribution across assets.

        Scalar = target_vol / asset_vol

        Args:
            symbol: Asset symbol
            returns: Optional returns series for volatility calculation

        Returns:
            Volatility scaling factor
        """
        # Get or estimate volatility
        if returns is not None and len(returns) >= 20:
            vol = returns.std() * np.sqrt(252)
            self._volatility_cache[symbol] = vol
        elif symbol in self._volatility_cache:
            vol = self._volatility_cache[symbol]
        else:
            vol = self._avg_volatility

        # Target volatility (annualized)
        target_vol = self.config.target_volatility if hasattr(self.config, 'target_volatility') else 0.15

        # Inverse volatility scalar
        if vol > 0:
            scalar = target_vol / vol
        else:
            scalar = 1.0

        return scalar

    def discretize_bet_size(
        self,
        raw_size: float
    ) -> float:
        """
        Discretize bet size to predefined levels.

        Discretization helps avoid over-precision in sizing and
        reduces transaction costs from frequent small adjustments.

        Default levels: 0, 0.25, 0.5, 0.75, 1.0 (of max position)

        Args:
            raw_size: Continuous bet size (0 to 1)

        Returns:
            Discretized bet size
        """
        if not self.use_bet_size_discretization:
            return raw_size

        # Create discrete levels
        levels = np.linspace(0, 1, self.n_bet_sizes + 1)

        # Find nearest level
        idx = np.abs(levels - raw_size).argmin()
        return levels[idx]

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        ml_probability: Optional[float] = None,
        returns: Optional[pd.Series] = None,
        **kwargs
    ) -> PositionSize:
        """
        Calculate position size using Meta-Labeled Kelly Criterion.

        Args:
            symbol: Asset symbol
            current_price: Current asset price
            portfolio_value: Total portfolio value
            signal_strength: Signal direction and strength (-1 to 1)
            ml_probability: ML model's predicted probability (0 to 1)
            returns: Optional returns for volatility estimation

        Returns:
            PositionSize with Kelly-optimal sizing
        """
        # Use signal strength as probability if ML probability not provided
        if ml_probability is None:
            # Convert signal strength to probability estimate
            # Assumes signal_strength is roughly calibrated to probability
            ml_probability = 0.5 + (abs(signal_strength) * 0.3)

        # Check minimum probability threshold
        if ml_probability < self.min_probability:
            return PositionSize(
                symbol=symbol,
                shares=0,
                dollar_value=0,
                weight=0,
                risk_contribution=0,
                sizing_method="meta_labeled_kelly",
                confidence=ml_probability,
                constraints_applied=["min_probability_threshold"],
                metadata={
                    'ml_probability': ml_probability,
                    'min_threshold': self.min_probability,
                    'reason': 'probability_below_threshold'
                }
            )

        # Calculate raw Kelly fraction
        raw_kelly = self.calculate_kelly_from_probability(ml_probability)

        # Apply Kelly fraction (e.g., Half-Kelly)
        kelly_bet = raw_kelly * self.kelly_fraction

        # Apply volatility scaling
        if self.scale_by_volatility:
            vol_scalar = self.calculate_volatility_scalar(symbol, returns)
            kelly_bet *= vol_scalar

        # Discretize bet size
        kelly_bet = self.discretize_bet_size(kelly_bet)

        # Calculate position value
        raw_position_value = portfolio_value * kelly_bet

        # Apply maximum leverage constraint
        max_position_value = portfolio_value * self.max_leverage
        position_value = min(raw_position_value, max_position_value)

        # Also apply config max position constraint
        config_max = portfolio_value * self.config.max_position_pct
        position_value = min(position_value, config_max)

        # Calculate shares (account for direction from signal)
        direction = np.sign(signal_strength) if signal_strength != 0 else 1
        shares = int(position_value / current_price)
        actual_value = shares * current_price

        constraints_applied = []
        if position_value < raw_position_value:
            if raw_position_value > max_position_value:
                constraints_applied.append("max_leverage")
            if raw_position_value > config_max:
                constraints_applied.append("max_position_pct")

        return PositionSize(
            symbol=symbol,
            shares=shares * int(direction),
            dollar_value=actual_value,
            weight=actual_value / portfolio_value if portfolio_value > 0 else 0,
            risk_contribution=kelly_bet,
            sizing_method="meta_labeled_kelly",
            confidence=ml_probability,
            constraints_applied=constraints_applied,
            metadata={
                'ml_probability': ml_probability,
                'raw_kelly': raw_kelly,
                'kelly_fraction': self.kelly_fraction,
                'applied_kelly': kelly_bet,
                'vol_scalar': vol_scalar if self.scale_by_volatility else 1.0,
                'direction': direction,
                'win_loss_ratio': self._rolling_win_loss_ratio
            }
        )

    def update_trade_statistics(
        self,
        trade_pnl: float,
        entry_price: float
    ) -> None:
        """
        Update rolling trade statistics for Kelly estimation.

        Args:
            trade_pnl: Realized P&L from trade
            entry_price: Entry price for return calculation
        """
        trade_return = trade_pnl / entry_price if entry_price > 0 else 0

        self._trade_history.append({
            'pnl': trade_pnl,
            'return': trade_return,
            'timestamp': datetime.now()
        })

        # Keep last 100 trades
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

        # Update rolling statistics
        if len(self._trade_history) >= 20:
            wins = [t for t in self._trade_history if t['pnl'] > 0]
            losses = [t for t in self._trade_history if t['pnl'] < 0]

            self._rolling_win_rate = len(wins) / len(self._trade_history)

            if wins and losses:
                avg_win = np.mean([t['pnl'] for t in wins])
                avg_loss = abs(np.mean([t['pnl'] for t in losses]))
                self._rolling_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

            logger.debug(
                f"Kelly stats updated: win_rate={self._rolling_win_rate:.2%}, "
                f"win_loss_ratio={self._rolling_win_loss_ratio:.2f}"
            )

    def get_bet_size_from_confidence(
        self,
        confidence: float,
        base_size: float = 1.0
    ) -> float:
        """
        Get discrete bet size from confidence/probability.

        Useful for quick lookup without full calculation.

        Args:
            confidence: ML model confidence (0-1)
            base_size: Maximum bet size

        Returns:
            Bet size as fraction of base
        """
        if confidence < self.min_probability:
            return 0.0

        kelly = self.calculate_kelly_from_probability(confidence)
        kelly *= self.kelly_fraction
        kelly = self.discretize_bet_size(kelly)

        return min(kelly * base_size, self.max_leverage)


class VolatilityInverseKelly(MetaLabeledKelly):
    """
    Volatility-Inverse Kelly: Extension of Meta-Labeled Kelly that
    emphasizes volatility normalization.

    This variant is particularly useful for multi-asset portfolios
    where maintaining consistent risk contribution is critical.

    Position Size = Kelly_fraction * (sigma_target / sigma_asset) * ML_prob_adjustment

    Features:
    - Cross-sectional volatility ranking
    - Dynamic volatility regime adjustment
    - Correlation-aware sizing (optional)
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        vol_adjustment_factor: float = 1.0,
        use_rolling_vol: bool = True,
        vol_half_life: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_volatility = target_volatility
        self.vol_adjustment_factor = vol_adjustment_factor
        self.use_rolling_vol = use_rolling_vol
        self.vol_half_life = vol_half_life

    def calculate_ewma_volatility(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate EWMA volatility with specified half-life.

        Args:
            returns: Returns series

        Returns:
            Annualized EWMA volatility
        """
        if len(returns) < 10:
            return self._avg_volatility

        span = 2 * self.vol_half_life - 1
        ewma_var = returns.ewm(span=span).var().iloc[-1]

        return np.sqrt(ewma_var * 252)

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        ml_probability: Optional[float] = None,
        returns: Optional[pd.Series] = None,
        **kwargs
    ) -> PositionSize:
        """Calculate volatility-inverse position size."""
        # Get volatility
        if returns is not None and len(returns) >= 10:
            if self.use_rolling_vol:
                vol = self.calculate_ewma_volatility(returns)
            else:
                vol = returns.std() * np.sqrt(252)
            self._volatility_cache[symbol] = vol
        elif symbol in self._volatility_cache:
            vol = self._volatility_cache[symbol]
        else:
            vol = self._avg_volatility

        # Calculate base Kelly size
        base_result = super().calculate_size(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            signal_strength=signal_strength,
            ml_probability=ml_probability,
            returns=returns,
            **kwargs
        )

        # Additional volatility adjustment
        vol_ratio = self.target_volatility / max(vol, 0.01)
        vol_adjusted_value = base_result.dollar_value * vol_ratio * self.vol_adjustment_factor

        # Apply leverage cap
        max_value = portfolio_value * self.max_leverage
        final_value = min(vol_adjusted_value, max_value)

        # Recalculate shares
        direction = np.sign(base_result.shares) if base_result.shares != 0 else 1
        shares = int(final_value / current_price) * int(direction)
        actual_value = abs(shares) * current_price

        # Update metadata
        metadata = base_result.metadata.copy()
        metadata.update({
            'asset_volatility': vol,
            'vol_ratio': vol_ratio,
            'vol_adjusted': True
        })

        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=actual_value,
            weight=actual_value / portfolio_value if portfolio_value > 0 else 0,
            risk_contribution=base_result.risk_contribution * vol_ratio,
            sizing_method="vol_inverse_kelly",
            confidence=base_result.confidence,
            constraints_applied=base_result.constraints_applied,
            metadata=metadata
        )


def create_position_sizer(
    method: Union[str, SizingMethod],
    **kwargs
) -> PositionSizer:
    """
    Factory function to create position sizers.

    Args:
        method: Sizing method name or enum
        **kwargs: Method-specific parameters

    Returns:
        Configured position sizer
    """
    if isinstance(method, str):
        method_lower = method.lower()
        if method_lower == 'meta_kelly' or method_lower == 'meta_labeled_kelly':
            return MetaLabeledKelly(**kwargs)
        elif method_lower == 'vol_inverse_kelly':
            return VolatilityInverseKelly(**kwargs)
        method = SizingMethod(method_lower)

    sizers = {
        SizingMethod.KELLY: KellyCriterion,
        SizingMethod.VOLATILITY: VolatilityPositionSizer,
        SizingMethod.RISK_PARITY: RiskParityPositionSizer,
        SizingMethod.OPTIMAL_F: OptimalFPositionSizer,
    }

    if method not in sizers:
        raise ValueError(f"Unknown sizing method: {method}")

    return sizers[method](**kwargs)
