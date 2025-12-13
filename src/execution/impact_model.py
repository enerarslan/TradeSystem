"""
Pre-Trade Market Impact Estimation
==================================
JPMorgan-Level Almgren-Chriss Market Impact Model

Estimates expected market impact BEFORE execution to:
1. Decide if trade is worth executing
2. Choose optimal execution strategy
3. Set realistic price expectations
4. Size positions accounting for impact

Market Impact Components:
- Permanent Impact: Information leakage, price doesn't revert
- Temporary Impact: Liquidity consumption, price reverts after execution
- Spread Cost: Bid-ask crossing cost

Models Implemented:
- Almgren-Chriss (2000): Classic optimal execution model
- Square-Root Model: Empirically validated impact model
- Kyle's Lambda: Information-based impact

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 5
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionUrgency(Enum):
    """Execution urgency level"""
    LOW = "low"           # Patient execution, minimize impact
    MEDIUM = "medium"     # Balance speed and impact
    HIGH = "high"         # Fast execution, accept higher impact
    IMMEDIATE = "immediate"  # Execute now regardless of impact


@dataclass
class ImpactEstimate:
    """Market impact estimation result"""
    # Impact components (in basis points)
    permanent_bps: float      # Won't revert
    temporary_bps: float      # Will revert after execution
    spread_cost_bps: float    # Bid-ask spread
    total_bps: float          # Total expected impact

    # Dollar amounts
    permanent_cost: float
    temporary_cost: float
    spread_cost: float
    total_cost: float

    # Optimal execution
    optimal_horizon_minutes: float
    recommended_strategy: str  # TWAP, VWAP, IS, etc.

    # Participation
    participation_rate: float
    adv_percentage: float  # % of average daily volume

    # Risk metrics
    execution_risk: float  # Uncertainty in impact estimate
    impact_confidence_interval: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'impact_bps': {
                'permanent': self.permanent_bps,
                'temporary': self.temporary_bps,
                'spread': self.spread_cost_bps,
                'total': self.total_bps
            },
            'cost_dollars': {
                'permanent': self.permanent_cost,
                'temporary': self.temporary_cost,
                'spread': self.spread_cost,
                'total': self.total_cost
            },
            'execution': {
                'optimal_horizon_minutes': self.optimal_horizon_minutes,
                'recommended_strategy': self.recommended_strategy,
                'participation_rate': self.participation_rate,
                'adv_percentage': self.adv_percentage
            },
            'risk': {
                'execution_risk': self.execution_risk,
                'ci_lower': self.impact_confidence_interval[0],
                'ci_upper': self.impact_confidence_interval[1]
            }
        }


@dataclass
class TradeDecision:
    """Decision on whether to execute trade"""
    should_execute: bool
    reason: str
    adjusted_quantity: Optional[int]  # Reduced quantity if needed
    expected_net_alpha: float  # Alpha minus impact
    impact_to_alpha_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'should_execute': self.should_execute,
            'reason': self.reason,
            'adjusted_quantity': self.adjusted_quantity,
            'expected_net_alpha': self.expected_net_alpha,
            'impact_to_alpha_ratio': self.impact_to_alpha_ratio
        }


class AlmgrenChrissModel:
    """
    Almgren-Chriss Optimal Execution Model.

    Classic model for estimating market impact and optimal execution.

    Reference:
    Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio
    transactions. Journal of Risk, 3, 5-40.

    Impact Model:
    - Permanent: gamma * (Q/V)^0.5 * sigma
    - Temporary: eta * (Q/V)^delta * sigma + spread/2

    Where:
    - Q = order quantity
    - V = average daily volume
    - sigma = daily volatility
    - gamma = permanent impact coefficient
    - eta = temporary impact coefficient
    - delta = temporary impact exponent (typically 0.5-0.8)
    """

    def __init__(
        self,
        # Impact coefficients (calibrate to your market)
        permanent_impact_coef: float = 0.1,    # gamma
        temporary_impact_coef: float = 0.2,    # eta
        temporary_impact_exp: float = 0.6,      # delta

        # Risk aversion for optimal horizon
        risk_aversion: float = 1e-6,

        # Default spread if not provided
        default_spread_bps: float = 5.0,

        # Trading hours per day
        trading_minutes_per_day: float = 390.0,  # 6.5 hours
    ):
        self.gamma = permanent_impact_coef
        self.eta = temporary_impact_coef
        self.delta = temporary_impact_exp
        self.risk_aversion = risk_aversion
        self.default_spread_bps = default_spread_bps
        self.trading_minutes = trading_minutes_per_day

    def estimate_impact(
        self,
        quantity: int,
        price: float,
        adv: float,
        volatility: float,
        spread_bps: Optional[float] = None,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    ) -> ImpactEstimate:
        """
        Estimate market impact for a trade.

        Args:
            quantity: Number of shares to trade
            price: Current price
            adv: Average daily volume (shares)
            volatility: Daily volatility (decimal, e.g., 0.02 for 2%)
            spread_bps: Bid-ask spread in bps (optional)
            urgency: Execution urgency level

        Returns:
            ImpactEstimate with full breakdown
        """
        # Validate inputs
        if adv <= 0:
            logger.warning("ADV is zero or negative, using minimum")
            adv = max(adv, quantity * 10)

        # Participation rate
        participation = quantity / adv
        adv_pct = participation * 100

        # Spread
        spread = spread_bps if spread_bps is not None else self.default_spread_bps
        spread_decimal = spread / 10000

        # Permanent impact: gamma * sqrt(participation) * volatility
        permanent_impact = self.gamma * np.sqrt(participation) * volatility

        # Temporary impact: eta * participation^delta * volatility + spread/2
        temporary_impact = self.eta * (participation ** self.delta) * volatility + spread_decimal / 2

        # Total impact
        total_impact = permanent_impact + temporary_impact

        # Convert to basis points
        permanent_bps = permanent_impact * 10000
        temporary_bps = temporary_impact * 10000
        spread_cost_bps = spread / 2  # Half spread on entry
        total_bps = total_impact * 10000

        # Dollar costs
        notional = quantity * price
        permanent_cost = permanent_impact * notional
        temporary_cost = temporary_impact * notional
        spread_cost = (spread_decimal / 2) * notional
        total_cost = total_impact * notional

        # Optimal execution horizon
        optimal_horizon = self._compute_optimal_horizon(
            quantity, adv, volatility, urgency
        )

        # Recommended strategy
        strategy = self._recommend_strategy(participation, optimal_horizon, urgency)

        # Execution risk (uncertainty in estimate)
        # Impact models have ~50% uncertainty
        execution_risk = total_bps * 0.5

        # Confidence interval (roughly 2 sigma)
        ci_lower = total_bps * 0.5
        ci_upper = total_bps * 2.0

        return ImpactEstimate(
            permanent_bps=permanent_bps,
            temporary_bps=temporary_bps,
            spread_cost_bps=spread_cost_bps,
            total_bps=total_bps,
            permanent_cost=permanent_cost,
            temporary_cost=temporary_cost,
            spread_cost=spread_cost,
            total_cost=total_cost,
            optimal_horizon_minutes=optimal_horizon,
            recommended_strategy=strategy,
            participation_rate=participation,
            adv_percentage=adv_pct,
            execution_risk=execution_risk,
            impact_confidence_interval=(ci_lower, ci_upper)
        )

    def _compute_optimal_horizon(
        self,
        quantity: int,
        adv: float,
        volatility: float,
        urgency: ExecutionUrgency
    ) -> float:
        """
        Compute optimal execution horizon.

        Balances:
        - Impact cost (shorter = higher impact)
        - Timing risk (longer = more variance)

        Almgren-Chriss optimal horizon:
        T* = (lambda * sigma^2 * Q) / (2 * eta)

        Where lambda is risk aversion.
        """
        # Urgency adjustments
        urgency_multipliers = {
            ExecutionUrgency.LOW: 2.0,
            ExecutionUrgency.MEDIUM: 1.0,
            ExecutionUrgency.HIGH: 0.5,
            ExecutionUrgency.IMMEDIATE: 0.1
        }
        urgency_mult = urgency_multipliers.get(urgency, 1.0)

        # Participation rate
        participation = quantity / adv

        # Base optimal horizon (in fraction of day)
        # Higher participation = longer horizon needed
        # Higher volatility = shorter horizon (timing risk)
        base_horizon_days = np.sqrt(participation) / (volatility + 0.01)

        # Convert to minutes and apply urgency
        horizon_minutes = base_horizon_days * self.trading_minutes * urgency_mult

        # Bounds
        horizon_minutes = max(5, min(horizon_minutes, self.trading_minutes))

        return horizon_minutes

    def _recommend_strategy(
        self,
        participation: float,
        horizon_minutes: float,
        urgency: ExecutionUrgency
    ) -> str:
        """Recommend execution strategy based on parameters"""
        if urgency == ExecutionUrgency.IMMEDIATE:
            return "MARKET"

        if participation < 0.01:  # < 1% of ADV
            if urgency == ExecutionUrgency.LOW:
                return "LIMIT"  # Patient, use limit orders
            else:
                return "MARKET"  # Small enough, just execute

        elif participation < 0.05:  # 1-5% of ADV
            return "TWAP"  # Time-weighted

        elif participation < 0.15:  # 5-15% of ADV
            return "VWAP"  # Volume-weighted

        else:  # > 15% of ADV
            if horizon_minutes > 60:
                return "IS"  # Implementation Shortfall algo
            else:
                return "POV"  # Percentage of Volume

    def should_execute(
        self,
        quantity: int,
        price: float,
        adv: float,
        volatility: float,
        expected_alpha_bps: float,
        spread_bps: Optional[float] = None,
        max_impact_to_alpha_ratio: float = 0.5
    ) -> TradeDecision:
        """
        Decide if trade should be executed given expected alpha.

        Rule: Don't trade if impact eats > 50% of expected alpha.

        Args:
            quantity: Shares to trade
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            expected_alpha_bps: Expected return in basis points
            spread_bps: Bid-ask spread
            max_impact_to_alpha_ratio: Max acceptable impact/alpha ratio

        Returns:
            TradeDecision
        """
        # Get impact estimate
        impact = self.estimate_impact(
            quantity, price, adv, volatility, spread_bps
        )

        # Calculate net alpha
        net_alpha = expected_alpha_bps - impact.total_bps

        # Impact to alpha ratio
        if expected_alpha_bps > 0:
            ratio = impact.total_bps / expected_alpha_bps
        else:
            ratio = float('inf')

        # Decision logic
        if ratio <= max_impact_to_alpha_ratio:
            # Trade is profitable after impact
            return TradeDecision(
                should_execute=True,
                reason=f"Impact ({impact.total_bps:.1f}bps) acceptable vs alpha ({expected_alpha_bps:.1f}bps)",
                adjusted_quantity=None,
                expected_net_alpha=net_alpha,
                impact_to_alpha_ratio=ratio
            )

        # Try reducing size
        reduction_factor = max_impact_to_alpha_ratio / ratio
        adjusted_qty = int(quantity * reduction_factor * 0.8)  # Extra buffer

        if adjusted_qty >= 100:  # Minimum viable size
            # Re-estimate with reduced size
            new_impact = self.estimate_impact(
                adjusted_qty, price, adv, volatility, spread_bps
            )
            new_ratio = new_impact.total_bps / expected_alpha_bps if expected_alpha_bps > 0 else float('inf')
            new_net_alpha = expected_alpha_bps - new_impact.total_bps

            return TradeDecision(
                should_execute=True,
                reason=f"Reduced size from {quantity} to {adjusted_qty} to manage impact",
                adjusted_quantity=adjusted_qty,
                expected_net_alpha=new_net_alpha,
                impact_to_alpha_ratio=new_ratio
            )

        # Don't trade
        return TradeDecision(
            should_execute=False,
            reason=f"Impact ({impact.total_bps:.1f}bps) exceeds {max_impact_to_alpha_ratio*100:.0f}% of alpha ({expected_alpha_bps:.1f}bps)",
            adjusted_quantity=None,
            expected_net_alpha=net_alpha,
            impact_to_alpha_ratio=ratio
        )


class SquareRootImpactModel:
    """
    Square-Root Market Impact Model.

    Empirically validated model that assumes:
    Impact ~ sigma * sqrt(Q/V)

    This is the "universal" impact model observed across
    many markets and time periods.

    Reference:
    Bouchaud, J.P. et al. (2009). How markets slowly digest
    changes in supply and demand.
    """

    def __init__(
        self,
        impact_coefficient: float = 0.5,  # Typical range: 0.1-1.0
        spread_coefficient: float = 0.5,
    ):
        self.impact_coef = impact_coefficient
        self.spread_coef = spread_coefficient

    def estimate_impact(
        self,
        quantity: int,
        price: float,
        adv: float,
        volatility: float,
        spread_bps: float = 5.0
    ) -> Dict[str, float]:
        """
        Simple square-root impact estimate.

        Impact = c * sigma * sqrt(Q/V)

        Args:
            quantity: Shares to trade
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            spread_bps: Bid-ask spread

        Returns:
            Dictionary with impact metrics
        """
        participation = quantity / adv

        # Square-root impact
        impact = self.impact_coef * volatility * np.sqrt(participation)

        # Add spread
        spread_decimal = spread_bps / 10000
        total_impact = impact + self.spread_coef * spread_decimal

        # Notional
        notional = quantity * price

        return {
            'impact_bps': impact * 10000,
            'spread_cost_bps': spread_bps * self.spread_coef,
            'total_impact_bps': total_impact * 10000,
            'total_cost_dollars': total_impact * notional,
            'participation_rate': participation
        }


class KyleLambdaModel:
    """
    Kyle's Lambda Impact Model.

    Based on Kyle (1985) model of informed trading.
    Lambda represents price impact per unit of order flow.

    Lambda = sigma / sqrt(V)

    Where informed traders move prices by revealing information.
    """

    def __init__(
        self,
        lambda_multiplier: float = 1.0,
    ):
        self.lambda_mult = lambda_multiplier

    def compute_kyle_lambda(
        self,
        volatility: float,
        volume: float
    ) -> float:
        """
        Compute Kyle's Lambda.

        Args:
            volatility: Price volatility
            volume: Trading volume

        Returns:
            Lambda value (price impact per share)
        """
        if volume <= 0:
            return 0.0

        return self.lambda_mult * volatility / np.sqrt(volume)

    def estimate_impact(
        self,
        quantity: int,
        price: float,
        volatility: float,
        volume: float
    ) -> Dict[str, float]:
        """
        Estimate impact using Kyle's Lambda.

        Impact = lambda * Q

        Args:
            quantity: Order size
            price: Current price
            volatility: Price volatility
            volume: Expected volume during execution

        Returns:
            Impact metrics
        """
        lambda_val = self.compute_kyle_lambda(volatility, volume)

        # Impact in price terms
        price_impact = lambda_val * quantity

        # Convert to bps
        impact_bps = (price_impact / price) * 10000

        # Dollar cost
        notional = quantity * price
        cost = price_impact * quantity

        return {
            'kyle_lambda': lambda_val,
            'price_impact': price_impact,
            'impact_bps': impact_bps,
            'total_cost': cost,
            'notional': notional
        }


# =============================================================================
# PRE-TRADE ANALYTICS
# =============================================================================

class PreTradeAnalytics:
    """
    Comprehensive pre-trade analysis.

    Combines multiple impact models and provides
    execution recommendations.
    """

    def __init__(
        self,
        almgren_chriss: Optional[AlmgrenChrissModel] = None,
        sqrt_model: Optional[SquareRootImpactModel] = None,
        kyle_model: Optional[KyleLambdaModel] = None
    ):
        self.ac_model = almgren_chriss or AlmgrenChrissModel()
        self.sqrt_model = sqrt_model or SquareRootImpactModel()
        self.kyle_model = kyle_model or KyleLambdaModel()

        # Market data cache
        self._adv_cache: Dict[str, float] = {}
        self._volatility_cache: Dict[str, float] = {}
        self._spread_cache: Dict[str, float] = {}

    def set_market_data(
        self,
        symbol: str,
        adv: float,
        volatility: float,
        spread_bps: float
    ) -> None:
        """Cache market data for a symbol"""
        self._adv_cache[symbol] = adv
        self._volatility_cache[symbol] = volatility
        self._spread_cache[symbol] = spread_bps

    def analyze_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        expected_alpha_bps: float = 0.0,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
        adv: Optional[float] = None,
        volatility: Optional[float] = None,
        spread_bps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive pre-trade analysis.

        Args:
            symbol: Trading symbol
            quantity: Shares to trade
            price: Current price
            side: 'buy' or 'sell'
            expected_alpha_bps: Expected return
            urgency: Execution urgency
            adv: Average daily volume (optional, uses cache)
            volatility: Daily volatility (optional, uses cache)
            spread_bps: Bid-ask spread (optional, uses cache)

        Returns:
            Comprehensive analysis dictionary
        """
        # Get market data from cache or parameters
        adv = adv or self._adv_cache.get(symbol, 1000000)
        volatility = volatility or self._volatility_cache.get(symbol, 0.02)
        spread = spread_bps or self._spread_cache.get(symbol, 5.0)

        # Almgren-Chriss estimate
        ac_estimate = self.ac_model.estimate_impact(
            quantity, price, adv, volatility, spread, urgency
        )

        # Square-root estimate
        sqrt_estimate = self.sqrt_model.estimate_impact(
            quantity, price, adv, volatility, spread
        )

        # Kyle estimate (use ADV as expected volume)
        kyle_estimate = self.kyle_model.estimate_impact(
            quantity, price, volatility, adv
        )

        # Should execute decision
        decision = self.ac_model.should_execute(
            quantity, price, adv, volatility,
            expected_alpha_bps, spread
        )

        # Combine estimates (conservative: use max)
        consensus_impact = max(
            ac_estimate.total_bps,
            sqrt_estimate['total_impact_bps'],
            kyle_estimate['impact_bps']
        )

        # Execution recommendation
        notional = quantity * price

        return {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'notional': notional,
            'market_data': {
                'adv': adv,
                'volatility': volatility,
                'spread_bps': spread,
                'adv_participation': quantity / adv
            },
            'impact_estimates': {
                'almgren_chriss': ac_estimate.to_dict(),
                'square_root': sqrt_estimate,
                'kyle': kyle_estimate,
                'consensus_bps': consensus_impact
            },
            'execution_recommendation': {
                'should_execute': decision.should_execute,
                'reason': decision.reason,
                'adjusted_quantity': decision.adjusted_quantity,
                'strategy': ac_estimate.recommended_strategy,
                'optimal_horizon_minutes': ac_estimate.optimal_horizon_minutes,
                'expected_net_alpha_bps': decision.expected_net_alpha
            },
            'cost_summary': {
                'expected_cost_bps': consensus_impact,
                'expected_cost_dollars': consensus_impact / 10000 * notional,
                'cost_as_pct_of_alpha': (
                    consensus_impact / expected_alpha_bps * 100
                    if expected_alpha_bps > 0 else float('inf')
                )
            }
        }

    def get_execution_schedule(
        self,
        symbol: str,
        quantity: int,
        price: float,
        horizon_minutes: float,
        strategy: str = "TWAP"
    ) -> List[Dict[str, Any]]:
        """
        Generate execution schedule.

        Args:
            symbol: Trading symbol
            quantity: Total shares to trade
            price: Current price
            horizon_minutes: Execution horizon
            strategy: TWAP, VWAP, etc.

        Returns:
            List of scheduled child orders
        """
        # Number of slices
        if strategy == "TWAP":
            # Equal time slices
            slice_interval = 5  # 5 minute intervals
            n_slices = max(1, int(horizon_minutes / slice_interval))
            slice_qty = quantity // n_slices
            remainder = quantity % n_slices

            schedule = []
            for i in range(n_slices):
                qty = slice_qty + (1 if i < remainder else 0)
                schedule.append({
                    'slice': i + 1,
                    'time_offset_minutes': i * slice_interval,
                    'quantity': qty,
                    'order_type': 'LIMIT',
                    'limit_offset_bps': 2  # 2 bps aggressive
                })

            return schedule

        elif strategy == "VWAP":
            # Volume-weighted slices (simplified U-shape)
            # More volume at open and close
            volume_profile = [0.15, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.12, 0.15, 0.20]
            n_slices = len(volume_profile)
            slice_interval = horizon_minutes / n_slices

            schedule = []
            remaining = quantity
            for i, vol_pct in enumerate(volume_profile):
                qty = int(quantity * vol_pct)
                if i == n_slices - 1:
                    qty = remaining  # Last slice gets remainder
                remaining -= qty

                schedule.append({
                    'slice': i + 1,
                    'time_offset_minutes': i * slice_interval,
                    'quantity': qty,
                    'volume_target_pct': vol_pct,
                    'order_type': 'LIMIT',
                    'limit_offset_bps': 3
                })

            return schedule

        else:
            # Default: single order
            return [{
                'slice': 1,
                'time_offset_minutes': 0,
                'quantity': quantity,
                'order_type': 'MARKET'
            }]


# =============================================================================
# IMPACT TRACKER
# =============================================================================

class ImpactTracker:
    """
    Tracks realized vs predicted impact for model calibration.

    Compares pre-trade estimates to actual execution costs
    to improve future predictions.
    """

    def __init__(self, max_history: int = 1000):
        self._predictions: List[Dict] = []
        self._actuals: List[Dict] = []
        self.max_history = max_history

    def record_prediction(
        self,
        trade_id: str,
        symbol: str,
        predicted_impact_bps: float,
        quantity: int,
        price: float
    ) -> None:
        """Record pre-trade prediction"""
        self._predictions.append({
            'trade_id': trade_id,
            'symbol': symbol,
            'timestamp': datetime.now(),
            'predicted_bps': predicted_impact_bps,
            'quantity': quantity,
            'price': price
        })

        # Trim history
        if len(self._predictions) > self.max_history:
            self._predictions = self._predictions[-self.max_history:]

    def record_actual(
        self,
        trade_id: str,
        avg_fill_price: float,
        decision_price: float
    ) -> None:
        """Record actual execution result"""
        # Find prediction
        pred = next((p for p in self._predictions if p['trade_id'] == trade_id), None)

        if pred is None:
            return

        # Calculate actual impact
        actual_impact = (avg_fill_price - decision_price) / decision_price * 10000

        self._actuals.append({
            'trade_id': trade_id,
            'symbol': pred['symbol'],
            'predicted_bps': pred['predicted_bps'],
            'actual_bps': abs(actual_impact),
            'prediction_error': abs(actual_impact) - pred['predicted_bps'],
            'timestamp': datetime.now()
        })

        # Trim history
        if len(self._actuals) > self.max_history:
            self._actuals = self._actuals[-self.max_history:]

    def get_model_accuracy(self) -> Dict[str, float]:
        """Get impact model accuracy metrics"""
        if not self._actuals:
            return {'error': 'No data'}

        errors = [a['prediction_error'] for a in self._actuals]
        pct_errors = [
            a['prediction_error'] / a['predicted_bps'] * 100
            if a['predicted_bps'] > 0 else 0
            for a in self._actuals
        ]

        return {
            'n_trades': len(self._actuals),
            'mean_error_bps': np.mean(errors),
            'median_error_bps': np.median(errors),
            'rmse_bps': np.sqrt(np.mean(np.array(errors) ** 2)),
            'mean_pct_error': np.mean(pct_errors),
            'underestimate_rate': sum(1 for e in errors if e > 0) / len(errors)
        }

    def get_calibration_factors(self) -> Dict[str, float]:
        """Get suggested calibration factors by symbol"""
        if not self._actuals:
            return {}

        by_symbol: Dict[str, List[float]] = {}
        for a in self._actuals:
            symbol = a['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []

            if a['predicted_bps'] > 0:
                ratio = a['actual_bps'] / a['predicted_bps']
                by_symbol[symbol].append(ratio)

        return {
            symbol: np.median(ratios)
            for symbol, ratios in by_symbol.items()
            if len(ratios) >= 10
        }
