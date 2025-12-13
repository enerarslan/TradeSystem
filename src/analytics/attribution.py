"""
P&L Attribution System
======================
JPMorgan-Level Trade and Portfolio P&L Decomposition

This module decomposes P&L into actionable components:

1. **Cost Attribution**:
   - Commission costs
   - Slippage costs
   - Spread costs
   - Market impact costs

2. **Alpha Attribution**:
   - Direction alpha (was signal correct?)
   - Timing alpha (did we enter/exit well?)
   - Sizing alpha (did we size correctly?)

3. **Risk Attribution**:
   - Factor exposures
   - Risk-adjusted returns
   - Drawdown attribution

Why Attribution Matters:
- Understand WHERE profits come from
- Identify which costs are eroding edge
- Optimize sizing based on alpha sources
- Detect strategy decay early

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TradeSide(Enum):
    """Trade side"""
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """Why the position was closed"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    LIQUIDATION = "liquidation"


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    symbol: str
    strategy: str
    side: TradeSide

    # Entry
    entry_time: datetime
    entry_price: float
    entry_quantity: int
    entry_signal_strength: float = 1.0

    # Exit
    exit_time: datetime
    exit_price: float
    exit_quantity: int
    exit_reason: ExitReason = ExitReason.SIGNAL_REVERSAL

    # Costs
    commission: float = 0.0
    slippage_bps: float = 0.0

    # Risk at entry
    risk_at_entry: float = 0.0  # Dollar risk (position * volatility)
    stop_loss_pct: float = 0.0

    # Market context
    entry_vwap: float = 0.0  # Market VWAP at entry
    exit_vwap: float = 0.0   # Market VWAP at exit
    entry_spread_bps: float = 0.0
    exit_spread_bps: float = 0.0

    @property
    def realized_pnl(self) -> float:
        """Gross realized P&L"""
        if self.side == TradeSide.LONG:
            return (self.exit_price - self.entry_price) * self.exit_quantity
        else:
            return (self.entry_price - self.exit_price) * self.exit_quantity

    @property
    def net_pnl(self) -> float:
        """Net P&L after costs"""
        return self.realized_pnl - self.commission

    @property
    def holding_period_hours(self) -> float:
        """Time held"""
        return (self.exit_time - self.entry_time).total_seconds() / 3600

    @property
    def return_pct(self) -> float:
        """Percentage return"""
        entry_notional = self.entry_price * self.entry_quantity
        if entry_notional == 0:
            return 0.0
        return self.realized_pnl / entry_notional


@dataclass
class TradeAttribution:
    """Attribution breakdown for a single trade"""
    trade_id: str
    symbol: str
    strategy: str

    # Gross P&L
    gross_pnl: float

    # Cost breakdown
    commission_cost: float
    slippage_cost: float
    spread_cost: float
    market_impact_cost: float

    # Alpha sources
    direction_alpha: float  # P&L from correct direction
    timing_alpha: float     # P&L from entry/exit timing vs VWAP
    sizing_alpha: float     # Extra/lost P&L from sizing

    # Risk-adjusted
    risk_adjusted_pnl: float  # P&L / risk at entry
    sharpe_contribution: float  # Contribution to portfolio Sharpe

    # Net P&L
    net_pnl: float

    # Metadata
    holding_period_hours: float
    exit_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'strategy': self.strategy,
            'gross_pnl': self.gross_pnl,
            'costs': {
                'commission': self.commission_cost,
                'slippage': self.slippage_cost,
                'spread': self.spread_cost,
                'market_impact': self.market_impact_cost,
                'total': self.commission_cost + self.slippage_cost + self.spread_cost + self.market_impact_cost
            },
            'alpha': {
                'direction': self.direction_alpha,
                'timing': self.timing_alpha,
                'sizing': self.sizing_alpha,
                'total': self.direction_alpha + self.timing_alpha + self.sizing_alpha
            },
            'risk_adjusted': {
                'pnl_per_unit_risk': self.risk_adjusted_pnl,
                'sharpe_contribution': self.sharpe_contribution
            },
            'net_pnl': self.net_pnl,
            'holding_period_hours': self.holding_period_hours,
            'exit_reason': self.exit_reason
        }


@dataclass
class PortfolioAttribution:
    """Attribution for entire portfolio over a period"""
    period_start: datetime
    period_end: datetime
    total_trades: int

    # Aggregate P&L
    gross_pnl: float
    total_costs: float
    net_pnl: float

    # Cost breakdown
    total_commission: float
    total_slippage: float
    total_spread: float
    total_impact: float

    # Alpha breakdown
    total_direction_alpha: float
    total_timing_alpha: float
    total_sizing_alpha: float

    # By strategy
    by_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # By symbol
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # By exit reason
    by_exit_reason: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Risk metrics
    risk_adjusted_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat()
            },
            'summary': {
                'total_trades': self.total_trades,
                'gross_pnl': self.gross_pnl,
                'total_costs': self.total_costs,
                'net_pnl': self.net_pnl
            },
            'costs': {
                'commission': self.total_commission,
                'slippage': self.total_slippage,
                'spread': self.total_spread,
                'market_impact': self.total_impact
            },
            'alpha': {
                'direction': self.total_direction_alpha,
                'timing': self.total_timing_alpha,
                'sizing': self.total_sizing_alpha
            },
            'by_strategy': self.by_strategy,
            'by_symbol': self.by_symbol,
            'by_exit_reason': self.by_exit_reason,
            'risk_metrics': {
                'risk_adjusted_return': self.risk_adjusted_return,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor
            }
        }


@dataclass
class AttributionReport:
    """Full attribution report with analysis"""
    portfolio: PortfolioAttribution
    trades: List[TradeAttribution]
    insights: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'portfolio': self.portfolio.to_dict(),
            'trades': [t.to_dict() for t in self.trades],
            'insights': self.insights,
            'recommendations': self.recommendations
        }


class PnLAttribution:
    """
    P&L Attribution Engine.

    Decomposes trade and portfolio P&L into:
    1. Cost components (commission, slippage, spread, impact)
    2. Alpha sources (direction, timing, sizing)
    3. Risk-adjusted metrics
    """

    def __init__(
        self,
        # Default cost assumptions
        default_spread_bps: float = 2.0,
        default_impact_bps: float = 1.0,

        # Sizing baseline for alpha calculation
        baseline_position_pct: float = 0.02,  # 2% baseline position

        # Risk-free rate for Sharpe
        risk_free_rate: float = 0.05,
    ):
        self.default_spread_bps = default_spread_bps
        self.default_impact_bps = default_impact_bps
        self.baseline_position_pct = baseline_position_pct
        self.risk_free_rate = risk_free_rate

        # Store trades
        self._trades: List[Trade] = []
        self._attributions: List[TradeAttribution] = []

    def add_trade(self, trade: Trade) -> TradeAttribution:
        """
        Add a trade and compute its attribution.

        Args:
            trade: Completed trade

        Returns:
            TradeAttribution for the trade
        """
        attribution = self.attribute_trade(trade)

        self._trades.append(trade)
        self._attributions.append(attribution)

        return attribution

    def attribute_trade(self, trade: Trade) -> TradeAttribution:
        """
        Decompose a single trade's P&L.

        Args:
            trade: Trade to attribute

        Returns:
            TradeAttribution with full breakdown
        """
        # 1. Calculate costs
        commission_cost = trade.commission
        slippage_cost = self._calculate_slippage_cost(trade)
        spread_cost = self._estimate_spread_cost(trade)
        impact_cost = self._estimate_impact_cost(trade)

        total_costs = commission_cost + slippage_cost + spread_cost + impact_cost

        # 2. Calculate alpha sources
        direction_alpha = self._calculate_direction_alpha(trade)
        timing_alpha = self._calculate_timing_alpha(trade)
        sizing_alpha = self._calculate_sizing_alpha(trade)

        # 3. Risk-adjusted metrics
        risk_adjusted = self._calculate_risk_adjusted(trade)
        sharpe_contribution = self._calculate_sharpe_contribution(trade)

        # Net P&L
        net_pnl = trade.realized_pnl - total_costs

        return TradeAttribution(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            strategy=trade.strategy,
            gross_pnl=trade.realized_pnl,
            commission_cost=commission_cost,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
            market_impact_cost=impact_cost,
            direction_alpha=direction_alpha,
            timing_alpha=timing_alpha,
            sizing_alpha=sizing_alpha,
            risk_adjusted_pnl=risk_adjusted,
            sharpe_contribution=sharpe_contribution,
            net_pnl=net_pnl,
            holding_period_hours=trade.holding_period_hours,
            exit_reason=trade.exit_reason.value
        )

    def _calculate_slippage_cost(self, trade: Trade) -> float:
        """
        Calculate slippage cost.

        Slippage = difference between expected and actual fill price.
        """
        if trade.slippage_bps == 0:
            return 0.0

        notional = trade.entry_price * trade.entry_quantity
        return abs(trade.slippage_bps / 10000 * notional)

    def _estimate_spread_cost(self, trade: Trade) -> float:
        """
        Estimate spread cost (bid-ask crossing).

        We cross spread on both entry and exit.
        """
        spread_bps = trade.entry_spread_bps if trade.entry_spread_bps > 0 else self.default_spread_bps
        exit_spread_bps = trade.exit_spread_bps if trade.exit_spread_bps > 0 else self.default_spread_bps

        # Half spread on entry, half on exit
        entry_cost = (spread_bps / 2 / 10000) * trade.entry_price * trade.entry_quantity
        exit_cost = (exit_spread_bps / 2 / 10000) * trade.exit_price * trade.exit_quantity

        return entry_cost + exit_cost

    def _estimate_impact_cost(self, trade: Trade) -> float:
        """
        Estimate market impact cost.

        Uses simplified square-root model:
        Impact = sigma * sqrt(Q/V) * price

        For now, use default assumption.
        """
        notional = trade.entry_price * trade.entry_quantity
        return (self.default_impact_bps / 10000) * notional

    def _calculate_direction_alpha(self, trade: Trade) -> float:
        """
        Calculate direction alpha.

        Direction alpha = P&L from being on the right side of the market.
        This is the P&L you would have gotten from a baseline position.
        """
        # Return as if we held baseline position
        if trade.side == TradeSide.LONG:
            move = trade.exit_price - trade.entry_price
        else:
            move = trade.entry_price - trade.exit_price

        move_pct = move / trade.entry_price

        # Direction alpha is the P&L from simply being long/short
        # regardless of timing or sizing
        baseline_notional = trade.entry_price * trade.entry_quantity

        return move_pct * baseline_notional

    def _calculate_timing_alpha(self, trade: Trade) -> float:
        """
        Calculate timing alpha.

        Timing alpha = difference between our entry/exit prices
        and the market VWAP during those periods.

        Positive = we entered below VWAP (for longs) or exited above VWAP.
        """
        if trade.entry_vwap == 0 or trade.exit_vwap == 0:
            return 0.0

        # Entry timing: did we buy below VWAP (good) or above (bad)?
        if trade.side == TradeSide.LONG:
            entry_timing = trade.entry_vwap - trade.entry_price
            exit_timing = trade.exit_price - trade.exit_vwap
        else:
            entry_timing = trade.entry_price - trade.entry_vwap
            exit_timing = trade.exit_vwap - trade.exit_price

        # Convert to dollar terms
        entry_alpha = entry_timing * trade.entry_quantity
        exit_alpha = exit_timing * trade.exit_quantity

        return entry_alpha + exit_alpha

    def _calculate_sizing_alpha(self, trade: Trade) -> float:
        """
        Calculate sizing alpha.

        Sizing alpha = extra P&L from over/undersizing vs baseline.

        If signal was correct and we oversized: positive sizing alpha
        If signal was wrong and we undersized: positive sizing alpha
        """
        # Baseline position for this strategy
        baseline_qty = int(
            (self.baseline_position_pct * trade.entry_price * 1000000) /
            trade.entry_price
        )

        # Actual vs baseline
        size_multiplier = trade.entry_quantity / baseline_qty if baseline_qty > 0 else 1.0

        # P&L per share
        pnl_per_share = trade.realized_pnl / trade.entry_quantity if trade.entry_quantity > 0 else 0

        # Sizing alpha = (actual_qty - baseline_qty) * pnl_per_share
        # = (actual_qty * pnl_per_share) - (baseline_qty * pnl_per_share)
        # = total_pnl - baseline_pnl
        baseline_pnl = pnl_per_share * baseline_qty
        sizing_alpha = trade.realized_pnl - baseline_pnl

        return sizing_alpha

    def _calculate_risk_adjusted(self, trade: Trade) -> float:
        """
        Calculate risk-adjusted P&L.

        P&L / risk at entry
        """
        if trade.risk_at_entry == 0:
            # Use stop loss as proxy for risk
            if trade.stop_loss_pct > 0:
                risk = trade.stop_loss_pct * trade.entry_price * trade.entry_quantity
            else:
                # Default 2% risk assumption
                risk = 0.02 * trade.entry_price * trade.entry_quantity

            if risk == 0:
                return 0.0

            return trade.realized_pnl / risk

        return trade.realized_pnl / trade.risk_at_entry

    def _calculate_sharpe_contribution(self, trade: Trade) -> float:
        """
        Calculate contribution to portfolio Sharpe.

        Simplified: return / sqrt(holding_period_days)
        """
        holding_days = trade.holding_period_hours / 24

        if holding_days <= 0:
            return 0.0

        daily_return = trade.return_pct / holding_days if holding_days > 0 else trade.return_pct
        annualized_return = daily_return * 252

        # Simplified Sharpe contribution
        return (annualized_return - self.risk_free_rate) / np.sqrt(252)

    def get_portfolio_attribution(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PortfolioAttribution:
        """
        Get aggregate attribution for portfolio.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            PortfolioAttribution
        """
        # Filter trades by date
        trades = self._trades
        attributions = self._attributions

        if start_date:
            mask = [t.entry_time >= start_date for t in trades]
            trades = [t for t, m in zip(trades, mask) if m]
            attributions = [a for a, m in zip(attributions, mask) if m]

        if end_date:
            mask = [t.exit_time <= end_date for t in trades]
            trades = [t for t, m in zip(trades, mask) if m]
            attributions = [a for a, m in zip(attributions, mask) if m]

        if not trades:
            return PortfolioAttribution(
                period_start=start_date or datetime.now(),
                period_end=end_date or datetime.now(),
                total_trades=0,
                gross_pnl=0,
                total_costs=0,
                net_pnl=0,
                total_commission=0,
                total_slippage=0,
                total_spread=0,
                total_impact=0,
                total_direction_alpha=0,
                total_timing_alpha=0,
                total_sizing_alpha=0
            )

        # Aggregate metrics
        gross_pnl = sum(a.gross_pnl for a in attributions)
        total_commission = sum(a.commission_cost for a in attributions)
        total_slippage = sum(a.slippage_cost for a in attributions)
        total_spread = sum(a.spread_cost for a in attributions)
        total_impact = sum(a.market_impact_cost for a in attributions)
        total_costs = total_commission + total_slippage + total_spread + total_impact

        total_direction_alpha = sum(a.direction_alpha for a in attributions)
        total_timing_alpha = sum(a.timing_alpha for a in attributions)
        total_sizing_alpha = sum(a.sizing_alpha for a in attributions)

        # By strategy
        by_strategy: Dict[str, Dict[str, float]] = defaultdict(lambda: {'pnl': 0, 'trades': 0})
        for t, a in zip(trades, attributions):
            by_strategy[t.strategy]['pnl'] += a.net_pnl
            by_strategy[t.strategy]['trades'] += 1

        # By symbol
        by_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: {'pnl': 0, 'trades': 0})
        for t, a in zip(trades, attributions):
            by_symbol[t.symbol]['pnl'] += a.net_pnl
            by_symbol[t.symbol]['trades'] += 1

        # By exit reason
        by_exit_reason: Dict[str, Dict[str, float]] = defaultdict(lambda: {'pnl': 0, 'trades': 0})
        for t, a in zip(trades, attributions):
            by_exit_reason[t.exit_reason.value]['pnl'] += a.net_pnl
            by_exit_reason[t.exit_reason.value]['trades'] += 1

        # Risk metrics
        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl <= 0]

        win_rate = len(wins) / len(trades) if trades else 0

        total_wins = sum(t.realized_pnl for t in wins)
        total_losses = abs(sum(t.realized_pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Risk adjusted return
        total_risk = sum(t.risk_at_entry for t in trades if t.risk_at_entry > 0)
        risk_adjusted = (gross_pnl - total_costs) / total_risk if total_risk > 0 else 0

        return PortfolioAttribution(
            period_start=start_date or min(t.entry_time for t in trades),
            period_end=end_date or max(t.exit_time for t in trades),
            total_trades=len(trades),
            gross_pnl=gross_pnl,
            total_costs=total_costs,
            net_pnl=gross_pnl - total_costs,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_spread=total_spread,
            total_impact=total_impact,
            total_direction_alpha=total_direction_alpha,
            total_timing_alpha=total_timing_alpha,
            total_sizing_alpha=total_sizing_alpha,
            by_strategy=dict(by_strategy),
            by_symbol=dict(by_symbol),
            by_exit_reason=dict(by_exit_reason),
            risk_adjusted_return=risk_adjusted,
            win_rate=win_rate,
            profit_factor=profit_factor
        )

    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AttributionReport:
        """
        Generate full attribution report with insights.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            AttributionReport with insights and recommendations
        """
        portfolio = self.get_portfolio_attribution(start_date, end_date)

        # Filter attributions
        attributions = self._attributions
        if start_date:
            trades = self._trades
            mask = [t.entry_time >= start_date for t in trades]
            attributions = [a for a, m in zip(attributions, mask) if m]
        if end_date:
            trades = self._trades
            mask = [t.exit_time <= end_date for t in trades]
            attributions = [a for a, m in zip(attributions, mask) if m]

        # Generate insights
        insights = self._generate_insights(portfolio, attributions)

        # Generate recommendations
        recommendations = self._generate_recommendations(portfolio, attributions)

        return AttributionReport(
            portfolio=portfolio,
            trades=attributions,
            insights=insights,
            recommendations=recommendations
        )

    def _generate_insights(
        self,
        portfolio: PortfolioAttribution,
        attributions: List[TradeAttribution]
    ) -> List[str]:
        """Generate insights from attribution data"""
        insights = []

        if portfolio.total_trades == 0:
            return ["No trades in period"]

        # Cost analysis
        cost_pct = portfolio.total_costs / abs(portfolio.gross_pnl) * 100 if portfolio.gross_pnl != 0 else 0
        if cost_pct > 30:
            insights.append(f"High cost drag: {cost_pct:.1f}% of gross P&L lost to costs")

        # Slippage analysis
        if portfolio.total_slippage > portfolio.total_commission:
            insights.append(f"Slippage (${portfolio.total_slippage:,.0f}) exceeds commissions - execution quality issue")

        # Alpha source analysis
        total_alpha = portfolio.total_direction_alpha + portfolio.total_timing_alpha + portfolio.total_sizing_alpha

        if total_alpha != 0:
            direction_pct = portfolio.total_direction_alpha / total_alpha * 100 if total_alpha > 0 else 0
            timing_pct = portfolio.total_timing_alpha / total_alpha * 100 if total_alpha > 0 else 0
            sizing_pct = portfolio.total_sizing_alpha / total_alpha * 100 if total_alpha > 0 else 0

            if abs(direction_pct) > 60:
                insights.append(f"Direction calling is primary alpha source ({direction_pct:.0f}%)")

            if timing_pct < -20:
                insights.append(f"Negative timing alpha ({timing_pct:.0f}%) - execution timing hurting returns")

            if sizing_pct > 30:
                insights.append(f"Position sizing adding significant alpha ({sizing_pct:.0f}%)")

        # Strategy analysis
        if portfolio.by_strategy:
            best_strategy = max(portfolio.by_strategy.items(), key=lambda x: x[1]['pnl'])
            worst_strategy = min(portfolio.by_strategy.items(), key=lambda x: x[1]['pnl'])

            if best_strategy[1]['pnl'] > 0:
                insights.append(f"Best performing strategy: {best_strategy[0]} (+${best_strategy[1]['pnl']:,.0f})")
            if worst_strategy[1]['pnl'] < 0:
                insights.append(f"Worst performing strategy: {worst_strategy[0]} (${worst_strategy[1]['pnl']:,.0f})")

        # Exit reason analysis
        if portfolio.by_exit_reason:
            stop_loss_pnl = portfolio.by_exit_reason.get('stop_loss', {}).get('pnl', 0)
            take_profit_pnl = portfolio.by_exit_reason.get('take_profit', {}).get('pnl', 0)

            if abs(stop_loss_pnl) > abs(take_profit_pnl) * 1.5:
                insights.append("Stop losses contributing more to P&L than take profits - review stop placement")

        return insights

    def _generate_recommendations(
        self,
        portfolio: PortfolioAttribution,
        attributions: List[TradeAttribution]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if portfolio.total_trades == 0:
            return ["Insufficient data for recommendations"]

        # Cost recommendations
        if portfolio.total_slippage > 1000:
            recommendations.append("Consider using limit orders or TWAP execution to reduce slippage")

        if portfolio.total_spread > portfolio.total_commission * 2:
            recommendations.append("Evaluate trading in higher-liquidity names to reduce spread costs")

        # Alpha recommendations
        if portfolio.total_timing_alpha < -500:
            recommendations.append("Review entry/exit timing - consider using VWAP benchmark for execution")

        if portfolio.total_sizing_alpha < -500:
            recommendations.append("Position sizing may be sub-optimal - review Kelly criterion calibration")

        # Risk recommendations
        if portfolio.win_rate < 0.45:
            recommendations.append("Win rate below 45% - ensure profit factor compensates for low hit rate")

        if portfolio.profit_factor < 1.2:
            recommendations.append("Profit factor below 1.2 - tighten stops or widen targets")

        # Strategy recommendations
        losing_strategies = [
            s for s, d in portfolio.by_strategy.items()
            if d['pnl'] < 0 and d['trades'] > 5
        ]
        if losing_strategies:
            recommendations.append(f"Review or disable losing strategies: {', '.join(losing_strategies)}")

        return recommendations

    def get_trade_attributions(self) -> List[TradeAttribution]:
        """Get all trade attributions"""
        return self._attributions.copy()

    def clear(self) -> None:
        """Clear all stored trades"""
        self._trades.clear()
        self._attributions.clear()


# =============================================================================
# REAL-TIME P&L TRACKER
# =============================================================================

class RealTimePnLTracker:
    """
    Tracks P&L in real-time during trading session.

    Provides:
    - Live P&L by position
    - Unrealized vs realized P&L
    - Daily P&L limit monitoring
    """

    def __init__(
        self,
        daily_loss_limit: float = 10000,
        position_loss_limit_pct: float = 0.05
    ):
        self.daily_loss_limit = daily_loss_limit
        self.position_loss_limit_pct = position_loss_limit_pct

        # Positions
        self._positions: Dict[str, Dict] = {}

        # Daily tracking
        self._daily_realized: float = 0.0
        self._daily_unrealized: float = 0.0
        self._session_start: datetime = datetime.now()

        # Alert callbacks
        self._alert_handlers: List = []

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float
    ) -> None:
        """Record position opening"""
        self._positions[symbol] = {
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'high_water_mark': entry_price,
            'low_water_mark': entry_price
        }

    def update_price(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Update position with current price.

        Returns alert dict if limits breached.
        """
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        pos['current_price'] = current_price

        # Calculate unrealized P&L
        if pos['side'] == 'long':
            pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['quantity']
        else:
            pos['unrealized_pnl'] = (pos['entry_price'] - current_price) * pos['quantity']

        # Update water marks
        if current_price > pos['high_water_mark']:
            pos['high_water_mark'] = current_price
        if current_price < pos['low_water_mark']:
            pos['low_water_mark'] = current_price

        # Check position-level limit
        pnl_pct = pos['unrealized_pnl'] / (pos['entry_price'] * pos['quantity'])
        if pnl_pct < -self.position_loss_limit_pct:
            return {
                'type': 'position_loss_limit',
                'symbol': symbol,
                'pnl_pct': pnl_pct,
                'limit': self.position_loss_limit_pct
            }

        # Update total unrealized
        self._daily_unrealized = sum(
            p['unrealized_pnl'] for p in self._positions.values()
        )

        # Check daily limit
        total_pnl = self._daily_realized + self._daily_unrealized
        if total_pnl < -self.daily_loss_limit:
            return {
                'type': 'daily_loss_limit',
                'total_pnl': total_pnl,
                'limit': self.daily_loss_limit
            }

        return None

    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close position and return realized P&L.

        Returns:
            Realized P&L
        """
        if symbol not in self._positions:
            return 0.0

        pos = self._positions.pop(symbol)

        if pos['side'] == 'long':
            realized = (exit_price - pos['entry_price']) * pos['quantity']
        else:
            realized = (pos['entry_price'] - exit_price) * pos['quantity']

        self._daily_realized += realized

        # Update unrealized total
        self._daily_unrealized = sum(
            p['unrealized_pnl'] for p in self._positions.values()
        )

        return realized

    def get_daily_pnl(self) -> Dict[str, float]:
        """Get current daily P&L status"""
        return {
            'realized': self._daily_realized,
            'unrealized': self._daily_unrealized,
            'total': self._daily_realized + self._daily_unrealized,
            'remaining_limit': self.daily_loss_limit + (self._daily_realized + self._daily_unrealized)
        }

    def get_position_pnl(self, symbol: str) -> Optional[Dict]:
        """Get P&L for specific position"""
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        return {
            'symbol': symbol,
            'unrealized_pnl': pos['unrealized_pnl'],
            'entry_price': pos['entry_price'],
            'current_price': pos['current_price'],
            'high_water_mark': pos['high_water_mark'],
            'low_water_mark': pos['low_water_mark'],
            'pnl_pct': pos['unrealized_pnl'] / (pos['entry_price'] * pos['quantity'])
        }

    def reset_daily(self) -> None:
        """Reset daily tracking (call at session start)"""
        self._daily_realized = 0.0
        self._session_start = datetime.now()
