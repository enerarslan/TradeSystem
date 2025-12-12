"""
Institutional Portfolio Management System
JPMorgan-Level Portfolio Tracking and Optimization

Features:
- Real-time position tracking
- Portfolio optimization (MVO, Risk Parity, Black-Litterman)
- Transaction cost analysis
- Performance attribution
- Rebalancing engine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from collections import defaultdict
import threading
import uuid
from scipy.optimize import minimize
import scipy.stats as stats

from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class PositionSide(Enum):
    """Position direction"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    side: PositionSide
    sector: str = "Other"
    beta: float = 1.0

    # Calculated fields
    market_value: float = 0
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0
    realized_pnl: float = 0
    cost_basis: float = 0
    weight: float = 0

    # Metadata
    entry_date: Optional[datetime] = None
    last_update: Optional[datetime] = None
    trade_count: int = 0

    def __post_init__(self):
        self.update_values()

    def update_values(self):
        """Update calculated values"""
        self.market_value = self.quantity * self.current_price
        self.cost_basis = abs(self.quantity) * self.avg_cost

        if self.cost_basis > 0:
            if self.side == PositionSide.LONG:
                self.unrealized_pnl = self.market_value - self.cost_basis
            else:
                self.unrealized_pnl = self.cost_basis - abs(self.market_value)

            self.unrealized_pnl_pct = self.unrealized_pnl / self.cost_basis
        else:
            self.unrealized_pnl = 0
            self.unrealized_pnl_pct = 0

        self.last_update = datetime.now()

    def update_price(self, price: float):
        """Update current price and recalculate"""
        self.current_price = price
        self.update_values()

    def add_shares(
        self,
        quantity: int,
        price: float
    ) -> float:
        """
        Add shares to position.

        Returns realized P&L if closing/reducing position
        """
        realized = 0

        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            # Reducing or closing position
            close_qty = min(abs(quantity), abs(self.quantity))

            if self.quantity > 0:
                realized = close_qty * (price - self.avg_cost)
            else:
                realized = close_qty * (self.avg_cost - price)

            self.realized_pnl += realized

            # Reduce position
            remaining = abs(quantity) - close_qty
            self.quantity += quantity

            if remaining > 0 and self.quantity != 0:
                # Opened opposite position
                self.avg_cost = price
                self.side = PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT
        else:
            # Adding to position
            total_cost = (abs(self.quantity) * self.avg_cost) + (abs(quantity) * price)
            self.quantity += quantity
            if self.quantity != 0:
                self.avg_cost = total_cost / abs(self.quantity)

        if self.quantity == 0:
            self.side = PositionSide.FLAT
        elif self.quantity > 0:
            self.side = PositionSide.LONG
        else:
            self.side = PositionSide.SHORT

        self.trade_count += 1
        self.update_values()

        return realized

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'side': self.side.value,
            'sector': self.sector,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl': self.realized_pnl,
            'weight': self.weight
        }


@dataclass
class Portfolio:
    """
    Portfolio state container.

    Tracks all positions, cash, and performance metrics.
    """
    portfolio_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Default"
    initial_capital: float = 1000000

    # Core state
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 0
    pending_cash: float = 0  # Cash from pending settlements

    # Performance tracking
    total_value: float = 0
    daily_pnl: float = 0
    total_pnl: float = 0
    total_realized_pnl: float = 0
    total_unrealized_pnl: float = 0

    # Historical tracking
    high_water_mark: float = 0
    max_drawdown: float = 0
    current_drawdown: float = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.cash = self.initial_capital
        self.total_value = self.initial_capital
        self.high_water_mark = self.initial_capital

    def update_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        sector: str = "Other"
    ) -> float:
        """
        Update or create position.

        Returns realized P&L
        """
        realized = 0

        if symbol in self.positions:
            realized = self.positions[symbol].add_shares(quantity, price)

            # Remove if flat
            if self.positions[symbol].quantity == 0:
                del self.positions[symbol]
        else:
            # New position
            side = PositionSide.LONG if quantity > 0 else PositionSide.SHORT
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                side=side,
                sector=sector,
                entry_date=datetime.now()
            )

        # Update cash
        self.cash -= quantity * price
        self.total_realized_pnl += realized

        self.recalculate()
        return realized

    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

        self.recalculate()

    def recalculate(self):
        """Recalculate portfolio values"""
        # Calculate position values
        position_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + position_value

        # Calculate unrealized P&L
        self.total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )

        # Total P&L
        self.total_pnl = self.total_value - self.initial_capital

        # Update drawdown
        if self.total_value > self.high_water_mark:
            self.high_water_mark = self.total_value

        if self.high_water_mark > 0:
            self.current_drawdown = (self.high_water_mark - self.total_value) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Update position weights
        for pos in self.positions.values():
            pos.weight = pos.market_value / self.total_value if self.total_value > 0 else 0

        self.last_update = datetime.now()

    def get_exposure(self) -> Dict[str, float]:
        """Get exposure metrics"""
        long_exposure = sum(
            pos.market_value for pos in self.positions.values()
            if pos.side == PositionSide.LONG
        )
        short_exposure = sum(
            abs(pos.market_value) for pos in self.positions.values()
            if pos.side == PositionSide.SHORT
        )

        return {
            'gross': long_exposure + short_exposure,
            'net': long_exposure - short_exposure,
            'long': long_exposure,
            'short': short_exposure,
            'leverage': (long_exposure + short_exposure) / self.total_value if self.total_value > 0 else 0
        }

    def get_sector_exposure(self) -> Dict[str, float]:
        """Get sector exposure breakdown"""
        sectors = defaultdict(float)
        for pos in self.positions.values():
            sectors[pos.sector] += pos.market_value

        return dict(sectors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl / self.initial_capital if self.initial_capital > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'exposure': self.get_exposure()
        }


class PortfolioManager:
    """
    Portfolio management and tracking system.

    Features:
    - Multi-portfolio management
    - Real-time position tracking
    - P&L calculation
    - Performance attribution
    - Historical tracking
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        portfolio_name: str = "Main"
    ):
        self.portfolio = Portfolio(
            name=portfolio_name,
            initial_capital=initial_capital
        )

        # Historical tracking
        self._daily_snapshots: List[Dict] = []
        self._trade_history: List[Dict] = []
        self._pnl_history: pd.DataFrame = pd.DataFrame()

        # Sector mapping
        self._sector_map: Dict[str, str] = {}

        # Thread safety
        self._lock = threading.RLock()

    def set_sector_map(self, sector_map: Dict[str, str]):
        """Set symbol to sector mapping"""
        with self._lock:
            self._sector_map = sector_map

    def execute_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        commission: float = 0,
        trade_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a trade and update portfolio.

        Returns:
            Trade execution details
        """
        with self._lock:
            sector = self._sector_map.get(symbol, 'Other')

            # Check if we have enough cash for buys
            if quantity > 0:
                required = quantity * price + commission
                if required > self.portfolio.cash:
                    raise ValueError(f"Insufficient cash: need ${required:,.2f}, have ${self.portfolio.cash:,.2f}")

            # Update position
            realized_pnl = self.portfolio.update_position(
                symbol, quantity, price, sector
            )

            # Deduct commission
            self.portfolio.cash -= commission

            # Record trade
            trade = {
                'trade_id': trade_id or str(uuid.uuid4())[:8],
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'notional': abs(quantity * price),
                'side': 'buy' if quantity > 0 else 'sell',
                'commission': commission,
                'realized_pnl': realized_pnl,
                'portfolio_value': self.portfolio.total_value
            }
            self._trade_history.append(trade)

            # Audit log
            audit_logger.log_trade({
                'trade_id': trade['trade_id'],
                'timestamp': trade['timestamp'].isoformat(),
                'symbol': symbol,
                'side': trade['side'].upper(),
                'quantity': abs(quantity),
                'price': price,
                'order_type': 'MARKET',
                'status': 'FILLED',
                'fill_price': price,
                'fill_quantity': abs(quantity),
                'commission': commission
            })

            logger.info(
                f"Trade executed: {trade['side'].upper()} {abs(quantity)} {symbol} "
                f"@ ${price:.2f}, P&L: ${realized_pnl:,.2f}"
            )

            return trade

    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        with self._lock:
            self.portfolio.update_prices(prices)

    def take_snapshot(self) -> Dict[str, Any]:
        """Take daily portfolio snapshot"""
        with self._lock:
            snapshot = {
                'date': date.today(),
                'timestamp': datetime.now(),
                'total_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.total_value - self.portfolio.cash,
                'total_pnl': self.portfolio.total_pnl,
                'realized_pnl': self.portfolio.total_realized_pnl,
                'unrealized_pnl': self.portfolio.total_unrealized_pnl,
                'drawdown': self.portfolio.current_drawdown,
                'num_positions': len(self.portfolio.positions),
                'exposure': self.portfolio.get_exposure()
            }

            self._daily_snapshots.append(snapshot)
            return snapshot

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        with self._lock:
            if len(self._daily_snapshots) < 2:
                return {}

            # Build returns series
            values = [s['total_value'] for s in self._daily_snapshots]
            returns = pd.Series(values).pct_change().dropna()

            if len(returns) == 0:
                return {}

            # Calculate metrics
            total_return = (self.portfolio.total_value / self.portfolio.initial_capital) - 1
            ann_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
            volatility = returns.std() * np.sqrt(252)
            sharpe = ann_return / volatility if volatility > 0 else 0
            sortino = ann_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0

            # Calmar ratio
            calmar = ann_return / self.portfolio.max_drawdown if self.portfolio.max_drawdown > 0 else 0

            # Win rate
            winning_days = (returns > 0).sum()
            total_days = len(returns)
            win_rate = winning_days / total_days if total_days > 0 else 0

            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            return {
                'total_return': total_return,
                'annualized_return': ann_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': self.portfolio.max_drawdown,
                'current_drawdown': self.portfolio.current_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(self._trade_history),
                'trading_days': len(returns)
            }

    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions"""
        with self._lock:
            if not self.portfolio.positions:
                return pd.DataFrame()

            data = [pos.to_dict() for pos in self.portfolio.positions.values()]
            return pd.DataFrame(data)

    def get_sector_summary(self) -> pd.DataFrame:
        """Get sector exposure summary"""
        with self._lock:
            sector_data = defaultdict(lambda: {
                'value': 0, 'pnl': 0, 'weight': 0, 'count': 0
            })

            for pos in self.portfolio.positions.values():
                sector_data[pos.sector]['value'] += pos.market_value
                sector_data[pos.sector]['pnl'] += pos.unrealized_pnl
                sector_data[pos.sector]['count'] += 1

            total_value = self.portfolio.total_value
            for sector in sector_data:
                sector_data[sector]['weight'] = sector_data[sector]['value'] / total_value if total_value > 0 else 0

            return pd.DataFrame(sector_data).T


class PortfolioOptimizer:
    """
    Portfolio optimization engine.

    Methods:
    - Mean-Variance Optimization (MVO)
    - Risk Parity
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Black-Litterman
    - Maximum Diversification
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.target_volatility = target_volatility

    def mean_variance_optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Mean-Variance Optimization (Markowitz).

        Args:
            returns: DataFrame of asset returns
            constraints: Optional constraints dict

        Returns:
            Optimal weights
        """
        n = len(returns.columns)
        symbols = list(returns.columns)

        # Expected returns and covariance
        mu = returns.mean() * 252
        cov = returns.cov() * 252

        # Objective: minimize variance for given return
        def portfolio_variance(weights):
            return weights @ cov.values @ weights

        def portfolio_return(weights):
            return weights @ mu.values

        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if self.target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda w: portfolio_return(w) - self.target_return
            })

        # Bounds
        min_weight = constraints.get('min_weight', 0) if constraints else 0
        max_weight = constraints.get('max_weight', 0.3) if constraints else 0.3
        bounds = [(min_weight, max_weight) for _ in range(n)]

        # Optimize
        x0 = np.ones(n) / n
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            weights = np.ones(n) / n

        return {symbols[i]: weights[i] for i in range(n)}

    def maximize_sharpe(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Maximize Sharpe Ratio.

        Returns:
            Optimal weights for max Sharpe
        """
        n = len(returns.columns)
        symbols = list(returns.columns)

        mu = returns.mean() * 252
        cov = returns.cov() * 252

        def neg_sharpe(weights):
            port_ret = weights @ mu.values
            port_vol = np.sqrt(weights @ cov.values @ weights)
            return -(port_ret - self.risk_free_rate) / port_vol

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        min_weight = constraints.get('min_weight', 0) if constraints else 0
        max_weight = constraints.get('max_weight', 0.3) if constraints else 0.3
        bounds = [(min_weight, max_weight) for _ in range(n)]

        x0 = np.ones(n) / n
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            weights = np.ones(n) / n

        return {symbols[i]: weights[i] for i in range(n)}

    def minimum_variance(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Minimum Variance Portfolio.

        Returns:
            Weights for minimum variance portfolio
        """
        n = len(returns.columns)
        symbols = list(returns.columns)

        cov = returns.cov() * 252

        def portfolio_variance(weights):
            return weights @ cov.values @ weights

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        min_weight = constraints.get('min_weight', 0) if constraints else 0
        max_weight = constraints.get('max_weight', 0.3) if constraints else 0.3
        bounds = [(min_weight, max_weight) for _ in range(n)]

        x0 = np.ones(n) / n
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            weights = np.ones(n) / n

        return {symbols[i]: weights[i] for i in range(n)}

    def risk_parity(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Risk Parity Portfolio.

        Equal risk contribution from each asset.
        """
        n = len(returns.columns)
        symbols = list(returns.columns)

        cov = returns.cov() * 252

        def risk_contribution(weights):
            port_vol = np.sqrt(weights @ cov.values @ weights)
            marginal = cov.values @ weights
            return weights * marginal / port_vol

        def objective(weights):
            rc = risk_contribution(weights)
            target = 1 / n
            return np.sum((rc - target) ** 2)

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        min_weight = constraints.get('min_weight', 0.01) if constraints else 0.01
        max_weight = constraints.get('max_weight', 0.5) if constraints else 0.5
        bounds = [(min_weight, max_weight) for _ in range(n)]

        x0 = np.ones(n) / n
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            # Fallback: inverse volatility
            vols = np.sqrt(np.diag(cov.values))
            inv_vols = 1 / vols
            weights = inv_vols / np.sum(inv_vols)

        return {symbols[i]: weights[i] for i in range(n)}

    def maximum_diversification(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Maximum Diversification Portfolio.

        Maximizes the diversification ratio:
        DR = (weighted avg of volatilities) / (portfolio volatility)
        """
        n = len(returns.columns)
        symbols = list(returns.columns)

        cov = returns.cov() * 252
        vols = np.sqrt(np.diag(cov.values))

        def neg_diversification_ratio(weights):
            weighted_vol = weights @ vols
            port_vol = np.sqrt(weights @ cov.values @ weights)
            return -weighted_vol / port_vol

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        min_weight = constraints.get('min_weight', 0) if constraints else 0
        max_weight = constraints.get('max_weight', 0.3) if constraints else 0.3
        bounds = [(min_weight, max_weight) for _ in range(n)]

        x0 = np.ones(n) / n
        result = minimize(
            neg_diversification_ratio,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            weights = np.ones(n) / n

        return {symbols[i]: weights[i] for i in range(n)}

    def black_litterman(
        self,
        returns: pd.DataFrame,
        market_caps: Dict[str, float],
        views: Dict[str, float],
        view_confidences: Dict[str, float],
        tau: float = 0.05
    ) -> Dict[str, float]:
        """
        Black-Litterman Model.

        Combines market equilibrium with investor views.

        Args:
            returns: Historical returns
            market_caps: Market capitalizations
            views: Expected returns views
            view_confidences: Confidence in each view (0-1)
            tau: Uncertainty scaling factor
        """
        n = len(returns.columns)
        symbols = list(returns.columns)

        cov = returns.cov() * 252

        # Market cap weights
        total_cap = sum(market_caps.get(s, 1e9) for s in symbols)
        market_weights = np.array([
            market_caps.get(s, 1e9) / total_cap for s in symbols
        ])

        # Implied equilibrium returns
        risk_aversion = 2.5
        pi = risk_aversion * cov.values @ market_weights

        # Views matrix
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        omega_diag = []

        for i, (symbol, view_return) in enumerate(views.items()):
            if symbol in symbols:
                idx = symbols.index(symbol)
                P[i, idx] = 1
                Q[i] = view_return
                conf = view_confidences.get(symbol, 0.5)
                omega_diag.append((1 - conf) * tau * cov.values[idx, idx])

        if len(omega_diag) == 0:
            # No valid views, return market weights
            return {symbols[i]: market_weights[i] for i in range(n)}

        Omega = np.diag(omega_diag)

        # Black-Litterman posterior
        tau_cov = tau * cov.values

        # M = inv(inv(tau*Sigma) + P'*inv(Omega)*P)
        M = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)

        # Posterior expected returns
        bl_returns = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)

        # Optimize using posterior returns
        def neg_utility(weights):
            port_ret = weights @ bl_returns
            port_var = weights @ cov.values @ weights
            return -(port_ret - 0.5 * risk_aversion * port_var)

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 0.3) for _ in range(n)]

        x0 = market_weights
        result = minimize(
            neg_utility,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        if result.success:
            weights = result.x
        else:
            weights = market_weights

        return {symbols[i]: weights[i] for i in range(n)}

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.

        Returns:
            DataFrame with frontier portfolios
        """
        mu = returns.mean() * 252
        cov = returns.cov() * 252

        min_ret = mu.min()
        max_ret = mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []

        for target in target_returns:
            self.target_return = target
            try:
                weights = self.mean_variance_optimize(returns)
                w = np.array(list(weights.values()))

                port_ret = w @ mu.values
                port_vol = np.sqrt(w @ cov.values @ w)
                sharpe = (port_ret - self.risk_free_rate) / port_vol

                frontier.append({
                    'return': port_ret,
                    'volatility': port_vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
            except:
                continue

        self.target_return = None
        return pd.DataFrame(frontier)


class RebalancingEngine:
    """
    Portfolio rebalancing engine.

    Features:
    - Threshold-based rebalancing
    - Calendar rebalancing
    - Tax-aware rebalancing
    - Transaction cost optimization
    """

    def __init__(
        self,
        threshold: float = 0.05,
        min_trade_value: float = 1000,
        commission_rate: float = 0.001
    ):
        self.threshold = threshold
        self.min_trade_value = min_trade_value
        self.commission_rate = commission_rate

    def calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to rebalance to target weights.

        Returns:
            List of trade orders
        """
        trades = []

        # Get all symbols
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current

            # Check if exceeds threshold
            if abs(diff) >= self.threshold:
                trade_value = diff * portfolio_value

                if abs(trade_value) >= self.min_trade_value:
                    price = prices.get(symbol, 0)
                    if price > 0:
                        shares = int(trade_value / price)
                        if shares != 0:
                            trades.append({
                                'symbol': symbol,
                                'shares': abs(shares),
                                'side': 'buy' if shares > 0 else 'sell',
                                'target_weight': target,
                                'current_weight': current,
                                'weight_change': diff,
                                'trade_value': abs(trade_value),
                                'estimated_commission': abs(trade_value) * self.commission_rate
                            })

        # Sort by trade value (largest first)
        trades.sort(key=lambda x: x['trade_value'], reverse=True)

        return trades

    def needs_rebalancing(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if rebalancing is needed.

        Returns:
            Tuple of (needs_rebalance, drift_by_symbol)
        """
        drifts = {}
        max_drift = 0

        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            drift = abs(target - current)
            drifts[symbol] = drift
            max_drift = max(max_drift, drift)

        return max_drift >= self.threshold, drifts

    def optimize_trade_sequence(
        self,
        trades: List[Dict[str, Any]],
        available_cash: float
    ) -> List[Dict[str, Any]]:
        """
        Optimize trade sequence to minimize cash usage.

        Executes sells before buys when cash is limited.
        """
        sells = [t for t in trades if t['side'] == 'sell']
        buys = [t for t in trades if t['side'] == 'buy']

        # Sort sells by value (highest first)
        sells.sort(key=lambda x: x['trade_value'], reverse=True)

        # Sort buys by value
        buys.sort(key=lambda x: x['trade_value'], reverse=True)

        # Calculate total buy value
        total_buy_value = sum(t['trade_value'] for t in buys)
        cash_from_sells = sum(t['trade_value'] for t in sells)

        # Check if we have enough cash
        cash_needed = total_buy_value - cash_from_sells

        if cash_needed > available_cash:
            # Scale down buys
            scale = (available_cash + cash_from_sells) / total_buy_value
            for buy in buys:
                buy['shares'] = int(buy['shares'] * scale)
                buy['trade_value'] *= scale
                buy['scaled'] = True

        # Return sells first, then buys
        return sells + buys
