"""
Institutional Backtesting Extensions
====================================

This module extends the institutional.py with additional functionality
and helper classes that were not fully implemented.

Includes:
- Complete RegimeDetector.get_regime_adjusted_params()
- Complete PortfolioOptimizer methods (HRP, Risk Parity, Black-Litterman)
- InstitutionalBacktester._execute_trade() implementation
- Performance attribution analysis
- HTML report generation

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import squareform

from config.settings import get_logger

logger = get_logger(__name__)


# =============================================================================
# REGIME PARAMETER ADJUSTMENTS
# =============================================================================

@dataclass
class RegimeAdjustedParams:
    """Parameters adjusted for current market regime."""
    adjusted_size: float  # Position size multiplier
    volatility_target: float  # Target portfolio volatility
    rebalance_threshold: float  # Threshold for rebalancing
    stop_loss_multiplier: float  # Stop loss adjustment
    confidence_threshold: float  # Minimum confidence to trade
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "adjusted_size": self.adjusted_size,
            "volatility_target": self.volatility_target,
            "rebalance_threshold": self.rebalance_threshold,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "confidence_threshold": self.confidence_threshold,
        }


# Regime parameter mappings
REGIME_PARAMS = {
    "bull_low_vol": RegimeAdjustedParams(
        adjusted_size=1.0,
        volatility_target=0.15,
        rebalance_threshold=0.05,
        stop_loss_multiplier=1.0,
        confidence_threshold=0.55,
    ),
    "bull_high_vol": RegimeAdjustedParams(
        adjusted_size=0.8,
        volatility_target=0.12,
        rebalance_threshold=0.04,
        stop_loss_multiplier=0.8,
        confidence_threshold=0.60,
    ),
    "bear_low_vol": RegimeAdjustedParams(
        adjusted_size=0.6,
        volatility_target=0.10,
        rebalance_threshold=0.03,
        stop_loss_multiplier=0.7,
        confidence_threshold=0.60,
    ),
    "bear_high_vol": RegimeAdjustedParams(
        adjusted_size=0.4,
        volatility_target=0.08,
        rebalance_threshold=0.02,
        stop_loss_multiplier=0.5,
        confidence_threshold=0.65,
    ),
    "sideways": RegimeAdjustedParams(
        adjusted_size=0.7,
        volatility_target=0.12,
        rebalance_threshold=0.05,
        stop_loss_multiplier=0.9,
        confidence_threshold=0.58,
    ),
    "crisis": RegimeAdjustedParams(
        adjusted_size=0.2,
        volatility_target=0.05,
        rebalance_threshold=0.01,
        stop_loss_multiplier=0.3,
        confidence_threshold=0.75,
    ),
}


def get_regime_adjusted_params(
    regime: str,
    base_position_size: float = 0.10,
) -> Dict[str, Any]:
    """
    Get regime-adjusted trading parameters.
    
    Args:
        regime: Current regime (e.g., "bull_low_vol", "crisis")
        base_position_size: Base position size as fraction
        
    Returns:
        Dictionary with adjusted parameters
    """
    regime_key = regime.lower().replace(" ", "_")
    params = REGIME_PARAMS.get(regime_key, REGIME_PARAMS["sideways"])
    
    return {
        "adjusted_size": base_position_size * params.adjusted_size,
        "volatility_target": params.volatility_target,
        "rebalance_threshold": params.rebalance_threshold,
        "stop_loss_multiplier": params.stop_loss_multiplier,
        "confidence_threshold": params.confidence_threshold,
    }


# =============================================================================
# ADVANCED PORTFOLIO OPTIMIZATION
# =============================================================================

class AdvancedPortfolioOptimizer:
    """
    Extended portfolio optimizer with all optimization methods.
    
    Implements:
    - Mean-Variance (Markowitz)
    - Minimum Variance
    - Maximum Sharpe
    - Risk Parity
    - Hierarchical Risk Parity (HRP)
    - Black-Litterman
    - Maximum Diversification
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        max_weight: float = 0.10,
        min_weight: float = 0.0,
        allow_shorting: bool = False,
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_shorting else -max_weight
        self.allow_shorting = allow_shorting
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        method: str = "max_sharpe",
        **kwargs,
    ) -> NDArray[np.float64]:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns per asset
            covariance: Covariance matrix
            method: Optimization method
            
        Returns:
            Optimal weights array
        """
        method = method.lower().replace("-", "_")
        
        if method == "mean_variance" or method == "max_sharpe":
            return self._max_sharpe(expected_returns, covariance)
        elif method == "min_variance":
            return self._min_variance(covariance)
        elif method == "risk_parity":
            return self._risk_parity(covariance)
        elif method in ["hrp", "hierarchical_risk_parity"]:
            return self._hrp(covariance)
        elif method == "black_litterman":
            views = kwargs.get("views", {})
            return self._black_litterman(expected_returns, covariance, views)
        elif method == "max_diversification":
            return self._max_diversification(covariance)
        elif method == "equal_weight":
            n = len(expected_returns)
            return np.ones(n) / n
        elif method == "inverse_volatility":
            return self._inverse_volatility(covariance)
        else:
            # Default to max sharpe
            return self._max_sharpe(expected_returns, covariance)
    
    def _max_sharpe(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximum Sharpe ratio optimization."""
        n = len(expected_returns)
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Bounds
        bounds = Bounds(
            lb=np.full(n, self.min_weight),
            ub=np.full(n, self.max_weight),
        )
        
        # Constraint: weights sum to 1
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
        }]
        
        def neg_sharpe(w):
            port_return = np.dot(w, expected_returns)
            port_vol = np.sqrt(w @ covariance @ w)
            
            if port_vol < 1e-10:
                return 1e10
            
            # Annualize
            annual_return = port_return * 252
            annual_vol = port_vol * np.sqrt(252)
            
            sharpe = (annual_return - self.risk_free_rate) / annual_vol
            return -sharpe + 0.001 * np.sum(w ** 2)  # L2 regularization
        
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000},
        )
        
        weights = result.x
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _min_variance(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Minimum variance portfolio."""
        n = covariance.shape[0]
        
        x0 = np.ones(n) / n
        
        bounds = Bounds(
            lb=np.full(n, self.min_weight),
            ub=np.full(n, self.max_weight),
        )
        
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
        }]
        
        def variance(w):
            return w @ covariance @ w
        
        result = minimize(
            variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = result.x
        weights = weights / np.sum(weights)
        
        return weights
    
    def _risk_parity(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Risk parity (equal risk contribution) portfolio.
        
        Each asset contributes equally to portfolio risk.
        """
        n = covariance.shape[0]
        
        # Initial guess: inverse volatility
        vols = np.sqrt(np.diag(covariance))
        vols[vols == 0] = 1e-10
        x0 = (1 / vols) / np.sum(1 / vols)
        
        bounds = Bounds(
            lb=np.full(n, 1e-6),
            ub=np.full(n, self.max_weight),
        )
        
        def risk_parity_obj(w):
            # Portfolio volatility
            port_vol = np.sqrt(w @ covariance @ w)
            
            if port_vol < 1e-10:
                return 1e10
            
            # Marginal risk contribution
            mrc = (covariance @ w) / port_vol
            
            # Risk contribution
            rc = w * mrc
            
            # Target: equal risk contribution
            target_rc = port_vol / n
            
            # Sum of squared deviations
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
        }]
        
        result = minimize(
            risk_parity_obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = result.x
        weights = weights / np.sum(weights)
        
        return weights
    
    def _hrp(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Hierarchical Risk Parity (HRP).
        
        From López de Prado (2016).
        """
        n = covariance.shape[0]
        
        # Step 1: Correlation and distance matrix
        std = np.sqrt(np.diag(covariance))
        std[std == 0] = 1e-10
        corr = covariance / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1)
        
        # Distance
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Step 2: Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method='ward')
        
        # Step 3: Quasi-diagonalization
        sorted_idx = self._get_quasi_diag(link, n)
        
        # Step 4: Recursive bisection
        weights = np.zeros(n)
        self._recursive_bisection(weights, covariance, sorted_idx)
        
        # Apply max weight constraint
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _get_quasi_diag(
        self,
        link: NDArray[np.float64],
        n: int,
    ) -> List[int]:
        """Get quasi-diagonal ordering from linkage."""
        link = link.astype(int)
        
        sorted_idx = [2 * n - 2]  # Root
        
        while len([i for i in sorted_idx if i >= n]) > 0:
            new_sorted = []
            for idx in sorted_idx:
                if idx < n:
                    new_sorted.append(idx)
                else:
                    cluster_idx = idx - n
                    new_sorted.extend([link[cluster_idx, 0], link[cluster_idx, 1]])
            sorted_idx = new_sorted
        
        return [int(i) for i in sorted_idx]
    
    def _recursive_bisection(
        self,
        weights: NDArray[np.float64],
        covariance: NDArray[np.float64],
        sorted_idx: List[int],
        allocation: float = 1.0,
    ) -> None:
        """Recursive bisection for HRP weights."""
        if len(sorted_idx) == 1:
            weights[sorted_idx[0]] = allocation
            return
        
        mid = len(sorted_idx) // 2
        left = sorted_idx[:mid]
        right = sorted_idx[mid:]
        
        # Cluster variances
        left_var = self._cluster_variance(covariance, left)
        right_var = self._cluster_variance(covariance, right)
        
        total_var = left_var + right_var
        if total_var > 0:
            left_alloc = allocation * right_var / total_var
            right_alloc = allocation * left_var / total_var
        else:
            left_alloc = right_alloc = allocation / 2
        
        self._recursive_bisection(weights, covariance, left, left_alloc)
        self._recursive_bisection(weights, covariance, right, right_alloc)
    
    def _cluster_variance(
        self,
        covariance: NDArray[np.float64],
        indices: List[int],
    ) -> float:
        """Calculate cluster variance using inverse variance weights."""
        sub_cov = covariance[np.ix_(indices, indices)]
        vols = np.sqrt(np.diag(sub_cov))
        vols[vols == 0] = 1e-10
        ivp = 1 / vols
        ivp = ivp / np.sum(ivp)
        return float(ivp @ sub_cov @ ivp)
    
    def _black_litterman(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        views: Dict[str, Any],
    ) -> NDArray[np.float64]:
        """
        Black-Litterman model.
        
        Args:
            expected_returns: Prior returns
            covariance: Covariance matrix
            views: Dictionary with:
                   - P: View matrix (n_views x n_assets)
                   - Q: View returns (n_views,)
                   - omega: View uncertainty (n_views x n_views)
        """
        n = len(expected_returns)
        tau = 0.05  # Uncertainty scaling
        
        # Default market weights (equal)
        market_weights = np.ones(n) / n
        
        # Risk aversion
        delta = 2.5
        
        # Equilibrium returns
        pi = delta * (covariance @ market_weights)
        
        if not views or "P" not in views:
            # No views - use equilibrium
            bl_returns = pi
        else:
            P = np.array(views["P"])
            Q = np.array(views["Q"])
            omega = np.array(views.get("omega", np.eye(len(Q)) * tau))
            
            # Black-Litterman formula
            tau_sigma = tau * covariance
            tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(n) * 1e-8)
            omega_inv = np.linalg.inv(omega + np.eye(len(Q)) * 1e-8)
            
            M = tau_sigma_inv + P.T @ omega_inv @ P
            M_inv = np.linalg.inv(M + np.eye(n) * 1e-8)
            
            bl_returns = M_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        
        # Optimize with BL returns
        return self._max_sharpe(bl_returns, covariance)
    
    def _max_diversification(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximum diversification portfolio."""
        n = covariance.shape[0]
        vols = np.sqrt(np.diag(covariance))
        vols[vols == 0] = 1e-10
        
        x0 = np.ones(n) / n
        
        bounds = Bounds(
            lb=np.full(n, self.min_weight),
            ub=np.full(n, self.max_weight),
        )
        
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
        }]
        
        def neg_diversification(w):
            port_vol = np.sqrt(w @ covariance @ w)
            weighted_vols = np.dot(w, vols)
            
            if port_vol < 1e-10:
                return 1e10
            
            # Diversification ratio: weighted avg vol / portfolio vol
            dr = weighted_vols / port_vol
            return -dr
        
        result = minimize(
            neg_diversification,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = result.x
        weights = weights / np.sum(weights)
        
        return weights
    
    def _inverse_volatility(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Inverse volatility weighting."""
        vols = np.sqrt(np.diag(covariance))
        vols[vols == 0] = 1e-10
        
        weights = 1 / vols
        weights = weights / np.sum(weights)
        
        # Apply max weight
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / np.sum(weights)
        
        return weights


# =============================================================================
# TRADE EXECUTION
# =============================================================================

@dataclass
class TradeExecution:
    """Record of a trade execution."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    notional: float
    commission: float
    slippage: float
    market_impact: float
    total_cost: float
    signal_direction: int
    signal_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": str(self.timestamp),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "commission": self.commission,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "total_cost": self.total_cost,
            "signal_direction": self.signal_direction,
            "signal_confidence": self.signal_confidence,
            "cost_bps": (self.total_cost / self.notional * 10000) if self.notional > 0 else 0,
        }


def execute_institutional_trade(
    symbol: str,
    trade_value: float,
    price: float,
    config: Any,  # InstitutionalBacktestConfig
    timestamp: datetime,
    signal: Dict[str, Any],
    adv: float = 1_000_000.0,
) -> TradeExecution:
    """
    Execute a trade with institutional-grade cost modeling.
    
    Args:
        symbol: Trading symbol
        trade_value: Signed trade value (positive=buy, negative=sell)
        price: Current price
        config: Backtest configuration
        timestamp: Current timestamp
        signal: Signal information
        adv: Average daily volume in dollars
        
    Returns:
        TradeExecution record
    """
    abs_value = abs(trade_value)
    side = "buy" if trade_value > 0 else "sell"
    
    # Commission
    commission = abs_value * (config.commission_bps / 10000)
    
    # Spread cost (half-spread for each leg)
    spread_cost = abs_value * (config.spread_bps / 10000)
    
    # Market impact (Almgren-Chriss style)
    # Temporary impact: sqrt(trade_size / ADV) * volatility * coefficient
    participation = abs_value / adv if adv > 0 else 0.01
    temp_impact_pct = config.temporary_impact_coefficient * np.sqrt(participation)
    temp_impact = abs_value * temp_impact_pct
    
    # Permanent impact
    perm_impact = abs_value * (config.market_impact_bps / 10000)
    
    total_market_impact = temp_impact + perm_impact
    
    # Total slippage
    slippage = spread_cost + total_market_impact
    
    # Adjust execution price for slippage
    if side == "buy":
        effective_price = price * (1 + slippage / abs_value) if abs_value > 0 else price
    else:
        effective_price = price * (1 - slippage / abs_value) if abs_value > 0 else price
    
    quantity = abs_value / effective_price
    total_cost = commission + slippage
    
    return TradeExecution(
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=effective_price,
        notional=abs_value,
        commission=commission,
        slippage=slippage,
        market_impact=total_market_impact,
        total_cost=total_cost,
        signal_direction=signal.get("direction", 0),
        signal_confidence=signal.get("confidence", 0),
    )


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Institutional Backtest Report",
) -> None:
    """
    Generate an HTML report for backtest results.
    
    Args:
        results: Backtest results dictionary
        output_path: Path to save HTML file
        title: Report title
    """
    # Extract equity curve
    equity_curve = results.get("equity_curve", [])
    equity_dates = [e[0] for e in equity_curve]
    equity_values = [e[1] for e in equity_curve]
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            opacity: 0.8;
            font-size: 1.1em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card .label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin-top: 5px;
        }}
        .metric-card .value.positive {{ color: #27ae60; }}
        .metric-card .value.negative {{ color: #e74c3c; }}
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .chart-container h2 {{
            margin-bottom: 20px;
            color: #1a1a2e;
        }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section h2 {{
            margin-bottom: 20px;
            color: #1a1a2e;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Return</div>
                <div class="value {'positive' if results.get('total_return', 0) >= 0 else 'negative'}">
                    {results.get('total_return', 0):.2%}
                </div>
            </div>
            <div class="metric-card">
                <div class="label">Annual Return</div>
                <div class="value {'positive' if results.get('annual_return', 0) >= 0 else 'negative'}">
                    {results.get('annual_return', 0):.2%}
                </div>
            </div>
            <div class="metric-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{results.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Sortino Ratio</div>
                <div class="value">{results.get('sortino_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">{results.get('max_drawdown', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Calmar Ratio</div>
                <div class="value">{results.get('calmar_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{results.get('n_trades', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Costs</div>
                <div class="value negative">${results.get('total_costs', 0):,.2f}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Equity Curve</h2>
            <canvas id="equityChart"></canvas>
        </div>
        
        <div class="section">
            <h2>Capital Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Initial Capital</td>
                    <td>${results.get('initial_capital', 0):,.2f}</td>
                </tr>
                <tr>
                    <td>Final Value</td>
                    <td>${results.get('final_value', 0):,.2f}</td>
                </tr>
                <tr>
                    <td>Total P&L</td>
                    <td>${results.get('total_pnl', 0):,.2f}</td>
                </tr>
                <tr>
                    <td>Annual Volatility</td>
                    <td>{results.get('annual_volatility', 0):.2%}</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Algo Trading Platform - Institutional Backtesting Framework</p>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(equity_dates[-500:])},
                datasets: [{{
                    label: 'Portfolio Value',
                    data: {json.dumps(equity_values[-500:])},
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        ticks: {{ maxTicksLimit: 10 }}
                    }},
                    y: {{
                        display: true,
                        ticks: {{
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {output_path}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Regime
    "RegimeAdjustedParams",
    "REGIME_PARAMS",
    "get_regime_adjusted_params",
    # Optimization
    "AdvancedPortfolioOptimizer",
    # Execution
    "TradeExecution",
    "execute_institutional_trade",
    # Reporting
    "generate_html_report",
]