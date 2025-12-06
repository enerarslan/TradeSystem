"""
Portfolio Optimizer Module
==========================

Portfolio optimization and allocation for the algorithmic trading platform.
Implements various optimization methods for portfolio construction.

Optimization Methods:
- Mean-Variance (Markowitz)
- Minimum Variance
- Maximum Sharpe Ratio
- Risk Parity
- Hierarchical Risk Parity (HRP)
- Black-Litterman (placeholder)
- Equal Weight
- Inverse Volatility

Features:
- Multiple optimization objectives
- Constraint handling (long-only, box constraints)
- Transaction cost consideration
- Rebalancing logic
- Performance attribution

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as scipy_optimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from config.settings import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOL = "inverse_volatility"
    HRP = "hrp"  # Hierarchical Risk Parity
    BLACK_LITTERMAN = "black_litterman"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CVAR = "min_cvar"


class RebalanceFrequency(str, Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """
    Portfolio optimization configuration.
    
    Attributes:
        method: Optimization method
        risk_free_rate: Annual risk-free rate
        target_return: Target annual return (for mean-variance)
        target_volatility: Target annual volatility
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
        long_only: Only allow long positions
        leverage: Maximum leverage (1.0 = no leverage)
        rebalance_frequency: How often to rebalance
        rebalance_threshold: Drift threshold for rebalancing
        transaction_cost: Transaction cost as fraction
        lookback_period: Days of history for estimation
        shrinkage: Covariance shrinkage parameter
        regularization: L2 regularization for weights
    """
    method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    risk_free_rate: float = 0.05
    target_return: float | None = None
    target_volatility: float | None = None
    max_weight: float = 0.30
    min_weight: float = 0.0
    long_only: bool = True
    leverage: float = 1.0
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_threshold: float = 0.05
    transaction_cost: float = 0.001
    lookback_period: int = 252
    shrinkage: float = 0.1
    regularization: float = 0.0


@dataclass
class OptimizationResult:
    """
    Result of portfolio optimization.
    
    Attributes:
        weights: Asset weights dictionary
        expected_return: Portfolio expected return
        expected_volatility: Portfolio expected volatility
        sharpe_ratio: Portfolio Sharpe ratio
        method: Method used
        timestamp: Optimization timestamp
        metadata: Additional metadata
    """
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: OptimizationMethod
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @property
    def weight_array(self) -> NDArray[np.float64]:
        """Get weights as numpy array."""
        return np.array(list(self.weights.values()))
    
    @property
    def symbols(self) -> list[str]:
        """Get list of symbols."""
        return list(self.weights.keys())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _portfolio_return(
    weights: NDArray[np.float64],
    expected_returns: NDArray[np.float64],
) -> float:
    """Calculate portfolio expected return."""
    return np.dot(weights, expected_returns)


def _portfolio_volatility(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
) -> float:
    """Calculate portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def _portfolio_sharpe(
    weights: NDArray[np.float64],
    expected_returns: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    risk_free_rate: float = 0.05,
) -> float:
    """Calculate portfolio Sharpe ratio."""
    ret = _portfolio_return(weights, expected_returns)
    vol = _portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol


def _shrink_covariance(
    cov_matrix: NDArray[np.float64],
    shrinkage: float = 0.1,
) -> NDArray[np.float64]:
    """
    Apply Ledoit-Wolf shrinkage to covariance matrix.
    
    Shrinks towards diagonal matrix for better conditioning.
    """
    n = cov_matrix.shape[0]
    
    # Target: diagonal matrix
    target = np.diag(np.diag(cov_matrix))
    
    # Shrink
    shrunk = (1 - shrinkage) * cov_matrix + shrinkage * target
    
    return shrunk


def _ensure_positive_definite(
    cov_matrix: NDArray[np.float64],
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    """Ensure covariance matrix is positive definite."""
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Reconstruct
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def mean_variance_optimize(
    expected_returns: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    target_return: float | None = None,
    target_volatility: float | None = None,
    risk_free_rate: float = 0.05,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    regularization: float = 0.0,
) -> NDArray[np.float64]:
    """
    Mean-variance optimization (Markowitz).
    
    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        target_return: Target portfolio return
        target_volatility: Target portfolio volatility
        risk_free_rate: Risk-free rate
        long_only: Only long positions
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
        regularization: L2 regularization
    
    Returns:
        Optimal weights array
    """
    n = len(expected_returns)
    
    # Initial guess: equal weight
    x0 = np.ones(n) / n
    
    # Bounds
    if long_only:
        bounds = [(min_weight, max_weight) for _ in range(n)]
    else:
        bounds = [(-max_weight, max_weight) for _ in range(n)]
    
    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]
    
    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: _portfolio_return(w, expected_returns) - target_return
        })
    
    if target_volatility is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: _portfolio_volatility(w, cov_matrix) - target_volatility
        })
    
    # Objective: minimize variance (or negative Sharpe if no target)
    if target_return is not None or target_volatility is not None:
        def objective(w):
            vol = _portfolio_volatility(w, cov_matrix)
            reg = regularization * np.sum(w ** 2)
            return vol + reg
    else:
        # Maximize Sharpe ratio
        def objective(w):
            sharpe = _portfolio_sharpe(w, expected_returns, cov_matrix, risk_free_rate)
            reg = regularization * np.sum(w ** 2)
            return -sharpe + reg
    
    # Optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scipy_optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
    
    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")
    
    # Normalize weights
    weights = result.x
    weights = np.maximum(weights, 0) if long_only else weights
    weights = weights / np.sum(np.abs(weights))
    
    return weights


def min_variance_portfolio(
    cov_matrix: NDArray[np.float64],
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """
    Minimum variance portfolio.
    
    Args:
        cov_matrix: Covariance matrix
        long_only: Only long positions
        max_weight: Maximum weight
        min_weight: Minimum weight
    
    Returns:
        Optimal weights
    """
    n = cov_matrix.shape[0]
    
    # Dummy expected returns (all equal)
    expected_returns = np.zeros(n)
    
    return mean_variance_optimize(
        expected_returns,
        cov_matrix,
        target_return=None,
        risk_free_rate=0,
        long_only=long_only,
        max_weight=max_weight,
        min_weight=min_weight,
    )


def max_sharpe_portfolio(
    expected_returns: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    risk_free_rate: float = 0.05,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """
    Maximum Sharpe ratio portfolio.
    
    Args:
        expected_returns: Expected returns
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        long_only: Only long positions
        max_weight: Maximum weight
        min_weight: Minimum weight
    
    Returns:
        Optimal weights
    """
    return mean_variance_optimize(
        expected_returns,
        cov_matrix,
        target_return=None,
        risk_free_rate=risk_free_rate,
        long_only=long_only,
        max_weight=max_weight,
        min_weight=min_weight,
    )


def risk_parity_portfolio(
    cov_matrix: NDArray[np.float64],
    risk_budget: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Risk parity portfolio (equal risk contribution).
    
    Each asset contributes equally to portfolio risk.
    
    Args:
        cov_matrix: Covariance matrix
        risk_budget: Risk budget per asset (None = equal)
    
    Returns:
        Optimal weights
    """
    n = cov_matrix.shape[0]
    
    if risk_budget is None:
        risk_budget = np.ones(n) / n
    
    # Normalize budget
    risk_budget = risk_budget / np.sum(risk_budget)
    
    def objective(w):
        """Minimize deviation from target risk contribution."""
        # Portfolio variance
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        
        # Marginal risk contribution
        mrc = np.dot(cov_matrix, w) / np.sqrt(port_var + 1e-10)
        
        # Risk contribution
        rc = w * mrc
        rc = rc / (np.sum(rc) + 1e-10)
        
        # Deviation from target
        return np.sum((rc - risk_budget) ** 2)
    
    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]
    
    # Bounds (long only)
    bounds = [(0.01, 1.0) for _ in range(n)]
    
    # Initial guess
    x0 = np.ones(n) / n
    
    # Optimize
    result = scipy_optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    return result.x / np.sum(result.x)


def equal_weight_portfolio(n_assets: int) -> NDArray[np.float64]:
    """
    Equal weight portfolio.
    
    Args:
        n_assets: Number of assets
    
    Returns:
        Equal weights array
    """
    return np.ones(n_assets) / n_assets


def inverse_volatility_portfolio(
    volatilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Inverse volatility weighted portfolio.
    
    Lower volatility assets get higher weights.
    
    Args:
        volatilities: Volatility for each asset
    
    Returns:
        Weights array
    """
    inv_vol = 1.0 / (volatilities + 1e-10)
    return inv_vol / np.sum(inv_vol)


def hrp_portfolio(
    returns: NDArray[np.float64],
    cov_matrix: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Hierarchical Risk Parity (HRP) portfolio.
    
    Uses hierarchical clustering for more stable weights.
    
    Args:
        returns: Historical returns matrix (T x N)
        cov_matrix: Covariance matrix (optional)
    
    Returns:
        Optimal weights
    """
    if cov_matrix is None:
        cov_matrix = np.cov(returns.T)
    
    n = cov_matrix.shape[0]
    
    # Correlation matrix
    std = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std, std)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Distance matrix
    dist = np.sqrt(0.5 * (1 - corr_matrix))
    dist = np.nan_to_num(dist, nan=1.0)
    
    # Hierarchical clustering
    try:
        dist_condensed = squareform(dist, checks=False)
        linkage = hierarchy.linkage(dist_condensed, method="ward")
        sort_ix = hierarchy.leaves_list(linkage)
    except Exception:
        sort_ix = np.arange(n)
    
    # Recursive bisection
    weights = np.ones(n)
    
    def _get_cluster_var(cov, ix):
        """Get cluster variance."""
        cov_slice = cov[np.ix_(ix, ix)]
        w = inverse_volatility_portfolio(np.sqrt(np.diag(cov_slice)))
        return np.dot(w.T, np.dot(cov_slice, w))
    
    def _recursive_bisect(cov, sort_ix, weights):
        """Recursively bisect portfolio."""
        if len(sort_ix) == 1:
            return
        
        # Split
        mid = len(sort_ix) // 2
        left_ix = sort_ix[:mid]
        right_ix = sort_ix[mid:]
        
        # Variance of each half
        left_var = _get_cluster_var(cov, left_ix)
        right_var = _get_cluster_var(cov, right_ix)
        
        # Allocate inverse variance
        alpha = 1 - left_var / (left_var + right_var + 1e-10)
        
        weights[left_ix] *= alpha
        weights[right_ix] *= (1 - alpha)
        
        # Recurse
        _recursive_bisect(cov, left_ix, weights)
        _recursive_bisect(cov, right_ix, weights)
    
    _recursive_bisect(cov_matrix, sort_ix, weights)
    
    return weights / np.sum(weights)


# =============================================================================
# PORTFOLIO OPTIMIZER CLASS
# =============================================================================

class PortfolioOptimizer:
    """
    Comprehensive portfolio optimization engine.
    
    Example:
        optimizer = PortfolioOptimizer(config)
        
        # Optimize portfolio
        result = optimizer.optimize(returns_df)
        
        # Get weights
        print(result.weights)
        
        # Check if rebalance needed
        if optimizer.should_rebalance(current_weights):
            new_weights = optimizer.optimize(returns_df)
    """
    
    def __init__(self, config: OptimizationConfig | None = None):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # State
        self._last_optimization: datetime | None = None
        self._last_weights: dict[str, float] | None = None
        self._optimization_history: list[OptimizationResult] = []
    
    def optimize(
        self,
        returns: NDArray[np.float64],
        symbols: list[str],
        expected_returns: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            returns: Historical returns matrix (T x N)
            symbols: Asset symbols
            expected_returns: Expected returns (None = use historical mean)
        
        Returns:
            OptimizationResult with optimal weights
        """
        n_assets = len(symbols)
        
        # Calculate statistics
        if expected_returns is None:
            expected_returns = np.mean(returns, axis=0) * 252  # Annualized
        
        cov_matrix = np.cov(returns.T) * 252  # Annualized
        
        # Shrink covariance
        cov_matrix = _shrink_covariance(cov_matrix, self.config.shrinkage)
        cov_matrix = _ensure_positive_definite(cov_matrix)
        
        # Calculate volatilities
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Optimize based on method
        method = self.config.method
        
        if method == OptimizationMethod.MEAN_VARIANCE:
            weights = mean_variance_optimize(
                expected_returns,
                cov_matrix,
                target_return=self.config.target_return,
                risk_free_rate=self.config.risk_free_rate,
                long_only=self.config.long_only,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight,
                regularization=self.config.regularization,
            )
        
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = min_variance_portfolio(
                cov_matrix,
                long_only=self.config.long_only,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight,
            )
        
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = max_sharpe_portfolio(
                expected_returns,
                cov_matrix,
                risk_free_rate=self.config.risk_free_rate,
                long_only=self.config.long_only,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight,
            )
        
        elif method == OptimizationMethod.RISK_PARITY:
            weights = risk_parity_portfolio(cov_matrix)
        
        elif method == OptimizationMethod.EQUAL_WEIGHT:
            weights = equal_weight_portfolio(n_assets)
        
        elif method == OptimizationMethod.INVERSE_VOL:
            weights = inverse_volatility_portfolio(volatilities)
        
        elif method == OptimizationMethod.HRP:
            weights = hrp_portfolio(returns, cov_matrix)
        
        else:
            logger.warning(f"Unknown method {method}, using equal weight")
            weights = equal_weight_portfolio(n_assets)
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        # Calculate portfolio metrics
        port_return = _portfolio_return(weights, expected_returns)
        port_vol = _portfolio_volatility(weights, cov_matrix)
        port_sharpe = (port_return - self.config.risk_free_rate) / (port_vol + 1e-10)
        
        # Create result
        result = OptimizationResult(
            weights={sym: w for sym, w in zip(symbols, weights)},
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_sharpe,
            method=method,
            metadata={
                "n_assets": n_assets,
                "lookback": returns.shape[0] if returns.ndim > 1 else 1,
            },
        )
        
        # Update state
        self._last_optimization = datetime.now()
        self._last_weights = result.weights.copy()
        self._optimization_history.append(result)
        
        logger.info(
            f"Optimized portfolio: {method.value}, "
            f"Return={port_return:.2%}, Vol={port_vol:.2%}, Sharpe={port_sharpe:.2f}"
        )
        
        return result
    
    def should_rebalance(
        self,
        current_weights: dict[str, float],
    ) -> bool:
        """
        Check if portfolio should be rebalanced.
        
        Args:
            current_weights: Current portfolio weights
        
        Returns:
            True if rebalance is recommended
        """
        if self._last_weights is None:
            return True
        
        if self._last_optimization is None:
            return True
        
        # Check frequency
        elapsed = datetime.now() - self._last_optimization
        freq = self.config.rebalance_frequency
        
        should_by_time = False
        if freq == RebalanceFrequency.DAILY:
            should_by_time = elapsed >= timedelta(days=1)
        elif freq == RebalanceFrequency.WEEKLY:
            should_by_time = elapsed >= timedelta(weeks=1)
        elif freq == RebalanceFrequency.MONTHLY:
            should_by_time = elapsed >= timedelta(days=30)
        elif freq == RebalanceFrequency.QUARTERLY:
            should_by_time = elapsed >= timedelta(days=90)
        elif freq == RebalanceFrequency.YEARLY:
            should_by_time = elapsed >= timedelta(days=365)
        
        if should_by_time:
            return True
        
        # Check threshold
        if freq == RebalanceFrequency.THRESHOLD or True:
            drift = self._calculate_drift(current_weights)
            if drift > self.config.rebalance_threshold:
                logger.info(f"Rebalance triggered by drift: {drift:.2%}")
                return True
        
        return False
    
    def _calculate_drift(
        self,
        current_weights: dict[str, float],
    ) -> float:
        """Calculate drift from target weights."""
        if self._last_weights is None:
            return 1.0
        
        drift = 0.0
        all_symbols = set(current_weights.keys()) | set(self._last_weights.keys())
        
        for sym in all_symbols:
            current = current_weights.get(sym, 0.0)
            target = self._last_weights.get(sym, 0.0)
            drift += abs(current - target)
        
        return drift / 2  # Normalize (max drift is 2.0)
    
    def _apply_constraints(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply weight constraints."""
        # Long only
        if self.config.long_only:
            weights = np.maximum(weights, 0)
        
        # Min/max weight
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        # Normalize
        total = np.sum(np.abs(weights))
        if total > 0:
            weights = weights / total * self.config.leverage
        
        return weights
    
    def get_rebalance_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate trades needed to rebalance.
        
        Args:
            current_weights: Current weights
            target_weights: Target weights
            portfolio_value: Total portfolio value
            prices: Current prices
        
        Returns:
            Dictionary of symbol -> shares to trade (positive = buy)
        """
        trades = {}
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for sym in all_symbols:
            current = current_weights.get(sym, 0.0)
            target = target_weights.get(sym, 0.0)
            
            # Value change
            value_change = (target - current) * portfolio_value
            
            # Shares to trade
            if sym in prices and prices[sym] > 0:
                shares = value_change / prices[sym]
                
                # Apply transaction cost filter
                if abs(value_change) > portfolio_value * self.config.transaction_cost * 10:
                    trades[sym] = shares
        
        return trades
    
    @property
    def last_result(self) -> OptimizationResult | None:
        """Get the last optimization result."""
        return self._optimization_history[-1] if self._optimization_history else None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "OptimizationMethod",
    "RebalanceFrequency",
    # Classes
    "OptimizationConfig",
    "OptimizationResult",
    "PortfolioOptimizer",
    # Functions
    "mean_variance_optimize",
    "min_variance_portfolio",
    "max_sharpe_portfolio",
    "risk_parity_portfolio",
    "equal_weight_portfolio",
    "inverse_volatility_portfolio",
    "hrp_portfolio",
]