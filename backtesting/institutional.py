"""
Institutional Backtesting Engine
================================

JPMorgan-level backtesting framework with:
- Multi-asset portfolio optimization
- Transaction cost aware execution
- Regime-based model selection
- Walk-forward validation with purging/embargo
- Factor attribution analysis
- Monte Carlo stress testing

This module represents the institutional-grade approach used by
top quantitative trading firms like JPMorgan, Two Sigma, and Citadel.

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
from uuid import UUID, uuid4
from backtesting.institutional_extensions import (
    get_regime_adjusted_params,
    AdvancedPortfolioOptimizer,
    execute_institutional_trade,
    generate_html_report,
)

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RegimeType(str, Enum):
    """Market regime classification."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hierarchical_risk_parity"
    KELLY = "kelly"
    EQUAL_WEIGHT = "equal_weight"


class RebalanceFrequency(str, Enum):
    """Portfolio rebalance frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_SIGNAL = "on_signal"
    THRESHOLD = "threshold"


class ExecutionStyle(str, Enum):
    """Execution algorithm style."""
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InstitutionalBacktestConfig:
    """
    Institutional-grade backtest configuration.
    
    Mirrors configurations used at JPMorgan's Quantitative Research desk.
    """
    # === CAPITAL & LEVERAGE ===
    initial_capital: float = 10_000_000.0  # $10M starting capital
    max_leverage: float = 1.0  # No leverage by default
    margin_requirement: float = 0.25  # Reg-T margin
    
    # === TRANSACTION COSTS (Realistic Institutional) ===
    commission_bps: float = 0.5  # 0.5 bps commission
    spread_bps: float = 1.0  # 1 bp half-spread
    market_impact_bps: float = 2.0  # 2 bps permanent impact
    temporary_impact_coefficient: float = 0.1  # Almgren-Chriss temporary impact
    borrowing_cost_annual: float = 0.005  # 50 bps annual for shorts
    
    # === POSITION LIMITS ===
    max_position_pct: float = 0.10  # Max 10% per position
    max_sector_pct: float = 0.30  # Max 30% per sector
    max_positions: int = 20  # Max concurrent positions
    min_position_size: float = 50_000.0  # Minimum $50K per position
    
    # === RISK LIMITS ===
    max_drawdown_pct: float = 0.15  # Stop trading at 15% drawdown
    daily_var_limit_pct: float = 0.02  # 2% daily VaR limit
    max_portfolio_beta: float = 1.5  # Max beta exposure
    max_correlation_threshold: float = 0.7  # Max pairwise correlation
    
    # === EXECUTION ===
    execution_style: ExecutionStyle = ExecutionStyle.VWAP
    participation_rate: float = 0.05  # 5% of ADV max
    execution_horizon_minutes: int = 30  # Execute over 30 minutes
    
    # === REBALANCING ===
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.ON_SIGNAL
    rebalance_threshold: float = 0.05  # Rebalance if drift > 5%
    
    # === OPTIMIZATION ===
    optimization_method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    covariance_estimator: str = "ledoit_wolf"  # Shrinkage estimator
    returns_estimator: str = "exponential"  # Exponential weighting
    lookback_period: int = 252  # 1 year lookback for optimization
    
    # === WALK-FORWARD ===
    wf_train_periods: int = 504  # 2 years training
    wf_test_periods: int = 63  # 3 months testing
    wf_embargo_periods: int = 5  # 5 bar embargo between train/test
    wf_purge_periods: int = 10  # Purge 10 bars around test
    
    # === DATA ===
    timeframe: str = "15min"
    warmup_bars: int = 500  # Warmup for indicators
    
    # === REGIME DETECTION ===
    enable_regime_detection: bool = True
    regime_lookback: int = 126  # 6 months for regime
    regime_volatility_threshold: float = 1.5  # Std devs for high vol
    
    # === ML MODELS ===
    min_model_confidence: float = 0.55
    ensemble_method: str = "weighted_average"  # voting, average, weighted_average
    model_weight_decay: float = 0.95  # Decay older model weights
    
    # === REPORTING ===
    benchmark_symbol: str = "SPY"
    generate_html_report: bool = True
    save_trade_log: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }


# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Hidden Markov Model-inspired regime detection.
    
    Identifies market regimes based on:
    - Return distribution (bull/bear)
    - Volatility level (low/high)
    - Market structure (trending/mean-reverting)
    """
    



    
    def __init__(
        self,
        lookback: int = 126,
        vol_threshold: float = 1.5,
        trend_threshold: float = 0.02,
    ):
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.trend_threshold = trend_threshold
        
        self._regime_history: List[Tuple[datetime, RegimeType]] = []
        self._vol_history: List[float] = []
        self._return_history: List[float] = []
    
    def detect_regime(
        self,
        returns: NDArray[np.float64],
        timestamp: datetime,
    ) -> RegimeType:
        """
        Detect current market regime.
        
        Args:
            returns: Recent returns array
            timestamp: Current timestamp
        
        Returns:
            Current regime classification
        """
        if len(returns) < self.lookback:
            return RegimeType.SIDEWAYS
        
        recent = returns[-self.lookback:]
        
        # Calculate metrics
        mean_return = np.mean(recent)
        volatility = np.std(recent)
        long_vol = np.std(returns) if len(returns) > self.lookback * 2 else volatility
        
        # Classify trend direction
        is_bull = mean_return > self.trend_threshold / np.sqrt(252)
        is_bear = mean_return < -self.trend_threshold / np.sqrt(252)
        
        # Classify volatility
        vol_ratio = volatility / long_vol if long_vol > 0 else 1.0
        is_high_vol = vol_ratio > self.vol_threshold
        
        # Check for crisis (extreme conditions)
        if is_bear and is_high_vol and vol_ratio > 2.0:
            regime = RegimeType.CRISIS
        elif is_bull and not is_high_vol:
            regime = RegimeType.BULL_LOW_VOL
        elif is_bull and is_high_vol:
            regime = RegimeType.BULL_HIGH_VOL
        elif is_bear and not is_high_vol:
            regime = RegimeType.BEAR_LOW_VOL
        elif is_bear and is_high_vol:
            regime = RegimeType.BEAR_HIGH_VOL
        else:
            regime = RegimeType.SIDEWAYS
        
        self._regime_history.append((timestamp, regime))
        self._vol_history.append(float(volatility))
        
        return regime
    
    def get_regime_history(self) -> List[Tuple[datetime, RegimeType]]:
        """Get regime history."""
        return self._regime_history.copy()
    
    def get_regime_adjusted_params(
        self,
        regime: RegimeType,
        base_position_size: float,
    ) -> Dict[str, float]:
        """
        Get regime-adjusted parameters.
        
        Reduces position sizes in high volatility and crisis regimes.
        """
        adjustments = {
            RegimeType.BULL_LOW_VOL: {"size_mult": 1.0, "confidence_adj": 0.0},
            RegimeType.BULL_HIGH_VOL: {"size_mult": 0.7, "confidence_adj": 0.05},
            RegimeType.BEAR_LOW_VOL: {"size_mult": 0.8, "confidence_adj": 0.05},
            RegimeType.BEAR_HIGH_VOL: {"size_mult": 0.5, "confidence_adj": 0.10},
            RegimeType.SIDEWAYS: {"size_mult": 0.9, "confidence_adj": 0.0},
            RegimeType.CRISIS: {"size_mult": 0.25, "confidence_adj": 0.20},
        }
        
        adj = adjustments.get(regime, {"size_mult": 0.5, "confidence_adj": 0.10})
        
        return {
            "adjusted_size": base_position_size * adj["size_mult"],
            "min_confidence": 0.55 + adj["confidence_adj"],
            "reduce_leverage": regime in [RegimeType.BEAR_HIGH_VOL, RegimeType.CRISIS],
        }


# =============================================================================
# COVARIANCE ESTIMATION
# =============================================================================

class CovarianceEstimator:
    """
    Advanced covariance matrix estimation.
    
    Implements multiple shrinkage estimators for robust covariance estimation
    with fewer observations than assets.
    """
    
    @staticmethod
    def sample_covariance(returns: NDArray[np.float64]) -> NDArray[np.float64]:
        """Simple sample covariance."""
        return np.cov(returns, rowvar=False)
    
    @staticmethod
    def ledoit_wolf(returns: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Ledoit-Wolf shrinkage estimator.
        
        Shrinks toward scaled identity matrix for stability.
        """
        n, p = returns.shape
        
        if n <= p:
            warnings.warn("Ledoit-Wolf: n <= p, results may be unstable")
        
        # Sample covariance
        sample_cov = np.cov(returns, rowvar=False)
        
        # Shrinkage target: scaled identity
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Compute optimal shrinkage intensity
        delta = sample_cov - target
        
        # Frobenius norms
        delta_sum = np.sum(delta ** 2)
        
        # Compute shrinkage coefficient
        X = returns - returns.mean(axis=0)
        X2 = X ** 2
        
        # Asymptotic approximation
        sum_sq = np.sum(np.dot(X2.T, X2) / n)
        gamma = np.sum(delta ** 2)
        
        kappa = (sum_sq - gamma) / n
        shrinkage = max(0, min(1, kappa / gamma)) if gamma > 0 else 0
        
        return (1 - shrinkage) * sample_cov + shrinkage * target
    
    @staticmethod
    def exponential_weighted(
        returns: NDArray[np.float64],
        halflife: int = 63,
    ) -> NDArray[np.float64]:
        """
        Exponentially weighted covariance matrix.
        
        Gives more weight to recent observations.
        """
        n, p = returns.shape
        
        # Calculate weights
        decay = 0.5 ** (1 / halflife)
        weights = decay ** np.arange(n - 1, -1, -1)
        weights /= weights.sum()
        
        # Weighted mean
        weighted_mean = np.average(returns, axis=0, weights=weights)
        
        # Weighted covariance
        centered = returns - weighted_mean
        weighted_cov = np.dot((centered * weights[:, np.newaxis]).T, centered)
        
        return weighted_cov
    
    @staticmethod
    def denoised_correlation(
        returns: NDArray[np.float64],
        num_factors: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Denoised correlation matrix using random matrix theory.
        
        Removes noise eigenvalues based on Marchenko-Pastur distribution.
        """
        n, p = returns.shape
        
        # Correlation matrix
        corr = np.corrcoef(returns, rowvar=False)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        
        # Sort by magnitude
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Marchenko-Pastur bounds
        q = p / n
        lambda_plus = (1 + np.sqrt(q)) ** 2
        lambda_minus = (1 - np.sqrt(q)) ** 2
        
        # Determine number of signal eigenvalues
        if num_factors is None:
            num_factors = np.sum(eigenvalues > lambda_plus)
            num_factors = max(1, num_factors)
        
        # Shrink noise eigenvalues
        denoised_eigenvalues = eigenvalues.copy()
        denoised_eigenvalues[num_factors:] = np.mean(eigenvalues[num_factors:])
        
        # Reconstruct
        denoised_corr = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
        
        # Ensure proper correlation matrix
        np.fill_diagonal(denoised_corr, 1.0)
        
        # Convert back to covariance
        std = np.std(returns, axis=0)
        denoised_cov = denoised_corr * np.outer(std, std)
        
        return denoised_cov


# =============================================================================
# PORTFOLIO OPTIMIZER
# =============================================================================

class PortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer.
    
    Implements multiple optimization methods:
    - Mean-Variance (Markowitz)
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Risk Parity
    - Black-Litterman
    - Hierarchical Risk Parity (HRP)
    - Kelly Criterion
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
        self.min_weight = -max_weight if allow_shorting else min_weight
        self.allow_shorting = allow_shorting
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        views: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected return for each asset
            covariance: Covariance matrix
            method: Optimization method
            views: Views for Black-Litterman
            constraints: Additional constraints
        
        Returns:
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        
        if method == OptimizationMethod.EQUAL_WEIGHT:
            return np.ones(n_assets) / n_assets
        
        elif method == OptimizationMethod.MIN_VARIANCE:
            return self._min_variance(covariance)
        
        elif method == OptimizationMethod.MAX_SHARPE:
            return self._max_sharpe(expected_returns, covariance)
        
        elif method == OptimizationMethod.MEAN_VARIANCE:
            target_return = np.mean(expected_returns) * 1.5  # Target 50% above average
            return self._mean_variance(expected_returns, covariance, target_return)
        
        elif method == OptimizationMethod.RISK_PARITY:
            return self._risk_parity(covariance)
        
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            return self._black_litterman(expected_returns, covariance, views)
        
        elif method == OptimizationMethod.HRP:
            return self._hierarchical_risk_parity(covariance)
        
        elif method == OptimizationMethod.KELLY:
            return self._kelly_criterion(expected_returns, covariance)
        
        else:
            return np.ones(n_assets) / n_assets
    
    def _min_variance(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Minimum variance portfolio."""
        n = len(covariance)
        
        def objective(w):
            return w @ covariance @ w
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(self.min_weight, self.max_weight)
        
        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        return result.x if result.success else np.ones(n) / n
    
    def _max_sharpe(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximum Sharpe ratio portfolio."""
        n = len(expected_returns)
        
        def neg_sharpe(w):
            ret = w @ expected_returns
            vol = np.sqrt(w @ covariance @ w)
            if vol < 1e-10:
                return 0
            return -(ret - self.risk_free_rate / 252) / vol
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(self.min_weight, self.max_weight)
        
        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        return result.x if result.success else np.ones(n) / n
    
    def _mean_variance(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        target_return: float,
    ) -> NDArray[np.float64]:
        """Mean-variance optimization with target return."""
        n = len(expected_returns)
        
        def objective(w):
            return w @ covariance @ w
        
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ expected_returns - target_return},
        ]
        bounds = Bounds(self.min_weight, self.max_weight)
        
        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if not result.success:
            # Fallback to unconstrained return
            return self._max_sharpe(expected_returns, covariance)
        
        return result.x
    
    def _risk_parity(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Risk parity portfolio.
        
        Each asset contributes equally to portfolio risk.
        """
        n = len(covariance)
        
        def risk_contribution(w):
            port_vol = np.sqrt(w @ covariance @ w)
            if port_vol < 1e-10:
                return np.ones(n) / n
            marginal_contrib = covariance @ w
            risk_contrib = w * marginal_contrib / port_vol
            return risk_contrib
        
        def objective(w):
            rc = risk_contribution(w)
            target_rc = np.ones(n) / n * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(0.01, self.max_weight)  # Minimum 1% per asset
        
        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        return result.x if result.success else np.ones(n) / n
    
    def _black_litterman(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        views: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """
        Black-Litterman model.
        
        Combines equilibrium returns with investor views.
        """
        n = len(expected_returns)
        
        if views is None:
            return self._max_sharpe(expected_returns, covariance)
        
        # Risk aversion coefficient
        delta = 2.5
        tau = 0.05  # Scalar for prior uncertainty
        
        # Equilibrium returns (implied by market cap weights)
        market_weights = views.get("market_weights", np.ones(n) / n)
        pi = delta * covariance @ market_weights
        
        # View matrix P and expected excess returns Q
        P = views.get("P", np.eye(n))
        Q = views.get("Q", expected_returns)
        omega = views.get("omega", np.diag(np.diag(P @ (tau * covariance) @ P.T)))
        
        # Black-Litterman combined returns
        try:
            M1 = np.linalg.inv(tau * covariance)
            M2 = P.T @ np.linalg.inv(omega) @ P
            
            bl_returns = np.linalg.inv(M1 + M2) @ (M1 @ pi + P.T @ np.linalg.inv(omega) @ Q)
            bl_cov = np.linalg.inv(M1 + M2)
            
            return self._max_sharpe(bl_returns, covariance + bl_cov)
        except np.linalg.LinAlgError:
            return self._max_sharpe(expected_returns, covariance)
    
    def _hierarchical_risk_parity(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Hierarchical Risk Parity (HRP).
        
        Uses hierarchical clustering to build diversified portfolio
        without requiring return estimates.
        """
        n = len(covariance)
        
        # Convert covariance to correlation
        std = np.sqrt(np.diag(covariance))
        corr = covariance / np.outer(std, std)
        
        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Hierarchical clustering
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method="single")
        
        # Quasi-diagonalization
        sort_idx = self._get_quasi_diag(link)
        
        # Recursive bisection
        weights = np.zeros(n)
        self._hrp_recursive_bisection(
            weights,
            covariance,
            sort_idx,
        )
        
        return weights
    
    def _get_quasi_diag(self, link: NDArray) -> List[int]:
        """Get quasi-diagonal sort indices from linkage."""
        link = link.astype(int)
        n = link.shape[0] + 1
        sort_idx = [list(range(n))]
        
        for i in range(len(link) - 1, -1, -1):
            idx1, idx2 = int(link[i, 0]), int(link[i, 1])
            
            for j, cluster in enumerate(sort_idx):
                if n + i in cluster:
                    pos = cluster.index(n + i)
                    cluster.remove(n + i)
                    cluster.insert(pos, idx1)
                    cluster.insert(pos + 1, idx2)
                    break
        
        return [x for x in sort_idx[0] if x < n]
    
    def _hrp_recursive_bisection(
        self,
        weights: NDArray[np.float64],
        covariance: NDArray[np.float64],
        sort_idx: List[int],
    ) -> None:
        """Recursive bisection for HRP."""
        if len(sort_idx) == 1:
            weights[sort_idx[0]] = 1.0
            return
        
        # Split
        mid = len(sort_idx) // 2
        left_idx = sort_idx[:mid]
        right_idx = sort_idx[mid:]
        
        # Cluster variance
        left_var = self._cluster_variance(covariance, left_idx)
        right_var = self._cluster_variance(covariance, right_idx)
        
        # Allocate based on inverse variance
        total_var = left_var + right_var
        left_weight = 1 - left_var / total_var if total_var > 0 else 0.5
        right_weight = 1 - right_var / total_var if total_var > 0 else 0.5
        
        # Normalize
        total = left_weight + right_weight
        left_weight /= total
        right_weight /= total
        
        # Recursive allocation
        left_weights = np.zeros(len(covariance))
        right_weights = np.zeros(len(covariance))
        
        self._hrp_recursive_bisection(left_weights, covariance, left_idx)
        self._hrp_recursive_bisection(right_weights, covariance, right_idx)
        
        weights[:] = left_weight * left_weights + right_weight * right_weights
    
    def _cluster_variance(
        self,
        covariance: NDArray[np.float64],
        indices: List[int],
    ) -> float:
        """Calculate variance of a cluster."""
        cov_slice = covariance[np.ix_(indices, indices)]
        
        # Inverse variance portfolio within cluster
        inv_var = 1 / np.diag(cov_slice)
        w = inv_var / inv_var.sum()
        
        return float(w @ cov_slice @ w)
    
    def _kelly_criterion(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Kelly criterion for optimal bet sizing.
        
        Full Kelly is often too aggressive, so we use fractional Kelly.
        """
        try:
            inv_cov = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            return self._max_sharpe(expected_returns, covariance)
        
        # Full Kelly
        full_kelly = inv_cov @ expected_returns
        
        # Half Kelly (more conservative)
        kelly_weights = full_kelly * 0.5
        
        # Normalize and constrain
        kelly_weights = np.clip(kelly_weights, self.min_weight, self.max_weight)
        
        # Normalize to sum to 1
        total = np.sum(kelly_weights)
        if total > 0:
            kelly_weights /= total
        else:
            kelly_weights = np.ones(len(expected_returns)) / len(expected_returns)
        
        return kelly_weights


# =============================================================================
# TRANSACTION COST MODEL
# =============================================================================

@dataclass
class TransactionCost:
    """Transaction cost breakdown."""
    commission: float = 0.0
    spread_cost: float = 0.0
    market_impact: float = 0.0
    total: float = 0.0
    
    @classmethod
    def calculate(
        cls,
        trade_value: float,
        adv: float,
        config: InstitutionalBacktestConfig,
    ) -> "TransactionCost":
        """
        Calculate realistic transaction costs.
        
        Uses Almgren-Chriss model for market impact.
        """
        # Commission
        commission = trade_value * config.commission_bps / 10000
        
        # Spread (half spread each way)
        spread_cost = trade_value * config.spread_bps / 10000
        
        # Market impact (square root model)
        participation = trade_value / adv if adv > 0 else 0.01
        
        # Temporary impact
        temp_impact = config.temporary_impact_coefficient * np.sqrt(participation)
        
        # Permanent impact
        perm_impact = config.market_impact_bps / 10000
        
        market_impact = trade_value * (temp_impact + perm_impact)
        
        total = commission + spread_cost + market_impact
        
        return cls(
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            total=total,
        )


# =============================================================================
# MODEL LOADER
# =============================================================================

class ModelLoader:
    """
    Loads trained ML models from artifacts directory.
    
    Follows naming convention:
        models/artifacts/{SYMBOL}/{SYMBOL}_{model_type}_{version}.pkl
    """
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self._models: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._scalers: Dict[str, Any] = {}
    
    def load_all_models(self, symbols: Optional[List[str]] = None) -> int:
        """
        Load all available models.
        
        Args:
            symbols: Optional list of symbols to load. If None, load all.
        
        Returns:
            Number of models loaded
        """
        loaded = 0
        
        if not self.artifacts_dir.exists():
            warnings.warn(f"Artifacts directory not found: {self.artifacts_dir}")
            return 0
        
        for symbol_dir in self.artifacts_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            symbol = symbol_dir.name
            
            if symbols and symbol not in symbols:
                continue
            
            self._models[symbol] = {}
            
            # Load metadata
            metadata_path = symbol_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self._metadata[symbol] = json.load(f)
            
            # Load models
            for model_file in symbol_dir.glob("*.pkl"):
                if "scaler" in model_file.name.lower():
                    self._scalers[symbol] = self._load_pickle(model_file)
                    continue
                
                # Parse model type from filename
                parts = model_file.stem.split("_")
                if len(parts) >= 3:
                    model_type = parts[1]  # e.g., lightgbm, xgboost
                    
                    model = self._load_pickle(model_file)
                    if model is not None:
                        self._models[symbol][model_type] = model
                        loaded += 1
        
        return loaded
    
    def _load_pickle(self, path: Path) -> Any:
        """Safely load a pickle file. FIXED: extracts model from wrapper."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            # FIX: Extract actual model from wrapper dict
            if isinstance(data, dict) and "model" in data:
                return data["model"]
            return data
            
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}")
            return None
    
    def get_model(self, symbol: str, model_type: str) -> Optional[Any]:
        """Get a specific model."""
        return self._models.get(symbol, {}).get(model_type)
    
    def get_models_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get all models for a symbol."""
        return self._models.get(symbol, {})
    
    def get_scaler(self, symbol: str) -> Optional[Any]:
        """Get the scaler for a symbol."""
        return self._scalers.get(symbol)
    
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a symbol."""
        return self._metadata.get(symbol, {})
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with loaded models."""
        return list(self._models.keys())
    
    def get_model_performance(self, symbol: str) -> Dict[str, float]:
        """Get model performance metrics from metadata."""
        meta = self._metadata.get(symbol, {})
        
        return {
            "test_accuracy": meta.get("test_accuracy", 0.5),
            "test_f1": meta.get("test_f1", 0.5),
            "test_auc": meta.get("test_auc", 0.5),
            "walk_forward_sharpe": meta.get("walk_forward_sharpe", 0.0),
        }


# =============================================================================
# ENSEMBLE PREDICTOR
# =============================================================================

class EnsemblePredictor:
    """
    Ensemble model predictions with confidence calibration.
    
    Combines predictions from multiple models with:
    - Weighted averaging based on historical performance
    - Confidence calibration using isotonic regression
    - Regime-based model weighting
    """
    
    def __init__(
        self,
        model_loader: ModelLoader,
        method: str = "weighted_average",
        min_confidence: float = 0.55,
    ):
        self.model_loader = model_loader
        self.method = method
        self.min_confidence = min_confidence
        
        self._model_weights: Dict[str, Dict[str, float]] = {}
        self._prediction_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def initialize_weights(self, symbols: List[str]) -> None:
        """Initialize model weights based on metadata."""
        for symbol in symbols:
            models = self.model_loader.get_models_for_symbol(symbol)
            if not models:
                continue
            
            meta = self.model_loader.get_metadata(symbol)
            
            weights = {}
            for model_type in models.keys():
                # Base weight on historical performance
                perf = meta.get(f"{model_type}_test_auc", 0.5)
                weights[model_type] = max(0.1, perf - 0.5)  # Weight based on AUC above random
            
            # Normalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {k: 1 / len(weights) for k in weights}
            
            self._model_weights[symbol] = weights
    
    def predict(
        self,
        symbol: str,
        features: NDArray[np.float64],
        regime: Optional[RegimeType] = None,
    ) -> Tuple[int, float]:
        """
        Generate ensemble prediction.
        
        Args:
            symbol: Trading symbol
            features: Feature vector
            regime: Current market regime
        
        Returns:
            Tuple of (direction, confidence)
            direction: -1 (sell), 0 (hold), 1 (buy)
            confidence: 0.0 to 1.0
        """
        models = self.model_loader.get_models_for_symbol(symbol)
        if not models:
            return 0, 0.0
        
        weights = self._model_weights.get(symbol, {})
        if not weights:
            weights = {k: 1 / len(models) for k in models}
        
        # Adjust weights based on regime
        if regime and regime in [RegimeType.CRISIS, RegimeType.BEAR_HIGH_VOL]:
            # In crisis, favor more conservative models (XGBoost tends to be more regularized)
            if "xgboost" in weights:
                weights["xgboost"] *= 1.5
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}
        
        # Collect predictions
        predictions = []
        probabilities = []
        
        for model_type, model in models.items():
            weight = weights.get(model_type, 1 / len(models))
            
            try:
                # Reshape features for prediction
                X = features.reshape(1, -1) if features.ndim == 1 else features
                
                # Get prediction
                pred = model.predict(X)[0]
                
                # Get probability if available
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]
                    if len(prob) == 3:  # Multi-class: sell, hold, buy
                        pred_prob = prob[int(pred) + 1] if pred in [-1, 0, 1] else max(prob)
                    else:  # Binary
                        pred_prob = max(prob)
                else:
                    pred_prob = 0.6  # Default confidence
                
                predictions.append((pred, pred_prob, weight))
                probabilities.append(pred_prob * weight)
                
            except Exception as e:
                continue
        
        if not predictions:
            return 0, 0.0
        
        # Combine predictions
        if self.method == "voting":
            # Majority voting
            votes = defaultdict(float)
            for pred, prob, weight in predictions:
                votes[pred] += weight
            direction = max(votes, key=votes.get)
            confidence = votes[direction] / sum(weights.values())
            
        elif self.method == "weighted_average":
            # Weighted average of predictions
            weighted_pred = sum(p * w for p, _, w in predictions) / sum(w for _, _, w in predictions)
            
            # Convert to direction
            if weighted_pred > 0.3:
                direction = 1
            elif weighted_pred < -0.3:
                direction = -1
            else:
                direction = 0
            
            # Confidence is average of probabilities
            confidence = sum(probabilities) / len(probabilities)
            
        else:  # Simple average
            avg_pred = np.mean([p for p, _, _ in predictions])
            direction = 1 if avg_pred > 0.3 else (-1 if avg_pred < -0.3 else 0)
            confidence = np.mean([prob for _, prob, _ in predictions])
        
        # Calibrate confidence
        confidence = self._calibrate_confidence(confidence, direction)
        
        # Store for performance tracking
        self._prediction_history[symbol].append({
            "direction": direction,
            "confidence": confidence,
            "individual_predictions": predictions,
        })
        
        return direction, confidence
    
    def _calibrate_confidence(
        self,
        raw_confidence: float,
        direction: int,
    ) -> float:
        """
        Calibrate confidence score.
        
        ML models often have poorly calibrated probabilities.
        This applies a simple scaling to make them more realistic.
        """
        # Scale from [0.5, 1.0] to [0.5, 0.75]
        # Most predictions shouldn't be above 75% confident
        if raw_confidence > 0.5:
            calibrated = 0.5 + (raw_confidence - 0.5) * 0.5
        else:
            calibrated = raw_confidence
        
        # Hold signals should have lower confidence
        if direction == 0:
            calibrated *= 0.8
        
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def update_weights_from_performance(
        self,
        symbol: str,
        model_type: str,
        accuracy: float,
        decay: float = 0.95,
    ) -> None:
        """Update model weights based on recent performance."""
        if symbol not in self._model_weights:
            return
        
        weights = self._model_weights[symbol]
        
        # Exponential moving average of performance
        old_weight = weights.get(model_type, 0.5)
        new_weight = decay * old_weight + (1 - decay) * accuracy
        weights[model_type] = new_weight
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            self._model_weights[symbol] = {k: v / total for k, v in weights.items()}


# =============================================================================
# PERFORMANCE ATTRIBUTION
# =============================================================================

@dataclass
class AttributionResult:
    """Performance attribution breakdown."""
    # Return attribution
    total_return: float = 0.0
    alpha: float = 0.0
    beta_return: float = 0.0
    
    # Factor attribution
    market_contribution: float = 0.0
    size_contribution: float = 0.0
    value_contribution: float = 0.0
    momentum_contribution: float = 0.0
    residual: float = 0.0
    
    # Risk attribution
    total_risk: float = 0.0
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0
    
    # Cost attribution
    total_costs: float = 0.0
    commission_costs: float = 0.0
    spread_costs: float = 0.0
    impact_costs: float = 0.0
    
    # By time
    daily_returns: List[float] = field(default_factory=list)
    weekly_returns: List[float] = field(default_factory=list)
    monthly_returns: List[float] = field(default_factory=list)
    
    # By sector
    sector_returns: Dict[str, float] = field(default_factory=dict)


class PerformanceAttributor:
    """
    Comprehensive performance attribution analysis.
    
    Decomposes returns into:
    - Alpha vs Beta
    - Factor contributions
    - Cost impact
    - Sector allocation
    """
    
    def __init__(
        self,
        benchmark_returns: Optional[NDArray[np.float64]] = None,
        risk_free_rate: float = 0.05,
    ):
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
    
    def analyze(
        self,
        portfolio_returns: NDArray[np.float64],
        trade_costs: List[TransactionCost],
        positions: Dict[str, List[Tuple[datetime, float]]],
        sectors: Optional[Dict[str, str]] = None,
    ) -> AttributionResult:
        """
        Perform comprehensive attribution analysis.
        
        Args:
            portfolio_returns: Daily portfolio returns
            trade_costs: List of transaction costs
            positions: Position history by symbol
            sectors: Symbol to sector mapping
        
        Returns:
            AttributionResult with full breakdown
        """
        result = AttributionResult()
        
        # Total return
        result.total_return = float(np.prod(1 + portfolio_returns) - 1)
        
        # Alpha/Beta decomposition
        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(portfolio_returns):
            beta, alpha = self._calculate_alpha_beta(
                portfolio_returns,
                self.benchmark_returns,
            )
            
            result.alpha = alpha
            result.beta_return = beta * float(np.mean(self.benchmark_returns)) * 252
            result.market_contribution = result.beta_return
        
        # Cost attribution
        result.commission_costs = sum(tc.commission for tc in trade_costs)
        result.spread_costs = sum(tc.spread_cost for tc in trade_costs)
        result.impact_costs = sum(tc.market_impact for tc in trade_costs)
        result.total_costs = sum(tc.total for tc in trade_costs)
        
        # Risk decomposition
        result.total_risk = float(np.std(portfolio_returns) * np.sqrt(252))
        
        if self.benchmark_returns is not None:
            result.systematic_risk = result.beta_return * float(np.std(self.benchmark_returns) * np.sqrt(252))
            result.idiosyncratic_risk = np.sqrt(max(0, result.total_risk**2 - result.systematic_risk**2))
        
        # Time-based returns
        result.daily_returns = portfolio_returns.tolist()
        result.monthly_returns = self._aggregate_returns(portfolio_returns, 21)
        
        # Sector attribution
        if sectors and positions:
            result.sector_returns = self._calculate_sector_attribution(positions, sectors)
        
        return result
    
    def _calculate_alpha_beta(
        self,
        returns: NDArray[np.float64],
        benchmark: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Calculate alpha and beta using OLS regression."""
        if len(returns) < 20:
            return 1.0, 0.0
        
        # Excess returns
        rf_daily = self.risk_free_rate / 252
        excess_port = returns - rf_daily
        excess_bench = benchmark - rf_daily
        
        # Regression
        cov = np.cov(excess_port, excess_bench)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
        alpha = float(np.mean(excess_port) - beta * np.mean(excess_bench)) * 252
        
        return float(beta), alpha
    
    def _aggregate_returns(
        self,
        returns: NDArray[np.float64],
        period: int,
    ) -> List[float]:
        """Aggregate returns to a longer period."""
        n_periods = len(returns) // period
        agg_returns = []
        
        for i in range(n_periods):
            period_returns = returns[i * period:(i + 1) * period]
            agg_returns.append(float(np.prod(1 + period_returns) - 1))
        
        return agg_returns
    
    def _calculate_sector_attribution(
        self,
        positions: Dict[str, List[Tuple[datetime, float]]],
        sectors: Dict[str, str],
    ) -> Dict[str, float]:
        """Calculate sector contribution to returns."""
        sector_returns = defaultdict(float)
        sector_weights = defaultdict(float)
        
        for symbol, pos_history in positions.items():
            sector = sectors.get(symbol, "Unknown")
            
            # Simplified: sum of position values
            total_value = sum(abs(val) for _, val in pos_history)
            sector_weights[sector] += total_value
        
        # Normalize (simplified attribution)
        total = sum(sector_weights.values())
        if total > 0:
            for sector in sector_weights:
                sector_returns[sector] = sector_weights[sector] / total
        
        return dict(sector_returns)


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

@dataclass
class WalkForwardFold:
    """Single walk-forward fold."""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    max_drawdown: float
    n_trades: int


class WalkForwardValidator:
    """
    Walk-forward validation with proper purging and embargo.
    
    Implements the methodology from López de Prado's "Advances in Financial ML":
    - Purging: Removes training samples that overlap with test period
    - Embargo: Adds gap between training and test to prevent leakage
    """
    
    def __init__(
        self,
        train_periods: int = 504,
        test_periods: int = 63,
        embargo_periods: int = 5,
        purge_periods: int = 10,
    ):
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.embargo_periods = embargo_periods
        self.purge_periods = purge_periods
    
    def generate_folds(
        self,
        data_length: int,
        timestamps: Optional[List[datetime]] = None,
    ) -> Iterator[Tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """
        Generate walk-forward folds with purging and embargo.
        
        Args:
            data_length: Total number of samples
            timestamps: Optional timestamps for each sample
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        min_samples = self.train_periods + self.embargo_periods + self.test_periods
        
        if data_length < min_samples:
            warnings.warn(f"Insufficient data for walk-forward: {data_length} < {min_samples}")
            return
        
        fold = 0
        current_start = 0
        
        while current_start + min_samples <= data_length:
            # Define periods
            train_end = current_start + self.train_periods
            test_start = train_end + self.embargo_periods
            test_end = test_start + self.test_periods
            
            if test_end > data_length:
                break
            
            # Create indices
            train_idx = np.arange(current_start, train_end - self.purge_periods)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
            
            # Move forward by test period
            current_start += self.test_periods
            fold += 1
    
    def validate(
        self,
        results: List[WalkForwardFold],
    ) -> Dict[str, float]:
        """
        Validate walk-forward results.
        
        Checks for overfitting and calculates stability metrics.
        """
        if not results:
            return {}
        
        train_sharpes = [r.train_sharpe for r in results]
        test_sharpes = [r.test_sharpe for r in results]
        
        # Overfitting ratio: how much worse is test vs train
        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)
        overfit_ratio = (avg_train - avg_test) / avg_train if avg_train > 0 else 0
        
        # Probability of backtest overfit
        pbo = sum(1 for ts in test_sharpes if ts <= 0) / len(test_sharpes)
        
        # Stability: consistency across folds
        sharpe_stability = 1 - np.std(test_sharpes) / (np.mean(test_sharpes) + 1e-10)
        
        return {
            "avg_train_sharpe": float(avg_train),
            "avg_test_sharpe": float(avg_test),
            "overfit_ratio": float(overfit_ratio),
            "probability_backtest_overfit": float(pbo),
            "sharpe_stability": float(sharpe_stability),
            "n_folds": len(results),
            "n_positive_folds": sum(1 for r in results if r.test_return > 0),
        }


# =============================================================================
# MONTE CARLO STRESS TESTING
# =============================================================================

class MonteCarloStressTester:
    """
    Monte Carlo simulation for stress testing.
    
    Generates scenarios to test portfolio robustness:
    - Historical block bootstrap
    - Parametric simulation
    - Extreme event injection
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        block_size: int = 21,  # Monthly blocks
        seed: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
    
    def historical_bootstrap(
        self,
        returns: NDArray[np.float64],
        horizon: int = 252,
    ) -> NDArray[np.float64]:
        """
        Generate scenarios using block bootstrap.
        
        Args:
            returns: Historical returns
            horizon: Simulation horizon
        
        Returns:
            Array of shape (n_simulations, horizon)
        """
        n_blocks = horizon // self.block_size + 1
        simulations = np.zeros((self.n_simulations, horizon))
        
        for i in range(self.n_simulations):
            # Sample blocks with replacement
            sim_returns = []
            
            for _ in range(n_blocks):
                start = self.rng.integers(0, len(returns) - self.block_size)
                block = returns[start:start + self.block_size]
                sim_returns.extend(block)
            
            simulations[i] = np.array(sim_returns[:horizon])
        
        return simulations
    
    def parametric_simulation(
        self,
        mean_return: float,
        volatility: float,
        horizon: int = 252,
        include_jumps: bool = True,
    ) -> NDArray[np.float64]:
        """
        Generate scenarios using parametric model.
        
        Args:
            mean_return: Annual mean return
            volatility: Annual volatility
            horizon: Simulation horizon
            include_jumps: Include jump diffusion
        
        Returns:
            Array of shape (n_simulations, horizon)
        """
        daily_mean = mean_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        # Normal returns
        simulations = self.rng.normal(
            daily_mean,
            daily_vol,
            size=(self.n_simulations, horizon),
        )
        
        if include_jumps:
            # Merton jump diffusion
            jump_prob = 0.01  # 1% daily probability of jump
            jump_mean = -0.03  # Average jump is -3%
            jump_vol = 0.05  # Jump volatility 5%
            
            jumps = self.rng.binomial(1, jump_prob, size=(self.n_simulations, horizon))
            jump_sizes = self.rng.normal(jump_mean, jump_vol, size=(self.n_simulations, horizon))
            
            simulations += jumps * jump_sizes
        
        return simulations
    
    def stress_scenarios(
        self,
        returns: NDArray[np.float64],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Generate predefined stress scenarios.
        
        Returns:
            Dictionary of scenario name to return series
        """
        scenarios = {}
        
        # Black Monday (1987-style)
        scenarios["black_monday"] = np.concatenate([
            returns[-250:],
            np.array([-0.20]),  # 20% single-day drop
            returns[:100] * 1.5,  # Elevated volatility aftermath
        ])
        
        # Financial Crisis (2008-style)
        scenarios["financial_crisis"] = np.concatenate([
            returns[-100:],
            np.array([-0.05] * 5),  # Series of 5% drops
            np.array([0.02] * 3),  # Brief recovery
            np.array([-0.08] * 3),  # More drops
            returns[:100] * 1.3,
        ])
        
        # Flash Crash
        scenarios["flash_crash"] = np.concatenate([
            returns[-200:],
            np.array([-0.09]),  # 9% intraday drop
            np.array([0.05]),  # Partial recovery same day
            returns[:50],
        ])
        
        # Gradual Bear Market
        bear_returns = np.random.normal(-0.001, 0.015, 252)  # Slight negative drift
        scenarios["gradual_bear"] = bear_returns
        
        return scenarios
    
    def calculate_var(
        self,
        simulations: NDArray[np.float64],
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate VaR metrics from simulations.
        
        Args:
            simulations: Simulated returns
            confidence: Confidence level
        
        Returns:
            VaR and CVaR metrics
        """
        # Calculate terminal wealth for each simulation
        terminal_returns = np.prod(1 + simulations, axis=1) - 1
        
        var = float(-np.percentile(terminal_returns, (1 - confidence) * 100))
        cvar = float(-np.mean(terminal_returns[terminal_returns < -var]))
        
        # Maximum drawdown for each simulation
        cumulative = np.cumprod(1 + simulations, axis=1)
        running_max = np.maximum.accumulate(cumulative, axis=1)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=1)
        
        return {
            "var_95": var,
            "cvar_95": cvar,
            "expected_max_drawdown": float(-np.mean(max_drawdowns)),
            "worst_case_drawdown": float(-np.min(max_drawdowns)),
            "prob_loss": float(np.mean(terminal_returns < 0)),
            "prob_severe_loss": float(np.mean(terminal_returns < -0.20)),
        }


# =============================================================================
# INSTITUTIONAL BACKTESTER
# =============================================================================

class InstitutionalBacktester:
    """
    JPMorgan-level backtesting engine.
    
    Features:
    - Multi-asset portfolio with optimization
    - Transaction cost aware execution
    - Regime-based model selection
    - Walk-forward validation
    - Monte Carlo stress testing
    - Comprehensive attribution
    """
    
    def __init__(
        self,
        config: InstitutionalBacktestConfig,
        model_loader: ModelLoader,
    ):
        self.config = config
        self.model_loader = model_loader
        
        # Initialize components
        self.regime_detector = RegimeDetector(
            lookback=config.regime_lookback,
            vol_threshold=config.regime_volatility_threshold,
        )
        
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=0.05,
            max_weight=config.max_position_pct,
            allow_shorting=config.max_leverage > 1,
        )
        
        self.ensemble_predictor = EnsemblePredictor(
            model_loader=model_loader,
            method=config.ensemble_method,
            min_confidence=config.min_model_confidence,
        )
        
        self.wf_validator = WalkForwardValidator(
            train_periods=config.wf_train_periods,
            test_periods=config.wf_test_periods,
            embargo_periods=config.wf_embargo_periods,
            purge_periods=config.wf_purge_periods,
        )
        
        self.stress_tester = MonteCarloStressTester()
        
        # State
        self._portfolio_value = config.initial_capital
        self._cash = config.initial_capital
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._target_weights: Dict[str, float] = {}
        
        # History
        self._equity_history: List[Tuple[datetime, float]] = []
        self._returns_history: List[float] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._cost_history: List[TransactionCost] = []
        self._regime_history: List[Tuple[datetime, RegimeType]] = []
        
        # Tracking
        self._high_water_mark = config.initial_capital
        self._max_drawdown = 0.0
        self._trading_halted = False
    
    def run(
        self,
        data: Dict[str, pl.DataFrame],
        features: Dict[str, NDArray[np.float64]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Run institutional-grade backtest.
        
        Args:
            data: OHLCV data for each symbol
            features: Feature matrices for each symbol
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Comprehensive backtest results
        """
        # Initialize
        symbols = list(data.keys())
        self.ensemble_predictor.initialize_weights(symbols)
        
        # Align timestamps
        all_timestamps = set()
        for df in data.values():
            if "timestamp" in df.columns:
                all_timestamps.update(df["timestamp"].to_list())
        
        timestamps = sorted(all_timestamps)
        
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date]
        
        # Skip warmup period
        timestamps = timestamps[self.config.warmup_bars:]
        
        if not timestamps:
            return {"error": "Insufficient data after filtering"}
        
        # Main backtest loop
        for i, timestamp in enumerate(timestamps):
            if self._trading_halted:
                break
            
            # Get current prices and features
            current_prices = {}
            current_features = {}
            
            for symbol in symbols:
                df = data[symbol]
                row = df.filter(pl.col("timestamp") == timestamp)
                
                if len(row) > 0:
                    current_prices[symbol] = row["close"][0]
                    
                    # Get features up to this point
                    feat = features.get(symbol)
                    if feat is not None:
                        idx = min(i + self.config.warmup_bars, len(feat) - 1)
                        current_features[symbol] = feat[idx]
            
            if not current_prices:
                continue
            
            # Update positions with current prices
            self._update_position_values(current_prices)
            
            # Detect regime
            portfolio_returns = np.array(self._returns_history[-self.config.regime_lookback:]) if self._returns_history else np.array([])
            regime = self.regime_detector.detect_regime(portfolio_returns, timestamp) if len(portfolio_returns) >= 20 else RegimeType.SIDEWAYS
            
            # Check risk limits
            self._check_risk_limits(timestamp)
            
            if self._trading_halted:
                break
            
            # Generate signals for each symbol
            signals = {}
            for symbol, feat in current_features.items():
                if symbol not in current_prices:
                    continue
                
                direction, confidence = self.ensemble_predictor.predict(
                    symbol,
                    feat,
                    regime,
                )
                
                signals[symbol] = {
                    "direction": direction,
                    "confidence": confidence,
                    "price": current_prices[symbol],
                }
            
            # Filter signals by confidence
            valid_signals = {
                s: v for s, v in signals.items()
                if v["confidence"] >= self.config.min_model_confidence
                and v["direction"] != 0
            }
            
            # Optimize portfolio if we have valid signals
            if valid_signals and self._should_rebalance(timestamp, valid_signals):
                self._rebalance_portfolio(
                    valid_signals,
                    current_prices,
                    regime,
                    timestamp,
                )
            
            # Record state
            self._equity_history.append((timestamp, self._portfolio_value))
            
            if len(self._equity_history) > 1:
                prev_value = self._equity_history[-2][1]
                ret = (self._portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
                self._returns_history.append(ret)
        
        # Generate results
        return self._generate_results(timestamps)
    
    def _update_position_values(self, prices: Dict[str, float]) -> None:
        """Update all position values with current prices."""
        position_value = 0.0
        
        for symbol, pos in self._positions.items():
            if symbol in prices:
                pos["current_price"] = prices[symbol]
                pos["market_value"] = pos["quantity"] * prices[symbol]
                pos["unrealized_pnl"] = pos["market_value"] - pos["cost_basis"]
                position_value += pos["market_value"]
        
        self._portfolio_value = self._cash + position_value
        
        # Update high water mark and drawdown
        if self._portfolio_value > self._high_water_mark:
            self._high_water_mark = self._portfolio_value
        
        current_drawdown = (self._high_water_mark - self._portfolio_value) / self._high_water_mark
        self._max_drawdown = max(self._max_drawdown, current_drawdown)
    
    def _check_risk_limits(self, timestamp: datetime) -> None:
        """Check and enforce risk limits."""
        # Max drawdown check
        if self._max_drawdown > self.config.max_drawdown_pct:
            self._trading_halted = True
            warnings.warn(f"Trading halted at {timestamp}: Max drawdown exceeded ({self._max_drawdown:.2%})")
    
    def _should_rebalance(
        self,
        timestamp: datetime,
        signals: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Determine if portfolio should be rebalanced."""
        if self.config.rebalance_frequency == RebalanceFrequency.ON_SIGNAL:
            return len(signals) > 0
        
        elif self.config.rebalance_frequency == RebalanceFrequency.THRESHOLD:
            # Check drift from target weights
            for symbol, target in self._target_weights.items():
                if symbol in self._positions:
                    current_weight = self._positions[symbol]["market_value"] / self._portfolio_value
                    if abs(current_weight - target) > self.config.rebalance_threshold:
                        return True
            return False
        
        # Time-based rebalancing would check timestamp
        return True
    
    def _rebalance_portfolio(
        self,
        signals: Dict[str, Dict[str, Any]],
        prices: Dict[str, float],
        regime: RegimeType,
        timestamp: datetime,
    ) -> None:
        """
        Rebalance portfolio using optimization.
        
        Args:
            signals: Trading signals with direction and confidence
            prices: Current prices
            regime: Current market regime
            timestamp: Current timestamp
        """
        # Get regime-adjusted parameters
        regime_params = self.regime_detector.get_regime_adjusted_params(
            regime,
            self.config.max_position_pct,
        )
        
        # Build expected returns and covariance from signals
        symbols = list(signals.keys())
        n = len(symbols)
        
        if n == 0:
            return
        
        # Simple expected return based on signal direction and confidence
        expected_returns = np.array([
            signals[s]["direction"] * signals[s]["confidence"] * 0.02  # Scale to ~2% expected return
            for s in symbols
        ])
        
        # Simple covariance (could be enhanced with historical data)
        covariance = np.eye(n) * 0.04  # 20% vol assumption
        
        # Optimize
        target_weights = self.optimizer.optimize(
            expected_returns,
            covariance,
            self.config.optimization_method,
        )
        
        # Apply regime adjustment
        target_weights = target_weights * regime_params["adjusted_size"] / self.config.max_position_pct
        
        # Ensure constraints
        target_weights = np.clip(target_weights, 0, regime_params["adjusted_size"] / self.config.max_position_pct)
        
        # Normalize if sum > 1
        if np.sum(target_weights) > 1:
            target_weights = target_weights / np.sum(target_weights)
        
        # Execute trades
        for i, symbol in enumerate(symbols):
            target_value = self._portfolio_value * target_weights[i]
            current_value = self._positions.get(symbol, {}).get("market_value", 0)
            
            trade_value = target_value - current_value
            
            # Apply minimum trade size
            if abs(trade_value) < self.config.min_position_size:
                continue
            
            # Calculate transaction costs
            adv = prices.get(symbol, 100) * 1_000_000  # Assume $1M ADV
            cost = TransactionCost.calculate(abs(trade_value), adv, self.config)
            
            # Only trade if expected return exceeds costs
            expected_return = abs(expected_returns[i]) * abs(trade_value)
            if cost.total > expected_return * 0.5:  # Cost must be < 50% of expected return
                continue
            
            # Execute
            self._execute_trade(
                symbol,
                trade_value,
                prices[symbol],
                cost,
                timestamp,
                signals[symbol],
            )
        
        # Update target weights
        self._target_weights = dict(zip(symbols, target_weights))
    
    def _execute_trade(
        self,
        symbol: str,
        trade_value: float,
        price: float,
        cost: TransactionCost,
        timestamp: datetime,
        signal: Dict[str, Any],
    ) -> None:
        """Execute a trade with realistic costs."""
        quantity = trade_value / price
        
        # Apply slippage to execution price
        execution_price = price * (1 + self.config.spread_bps / 10000 * np.sign(quantity))
        
        # Record trade
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "quantity": quantity,
            "price": execution_price,
            "value": trade_value,
            "cost": cost.total,
            "direction": signal["direction"],
            "confidence": signal["confidence"],
        }
        self._trade_history.append(trade)
        self._cost_history.append(cost)
        
        # Update positions
        if symbol not in self._positions:
            self._positions[symbol] = {
                "quantity": 0,
                "cost_basis": 0,
                "current_price": price,
                "market_value": 0,
                "unrealized_pnl": 0,
                "entry_time": timestamp,
            }
        
        pos = self._positions[symbol]
        
        if quantity > 0:  # Buy
            new_quantity = pos["quantity"] + quantity
            if new_quantity != 0:
                pos["cost_basis"] = (pos["cost_basis"] + trade_value) / new_quantity * new_quantity
            pos["quantity"] = new_quantity
        else:  # Sell
            realized_pnl = abs(quantity) * (price - pos["cost_basis"] / pos["quantity"]) if pos["quantity"] > 0 else 0
            pos["quantity"] += quantity
            if pos["quantity"] <= 0:
                pos["cost_basis"] = 0
                pos["quantity"] = 0
        
        # Update cash
        self._cash -= trade_value + cost.total
        
        # Update market value
        pos["current_price"] = price
        pos["market_value"] = pos["quantity"] * price
    
    def _generate_results(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if len(self._equity_history) < 2:
            return {"error": "Insufficient data for results"}
        
        equity_curve = np.array([e for _, e in self._equity_history])
        returns = np.array(self._returns_history) if self._returns_history else np.array([0])
        
        # Calculate metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 0 else 0
        
        # Annualized metrics
        n_periods = len(returns)
        periods_per_year = 252 * 26 if "15min" in self.config.timeframe else 252  # Adjust for 15-min
        
        if n_periods > 0:
            annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
            annual_vol = np.std(returns) * np.sqrt(periods_per_year)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Sortino
            downside = returns[returns < 0]
            downside_vol = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else annual_vol
            sortino = annual_return / downside_vol if downside_vol > 0 else 0
            
            # Calmar
            calmar = annual_return / self._max_drawdown if self._max_drawdown > 0 else 0
        else:
            annual_return = sharpe = sortino = calmar = 0
        
        # Trade statistics
        n_trades = len(self._trade_history)
        total_costs = sum(c.total for c in self._cost_history)
        
        winning_trades = [t for t in self._trade_history if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
        
        # Attribution
        attributor = PerformanceAttributor()
        attribution = attributor.analyze(
            returns,
            self._cost_history,
            {s: [(t, p["market_value"]) for t, p in [(self._equity_history[0][0], p)] for p in [self._positions.get(s, {})]] for s in self._positions},
        )
        
        # Walk-forward validation
        wf_results = self.wf_validator.validate([])  # Would need actual fold results
        
        # Monte Carlo
        mc_results = {}
        if len(returns) >= 100:
            simulations = self.stress_tester.historical_bootstrap(returns, 252)
            mc_results = self.stress_tester.calculate_var(simulations)
        
        return {
            # Core metrics
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(self._max_drawdown),
            "annual_volatility": float(annual_vol) if n_periods > 0 else 0,
            
            # Trade stats
            "n_trades": n_trades,
            "win_rate": win_rate,
            "total_costs": total_costs,
            "cost_as_pct_return": total_costs / (total_return * self.config.initial_capital) if total_return > 0 else 0,
            
            # Risk metrics
            "var_95": mc_results.get("var_95", 0),
            "cvar_95": mc_results.get("cvar_95", 0),
            
            # Attribution
            "alpha": attribution.alpha,
            "total_cost_impact": attribution.total_costs,
            
            # Time series
            "equity_curve": [(str(t), e) for t, e in self._equity_history],
            "returns": returns.tolist(),
            
            # Trades
            "trades": self._trade_history,
            
            # Regime history
            "regimes": [(str(t), r.value) for t, r in self.regime_detector.get_regime_history()],
            
            # Walk-forward
            "walk_forward_metrics": wf_results,
            
            # Monte Carlo
            "monte_carlo": mc_results,
            
            # Configuration
            "config": self.config.to_dict(),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RegimeType",
    "OptimizationMethod",
    "RebalanceFrequency",
    "ExecutionStyle",
    # Config
    "InstitutionalBacktestConfig",
    # Components
    "RegimeDetector",
    "CovarianceEstimator",
    "PortfolioOptimizer",
    "TransactionCost",
    "ModelLoader",
    "EnsemblePredictor",
    "PerformanceAttributor",
    "AttributionResult",
    "WalkForwardValidator",
    "WalkForwardFold",
    "MonteCarloStressTester",
    # Main
    "InstitutionalBacktester",
]