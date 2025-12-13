"""
Institutional-Grade Portfolio Allocation Algorithms
Advanced HRP Implementation for Multi-Asset Portfolio Management

Features:
- Hierarchical Risk Parity (HRP) with full Lopez de Prado implementation
- Multiple distance metrics and linkage methods
- Rolling covariance estimation with shrinkage
- Daily rebalancing support
- Integration with BacktestEngine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import minimize
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DistanceMetric(Enum):
    """Distance metrics for correlation-based clustering"""
    CORRELATION = "correlation"  # d = sqrt(0.5 * (1 - rho))
    ANGULAR = "angular"  # d = arccos(rho) / pi
    ABSOLUTE = "absolute"  # d = sqrt(1 - |rho|)
    EUCLIDEAN = "euclidean"  # Standard euclidean on returns


class LinkageMethod(Enum):
    """Hierarchical clustering linkage methods"""
    SINGLE = "single"  # Minimum distance
    COMPLETE = "complete"  # Maximum distance
    AVERAGE = "average"  # UPGMA
    WARD = "ward"  # Ward's variance minimization


class RiskMeasure(Enum):
    """Risk measures for allocation"""
    VARIANCE = "variance"
    STANDARD_DEVIATION = "std"
    SEMI_VARIANCE = "semi_variance"
    CVAR = "cvar"  # Conditional VaR
    MAD = "mad"  # Mean Absolute Deviation


@dataclass
class HRPConfig:
    """Configuration for HRP optimization"""
    # Clustering parameters
    distance_metric: DistanceMetric = DistanceMetric.CORRELATION
    linkage_method: LinkageMethod = LinkageMethod.SINGLE

    # Risk parameters
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE
    cvar_confidence: float = 0.95

    # Covariance estimation
    use_shrinkage: bool = True
    shrinkage_target: str = "constant_correlation"  # or "diagonal", "identity"
    shrinkage_intensity: Optional[float] = None  # None = auto (Ledoit-Wolf)

    # Rolling window parameters
    lookback_window: int = 252  # Days for covariance estimation
    min_observations: int = 60  # Minimum observations required

    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector_weight: float = 0.40

    # Rebalancing
    rebalance_threshold: float = 0.05  # 5% drift triggers rebalance


@dataclass
class AllocationResult:
    """Result of portfolio allocation"""
    weights: Dict[str, float]
    timestamp: datetime
    method: str

    # Diagnostics
    portfolio_volatility: float
    effective_n_assets: float
    diversification_ratio: float
    risk_contributions: Dict[str, float]

    # Clustering info
    cluster_order: List[str]
    linkage_matrix: Optional[np.ndarray] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class CovarianceEstimator:
    """
    Advanced covariance matrix estimation with shrinkage.

    Implements Ledoit-Wolf shrinkage to improve estimation stability
    for high-dimensional portfolios where n_assets > n_observations.
    """

    def __init__(
        self,
        shrinkage_target: str = "constant_correlation",
        shrinkage_intensity: Optional[float] = None
    ):
        self.shrinkage_target = shrinkage_target
        self.shrinkage_intensity = shrinkage_intensity

    def estimate(
        self,
        returns: pd.DataFrame,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix with optional shrinkage.

        Args:
            returns: DataFrame of asset returns
            annualize: Whether to annualize the covariance

        Returns:
            Estimated covariance matrix
        """
        n_obs, n_assets = returns.shape

        # Sample covariance
        sample_cov = returns.cov()

        if self.shrinkage_intensity == 0:
            cov = sample_cov
        else:
            # Compute shrinkage target
            target = self._compute_shrinkage_target(returns, sample_cov)

            # Compute shrinkage intensity
            if self.shrinkage_intensity is not None:
                delta = self.shrinkage_intensity
            else:
                delta = self._ledoit_wolf_intensity(returns, sample_cov, target)

            # Apply shrinkage: Sigma_shrunk = delta * Target + (1-delta) * Sample
            cov = delta * target + (1 - delta) * sample_cov

        if annualize:
            cov = cov * 252

        return cov

    def _compute_shrinkage_target(
        self,
        returns: pd.DataFrame,
        sample_cov: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the shrinkage target matrix."""
        n_assets = len(sample_cov)

        if self.shrinkage_target == "diagonal":
            # Diagonal matrix with sample variances
            target = pd.DataFrame(
                np.diag(np.diag(sample_cov.values)),
                index=sample_cov.index,
                columns=sample_cov.columns
            )

        elif self.shrinkage_target == "identity":
            # Scaled identity matrix
            avg_var = np.trace(sample_cov.values) / n_assets
            target = pd.DataFrame(
                np.eye(n_assets) * avg_var,
                index=sample_cov.index,
                columns=sample_cov.columns
            )

        elif self.shrinkage_target == "constant_correlation":
            # Constant correlation matrix
            std_devs = np.sqrt(np.diag(sample_cov.values))
            corr = sample_cov.values / np.outer(std_devs, std_devs)
            np.fill_diagonal(corr, 1.0)

            # Average correlation (excluding diagonal)
            n = len(corr)
            avg_corr = (corr.sum() - n) / (n * (n - 1))

            # Constant correlation matrix
            target_corr = np.full((n, n), avg_corr)
            np.fill_diagonal(target_corr, 1.0)

            target = pd.DataFrame(
                target_corr * np.outer(std_devs, std_devs),
                index=sample_cov.index,
                columns=sample_cov.columns
            )
        else:
            target = sample_cov

        return target

    def _ledoit_wolf_intensity(
        self,
        returns: pd.DataFrame,
        sample_cov: pd.DataFrame,
        target: pd.DataFrame
    ) -> float:
        """
        Compute optimal Ledoit-Wolf shrinkage intensity.

        Based on Ledoit & Wolf (2004) "A Well-Conditioned Estimator
        for Large-Dimensional Covariance Matrices"
        """
        n_obs, n_assets = returns.shape
        X = returns.values

        # Demeaned returns
        X_centered = X - X.mean(axis=0)

        # Sample covariance (without Bessel correction for this computation)
        S = sample_cov.values * (n_obs - 1) / n_obs
        F = target.values

        # Estimate pi (sum of squared estimation errors)
        X2 = X_centered ** 2
        pi_mat = np.dot(X2.T, X2) / n_obs - S ** 2
        pi = np.sum(pi_mat)

        # Estimate gamma (distance between sample and target)
        gamma = np.sum((S - F) ** 2)

        # Estimate rho (asymptotic variance of target estimator)
        # Simplified version
        rho = np.sum(np.diag(pi_mat))

        # Optimal shrinkage intensity
        kappa = (pi - rho) / gamma
        delta = max(0, min(1, kappa / n_obs))

        return delta

    def estimate_correlation(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimate correlation matrix."""
        cov = self.estimate(returns, annualize=False)
        std_devs = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std_devs, std_devs)
        np.fill_diagonal(corr, 1.0)

        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


class HierarchicalRiskParityAllocator:
    """
    Institutional-Grade Hierarchical Risk Parity (HRP) Allocator.

    Implements the complete HRP algorithm from Lopez de Prado's
    "Building Diversified Portfolios that Outperform Out-of-Sample".

    Key advantages over traditional MVO:
    1. No matrix inversion required (stable with singular matrices)
    2. Respects hierarchical structure of asset relationships
    3. Superior out-of-sample performance
    4. Works well when n_assets > n_observations

    The algorithm has three stages:
    1. Tree Clustering: Build hierarchy from correlation-based distances
    2. Quasi-Diagonalization: Reorder covariance matrix by similarity
    3. Recursive Bisection: Allocate by inverse cluster variance
    """

    def __init__(self, config: Optional[HRPConfig] = None):
        self.config = config or HRPConfig()
        self.cov_estimator = CovarianceEstimator(
            shrinkage_target=self.config.shrinkage_target,
            shrinkage_intensity=self.config.shrinkage_intensity if not self.config.use_shrinkage else None
        )

        # Cache for rolling optimization
        self._weights_cache: Dict[str, Dict[str, float]] = {}
        self._last_rebalance: Optional[datetime] = None

    def compute_distance_matrix(
        self,
        returns: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Stage 1a: Compute correlation-based distance matrix.

        The distance metric must satisfy triangle inequality for valid clustering.

        Common choices:
        - Correlation distance: d_ij = sqrt(0.5 * (1 - rho_ij))
        - Angular distance: d_ij = arccos(rho_ij) / pi
        - Absolute distance: d_ij = sqrt(1 - |rho_ij|)

        Returns:
            Tuple of (condensed distance matrix, correlation matrix)
        """
        # Estimate correlation matrix
        corr = self.cov_estimator.estimate_correlation(returns)
        corr_values = corr.values.copy()

        # Ensure correlation is in valid range [-1, 1]
        corr_values = np.clip(corr_values, -1.0, 1.0)

        # Compute distance based on metric
        metric = self.config.distance_metric

        if metric == DistanceMetric.CORRELATION:
            # d_ij = sqrt(0.5 * (1 - rho_ij))
            dist = np.sqrt(0.5 * (1 - corr_values))

        elif metric == DistanceMetric.ANGULAR:
            # d_ij = arccos(rho_ij) / pi
            dist = np.arccos(corr_values) / np.pi

        elif metric == DistanceMetric.ABSOLUTE:
            # d_ij = sqrt(1 - |rho_ij|)
            dist = np.sqrt(1 - np.abs(corr_values))

        else:  # EUCLIDEAN
            # Standard euclidean on standardized returns
            standardized = (returns - returns.mean()) / returns.std()
            dist = squareform(pdist(standardized.T.values, metric='euclidean'))

        # Ensure diagonal is exactly zero and matrix is symmetric
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2

        # Convert to condensed form for scipy
        condensed_dist = squareform(dist, checks=False)

        return condensed_dist, corr

    def hierarchical_clustering(
        self,
        condensed_dist: np.ndarray
    ) -> np.ndarray:
        """
        Stage 1b: Perform agglomerative hierarchical clustering.

        Uses scipy's linkage function with the specified method.

        Args:
            condensed_dist: Condensed distance matrix from squareform

        Returns:
            Linkage matrix (n-1 x 4 array)
        """
        method = self.config.linkage_method.value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linkage = sch.linkage(condensed_dist, method=method)

        return linkage

    def quasi_diagonalize(
        self,
        linkage: np.ndarray,
        n_assets: int
    ) -> List[int]:
        """
        Stage 2: Quasi-diagonalize the covariance matrix.

        Reorders assets so that similar assets are placed together,
        making the covariance matrix approximately block-diagonal.

        This is achieved by traversing the dendrogram and extracting
        the leaf ordering.

        Args:
            linkage: Linkage matrix from hierarchical clustering
            n_assets: Number of assets

        Returns:
            List of asset indices in quasi-diagonal order
        """
        return list(sch.leaves_list(linkage))

    def get_cluster_variance(
        self,
        cov: np.ndarray,
        cluster_indices: List[int],
        returns: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate cluster risk measure (variance, CVaR, etc.).

        For variance: computes the variance of the minimum-variance
        portfolio within the cluster.

        For other measures: uses the specified risk measure.
        """
        if len(cluster_indices) == 0:
            return 1e-10

        # Get cluster covariance submatrix
        cov_cluster = cov[np.ix_(cluster_indices, cluster_indices)]

        risk_measure = self.config.risk_measure

        if risk_measure == RiskMeasure.VARIANCE:
            # Inverse-variance portfolio within cluster
            diag = np.diag(cov_cluster)
            diag = np.where(diag > 0, diag, 1e-10)
            inv_var = 1.0 / diag
            weights = inv_var / np.sum(inv_var)

            # Cluster variance
            cluster_var = np.dot(weights, np.dot(cov_cluster, weights))
            return max(cluster_var, 1e-10)

        elif risk_measure == RiskMeasure.STANDARD_DEVIATION:
            diag = np.diag(cov_cluster)
            diag = np.where(diag > 0, diag, 1e-10)
            inv_std = 1.0 / np.sqrt(diag)
            weights = inv_std / np.sum(inv_std)
            cluster_var = np.dot(weights, np.dot(cov_cluster, weights))
            return max(np.sqrt(cluster_var), 1e-10)

        elif risk_measure == RiskMeasure.SEMI_VARIANCE:
            if returns is not None:
                cluster_returns = returns.iloc[:, cluster_indices]
                downside = cluster_returns[cluster_returns < 0]
                semi_cov = downside.cov().values * 252
                if not np.isnan(semi_cov).any():
                    diag = np.diag(semi_cov)
                    diag = np.where(diag > 0, diag, 1e-10)
                    inv_var = 1.0 / diag
                    weights = inv_var / np.sum(inv_var)
                    return max(np.dot(weights, np.dot(semi_cov, weights)), 1e-10)
            # Fallback to variance
            diag = np.diag(cov_cluster)
            diag = np.where(diag > 0, diag, 1e-10)
            inv_var = 1.0 / diag
            weights = inv_var / np.sum(inv_var)
            return max(np.dot(weights, np.dot(cov_cluster, weights)), 1e-10)

        elif risk_measure == RiskMeasure.CVAR:
            if returns is not None:
                cluster_returns = returns.iloc[:, cluster_indices]
                # Equal-weight CVaR for cluster
                portfolio_returns = cluster_returns.mean(axis=1)
                var = np.percentile(portfolio_returns, (1 - self.config.cvar_confidence) * 100)
                cvar = -portfolio_returns[portfolio_returns <= var].mean()
                return max(cvar if not np.isnan(cvar) else 0.01, 1e-10)
            return 0.01

        elif risk_measure == RiskMeasure.MAD:
            if returns is not None:
                cluster_returns = returns.iloc[:, cluster_indices]
                portfolio_returns = cluster_returns.mean(axis=1)
                mad = np.abs(portfolio_returns - portfolio_returns.mean()).mean()
                return max(mad * np.sqrt(252), 1e-10)
            return 0.01

        # Default to variance
        diag = np.diag(cov_cluster)
        diag = np.where(diag > 0, diag, 1e-10)
        inv_var = 1.0 / diag
        weights = inv_var / np.sum(inv_var)
        return max(np.dot(weights, np.dot(cov_cluster, weights)), 1e-10)

    def recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_indices: List[int],
        returns: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Stage 3: Recursive bisection for weight allocation.

        This is the core HRP allocation algorithm:
        1. Start with all assets in one cluster with weight 1
        2. Split cluster at midpoint into left and right
        3. Calculate variance of each sub-cluster
        4. Allocate parent weight inversely proportional to variance
        5. Recurse until each cluster has single asset

        Args:
            cov: Covariance matrix (in original asset order)
            sorted_indices: Quasi-diagonalized asset ordering
            returns: Optional returns for non-variance risk measures

        Returns:
            Weight array (in original asset order)
        """
        n = len(sorted_indices)
        weights = np.zeros(n)

        # Initialize: all items in one cluster with weight 1
        # Use iterative approach with explicit stack
        stack = [(sorted_indices, 1.0)]

        while stack:
            cluster, cluster_weight = stack.pop()

            if len(cluster) == 1:
                # Terminal case: assign weight to single asset
                weights[cluster[0]] = cluster_weight
            else:
                # Bisect cluster
                mid = len(cluster) // 2
                left_cluster = cluster[:mid]
                right_cluster = cluster[mid:]

                # Calculate variance of each sub-cluster
                left_var = self.get_cluster_variance(cov, left_cluster, returns)
                right_var = self.get_cluster_variance(cov, right_cluster, returns)

                # Allocate inversely proportional to variance
                # alpha = 1 - right_var / (left_var + right_var)
                #       = left_var / (left_var + right_var) <- for right cluster
                total_inv_var = 1.0 / left_var + 1.0 / right_var
                left_weight = (1.0 / left_var) / total_inv_var
                right_weight = 1.0 - left_weight

                # Add sub-clusters to stack
                stack.append((left_cluster, cluster_weight * left_weight))
                stack.append((right_cluster, cluster_weight * right_weight))

        return weights

    def allocate(
        self,
        returns: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None
    ) -> AllocationResult:
        """
        Compute HRP portfolio weights.

        Full implementation of the three-stage HRP algorithm:
        1. Tree Clustering
        2. Quasi-Diagonalization
        3. Recursive Bisection

        Args:
            returns: DataFrame of asset returns (columns = assets)
            sector_map: Optional mapping of symbols to sectors for constraints

        Returns:
            AllocationResult with weights and diagnostics
        """
        symbols = list(returns.columns)
        n_assets = len(symbols)

        if n_assets == 0:
            return AllocationResult(
                weights={},
                timestamp=datetime.now(),
                method="HRP",
                portfolio_volatility=0,
                effective_n_assets=0,
                diversification_ratio=0,
                risk_contributions={},
                cluster_order=[]
            )

        if n_assets == 1:
            return AllocationResult(
                weights={symbols[0]: 1.0},
                timestamp=datetime.now(),
                method="HRP",
                portfolio_volatility=returns.iloc[:, 0].std() * np.sqrt(252),
                effective_n_assets=1,
                diversification_ratio=1,
                risk_contributions={symbols[0]: 1.0},
                cluster_order=symbols
            )

        # Check minimum observations
        if len(returns) < self.config.min_observations:
            logger.warning(
                f"Insufficient observations ({len(returns)} < {self.config.min_observations}), "
                "using equal weights"
            )
            eq_weight = 1.0 / n_assets
            return AllocationResult(
                weights={s: eq_weight for s in symbols},
                timestamp=datetime.now(),
                method="HRP_EQUAL_FALLBACK",
                portfolio_volatility=0,
                effective_n_assets=n_assets,
                diversification_ratio=1,
                risk_contributions={s: eq_weight for s in symbols},
                cluster_order=symbols
            )

        # Stage 1: Tree Clustering
        condensed_dist, corr = self.compute_distance_matrix(returns)
        linkage = self.hierarchical_clustering(condensed_dist)

        # Stage 2: Quasi-Diagonalization
        sorted_indices = self.quasi_diagonalize(linkage, n_assets)

        # Stage 3: Estimate covariance and allocate
        cov = self.cov_estimator.estimate(returns, annualize=True)
        weights = self.recursive_bisection(cov.values, sorted_indices, returns)

        # Apply constraints
        weights = self._apply_constraints(weights, symbols, sector_map)

        # Compute diagnostics
        w = weights
        cov_np = cov.values

        # Portfolio volatility
        port_var = np.dot(w, np.dot(cov_np, w))
        port_vol = np.sqrt(port_var)

        # Effective number of assets (inverse HHI)
        effective_n = 1.0 / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0

        # Diversification ratio = weighted avg vol / portfolio vol
        asset_vols = np.sqrt(np.diag(cov_np))
        weighted_vol = np.dot(w, asset_vols)
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1

        # Risk contributions
        marginal_contrib = np.dot(cov_np, w)
        risk_contrib = w * marginal_contrib / port_vol if port_vol > 0 else w

        # Build result
        cluster_order = [symbols[i] for i in sorted_indices]
        weight_dict = {symbols[i]: weights[i] for i in range(n_assets)}
        risk_dict = {symbols[i]: risk_contrib[i] for i in range(n_assets)}

        return AllocationResult(
            weights=weight_dict,
            timestamp=datetime.now(),
            method="HRP",
            portfolio_volatility=port_vol,
            effective_n_assets=effective_n,
            diversification_ratio=div_ratio,
            risk_contributions=risk_dict,
            cluster_order=cluster_order,
            linkage_matrix=linkage,
            metadata={
                'n_assets': n_assets,
                'n_observations': len(returns),
                'distance_metric': self.config.distance_metric.value,
                'linkage_method': self.config.linkage_method.value,
                'risk_measure': self.config.risk_measure.value,
                'shrinkage_used': self.config.use_shrinkage
            }
        )

    def _apply_constraints(
        self,
        weights: np.ndarray,
        symbols: List[str],
        sector_map: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """Apply weight constraints."""
        # Min/max weight constraints
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        # Sector constraints
        if sector_map and self.config.max_sector_weight < 1.0:
            sector_weights = {}
            for i, symbol in enumerate(symbols):
                sector = sector_map.get(symbol, 'Other')
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weights[i]

            # Scale down sectors exceeding limit
            for sector, total_weight in sector_weights.items():
                if total_weight > self.config.max_sector_weight:
                    scale = self.config.max_sector_weight / total_weight
                    for i, symbol in enumerate(symbols):
                        if sector_map.get(symbol, 'Other') == sector:
                            weights[i] *= scale

        # Renormalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total

        return weights

    def rolling_allocate(
        self,
        returns: pd.DataFrame,
        rebalance_dates: Optional[List[datetime]] = None,
        sector_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Perform rolling HRP allocation for backtesting.

        Args:
            returns: Full returns DataFrame
            rebalance_dates: Specific dates to rebalance (default: daily)
            sector_map: Optional sector mapping

        Returns:
            DataFrame of weights over time (index=date, columns=symbols)
        """
        symbols = list(returns.columns)
        lookback = self.config.lookback_window

        if rebalance_dates is None:
            # Daily rebalancing after warmup
            rebalance_dates = returns.index[lookback:].tolist()

        weight_history = []

        for date in rebalance_dates:
            # Get lookback window
            date_loc = returns.index.get_loc(date)
            if date_loc < lookback:
                continue

            window_returns = returns.iloc[date_loc - lookback:date_loc]

            # Allocate
            result = self.allocate(window_returns, sector_map)

            # Store weights
            row = {'date': date}
            row.update(result.weights)
            weight_history.append(row)

        df = pd.DataFrame(weight_history)
        if not df.empty:
            df = df.set_index('date')
            # Fill missing columns with 0
            for symbol in symbols:
                if symbol not in df.columns:
                    df[symbol] = 0

        return df


class HRPBacktestIntegration:
    """
    Integration layer between HRP allocator and BacktestEngine.

    Manages daily rebalancing and weight updates during backtesting.
    """

    def __init__(
        self,
        config: Optional[HRPConfig] = None,
        rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    ):
        self.allocator = HierarchicalRiskParityAllocator(config)
        self.rebalance_frequency = rebalance_frequency

        # State
        self._returns_buffer: Dict[str, List[float]] = {}
        self._last_weights: Dict[str, float] = {}
        self._last_rebalance_date: Optional[datetime] = None
        self._dates_since_rebalance: int = 0

    def update_returns(
        self,
        symbol: str,
        daily_return: float
    ) -> None:
        """Add daily return for symbol."""
        if symbol not in self._returns_buffer:
            self._returns_buffer[symbol] = []
        self._returns_buffer[symbol].append(daily_return)

        # Trim to lookback window
        max_len = self.allocator.config.lookback_window
        if len(self._returns_buffer[symbol]) > max_len:
            self._returns_buffer[symbol] = self._returns_buffer[symbol][-max_len:]

    def get_allocation(
        self,
        current_date: datetime,
        available_symbols: List[str],
        sector_map: Optional[Dict[str, str]] = None,
        force_rebalance: bool = False
    ) -> Dict[str, float]:
        """
        Get current allocation weights.

        Rebalances if needed based on frequency setting.
        """
        # Check if rebalance needed
        should_rebalance = force_rebalance

        if self._last_rebalance_date is None:
            should_rebalance = True
        elif self.rebalance_frequency == 'daily':
            should_rebalance = True
        elif self.rebalance_frequency == 'weekly':
            self._dates_since_rebalance += 1
            if self._dates_since_rebalance >= 5:
                should_rebalance = True
                self._dates_since_rebalance = 0
        elif self.rebalance_frequency == 'monthly':
            self._dates_since_rebalance += 1
            if self._dates_since_rebalance >= 21:
                should_rebalance = True
                self._dates_since_rebalance = 0

        if not should_rebalance and self._last_weights:
            return self._last_weights

        # Build returns DataFrame
        symbols_with_data = [
            s for s in available_symbols
            if s in self._returns_buffer and len(self._returns_buffer[s]) >= self.allocator.config.min_observations
        ]

        if not symbols_with_data:
            # Equal weight fallback
            n = len(available_symbols)
            return {s: 1.0 / n for s in available_symbols} if n > 0 else {}

        # Create aligned returns DataFrame
        min_len = min(len(self._returns_buffer[s]) for s in symbols_with_data)
        returns_data = {
            s: self._returns_buffer[s][-min_len:]
            for s in symbols_with_data
        }
        returns_df = pd.DataFrame(returns_data)

        # Allocate
        result = self.allocator.allocate(returns_df, sector_map)

        # Add zero weights for symbols without data
        for s in available_symbols:
            if s not in result.weights:
                result.weights[s] = 0.0

        self._last_weights = result.weights
        self._last_rebalance_date = current_date

        return result.weights

    def reset(self) -> None:
        """Reset state for new backtest."""
        self._returns_buffer = {}
        self._last_weights = {}
        self._last_rebalance_date = None
        self._dates_since_rebalance = 0


# Convenience functions for direct use

def compute_hrp_weights(
    returns: pd.DataFrame,
    config: Optional[HRPConfig] = None
) -> Dict[str, float]:
    """
    Compute HRP weights for a returns DataFrame.

    Args:
        returns: DataFrame of asset returns
        config: Optional HRP configuration

    Returns:
        Dictionary of symbol -> weight
    """
    allocator = HierarchicalRiskParityAllocator(config)
    result = allocator.allocate(returns)
    return result.weights


def compute_correlation_distance(
    correlation: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute correlation-based distance matrix.

    d_ij = sqrt(0.5 * (1 - rho_ij))

    Args:
        correlation: Correlation matrix

    Returns:
        Distance matrix
    """
    dist = np.sqrt(0.5 * (1 - correlation.values))
    np.fill_diagonal(dist, 0)
    return pd.DataFrame(dist, index=correlation.index, columns=correlation.columns)
