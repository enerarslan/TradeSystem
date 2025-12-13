#!/usr/bin/env python3
"""
Monte Carlo Stress Testing with Combinatorial Purged Cross-Validation (CPCV)

Institutional-Grade Strategy Validation Framework

Features:
1. Bootstrap resampling with replacement preserving serial correlation
2. Combinatorial Purged Cross-Validation (CPCV) for unbiased performance
3. Value at Risk (VaR) and Conditional VaR (CVaR) under stress scenarios
4. Probability of Loss calculation
5. Deflated Sharpe Ratio integration
6. Tail risk analysis and drawdown distribution

Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado

Usage:
    python scripts/stress_test.py --backtest-results results.pkl --n-simulations 1000
    python scripts/stress_test.py --trades-csv trades.csv --n-simulations 1000
"""

import argparse
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for Monte Carlo stress testing."""
    # Monte Carlo settings
    n_simulations: int = 1000
    random_seed: Optional[int] = 42
    preserve_autocorrelation: bool = True
    block_size: Optional[int] = None  # Auto-calculate if None

    # CPCV settings
    n_splits: int = 10
    n_test_splits: int = 2
    purge_window: int = 5  # Days to purge around test set
    embargo_window: int = 2  # Days to embargo after test set

    # Risk metrics
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95

    # Validation thresholds
    min_dsr: float = 0.95  # Minimum Deflated Sharpe Ratio
    max_prob_loss: float = 0.05  # Maximum probability of loss
    max_drawdown_95: float = 0.20  # 95th percentile max drawdown


@dataclass
class StressTestResult:
    """Results from Monte Carlo stress testing."""
    # Basic statistics
    original_sharpe: float
    original_return: float
    original_max_drawdown: float

    # Monte Carlo distributions
    sharpe_distribution: np.ndarray
    return_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray

    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Probability metrics
    prob_loss: float
    prob_worse_than_original: float
    prob_drawdown_exceeds_20: float

    # Percentiles
    sharpe_5th_percentile: float
    sharpe_95th_percentile: float
    return_5th_percentile: float
    return_95th_percentile: float

    # Validation
    passed_validation: bool
    validation_details: Dict[str, Any]

    # CPCV results
    cpcv_sharpe_mean: float
    cpcv_sharpe_std: float
    cpcv_paths: List[Dict]

    # DSR integration
    deflated_sharpe_ratio: float
    haircut_sharpe_ratio: float

    # Metadata
    n_simulations: int
    timestamp: datetime = field(default_factory=datetime.now)


class BlockBootstrap:
    """
    Block Bootstrap for time series with serial correlation.

    Standard bootstrap assumes i.i.d. samples, which is violated by
    financial time series. Block bootstrap preserves autocorrelation
    by resampling contiguous blocks of observations.

    Methods:
    - Non-overlapping blocks (Carlstein, 1986)
    - Overlapping blocks/Moving blocks (Kunsch, 1989)
    - Stationary bootstrap (Politis & Romano, 1994)
    """

    def __init__(
        self,
        block_size: Optional[int] = None,
        method: str = 'moving',
        random_seed: Optional[int] = None
    ):
        """
        Initialize BlockBootstrap.

        Args:
            block_size: Size of bootstrap blocks (auto if None)
            method: 'moving', 'non_overlapping', or 'stationary'
            random_seed: Random seed for reproducibility
        """
        self.block_size = block_size
        self.method = method
        self._rng = np.random.default_rng(random_seed)

    def _optimal_block_size(
        self,
        series: np.ndarray
    ) -> int:
        """
        Calculate optimal block size using Politis-White method.

        Based on autocorrelation structure of the series.
        """
        n = len(series)

        # Compute autocorrelations
        max_lag = min(n // 4, 50)
        acf = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
        acf = acf[len(acf) // 2:len(acf) // 2 + max_lag + 1]
        acf = acf / acf[0]

        # Find lag where autocorrelation becomes insignificant
        se = 1 / np.sqrt(n)
        significant_lags = np.where(np.abs(acf[1:]) > 2 * se)[0]

        if len(significant_lags) == 0:
            optimal = max(1, int(n ** (1 / 3)))
        else:
            # Rule: block_size ~ 1.5 * max(significant_lag)
            optimal = max(1, int(1.5 * (significant_lags[-1] + 1)))

        # Bound by practical limits
        return max(5, min(optimal, n // 10))

    def resample(
        self,
        data: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate bootstrap sample preserving serial correlation.

        Args:
            data: Original time series
            n_samples: Desired sample size (default: same as original)

        Returns:
            Resampled time series
        """
        n = len(data)
        n_samples = n_samples or n

        # Determine block size
        if self.block_size is None:
            block_size = self._optimal_block_size(data)
        else:
            block_size = self.block_size

        if self.method == 'non_overlapping':
            return self._non_overlapping_bootstrap(data, n_samples, block_size)
        elif self.method == 'stationary':
            return self._stationary_bootstrap(data, n_samples, block_size)
        else:  # moving/overlapping
            return self._moving_block_bootstrap(data, n_samples, block_size)

    def _moving_block_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        block_size: int
    ) -> np.ndarray:
        """Moving block bootstrap with overlapping blocks."""
        n = len(data)
        n_blocks = int(np.ceil(n_samples / block_size))

        # Number of possible starting positions
        n_positions = n - block_size + 1

        # Sample block starting positions
        starts = self._rng.integers(0, n_positions, size=n_blocks)

        # Concatenate blocks
        blocks = [data[start:start + block_size] for start in starts]
        result = np.concatenate(blocks)

        return result[:n_samples]

    def _non_overlapping_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        block_size: int
    ) -> np.ndarray:
        """Non-overlapping block bootstrap."""
        n = len(data)
        n_blocks = n // block_size

        # Create non-overlapping block indices
        block_indices = np.arange(n_blocks)

        # Sample blocks
        n_needed = int(np.ceil(n_samples / block_size))
        sampled = self._rng.choice(block_indices, size=n_needed, replace=True)

        # Build result
        result = []
        for idx in sampled:
            start = idx * block_size
            result.extend(data[start:start + block_size])

        return np.array(result[:n_samples])

    def _stationary_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        avg_block_size: int
    ) -> np.ndarray:
        """
        Stationary bootstrap with random block lengths.

        Block lengths follow geometric distribution with mean = avg_block_size.
        This produces stationary resampled series.
        """
        n = len(data)
        p = 1 / avg_block_size  # Geometric distribution parameter

        result = []
        current_pos = self._rng.integers(0, n)

        while len(result) < n_samples:
            result.append(data[current_pos])

            # Decide whether to continue block or start new
            if self._rng.random() < p:
                # Start new block
                current_pos = self._rng.integers(0, n)
            else:
                # Continue block
                current_pos = (current_pos + 1) % n

        return np.array(result[:n_samples])


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Standard k-fold CV is biased for financial data due to:
    1. Temporal leakage (future information in training)
    2. Overlapping labels (same event in train and test)

    CPCV addresses these through:
    1. Purging: Removing training samples overlapping with test labels
    2. Embargo: Removing samples immediately after test set
    3. Combinatorial: Testing all possible train/test combinations

    Reference: Lopez de Prado, "Advances in Financial Machine Learning"
    """

    def __init__(
        self,
        n_splits: int = 10,
        n_test_splits: int = 2,
        purge_window: int = 5,
        embargo_window: int = 2
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of groups
            n_test_splits: Number of groups in test set
            purge_window: Observations to purge around test
            embargo_window: Observations to embargo after test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

        # Number of combinations
        self.n_paths = int(comb(n_splits, n_test_splits))

    def get_n_paths(self) -> int:
        """Get number of backtest paths."""
        return self.n_paths

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None
    ):
        """
        Generate train/test indices for each CPCV path.

        Args:
            X: Features array
            y: Labels (optional)
            groups: Group labels (optional)
            times: Time indices for purging (optional)

        Yields:
            Tuple of (train_indices, test_indices, path_info)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Split into n_splits groups
        group_size = n_samples // self.n_splits
        group_boundaries = [i * group_size for i in range(self.n_splits + 1)]
        group_boundaries[-1] = n_samples  # Include remainder

        # Generate all combinations of test groups
        from itertools import combinations
        test_group_combos = list(combinations(range(self.n_splits), self.n_test_splits))

        for path_idx, test_groups in enumerate(test_group_combos):
            test_groups = set(test_groups)
            train_groups = set(range(self.n_splits)) - test_groups

            # Build test indices
            test_indices = []
            for g in sorted(test_groups):
                start = group_boundaries[g]
                end = group_boundaries[g + 1]
                test_indices.extend(indices[start:end])

            # Build train indices with purging
            train_indices = []
            test_start = min(test_indices)
            test_end = max(test_indices)

            for g in sorted(train_groups):
                start = group_boundaries[g]
                end = group_boundaries[g + 1]

                for idx in indices[start:end]:
                    # Purge: skip if too close to test set
                    if abs(idx - test_start) < self.purge_window:
                        continue
                    if abs(idx - test_end) < self.purge_window:
                        continue

                    # Embargo: skip if immediately after test set
                    if test_end < idx < test_end + self.embargo_window:
                        continue

                    train_indices.append(idx)

            path_info = {
                'path_idx': path_idx,
                'test_groups': list(test_groups),
                'train_groups': list(train_groups),
                'n_train': len(train_indices),
                'n_test': len(test_indices)
            }

            yield np.array(train_indices), np.array(test_indices), path_info


class MonteCarloStressTester:
    """
    Monte Carlo Stress Testing for Strategy Validation.

    Generates synthetic equity curves by:
    1. Shuffling trade sequence (preserving correlation structure)
    2. Calculating performance metrics for each permutation
    3. Estimating distribution of outcomes
    4. Computing probability metrics (VaR, CVaR, P(Loss))

    A strategy is considered robust if the original performance
    is NOT in the bottom 5% of random permutations.
    """

    def __init__(self, config: Optional[StressTestConfig] = None):
        self.config = config or StressTestConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        self.bootstrap = BlockBootstrap(
            block_size=self.config.block_size,
            method='moving',
            random_seed=self.config.random_seed
        )

        self.cpcv = CombinatorialPurgedCV(
            n_splits=self.config.n_splits,
            n_test_splits=self.config.n_test_splits,
            purge_window=self.config.purge_window,
            embargo_window=self.config.embargo_window
        )

    def load_trades(
        self,
        filepath: str
    ) -> pd.DataFrame:
        """Load trades from CSV file."""
        return pd.read_csv(filepath, parse_dates=['timestamp'])

    def load_backtest_results(
        self,
        filepath: str
    ) -> Dict:
        """Load backtest results from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def compute_returns_from_trades(
        self,
        trades: pd.DataFrame
    ) -> pd.Series:
        """
        Compute daily returns from trade-level data.

        Args:
            trades: DataFrame with columns ['timestamp', 'pnl', 'return']

        Returns:
            Series of daily returns
        """
        trades = trades.copy()
        trades['date'] = pd.to_datetime(trades['timestamp']).dt.date

        daily = trades.groupby('date').agg({
            'pnl': 'sum',
            'return': lambda x: (1 + x).prod() - 1 if 'return' in trades else x.sum()
        })

        return daily['return'] if 'return' in daily else daily['pnl']

    def bootstrap_equity_curves(
        self,
        returns: np.ndarray,
        n_simulations: int = None
    ) -> np.ndarray:
        """
        Generate bootstrapped equity curves.

        Args:
            returns: Original return series
            n_simulations: Number of simulations

        Returns:
            Array of shape (n_simulations, n_periods)
        """
        n_simulations = n_simulations or self.config.n_simulations

        curves = []
        for _ in range(n_simulations):
            resampled = self.bootstrap.resample(returns)
            equity = np.cumprod(1 + resampled)
            curves.append(equity)

        return np.array(curves)

    def calculate_metrics(
        self,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from returns.

        Args:
            returns: Return series

        Returns:
            Dictionary of metrics
        """
        returns = np.array(returns)

        # Handle edge cases
        if len(returns) < 2:
            return {
                'sharpe': 0,
                'return': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'skewness': 0,
                'kurtosis': 0
            }

        # Sharpe ratio (annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # Total return
        total_return = np.prod(1 + returns) - 1

        # Volatility (annualized)
        volatility = std_return * np.sqrt(252)

        # Max drawdown
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity / running_max - 1
        max_drawdown = abs(np.min(drawdowns))

        # Higher moments
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0

        return {
            'sharpe': sharpe,
            'return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def run_monte_carlo(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Monte Carlo simulation.

        Args:
            returns: Original return series

        Returns:
            Tuple of (sharpe_dist, return_dist, drawdown_dist)
        """
        n_simulations = self.config.n_simulations

        sharpes = []
        total_returns = []
        max_drawdowns = []

        for i in range(n_simulations):
            # Bootstrap resample
            resampled = self.bootstrap.resample(returns)

            # Calculate metrics
            metrics = self.calculate_metrics(resampled)

            sharpes.append(metrics['sharpe'])
            total_returns.append(metrics['return'])
            max_drawdowns.append(metrics['max_drawdown'])

            if (i + 1) % 100 == 0:
                logger.info(f"Monte Carlo: {i + 1}/{n_simulations} simulations complete")

        return np.array(sharpes), np.array(total_returns), np.array(max_drawdowns)

    def run_cpcv(
        self,
        returns: np.ndarray
    ) -> List[Dict]:
        """
        Run Combinatorial Purged Cross-Validation.

        Args:
            returns: Return series

        Returns:
            List of dictionaries with results for each path
        """
        returns = np.array(returns)
        n = len(returns)

        results = []

        for train_idx, test_idx, path_info in self.cpcv.split(returns):
            if len(test_idx) < 10:
                continue

            test_returns = returns[test_idx]
            metrics = self.calculate_metrics(test_returns)

            results.append({
                'path_idx': path_info['path_idx'],
                'n_train': path_info['n_train'],
                'n_test': path_info['n_test'],
                **metrics
            })

        logger.info(f"CPCV: {len(results)} paths evaluated")
        return results

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk.

        VaR is the loss threshold at the given confidence level.
        """
        return -np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        CVaR is the expected loss given that loss exceeds VaR.
        """
        var = self.calculate_var(returns, confidence)
        tail = returns[returns <= -var]
        return -np.mean(tail) if len(tail) > 0 else var

    def calculate_deflated_sharpe(
        self,
        returns: np.ndarray,
        n_trials: int = 50
    ) -> Tuple[float, float]:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            returns: Return series
            n_trials: Number of strategy variants tested

        Returns:
            Tuple of (DSR, Haircut SR)
        """
        try:
            from src.backtest.metrics import DeflatedSharpeRatio
            dsr_calc = DeflatedSharpeRatio()
            returns_series = pd.Series(returns)

            dsr = dsr_calc.deflated_sharpe_ratio(returns_series, n_trials)
            haircut = dsr_calc.haircut_sharpe_ratio(returns_series, n_trials)

            return dsr, haircut
        except Exception as e:
            logger.warning(f"Could not calculate DSR: {e}")
            return 0.5, 0

    def run_stress_test(
        self,
        returns: Union[np.ndarray, pd.Series],
        n_trials: int = 50
    ) -> StressTestResult:
        """
        Run complete stress test suite.

        Args:
            returns: Return series (daily)
            n_trials: Number of strategy variants tested (for DSR)

        Returns:
            StressTestResult with all metrics
        """
        returns = np.array(returns)

        logger.info(f"Starting stress test with {self.config.n_simulations} simulations")

        # Original metrics
        original_metrics = self.calculate_metrics(returns)
        logger.info(f"Original Sharpe: {original_metrics['sharpe']:.3f}")
        logger.info(f"Original Return: {original_metrics['return']:.2%}")
        logger.info(f"Original Max DD: {original_metrics['max_drawdown']:.2%}")

        # Monte Carlo simulation
        sharpe_dist, return_dist, dd_dist = self.run_monte_carlo(returns)

        # CPCV
        cpcv_results = self.run_cpcv(returns)
        cpcv_sharpes = [r['sharpe'] for r in cpcv_results]

        # Risk metrics
        var_95 = self.calculate_var(return_dist, 0.95)
        var_99 = self.calculate_var(return_dist, 0.99)
        cvar_95 = self.calculate_cvar(return_dist, 0.95)
        cvar_99 = self.calculate_cvar(return_dist, 0.99)

        # Probability metrics
        prob_loss = np.mean(return_dist < 0)
        prob_worse = np.mean(sharpe_dist < original_metrics['sharpe'])
        prob_dd_20 = np.mean(dd_dist > 0.20)

        # Percentiles
        sharpe_5 = np.percentile(sharpe_dist, 5)
        sharpe_95 = np.percentile(sharpe_dist, 95)
        return_5 = np.percentile(return_dist, 5)
        return_95 = np.percentile(return_dist, 95)

        # Deflated Sharpe Ratio
        dsr, haircut_sr = self.calculate_deflated_sharpe(returns, n_trials)

        # Validation
        validation = {
            'dsr_pass': dsr >= self.config.min_dsr,
            'prob_loss_pass': prob_loss <= self.config.max_prob_loss,
            'drawdown_pass': np.percentile(dd_dist, 95) <= self.config.max_drawdown_95,
            'not_bottom_5pct': prob_worse > 0.05
        }

        passed = all(validation.values())

        validation['details'] = {
            'dsr_threshold': self.config.min_dsr,
            'prob_loss_threshold': self.config.max_prob_loss,
            'drawdown_threshold': self.config.max_drawdown_95
        }

        logger.info(f"\nValidation Results:")
        logger.info(f"  DSR: {dsr:.3f} (threshold: {self.config.min_dsr}) - {'PASS' if validation['dsr_pass'] else 'FAIL'}")
        logger.info(f"  P(Loss): {prob_loss:.2%} (threshold: {self.config.max_prob_loss:.0%}) - {'PASS' if validation['prob_loss_pass'] else 'FAIL'}")
        logger.info(f"  95th DD: {np.percentile(dd_dist, 95):.2%} (threshold: {self.config.max_drawdown_95:.0%}) - {'PASS' if validation['drawdown_pass'] else 'FAIL'}")
        logger.info(f"  Not Bottom 5%: {prob_worse:.2%} > 5% - {'PASS' if validation['not_bottom_5pct'] else 'FAIL'}")
        logger.info(f"\nOverall: {'PASSED' if passed else 'FAILED'}")

        return StressTestResult(
            original_sharpe=original_metrics['sharpe'],
            original_return=original_metrics['return'],
            original_max_drawdown=original_metrics['max_drawdown'],
            sharpe_distribution=sharpe_dist,
            return_distribution=return_dist,
            max_drawdown_distribution=dd_dist,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            prob_loss=prob_loss,
            prob_worse_than_original=prob_worse,
            prob_drawdown_exceeds_20=prob_dd_20,
            sharpe_5th_percentile=sharpe_5,
            sharpe_95th_percentile=sharpe_95,
            return_5th_percentile=return_5,
            return_95th_percentile=return_95,
            passed_validation=passed,
            validation_details=validation,
            cpcv_sharpe_mean=np.mean(cpcv_sharpes) if cpcv_sharpes else 0,
            cpcv_sharpe_std=np.std(cpcv_sharpes) if cpcv_sharpes else 0,
            cpcv_paths=cpcv_results,
            deflated_sharpe_ratio=dsr,
            haircut_sharpe_ratio=haircut_sr,
            n_simulations=self.config.n_simulations
        )

    def generate_report(
        self,
        result: StressTestResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed stress test report.

        Args:
            result: StressTestResult from run_stress_test
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("MONTE CARLO STRESS TEST REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Simulations: {result.n_simulations:,}")
        report.append("")

        # Original Performance
        report.append("-" * 70)
        report.append("ORIGINAL STRATEGY PERFORMANCE")
        report.append("-" * 70)
        report.append(f"Sharpe Ratio:        {result.original_sharpe:>10.3f}")
        report.append(f"Total Return:        {result.original_return:>10.2%}")
        report.append(f"Max Drawdown:        {result.original_max_drawdown:>10.2%}")
        report.append("")

        # Monte Carlo Distribution
        report.append("-" * 70)
        report.append("MONTE CARLO DISTRIBUTION")
        report.append("-" * 70)
        report.append(f"{'Metric':<25} {'5th %ile':<12} {'Median':<12} {'95th %ile':<12}")
        report.append("-" * 70)
        report.append(
            f"{'Sharpe Ratio':<25} "
            f"{result.sharpe_5th_percentile:<12.3f} "
            f"{np.median(result.sharpe_distribution):<12.3f} "
            f"{result.sharpe_95th_percentile:<12.3f}"
        )
        report.append(
            f"{'Total Return':<25} "
            f"{result.return_5th_percentile:<12.2%} "
            f"{np.median(result.return_distribution):<12.2%} "
            f"{result.return_95th_percentile:<12.2%}"
        )
        report.append(
            f"{'Max Drawdown':<25} "
            f"{np.percentile(result.max_drawdown_distribution, 5):<12.2%} "
            f"{np.median(result.max_drawdown_distribution):<12.2%} "
            f"{np.percentile(result.max_drawdown_distribution, 95):<12.2%}"
        )
        report.append("")

        # Risk Metrics
        report.append("-" * 70)
        report.append("RISK METRICS")
        report.append("-" * 70)
        report.append(f"VaR 95%:             {result.var_95:>10.2%}")
        report.append(f"VaR 99%:             {result.var_99:>10.2%}")
        report.append(f"CVaR 95%:            {result.cvar_95:>10.2%}")
        report.append(f"CVaR 99%:            {result.cvar_99:>10.2%}")
        report.append("")

        # Probability Metrics
        report.append("-" * 70)
        report.append("PROBABILITY METRICS")
        report.append("-" * 70)
        report.append(f"P(Total Loss):       {result.prob_loss:>10.2%}")
        report.append(f"P(Worse than Orig.): {result.prob_worse_than_original:>10.2%}")
        report.append(f"P(DD > 20%):         {result.prob_drawdown_exceeds_20:>10.2%}")
        report.append("")

        # Deflated Sharpe Ratio
        report.append("-" * 70)
        report.append("OVERFITTING ANALYSIS")
        report.append("-" * 70)
        report.append(f"Deflated SR:         {result.deflated_sharpe_ratio:>10.3f}")
        report.append(f"Haircut SR:          {result.haircut_sharpe_ratio:>10.3f}")
        report.append(f"CPCV Mean SR:        {result.cpcv_sharpe_mean:>10.3f}")
        report.append(f"CPCV Std SR:         {result.cpcv_sharpe_std:>10.3f}")
        report.append("")

        # Validation
        report.append("-" * 70)
        report.append("VALIDATION RESULTS")
        report.append("-" * 70)

        for check, passed in result.validation_details.items():
            if check != 'details':
                status = "PASS" if passed else "FAIL"
                report.append(f"{check:<30} {status:>10}")

        report.append("")
        report.append("=" * 70)
        final_status = "STRATEGY PASSED" if result.passed_validation else "STRATEGY FAILED"
        report.append(f"{final_status:^70}")
        report.append("=" * 70)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

    def plot_distributions(
        self,
        result: StressTestResult,
        output_path: Optional[str] = None
    ) -> None:
        """
        Generate distribution plots.

        Args:
            result: StressTestResult
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sharpe distribution
        ax = axes[0, 0]
        ax.hist(result.sharpe_distribution, bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(result.original_sharpe, color='red', linestyle='--', linewidth=2,
                   label=f'Original: {result.original_sharpe:.2f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Density')
        ax.set_title('Sharpe Ratio Distribution')
        ax.legend()

        # Return distribution
        ax = axes[0, 1]
        ax.hist(result.return_distribution * 100, bins=50, density=True, alpha=0.7, color='forestgreen')
        ax.axvline(result.original_return * 100, color='red', linestyle='--', linewidth=2,
                   label=f'Original: {result.original_return:.1%}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution')
        ax.legend()

        # Max Drawdown distribution
        ax = axes[1, 0]
        ax.hist(result.max_drawdown_distribution * 100, bins=50, density=True, alpha=0.7, color='indianred')
        ax.axvline(result.original_max_drawdown * 100, color='red', linestyle='--', linewidth=2,
                   label=f'Original: {result.original_max_drawdown:.1%}')
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Density')
        ax.set_title('Max Drawdown Distribution')
        ax.legend()

        # CPCV Sharpes
        ax = axes[1, 1]
        if result.cpcv_paths:
            cpcv_sharpes = [p['sharpe'] for p in result.cpcv_paths]
            ax.bar(range(len(cpcv_sharpes)), cpcv_sharpes, alpha=0.7, color='purple')
            ax.axhline(result.original_sharpe, color='red', linestyle='--', linewidth=2,
                       label=f'Original: {result.original_sharpe:.2f}')
            ax.axhline(result.cpcv_sharpe_mean, color='green', linestyle=':', linewidth=2,
                       label=f'CPCV Mean: {result.cpcv_sharpe_mean:.2f}')
            ax.set_xlabel('CPCV Path')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('CPCV Path Analysis')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No CPCV data', ha='center', va='center')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main entry point for stress testing."""
    parser = argparse.ArgumentParser(
        description='Monte Carlo Stress Testing for Trading Strategies'
    )

    parser.add_argument(
        '--backtest-results',
        type=str,
        help='Path to backtest results pickle file'
    )
    parser.add_argument(
        '--trades-csv',
        type=str,
        help='Path to trades CSV file'
    )
    parser.add_argument(
        '--returns-csv',
        type=str,
        help='Path to daily returns CSV file'
    )
    parser.add_argument(
        '--n-simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./stress_test_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of strategy variants tested (for DSR calculation)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize stress tester
    config = StressTestConfig(
        n_simulations=args.n_simulations,
        random_seed=args.random_seed
    )
    tester = MonteCarloStressTester(config)

    # Load data
    if args.returns_csv:
        logger.info(f"Loading returns from {args.returns_csv}")
        df = pd.read_csv(args.returns_csv, parse_dates=['date'], index_col='date')
        returns = df['return'].values

    elif args.trades_csv:
        logger.info(f"Loading trades from {args.trades_csv}")
        trades = tester.load_trades(args.trades_csv)
        returns = tester.compute_returns_from_trades(trades).values

    elif args.backtest_results:
        logger.info(f"Loading backtest results from {args.backtest_results}")
        results = tester.load_backtest_results(args.backtest_results)
        if hasattr(results, 'daily_returns'):
            returns = results.daily_returns.values
        elif 'daily_returns' in results:
            returns = np.array(results['daily_returns'])
        else:
            raise ValueError("Could not extract returns from backtest results")

    else:
        # Demo with synthetic data
        logger.info("No data provided, running demo with synthetic returns")
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 500)  # ~500 trading days

    # Run stress test
    result = tester.run_stress_test(returns, n_trials=args.n_trials)

    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f'stress_test_report_{timestamp}.txt'
    report = tester.generate_report(result, str(report_path))
    print("\n" + report)

    # Generate plots
    plot_path = output_dir / f'stress_test_plots_{timestamp}.png'
    tester.plot_distributions(result, str(plot_path))

    # Save results
    results_path = output_dir / f'stress_test_results_{timestamp}.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Results saved to {results_path}")

    # Return exit code based on validation
    return 0 if result.passed_validation else 1


if __name__ == '__main__':
    sys.exit(main())
