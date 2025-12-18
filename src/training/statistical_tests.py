"""
Statistical Significance Tests Module.

JPMorgan Institutional-Level Statistical Validation for Trading Strategies.

Implements rigorous statistical tests to validate strategy performance:
1. Walk-Forward Analysis
2. Monte Carlo Simulation
3. Bootstrap Analysis
4. Multiple Hypothesis Testing Corrections
5. Deflated Sharpe Ratio
6. Probability of Backtest Overfitting (PBO)

Reference:
    "Advances in Financial Machine Learning" by de Prado (2018)
    Chapters 11-14: Backtesting through Monte Carlo

These tests are essential to distinguish between genuine alpha
and statistical noise/overfitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    details: Dict[str, Any]


class SharpeRatioTest:
    """
    Statistical tests for Sharpe Ratio significance.

    Tests whether the Sharpe Ratio is statistically different from:
    - Zero (strategy is profitable)
    - A benchmark Sharpe Ratio

    Includes correction for non-normality and autocorrelation.
    """

    @staticmethod
    def test_sharpe_significance(
        returns: pd.Series,
        benchmark_sharpe: float = 0.0,
        confidence_level: float = 0.95,
    ) -> StatisticalTestResult:
        """
        Test if Sharpe Ratio is significantly different from benchmark.

        Uses the Jobson-Korkie test statistic with Lo (2002) adjustment
        for autocorrelation.

        Args:
            returns: Strategy returns series
            benchmark_sharpe: Sharpe ratio to test against (default 0)
            confidence_level: Confidence level for significance

        Returns:
            StatisticalTestResult
        """
        returns = returns.dropna()
        n = len(returns)
        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret == 0:
            return StatisticalTestResult(
                test_name="sharpe_significance",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=confidence_level,
                details={"error": "zero_volatility"}
            )

        sharpe = mean_ret / std_ret * np.sqrt(252)

        # Standard error of Sharpe ratio (Lo, 2002)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # Adjust for non-normality
        se_sharpe = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4) * sharpe**2) / n
        ) * np.sqrt(252)

        # Adjust for autocorrelation
        autocorr = returns.autocorr()
        if not np.isnan(autocorr) and abs(autocorr) < 1:
            se_sharpe *= np.sqrt((1 + 2 * autocorr) / (1 - autocorr))

        # Test statistic
        t_stat = (sharpe - benchmark_sharpe) / se_sharpe
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

        is_significant = p_value < (1 - confidence_level)

        return StatisticalTestResult(
            test_name="sharpe_significance",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=confidence_level,
            details={
                "sharpe_ratio": sharpe,
                "benchmark_sharpe": benchmark_sharpe,
                "standard_error": se_sharpe,
                "n_observations": n,
                "skewness": skew,
                "kurtosis": kurt,
                "autocorrelation": autocorr,
            }
        )

    @staticmethod
    def deflated_sharpe_ratio(
        sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        expected_max_sharpe: Optional[float] = None,
    ) -> StatisticalTestResult:
        """
        Calculate the Deflated Sharpe Ratio (DSR).

        DSR adjusts the observed Sharpe Ratio for multiple testing,
        giving the probability that the Sharpe was achieved by luck.

        Reference:
            Bailey & Lopez de Prado (2014)
            "The Deflated Sharpe Ratio"

        Args:
            sharpe: Observed Sharpe Ratio
            n_trials: Number of independent backtests tried
            n_observations: Number of observations per backtest
            skewness: Skewness of returns
            kurtosis: Kurtosis of returns
            expected_max_sharpe: Expected maximum Sharpe under null

        Returns:
            StatisticalTestResult with DSR
        """
        # Expected maximum Sharpe under null hypothesis
        if expected_max_sharpe is None:
            # Approximate expected max of n_trials standard normals
            from scipy.special import erfinv
            expected_max_sharpe = (1 - np.euler_gamma) * stats.norm.ppf(
                1 - 1 / n_trials
            ) + np.euler_gamma * stats.norm.ppf(
                1 - 1 / (n_trials * np.e)
            )

        # Standard deviation of Sharpe estimator
        se_sharpe = np.sqrt(
            (1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2)
            / (n_observations - 1)
        )

        # PSR (Probabilistic Sharpe Ratio)
        psr = stats.norm.cdf((sharpe - expected_max_sharpe) / se_sharpe)

        # DSR is essentially 1 - probability of achieving sharpe by luck
        dsr = psr

        return StatisticalTestResult(
            test_name="deflated_sharpe_ratio",
            statistic=dsr,
            p_value=1 - dsr,
            is_significant=dsr > 0.95,
            confidence_level=0.95,
            details={
                "sharpe_ratio": sharpe,
                "expected_max_sharpe": expected_max_sharpe,
                "n_trials": n_trials,
                "n_observations": n_observations,
                "standard_error": se_sharpe,
            }
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy validation.

    Tests strategy robustness by simulating alternative
    market scenarios and checking performance consistency.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.n_jobs = n_jobs

    def permutation_test(
        self,
        returns: pd.Series,
        metric_func: Callable[[pd.Series], float] = None,
    ) -> StatisticalTestResult:
        """
        Perform permutation test on returns.

        Shuffles return ordering to test if strategy performance
        is significantly different from random.

        Args:
            returns: Strategy returns series
            metric_func: Function to calculate metric (default: Sharpe)

        Returns:
            StatisticalTestResult
        """
        if metric_func is None:
            metric_func = lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

        np.random.seed(self.random_state)
        returns_array = returns.dropna().values

        # Observed metric
        observed_metric = metric_func(returns)

        # Simulated metrics under null hypothesis
        def simulate_one(seed):
            np.random.seed(seed)
            shuffled = np.random.permutation(returns_array)
            return metric_func(pd.Series(shuffled))

        if JOBLIB_AVAILABLE and self.n_jobs != 1:
            simulated = Parallel(n_jobs=self.n_jobs)(
                delayed(simulate_one)(self.random_state + i)
                for i in range(self.n_simulations)
            )
        else:
            simulated = [
                simulate_one(self.random_state + i)
                for i in range(self.n_simulations)
            ]

        simulated = np.array(simulated)

        # P-value: fraction of simulations better than observed
        p_value = (simulated >= observed_metric).mean()

        return StatisticalTestResult(
            test_name="permutation_test",
            statistic=observed_metric,
            p_value=p_value,
            is_significant=p_value < 0.05,
            confidence_level=0.95,
            details={
                "observed_metric": observed_metric,
                "simulated_mean": simulated.mean(),
                "simulated_std": simulated.std(),
                "percentile_rank": (simulated < observed_metric).mean() * 100,
                "n_simulations": self.n_simulations,
            }
        )

    def bootstrap_confidence_interval(
        self,
        returns: pd.Series,
        metric_func: Callable[[pd.Series], float] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for a metric.

        Args:
            returns: Strategy returns series
            metric_func: Function to calculate metric
            confidence_level: Confidence level

        Returns:
            Dictionary with lower, upper bounds and point estimate
        """
        if metric_func is None:
            metric_func = lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

        np.random.seed(self.random_state)
        returns_array = returns.dropna().values
        n = len(returns_array)

        # Bootstrap samples
        def bootstrap_one(seed):
            np.random.seed(seed)
            sample = np.random.choice(returns_array, size=n, replace=True)
            return metric_func(pd.Series(sample))

        if JOBLIB_AVAILABLE and self.n_jobs != 1:
            bootstrap_metrics = Parallel(n_jobs=self.n_jobs)(
                delayed(bootstrap_one)(self.random_state + i)
                for i in range(self.n_simulations)
            )
        else:
            bootstrap_metrics = [
                bootstrap_one(self.random_state + i)
                for i in range(self.n_simulations)
            ]

        bootstrap_metrics = np.array(bootstrap_metrics)

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)

        return {
            "point_estimate": metric_func(returns),
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence_level": confidence_level,
            "bootstrap_std": bootstrap_metrics.std(),
        }


class MultipleTestingCorrection:
    """
    Multiple hypothesis testing corrections.

    When testing multiple strategies or parameters, we need to
    correct for the increased false positive rate.
    """

    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Bonferroni correction.

        Most conservative correction - divides alpha by number of tests.

        Args:
            p_values: List of p-values from individual tests
            alpha: Desired family-wise error rate

        Returns:
            Dictionary with corrected significance decisions
        """
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests

        return {
            "method": "bonferroni",
            "original_alpha": alpha,
            "corrected_alpha": corrected_alpha,
            "n_tests": n_tests,
            "is_significant": [p < corrected_alpha for p in p_values],
            "n_significant": sum(p < corrected_alpha for p in p_values),
        }

    @staticmethod
    def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Holm-Bonferroni step-down correction.

        Less conservative than Bonferroni while still controlling FWER.

        Args:
            p_values: List of p-values
            alpha: Desired FWER

        Returns:
            Dictionary with corrected significance decisions
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = np.array(p_values)[sorted_indices]

        is_significant = [False] * n_tests

        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_pvals)):
            corrected_alpha = alpha / (n_tests - i)
            if p <= corrected_alpha:
                is_significant[idx] = True
            else:
                break  # All subsequent are not significant

        return {
            "method": "holm_bonferroni",
            "original_alpha": alpha,
            "n_tests": n_tests,
            "is_significant": is_significant,
            "n_significant": sum(is_significant),
        }

    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Benjamini-Hochberg procedure.

        Controls False Discovery Rate (FDR) - less conservative than FWER methods.

        Args:
            p_values: List of p-values
            alpha: Desired FDR

        Returns:
            Dictionary with corrected significance decisions
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = np.array(p_values)[sorted_indices]

        # Find largest k where p(k) <= k*alpha/n
        is_significant = [False] * n_tests
        threshold_idx = -1

        for k, (idx, p) in enumerate(zip(sorted_indices, sorted_pvals), 1):
            threshold = k * alpha / n_tests
            if p <= threshold:
                threshold_idx = k - 1

        # All p-values up to and including threshold_idx are significant
        if threshold_idx >= 0:
            for i in range(threshold_idx + 1):
                is_significant[sorted_indices[i]] = True

        return {
            "method": "benjamini_hochberg",
            "original_alpha": alpha,
            "n_tests": n_tests,
            "is_significant": is_significant,
            "n_significant": sum(is_significant),
            "fdr_control": alpha,
        }


class ProbabilityOfBacktestOverfitting:
    """
    Probability of Backtest Overfitting (PBO).

    Estimates the probability that a backtest's performance is
    due to overfitting rather than genuine predictive ability.

    Reference:
        Bailey et al. (2016)
        "Probability of Backtest Overfitting"
    """

    def __init__(
        self,
        n_partitions: int = 16,
        random_state: int = 42,
    ):
        """
        Initialize PBO calculator.

        Args:
            n_partitions: Number of partitions for CSCV
            random_state: Random seed
        """
        self.n_partitions = n_partitions
        self.random_state = random_state

    def calculate_pbo(
        self,
        returns_matrix: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate Probability of Backtest Overfitting.

        Uses Combinatorially Symmetric Cross-Validation (CSCV)
        to estimate overfitting probability.

        Args:
            returns_matrix: DataFrame where each column is a strategy
                           and each row is a time period

        Returns:
            Dictionary with PBO and related statistics
        """
        n_strategies = returns_matrix.shape[1]
        n_periods = returns_matrix.shape[0]

        # Partition data
        partition_size = n_periods // self.n_partitions
        partitions = [
            returns_matrix.iloc[i*partition_size:(i+1)*partition_size]
            for i in range(self.n_partitions)
        ]

        # CSCV: for each way of splitting into train/test
        n_combinations = comb(self.n_partitions, self.n_partitions // 2, exact=True)
        max_combinations = min(n_combinations, 1000)  # Limit for computational reasons

        np.random.seed(self.random_state)

        # Track rank correlations
        rank_correlations = []
        is_overfit = []

        from itertools import combinations
        all_combos = list(combinations(range(self.n_partitions), self.n_partitions // 2))
        selected_combos = np.random.choice(
            len(all_combos), size=min(len(all_combos), max_combinations), replace=False
        )

        for combo_idx in selected_combos:
            train_indices = all_combos[combo_idx]
            test_indices = [i for i in range(self.n_partitions) if i not in train_indices]

            # Combine partitions
            train_data = pd.concat([partitions[i] for i in train_indices])
            test_data = pd.concat([partitions[i] for i in test_indices])

            # Calculate Sharpe for each strategy in train and test
            train_sharpes = train_data.mean() / train_data.std() * np.sqrt(252)
            test_sharpes = test_data.mean() / test_data.std() * np.sqrt(252)

            # Find best strategy in-sample
            best_is = train_sharpes.idxmax()

            # Check if best IS strategy is also best OOS
            best_oos = test_sharpes.idxmax()
            is_overfit.append(best_is != best_oos)

            # Rank correlation
            if len(train_sharpes) > 1:
                rho, _ = stats.spearmanr(train_sharpes, test_sharpes)
                if not np.isnan(rho):
                    rank_correlations.append(rho)

        # PBO is fraction of times IS winner underperforms OOS
        # More precisely: P(IS rank < OOS rank for the best IS strategy)

        # Calculate relative rank of IS-best strategy in OOS
        pbo = np.mean(is_overfit)

        return {
            "pbo": pbo,
            "mean_rank_correlation": np.mean(rank_correlations) if rank_correlations else 0,
            "std_rank_correlation": np.std(rank_correlations) if rank_correlations else 0,
            "n_combinations_tested": len(selected_combos),
            "n_strategies": n_strategies,
            "interpretation": self._interpret_pbo(pbo),
        }

    def _interpret_pbo(self, pbo: float) -> str:
        """Interpret PBO value."""
        if pbo < 0.1:
            return "Very low overfitting risk"
        elif pbo < 0.3:
            return "Low overfitting risk"
        elif pbo < 0.5:
            return "Moderate overfitting risk"
        elif pbo < 0.7:
            return "High overfitting risk"
        else:
            return "Very high overfitting risk - likely overfit"


class StrategyStatisticalValidator:
    """
    Comprehensive statistical validation for trading strategies.

    Combines multiple statistical tests to provide thorough
    validation of strategy performance.

    Example:
        validator = StrategyStatisticalValidator()
        results = validator.validate(returns)
        print(results.summary())
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_simulations: int = 1000,
        random_state: int = 42,
    ):
        """
        Initialize validator.

        Args:
            confidence_level: Confidence level for tests
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed
        """
        self.confidence_level = confidence_level
        self.n_simulations = n_simulations
        self.random_state = random_state

        self.mc_simulator = MonteCarloSimulator(
            n_simulations=n_simulations,
            random_state=random_state,
        )

    def validate(
        self,
        returns: pd.Series,
        n_trials: int = 1,
        benchmark_sharpe: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation.

        Args:
            returns: Strategy returns series
            n_trials: Number of strategies/configs tried
            benchmark_sharpe: Benchmark Sharpe to test against

        Returns:
            Dictionary with all test results
        """
        results = {}

        # 1. Sharpe Ratio Significance Test
        sharpe_test = SharpeRatioTest.test_sharpe_significance(
            returns,
            benchmark_sharpe=benchmark_sharpe,
            confidence_level=self.confidence_level,
        )
        results["sharpe_significance"] = sharpe_test

        # 2. Deflated Sharpe Ratio (if multiple trials)
        if n_trials > 1:
            returns_clean = returns.dropna()
            sharpe = returns_clean.mean() / returns_clean.std() * np.sqrt(252)
            dsr_test = SharpeRatioTest.deflated_sharpe_ratio(
                sharpe=sharpe,
                n_trials=n_trials,
                n_observations=len(returns_clean),
                skewness=stats.skew(returns_clean),
                kurtosis=stats.kurtosis(returns_clean),
            )
            results["deflated_sharpe_ratio"] = dsr_test

        # 3. Permutation Test
        perm_test = self.mc_simulator.permutation_test(returns)
        results["permutation_test"] = perm_test

        # 4. Bootstrap Confidence Interval
        bootstrap_ci = self.mc_simulator.bootstrap_confidence_interval(
            returns,
            confidence_level=self.confidence_level,
        )
        results["bootstrap_confidence_interval"] = bootstrap_ci

        # 5. Summary statistics
        results["summary"] = self._calculate_summary(returns)

        return results

    def _calculate_summary(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate summary statistics."""
        returns_clean = returns.dropna()

        if len(returns_clean) == 0 or returns_clean.std() == 0:
            return {"error": "insufficient_data"}

        return {
            "mean_return": returns_clean.mean() * 252,
            "volatility": returns_clean.std() * np.sqrt(252),
            "sharpe_ratio": returns_clean.mean() / returns_clean.std() * np.sqrt(252),
            "skewness": stats.skew(returns_clean),
            "kurtosis": stats.kurtosis(returns_clean),
            "max_drawdown": self._calculate_max_drawdown(returns_clean),
            "n_observations": len(returns_clean),
            "positive_days_pct": (returns_clean > 0).mean(),
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def validate_multiple_strategies(
        self,
        returns_dict: Dict[str, pd.Series],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Validate multiple strategies with multiple testing correction.

        Args:
            returns_dict: Dictionary of strategy_name -> returns series
            alpha: Desired significance level

        Returns:
            Dictionary with individual and corrected results
        """
        individual_results = {}
        p_values = []

        for name, returns in returns_dict.items():
            result = self.validate(returns)
            individual_results[name] = result

            # Collect p-value from sharpe test
            sharpe_pval = result["sharpe_significance"].p_value
            p_values.append(sharpe_pval)

        # Apply multiple testing corrections
        corrections = {
            "bonferroni": MultipleTestingCorrection.bonferroni(p_values, alpha),
            "holm_bonferroni": MultipleTestingCorrection.holm_bonferroni(p_values, alpha),
            "benjamini_hochberg": MultipleTestingCorrection.benjamini_hochberg(p_values, alpha),
        }

        # Add corrected significance to individual results
        strategy_names = list(returns_dict.keys())
        for i, name in enumerate(strategy_names):
            individual_results[name]["multiple_testing"] = {
                method: corrections[method]["is_significant"][i]
                for method in corrections
            }

        return {
            "individual_results": individual_results,
            "multiple_testing_corrections": corrections,
            "strategy_ranking": self._rank_strategies(returns_dict),
        }

    def _rank_strategies(self, returns_dict: Dict[str, pd.Series]) -> List[Tuple[str, float]]:
        """Rank strategies by Sharpe Ratio."""
        sharpes = []
        for name, returns in returns_dict.items():
            returns_clean = returns.dropna()
            if len(returns_clean) > 0 and returns_clean.std() > 0:
                sharpe = returns_clean.mean() / returns_clean.std() * np.sqrt(252)
                sharpes.append((name, sharpe))

        return sorted(sharpes, key=lambda x: x[1], reverse=True)
