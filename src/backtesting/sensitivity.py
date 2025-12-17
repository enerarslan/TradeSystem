"""
Transaction Cost Sensitivity Analysis Module.

This module provides institutional-grade analysis of how trading costs
affect strategy performance, critical for understanding strategy robustness.

JPMorgan-level requirements:
- Analyze impact of varying commission rates
- Analyze impact of varying slippage
- Market impact sensitivity
- Combined cost scenarios
- Break-even cost analysis

Reference:
    - Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


@dataclass
class CostScenario:
    """A single cost scenario configuration."""
    name: str
    commission_pct: float
    slippage_pct: float
    market_impact_factor: float = 1.0


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis."""
    scenario: CostScenario
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_costs: float
    cost_as_pct_return: float
    n_trades: int


@dataclass
class BreakEvenAnalysis:
    """Break-even cost analysis results."""
    break_even_commission: float
    break_even_slippage: float
    break_even_total_cost: float
    current_margin: float  # How much room before break-even
    strategy_gross_return: float


class TransactionCostSensitivity:
    """
    Analyzes strategy sensitivity to transaction costs.

    This is critical for:
    1. Understanding if a strategy survives realistic costs
    2. Finding the break-even cost level
    3. Assessing strategy robustness
    """

    def __init__(
        self,
        backtest_engine,
        strategy,
        data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        base_config: Dict[str, Any],
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            backtest_engine: BacktestEngine class (not instance)
            strategy: Strategy instance to test
            data: Market data
            features: Feature data
            base_config: Base configuration
        """
        self.engine_class = backtest_engine
        self.strategy = strategy
        self.data = data
        self.features = features
        self.base_config = base_config

    def _run_scenario(self, scenario: CostScenario) -> SensitivityResult:
        """Run a single cost scenario."""
        # Create engine with scenario costs
        engine = self.engine_class(
            initial_capital=self.base_config.get("initial_capital", 1_000_000),
            commission_pct=scenario.commission_pct,
            slippage_pct=scenario.slippage_pct,
        )

        # Run backtest
        result = engine.run(
            self.strategy,
            self.data,
            self.features,
        )

        # Calculate cost percentage
        total_return = result.metrics.get("total_return", 0)
        total_costs = result.metrics.get("total_costs", 0)

        cost_pct = abs(total_costs / result.final_capital) if result.final_capital > 0 else 0

        return SensitivityResult(
            scenario=scenario,
            total_return=total_return,
            sharpe_ratio=result.metrics.get("sharpe_ratio", 0),
            max_drawdown=result.metrics.get("max_drawdown", 0),
            total_costs=total_costs,
            cost_as_pct_return=cost_pct,
            n_trades=result.metrics.get("n_trades", 0),
        )

    def analyze_commission_sensitivity(
        self,
        commission_range: Optional[List[float]] = None,
        n_points: int = 10,
    ) -> List[SensitivityResult]:
        """
        Analyze sensitivity to commission rates.

        Args:
            commission_range: Min/max commission percentages
            n_points: Number of points to test

        Returns:
            List of sensitivity results
        """
        if commission_range is None:
            commission_range = [0.0001, 0.005]  # 1 bps to 50 bps

        commissions = np.linspace(commission_range[0], commission_range[1], n_points)
        base_slippage = self.base_config.get("slippage_pct", 0.0005)

        results = []
        for comm in commissions:
            scenario = CostScenario(
                name=f"commission_{comm*10000:.1f}bps",
                commission_pct=comm,
                slippage_pct=base_slippage,
            )
            try:
                result = self._run_scenario(scenario)
                results.append(result)
                logger.debug(f"Commission {comm*10000:.1f}bps: Sharpe={result.sharpe_ratio:.2f}")
            except Exception as e:
                logger.warning(f"Scenario failed: {e}")

        return results

    def analyze_slippage_sensitivity(
        self,
        slippage_range: Optional[List[float]] = None,
        n_points: int = 10,
    ) -> List[SensitivityResult]:
        """
        Analyze sensitivity to slippage.

        Args:
            slippage_range: Min/max slippage percentages
            n_points: Number of points to test

        Returns:
            List of sensitivity results
        """
        if slippage_range is None:
            slippage_range = [0.0, 0.003]  # 0 to 30 bps

        slippages = np.linspace(slippage_range[0], slippage_range[1], n_points)
        base_commission = self.base_config.get("commission_pct", 0.001)

        results = []
        for slip in slippages:
            scenario = CostScenario(
                name=f"slippage_{slip*10000:.1f}bps",
                commission_pct=base_commission,
                slippage_pct=slip,
            )
            try:
                result = self._run_scenario(scenario)
                results.append(result)
                logger.debug(f"Slippage {slip*10000:.1f}bps: Sharpe={result.sharpe_ratio:.2f}")
            except Exception as e:
                logger.warning(f"Scenario failed: {e}")

        return results

    def analyze_combined_sensitivity(
        self,
        commission_range: Optional[List[float]] = None,
        slippage_range: Optional[List[float]] = None,
        n_points: int = 5,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to combined commission and slippage.

        Creates a grid of cost scenarios.

        Returns:
            DataFrame with Sharpe ratios for each combination
        """
        if commission_range is None:
            commission_range = [0.0001, 0.003]
        if slippage_range is None:
            slippage_range = [0.0, 0.002]

        commissions = np.linspace(commission_range[0], commission_range[1], n_points)
        slippages = np.linspace(slippage_range[0], slippage_range[1], n_points)

        # Create result grid
        sharpe_grid = np.zeros((n_points, n_points))

        for i, comm in enumerate(commissions):
            for j, slip in enumerate(slippages):
                scenario = CostScenario(
                    name=f"c{comm*10000:.0f}_s{slip*10000:.0f}",
                    commission_pct=comm,
                    slippage_pct=slip,
                )
                try:
                    result = self._run_scenario(scenario)
                    sharpe_grid[i, j] = result.sharpe_ratio
                except Exception:
                    sharpe_grid[i, j] = np.nan

        # Create DataFrame
        df = pd.DataFrame(
            sharpe_grid,
            index=[f"{c*10000:.1f}bps" for c in commissions],
            columns=[f"{s*10000:.1f}bps" for s in slippages],
        )
        df.index.name = "Commission"
        df.columns.name = "Slippage"

        return df

    def find_break_even_costs(
        self,
        tolerance: float = 0.01,
        max_iterations: int = 20,
    ) -> BreakEvenAnalysis:
        """
        Find the break-even cost level where strategy return goes to zero.

        Uses binary search to find the cost level.

        Returns:
            BreakEvenAnalysis with break-even costs
        """
        # First, run with zero costs to get gross return
        zero_scenario = CostScenario(
            name="zero_cost",
            commission_pct=0,
            slippage_pct=0,
        )
        zero_result = self._run_scenario(zero_scenario)
        gross_return = zero_result.total_return

        if gross_return <= 0:
            logger.warning("Strategy has negative gross return - no break-even point")
            return BreakEvenAnalysis(
                break_even_commission=0,
                break_even_slippage=0,
                break_even_total_cost=0,
                current_margin=0,
                strategy_gross_return=gross_return,
            )

        # Binary search for break-even commission
        low, high = 0.0, 0.05  # 0 to 500 bps
        break_even_comm = high

        for _ in range(max_iterations):
            mid = (low + high) / 2
            scenario = CostScenario(
                name="break_even_search",
                commission_pct=mid,
                slippage_pct=0,
            )
            result = self._run_scenario(scenario)

            if result.total_return > tolerance:
                low = mid
            else:
                high = mid
                break_even_comm = mid

            if high - low < 0.0001:  # 1 bps precision
                break

        # Calculate break-even slippage (with base commission)
        base_comm = self.base_config.get("commission_pct", 0.001)
        low, high = 0.0, 0.05
        break_even_slip = high

        for _ in range(max_iterations):
            mid = (low + high) / 2
            scenario = CostScenario(
                name="break_even_search",
                commission_pct=base_comm,
                slippage_pct=mid,
            )
            result = self._run_scenario(scenario)

            if result.total_return > tolerance:
                low = mid
            else:
                high = mid
                break_even_slip = mid

            if high - low < 0.0001:
                break

        # Calculate current margin
        current_comm = self.base_config.get("commission_pct", 0.001)
        current_slip = self.base_config.get("slippage_pct", 0.0005)

        current_scenario = CostScenario(
            name="current",
            commission_pct=current_comm,
            slippage_pct=current_slip,
        )
        current_result = self._run_scenario(current_scenario)
        current_margin = current_result.total_return / gross_return if gross_return > 0 else 0

        return BreakEvenAnalysis(
            break_even_commission=break_even_comm,
            break_even_slippage=break_even_slip,
            break_even_total_cost=break_even_comm + break_even_slip,
            current_margin=current_margin,
            strategy_gross_return=gross_return,
        )

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cost sensitivity report.

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Running transaction cost sensitivity analysis...")

        # Run all analyses
        commission_results = self.analyze_commission_sensitivity()
        slippage_results = self.analyze_slippage_sensitivity()
        break_even = self.find_break_even_costs()

        # Create summary
        report = {
            "commission_sensitivity": {
                "scenarios": len(commission_results),
                "min_sharpe": min(r.sharpe_ratio for r in commission_results),
                "max_sharpe": max(r.sharpe_ratio for r in commission_results),
                "sharpe_at_1bps": next((r.sharpe_ratio for r in commission_results
                                        if r.scenario.commission_pct < 0.0002), None),
                "sharpe_at_10bps": next((r.sharpe_ratio for r in commission_results
                                         if 0.0009 < r.scenario.commission_pct < 0.0011), None),
            },
            "slippage_sensitivity": {
                "scenarios": len(slippage_results),
                "min_sharpe": min(r.sharpe_ratio for r in slippage_results),
                "max_sharpe": max(r.sharpe_ratio for r in slippage_results),
            },
            "break_even_analysis": {
                "break_even_commission_bps": break_even.break_even_commission * 10000,
                "break_even_slippage_bps": break_even.break_even_slippage * 10000,
                "current_margin_pct": break_even.current_margin * 100,
                "gross_return_pct": break_even.strategy_gross_return * 100,
            },
            "robustness_assessment": self._assess_robustness(
                commission_results, slippage_results, break_even
            ),
        }

        return report

    def _assess_robustness(
        self,
        commission_results: List[SensitivityResult],
        slippage_results: List[SensitivityResult],
        break_even: BreakEvenAnalysis,
    ) -> Dict[str, Any]:
        """Assess overall cost robustness."""
        # Check if strategy survives realistic costs
        realistic_comm = 0.001  # 10 bps
        realistic_slip = 0.0005  # 5 bps

        sharpe_at_realistic = None
        for r in commission_results:
            if abs(r.scenario.commission_pct - realistic_comm) < 0.0001:
                sharpe_at_realistic = r.sharpe_ratio
                break

        # Calculate cost elasticity
        if len(commission_results) >= 2:
            low_cost_sharpe = commission_results[0].sharpe_ratio
            high_cost_sharpe = commission_results[-1].sharpe_ratio
            cost_range = (commission_results[-1].scenario.commission_pct -
                         commission_results[0].scenario.commission_pct)
            elasticity = (high_cost_sharpe - low_cost_sharpe) / cost_range if cost_range > 0 else 0
        else:
            elasticity = 0

        # Robustness score (0-100)
        score = 0
        if break_even.current_margin > 0.5:  # >50% margin to break-even
            score += 40
        elif break_even.current_margin > 0.2:
            score += 20

        if sharpe_at_realistic and sharpe_at_realistic > 1.0:
            score += 30
        elif sharpe_at_realistic and sharpe_at_realistic > 0.5:
            score += 15

        if break_even.break_even_commission > 0.003:  # >30 bps break-even
            score += 30
        elif break_even.break_even_commission > 0.001:
            score += 15

        return {
            "robustness_score": score,
            "survives_realistic_costs": sharpe_at_realistic is not None and sharpe_at_realistic > 0,
            "sharpe_at_realistic_costs": sharpe_at_realistic,
            "cost_elasticity": elasticity,
            "assessment": (
                "ROBUST" if score >= 70 else
                "MODERATE" if score >= 40 else
                "FRAGILE"
            ),
        }

    def plot_sensitivity(
        self,
        commission_results: List[SensitivityResult],
        slippage_results: List[SensitivityResult],
    ) -> Optional[str]:
        """Generate sensitivity plots."""
        if not HAS_PLOTLY:
            logger.warning("Plotly not available for plotting")
            return None

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Commission Sensitivity", "Slippage Sensitivity")
        )

        # Commission sensitivity
        comms = [r.scenario.commission_pct * 10000 for r in commission_results]
        sharpes_comm = [r.sharpe_ratio for r in commission_results]

        fig.add_trace(
            go.Scatter(x=comms, y=sharpes_comm, mode="lines+markers", name="Sharpe"),
            row=1, col=1
        )

        # Slippage sensitivity
        slips = [r.scenario.slippage_pct * 10000 for r in slippage_results]
        sharpes_slip = [r.sharpe_ratio for r in slippage_results]

        fig.add_trace(
            go.Scatter(x=slips, y=sharpes_slip, mode="lines+markers", name="Sharpe"),
            row=1, col=2
        )

        fig.update_layout(
            title="Transaction Cost Sensitivity Analysis",
            height=400,
            template="plotly_dark",
        )
        fig.update_xaxes(title_text="Commission (bps)", row=1, col=1)
        fig.update_xaxes(title_text="Slippage (bps)", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)

        return fig.to_html(full_html=False, include_plotlyjs=False)


def run_cost_sensitivity_analysis(
    engine_class,
    strategy,
    data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience function to run cost sensitivity analysis.

    Args:
        engine_class: BacktestEngine class
        strategy: Strategy to test
        data: Market data
        features: Features
        config: Configuration

    Returns:
        Sensitivity analysis report
    """
    analyzer = TransactionCostSensitivity(
        backtest_engine=engine_class,
        strategy=strategy,
        data=data,
        features=features,
        base_config=config,
    )

    return analyzer.generate_report()
