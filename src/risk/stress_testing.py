"""
Stress Testing Module.

This module provides institutional-grade stress testing capabilities:
- Historical scenario analysis
- Hypothetical stress scenarios
- Reverse stress testing
- Sensitivity analysis

JPMorgan-level requirements:
- 2008 Financial Crisis scenario
- COVID-19 March 2020 scenario
- Custom scenario definition
- Tail risk analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of stress scenarios."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    REVERSE = "reverse"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    scenario_type: ScenarioType
    description: str
    shocks: Dict[str, float]  # Asset/factor -> shock percentage
    duration_days: int = 1
    correlation_adjustment: float = 1.0  # Correlation increase during stress


@dataclass
class ScenarioResult:
    """Results from running a stress scenario."""
    scenario_name: str
    portfolio_impact_pct: float
    var_exceedance: float  # How many VaR multiples
    expected_shortfall: float
    max_drawdown: float
    recovery_time_days: Optional[int] = None
    positions_at_risk: List[str] = field(default_factory=list)


@dataclass
class StressTestReport:
    """Complete stress test report."""
    timestamp: str
    portfolio_value: float
    results: List[ScenarioResult]
    worst_case_scenario: str
    worst_case_loss_pct: float
    tail_risk_summary: Dict[str, float]
    recommendations: List[str]


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="Lehman Brothers collapse and global financial crisis",
        shocks={
            "equity": -0.55,  # S&P 500 peak-to-trough
            "financials": -0.80,
            "credit_spread": 0.06,  # 600bps widening
            "volatility": 3.0,  # VIX tripled
            "treasury": 0.02,  # Flight to safety
        },
        duration_days=252,  # About 1 year of decline
        correlation_adjustment=1.5,
    ),
    "2020_covid_crash": StressScenario(
        name="COVID-19 March 2020",
        scenario_type=ScenarioType.HISTORICAL,
        description="Fastest bear market in history",
        shocks={
            "equity": -0.34,  # 34% decline in ~30 days
            "energy": -0.65,
            "travel": -0.70,
            "volatility": 4.0,  # VIX to 80+
            "credit_spread": 0.04,
        },
        duration_days=23,  # Very fast
        correlation_adjustment=1.8,
    ),
    "2000_dotcom_bust": StressScenario(
        name="Dot-Com Bust 2000-2002",
        scenario_type=ScenarioType.HISTORICAL,
        description="Technology bubble burst",
        shocks={
            "equity": -0.49,
            "technology": -0.78,
            "nasdaq": -0.78,
            "volatility": 2.0,
        },
        duration_days=756,  # ~3 years
        correlation_adjustment=1.2,
    ),
    "2011_euro_crisis": StressScenario(
        name="European Debt Crisis 2011",
        scenario_type=ScenarioType.HISTORICAL,
        description="European sovereign debt crisis",
        shocks={
            "equity": -0.19,
            "financials": -0.30,
            "euro": -0.10,
            "credit_spread": 0.03,
        },
        duration_days=120,
        correlation_adjustment=1.3,
    ),
    "flash_crash_2010": StressScenario(
        name="Flash Crash May 2010",
        scenario_type=ScenarioType.HISTORICAL,
        description="Intraday flash crash",
        shocks={
            "equity": -0.09,  # Quick recovery but shows vulnerability
            "volatility": 2.5,
        },
        duration_days=1,
        correlation_adjustment=2.0,
    ),
}

# Hypothetical stress scenarios
HYPOTHETICAL_SCENARIOS = {
    "severe_recession": StressScenario(
        name="Severe Recession",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Severe economic recession scenario",
        shocks={
            "equity": -0.40,
            "credit_spread": 0.05,
            "real_estate": -0.25,
            "volatility": 2.5,
        },
        duration_days=252,
        correlation_adjustment=1.6,
    ),
    "rate_spike": StressScenario(
        name="Interest Rate Spike",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Sudden 200bps rate increase",
        shocks={
            "treasury": -0.10,  # Bond prices fall
            "equity": -0.15,
            "utilities": -0.25,
            "real_estate": -0.20,
            "financials": 0.05,  # Banks benefit
        },
        duration_days=60,
        correlation_adjustment=1.3,
    ),
    "inflation_surge": StressScenario(
        name="Inflation Surge",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Inflation spikes to double digits",
        shocks={
            "equity": -0.20,
            "treasury": -0.15,
            "gold": 0.15,
            "energy": 0.30,
            "utilities": -0.15,
        },
        duration_days=180,
        correlation_adjustment=1.2,
    ),
    "liquidity_crisis": StressScenario(
        name="Liquidity Crisis",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Market-wide liquidity seizure",
        shocks={
            "equity": -0.25,
            "small_cap": -0.40,
            "high_yield": -0.15,
            "volatility": 3.0,
        },
        duration_days=30,
        correlation_adjustment=2.0,
    ),
}


class StressTester:
    """
    Performs stress testing on portfolios.

    Capabilities:
    - Historical scenario replay
    - Hypothetical scenario simulation
    - Reverse stress testing
    - Portfolio sensitivity analysis
    """

    def __init__(
        self,
        portfolio_value: float,
        position_values: Dict[str, float],
        position_betas: Optional[Dict[str, float]] = None,
        position_sectors: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize stress tester.

        Args:
            portfolio_value: Total portfolio value
            position_values: Position values by symbol
            position_betas: Market betas by symbol
            position_sectors: Sector classifications
        """
        self.portfolio_value = portfolio_value
        self.positions = position_values
        self.betas = position_betas or {s: 1.0 for s in position_values}
        self.sectors = position_sectors or {}

        # Infer sector exposures
        self.sector_exposure = self._calculate_sector_exposure()

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate portfolio exposure by sector."""
        exposure = {}
        for symbol, value in self.positions.items():
            sector = self.sectors.get(symbol, "other")
            exposure[sector] = exposure.get(sector, 0) + value / self.portfolio_value
        return exposure

    def run_scenario(
        self,
        scenario: StressScenario,
    ) -> ScenarioResult:
        """
        Run a single stress scenario.

        Args:
            scenario: Stress scenario to run

        Returns:
            Scenario results
        """
        # Calculate portfolio impact
        total_impact = 0.0
        positions_at_risk = []

        for symbol, value in self.positions.items():
            # Get beta
            beta = self.betas.get(symbol, 1.0)
            sector = self.sectors.get(symbol, "other")

            # Apply shocks
            position_impact = 0.0

            # Market-wide shock
            if "equity" in scenario.shocks:
                position_impact += beta * scenario.shocks["equity"]

            # Sector-specific shock
            if sector.lower() in scenario.shocks:
                position_impact += scenario.shocks[sector.lower()]

            # Symbol-specific shock (if defined)
            if symbol in scenario.shocks:
                position_impact = scenario.shocks[symbol]  # Override

            # Weight by position size
            weight = value / self.portfolio_value
            total_impact += weight * position_impact

            # Track positions with >20% loss
            if position_impact < -0.20:
                positions_at_risk.append(symbol)

        # Apply correlation adjustment (makes things worse in stress)
        total_impact *= scenario.correlation_adjustment

        # Cap at -100%
        total_impact = max(-1.0, total_impact)

        # Calculate VaR exceedance (assuming 1-day 95% VaR of ~2%)
        daily_var = 0.02
        var_exceedance = abs(total_impact) / daily_var if daily_var > 0 else 0

        return ScenarioResult(
            scenario_name=scenario.name,
            portfolio_impact_pct=total_impact,
            var_exceedance=var_exceedance,
            expected_shortfall=total_impact * 1.2,  # Approximate ES
            max_drawdown=abs(total_impact),
            recovery_time_days=scenario.duration_days,
            positions_at_risk=positions_at_risk,
        )

    def run_all_scenarios(
        self,
        include_historical: bool = True,
        include_hypothetical: bool = True,
        custom_scenarios: Optional[List[StressScenario]] = None,
    ) -> List[ScenarioResult]:
        """
        Run all relevant stress scenarios.

        Returns:
            List of scenario results
        """
        scenarios = []

        if include_historical:
            scenarios.extend(HISTORICAL_SCENARIOS.values())

        if include_hypothetical:
            scenarios.extend(HYPOTHETICAL_SCENARIOS.values())

        if custom_scenarios:
            scenarios.extend(custom_scenarios)

        results = []
        for scenario in scenarios:
            try:
                result = self.run_scenario(scenario)
                results.append(result)
                logger.debug(
                    f"{scenario.name}: {result.portfolio_impact_pct:.1%} impact"
                )
            except Exception as e:
                logger.error(f"Scenario {scenario.name} failed: {e}")

        return results

    def find_breaking_point(
        self,
        max_acceptable_loss: float = -0.20,
    ) -> Dict[str, float]:
        """
        Reverse stress test: find what scenarios cause unacceptable losses.

        Args:
            max_acceptable_loss: Maximum acceptable loss (e.g., -20%)

        Returns:
            Dict of factor -> shock level that causes breaking point
        """
        factors = ["equity", "volatility", "credit_spread", "treasury"]
        breaking_points = {}

        for factor in factors:
            # Binary search for breaking point
            low, high = 0.0, -1.0 if factor != "volatility" else 5.0

            for _ in range(20):
                mid = (low + high) / 2 if factor != "volatility" else (low + high) / 2

                test_scenario = StressScenario(
                    name=f"reverse_{factor}",
                    scenario_type=ScenarioType.REVERSE,
                    description="Reverse stress test",
                    shocks={factor: mid},
                    duration_days=1,
                )

                result = self.run_scenario(test_scenario)

                if result.portfolio_impact_pct < max_acceptable_loss:
                    high = mid
                else:
                    low = mid

                if abs(high - low) < 0.01:
                    break

            breaking_points[factor] = high

        return breaking_points

    def generate_report(self) -> StressTestReport:
        """
        Generate comprehensive stress test report.

        Returns:
            Complete stress test report
        """
        from datetime import datetime

        # Run all scenarios
        results = self.run_all_scenarios()

        if not results:
            return StressTestReport(
                timestamp=datetime.now().isoformat(),
                portfolio_value=self.portfolio_value,
                results=[],
                worst_case_scenario="N/A",
                worst_case_loss_pct=0,
                tail_risk_summary={},
                recommendations=["Unable to run stress tests"],
            )

        # Find worst case
        worst = min(results, key=lambda r: r.portfolio_impact_pct)

        # Tail risk summary
        impacts = [r.portfolio_impact_pct for r in results]
        tail_risk = {
            "avg_scenario_loss": np.mean(impacts),
            "worst_scenario_loss": min(impacts),
            "scenarios_exceeding_5pct_loss": sum(1 for i in impacts if i < -0.05),
            "scenarios_exceeding_10pct_loss": sum(1 for i in impacts if i < -0.10),
            "scenarios_exceeding_20pct_loss": sum(1 for i in impacts if i < -0.20),
        }

        # Generate recommendations
        recommendations = []

        if worst.portfolio_impact_pct < -0.30:
            recommendations.append(
                f"CRITICAL: Portfolio vulnerable to >{abs(worst.portfolio_impact_pct):.0%} "
                f"loss in {worst.scenario_name} scenario"
            )

        if tail_risk["scenarios_exceeding_20pct_loss"] > 3:
            recommendations.append(
                "Consider reducing overall portfolio beta or adding hedges"
            )

        if worst.positions_at_risk:
            recommendations.append(
                f"High-risk positions: {', '.join(worst.positions_at_risk[:5])}"
            )

        # Sector concentration check
        max_sector_exposure = max(self.sector_exposure.values()) if self.sector_exposure else 0
        if max_sector_exposure > 0.30:
            sector = max(self.sector_exposure, key=self.sector_exposure.get)
            recommendations.append(
                f"High sector concentration in {sector} ({max_sector_exposure:.0%})"
            )

        return StressTestReport(
            timestamp=datetime.now().isoformat(),
            portfolio_value=self.portfolio_value,
            results=results,
            worst_case_scenario=worst.scenario_name,
            worst_case_loss_pct=worst.portfolio_impact_pct,
            tail_risk_summary=tail_risk,
            recommendations=recommendations,
        )


def run_stress_test(
    portfolio_value: float,
    positions: Dict[str, float],
    betas: Optional[Dict[str, float]] = None,
    sectors: Optional[Dict[str, str]] = None,
) -> StressTestReport:
    """
    Convenience function to run stress tests.

    Args:
        portfolio_value: Total portfolio value
        positions: Position values by symbol
        betas: Market betas
        sectors: Sector mappings

    Returns:
        Stress test report
    """
    tester = StressTester(
        portfolio_value=portfolio_value,
        position_values=positions,
        position_betas=betas,
        position_sectors=sectors,
    )

    return tester.generate_report()
