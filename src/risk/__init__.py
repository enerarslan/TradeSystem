"""
Risk management module for AlphaTrade system.

This module provides comprehensive risk management:
- Position sizing algorithms
- VaR calculations
- Drawdown controls
- Correlation analysis
- Circuit breakers
- Real-time risk monitoring
- Pre-trade compliance
"""

from src.risk.position_sizing import (
    PositionSizer,
    fixed_fraction,
    kelly_criterion,
    volatility_target,
    risk_parity_weights,
)
from src.risk.var_models import (
    VaRCalculator,
    calculate_var,
    calculate_cvar,
)
from src.risk.drawdown import (
    DrawdownController,
    calculate_drawdown,
    calculate_max_drawdown,
)
from src.risk.correlation import (
    CorrelationAnalyzer,
    calculate_correlation_matrix,
)
from src.risk.circuit_breakers import (
    CircuitBreakerManager,
    CircuitBreakerAction,
    MarketCircuitBreaker,
    PortfolioCircuitBreaker,
    VolatilityCircuitBreaker,
)
from src.risk.realtime_monitor import (
    RealTimeRiskMonitor,
    RiskAlert,
    AlertSeverity,
    PortfolioRiskSnapshot,
)
from src.risk.pretrade_compliance import (
    PreTradeComplianceChecker,
    Order,
    ComplianceResult,
    ComplianceReport,
)

# New institutional-grade modules
from src.risk.execution_quality import (
    ExecutionQualityAnalyzer,
    TradeExecution,
    ExecutionMetrics,
    AggregateExecutionReport,
    analyze_execution_quality,
)
from src.risk.attribution import (
    RiskAttribution,
    FactorExposure,
    SectorAttribution,
    FactorModel,
    BrinsonAttribution,
    run_risk_attribution,
)
from src.risk.stress_testing import (
    StressTester,
    StressScenario,
    ScenarioResult,
    StressTestReport,
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
    run_stress_test,
)
from src.risk.monitoring import (
    RiskMonitor as InstitutionalRiskMonitor,
    LiquidityMonitor,
    ModelDecayMonitor,
    LiquidityMetrics,
    ModelDecayMetrics,
    MonitoringReport,
)

__all__ = [
    # Position sizing
    "PositionSizer",
    "fixed_fraction",
    "kelly_criterion",
    "volatility_target",
    "risk_parity_weights",
    # VaR
    "VaRCalculator",
    "calculate_var",
    "calculate_cvar",
    # Drawdown
    "DrawdownController",
    "calculate_drawdown",
    "calculate_max_drawdown",
    # Correlation
    "CorrelationAnalyzer",
    "calculate_correlation_matrix",
    # Circuit breakers
    "CircuitBreakerManager",
    "CircuitBreakerAction",
    "MarketCircuitBreaker",
    "PortfolioCircuitBreaker",
    "VolatilityCircuitBreaker",
    # Real-time monitoring
    "RealTimeRiskMonitor",
    "RiskAlert",
    "AlertSeverity",
    "PortfolioRiskSnapshot",
    # Compliance
    "PreTradeComplianceChecker",
    "Order",
    "ComplianceResult",
    "ComplianceReport",
    # Execution Quality
    "ExecutionQualityAnalyzer",
    "TradeExecution",
    "ExecutionMetrics",
    "AggregateExecutionReport",
    "analyze_execution_quality",
    # Risk Attribution
    "RiskAttribution",
    "FactorExposure",
    "SectorAttribution",
    "FactorModel",
    "BrinsonAttribution",
    "run_risk_attribution",
    # Stress Testing
    "StressTester",
    "StressScenario",
    "ScenarioResult",
    "StressTestReport",
    "HISTORICAL_SCENARIOS",
    "HYPOTHETICAL_SCENARIOS",
    "run_stress_test",
    # Institutional Monitoring
    "InstitutionalRiskMonitor",
    "LiquidityMonitor",
    "ModelDecayMonitor",
    "LiquidityMetrics",
    "ModelDecayMetrics",
    "MonitoringReport",
]
