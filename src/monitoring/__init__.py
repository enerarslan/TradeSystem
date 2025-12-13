"""
Monitoring Module
=================
Real-time monitoring and observability for the AlphaTrade System.

Components:
- ExecutionMonitor: Real-time execution quality tracking
- MetricsCollector: Prometheus-compatible metrics export
- AlertManager: Alert generation and routing
"""

from .execution_dashboard import (
    ExecutionMonitor,
    ExecutionMetricsCollector,
    ExecutionQualityReport,
    SlippageAnalyzer,
)

__all__ = [
    'ExecutionMonitor',
    'ExecutionMetricsCollector',
    'ExecutionQualityReport',
    'SlippageAnalyzer',
]
