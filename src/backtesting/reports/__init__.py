"""
Report generation module for AlphaTrade system.

This module provides:
- HTML report generation
- Interactive performance dashboards
- Institutional-grade tear sheets
- Multi-strategy comparison

Designed for JPMorgan-level requirements:
- Comprehensive risk metrics
- Interactive visualizations
- Professional styling
"""

from src.backtesting.reports.report_generator import (
    ReportGenerator,
    generate_html_report,
)
from src.backtesting.reports.dashboard import (
    PerformanceDashboard,
    MetricsCalculator,
    TearSheetMetrics,
    create_tear_sheet,
)

__all__ = [
    "ReportGenerator",
    "generate_html_report",
    "PerformanceDashboard",
    "MetricsCalculator",
    "TearSheetMetrics",
    "create_tear_sheet",
]
