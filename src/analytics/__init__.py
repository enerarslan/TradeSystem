"""
Analytics Module
================
Performance analytics and P&L attribution for the AlphaTrade System.

Components:
- PnLAttribution: Trade-level P&L decomposition
- PerformanceAnalyzer: Portfolio performance metrics
- RiskAttribution: Risk factor attribution
"""

from .attribution import (
    PnLAttribution,
    TradeAttribution,
    PortfolioAttribution,
    AttributionReport,
)

__all__ = [
    'PnLAttribution',
    'TradeAttribution',
    'PortfolioAttribution',
    'AttributionReport',
]
