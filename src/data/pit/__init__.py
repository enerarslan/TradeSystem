"""
Point-in-Time (PIT) data infrastructure for AlphaTrade system.

This module provides institutional-grade data handling that prevents:
1. Survivorship bias through historical universe management
2. Look-ahead bias through as-of timestamp queries
3. Incorrect data through corporate action adjustments

Reference: "Quantitative Portfolio Management" by Isichenko (2021)
"""

from .universe_manager import UniverseManager, SymbolMetadata
from .pit_loader import PITDataLoader
from .corporate_actions import CorporateActionAdjuster, CorporateAction
from .as_of_query import AsOfQueryEngine

__all__ = [
    "UniverseManager",
    "SymbolMetadata",
    "PITDataLoader",
    "CorporateActionAdjuster",
    "CorporateAction",
    "AsOfQueryEngine",
]
