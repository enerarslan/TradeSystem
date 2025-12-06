"""
Configuration Module
====================

Centralized configuration management for the algorithmic trading platform.
Uses Pydantic v2 for validation and environment variable loading.

Author: Algo Trading Platform
License: MIT
"""

from config.settings import (
    Settings,
    get_settings,
    TradingMode,
    LogLevel,
    TimeFrame,
    OrderSide,
    OrderType,
    PositionSide,
)

__all__ = [
    "Settings",
    "get_settings",
    "TradingMode",
    "LogLevel",
    "TimeFrame",
    "OrderSide",
    "OrderType",
    "PositionSide",
]