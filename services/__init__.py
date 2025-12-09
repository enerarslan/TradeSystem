"""
Services Module
===============

Microservices for distributed trading system architecture.

Each service runs independently and communicates via message bus:

Services:
- DataIngestionService: Connects to market data, pushes to bus
- StrategyService: Runs ML inference, generates signals
- RiskService: Validates signals, enforces limits
- OEMSService: Manages order lifecycle
- WatchdogService: Independent health monitoring

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from services.data_ingestion import DataIngestionService
from services.strategy_engine import StrategyService
from services.risk_engine import RiskService
from services.oems import OEMSService
from services.watchdog import WatchdogService

__all__ = [
    "DataIngestionService",
    "StrategyService",
    "RiskService",
    "OEMSService",
    "WatchdogService",
]
