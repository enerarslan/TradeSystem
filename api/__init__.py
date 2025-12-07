"""
API Module
==========

REST API for the algorithmic trading platform using FastAPI.

Components:
- main: FastAPI application with all routes

Endpoints:
- /health: Health check
- /backtest: Run backtests
- /models: Model management
- /strategies: Strategy management
- /data: Data access
- /portfolio: Portfolio information
- /ws: WebSocket for real-time updates

Author: Algo Trading Platform
License: MIT
"""

from api.main import (
    app,
    create_app,
)

__all__ = [
    "app",
    "create_app",
]