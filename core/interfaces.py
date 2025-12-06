"""
Interfaces Module
=================

Abstract base classes (protocols) defining the contracts for all
major components in the algorithmic trading platform.

These interfaces enable:
- Loose coupling between components
- Easy testing with mock implementations
- Plugin architecture for strategies and models

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from core.events import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.types import (
    Order,
    Position,
    PortfolioState,
    Signal,
    Trade,
    PerformanceMetrics,
)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="Model")


# =============================================================================
# DATA PROVIDER INTERFACE
# =============================================================================

@runtime_checkable
class DataProvider(Protocol):
    """
    Protocol for data providers.
    
    Data providers are responsible for loading and providing
    historical and real-time market data.
    """
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "15min",
    ) -> pl.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        ...
    
    def get_latest_bar(self, symbol: str) -> dict[str, Any] | None:
        """
        Get the latest bar for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Latest bar as dictionary or None
        """
        ...
    
    def get_symbols(self) -> list[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of symbol strings
        """
        ...
    
    def subscribe(self, symbol: str) -> None:
        """
        Subscribe to real-time data for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        ...
    
    def unsubscribe(self, symbol: str) -> None:
        """
        Unsubscribe from real-time data.
        
        Args:
            symbol: Trading symbol
        """
        ...


# =============================================================================
# STRATEGY INTERFACE
# =============================================================================

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies must inherit from this class and
    implement the required methods.
    
    Lifecycle:
        1. __init__: Strategy initialization with parameters
        2. initialize: Called before backtest/trading starts
        3. on_bar: Called for each new bar (main logic)
        4. on_signal: Called when signal is generated (optional)
        5. on_fill: Called when order is filled (optional)
        6. shutdown: Called when strategy stops
    """
    
    def __init__(self, name: str, parameters: dict[str, Any] | None = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self._is_initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if strategy is initialized."""
        return self._is_initialized
    
    def initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Initialize strategy before trading.
        
        Override this method to set up indicators, state, etc.
        
        Args:
            symbols: List of symbols to trade
            start_date: Trading start date
            end_date: Trading end date
        """
        self._is_initialized = True
    
    @abstractmethod
    def on_bar(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """
        Process a new bar and generate signals.
        
        This is the main strategy logic. Called for each new bar.
        
        Args:
            event: Market event with bar data
            portfolio: Current portfolio state
        
        Returns:
            List of signal events (can be empty)
        """
        pass
    
    def on_signal(self, event: SignalEvent) -> None:
        """
        Called when a signal is generated.
        
        Override for signal post-processing.
        
        Args:
            event: Generated signal event
        """
        pass
    
    def on_fill(self, event: FillEvent) -> None:
        """
        Called when an order is filled.
        
        Override to update strategy state on fills.
        
        Args:
            event: Fill event
        """
        pass
    
    def on_order(self, event: OrderEvent) -> None:
        """
        Called when an order is created.
        
        Override to track pending orders.
        
        Args:
            event: Order event
        """
        pass
    
    def shutdown(self) -> None:
        """
        Cleanup when strategy stops.
        
        Override to close resources, save state, etc.
        """
        self._is_initialized = False
    
    def get_parameters(self) -> dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()
    
    def set_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            parameters: New parameters to set
        """
        self.parameters.update(parameters)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# =============================================================================
# RISK MANAGER INTERFACE
# =============================================================================

class RiskManager(ABC):
    """
    Abstract base class for risk management.
    
    Risk managers validate orders and monitor portfolio risk.
    """
    
    @abstractmethod
    def validate_order(
        self,
        order: Order,
        portfolio: PortfolioState,
    ) -> tuple[bool, str]:
        """
        Validate an order against risk limits.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        signal: SignalEvent,
        portfolio: PortfolioState,
    ) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            portfolio: Current portfolio state
        
        Returns:
            Recommended position size
        """
        pass
    
    @abstractmethod
    def check_portfolio_risk(
        self,
        portfolio: PortfolioState,
    ) -> list[tuple[str, str, float]]:
        """
        Check portfolio for risk limit breaches.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            List of (risk_type, level, value) tuples
        """
        pass
    
    @abstractmethod
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
        
        Returns:
            VaR value
        """
        pass
    
    @abstractmethod
    def get_risk_metrics(
        self,
        portfolio: PortfolioState,
    ) -> dict[str, float]:
        """
        Calculate current risk metrics.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            Dictionary of risk metrics
        """
        pass


# =============================================================================
# EXECUTION HANDLER INTERFACE
# =============================================================================

class ExecutionHandler(ABC):
    """
    Abstract base class for order execution.
    
    Execution handlers submit orders to brokers and
    handle fills.
    """
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
        
        Returns:
            Broker order ID
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Broker order ID
        
        Returns:
            True if cancellation was successful
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get current order status.
        
        Args:
            order_id: Broker order ID
        
        Returns:
            Order status dictionary
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> list[Position]:
        """
        Get current positions from broker.
        
        Returns:
            List of positions
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account info dictionary
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open
        """
        pass


# =============================================================================
# PORTFOLIO MANAGER INTERFACE
# =============================================================================

class PortfolioManager(ABC):
    """
    Abstract base class for portfolio management.
    
    Portfolio managers track positions, calculate PnL,
    and manage portfolio state.
    """
    
    @abstractmethod
    def get_state(self) -> PortfolioState:
        """
        Get current portfolio state.
        
        Returns:
            Current portfolio state
        """
        pass
    
    @abstractmethod
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
    ) -> Position:
        """
        Update a position.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to add/remove
            price: Transaction price
            side: Transaction side (buy/sell)
        
        Returns:
            Updated position
        """
        pass
    
    @abstractmethod
    def close_position(
        self,
        symbol: str,
        price: float,
    ) -> Trade:
        """
        Close a position completely.
        
        Args:
            symbol: Trading symbol
            price: Closing price
        
        Returns:
            Completed trade record
        """
        pass
    
    @abstractmethod
    def update_prices(
        self,
        prices: dict[str, float],
    ) -> None:
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of symbol to price
        """
        pass
    
    @abstractmethod
    def get_trades(self) -> list[Trade]:
        """
        Get all completed trades.
        
        Returns:
            List of trades
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate performance metrics.
        
        Returns:
            Performance metrics
        """
        pass


# =============================================================================
# MODEL INTERFACE
# =============================================================================

class Model(ABC):
    """
    Abstract base class for machine learning models.
    
    Provides a unified interface for training, prediction,
    and model management.
    """
    
    def __init__(self, name: str, parameters: dict[str, Any] | None = None):
        """
        Initialize model.
        
        Args:
            name: Model name
            parameters: Model hyperparameters
        """
        self.name = name
        self.parameters = parameters or {}
        self._is_trained = False
        self._feature_names: list[str] = []
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
    
    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names
    
    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelT:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            **kwargs: Additional training arguments
        
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """
        Predict class probabilities (for classifiers).
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability matrix or None if not supported
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> ModelT:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        
        Returns:
            Self for chaining
        """
        pass
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importances.
        
        Returns:
            Dictionary of feature name to importance or None
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, trained={self._is_trained})"


# =============================================================================
# BACKTEST ENGINE INTERFACE
# =============================================================================

class BacktestEngine(ABC):
    """
    Abstract base class for backtesting engines.
    """
    
    @abstractmethod
    def run(
        self,
        strategy: Strategy,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
    ) -> PerformanceMetrics:
        """
        Run a backtest.
        
        Args:
            strategy: Strategy to test
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
        
        Returns:
            Performance metrics
        """
        pass
    
    @abstractmethod
    def get_trades(self) -> list[Trade]:
        """
        Get all trades from backtest.
        
        Returns:
            List of trades
        """
        pass
    
    @abstractmethod
    def get_equity_curve(self) -> pl.DataFrame:
        """
        Get equity curve from backtest.
        
        Returns:
            DataFrame with timestamp and equity columns
        """
        pass
    
    @abstractmethod
    def generate_report(self, output_path: str) -> None:
        """
        Generate backtest report.
        
        Args:
            output_path: Path to save report
        """
        pass


# =============================================================================
# FEATURE PIPELINE INTERFACE
# =============================================================================

@runtime_checkable
class FeatureGenerator(Protocol):
    """
    Protocol for feature generators.
    """
    
    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate features from OHLCV data.
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with features added
        """
        ...
    
    def get_feature_names(self) -> list[str]:
        """
        Get names of generated features.
        
        Returns:
            List of feature names
        """
        ...


# =============================================================================
# OPTIMIZER INTERFACE
# =============================================================================

class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """
    
    @abstractmethod
    def optimize(
        self,
        objective_fn: callable,
        param_space: dict[str, Any],
        n_trials: int = 100,
    ) -> dict[str, Any]:
        """
        Run optimization.
        
        Args:
            objective_fn: Function to optimize
            param_space: Parameter search space
            n_trials: Number of optimization trials
        
        Returns:
            Best parameters found
        """
        pass
    
    @abstractmethod
    def get_results(self) -> pl.DataFrame:
        """
        Get optimization results.
        
        Returns:
            DataFrame with trial results
        """
        pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DataProvider",
    "Strategy",
    "RiskManager",
    "ExecutionHandler",
    "PortfolioManager",
    "Model",
    "BacktestEngine",
    "FeatureGenerator",
    "Optimizer",
]