#!/usr/bin/env python3
"""
Backtesting Integration Module
==============================

JPMorgan-level backtesting integration for AlphaML Strategy V2.
Provides comprehensive model validation, performance analysis, and reporting.

Features:
- Full integration with AlphaMLStrategyV2
- Multi-symbol backtesting support
- Realistic execution simulation (slippage, commission)
- Comprehensive performance metrics
- Equity curve analysis
- Trade-level analytics
- Benchmark comparison (SPY)

Usage:
    from phase1.backtesting_integration import (
        BacktestRunner,
        run_symbol_backtest,
        run_portfolio_backtest,
    )
    
    # Single symbol
    results = run_symbol_backtest("AAPL", strategy_config)
    
    # Multi-symbol portfolio
    portfolio_results = run_portfolio_backtest(
        symbols=["AAPL", "GOOGL", "MSFT"],
        strategy_config=config,
    )

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging, TimeFrame
from config.symbols import ALL_SYMBOLS, CORE_SYMBOLS, get_symbol_info
from core.types import Order, Trade, Position, PortfolioState, BacktestError
from core.events import MarketEvent, SignalEvent
from data.loader import CSVLoader
from data.processor import DataProcessor
from features.pipeline import FeaturePipeline, create_default_config
from features.advanced import (
    TripleBarrierLabeler, 
    TripleBarrierConfig,
    MicrostructureFeatures,
    CalendarFeatures,
)
from backtesting.engine import (
    BacktestEngine, 
    BacktestConfig, 
    PortfolioTracker,
    ReportGenerator,
)
from backtesting.metrics import (
    PerformanceReport,
    MetricsCalculator,
    calculate_trade_stats,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)
from backtesting.execution import (
    create_realistic_simulator,
    PercentageSlippage,
    PerShareCommission,
)
from strategies.alpha_ml_v2 import (
    AlphaMLStrategyV2,
    AlphaMLConfigV2,
    PredictionMode,
    MarketRegime,
)
from models.model_manager import ModelManager
from risk.manager import RiskManager, RiskConfig, PositionSizingMethod

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestRunConfig:
    """Configuration for backtest run."""
    # Capital
    initial_capital: float = 100_000.0
    
    # Execution costs
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.0005  # 0.05% slippage
    
    # Position sizing
    max_position_size: float = 0.10  # 10% max per position
    max_positions: int = 5
    
    # Risk management
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2% stop loss
    use_take_profit: bool = True
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Timeframe
    warmup_bars: int = 200  # Bars to skip for indicator calculation
    
    # Benchmark
    benchmark_symbol: str = "SPY"
    
    # Reporting
    save_trades: bool = True
    save_equity_curve: bool = True
    output_dir: Path = field(default_factory=lambda: Path("reports/backtests"))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["output_dir"] = str(d["output_dir"])
        return d


@dataclass
class BacktestResult:
    """Result of a single backtest."""
    symbol: str
    strategy_name: str
    
    # Period
    start_date: datetime
    end_date: datetime
    total_bars: int
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    
    # Risk
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars
    var_95: float
    cvar_95: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_holding_period: float
    
    # Model-specific
    total_signals: int
    signals_executed: int
    avg_confidence: float
    regime_distribution: dict[str, float] = field(default_factory=dict)
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    
    # Metadata
    config: dict[str, Any] = field(default_factory=dict)
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Path | str | None = None) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        # Convert datetime objects
        data["start_date"] = str(data["start_date"])
        data["end_date"] = str(data["end_date"])
        
        json_str = json.dumps(data, indent=2, default=str)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

class BacktestRunner:
    """
    Comprehensive backtest runner for AlphaML Strategy.
    
    Handles data loading, strategy initialization, execution simulation,
    and performance reporting.
    
    Example:
        runner = BacktestRunner(config)
        result = runner.run_single("AAPL", start_date, end_date)
        
        # Or multiple symbols
        results = runner.run_multiple(["AAPL", "GOOGL", "MSFT"])
    """
    
    def __init__(
        self,
        config: BacktestRunConfig | None = None,
        strategy_config: AlphaMLConfigV2 | None = None,
    ):
        """Initialize backtest runner."""
        self.config = config or BacktestRunConfig()
        self.strategy_config = strategy_config or AlphaMLConfigV2()
        
        settings = get_settings()
        
        # Initialize components
        self._loader = CSVLoader(storage_path=settings.data.storage_path)
        self._processor = DataProcessor()
        self._feature_pipeline = FeaturePipeline(create_default_config())
        
        # Model manager
        self._model_manager = ModelManager()
        
        # Risk manager
        self._risk_manager = RiskManager(RiskConfig(
            max_position_size=self.config.max_position_size,
            max_positions=self.config.max_positions,
            use_stop_loss=self.config.use_stop_loss,
            default_stop_loss_pct=self.config.stop_loss_pct,
            use_take_profit=self.config.use_take_profit,
            default_take_profit_pct=self.config.take_profit_pct,
        ))
        
        # Results storage
        self._results: list[BacktestResult] = []
        
        # Trade tracking
        self._all_trades: list[Trade] = []
        self._all_signals: list[SignalEvent] = []
        
        logger.info("BacktestRunner initialized")
    
    def run_single(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str = "v1",
    ) -> BacktestResult:
        """
        Run backtest for a single symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            model_version: Model version to use
        
        Returns:
            BacktestResult with comprehensive metrics
        """
        symbol = symbol.upper()
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest for {symbol}")
        logger.info(f"{'='*60}")
        
        # Load and prepare data
        data = self._load_and_prepare_data(symbol, start_date, end_date)
        
        if data is None or len(data) < self.config.warmup_bars:
            raise BacktestError(f"Insufficient data for {symbol}")
        
        # Create strategy instance
        strategy = self._create_strategy(symbol, model_version)
        
        # Create backtest engine
        engine = self._create_engine()
        engine.add_data(symbol, data)
        engine.add_strategy(strategy)
        
        # Load benchmark if available
        try:
            benchmark = self._load_benchmark_data(start_date, end_date)
            if benchmark is not None:
                engine.set_benchmark(benchmark)
        except Exception as e:
            logger.warning(f"Could not load benchmark: {e}")
        
        # Run backtest
        actual_start = data["timestamp"].min()
        actual_end = data["timestamp"].max()
        
        logger.info(f"Period: {actual_start} to {actual_end}")
        logger.info(f"Bars: {len(data)}")
        
        report = engine.run(start_date, end_date, show_progress=True)
        
        # Collect results
        trades = engine.get_trades()
        signals = engine.get_signals()
        equity_curve = engine.get_equity_curve()
        
        # Build result
        result = self._build_result(
            symbol=symbol,
            strategy=strategy,
            report=report,
            trades=trades,
            signals=signals,
            equity_curve=equity_curve,
            data=data,
        )
        
        # Store
        self._results.append(result)
        self._all_trades.extend(trades)
        self._all_signals.extend(signals)
        
        # Save if configured
        if self.config.save_trades or self.config.save_equity_curve:
            self._save_results(symbol, result, trades, equity_curve)
        
        logger.info(f"\nBacktest complete for {symbol}")
        logger.info(f"  Total Return: {result.total_return_pct:.2%}")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        
        return result
    
    def run_multiple(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str = "v1",
        parallel: bool = False,
    ) -> list[BacktestResult]:
        """
        Run backtests for multiple symbols.
        
        Args:
            symbols: List of symbols
            start_date: Backtest start date
            end_date: Backtest end date
            model_version: Model version to use
            parallel: Run in parallel (not implemented yet)
        
        Returns:
            List of BacktestResults
        """
        results = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
            
            try:
                result = self.run_single(symbol, start_date, end_date, model_version)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to backtest {symbol}: {e}")
                continue
        
        # Generate summary
        self._generate_summary_report(results)
        
        return results
    
    def run_portfolio(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str = "v1",
        allocation_method: str = "equal",  # equal, risk_parity, optimized
    ) -> BacktestResult:
        """
        Run portfolio backtest with multiple symbols trading together.
        
        This simulates a real portfolio with position limits across all symbols.
        
        Args:
            symbols: List of symbols
            start_date: Backtest start date
            end_date: Backtest end date
            model_version: Model version
            allocation_method: How to allocate capital
        
        Returns:
            Combined portfolio BacktestResult
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running PORTFOLIO backtest")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"{'='*60}")
        
        # Load all data
        all_data: dict[str, pl.DataFrame] = {}
        strategies: dict[str, AlphaMLStrategyV2] = {}
        
        for symbol in symbols:
            try:
                data = self._load_and_prepare_data(symbol, start_date, end_date)
                if data is not None and len(data) >= self.config.warmup_bars:
                    all_data[symbol] = data
                    strategies[symbol] = self._create_strategy(symbol, model_version)
            except Exception as e:
                logger.warning(f"Could not load {symbol}: {e}")
        
        if not all_data:
            raise BacktestError("No valid data loaded for portfolio")
        
        # Create multi-asset engine
        engine = self._create_engine()
        
        for symbol, data in all_data.items():
            engine.add_data(symbol, data)
        
        # Add combined strategy wrapper
        for strategy in strategies.values():
            engine.add_strategy(strategy)
        
        # Run backtest
        report = engine.run(start_date, end_date, show_progress=True)
        
        # Build portfolio result
        result = self._build_portfolio_result(
            symbols=list(all_data.keys()),
            report=report,
            trades=engine.get_trades(),
            equity_curve=engine.get_equity_curve(),
        )
        
        return result
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _load_and_prepare_data(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame | None:
        """Load and prepare data for backtesting."""
        try:
            # Load raw data
            df = self._loader.load(symbol, start_date=start_date, end_date=end_date)
            
            if df is None or len(df) == 0:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Process data
            df = self._processor.process(df)
            
            # Generate features
            df = self._feature_pipeline.generate(df)
            
            # Add advanced features
            df = MicrostructureFeatures.add_features(df)
            df = CalendarFeatures.add_features(df)
            
            # Drop rows with NaN in critical columns
            critical_cols = ["open", "high", "low", "close", "volume"]
            df = df.drop_nulls(subset=critical_cols)
            
            logger.info(f"Loaded {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _create_strategy(
        self,
        symbol: str,
        model_version: str,
    ) -> AlphaMLStrategyV2:
        """Create strategy instance for symbol."""
        config = AlphaMLConfigV2(
            symbol=symbol,
            use_lightgbm=self.strategy_config.use_lightgbm,
            use_xgboost=self.strategy_config.use_xgboost,
            model_version=model_version,
            models_dir=self.strategy_config.models_dir,
            prediction_mode=self.strategy_config.prediction_mode,
            min_confidence=self.strategy_config.min_confidence,
            use_stop_loss=self.config.use_stop_loss,
            stop_loss_pct=self.config.stop_loss_pct,
            use_take_profit=self.config.use_take_profit,
            take_profit_pct=self.config.take_profit_pct,
            max_positions=self.config.max_positions,
            max_position_size=self.config.max_position_size,
        )
        
        return AlphaMLStrategyV2(config)
    
    def _create_engine(self) -> BacktestEngine:
        """Create backtest engine with configured settings."""
        config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
            warmup_bars=self.config.warmup_bars,
            allow_shorting=False,  # Long-only for now
        )
        
        return BacktestEngine(config)
    
    def _load_benchmark_data(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame | None:
        """Load benchmark data (SPY)."""
        try:
            return self._loader.load(
                self.config.benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            return None
    
    def _build_result(
        self,
        symbol: str,
        strategy: AlphaMLStrategyV2,
        report: PerformanceReport,
        trades: list[Trade],
        signals: list[SignalEvent],
        equity_curve: pl.DataFrame,
        data: pl.DataFrame,
    ) -> BacktestResult:
        """Build comprehensive backtest result."""
        # Calculate additional metrics
        equity = equity_curve["equity"].to_numpy()
        returns = np.diff(equity) / equity[:-1]
        
        # Regime distribution
        regime_dist = {}
        regime_summary = strategy.get_regime_summary(symbol)
        if regime_summary:
            regime_dist = {"current": regime_summary.get("current_regime", "unknown")}
        
        # Trade statistics
        trade_stats = calculate_trade_stats(trades) if trades else None
        
        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            start_date=report.start_date,
            end_date=report.end_date,
            total_bars=len(data),
            initial_capital=report.initial_capital,
            final_capital=report.final_capital,
            total_return=report.final_capital - report.initial_capital,
            total_return_pct=report.total_return_pct,
            annualized_return=report.annualized_return,
            volatility=report.annualized_volatility,
            max_drawdown=report.max_drawdown,
            max_drawdown_duration=report.max_drawdown_duration,
            var_95=report.var_95,
            cvar_95=report.cvar_95,
            sharpe_ratio=report.sharpe_ratio,
            sortino_ratio=report.sortino_ratio,
            calmar_ratio=report.calmar_ratio,
            total_trades=trade_stats.total_trades if trade_stats else 0,
            winning_trades=trade_stats.winning_trades if trade_stats else 0,
            losing_trades=trade_stats.losing_trades if trade_stats else 0,
            win_rate=trade_stats.win_rate if trade_stats else 0,
            profit_factor=trade_stats.profit_factor if trade_stats else 0,
            avg_trade_return=trade_stats.avg_trade if trade_stats else 0,
            avg_win=trade_stats.avg_win if trade_stats else 0,
            avg_loss=trade_stats.avg_loss if trade_stats else 0,
            max_win=trade_stats.max_win if trade_stats else 0,
            max_loss=trade_stats.max_loss if trade_stats else 0,
            avg_holding_period=trade_stats.avg_holding_period if trade_stats else 0,
            total_signals=len(signals),
            signals_executed=len(trades),
            avg_confidence=np.mean([s.confidence for s in signals]) if signals else 0,
            regime_distribution=regime_dist,
            benchmark_return=report.benchmark_return if hasattr(report, 'benchmark_return') else 0,
            alpha=report.alpha if hasattr(report, 'alpha') else 0,
            beta=report.beta if hasattr(report, 'beta') else 0,
            information_ratio=report.information_ratio if hasattr(report, 'information_ratio') else 0,
            config=self.config.to_dict(),
        )
    
    def _build_portfolio_result(
        self,
        symbols: list[str],
        report: PerformanceReport,
        trades: list[Trade],
        equity_curve: pl.DataFrame,
    ) -> BacktestResult:
        """Build portfolio-level result."""
        trade_stats = calculate_trade_stats(trades) if trades else None
        
        return BacktestResult(
            symbol=",".join(symbols),
            strategy_name="AlphaML_Portfolio",
            start_date=report.start_date,
            end_date=report.end_date,
            total_bars=len(equity_curve),
            initial_capital=report.initial_capital,
            final_capital=report.final_capital,
            total_return=report.final_capital - report.initial_capital,
            total_return_pct=report.total_return_pct,
            annualized_return=report.annualized_return,
            volatility=report.annualized_volatility,
            max_drawdown=report.max_drawdown,
            max_drawdown_duration=report.max_drawdown_duration,
            var_95=report.var_95,
            cvar_95=report.cvar_95,
            sharpe_ratio=report.sharpe_ratio,
            sortino_ratio=report.sortino_ratio,
            calmar_ratio=report.calmar_ratio,
            total_trades=trade_stats.total_trades if trade_stats else 0,
            winning_trades=trade_stats.winning_trades if trade_stats else 0,
            losing_trades=trade_stats.losing_trades if trade_stats else 0,
            win_rate=trade_stats.win_rate if trade_stats else 0,
            profit_factor=trade_stats.profit_factor if trade_stats else 0,
            avg_trade_return=trade_stats.avg_trade if trade_stats else 0,
            avg_win=trade_stats.avg_win if trade_stats else 0,
            avg_loss=trade_stats.avg_loss if trade_stats else 0,
            max_win=trade_stats.max_win if trade_stats else 0,
            max_loss=trade_stats.max_loss if trade_stats else 0,
            avg_holding_period=trade_stats.avg_holding_period if trade_stats else 0,
            total_signals=0,
            signals_executed=len(trades),
            avg_confidence=0,
            config=self.config.to_dict(),
        )
    
    def _save_results(
        self,
        symbol: str,
        result: BacktestResult,
        trades: list[Trade],
        equity_curve: pl.DataFrame,
    ) -> None:
        """Save backtest results to disk."""
        output_dir = self.config.output_dir / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save result JSON
        result_path = output_dir / f"backtest_{timestamp}.json"
        result.to_json(result_path)
        
        # Save equity curve
        if self.config.save_equity_curve:
            equity_path = output_dir / f"equity_{timestamp}.csv"
            equity_curve.write_csv(equity_path)
        
        # Save trades
        if self.config.save_trades and trades:
            trades_data = [t.to_dict() if hasattr(t, 'to_dict') else vars(t) for t in trades]
            trades_path = output_dir / f"trades_{timestamp}.json"
            with open(trades_path, "w") as f:
                json.dump(trades_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _generate_summary_report(self, results: list[BacktestResult]) -> None:
        """Generate summary report for multiple backtests."""
        if not results:
            return
        
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate aggregate statistics
        summary = {
            "total_symbols": len(results),
            "timestamp": datetime.now().isoformat(),
            "aggregate": {
                "avg_return": np.mean([r.total_return_pct for r in results]),
                "avg_sharpe": np.mean([r.sharpe_ratio for r in results]),
                "avg_max_drawdown": np.mean([r.max_drawdown for r in results]),
                "avg_win_rate": np.mean([r.win_rate for r in results]),
                "total_trades": sum([r.total_trades for r in results]),
            },
            "best_performers": sorted(
                [(r.symbol, r.total_return_pct) for r in results],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "worst_performers": sorted(
                [(r.symbol, r.total_return_pct) for r in results],
                key=lambda x: x[1],
            )[:5],
            "symbols": [r.to_dict() for r in results],
        }
        
        # Save summary
        summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Symbols: {len(results)}")
        logger.info(f"Avg Return: {summary['aggregate']['avg_return']:.2%}")
        logger.info(f"Avg Sharpe: {summary['aggregate']['avg_sharpe']:.2f}")
        logger.info(f"Avg Max Drawdown: {summary['aggregate']['avg_max_drawdown']:.2%}")
        logger.info(f"Avg Win Rate: {summary['aggregate']['avg_win_rate']:.2%}")
        logger.info(f"Total Trades: {summary['aggregate']['total_trades']}")
        logger.info("="*60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_symbol_backtest(
    symbol: str,
    strategy_config: AlphaMLConfigV2 | None = None,
    backtest_config: BacktestRunConfig | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> BacktestResult:
    """
    Convenience function to run a single symbol backtest.
    
    Args:
        symbol: Trading symbol
        strategy_config: Strategy configuration
        backtest_config: Backtest configuration
        start_date: Start date
        end_date: End date
    
    Returns:
        BacktestResult
    """
    runner = BacktestRunner(backtest_config, strategy_config)
    return runner.run_single(symbol, start_date, end_date)


def run_portfolio_backtest(
    symbols: list[str],
    strategy_config: AlphaMLConfigV2 | None = None,
    backtest_config: BacktestRunConfig | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> BacktestResult:
    """
    Convenience function to run portfolio backtest.
    
    Args:
        symbols: List of trading symbols
        strategy_config: Strategy configuration
        backtest_config: Backtest configuration
        start_date: Start date
        end_date: End date
    
    Returns:
        Combined BacktestResult
    """
    runner = BacktestRunner(backtest_config, strategy_config)
    return runner.run_portfolio(symbols, start_date, end_date)


def run_all_symbols_backtest(
    strategy_config: AlphaMLConfigV2 | None = None,
    backtest_config: BacktestRunConfig | None = None,
    core_only: bool = False,
) -> list[BacktestResult]:
    """
    Run backtest for all available symbols.
    
    Args:
        strategy_config: Strategy configuration
        backtest_config: Backtest configuration
        core_only: Use only core (most liquid) symbols
    
    Returns:
        List of BacktestResults
    """
    symbols = CORE_SYMBOLS if core_only else ALL_SYMBOLS
    runner = BacktestRunner(backtest_config, strategy_config)
    return runner.run_multiple(symbols)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AlphaML backtests")
    parser.add_argument("--symbol", "-s", type=str, help="Single symbol to backtest")
    parser.add_argument("--symbols", "-S", type=str, nargs="+", help="Multiple symbols")
    parser.add_argument("--all", "-a", action="store_true", help="Backtest all symbols")
    parser.add_argument("--core", "-c", action="store_true", help="Core symbols only")
    parser.add_argument("--portfolio", "-p", action="store_true", help="Portfolio backtest")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    # Create configs
    backtest_config = BacktestRunConfig(initial_capital=args.capital)
    if args.output:
        backtest_config.output_dir = Path(args.output)
    
    strategy_config = AlphaMLConfigV2()
    
    # Run appropriate backtest
    if args.symbol:
        result = run_symbol_backtest(
            args.symbol, strategy_config, backtest_config, start_date, end_date
        )
        print(f"\nResult: {result.total_return_pct:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
    
    elif args.symbols:
        if args.portfolio:
            result = run_portfolio_backtest(
                args.symbols, strategy_config, backtest_config, start_date, end_date
            )
            print(f"\nPortfolio Result: {result.total_return_pct:.2%} return")
        else:
            runner = BacktestRunner(backtest_config, strategy_config)
            results = runner.run_multiple(args.symbols, start_date, end_date)
            print(f"\nCompleted {len(results)} backtests")
    
    elif args.all:
        results = run_all_symbols_backtest(strategy_config, backtest_config, args.core)
        print(f"\nCompleted {len(results)} backtests")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    configure_logging(get_settings())
    main()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "BacktestRunConfig",
    "BacktestResult",
    # Main class
    "BacktestRunner",
    # Convenience functions
    "run_symbol_backtest",
    "run_portfolio_backtest",
    "run_all_symbols_backtest",
]