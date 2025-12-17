"""
AlphaTrade System - Main Entry Point

This script orchestrates the full backtesting workflow:
1. Load and validate data
2. Generate features
3. Run strategies
4. Execute backtest
5. Generate performance report
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders.data_loader import DataLoader
from src.data.validators.data_validator import DataValidator
from src.data.processors.data_processor import DataProcessor
from src.features.technical.indicators import TechnicalIndicators
from src.strategies.momentum.multi_factor_momentum import MultiFactorMomentumStrategy
from src.strategies.mean_reversion.mean_reversion import MeanReversionStrategy
from src.strategies.multi_factor.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.backtesting.engine import BacktestEngine, VectorizedBacktest
from src.backtesting.analysis import BacktestAnalyzer
from src.backtesting.reports.report_generator import ReportGenerator
from src.risk.position_sizing import PositionSizer
from src.risk.var_models import VaRCalculator
from src.risk.drawdown import DrawdownController


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        "logs/alphatrade_{time}.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_validate_data(
    data_path: str,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load and validate market data.

    Args:
        data_path: Path to data directory
        symbols: Optional list of symbols to load

    Returns:
        Dictionary of validated DataFrames
    """
    logger.info(f"Loading data from {data_path}")

    loader = DataLoader(data_dir=data_path)
    validator = DataValidator()
    processor = DataProcessor()

    # Discover available symbols
    available_symbols = loader.symbols
    logger.info(f"Found {len(available_symbols)} symbols")

    if symbols:
        symbols = [s for s in symbols if s in available_symbols]
    else:
        symbols = available_symbols

    # Load and validate each symbol
    data = {}
    for symbol in symbols:
        try:
            df = loader.load_symbol(symbol)

            # Validate
            result = validator.validate(df, symbol=symbol)
            if not result.is_valid:
                logger.warning(f"{symbol}: Validation issues - {result.errors}")

            # Clean data
            df = processor.process(df)

            if len(df) > 100:  # Minimum data requirement
                data[symbol] = df
                logger.debug(f"Loaded {symbol}: {len(df)} bars")
            else:
                logger.warning(f"{symbol}: Insufficient data ({len(df)} bars)")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    logger.info(f"Successfully loaded {len(data)} symbols")
    return data


def create_strategies(config: dict[str, Any]) -> list:
    """
    Create strategy instances from configuration.

    Args:
        config: Strategy configuration

    Returns:
        List of strategy instances
    """
    strategies = []

    # Momentum Strategy
    if config.get("momentum", {}).get("enabled", True):
        momentum_params = config.get("momentum", {}).get("parameters", {})
        strategy = MultiFactorMomentumStrategy(
            name="Momentum",
            params={
                "lookback_periods": momentum_params.get("lookback_periods", [5, 10, 20]),
                "top_n_long": momentum_params.get("top_n_long", 5),
                "top_n_short": momentum_params.get("top_n_short", 0),
                "volatility_adjusted": momentum_params.get("volatility_adjusted", True),
            },
        )
        strategies.append(strategy)
        logger.info("Created Momentum strategy")

    # Mean Reversion Strategy
    if config.get("mean_reversion", {}).get("enabled", True):
        mr_params = config.get("mean_reversion", {}).get("parameters", {})
        strategy = MeanReversionStrategy(
            name="MeanReversion",
            params={
                "lookback_period": mr_params.get("lookback_period", 20),
                "entry_zscore": mr_params.get("entry_zscore", 2.0),
                "exit_zscore": mr_params.get("exit_zscore", 0.5),
            },
        )
        strategies.append(strategy)
        logger.info("Created MeanReversion strategy")

    # Volatility Breakout Strategy
    if config.get("volatility_breakout", {}).get("enabled", True):
        vb_params = config.get("volatility_breakout", {}).get("parameters", {})
        strategy = VolatilityBreakoutStrategy(
            name="VolatilityBreakout",
            params={
                "atr_period": vb_params.get("atr_period", 14),
                "atr_multiplier": vb_params.get("atr_multiplier", 2.0),
                "lookback_period": vb_params.get("lookback_period", 20),
            },
        )
        strategies.append(strategy)
        logger.info("Created VolatilityBreakout strategy")

    return strategies


def run_backtest(
    strategy,
    data: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> Any:
    """
    Run backtest for a strategy.

    Args:
        strategy: Strategy instance
        data: Market data
        config: Backtest configuration

    Returns:
        BacktestResult
    """
    logger.info(f"Running backtest for {strategy.name}")

    engine = BacktestEngine(
        initial_capital=config.get("initial_capital", 1000000),
        commission_pct=config.get("commission_pct", 0.001),
        slippage_pct=config.get("slippage_pct", 0.0005),
    )

    result = engine.run(strategy, data)

    logger.info(
        f"{strategy.name} - Return: {result.metrics.get('total_return', 0):.2%}, "
        f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}, "
        f"MaxDD: {result.metrics.get('max_drawdown', 0):.2%}"
    )

    return result


def run_ensemble_backtest(
    strategies: list,
    data: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> Any:
    """
    Run backtest for ensemble strategy.

    Args:
        strategies: List of strategy instances
        data: Market data
        config: Backtest configuration

    Returns:
        BacktestResult
    """
    logger.info("Running Ensemble strategy backtest")

    ensemble_config = config.get("ensemble", {}).get("parameters", {})

    ensemble = EnsembleStrategy(
        strategies=strategies,
        weights=ensemble_config.get("weights"),
        name="Ensemble",
        params={
            "combination_method": ensemble_config.get("combination_method", "weighted_average"),
            "min_agreement": ensemble_config.get("min_agreement", 0.6),
            "dynamic_weights": ensemble_config.get("dynamic_weights", True),
        },
    )

    engine = BacktestEngine(
        initial_capital=config.get("initial_capital", 1000000),
        commission_pct=config.get("commission_pct", 0.001),
        slippage_pct=config.get("slippage_pct", 0.0005),
    )

    result = engine.run(ensemble, data)

    logger.info(
        f"Ensemble - Return: {result.metrics.get('total_return', 0):.2%}, "
        f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}, "
        f"MaxDD: {result.metrics.get('max_drawdown', 0):.2%}"
    )

    return result


def generate_report(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Generate HTML performance report.

    Args:
        results: Dictionary of strategy results
        output_path: Output file path
    """
    logger.info(f"Generating report to {output_path}")

    # Use the best result for the main report
    best_strategy = max(
        results.keys(),
        key=lambda k: results[k].metrics.get("sharpe_ratio", 0),
    )
    best_result = results[best_strategy]

    generator = ReportGenerator(best_result)
    generator.generate_html_report(output_path)

    # Also generate comparison summary
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "Strategy": name,
            "Total Return": f"{result.metrics.get('total_return', 0):.2%}",
            "Annualized Return": f"{result.metrics.get('annualized_return', 0):.2%}",
            "Sharpe Ratio": f"{result.metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino Ratio": f"{result.metrics.get('sortino_ratio', 0):.2f}",
            "Max Drawdown": f"{result.metrics.get('max_drawdown', 0):.2%}",
            "Win Rate": f"{result.metrics.get('win_rate', 0):.2%}",
            "Num Trades": result.metrics.get("num_trades", 0),
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = output_path.replace(".html", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved strategy comparison to {summary_path}")

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AlphaTrade Backtesting System")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["momentum", "mean_reversion", "volatility_breakout", "ensemble", "all"],
        default="all",
        help="Strategy to backtest",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw",
        help="Path to data directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/strategy_params.yaml",
        help="Strategy configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/backtest_report.html",
        help="Output report path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to backtest (default: all)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000000,
        help="Initial capital",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("AlphaTrade System Starting")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Initial Capital: ${args.capital:,.0f}")

    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Load configuration
    try:
        strategy_config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        strategy_config = {}

    # Backtest configuration
    backtest_config = {
        "initial_capital": args.capital,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
    }
    backtest_config.update(strategy_config)

    # Load data
    data = load_and_validate_data(args.data_path, args.symbols)

    if not data:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Create strategies
    strategies = create_strategies(strategy_config)

    if not strategies:
        logger.error("No strategies created. Exiting.")
        sys.exit(1)

    # Run backtests
    results = {}

    if args.strategy == "all":
        # Run all individual strategies
        for strategy in strategies:
            try:
                result = run_backtest(strategy, data, backtest_config)
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Error running {strategy.name}: {e}")

        # Run ensemble
        if len(strategies) > 1:
            try:
                result = run_ensemble_backtest(strategies, data, backtest_config)
                results["Ensemble"] = result
            except Exception as e:
                logger.error(f"Error running Ensemble: {e}")

    elif args.strategy == "ensemble":
        result = run_ensemble_backtest(strategies, data, backtest_config)
        results["Ensemble"] = result

    else:
        # Run specific strategy
        strategy_map = {s.name.lower(): s for s in strategies}
        strategy_key = args.strategy.replace("_", "")

        for name, strategy in strategy_map.items():
            if strategy_key in name.lower():
                result = run_backtest(strategy, data, backtest_config)
                results[strategy.name] = result
                break

    if not results:
        logger.error("No backtest results. Exiting.")
        sys.exit(1)

    # Generate report
    generate_report(results, args.output)

    logger.info("AlphaTrade System Complete")
    logger.info(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
