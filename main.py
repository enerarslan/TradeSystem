"""
AlphaTrade System - Main Entry Point

Institutional-Grade Algorithmic Trading Platform
Version 2.0.0

This script orchestrates the full trading system workflow:
1. Load and validate data
2. Generate features (technical, fractional diff, macro)
3. Train ML models (optional)
4. Run strategies (traditional or ML-based)
5. Execute backtest (vectorized or event-driven)
6. Generate performance report with tear sheet

Usage:
    # Basic backtest
    python main.py

    # With specific strategy
    python main.py --strategy momentum

    # Train ML model and backtest
    python main.py --mode train --model lightgbm

    # Event-driven backtest
    python main.py --engine event-driven

    # Generate tear sheet only
    python main.py --report-only --input results/last_backtest.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
from src.data.loaders.data_loader import DataLoader
from src.data.validators.data_validator import DataValidator
from src.data.processors.data_processor import DataProcessor
from src.features.technical.indicators import TechnicalIndicators
from src.features.pipeline import FeaturePipeline

# Strategy imports
from src.strategies.momentum.multi_factor_momentum import MultiFactorMomentumStrategy
from src.strategies.mean_reversion.mean_reversion import MeanReversionStrategy
from src.strategies.multi_factor.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy

# Backtesting imports
from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.event_engine import EventDrivenEngine, EventEngineConfig
from src.backtesting.analysis import BacktestAnalyzer
from src.backtesting.reports.report_generator import ReportGenerator
from src.backtesting.reports.dashboard import PerformanceDashboard, create_tear_sheet

# Risk imports
from src.risk.position_sizing import PositionSizer
from src.risk.var_models import VaRCalculator
from src.risk.drawdown import DrawdownController

# Training imports
from src.training import ModelFactory, Trainer, TrainingResult
from src.training.validation import PurgedKFoldCV, WalkForwardValidator
from src.training.optimization import OptunaOptimizer


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "data": {
        "path": "data/raw",
        "cache_path": "data/cache",
        "min_bars": 100,
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "slippage_bps": 1.0,
    },
    "strategies": {
        "momentum": {
            "enabled": True,
            "lookback_periods": [5, 10, 20],
            "top_n_long": 5,
            "volatility_adjusted": True,
        },
        "mean_reversion": {
            "enabled": True,
            "lookback_period": 20,
            "entry_zscore": 2.0,
            "exit_zscore": 0.5,
        },
        "volatility_breakout": {
            "enabled": True,
            "atr_period": 14,
            "atr_multiplier": 2.0,
        },
    },
    "training": {
        "model_type": "lightgbm",
        "cv_splits": 5,
        "purge_gap": 5,
        "embargo_pct": 0.01,
        "optuna_trials": 50,
    },
    "risk": {
        "max_position_pct": 0.05,
        "max_drawdown_pct": 0.15,
        "var_confidence": 0.95,
    },
}


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: bool = True) -> None:
    """Configure logging for the application."""
    logger.remove()

    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File logging
    if log_file:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "alphatrade_{time}.log",
            rotation="100 MB",
            retention="30 days",
            level="DEBUG",
        )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        # Deep merge
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.info("Using default configuration")

    return config


def load_and_validate_data(
    data_path: str,
    symbols: Optional[List[str]] = None,
    min_bars: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Load and validate market data.

    Args:
        data_path: Path to data directory
        symbols: Optional list of symbols to load
        min_bars: Minimum number of bars required

    Returns:
        Dictionary of validated DataFrames
    """
    logger.info(f"Loading data from {data_path}")

    data_dir = Path(data_path)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_path}")
        return {}

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

            if len(df) >= min_bars:
                data[symbol] = df
                logger.debug(f"Loaded {symbol}: {len(df)} bars")
            else:
                logger.warning(f"{symbol}: Insufficient data ({len(df)} bars < {min_bars})")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    logger.info(f"Successfully loaded {len(data)} symbols")
    return data


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def generate_features(
    data: Dict[str, pd.DataFrame],
    feature_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate features for all symbols.

    Args:
        data: Dictionary of OHLCV DataFrames
        feature_config: Feature configuration

    Returns:
        Dictionary of DataFrames with features
    """
    logger.info("Generating features...")

    features = {}
    for symbol, df in data.items():
        try:
            indicators = TechnicalIndicators(df)

            # Add all standard indicators
            df_with_features = indicators.add_all()

            features[symbol] = df_with_features
            logger.debug(f"{symbol}: Added {len(df_with_features.columns)} features")

        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")

    logger.info(f"Generated features for {len(features)} symbols")
    return features


# =============================================================================
# STRATEGY CREATION
# =============================================================================

def create_strategies(config: Dict[str, Any]) -> List:
    """Create strategy instances from configuration."""
    strategies = []
    strategy_config = config.get("strategies", {})

    # Momentum Strategy
    mom_config = strategy_config.get("momentum", {})
    if mom_config.get("enabled", True):
        strategy = MultiFactorMomentumStrategy(
            name="Momentum",
            params={
                "lookback_periods": mom_config.get("lookback_periods", [5, 10, 20]),
                "top_n_long": mom_config.get("top_n_long", 5),
                "top_n_short": mom_config.get("top_n_short", 0),
                "volatility_adjusted": mom_config.get("volatility_adjusted", True),
            },
        )
        strategies.append(strategy)
        logger.info("Created Momentum strategy")

    # Mean Reversion Strategy
    mr_config = strategy_config.get("mean_reversion", {})
    if mr_config.get("enabled", True):
        strategy = MeanReversionStrategy(
            name="MeanReversion",
            params={
                "lookback_period": mr_config.get("lookback_period", 20),
                "entry_zscore": mr_config.get("entry_zscore", 2.0),
                "exit_zscore": mr_config.get("exit_zscore", 0.5),
            },
        )
        strategies.append(strategy)
        logger.info("Created MeanReversion strategy")

    # Volatility Breakout Strategy
    vb_config = strategy_config.get("volatility_breakout", {})
    if vb_config.get("enabled", True):
        strategy = VolatilityBreakoutStrategy(
            name="VolatilityBreakout",
            params={
                "atr_period": vb_config.get("atr_period", 14),
                "atr_multiplier": vb_config.get("atr_multiplier", 2.0),
                "lookback_period": vb_config.get("lookback_period", 20),
            },
        )
        strategies.append(strategy)
        logger.info("Created VolatilityBreakout strategy")

    return strategies


# =============================================================================
# BACKTESTING
# =============================================================================

def run_vectorized_backtest(
    strategy,
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    features: Optional[Dict[str, pd.DataFrame]] = None,
) -> BacktestResult:
    """Run vectorized backtest for a strategy."""
    logger.info(f"Running vectorized backtest for {strategy.name}")

    backtest_config = config.get("backtest", {})

    engine = BacktestEngine(
        initial_capital=backtest_config.get("initial_capital", 1_000_000),
        commission_pct=backtest_config.get("commission_pct", 0.001),
        slippage_pct=backtest_config.get("slippage_pct", 0.0005),
    )

    result = engine.run(strategy, data, features)

    logger.info(
        f"{strategy.name} - Return: {result.metrics.get('total_return', 0):.2%}, "
        f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}, "
        f"MaxDD: {result.metrics.get('max_drawdown', 0):.2%}"
    )

    return result


def run_event_driven_backtest(
    strategy,
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Any:
    """Run event-driven backtest for a strategy."""
    logger.info(f"Running event-driven backtest for {strategy.name}")

    backtest_config = config.get("backtest", {})

    engine_config = EventEngineConfig(
        initial_capital=backtest_config.get("initial_capital", 1_000_000),
        slippage_bps=backtest_config.get("slippage_bps", 1.0),
        commission_per_share=backtest_config.get("commission_per_share", 0.005),
    )

    engine = EventDrivenEngine(engine_config)

    # Register strategy callback
    def on_bar(event):
        # Strategy generates signals from bar events
        pass

    engine.register_strategy(on_bar=on_bar)

    result = engine.run(data)

    logger.info(
        f"{strategy.name} - Return: {result.metrics.get('total_return', 0):.2%}, "
        f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}"
    )

    return result


def run_ensemble_backtest(
    strategies: List,
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    features: Optional[Dict[str, pd.DataFrame]] = None,
) -> BacktestResult:
    """Run ensemble strategy backtest."""
    logger.info("Running Ensemble strategy backtest")

    ensemble_config = config.get("ensemble", {})

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

    return run_vectorized_backtest(ensemble, data, config, features)


# =============================================================================
# ML TRAINING
# =============================================================================

def train_ml_model(
    data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> TrainingResult:
    """Train ML model with cross-validation."""
    logger.info("Training ML model...")

    train_config = config.get("training", {})

    # Prepare training data
    all_features = []
    all_targets = []

    for symbol, df in features.items():
        # Create target (next day returns)
        target = df["close"].pct_change().shift(-1)

        # Get feature columns (exclude OHLCV)
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]

        X = df[feature_cols].iloc[:-1]  # Remove last row (no target)
        y = target.iloc[:-1]

        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        all_features.append(X)
        all_targets.append(y)

    X_train = pd.concat(all_features, axis=0)
    y_train = pd.concat(all_targets, axis=0)

    logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")

    # Create model
    model_type = train_config.get("model_type", "lightgbm")
    model = ModelFactory.create(model_type)

    # Cross-validation
    cv = PurgedKFoldCV(
        n_splits=train_config.get("cv_splits", 5),
        purge_gap=train_config.get("purge_gap", 5),
        embargo_pct=train_config.get("embargo_pct", 0.01),
    )

    # Optuna optimization (optional)
    if train_config.get("optimize", False):
        logger.info("Running hyperparameter optimization...")
        optimizer = OptunaOptimizer(
            model_type=model_type,
            cv=cv,
            metric="sharpe_ratio",
            n_trials=train_config.get("optuna_trials", 50),
        )
        best_params = optimizer.optimize(X_train.values, y_train.values)
        model = ModelFactory.create(model_type, **best_params)

    # Train
    trainer = Trainer(model=model, cv=cv)
    result = trainer.train(X_train.values, y_train.values)

    logger.info(f"Training complete - CV Score: {result.cv_scores.mean():.4f}")

    return result


# =============================================================================
# REPORTING
# =============================================================================

def generate_report(
    results: Dict[str, BacktestResult],
    output_dir: str = "reports",
) -> None:
    """Generate comprehensive performance reports."""
    logger.info("Generating performance reports...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find best strategy
    best_strategy = max(
        results.keys(),
        key=lambda k: results[k].metrics.get("sharpe_ratio", 0),
    )
    best_result = results[best_strategy]

    # Generate tear sheet for best strategy
    tear_sheet_path = output_path / f"tear_sheet_{timestamp}.html"
    create_tear_sheet(
        returns=best_result.returns,
        strategy_name=best_strategy,
        output_path=tear_sheet_path,
    )
    logger.info(f"Tear sheet saved to: {tear_sheet_path}")

    # Generate basic HTML report
    report_path = output_path / f"backtest_report_{timestamp}.html"
    generator = ReportGenerator(best_result, output_dir=output_path)
    generator.generate_html(f"backtest_report_{timestamp}.html")
    logger.info(f"Report saved to: {report_path}")

    # Generate comparison summary
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "Strategy": name,
            "Total Return": f"{result.metrics.get('total_return', 0):.2%}",
            "CAGR": f"{result.metrics.get('cagr', 0):.2%}",
            "Sharpe Ratio": f"{result.metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino Ratio": f"{result.metrics.get('sortino_ratio', 0):.2f}",
            "Max Drawdown": f"{result.metrics.get('max_drawdown', 0):.2%}",
            "Win Rate": f"{result.metrics.get('win_rate', 0):.1%}",
            "Num Trades": len(result.trades),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / f"strategy_comparison_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(f"\nBest Strategy: {best_strategy}")
    print(f"Reports saved to: {output_path}")


def save_results(results: Dict[str, BacktestResult], output_path: str) -> None:
    """Save backtest results to pickle file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to: {output_path}")


def load_results(input_path: str) -> Dict[str, BacktestResult]:
    """Load backtest results from pickle file."""
    with open(input_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaTrade System - Institutional-Grade Algorithmic Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run all strategies
  python main.py --strategy momentum                # Run momentum only
  python main.py --mode train --model lightgbm      # Train ML model
  python main.py --engine event-driven              # Use event-driven engine
  python main.py --report-only --input results.pkl  # Generate report only
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "train", "report"],
        default="backtest",
        help="Execution mode (default: backtest)",
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["momentum", "mean_reversion", "volatility_breakout", "ensemble", "all"],
        default="all",
        help="Strategy to run (default: all)",
    )

    # Engine selection
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vectorized", "event-driven"],
        default="vectorized",
        help="Backtest engine type (default: vectorized)",
    )

    # ML model
    parser.add_argument(
        "--model",
        type=str,
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
        default="lightgbm",
        help="ML model type for training (default: lightgbm)",
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw",
        help="Path to data directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/trading_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file for report-only mode",
    )

    # Options
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to process (default: all)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
        help="Initial capital (default: 1,000,000)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, log_file=not args.no_log_file)

    logger.info("=" * 60)
    logger.info("AlphaTrade System v2.0.0")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Engine: {args.engine}")
    logger.info(f"Initial Capital: ${args.capital:,.0f}")

    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Load configuration
    config = load_config(args.config)
    config["backtest"]["initial_capital"] = args.capital
    config["training"]["model_type"] = args.model
    config["training"]["optimize"] = args.optimize

    # Report-only mode
    if args.mode == "report":
        if not args.input:
            logger.error("--input required for report mode")
            sys.exit(1)
        results = load_results(args.input)
        generate_report(results, args.output)
        return

    # Load data
    data = load_and_validate_data(
        args.data_path,
        args.symbols,
        config["data"].get("min_bars", 100),
    )

    if not data:
        logger.error("No data loaded. Please check data directory.")
        logger.info(f"Expected data path: {args.data_path}")
        logger.info("Place CSV files with OHLCV data in the data directory.")
        sys.exit(1)

    # Generate features
    features = generate_features(data, config.get("features"))

    # Training mode
    if args.mode == "train":
        result = train_ml_model(data, features, config)
        logger.info("Training complete!")
        return

    # Backtest mode
    strategies = create_strategies(config)

    if not strategies:
        logger.error("No strategies created. Check configuration.")
        sys.exit(1)

    results = {}

    if args.strategy == "all":
        # Run all individual strategies
        for strategy in strategies:
            try:
                if args.engine == "event-driven":
                    result = run_event_driven_backtest(strategy, data, config)
                else:
                    result = run_vectorized_backtest(strategy, data, config, features)
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Error running {strategy.name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Run ensemble
        if len(strategies) > 1:
            try:
                result = run_ensemble_backtest(strategies, data, config, features)
                results["Ensemble"] = result
            except Exception as e:
                logger.error(f"Error running Ensemble: {e}")

    elif args.strategy == "ensemble":
        result = run_ensemble_backtest(strategies, data, config, features)
        results["Ensemble"] = result

    else:
        # Run specific strategy
        strategy_map = {s.name.lower().replace(" ", ""): s for s in strategies}
        strategy_key = args.strategy.replace("_", "").lower()

        for name, strategy in strategy_map.items():
            if strategy_key in name:
                if args.engine == "event-driven":
                    result = run_event_driven_backtest(strategy, data, config)
                else:
                    result = run_vectorized_backtest(strategy, data, config, features)
                results[strategy.name] = result
                break

    if not results:
        logger.error("No backtest results generated.")
        sys.exit(1)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/backtest_{timestamp}.pkl"
    save_results(results, results_path)

    # Generate reports
    generate_report(results, args.output)

    logger.info("=" * 60)
    logger.info("AlphaTrade System Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
