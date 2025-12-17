#!/usr/bin/env python3
"""
AlphaTrade System Orchestrator - Single Command Entry Point

This script provides a unified, single-command interface to run the complete
AlphaTrade pipeline with institutional-grade defaults and validation.

JPMorgan-level requirements:
- Single command: python orchestrate.py
- Pre-flight data validation
- Complete pipeline execution
- Comprehensive reporting

Usage:
    python orchestrate.py                    # Full pipeline with defaults
    python orchestrate.py --mode backtest    # Backtest only
    python orchestrate.py --mode train       # Train models only
    python orchestrate.py --validate-only    # Data validation only
    python orchestrate.py --quick            # Quick run with reduced data
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrate")


def banner():
    """Print banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗ ████████╗██████╗      ║
    ║    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗╚══██╔══╝██╔══██╗     ║
    ║    ███████║██║     ██████╔╝███████║███████║   ██║   ██████╔╝     ║
    ║    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║   ██║   ██╔══██╗     ║
    ║    ██║  ██║███████╗██║     ██║  ██║██║  ██║   ██║   ██║  ██║     ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝     ║
    ║                                                                   ║
    ║           Institutional-Grade Trading System Orchestrator         ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


def load_config(config_path: str = "config/trading_config.yaml") -> Dict[str, Any]:
    """Load configuration with institutional defaults."""
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = {}

    # Apply institutional defaults
    defaults = {
        "data": {
            "path": "data/raw",
            "min_bars": 500,
            "validate": True,
        },
        "training": {
            "model_type": "lightgbm",
            "cv_type": "combinatorial_purged",
            "cv_splits": 6,
            "n_test_splits": 2,
            "purge_gap": "auto",
            "embargo_pct": "auto",
            "mlflow_tracking": True,
        },
        "backtest": {
            "initial_capital": 1_000_000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "walk_forward": {"enabled": True},
        },
        "features": {
            "technical": True,
            "fractional_diff": True,
            "leakage_check": "strict",
        },
    }

    # Deep merge defaults
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(defaults, config)


def validate_environment() -> List[str]:
    """Validate Python environment and dependencies."""
    issues = []

    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required, got {sys.version_info}")

    # Check critical dependencies
    required = [
        "pandas", "numpy", "scipy", "sklearn",
        "yaml", "loguru",
    ]
    optional = [
        ("lightgbm", "ML models"),
        ("optuna", "hyperparameter optimization"),
        ("mlflow", "experiment tracking"),
        ("plotly", "interactive reports"),
        ("psycopg2", "TimescaleDB support"),
    ]

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"Missing required package: {pkg}")

    for pkg, desc in optional:
        try:
            __import__(pkg)
            logger.debug(f"Optional: {pkg} available ({desc})")
        except ImportError:
            logger.warning(f"Optional: {pkg} not installed ({desc})")

    return issues


def validate_data(data_path: str) -> Dict[str, Any]:
    """
    Validate data files before running pipeline.

    Returns:
        Validation report dictionary
    """
    logger.info("=" * 60)
    logger.info("PRE-FLIGHT DATA VALIDATION")
    logger.info("=" * 60)

    data_dir = Path(data_path)
    report = {
        "valid": True,
        "symbols": [],
        "issues": [],
        "warnings": [],
        "total_bars": 0,
    }

    if not data_dir.exists():
        report["valid"] = False
        report["issues"].append(f"Data directory not found: {data_path}")
        logger.error(f"Data directory not found: {data_path}")
        return report

    # Find data files
    files = list(data_dir.glob("*_15min.csv"))
    if not files:
        files = list(data_dir.glob("*.csv"))
        if not files:
            report["valid"] = False
            report["issues"].append("No CSV data files found")
            logger.error("No CSV data files found")
            return report
        logger.warning("Using fallback file format (*.csv instead of *_15min.csv)")
        report["warnings"].append("Using fallback file format")

    logger.info(f"Found {len(files)} data files")

    # Validate each file
    import pandas as pd

    for file_path in files:
        symbol = file_path.stem.replace("_15min", "")
        try:
            # Read file
            df = pd.read_csv(file_path)

            # Check required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [c for c in required_cols if c.lower() not in [col.lower() for col in df.columns]]
            if missing_cols:
                report["issues"].append(f"{symbol}: Missing columns {missing_cols}")
                continue

            # Check date column
            date_col = None
            for col in ["timestamp", "date", "Date", "Timestamp"]:
                if col in df.columns:
                    date_col = col
                    break

            if not date_col and df.columns[0] not in required_cols:
                date_col = df.columns[0]

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                report["warnings"].append(f"{symbol}: No date column found")

            # Check for NaN
            nan_pct = df.isna().sum().sum() / df.size * 100
            if nan_pct > 5:
                report["warnings"].append(f"{symbol}: {nan_pct:.1f}% missing values")

            # Check data length
            n_bars = len(df)
            report["total_bars"] += n_bars
            report["symbols"].append({
                "symbol": symbol,
                "bars": n_bars,
                "start": str(df[date_col].min()) if date_col else "N/A",
                "end": str(df[date_col].max()) if date_col else "N/A",
            })

            if n_bars < 100:
                report["warnings"].append(f"{symbol}: Only {n_bars} bars (need 100+)")

        except Exception as e:
            report["issues"].append(f"{symbol}: Error reading file - {e}")

    # Summary
    logger.info(f"Validated {len(report['symbols'])} symbols")
    logger.info(f"Total bars: {report['total_bars']:,}")

    if report["issues"]:
        report["valid"] = False
        logger.error(f"VALIDATION FAILED: {len(report['issues'])} critical issues")
        for issue in report["issues"]:
            logger.error(f"  - {issue}")

    if report["warnings"]:
        logger.warning(f"{len(report['warnings'])} warnings:")
        for warning in report["warnings"]:
            logger.warning(f"  - {warning}")

    if report["valid"]:
        logger.info("DATA VALIDATION PASSED")

    return report


def generate_sample_data_if_needed(data_path: str) -> bool:
    """Generate sample data if none exists."""
    data_dir = Path(data_path)

    if data_dir.exists() and list(data_dir.glob("*.csv")):
        return True  # Data exists

    logger.info("No data found. Generating sample data...")

    try:
        from scripts.generate_sample_data import main as generate_data
        generate_data()
        return True
    except ImportError:
        logger.error("Could not import sample data generator")
        return False
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return False


def run_pipeline(
    config: Dict[str, Any],
    mode: str = "full",
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete AlphaTrade pipeline.

    Args:
        config: Configuration dictionary
        mode: "full", "train", "backtest", or "validate"
        quick: Use reduced data for quick testing

    Returns:
        Pipeline results
    """
    logger.info("=" * 60)
    logger.info(f"RUNNING PIPELINE: mode={mode}, quick={quick}")
    logger.info("=" * 60)

    results = {
        "status": "started",
        "start_time": datetime.now().isoformat(),
        "mode": mode,
    }

    try:
        # Import main module
        from main import (
            load_data,
            generate_features,
            train_ml_model,
            run_backtest,
            create_strategies,
        )

        # Load data
        logger.info("Step 1/4: Loading data...")
        data_path = config["data"]["path"]
        min_bars = 100 if quick else config["data"].get("min_bars", 500)

        data = load_data(
            data_path,
            min_bars=min_bars,
            config=config,
        )

        if not data:
            raise ValueError("No data loaded")

        results["symbols_loaded"] = len(data)
        logger.info(f"Loaded {len(data)} symbols")

        if quick:
            # Limit symbols for quick run
            symbols = list(data.keys())[:5]
            data = {s: data[s] for s in symbols}
            logger.info(f"Quick mode: Limited to {len(data)} symbols")

        # Generate features
        logger.info("Step 2/4: Generating features...")
        features, pipelines = generate_features(data, config)
        results["features_generated"] = True

        if mode == "train" or mode == "full":
            # Train ML model
            logger.info("Step 3/4: Training ML model...")
            training_result = train_ml_model(data, features, config)
            results["training"] = {
                "cv_mean": float(training_result.cv_scores.mean()),
                "cv_std": float(training_result.cv_scores.std()),
            }
            logger.info(
                f"Training complete: CV={training_result.cv_scores.mean():.4f} "
                f"(+/- {training_result.cv_scores.std():.4f})"
            )

        if mode == "backtest" or mode == "full":
            # Run backtest
            logger.info("Step 4/4: Running backtest...")
            strategies = create_strategies(config)

            backtest_results = {}
            for strategy in strategies:
                result = run_backtest(
                    strategy=strategy,
                    data=data,
                    features=features,
                    config=config,
                )
                backtest_results[strategy.name] = {
                    "total_return": float(result.metrics.get("total_return", 0)),
                    "sharpe_ratio": float(result.metrics.get("sharpe_ratio", 0)),
                    "max_drawdown": float(result.metrics.get("max_drawdown", 0)),
                }
                logger.info(
                    f"{strategy.name}: Return={result.metrics.get('total_return', 0):.2%}, "
                    f"Sharpe={result.metrics.get('sharpe_ratio', 0):.2f}"
                )

            results["backtest"] = backtest_results

        results["status"] = "completed"
        results["end_time"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)

    return results


def print_summary(results: Dict[str, Any]):
    """Print pipeline summary."""
    print("\n")
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    if results["status"] == "completed":
        print(f"Status: SUCCESS")
        print(f"Mode: {results['mode']}")
        print(f"Symbols: {results.get('symbols_loaded', 'N/A')}")

        if "training" in results:
            print(f"\nTraining Results:")
            print(f"  CV Score: {results['training']['cv_mean']:.4f} (+/- {results['training']['cv_std']:.4f})")

        if "backtest" in results:
            print(f"\nBacktest Results:")
            for strategy, metrics in results["backtest"].items():
                print(f"  {strategy}:")
                print(f"    Return: {metrics['total_return']:.2%}")
                print(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
                print(f"    Max DD: {metrics['max_drawdown']:.2%}")
    else:
        print(f"Status: FAILED")
        print(f"Error: {results.get('error', 'Unknown')}")

    print("=" * 60)


def main():
    """Main entry point."""
    banner()

    parser = argparse.ArgumentParser(
        description="AlphaTrade System Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "backtest", "validate"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--config",
        default="config/trading_config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override data path from config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data, don't run pipeline",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with reduced data for testing",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate sample data if none exists",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)
    if args.data_path:
        config["data"]["path"] = args.data_path

    data_path = config["data"]["path"]

    # Validate environment
    logger.info("Validating environment...")
    env_issues = validate_environment()
    if env_issues:
        for issue in env_issues:
            logger.error(f"Environment issue: {issue}")
        sys.exit(1)
    logger.info("Environment OK")

    # Generate sample data if requested/needed
    if args.generate_data:
        generate_sample_data_if_needed(data_path)

    # Validate data
    validation = validate_data(data_path)
    if not validation["valid"]:
        logger.error("Data validation failed. Run with --generate-data to create sample data.")
        sys.exit(1)

    if args.validate_only or args.mode == "validate":
        logger.info("Validation-only mode. Exiting.")
        return

    # Run pipeline
    results = run_pipeline(config, mode=args.mode, quick=args.quick)

    # Print summary
    print_summary(results)

    # Exit code
    sys.exit(0 if results["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
