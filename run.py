#!/usr/bin/env python3
"""
AlphaTrade System - Unified Launcher

This is the SINGLE entry point for the entire trading system.
All modules are REQUIRED and fully integrated.

Usage:
    python run.py                    # Quick start - full pipeline
    python run.py setup              # First time setup
    python run.py backtest           # Run backtest only
    python run.py train              # Train ML model
    python run.py train --deep       # Train deep learning
    python run.py report             # Generate reports
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def check_installation():
    """Check if all required packages are installed."""
    required = [
        "pandas", "numpy", "polars", "scikit-learn",
        "lightgbm", "xgboost", "catboost", "torch",
        "optuna", "mlflow", "plotly", "loguru",
        "psycopg2", "fredapi", "pydantic",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    return missing


def setup():
    """First time setup."""
    print("=" * 60)
    print("  ALPHATRADE SYSTEM - SETUP")
    print("=" * 60)

    # Check missing packages
    missing = check_installation()
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstalling dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    else:
        print("\nAll dependencies installed!")

    # Create directories
    dirs = ["data/raw", "data/cache", "data/processed", "logs", "reports", "results", "models"]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("\nDirectories created!")

    # Check for data
    data_files = list((PROJECT_ROOT / "data/raw").glob("*.csv"))
    if not data_files:
        print("\nNo data found. Generating sample data...")
        subprocess.run([sys.executable, "scripts/generate_sample_data.py"], check=True)

    print("\nSetup complete! Run: python run.py")


def main():
    """Main launcher."""
    if len(sys.argv) < 2:
        # Default: full pipeline
        subprocess.run([sys.executable, "main.py", "--mode", "full"])
        return

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if command == "setup":
        setup()

    elif command == "backtest":
        args = [sys.executable, "main.py", "--mode", "backtest"] + extra_args
        subprocess.run(args)

    elif command == "train":
        if "--deep" in extra_args:
            extra_args.remove("--deep")
            args = [sys.executable, "main.py", "--mode", "train", "--deep-learning"] + extra_args
        else:
            args = [sys.executable, "main.py", "--mode", "train"] + extra_args
        subprocess.run(args)

    elif command == "report":
        args = [sys.executable, "main.py", "--mode", "report"] + extra_args
        subprocess.run(args)

    elif command == "full":
        args = [sys.executable, "main.py", "--mode", "full"] + extra_args
        subprocess.run(args)

    elif command == "help":
        print(__doc__)
        print("\nCommands:")
        print("  setup      - First time setup")
        print("  backtest   - Run backtesting")
        print("  train      - Train ML model")
        print("  train --deep - Train deep learning model")
        print("  report     - Generate reports")
        print("  full       - Full pipeline (default)")
        print("\nOptions:")
        print("  --engine event-driven  - Use event-driven engine")
        print("  --model xgboost       - Use specific ML model")
        print("  --optimize            - Run Optuna optimization")
        print("  --capital 500000      - Set initial capital")

    else:
        print(f"Unknown command: {command}")
        print("Run 'python run.py help' for usage")


if __name__ == "__main__":
    main()
