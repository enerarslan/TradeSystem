#!/usr/bin/env python3
"""
Algo Trading Platform - Main Entry Point
=========================================

JPMorgan-level algorithmic trading platform.
Supports backtesting, paper trading, and live trading.

Usage:
    python main.py                    # Interactive menu
    python main.py backtest           # Run backtest
    python main.py paper              # Start paper trading
    python main.py api                # Start API server

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_environment() -> None:
    """Setup environment and create necessary directories."""
    directories = [
        "data/storage",
        "data/processed",
        "data/cache",
        "models/artifacts",
        "backtesting/reports",
        "logs",
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_backtest(args: argparse.Namespace) -> int:
    """Run backtesting mode."""
    from config.settings import get_settings, get_logger, configure_logging
    
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("ALGO TRADING PLATFORM - BACKTEST MODE")
    logger.info("=" * 60)
    
    try:
        from scripts.run_backtest import main as backtest_main
        return backtest_main(args)
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1


def run_paper_trading(args: argparse.Namespace) -> int:
    """Run paper trading mode."""
    from config.settings import get_settings, get_logger, configure_logging
    
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("ALGO TRADING PLATFORM - PAPER TRADING MODE")
    logger.info("=" * 60)
    
    try:
        # TODO: Implement paper trading in Phase 5
        logger.warning("Paper trading not yet implemented (Phase 5)")
        logger.info("Please use backtest mode for now")
        return 0
    except Exception as e:
        logger.exception(f"Paper trading failed: {e}")
        return 1


def run_api_server(args: argparse.Namespace) -> int:
    """Run API server."""
    from config.settings import get_settings, get_logger, configure_logging
    
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("ALGO TRADING PLATFORM - API SERVER")
    logger.info("=" * 60)
    
    try:
        import uvicorn
        
        host = args.host if hasattr(args, 'host') else "0.0.0.0"
        port = args.port if hasattr(args, 'port') else 8000
        
        logger.info(f"Starting API server on {host}:{port}")
        
        # TODO: Implement API in Phase 6
        logger.warning("API server not yet implemented (Phase 6)")
        return 0
        
        # Uncomment when API is ready:
        # uvicorn.run("api.main:app", host=host, port=port, reload=settings.debug)
        
    except Exception as e:
        logger.exception(f"API server failed: {e}")
        return 1


def run_interactive() -> int:
    """Run interactive menu."""
    from config.settings import get_settings, get_logger, configure_logging
    
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)
    
    print("\n" + "=" * 60)
    print("  ALGO TRADING PLATFORM")
    print("  JPMorgan-Level Algorithmic Trading System")
    print("=" * 60)
    print(f"\n  Version: {settings.app_version}")
    print(f"  Mode: {settings.trading_mode.value}")
    print(f"  Debug: {settings.debug}")
    print("\n" + "-" * 60)
    print("\n  Available Commands:")
    print("    1. Run Backtest")
    print("    2. Paper Trading (Coming Soon)")
    print("    3. Start API Server (Coming Soon)")
    print("    4. Run Tests")
    print("    5. Show System Info")
    print("    0. Exit")
    print("\n" + "-" * 60)
    
    while True:
        try:
            choice = input("\n  Enter choice [0-5]: ").strip()
            
            if choice == "0":
                print("\n  Goodbye!\n")
                return 0
            elif choice == "1":
                args = argparse.Namespace()
                return run_backtest(args)
            elif choice == "2":
                args = argparse.Namespace()
                return run_paper_trading(args)
            elif choice == "3":
                args = argparse.Namespace()
                return run_api_server(args)
            elif choice == "4":
                import subprocess
                result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
                return result.returncode
            elif choice == "5":
                show_system_info()
            else:
                print("  Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user. Goodbye!\n")
            return 0


def show_system_info() -> None:
    """Display system information."""
    from config.settings import get_settings
    
    settings = get_settings()
    
    print("\n" + "-" * 60)
    print("  SYSTEM INFORMATION")
    print("-" * 60)
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"\n  Configuration:")
    print(f"    Trading Mode: {settings.trading_mode.value}")
    print(f"    Debug: {settings.debug}")
    print(f"    Log Level: {settings.log_level.value}")
    print(f"    Timezone: {settings.timezone}")
    print(f"\n  Data Paths:")
    print(f"    Storage: {settings.data.storage_path}")
    print(f"    Processed: {settings.data.processed_path}")
    print(f"    Cache: {settings.data.cache_path}")
    print(f"\n  Backtest Settings:")
    print(f"    Initial Capital: ${settings.backtest.initial_capital:,.2f}")
    print(f"    Commission: {settings.backtest.commission_pct:.4%}")
    print(f"    Slippage: {settings.backtest.slippage_pct:.4%}")
    
    # Check available data
    storage_path = Path(settings.data.storage_path)
    if storage_path.exists():
        csv_files = list(storage_path.glob("*.csv"))
        print(f"\n  Available Data Files: {len(csv_files)}")
        for f in csv_files[:5]:
            print(f"    - {f.name}")
        if len(csv_files) > 5:
            print(f"    ... and {len(csv_files) - 5} more")
    
    print("-" * 60)


def main() -> int:
    """Main entry point."""
    setup_environment()
    
    parser = argparse.ArgumentParser(
        description="Algo Trading Platform - JPMorgan-Level Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Interactive menu
  python main.py backtest            Run backtest
  python main.py backtest --symbol AAPL --strategy trend_following
  python main.py paper               Start paper trading
  python main.py api --port 8080     Start API server
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtesting")
    bt_parser.add_argument("--symbol", "-s", type=str, nargs="+", 
                          help="Symbols to backtest")
    bt_parser.add_argument("--strategy", "-t", type=str, default="trend_following",
                          help="Strategy to use")
    bt_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--capital", "-c", type=float, default=100000,
                          help="Initial capital")
    bt_parser.add_argument("--output", "-o", type=str, help="Output report path")
    
    # Paper trading command
    paper_parser = subparsers.add_parser("paper", help="Run paper trading")
    paper_parser.add_argument("--symbol", "-s", type=str, nargs="+",
                             help="Symbols to trade")
    paper_parser.add_argument("--strategy", "-t", type=str, default="trend_following",
                             help="Strategy to use")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0",
                           help="Host to bind")
    api_parser.add_argument("--port", "-p", type=int, default=8000,
                           help="Port to bind")
    
    args = parser.parse_args()
    
    if args.command is None:
        return run_interactive()
    elif args.command == "backtest":
        return run_backtest(args)
    elif args.command == "paper":
        return run_paper_trading(args)
    elif args.command == "api":
        return run_api_server(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())