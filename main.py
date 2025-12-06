#!/usr/bin/env python3
"""
============================================================================
ALPHATRADE - MAIN ENTRY POINT
============================================================================
JPMorgan-Style Enterprise Trading System

Usage:
    python main.py --mode backtest           # Run backtests
    python main.py --mode paper              # Paper trading (simulated)
    python main.py --mode live               # Live trading (CAUTION!)
    python main.py --mode dashboard          # Start monitoring dashboard
    python main.py --help                    # Show all options

============================================================================
"""

import asyncio
import argparse
import sys
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import log
from config.settings import settings
from config.yaml_config import get_config


class AlphaTradeApp:
    """
    Main Application Controller.
    Manages all trading modes and system lifecycle.
    """
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.mode = "backtest"
        
        # Components (initialized based on mode)
        self.event_bus = None
        self.data_feed = None
        self.strategy_orchestrator = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.execution_engine = None
        
    def print_banner(self):
        """Print startup banner"""
        print("\n" + "=" * 70)
        print("   🏦 ALPHATRADE - Enterprise Trading System")
        print("   JPMorgan-Style Quantitative Trading Platform")
        print("=" * 70)
        print(f"   Version    : {self.config.version}")
        print(f"   Environment: {self.config.environment}")
        print(f"   Mode       : {self.mode.upper()}")
        print(f"   Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
    
    async def initialize(self, mode: str):
        """Initialize system components based on mode"""
        self.mode = mode
        log.info(f"🚀 Initializing AlphaTrade in {mode.upper()} mode...")
        
        # Initialize Event Bus
        from core.bus import init_event_bus
        self.event_bus = await init_event_bus()
        
        # Initialize Risk Manager
        from risk.core import EnterpriseRiskManager, RiskLimitConfig
        self.risk_manager = EnterpriseRiskManager(config=RiskLimitConfig())
        
        # Initialize Portfolio Manager
        from execution.portfolio import PortfolioManager
        self.portfolio_manager = PortfolioManager(
            initial_balance=self.config.trading.initial_capital
        )
        
        # Mode-specific initialization
        if mode == "backtest":
            await self._init_backtest_mode()
        elif mode == "paper":
            await self._init_paper_mode()
        elif mode == "live":
            await self._init_live_mode()
        elif mode == "dashboard":
            await self._init_dashboard_mode()
        
        log.success("✅ System initialized successfully")
    
    async def _init_backtest_mode(self):
        """Initialize for backtest mode"""
        log.info("📊 Backtest mode - Using historical data")
        from data.csv_loader import LocalCSVLoader
        self.data_feed = LocalCSVLoader(
            storage_path=self.config.data_feed.storage_path,
            validate_data=True
        )
    
    async def _init_paper_mode(self):
        """Initialize for paper trading mode"""
        log.info("📝 Paper trading mode - Simulated execution")
        # TODO: Initialize paper trading components
        log.warning("⚠️ Paper trading mode not fully implemented yet")
    
    async def _init_live_mode(self):
        """Initialize for live trading mode"""
        log.warning("🔴 LIVE TRADING MODE - Real money at risk!")
        log.warning("⚠️ Live trading requires broker integration")
        # TODO: Initialize broker connections
        raise NotImplementedError("Live trading requires broker integration")
    
    async def _init_dashboard_mode(self):
        """Initialize monitoring dashboard"""
        log.info("📈 Dashboard mode - Starting monitoring server")
        # TODO: Initialize FastAPI dashboard
        log.warning("⚠️ Dashboard not implemented yet")
    
    async def run_backtest(
        self,
        symbol: Optional[str] = None,
        mode: str = "single"
    ):
        """Run backtest"""
        from run_backtest import (
            run_single_backtest,
            run_portfolio_backtest,
            run_walk_forward,
            run_all_stocks_sequential
        )
        
        capital = self.config.trading.initial_capital
        
        if mode == "single":
            symbol = symbol or "AAPL"
            return await run_single_backtest(symbol, capital)
        elif mode == "portfolio":
            return await run_portfolio_backtest(
                capital=capital,
                max_positions=self.config.risk.max_open_positions,
                allocation="risk_parity"
            )
        elif mode == "walkforward":
            symbol = symbol or "AAPL"
            return await run_walk_forward(symbol, capital, 180, 30)
        elif mode == "sequential":
            return await run_all_stocks_sequential(capital)
    
    async def shutdown(self):
        """Graceful shutdown"""
        log.info("🛑 Shutting down AlphaTrade...")
        self.running = False
        
        if self.event_bus:
            from core.bus import shutdown_event_bus
            await shutdown_event_bus()
        
        log.success("✅ Shutdown complete")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            log.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AlphaTrade - JPMorgan-Style Trading System"
    )
    
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live", "dashboard"],
        default="backtest",
        help="Trading mode"
    )
    
    parser.add_argument(
        "--backtest-type",
        choices=["single", "portfolio", "walkforward", "sequential"],
        default="single",
        help="Backtest type (only for backtest mode)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol for single backtest"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital"
    )
    
    args = parser.parse_args()
    
    # Create app
    app = AlphaTradeApp()
    app.print_banner()
    app.setup_signal_handlers()
    
    try:
        # Initialize
        await app.initialize(args.mode)
        
        # Run based on mode
        if args.mode == "backtest":
            await app.run_backtest(
                symbol=args.symbol,
                mode=args.backtest_type
            )
        elif args.mode == "paper":
            log.info("Starting paper trading loop...")
            app.running = True
            while app.running:
                await asyncio.sleep(1)
        elif args.mode == "live":
            log.error("Live trading not implemented")
        elif args.mode == "dashboard":
            log.error("Dashboard not implemented")
        
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())