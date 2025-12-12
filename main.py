"""
AlphaTrade System - Main Orchestrator
JPMorgan-Level Institutional Trading Platform

This is the main entry point for the trading system.
Coordinates all components for live and paper trading.
"""

import asyncio
import signal
import sys
import os
import argparse
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger, get_audit_logger, setup_logging
from src.utils.helpers import load_config, Timer

from src.data.loader import DataLoader, MultiAssetLoader
from src.data.preprocessor import DataPreprocessor
from src.data.live_feed import WebSocketManager

from src.features.builder import FeatureBuilder, FeaturePipeline
from src.features.regime import RegimeDetector

from src.models.training import ModelTrainer, WalkForwardValidator
from src.models.ensemble import EnsembleModel

from src.strategy.ml_strategy import MLStrategy, EnsembleMLStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.mean_reversion import MeanReversionStrategy

from src.risk.risk_manager import RiskManager, RiskLimits
from src.risk.position_sizer import VolatilityPositionSizer, RiskParityPositionSizer
from src.risk.portfolio import PortfolioManager

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.metrics import MetricsCalculator, ReportGenerator

from src.execution.broker_api import BrokerFactory, AlpacaBroker
from src.execution.order_manager import OrderManager
from src.execution.executor import ExecutionEngine


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class TradingMode:
    """Trading mode enum"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class AlphaTradeSystem:
    """
    Main trading system orchestrator.

    Coordinates all components:
    - Data ingestion
    - Feature engineering
    - Model prediction
    - Signal generation
    - Risk management
    - Order execution
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        mode: str = TradingMode.PAPER
    ):
        self.config_path = config_path
        self.mode = mode

        # Load configurations
        self.config = self._load_config(config_path)
        self.symbols_config = self._load_config("config/symbols.yaml")
        self.risk_config = self._load_config("config/risk_params.yaml")

        # Component references
        self.data_loader: Optional[MultiAssetLoader] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.strategies: List[Any] = []
        self.risk_manager: Optional[RiskManager] = None
        self.position_sizer: Optional[VolatilityPositionSizer] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.broker: Optional[AlpacaBroker] = None
        self.order_manager: Optional[OrderManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.ws_manager: Optional[WebSocketManager] = None

        # State
        self._running = False
        self._initialized = False
        self._last_signals: Dict[str, Any] = {}
        self._market_data: Dict[str, pd.DataFrame] = {}

        # Setup logging
        setup_logging(
            log_dir=self.config.get('logging', {}).get('log_dir', 'logs'),
            level=self.config.get('logging', {}).get('level', 'INFO')
        )

    def _load_config(self, path: str) -> Dict:
        """Load YAML configuration"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found: {path}")
            return {}

    async def initialize(self) -> bool:
        """
        Initialize all trading components.

        Returns:
            True if initialization successful
        """
        logger.info("=" * 60)
        logger.info("Initializing AlphaTrade System")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info("=" * 60)

        try:
            # Get symbols
            self.symbols = self.symbols_config.get('universe', {}).get('symbols', [])
            if not self.symbols:
                self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

            logger.info(f"Trading universe: {len(self.symbols)} symbols")

            # Initialize data loader
            logger.info("Initializing data loader...")
            self.data_loader = MultiAssetLoader(
                symbols=self.symbols,
                data_dir="data/raw"
            )

            # Initialize preprocessor
            logger.info("Initializing preprocessor...")
            self.preprocessor = DataPreprocessor()

            # Initialize feature builder
            logger.info("Initializing feature builder...")
            self.feature_builder = FeatureBuilder()

            # Initialize regime detector
            logger.info("Initializing regime detector...")
            self.regime_detector = RegimeDetector()

            # Initialize risk manager
            logger.info("Initializing risk manager...")
            risk_limits = RiskLimits(
                max_position_pct=self.risk_config.get('position_limits', {}).get('max_position_pct', 0.10),
                max_sector_pct=self.risk_config.get('position_limits', {}).get('max_sector_pct', 0.30),
                max_drawdown=self.risk_config.get('drawdown', {}).get('max_drawdown', 0.15),
                max_daily_loss=self.risk_config.get('drawdown', {}).get('daily_loss_limit', 0.03),
                target_volatility=self.risk_config.get('volatility', {}).get('target_annual', 0.15)
            )
            self.risk_manager = RiskManager(limits=risk_limits)

            # Set sector map
            sector_map = {}
            for sector_info in self.symbols_config.get('universe', {}).get('sector_weights', []):
                for symbol in sector_info.get('symbols', []):
                    sector_map[symbol] = sector_info.get('sector', 'Other')
            self.risk_manager.set_sector_map(sector_map)

            # Initialize position sizer
            logger.info("Initializing position sizer...")
            self.position_sizer = VolatilityPositionSizer(
                target_volatility=self.risk_config.get('volatility', {}).get('target_annual', 0.15)
            )

            # Initialize portfolio manager
            logger.info("Initializing portfolio manager...")
            initial_capital = self.config.get('trading', {}).get('initial_capital', 1000000)
            self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
            self.portfolio_manager.set_sector_map(sector_map)

            # Initialize strategies
            logger.info("Initializing strategies...")
            self._init_strategies()

            # Initialize broker (if not backtest mode)
            if self.mode != TradingMode.BACKTEST:
                await self._init_broker()

            self._initialized = True
            logger.info("Initialization complete!")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_strategies(self) -> None:
        """Initialize trading strategies"""
        # Momentum strategy
        momentum = MomentumStrategy(
            name="momentum_main",
            lookback=20,
            threshold=0.02
        )
        self.strategies.append(momentum)

        # Mean reversion strategy
        mean_rev = MeanReversionStrategy(
            name="mean_reversion_main",
            lookback=20,
            entry_zscore=2.0,
            exit_zscore=0.5
        )
        self.strategies.append(mean_rev)

        # ML strategy (if models exist)
        model_path = Path("models/ensemble_model.pkl")
        if model_path.exists():
            try:
                ml_strategy = MLStrategy(
                    name="ml_ensemble",
                    model_path=str(model_path),
                    feature_builder=self.feature_builder
                )
                self.strategies.append(ml_strategy)
                logger.info("ML strategy loaded")
            except Exception as e:
                logger.warning(f"Could not load ML strategy: {e}")

    async def _init_broker(self) -> None:
        """Initialize broker connection"""
        broker_config = self.config.get('broker', {})
        broker_name = broker_config.get('name', 'alpaca')

        logger.info(f"Connecting to broker: {broker_name}")

        # Get credentials from environment
        api_key = os.environ.get('ALPACA_API_KEY', broker_config.get('api_key', ''))
        api_secret = os.environ.get('ALPACA_API_SECRET', broker_config.get('api_secret', ''))

        if not api_key or not api_secret:
            logger.warning("Broker credentials not found - running in simulation mode")
            return

        # Create broker
        self.broker = BrokerFactory.create(
            broker_name=broker_name,
            api_key=api_key,
            api_secret=api_secret,
            paper=(self.mode == TradingMode.PAPER)
        )

        # Connect
        connected = await self.broker.connect()
        if connected:
            logger.info("Broker connected successfully")

            # Initialize order manager
            self.order_manager = OrderManager(
                broker=self.broker,
                risk_manager=self.risk_manager
            )

            # Initialize execution engine
            self.execution_engine = ExecutionEngine(self.order_manager)

            # Get account info
            account = await self.broker.get_account()
            logger.info(f"Account value: ${account.portfolio_value:,.2f}")
            logger.info(f"Buying power: ${account.buying_power:,.2f}")
        else:
            logger.error("Failed to connect to broker")

    async def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols"""
        logger.info("Loading historical data...")

        data = {}
        for symbol in self.symbols:
            try:
                df = self.data_loader.load_symbol(symbol)
                if df is not None and len(df) > 0:
                    # Preprocess
                    df = self.preprocessor.clean_data(df)
                    data[symbol] = df
                    logger.debug(f"Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")

        logger.info(f"Loaded data for {len(data)} symbols")
        self._market_data = data
        return data

    async def generate_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Generate features for all symbols"""
        logger.info("Generating features...")

        features = {}
        for symbol, df in data.items():
            try:
                feature_df = self.feature_builder.build_features(df)
                features[symbol] = feature_df
            except Exception as e:
                logger.warning(f"Feature generation failed for {symbol}: {e}")

        return features

    async def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate trading signals from all strategies"""
        all_signals = {}

        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                for symbol, signal in signals.items():
                    if symbol not in all_signals:
                        all_signals[symbol] = []
                    all_signals[symbol].append(signal)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} error: {e}")

        # Combine signals (simple average for now)
        combined = {}
        for symbol, signal_list in all_signals.items():
            if signal_list:
                avg_strength = np.mean([s.strength for s in signal_list])
                # Use signal with highest absolute strength
                best_signal = max(signal_list, key=lambda s: abs(s.strength))
                combined[symbol] = best_signal

        self._last_signals = combined
        return combined

    async def execute_signals(
        self,
        signals: Dict[str, Any],
        prices: Dict[str, float]
    ) -> None:
        """Execute trading signals"""
        if not self.order_manager:
            logger.warning("Order manager not initialized")
            return

        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            try:
                # Calculate position size
                size = self.position_sizer.calculate_size(
                    symbol=symbol,
                    current_price=prices[symbol],
                    portfolio_value=self.portfolio_manager.portfolio.total_value,
                    signal_strength=signal.strength
                )

                if size.shares == 0:
                    continue

                # Determine side
                side = 'buy' if signal.strength > 0 else 'sell'

                # Create and submit order
                order = await self.order_manager.create_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(size.shares),
                    strategy_name=signal.strategy_name,
                    signal_strength=signal.strength,
                    signal_price=prices[symbol]
                )

                success = await self.order_manager.submit_order(order)

                if success:
                    logger.info(
                        f"Order submitted: {side.upper()} {abs(size.shares)} {symbol} "
                        f"@ ${prices[symbol]:.2f}"
                    )

            except Exception as e:
                logger.error(f"Execution error for {symbol}: {e}")

    async def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """Run backtest on historical data"""
        logger.info("Starting backtest...")

        # Load data
        data = await self.load_historical_data()

        if not data:
            logger.error("No data available for backtest")
            return

        # Create backtest config
        config = BacktestConfig(
            initial_capital=self.config.get('trading', {}).get('initial_capital', 1000000),
            commission_per_share=0.005,
            slippage_bps=5,
            warmup_period=100
        )

        # Use first strategy for backtest
        if not self.strategies:
            logger.error("No strategies configured")
            return

        strategy = self.strategies[0]

        # Run backtest
        engine = BacktestEngine(
            strategy=strategy,
            config=config,
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager
        )

        result = engine.run(data)

        # Generate report
        report_gen = ReportGenerator()
        report = report_gen.generate_text_report(
            result.equity_curve,
            [t.to_dict() for t in result.trades]
        )
        print(report)

        # Save results
        result.equity_curve.to_csv("results/equity_curve.csv")
        pd.DataFrame([t.to_dict() for t in result.trades]).to_csv("results/trades.csv", index=False)

        logger.info("Backtest complete")

    async def run_live(self) -> None:
        """Run live trading loop"""
        logger.info("Starting live trading...")

        self._running = True

        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Main trading loop
        while self._running:
            try:
                # Check if market is open (simplified)
                now = datetime.now()
                market_open = time(9, 30)
                market_close = time(16, 0)

                if market_open <= now.time() <= market_close and now.weekday() < 5:
                    await self._trading_cycle()
                else:
                    logger.debug("Market closed, waiting...")

                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute cycle

            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(60)

    async def _trading_cycle(self) -> None:
        """Execute one trading cycle"""
        logger.debug("Starting trading cycle...")

        with Timer() as timer:
            # Update market data
            if self.broker:
                # In live mode, we would fetch real-time data
                # For now, use historical
                pass

            # Generate features
            features = await self.generate_features(self._market_data)

            # Generate signals
            signals = await self.generate_signals(self._market_data)

            if signals:
                # Get current prices
                prices = {}
                for symbol, df in self._market_data.items():
                    if len(df) > 0:
                        prices[symbol] = df['close'].iloc[-1]

                # Execute signals
                await self.execute_signals(signals, prices)

            # Update risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics()

            logger.debug(f"Trading cycle complete in {timer.elapsed:.2f}s")

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down...")
        self._running = False

        # Cancel all orders
        if self.order_manager:
            cancelled = await self.order_manager.cancel_all()
            logger.info(f"Cancelled {cancelled} orders")

        # Disconnect broker
        if self.broker:
            await self.broker.disconnect()

        # Save state
        self._save_state()

        logger.info("Shutdown complete")

    def _save_state(self) -> None:
        """Save trading state"""
        try:
            os.makedirs("results", exist_ok=True)

            # Save portfolio state
            portfolio_state = self.portfolio_manager.portfolio.to_dict()
            with open("results/portfolio_state.yaml", "w") as f:
                yaml.dump(portfolio_state, f)

            logger.info("State saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def run(self) -> None:
        """Main entry point"""
        # Initialize
        if not await self.initialize():
            logger.error("Initialization failed, exiting")
            return

        # Load data
        await self.load_historical_data()

        # Run based on mode
        if self.mode == TradingMode.BACKTEST:
            await self.run_backtest()
        else:
            await self.run_live()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AlphaTrade Trading System")
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["backtest", "paper", "live"],
        help="Trading mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade"
    )

    args = parser.parse_args()

    # Create and run system
    system = AlphaTradeSystem(
        config_path=args.config,
        mode=args.mode
    )

    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
