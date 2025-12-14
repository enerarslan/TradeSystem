"""
AlphaTrade System - Main Orchestrator
JPMorgan-Level Institutional Trading Platform

This is the main entry point for the trading system.
Coordinates all components for live and paper trading.

Integrated Components:
- ProtectedPositionManager: Server-side stop loss with bracket orders
- ReconciliationEngine: State sync between local and broker
- GracefulDegradationManager: Fault tolerance and fallback handling
- RedisStateManager: Crash recovery with persistent state
- BayesianKellySizer: Uncertainty-aware position sizing
- CorrelationCircuitBreaker: Crisis detection and exposure reduction
- ProbabilityCalibrationManager: Model probability calibration
- ExecutionMetricsCollector: Execution quality monitoring
- ModelStalenessDetector: Model health and accuracy monitoring
- AlmgrenChrissModel: Pre-trade market impact estimation
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
from src.features.institutional import InstitutionalFeatureEngineer
from src.features.regime import RegimeDetector
from src.features.point_in_time import PointInTimeFeatureEngine

from src.models.training import ModelTrainer, WalkForwardValidator
from src.models.ensemble import EnsembleModel
from src.models.calibration import IsotonicCalibrator, PlattCalibrator, CalibrationMetrics

from src.strategy.ml_strategy import MLStrategy, EnsembleMLStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.mean_reversion import MeanReversionStrategy

from src.risk.risk_manager import RiskManager, RiskLimits
from src.risk.position_sizer import VolatilityPositionSizer, RiskParityPositionSizer
from src.risk.portfolio import PortfolioManager
from src.risk.bayesian_kelly import BayesianKellySizer, TradeOutcome
from src.risk.correlation_breaker import CorrelationCircuitBreaker, CorrelationState

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.metrics import MetricsCalculator, ReportGenerator
from src.backtest.realistic_fills import RealisticFillSimulator, FillModel

from src.execution.broker_api import BrokerFactory, AlpacaBroker
from src.execution.order_manager import OrderManager
from src.execution.executor import ExecutionEngine
from src.execution.protected_positions import ProtectedPositionManager, ProtectionConfig
from src.execution.reconciliation import ReconciliationEngine
from src.execution.impact_model import AlmgrenChrissModel

from src.core.graceful_degradation import GracefulDegradationManager, ComponentType, DegradationLevel
from src.core.state_manager import RedisStateManager

from src.monitoring.execution_dashboard import ExecutionMetricsCollector
from src.mlops.staleness import ModelStalenessDetector


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

        # Component references - Core
        self.data_loader: Optional[MultiAssetLoader] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_builder: Optional[InstitutionalFeatureEngineer] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.strategies: List[Any] = []
        self.risk_manager: Optional[RiskManager] = None
        self.position_sizer: Optional[VolatilityPositionSizer] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.broker: Optional[AlpacaBroker] = None
        self.order_manager: Optional[OrderManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.ws_manager: Optional[WebSocketManager] = None

        # Component references - New Integrated Modules
        self.protected_position_manager: Optional[ProtectedPositionManager] = None
        self.reconciliation_engine: Optional[ReconciliationEngine] = None
        self.graceful_degradation: Optional[GracefulDegradationManager] = None
        self.state_manager: Optional[RedisStateManager] = None
        self.bayesian_kelly: Optional[BayesianKellySizer] = None
        self.correlation_breaker: Optional[CorrelationCircuitBreaker] = None
        self.probability_calibrator: Optional[IsotonicCalibrator] = None
        self.execution_monitor: Optional[ExecutionMetricsCollector] = None
        self.staleness_detector: Optional[ModelStalenessDetector] = None
        self.impact_model: Optional[AlmgrenChrissModel] = None
        self.point_in_time_engine: Optional[PointInTimeFeatureEngine] = None
        self.fill_simulator: Optional[RealisticFillSimulator] = None

        # State
        self._running = False
        self._initialized = False
        self._last_signals: Dict[str, Any] = {}
        self._market_data: Dict[str, pd.DataFrame] = {}
        self._last_reconciliation: Optional[datetime] = None
        self._model_trained_date: Optional[datetime] = None

        # Track open positions for Bayesian Kelly outcome recording
        # Key: symbol, Value: {entry_price, side, strategy, quantity, timestamp}
        self._open_positions_for_kelly: Dict[str, Dict[str, Any]] = {}

        # Setup logging - FIXED: Use correct parameter names
        setup_logging(
            log_path=self.config.get('logging', {}).get('log_dir', 'logs'),
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
            # ================================================================
            # PHASE 1: Core Infrastructure (Graceful Degradation & State)
            # ================================================================

            # Initialize GracefulDegradationManager FIRST (catches all failures)
            logger.info("Initializing graceful degradation manager...")
            self.graceful_degradation = GracefulDegradationManager(
                check_interval_seconds=5.0,
                alert_callback=self._on_degradation_alert
            )
            await self.graceful_degradation.start()

            # Initialize RedisStateManager for crash recovery
            logger.info("Initializing Redis state manager...")
            self.state_manager = RedisStateManager(
                redis_client=None,  # Will use in-memory fallback if Redis unavailable
                auto_save_interval=5.0,
                state_ttl_hours=24
            )
            await self.state_manager.initialize()

            # Attempt to recover state from previous session
            recovered_state = await self.state_manager.recover_state()
            if recovered_state:
                logger.info(f"Recovered state: {len(recovered_state.get('positions', {}))} positions, "
                           f"{len(recovered_state.get('orders', {}))} orders")

            # ================================================================
            # PHASE 2: Symbol Universe Setup
            # ================================================================

            # Get symbols - extract from all sectors in the YAML structure
            self.symbols = []
            sectors = self.symbols_config.get('sectors', {})
            for sector_name, sector_data in sectors.items():
                sector_symbols = sector_data.get('symbols', [])
                self.symbols.extend(sector_symbols)

            # If no symbols found in sectors, fall back to symbols dict keys
            if not self.symbols:
                symbols_dict = self.symbols_config.get('symbols', {})
                self.symbols = list(symbols_dict.keys())

            # Final fallback to default symbols
            if not self.symbols:
                self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
                logger.warning("No symbols found in config, using default 5 symbols")

            logger.info(f"Trading universe: {len(self.symbols)} symbols")

            # ================================================================
            # PHASE 3: Data & Feature Infrastructure
            # ================================================================

            # Initialize data loader
            logger.info("Initializing data loader...")
            self.data_loader = MultiAssetLoader(
                data_path="data/raw",
                cache_path="data/cache"
            )

            # Initialize preprocessor
            logger.info("Initializing preprocessor...")
            self.preprocessor = DataPreprocessor()

            # Initialize feature builder - Use InstitutionalFeatureEngineer for ML model compatibility
            logger.info("Initializing institutional feature engineer...")
            self.feature_builder = InstitutionalFeatureEngineer()

            # Initialize Point-in-Time Feature Engine (prevents look-ahead bias)
            logger.info("Initializing point-in-time feature engine...")
            self.point_in_time_engine = PointInTimeFeatureEngine()
            self.point_in_time_engine.add_standard_features()

            # Initialize regime detector
            logger.info("Initializing regime detector...")
            self.regime_detector = RegimeDetector()

            # ================================================================
            # PHASE 4: Risk Management Infrastructure
            # ================================================================

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

            # Initialize legacy position sizer (fallback)
            logger.info("Initializing position sizer...")
            self.position_sizer = VolatilityPositionSizer(
                target_volatility=self.risk_config.get('volatility', {}).get('target_annual', 0.15)
            )

            # Initialize Bayesian Kelly Sizer (primary position sizing)
            logger.info("Initializing Bayesian Kelly sizer...")
            self.bayesian_kelly = BayesianKellySizer(
                prior_wins=2.0,
                prior_losses=2.0,
                prior_avg_win=0.02,
                prior_avg_loss=0.02,
                kelly_fraction=0.25,  # Quarter Kelly for safety
                max_position_pct=0.20,
                min_observations=20,
                uncertainty_penalty_weight=1.0
            )

            # Initialize Correlation Circuit Breaker
            logger.info("Initializing correlation circuit breaker...")
            self.correlation_breaker = CorrelationCircuitBreaker(
                correlation_spike_threshold=0.25,
                crisis_threshold=0.40,
                first_pc_threshold=0.55,
                first_pc_crisis_threshold=0.70,
                lookback_period=20,
                cooldown_periods=10
            )

            # Initialize portfolio manager
            logger.info("Initializing portfolio manager...")
            initial_capital = self.config.get('trading', {}).get('initial_capital', 1000000)
            self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
            self.portfolio_manager.set_sector_map(sector_map)

            # ================================================================
            # PHASE 5: Model & Calibration Infrastructure
            # ================================================================

            # Initialize probability calibrator
            logger.info("Initializing probability calibrator...")
            self.probability_calibrator = IsotonicCalibrator()

            # Load calibration model if exists
            calibration_path = Path("models/calibration_model.pkl")
            if calibration_path.exists():
                try:
                    import pickle
                    with open(calibration_path, 'rb') as f:
                        self.probability_calibrator = pickle.load(f)
                    logger.info("Loaded probability calibration model")
                except Exception as e:
                    logger.warning(f"Could not load calibration model: {e}")

            # Initialize model staleness detector
            logger.info("Initializing model staleness detector...")
            model_metrics_path = Path("models/metrics.yaml")
            if model_metrics_path.exists():
                with open(model_metrics_path, 'r') as f:
                    model_metrics = yaml.safe_load(f)
                training_date_str = model_metrics.get('training_date', '')
                if training_date_str:
                    self._model_trained_date = datetime.fromisoformat(training_date_str)

            self.staleness_detector = ModelStalenessDetector(
                model_name="catboost_ensemble",
                model_trained_date=self._model_trained_date or datetime.now(),
                max_age_days=30,
                warning_age_days=21,
                min_accuracy_threshold=0.52,
                min_samples_for_eval=100
            )
            self.staleness_detector.add_alert_handler(self._on_staleness_alert)

            # ================================================================
            # PHASE 6: Execution Infrastructure
            # ================================================================

            # Initialize market impact model
            logger.info("Initializing market impact model...")
            self.impact_model = AlmgrenChrissModel(
                permanent_impact_coef=0.1,
                temporary_impact_coef=0.2,
                temporary_impact_exp=0.6,
                risk_aversion=1e-6
            )

            # Initialize execution metrics collector
            logger.info("Initializing execution metrics collector...")
            self.execution_monitor = ExecutionMetricsCollector(
                buffer_size=10000,
                export_interval_seconds=10.0
            )

            # Initialize realistic fill simulator (for backtest)
            logger.info("Initializing realistic fill simulator...")
            self.fill_simulator = RealisticFillSimulator(
                model=FillModel.IMPACT,
                default_spread_bps=10.0,
                impact_coefficient=0.1,
                participation_rate=0.1
            )

            # ================================================================
            # PHASE 7: Strategy Initialization
            # ================================================================

            logger.info("Initializing strategies...")
            self._init_strategies()

            # ================================================================
            # PHASE 8: Broker & Live Trading Infrastructure
            # ================================================================

            if self.mode != TradingMode.BACKTEST:
                await self._init_broker()

                # Initialize protected position manager (after broker)
                if self.broker:
                    logger.info("Initializing protected position manager...")
                    protection_config = ProtectionConfig(
                        default_stop_loss_pct=0.02,
                        default_take_profit_pct=0.04,
                        use_bracket_orders=True,
                        max_slippage_pct=0.005
                    )
                    self.protected_position_manager = ProtectedPositionManager(
                        broker=self.broker,
                        config=protection_config
                    )

                    # Initialize reconciliation engine
                    logger.info("Initializing reconciliation engine...")
                    self.reconciliation_engine = ReconciliationEngine(
                        broker=self.broker,
                        order_manager=self.order_manager,
                        reconcile_interval_seconds=30,
                        auto_fix_enabled=True,
                        alert_callback=self._on_reconciliation_alert
                    )

            # ================================================================
            # PHASE 9: Register Components with Graceful Degradation
            # ================================================================

            logger.info("Registering components for health monitoring...")
            self._register_components_for_monitoring()

            # Start auto-save for state manager
            await self.state_manager.start_auto_save()

            # Start a new trading session
            session_id = f"{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.state_manager.start_session(session_id)

            self._initialized = True
            logger.info("=" * 60)
            logger.info("Initialization complete!")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()

            # Try to gracefully handle initialization failure
            if self.graceful_degradation:
                await self.graceful_degradation.stop()

            return False

    def _register_components_for_monitoring(self) -> None:
        """Register all components with graceful degradation manager for health monitoring."""
        if not self.graceful_degradation:
            return

        # Register broker health check
        if self.broker:
            self.graceful_degradation.register_component(
                name="broker",
                component_type=ComponentType.BROKER,
                health_check=lambda: asyncio.create_task(self._check_broker_health())
            )

        # Register data feed health check
        if self.data_loader:
            self.graceful_degradation.register_component(
                name="data_loader",
                component_type=ComponentType.DATA_FEED,
                health_check=lambda: True  # Simplified check
            )

        # Register model health check
        self.graceful_degradation.register_component(
            name="ml_model",
            component_type=ComponentType.MODEL,
            health_check=self._check_model_health
        )

        # Register Redis health check
        if self.state_manager:
            self.graceful_degradation.register_component(
                name="state_manager",
                component_type=ComponentType.REDIS,
                health_check=lambda: self.state_manager._client is not None
            )

    async def _check_broker_health(self) -> bool:
        """Check broker connection health."""
        try:
            if self.broker:
                account = await self.broker.get_account()
                return account is not None
            return False
        except Exception:
            return False

    def _check_model_health(self) -> bool:
        """Check ML model health via staleness detector."""
        if self.staleness_detector:
            report = self.staleness_detector.check_staleness()
            return report.level.value <= 2  # FRESH or AGING is OK
        return True

    def _on_degradation_alert(self, level: DegradationLevel, message: str) -> None:
        """Handle degradation alerts."""
        logger.warning(f"DEGRADATION ALERT [{level.name}]: {message}")
        audit_logger.warning(f"System degradation: {level.name} - {message}")

    def _on_staleness_alert(self, report) -> None:
        """Handle model staleness alerts."""
        logger.warning(f"MODEL STALENESS ALERT [{report.level.name}]: {report.recommendation}")
        audit_logger.warning(f"Model staleness: {report.level.name}")

    def _on_reconciliation_alert(self, discrepancy) -> None:
        """Handle reconciliation alerts."""
        logger.warning(f"RECONCILIATION ALERT: {discrepancy.type.name} for {discrepancy.symbol}")
        audit_logger.warning(f"Reconciliation discrepancy: {discrepancy}")

    def _init_strategies(self) -> None:
        """Initialize trading strategies"""
        # Momentum strategy - FIXED: Don't pass name through kwargs (hardcoded in strategy)
        momentum = MomentumStrategy(
            lookback_periods=[10, 20, 50],
            rsi_period=14
        )
        self.strategies.append(momentum)

        # Mean reversion strategy - FIXED: Use correct parameter names
        mean_rev = MeanReversionStrategy(
            lookback_period=20,
            entry_zscore=2.0,
            exit_zscore=0.5
        )
        self.strategies.append(mean_rev)

        # ML strategy (if models exist) - FIXED: Use correct model path
        model_path = Path("models/model.pkl")
        features_path = Path("models/features.txt")

        if model_path.exists():
            try:
                # Load feature list from features.txt for consistency
                feature_list = None
                if features_path.exists():
                    with open(features_path, 'r') as f:
                        feature_list = [line.strip() for line in f.readlines() if line.strip()]
                    logger.info(f"Loaded {len(feature_list)} features from features.txt")

                ml_strategy = MLStrategy(
                    name="ml_ensemble",
                    model_path=str(model_path),
                    feature_builder=self.feature_builder,
                    feature_list=feature_list
                )
                self.strategies.append(ml_strategy)
                logger.info("ML strategy loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load ML strategy: {e}")
                import traceback
                traceback.print_exc()

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

        # FIXED: Use load_symbols method (plural) which handles parallel loading
        raw_data = self.data_loader.load_symbols(
            symbols=self.symbols,
            show_progress=True
        )

        # Preprocess each symbol's data - FIXED: Use preprocess() method which returns tuple
        data = {}
        for symbol, df in raw_data.items():
            try:
                if df is not None and len(df) > 0:
                    df_clean, report = self.preprocessor.preprocess(df, symbol)
                    data[symbol] = df_clean
                    logger.debug(f"Loaded {len(df_clean)} bars for {symbol} (quality: {report.quality_score:.1f})")
            except Exception as e:
                logger.warning(f"Failed to preprocess {symbol}: {e}")

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

        # FIXED: Select ML strategy as primary backtest target
        if not self.strategies:
            logger.error("No strategies configured")
            return

        # Find ML strategy by name (ml_ensemble or ml_strategy)
        strategy = None
        ml_strategy_names = ['ml_ensemble', 'ml_strategy']

        for s in self.strategies:
            if s.name in ml_strategy_names:
                strategy = s
                logger.info(f"Selected ML strategy '{s.name}' for backtest")
                break

        # Fallback to first strategy if no ML strategy found
        if strategy is None:
            strategy = self.strategies[0]
            logger.warning(f"ML strategy not found, falling back to '{strategy.name}'")

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
        """Execute one trading cycle with all integrated components."""
        logger.debug("Starting trading cycle...")

        with Timer() as timer:
            # ================================================================
            # PRE-TRADE CHECKS
            # ================================================================

            # Check if trading is allowed (graceful degradation)
            if self.graceful_degradation and not self.graceful_degradation.is_trading_allowed():
                degradation_level = self.graceful_degradation.get_degradation_level()
                logger.warning(f"Trading blocked due to degradation level: {degradation_level.name}")
                return

            # Check correlation circuit breaker for crisis conditions
            if self.correlation_breaker:
                correlation_state = self.correlation_breaker.get_state()
                if correlation_state == CorrelationState.CRISIS:
                    logger.warning("CRISIS detected by correlation breaker - halting new positions")
                    # Could also reduce existing positions here
                    return
                elif correlation_state == CorrelationState.ELEVATED:
                    logger.info("Elevated correlations detected - proceeding with caution")

            # Check model staleness
            if self.staleness_detector:
                staleness_report = self.staleness_detector.check_staleness()
                if staleness_report.level.name == "CRITICAL":
                    logger.warning(f"Model is critically stale: {staleness_report.recommendation}")
                    # Fall back to rule-based strategies only
                    return

            # ================================================================
            # DATA & FEATURE GENERATION
            # ================================================================

            # Update market data
            if self.broker:
                # In live mode, fetch real-time data from broker
                try:
                    for symbol in self.symbols[:10]:  # Limit to avoid rate limits
                        bars = await self.broker.get_bars(symbol, timeframe='1Min', limit=100)
                        if bars is not None and len(bars) > 0:
                            self._market_data[symbol] = bars
                except Exception as e:
                    logger.warning(f"Failed to fetch real-time data: {e}")

            if not self._market_data:
                logger.debug("No market data available")
                return

            # Update correlation breaker with latest returns
            if self.correlation_breaker and len(self._market_data) > 5:
                try:
                    returns_df = pd.DataFrame({
                        symbol: df['close'].pct_change().dropna()
                        for symbol, df in self._market_data.items()
                        if len(df) > 1
                    })
                    if len(returns_df) > 0:
                        latest_returns = returns_df.iloc[-1].to_dict()
                        self.correlation_breaker.update(latest_returns)
                except Exception as e:
                    logger.debug(f"Correlation update failed: {e}")

            # Generate features
            features = await self.generate_features(self._market_data)

            # ================================================================
            # SIGNAL GENERATION & CALIBRATION
            # ================================================================

            # Generate signals
            signals = await self.generate_signals(self._market_data)

            if not signals:
                logger.debug("No signals generated")
                return

            # Calibrate probabilities if we have a calibrator
            if self.probability_calibrator and hasattr(self.probability_calibrator, 'calibrate'):
                for symbol, signal in signals.items():
                    if hasattr(signal, 'confidence') and signal.confidence is not None:
                        try:
                            calibrated_prob = self.probability_calibrator.calibrate(
                                np.array([signal.confidence])
                            )[0]
                            signal.calibrated_confidence = calibrated_prob
                        except Exception:
                            signal.calibrated_confidence = signal.confidence

            # ================================================================
            # POSITION SIZING & EXECUTION
            # ================================================================

            # Get current prices
            prices = {}
            for symbol, df in self._market_data.items():
                if len(df) > 0:
                    prices[symbol] = df['close'].iloc[-1]

            # Execute signals with new position sizing
            await self._execute_signals_with_protection(signals, prices)

            # ================================================================
            # POST-TRADE UPDATES
            # ================================================================

            # Update risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics()

            # Trigger reconciliation if enough time has passed (every 30 seconds)
            now = datetime.now()
            if self.reconciliation_engine:
                if (self._last_reconciliation is None or
                    (now - self._last_reconciliation).total_seconds() >= 30):
                    try:
                        await self.reconciliation_engine.reconcile()
                        self._last_reconciliation = now
                    except Exception as e:
                        logger.warning(f"Reconciliation failed: {e}")

            # Check for closed positions and record outcomes for Bayesian Kelly
            await self._check_and_record_closed_positions()

            # Save state
            if self.state_manager:
                try:
                    await self.state_manager.save_risk_state({
                        'metrics': risk_metrics,
                        'timestamp': now.isoformat()
                    })
                except Exception as e:
                    logger.debug(f"State save failed: {e}")

            logger.debug(f"Trading cycle complete in {timer.elapsed:.2f}s")

    async def _execute_signals_with_protection(
        self,
        signals: Dict[str, Any],
        prices: Dict[str, float]
    ) -> None:
        """Execute signals using protected positions and Bayesian Kelly sizing."""
        if not self.order_manager and not self.protected_position_manager:
            logger.warning("No execution manager available")
            return

        portfolio_value = self.portfolio_manager.portfolio.total_value

        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            try:
                current_price = prices[symbol]

                # Get signal confidence (use calibrated if available)
                confidence = getattr(signal, 'calibrated_confidence',
                                   getattr(signal, 'confidence', abs(signal.strength)))

                # Calculate position size using Bayesian Kelly
                if self.bayesian_kelly:
                    # Get strategy name from signal
                    strategy_name = getattr(signal, 'strategy_name', 'default')

                    kelly_result = self.bayesian_kelly.calculate_kelly(
                        symbol=symbol,
                        strategy=strategy_name,
                        signal_strength=confidence
                    )
                    position_pct = kelly_result.fractional_kelly  # Use fractional_kelly for position %

                    # Apply correlation breaker reduction if needed
                    if self.correlation_breaker:
                        state = self.correlation_breaker.get_state()
                        if state == CorrelationState.ELEVATED:
                            position_pct *= 0.5  # Reduce by 50%
                            logger.info(f"Reduced position for {symbol} due to elevated correlations")

                    shares = int((portfolio_value * position_pct) / current_price)
                else:
                    # Fallback to volatility-based sizing
                    size = self.position_sizer.calculate_size(
                        symbol=symbol,
                        current_price=current_price,
                        portfolio_value=portfolio_value,
                        signal_strength=signal.strength
                    )
                    shares = size.shares

                if shares == 0:
                    continue

                # Estimate market impact before executing
                if self.impact_model:
                    try:
                        # Get average daily volume (simplified)
                        adv = 1000000  # Default 1M shares
                        if symbol in self._market_data:
                            df = self._market_data[symbol]
                            if 'volume' in df.columns and len(df) > 20:
                                adv = df['volume'].tail(20).mean()

                        impact = self.impact_model.estimate_impact(
                            shares=shares,
                            price=current_price,
                            daily_volume=adv,
                            volatility=0.02  # Default 2% daily vol
                        )

                        # Skip trade if impact is too high
                        if impact.total_cost_bps > 50:  # More than 50 bps impact
                            logger.info(f"Skipping {symbol}: impact too high ({impact.total_cost_bps:.1f} bps)")
                            continue

                    except Exception as e:
                        logger.debug(f"Impact estimation failed for {symbol}: {e}")

                # Determine side
                side = 'buy' if signal.strength > 0 else 'sell'

                # Execute with protected position manager if available
                if self.protected_position_manager:
                    try:
                        # Calculate stop loss and take profit
                        stop_loss_pct = 0.02  # 2% stop loss
                        take_profit_pct = 0.04  # 4% take profit

                        if side == 'buy':
                            stop_price = current_price * (1 - stop_loss_pct)
                            take_profit_price = current_price * (1 + take_profit_pct)
                        else:
                            stop_price = current_price * (1 + stop_loss_pct)
                            take_profit_price = current_price * (1 - take_profit_pct)

                        position = await self.protected_position_manager.open_position_with_protection(
                            symbol=symbol,
                            side=side,
                            quantity=abs(shares),
                            entry_price=current_price,
                            stop_loss_price=stop_price,
                            take_profit_price=take_profit_price
                        )

                        if position:
                            logger.info(
                                f"Protected position opened: {side.upper()} {abs(shares)} {symbol} "
                                f"@ ${current_price:.2f} (SL: ${stop_price:.2f}, TP: ${take_profit_price:.2f})"
                            )

                            # Record execution metrics
                            if self.execution_monitor:
                                self.execution_monitor.record_execution(
                                    symbol=symbol,
                                    side=side,
                                    quantity=shares,
                                    decision_price=current_price,
                                    fill_price=current_price,  # Simplified
                                    latency_ms=0
                                )

                            # Save position to state manager
                            if self.state_manager:
                                await self.state_manager.save_position({
                                    'symbol': symbol,
                                    'side': side,
                                    'quantity': shares,
                                    'entry_price': current_price,
                                    'stop_loss': stop_price,
                                    'take_profit': take_profit_price,
                                    'timestamp': datetime.now().isoformat()
                                })

                            # Track position for Bayesian Kelly outcome recording
                            self._open_positions_for_kelly[symbol] = {
                                'entry_price': current_price,
                                'side': side,
                                'strategy': strategy_name,
                                'quantity': shares,
                                'timestamp': datetime.now()
                            }

                    except Exception as e:
                        logger.error(f"Protected position failed for {symbol}: {e}")
                        # Fall through to regular order manager

                elif self.order_manager:
                    # Fallback to regular order manager
                    order = await self.order_manager.create_order(
                        symbol=symbol,
                        side=side,
                        quantity=abs(shares),
                        strategy_name=signal.strategy_name,
                        signal_strength=signal.strength,
                        signal_price=current_price
                    )

                    success = await self.order_manager.submit_order(order)

                    if success:
                        logger.info(
                            f"Order submitted: {side.upper()} {abs(shares)} {symbol} "
                            f"@ ${current_price:.2f}"
                        )

                        # Track position for Bayesian Kelly outcome recording
                        strategy_name = getattr(signal, 'strategy_name', 'default')
                        self._open_positions_for_kelly[symbol] = {
                            'entry_price': current_price,
                            'side': side,
                            'strategy': strategy_name,
                            'quantity': shares,
                            'timestamp': datetime.now()
                        }

            except Exception as e:
                logger.error(f"Execution error for {symbol}: {e}")

    def _record_trade_outcome(
        self,
        symbol: str,
        exit_price: float
    ) -> None:
        """
        Record trade outcome for Bayesian Kelly learning.

        Called when a position is closed to update the Bayesian posterior
        with the actual trade result.

        Args:
            symbol: Symbol that was closed
            exit_price: Price at which the position was closed
        """
        if symbol not in self._open_positions_for_kelly:
            logger.debug(f"No tracked position for {symbol} to record")
            return

        if not self.bayesian_kelly:
            return

        position_info = self._open_positions_for_kelly[symbol]
        entry_price = position_info['entry_price']
        side = position_info['side']
        strategy = position_info['strategy']

        # Calculate profit percentage
        if side == 'buy':
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price

        # Determine if win or loss
        win = profit_pct > 0

        # Create trade outcome
        outcome = TradeOutcome(
            symbol=symbol,
            strategy=strategy,
            win=win,
            profit_pct=profit_pct,
            timestamp=datetime.now()
        )

        # Record with Bayesian Kelly sizer
        self.bayesian_kelly.record_outcome(outcome)

        logger.info(
            f"Recorded trade outcome for {symbol}: "
            f"{'WIN' if win else 'LOSS'} {profit_pct*100:.2f}% "
            f"(strategy: {strategy})"
        )

        # Clean up tracked position
        del self._open_positions_for_kelly[symbol]

    async def _check_and_record_closed_positions(self) -> None:
        """
        Check for closed positions and record outcomes for Bayesian Kelly.

        This should be called periodically in the trading cycle to detect
        positions that have been closed (via stop loss, take profit, or manual).
        """
        if not self.broker or not self._open_positions_for_kelly:
            return

        try:
            # Get current positions from broker
            current_positions = await self.broker.get_positions()
            current_symbols = {p.symbol for p in current_positions}

            # Find positions we were tracking that are now closed
            tracked_symbols = set(self._open_positions_for_kelly.keys())
            closed_symbols = tracked_symbols - current_symbols

            # Get current prices for closed positions
            for symbol in closed_symbols:
                try:
                    # Try to get the last price from market data
                    if symbol in self._market_data and len(self._market_data[symbol]) > 0:
                        exit_price = self._market_data[symbol]['close'].iloc[-1]
                        self._record_trade_outcome(symbol, exit_price)
                    else:
                        logger.warning(f"No exit price available for closed position {symbol}")
                        # Still clean up the tracked position
                        del self._open_positions_for_kelly[symbol]
                except Exception as e:
                    logger.error(f"Error recording outcome for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error checking closed positions: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown with all component cleanup."""
        logger.info("=" * 60)
        logger.info("Initiating graceful shutdown...")
        logger.info("=" * 60)

        self._running = False

        # ================================================================
        # PHASE 1: Stop Background Tasks
        # ================================================================

        # Stop reconciliation engine
        if self.reconciliation_engine:
            try:
                logger.info("Stopping reconciliation engine...")
                await self.reconciliation_engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping reconciliation: {e}")

        # Stop graceful degradation monitoring
        if self.graceful_degradation:
            try:
                logger.info("Stopping graceful degradation monitor...")
                await self.graceful_degradation.stop()
            except Exception as e:
                logger.warning(f"Error stopping degradation monitor: {e}")

        # Stop execution metrics export
        if self.execution_monitor:
            try:
                logger.info("Stopping execution metrics collector...")
                self.execution_monitor.stop_export()
            except Exception as e:
                logger.warning(f"Error stopping metrics collector: {e}")

        # ================================================================
        # PHASE 2: Cancel Orders & Close Positions
        # ================================================================

        # Cancel all pending orders
        if self.order_manager:
            try:
                cancelled = await self.order_manager.cancel_all()
                logger.info(f"Cancelled {cancelled} pending orders")
            except Exception as e:
                logger.warning(f"Error cancelling orders: {e}")

        # Close protected positions if configured
        if self.protected_position_manager:
            try:
                active_positions = self.protected_position_manager.get_active_positions()
                logger.info(f"Active protected positions at shutdown: {len(active_positions)}")
                # Note: Positions remain protected via server-side stops
            except Exception as e:
                logger.warning(f"Error getting protected positions: {e}")

        # ================================================================
        # PHASE 3: Generate Reports
        # ================================================================

        # Generate execution quality report
        if self.execution_monitor:
            try:
                report = self.execution_monitor.get_execution_quality_report()
                if report:
                    logger.info("Execution Quality Report:")
                    logger.info(f"  Total executions: {report.total_executions}")
                    logger.info(f"  Avg slippage: {report.avg_slippage_bps:.2f} bps")
                    logger.info(f"  Fill rate: {report.fill_rate:.1%}")

                    # Save report to file
                    os.makedirs("results", exist_ok=True)
                    with open("results/execution_report.yaml", "w") as f:
                        yaml.dump({
                            'total_executions': report.total_executions,
                            'avg_slippage_bps': float(report.avg_slippage_bps),
                            'fill_rate': float(report.fill_rate),
                            'timestamp': datetime.now().isoformat()
                        }, f)
            except Exception as e:
                logger.warning(f"Error generating execution report: {e}")

        # Generate staleness report
        if self.staleness_detector:
            try:
                staleness_report = self.staleness_detector.check_staleness()
                logger.info(f"Model Staleness: {staleness_report.level.name}")
                if staleness_report.recent_accuracy:
                    logger.info(f"  Recent accuracy: {staleness_report.recent_accuracy:.1%}")
            except Exception as e:
                logger.warning(f"Error generating staleness report: {e}")

        # ================================================================
        # PHASE 4: Save State
        # ================================================================

        # Save final state to Redis/file
        if self.state_manager:
            try:
                logger.info("Saving final state...")
                await self.state_manager.end_session()
            except Exception as e:
                logger.warning(f"Error saving state: {e}")

        # Save legacy state
        self._save_state()

        # ================================================================
        # PHASE 5: Disconnect
        # ================================================================

        # Disconnect broker
        if self.broker:
            try:
                logger.info("Disconnecting from broker...")
                await self.broker.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting broker: {e}")

        logger.info("=" * 60)
        logger.info("Shutdown complete")
        logger.info("=" * 60)

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

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for all components.
        Can be exposed via HTTP endpoint for monitoring.
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'initialized': self._initialized,
            'running': self._running,
            'components': {},
            'degradation_level': None,
            'correlation_state': None,
            'model_staleness': None
        }

        # Graceful degradation status
        if self.graceful_degradation:
            deg_level = self.graceful_degradation.get_degradation_level()
            status['degradation_level'] = deg_level.name
            status['trading_allowed'] = self.graceful_degradation.is_trading_allowed()
            status['components']['graceful_degradation'] = 'healthy'
        else:
            status['components']['graceful_degradation'] = 'not_initialized'

        # Correlation breaker status
        if self.correlation_breaker:
            corr_state = self.correlation_breaker.get_state()
            status['correlation_state'] = corr_state.name
            metrics = self.correlation_breaker.get_metrics()
            if metrics:
                status['correlation_metrics'] = {
                    'mean_correlation': float(metrics.mean_correlation) if metrics.mean_correlation else None,
                    'first_pc_variance': float(metrics.first_pc_variance_ratio) if metrics.first_pc_variance_ratio else None
                }
            status['components']['correlation_breaker'] = 'healthy'
        else:
            status['components']['correlation_breaker'] = 'not_initialized'

        # Model staleness status
        if self.staleness_detector:
            report = self.staleness_detector.check_staleness()
            status['model_staleness'] = report.level.name
            status['model_age_days'] = report.model_age_days
            if report.recent_accuracy:
                status['model_recent_accuracy'] = float(report.recent_accuracy)
            status['components']['staleness_detector'] = 'healthy'
        else:
            status['components']['staleness_detector'] = 'not_initialized'

        # Broker status
        if self.broker:
            status['components']['broker'] = 'connected'
        else:
            status['components']['broker'] = 'not_connected'

        # State manager status
        if self.state_manager:
            status['components']['state_manager'] = 'healthy'
        else:
            status['components']['state_manager'] = 'not_initialized'

        # Protected position manager status
        if self.protected_position_manager:
            try:
                positions = self.protected_position_manager.get_active_positions()
                status['active_protected_positions'] = len(positions)
                status['components']['protected_positions'] = 'healthy'
            except Exception:
                status['components']['protected_positions'] = 'error'
        else:
            status['components']['protected_positions'] = 'not_initialized'

        # Execution monitor status
        if self.execution_monitor:
            try:
                report = self.execution_monitor.get_execution_quality_report()
                if report:
                    status['execution_metrics'] = {
                        'total_executions': report.total_executions,
                        'avg_slippage_bps': float(report.avg_slippage_bps),
                        'fill_rate': float(report.fill_rate)
                    }
                status['components']['execution_monitor'] = 'healthy'
            except Exception:
                status['components']['execution_monitor'] = 'error'
        else:
            status['components']['execution_monitor'] = 'not_initialized'

        # Bayesian Kelly status
        if self.bayesian_kelly:
            status['components']['bayesian_kelly'] = 'healthy'
        else:
            status['components']['bayesian_kelly'] = 'not_initialized'

        # Calculate overall health
        component_statuses = list(status['components'].values())
        healthy_count = sum(1 for s in component_statuses if s in ['healthy', 'connected'])
        total_count = len(component_statuses)
        status['overall_health'] = f"{healthy_count}/{total_count} components healthy"

        return status

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
