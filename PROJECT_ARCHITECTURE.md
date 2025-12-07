# AlphaTrade System - Project Architecture

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗ ████████╗██████╗  █████╗ ██████╗ ██████╗ ║
║    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔═══╝ ║
║    ███████║██║     ██████╔╝███████║███████║   ██║   ██████╔╝███████║██║  ██║█████╗  ║
║    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ║
║    ██║  ██║███████╗██║     ██║  ██║██║  ██║   ██║   ██║  ██║██║  ██║██████╔╝██████╗ ║
║    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ║
║                                                                               ║
║                    INSTITUTIONAL-GRADE ALGORITHMIC TRADING SYSTEM             ║
║                                                                               ║
║                              Version 2.0.0                                    ║
║                          Last Updated: 2025-12-07                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## Executive Summary

AlphaTrade is a **production-grade algorithmic trading platform** built to institutional (JPMorgan/Goldman Sachs) standards. The system provides end-to-end capabilities from data ingestion through live execution, with advanced ML-based alpha generation.

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Lines of Code | ~25,000+ |
| Test Coverage Target | 85%+ |
| Supported Symbols | 46 (Dow Jones + NASDAQ 100 subset) |
| ML Models | 7 (LightGBM, XGBoost, CatBoost, RF, LSTM, Transformer, TCN) |
| Strategies | 12+ (Momentum, Statistical, ML-Based) |
| API Endpoints | 15+ |

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ALPHATRADE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   DATA      │───▶│  FEATURES   │───▶│   MODELS    │───▶│  SIGNALS    │      │
│  │   LAYER     │    │   ENGINE    │    │   (ML/DL)   │    │  GENERATOR  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘      │
│         │                                                         │             │
│         │                                                         ▼             │
│  ┌──────▼──────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   MARKET    │    │    RISK     │◀───│  PORTFOLIO  │◀───│  STRATEGY   │      │
│  │    DATA     │    │   MANAGER   │    │  OPTIMIZER  │    │   ENGINE    │      │
│  └─────────────┘    └──────┬──────┘    └─────────────┘    └─────────────┘      │
│                            │                                                    │
│                            ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  BACKTEST   │    │  EXECUTION  │───▶│   BROKER    │───▶│   MARKET    │      │
│  │   ENGINE    │    │  ALGORITHMS │    │   (ALPACA)  │    │  (LIVE/SIM) │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            REST API (FastAPI)                            │   │
│  │   /backtest  │  /models  │  /strategies  │  /data  │  /health  │  /ws   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
AlphaTrade_System/
│
├── 📁 config/                          # ═══ CONFIGURATION LAYER ═══
│   ├── __init__.py                     # Module exports
│   └── settings.py                     # Centralized settings (Pydantic v2)
│       ├── Settings                    # Main configuration class
│       ├── DatabaseSettings            # DB connection settings
│       ├── AlpacaSettings              # Broker API settings
│       ├── DataSettings                # Data paths & cache
│       ├── BacktestSettings            # Backtest parameters
│       ├── RiskSettings                # Risk limits
│       ├── MLSettings                  # ML hyperparameters
│       ├── LogLevel (enum)             # Logging levels
│       ├── TradingMode (enum)          # BACKTEST/PAPER/LIVE
│       ├── TimeFrame (enum)            # 1m/5m/15m/1h/1d
│       └── configure_logging()         # Structlog setup
│
├── 📁 core/                            # ═══ CORE BUILDING BLOCKS ═══
│   ├── __init__.py                     # Module exports (50+ items)
│   ├── events.py                       # Event-driven architecture
│   │   ├── EventType (enum)            # MARKET/SIGNAL/ORDER/FILL/etc.
│   │   ├── EventPriority (enum)        # LOW/NORMAL/HIGH/CRITICAL
│   │   ├── Event (base)                # Base event class
│   │   ├── MarketEvent                 # Price update events
│   │   ├── SignalEvent                 # Strategy signals
│   │   ├── OrderEvent                  # Order lifecycle
│   │   ├── FillEvent                   # Execution fills
│   │   ├── RiskEvent                   # Risk alerts
│   │   └── EventBus                    # Pub/sub message bus
│   │
│   ├── types.py                        # Domain objects & exceptions
│   │   ├── OHLCV                       # Price bar dataclass
│   │   ├── Bar                         # Extended bar with metadata
│   │   ├── Trade                       # Executed trade record
│   │   ├── Position                    # Open position tracking
│   │   ├── Order                       # Order representation
│   │   ├── Signal                      # Trading signal
│   │   ├── SignalStrength (enum)       # WEAK/MODERATE/STRONG
│   │   ├── PortfolioState              # Portfolio snapshot
│   │   ├── PerformanceMetrics          # Performance stats
│   │   └── [15+ Exception Classes]     # Typed exceptions
│   │
│   └── interfaces.py                   # Abstract protocols
│       ├── DataProvider                # Data source interface
│       ├── Strategy                    # Strategy interface
│       ├── RiskManager                 # Risk interface
│       ├── ExecutionHandler            # Execution interface
│       ├── PortfolioManager            # Portfolio interface
│       ├── Model                       # ML model interface
│       └── FeatureGenerator            # Feature interface
│
├── 📁 data/                            # ═══ DATA LAYER ═══
│   ├── __init__.py                     # Module exports
│   │
│   ├── 📁 storage/                     # Raw market data (CSV)
│   │   ├── AAPL_15min.csv              # 72,261 bars per symbol
│   │   ├── GOOGL_15min.csv             # ~2.5 years of data
│   │   ├── MSFT_15min.csv              # 15-minute timeframe
│   │   └── ... (46 symbols total)      # Dow Jones + NASDAQ 100
│   │
│   ├── 📁 processed/                   # Cleaned parquet files
│   │   └── [auto-generated]            # Optimized format
│   │
│   ├── 📁 cache/                       # Runtime cache
│   │   └── [auto-generated]            # LRU cache files
│   │
│   ├── loader.py                       # Data loading utilities
│   │   ├── DataLoader (protocol)       # Abstract loader
│   │   ├── CSVLoader                   # CSV file loader
│   │   ├── load_csv_data()             # Single symbol load
│   │   └── load_all_symbols()          # Batch load
│   │
│   ├── processor.py                    # Data processing
│   │   ├── DataProcessor               # Main processor class
│   │   ├── DataValidator               # Validation rules
│   │   ├── clean_ohlcv_data()          # Remove anomalies
│   │   ├── resample_ohlcv()            # Timeframe conversion
│   │   └── normalize_data()            # Standardization
│   │
│   └── provider.py                     # Unified data access
│       ├── HistoricalDataProvider      # Historical data
│       └── DataProviderFactory         # Provider factory
│
├── 📁 features/                        # ═══ FEATURE ENGINEERING ═══
│   ├── __init__.py                     # Module exports
│   │
│   ├── technical.py                    # Technical indicators (167 total)
│   │   ├── MomentumIndicators          # RSI, MACD, Stochastic, etc.
│   │   ├── TrendIndicators             # SMA, EMA, ADX, Supertrend
│   │   ├── VolatilityIndicators        # ATR, BB, Keltner, etc.
│   │   ├── VolumeIndicators            # OBV, VWAP, MFI, etc.
│   │   └── CustomIndicators            # Proprietary indicators
│   │
│   ├── statistical.py                  # Statistical features
│   │   ├── ReturnFeatures              # Returns, log returns
│   │   ├── CorrelationFeatures         # Rolling correlations
│   │   ├── DistributionFeatures        # Skew, kurtosis
│   │   └── RegimeDetection             # Market regime classifier
│   │
│   └── pipeline.py                     # Feature orchestration
│       ├── FeatureConfig               # Pipeline configuration
│       ├── FeatureCategory (enum)      # MOMENTUM/TREND/VOLUME/etc.
│       ├── FeaturePipeline             # Main pipeline class
│       ├── create_default_config()     # Default settings
│       └── generate_all_features()     # Full generation
│
├── 📁 strategies/                      # ═══ TRADING STRATEGIES ═══
│   ├── __init__.py                     # Registry & exports
│   │
│   ├── base.py                         # Strategy foundation
│   │   ├── StrategyState (enum)        # INITIALIZED/RUNNING/STOPPED
│   │   ├── SignalAction (enum)         # BUY/SELL/HOLD
│   │   ├── StrategyConfig              # Base configuration
│   │   ├── StrategyMetrics             # Performance tracking
│   │   ├── BaseStrategy                # Abstract base class
│   │   └── StrategyCombiner            # Multi-strategy ensemble
│   │
│   ├── momentum.py                     # Momentum strategies
│   │   ├── TrendFollowingStrategy      # MA crossover, ADX filter
│   │   ├── BreakoutStrategy            # Price channel breakouts
│   │   ├── MeanReversionStrategy       # Bollinger band reversal
│   │   ├── DualMomentumStrategy        # Relative + absolute momentum
│   │   ├── RSIDivergenceStrategy       # RSI divergence detection
│   │   └── MACDStrategy                # MACD signal generation
│   │
│   ├── statistical.py                  # Statistical arbitrage
│   │   ├── PairsTradingStrategy        # Pairs trading
│   │   ├── CointegrationStrategy       # Cointegration-based
│   │   ├── KalmanFilterStrategy        # Kalman filter hedge ratio
│   │   └── OUProcessStrategy           # Ornstein-Uhlenbeck model
│   │
│   └── alpha_ml.py                     # ML-based strategy
│       ├── MarketRegime (enum)         # TRENDING/RANGING/VOLATILE
│       ├── AlphaMLConfig               # ML strategy config
│       └── AlphaMLStrategy             # Ensemble ML strategy
│           ├── LightGBM (40% weight)   # Primary model
│           ├── XGBoost (40% weight)    # Secondary model
│           └── Neural (20% weight)     # Deep learning model
│
├── 📁 models/                          # ═══ MACHINE LEARNING ═══
│   ├── __init__.py                     # Registry & exports
│   │
│   ├── 📁 artifacts/                   # Saved models
│   │   ├── AAPL_lightgbm_*.pkl         # Trained models
│   │   ├── AAPL_lightgbm_*.json        # Training results
│   │   └── ... (per symbol/model)      # Auto-organized
│   │
│   ├── base.py                         # Model foundation
│   │   ├── ModelType (enum)            # CLASSIFIER/REGRESSOR/RL
│   │   ├── ModelState (enum)           # UNTRAINED/TRAINED/DEPLOYED
│   │   ├── ValidationMethod (enum)     # HOLDOUT/KFOLD/PURGED
│   │   ├── ModelConfig                 # Base configuration
│   │   ├── ClassificationMetrics       # Accuracy, F1, AUC
│   │   ├── RegressionMetrics           # MSE, MAE, R2
│   │   ├── BaseModel                   # Abstract base class
│   │   └── ModelRegistry               # Model registration
│   │
│   ├── classifiers.py                  # Gradient boosting models
│   │   ├── LightGBMClassifier          # LightGBM wrapper
│   │   ├── XGBoostClassifier           # XGBoost wrapper
│   │   ├── CatBoostClassifier          # CatBoost wrapper
│   │   ├── RandomForestClassifier      # RF wrapper
│   │   ├── ExtraTreesClassifier        # Extra Trees wrapper
│   │   ├── StackingClassifier          # Stacked ensemble
│   │   ├── VotingClassifier            # Voting ensemble
│   │   └── create_classifier()         # Factory function
│   │
│   ├── deep.py                         # Deep learning models
│   │   ├── DeepLearningConfig          # DL configuration
│   │   ├── LSTMConfig                  # LSTM-specific config
│   │   ├── TransformerConfig           # Transformer config
│   │   ├── TCNConfig                   # TCN config
│   │   ├── LSTMModel                   # LSTM implementation
│   │   ├── TransformerModel            # Attention-based model
│   │   ├── TCNModel                    # Temporal CNN
│   │   └── create_deep_model()         # Factory function
│   │
│   ├── reinforcement.py                # Reinforcement learning
│   │   ├── RLAction (enum)             # SELL/HOLD/BUY
│   │   ├── RLConfig                    # RL configuration
│   │   ├── ReplayBuffer                # Experience replay
│   │   ├── TradingEnvironment          # Gym-like environment
│   │   ├── DQNAgent                    # Deep Q-Network
│   │   ├── PPOAgent                    # Proximal Policy Opt
│   │   └── create_rl_agent()           # Factory function
│   │
│   └── training.py                     # Training infrastructure
│       ├── OptimizationDirection       # MINIMIZE/MAXIMIZE
│       ├── SamplerType (enum)          # TPE/CMA-ES/RANDOM
│       ├── PrunerType (enum)           # MEDIAN/HYPERBAND
│       ├── OptimizationConfig          # Optuna configuration
│       ├── TrainingConfig              # Training configuration
│       ├── PurgedKFold                 # Purged cross-validation
│       ├── CombinatorialPurgedKFold    # Combinatorial CV
│       ├── HyperparameterOptimizer     # Optuna wrapper
│       ├── TrainingPipeline            # Main training class
│       ├── quick_train()               # Quick training
│       ├── auto_ml()                   # Automated ML
│       └── PARAM_SPACES                # Hyperparameter spaces
│
├── 📁 backtesting/                     # ═══ BACKTESTING ENGINE ═══
│   ├── __init__.py                     # Module exports
│   │
│   ├── 📁 reports/                     # Generated reports
│   │   ├── backtest_*.html             # HTML reports
│   │   └── backtest_*.json             # JSON results
│   │
│   ├── engine.py                       # Core backtester
│   │   ├── BacktestMode (enum)         # VECTORIZED/EVENT_DRIVEN
│   │   ├── OrderFillMode (enum)        # CLOSE/NEXT_OPEN/VWAP
│   │   ├── BacktestConfig              # Backtest configuration
│   │   ├── PortfolioTracker            # Portfolio state tracking
│   │   ├── BacktestEngine              # Main engine class
│   │   ├── WalkForwardResult           # WF analysis result
│   │   ├── WalkForwardAnalyzer         # Walk-forward testing
│   │   ├── ReportGenerator             # Report generation
│   │   ├── run_backtest()              # Full backtest
│   │   └── quick_backtest()            # Quick test
│   │
│   ├── execution.py                    # Execution simulation
│   │   ├── SlippageModel (protocol)    # Slippage interface
│   │   ├── NoSlippage                  # Zero slippage
│   │   ├── FixedSlippage               # Fixed amount
│   │   ├── PercentageSlippage          # Percentage-based
│   │   ├── VolumeSlippage              # Volume-aware
│   │   ├── MarketImpactSlippage        # Market impact model
│   │   ├── CommissionModel (protocol)  # Commission interface
│   │   ├── IBKRCommission              # Interactive Brokers
│   │   ├── FillModel (protocol)        # Fill interface
│   │   ├── ExecutionSimulator          # Full simulator
│   │   └── create_realistic_simulator()# Factory
│   │
│   └── metrics.py                      # Performance analytics
│       ├── ReturnMetrics               # Return calculations
│       ├── RiskMetrics                 # Risk calculations
│       ├── TradeMetrics                # Trade statistics
│       ├── PerformanceAnalyzer         # Main analyzer
│       ├── calculate_sharpe_ratio()    # Sharpe calculation
│       ├── calculate_sortino_ratio()   # Sortino calculation
│       ├── calculate_max_drawdown()    # Drawdown analysis
│       ├── calculate_calmar_ratio()    # Calmar ratio
│       └── calculate_information_ratio()# IR calculation
│
├── 📁 risk/                            # ═══ RISK MANAGEMENT ═══
│   ├── __init__.py                     # Module exports
│   │
│   └── manager.py                      # Risk management
│       ├── RiskMetricType (enum)       # VAR/CVAR/VOLATILITY
│       ├── RiskLevel (enum)            # LOW/MEDIUM/HIGH/CRITICAL
│       ├── RiskConfig                  # Risk configuration
│       ├── RiskLimits                  # Position/portfolio limits
│       ├── PositionSizer               # Position sizing
│       │   ├── fixed_size()            # Fixed dollar amount
│       │   ├── percent_risk()          # Percentage risk
│       │   ├── kelly_criterion()       # Kelly sizing
│       │   └── volatility_target()     # Vol-targeted
│       ├── VaRCalculator               # Value at Risk
│       │   ├── historical_var()        # Historical VaR
│       │   ├── parametric_var()        # Parametric VaR
│       │   └── monte_carlo_var()       # MC simulation
│       ├── DrawdownMonitor             # Drawdown tracking
│       ├── RiskManager                 # Main risk class
│       └── CircuitBreaker              # Emergency stops
│
├── 📁 portfolio/                       # ═══ PORTFOLIO MANAGEMENT ═══
│   ├── __init__.py                     # Module exports
│   │
│   └── optimizer.py                    # Portfolio optimization
│       ├── OptimizationMethod (enum)   # MVO/RISK_PARITY/BLACK_LITTERMAN
│       ├── ConstraintType (enum)       # LONG_ONLY/LEVERAGE/etc.
│       ├── PortfolioConfig             # Configuration
│       ├── PortfolioOptimizer          # Main optimizer
│       │   ├── mean_variance()         # Markowitz MVO
│       │   ├── min_variance()          # Minimum variance
│       │   ├── max_sharpe()            # Maximum Sharpe
│       │   ├── risk_parity()           # Risk parity
│       │   ├── hierarchical_risk_parity()# HRP
│       │   └── black_litterman()       # BL model
│       ├── PortfolioRebalancer         # Rebalancing logic
│       └── TransactionCostOptimizer    # Cost-aware optimization
│
├── 📁 execution/                       # ═══ ORDER EXECUTION ═══
│   ├── __init__.py                     # Module exports
│   │
│   ├── broker.py                       # Broker integration
│   │   ├── BrokerType (enum)           # ALPACA/PAPER/IBKR
│   │   ├── OrderType (enum)            # MARKET/LIMIT/STOP
│   │   ├── TimeInForce (enum)          # DAY/GTC/IOC
│   │   ├── BrokerConfig                # Broker configuration
│   │   ├── BaseBroker (protocol)       # Broker interface
│   │   ├── AlpacaBroker                # Alpaca implementation
│   │   ├── PaperBroker                 # Paper trading
│   │   └── BrokerFactory               # Broker factory
│   │
│   ├── algorithms.py                   # Execution algorithms
│   │   ├── ExecutionStyle (enum)       # AGGRESSIVE/PASSIVE
│   │   ├── TWAPExecutor                # Time-weighted avg
│   │   ├── VWAPExecutor                # Volume-weighted avg
│   │   ├── IcebergExecutor             # Iceberg orders
│   │   ├── SmartRouter                 # Smart order routing
│   │   └── ExecutionAlgorithmFactory   # Factory
│   │
│   └── live_engine.py                  # Live trading engine
│       ├── TradingEngineState (enum)   # IDLE/RUNNING/STOPPED
│       ├── LiveTradingConfig           # Configuration
│       ├── LiveTradingEngine           # Main engine
│       └── run_paper_trading()         # Paper trading runner
│
├── 📁 api/                             # ═══ REST API ═══
│   ├── __init__.py                     # Module exports
│   │
│   └── main.py                         # FastAPI application
│       ├── /health                     # Health check
│       ├── /status                     # System status
│       ├── POST /backtest              # Queue backtest
│       ├── GET /backtest/{id}          # Get backtest result
│       ├── POST /models/train          # Train model
│       ├── GET /models                 # List models
│       ├── POST /models/{id}/predict   # Make prediction
│       ├── GET /strategies             # List strategies
│       ├── GET /data/symbols           # List symbols
│       ├── POST /data                  # Load data
│       └── WS /ws                      # WebSocket updates
│
├── 📁 tests/                           # ═══ TEST SUITE ═══
│   ├── __init__.py                     # Test module
│   ├── conftest.py                     # Shared fixtures
│   │   ├── sample_ohlcv_data           # OHLCV fixture
│   │   ├── sample_multi_symbol_data    # Multi-symbol fixture
│   │   ├── sample_features_data        # Feature fixture
│   │   ├── sample_ml_data              # ML data fixture
│   │   ├── trend_following_strategy    # Strategy fixture
│   │   ├── lightgbm_model              # Model fixture
│   │   └── backtest_engine             # Engine fixture
│   │
│   ├── test_data.py                    # Data layer tests
│   ├── test_features.py                # Feature tests
│   ├── test_strategies.py              # Strategy tests
│   ├── test_models.py                  # ML model tests
│   └── test_backtesting.py             # Backtest tests
│
├── 📁 scripts/                         # ═══ CLI SCRIPTS ═══
│   ├── run_backtest.py                 # Backtest runner
│   │   └── Interactive/CLI backtest execution
│   │
│   ├── train_model.py                  # ML training CLI
│   │   ├── --symbol AAPL               # Single symbol
│   │   ├── --symbols AAPL GOOGL        # Multiple symbols
│   │   ├── --model lightgbm            # Model selection
│   │   ├── --optimize                  # Hyperparameter opt
│   │   ├── --n-trials 50               # Optuna trials
│   │   ├── --walk-forward              # Walk-forward CV
│   │   ├── --compare-models            # Model comparison
│   │   └── --epochs 100                # DL epochs
│   │
│   ├── paper_trade.py                  # Paper trading
│   │   ├── --symbols AAPL GOOGL        # Symbols to trade
│   │   ├── --capital 100000            # Starting capital
│   │   ├── --strategy alpha_ml         # Strategy
│   │   └── --duration 60               # Duration (mins)
│   │
│   └── validate_backtest.py            # Report validation
│
├── 📁 notebooks/                       # ═══ RESEARCH ═══
│   ├── 01_data_exploration.ipynb       # Data analysis
│   ├── 02_feature_engineering.ipynb    # Feature research
│   ├── 03_strategy_research.ipynb      # Strategy dev
│   └── 04_model_training.ipynb         # ML experiments
│
├── 📁 logs/                            # ═══ LOGGING ═══
│   └── [auto-generated]                # Structured logs
│
├── .env.example                        # Environment template
├── .env                                # Local configuration
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Production deps
├── requirements-dev.txt                # Development deps
├── Makefile                            # Build commands
├── main.py                             # Application entry
├── README.md                           # Documentation
├── ML_EXECUTION_GUIDE.md               # ML workflow guide
└── PROJECT_ARCHITECTURE.md             # This file
```

---

## Component Details

### Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  CSV Files   │────▶│  CSVLoader   │────▶│DataProcessor │────▶│   Features   │
│  (46 files)  │     │   (Polars)   │     │   (Clean)    │     │  (167 cols)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
                                                                       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Signals    │◀────│    Model     │◀────│   Training   │◀────│  ML Dataset  │
│  (Actions)   │     │ (Prediction) │     │  (Optuna)    │     │   (X, y)     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Model Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           ENSEMBLE PREDICTOR            │
                    │  ┌───────────────────────────────────┐  │
                    │  │         Weighted Voting           │  │
                    │  └───────────────────────────────────┘  │
                    │       ▲           ▲           ▲         │
                    │       │           │           │         │
                    │  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐   │
                    │  │LightGBM │ │ XGBoost │ │  LSTM   │   │
                    │  │  (40%)  │ │  (40%)  │ │  (20%)  │   │
                    │  └─────────┘ └─────────┘ └─────────┘   │
                    └─────────────────────────────────────────┘
                                       ▲
                                       │
                    ┌─────────────────────────────────────────┐
                    │            167 FEATURES                 │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
                    │  │Momentum │ │  Trend  │ │Volatil. │   │
                    │  │ (35)    │ │  (28)   │ │  (24)   │   │
                    │  └─────────┘ └─────────┘ └─────────┘   │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
                    │  │ Volume  │ │  Stats  │ │ Custom  │   │
                    │  │  (20)   │ │  (35)   │ │  (25)   │   │
                    │  └─────────┘ └─────────┘ └─────────┘   │
                    └─────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Language | Python | 3.11+ |
| Data Processing | Polars | 0.20+ |
| ML Framework | scikit-learn | 1.4+ |
| Gradient Boosting | LightGBM, XGBoost, CatBoost | Latest |
| Deep Learning | PyTorch | 2.0+ |
| Hyperparameter Optimization | Optuna | 3.5+ |
| API Framework | FastAPI | 0.109+ |
| Validation | Pydantic | 2.0+ |
| Logging | structlog | 24.0+ |
| Testing | pytest | 8.0+ |
| Broker | Alpaca | Latest |

---

*Document Version: 2.0.0 | Last Updated: 2025-12-07*
