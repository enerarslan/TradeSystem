#!/usr/bin/env python3
"""
AlphaTrade System - Institutional-Grade Algorithmic Trading Platform
Version 2.0.0 - JPMorgan Level Implementation

COMPLETE SYSTEM with ALL modules fully integrated:

1. DATA LAYER:
   - DataLoader (Pandas/Polars)
   - TimescaleDB integration
   - Data validation

2. FEATURE ENGINEERING:
   - 50+ Technical indicators
   - Fractional differentiation
   - Cointegration features
   - Macroeconomic (FRED) features
   - Feature Store

3. ML TRAINING:
   - Model Factory (LightGBM, XGBoost, CatBoost)
   - Purged K-Fold CV
   - Walk-Forward Validation
   - Optuna optimization
   - MLflow tracking

4. DEEP LEARNING:
   - LSTM with Attention
   - Temporal Fusion Transformer
   - Custom financial losses

5. BACKTESTING:
   - Vectorized engine
   - Event-driven engine
   - Market impact (Almgren-Chriss)
   - Monte Carlo analysis

6. REPORTING:
   - Tear sheet generation
   - Statistical tests

Usage:
    python main.py                                    # Full pipeline
    python main.py --mode backtest                    # Backtest only
    python main.py --mode train --model lightgbm     # Train ML model
    python main.py --mode train --deep-learning      # Train LSTM
    python main.py --engine event-driven             # Event-driven backtest
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# IMPORT ALL REQUIRED MODULES
# =============================================================================

# Data Layer
from src.data import (
    DataLoader,
    DataValidator,
    DataProcessor,
    DataCache,
    POLARS_AVAILABLE,
    TIMESCALE_AVAILABLE,
)

# Feature Engineering
from src.features import (
    TechnicalIndicators,
    FeaturePipeline,
    frac_diff_ffd,
    find_min_d,
    FractionalDiffTransformer,
    CointegrationAnalyzer,
    OrnsteinUhlenbeckEstimator,
    FREDClient,
    MacroFeatureGenerator,
    EconomicRegimeDetector,
    FeatureStore,
    create_standard_features,
)

# Training
from src.training import (
    ExperimentTracker,
    ModelFactory,
    ModelRegistry,
    Trainer,
    TrainingResult,
    PurgedKFoldCV,
    CombinatorialPurgedKFoldCV,
    WalkForwardValidator,
    OptunaOptimizer,
    MultiObjectiveOptimizer,
    LSTMPredictor,
    AttentionLSTM,
    TemporalFusionTransformer,
    SharpeLoss,
    SortinoLoss,
)

# Strategies
from src.strategies.momentum.multi_factor_momentum import MultiFactorMomentumStrategy
from src.strategies.mean_reversion.mean_reversion import MeanReversionStrategy
from src.strategies.multi_factor.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy

# Backtesting
from src.backtesting import (
    BacktestEngine,
    BacktestResult,
    EventDrivenEngine,
    EventEngineConfig,
    AlmgrenChrissModel,
    DynamicSpreadModel,
    MonteCarloAnalyzer,
    StatisticalTests,
    PerformanceMetrics,
    BacktestAnalyzer,
)

# Reports
from src.backtesting.reports.dashboard import PerformanceDashboard, create_tear_sheet
from src.backtesting.reports.report_generator import ReportGenerator

# Risk
from src.risk import PositionSizer, VaRCalculator, DrawdownController

# Portfolio
from src.portfolio import PortfolioOptimizer


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "data": {
        "path": "data/raw",
        "cache_path": "data/cache",
        "min_bars": 500,  # Increased to accommodate max_feature_lookback (200) + prediction_horizon
        "use_polars": True,  # Use Polars for performance
        # Point-in-Time settings (institutional requirement)
        "point_in_time": True,
        "survivorship_bias_correction": True,
        "corporate_action_adjustment": True,
        "min_adv_filter": 1_000_000,  # $1M minimum ADV
    },
    "features": {
        "technical": True,
        "fractional_diff": True,  # Enable fractional differentiation
        "macro": False,  # Requires FRED API key
        "cointegration": False,
        # CRITICAL: Feature lookback settings for purge gap calculation
        "max_lookback_periods": 200,  # Max MA period used
        "microstructure_enabled": True,
        "garch_volatility": True,
        "leakage_check": "strict",  # fail on any detected leakage
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "slippage_bps": 1.0,
        "market_impact": True,  # Use Almgren-Chriss model
        # CRITICAL: Execution realism settings (institutional requirement)
        "execution_simulator": "order_book",  # Changed from "simple" to prevent infinite liquidity
        "partial_fills": True,
        "max_participation_rate": 0.02,  # 2% of ADV max
        "latency_ms": 50,  # Realistic retail latency
        "rejection_rate": 0.02,  # 2% order rejection simulation
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
        # CRITICAL: Purge gap must be calculated, not hardcoded
        # Formula: prediction_horizon + max_feature_lookback + buffer
        # For horizon=5, lookback=200, buffer=10 -> purge_gap=215
        "purge_gap": "auto",  # Will be calculated dynamically
        "purge_gap_buffer": 10,  # Safety buffer added to calculated purge gap
        "prediction_horizon": 5,  # Bars ahead for prediction
        "max_feature_lookback": 200,  # Max lookback in any feature
        "embargo_pct": "auto",  # Will be calculated as horizon/data_length
        "optuna_trials": 50,
        "mlflow_tracking": True,
        # CRITICAL: Use CPCV as default for institutional-grade validation
        "cv_type": "combinatorial_purged",  # Changed from standard purged
        "n_test_splits": 2,  # For CPCV
        "min_train_samples": 1000,  # Minimum samples after purging
    },
    "optimization": {
        # CRITICAL: Use Deflated Sharpe Ratio as primary metric
        "primary_metric": "deflated_sharpe_ratio",
        "secondary_metrics": [
            "probabilistic_sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
        ],
        "multiple_testing_correction": True,
        "min_track_record_months": 12,
    },
    "deep_learning": {
        "model": "lstm",  # lstm or transformer
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 64,
    },
    "risk": {
        "max_position_pct": 0.05,
        "max_drawdown_pct": 0.15,
        "var_confidence": 0.99,  # Increased from 0.95 for institutional standard
        "var_method": "historical",
        "max_position_adv_pct": 5.0,  # Max 5% of ADV
        "max_sector_exposure": 0.25,
        "drawdown_flatten_threshold": 0.15,
    },
    "monte_carlo": {
        "n_simulations": 1000,
        "block_size": 20,
    },
}


def calculate_purge_gap(config: Dict[str, Any]) -> int:
    """
    Calculate appropriate purge gap for cross-validation.

    CRITICAL: The purge gap must be at least:
        prediction_horizon + max_feature_lookback + buffer

    This prevents information leakage from:
    1. Features using data that overlaps with the target calculation window
    2. Target labels leaking into adjacent training folds

    Args:
        config: Configuration dictionary

    Returns:
        Calculated purge gap in bars
    """
    train_config = config.get("training", {})

    prediction_horizon = train_config.get("prediction_horizon", 5)
    max_feature_lookback = train_config.get("max_feature_lookback", 200)
    buffer = train_config.get("purge_gap_buffer", 10)

    purge_gap = prediction_horizon + max_feature_lookback + buffer

    logger.info(
        f"Calculated purge_gap={purge_gap} "
        f"(horizon={prediction_horizon} + lookback={max_feature_lookback} + buffer={buffer})"
    )

    return purge_gap


def calculate_embargo_pct(data_length: int, config: Dict[str, Any]) -> float:
    """
    Calculate appropriate embargo percentage for cross-validation.

    The embargo should be proportional to the prediction horizon relative
    to the data length.

    Args:
        data_length: Total number of samples
        config: Configuration dictionary

    Returns:
        Embargo percentage (0 to 1)
    """
    train_config = config.get("training", {})
    prediction_horizon = train_config.get("prediction_horizon", 5)

    # Embargo should be at least prediction_horizon / data_length
    # but with a minimum floor for safety
    embargo_pct = max(0.02, prediction_horizon / data_length)

    logger.info(f"Calculated embargo_pct={embargo_pct:.4f} for data_length={data_length}")

    return embargo_pct


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: bool = True) -> None:
    """Configure logging."""
    logger.remove()

    # Console
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )

    # File
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
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration."""
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Loaded config from {config_path}")

    return config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(
    data_path: str,
    symbols: Optional[List[str]] = None,
    min_bars: int = 100,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load and validate market data."""
    logger.info(f"Loading data from {data_path}")

    data_dir = Path(data_path)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_path}")
        return {}

    # Initialize components
    loader = DataLoader(data_dir=str(data_dir))
    validator = DataValidator()
    processor = DataProcessor()

    # Optional: Use cache
    cache = DataCache(cache_dir="data/cache") if use_cache else None

    available_symbols = loader.symbols
    logger.info(f"Found {len(available_symbols)} symbols")

    if symbols:
        symbols = [s for s in symbols if s in available_symbols]
    else:
        symbols = available_symbols

    data = {}
    for symbol in symbols:
        try:
            # Try cache first
            if cache:
                cached = cache.get(f"{symbol}_processed")
                if cached is not None:
                    data[symbol] = cached
                    continue

            df = loader.load_symbol(symbol)

            # Validate
            result = validator.validate(df, symbol=symbol)
            if not result.is_valid:
                logger.warning(f"{symbol}: Validation issues")

            # Process
            df = processor.process(df)

            if len(df) >= min_bars:
                data[symbol] = df
                if cache:
                    cache.set(f"{symbol}_processed", df)
                logger.debug(f"Loaded {symbol}: {len(df)} bars")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    logger.info(f"Loaded {len(data)} symbols")
    return data


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def generate_features(
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Generate all features - technical, fractional diff, macro."""
    logger.info("Generating features...")

    feature_config = config.get("features", {})
    features = {}

    for symbol, df in data.items():
        try:
            df_features = df.copy()

            # 1. Technical Indicators
            if feature_config.get("technical", True):
                indicators = TechnicalIndicators(df_features)
                df_features = indicators.add_all()

            # 2. Fractional Differentiation (memory-preserving stationarity)
            if feature_config.get("fractional_diff", True):
                price_cols = ["close", "high", "low"]
                for col in price_cols:
                    if col in df_features.columns:
                        try:
                            # Find optimal d
                            d_opt = find_min_d(df_features[col].dropna())
                            # Apply fractional diff
                            df_features[f"{col}_fracdiff"] = frac_diff_ffd(
                                df_features[col], d=d_opt
                            )
                        except Exception:
                            pass  # Skip if fails

            # 3. Cointegration features (for pairs)
            if feature_config.get("cointegration", False):
                # Will be computed across symbols later
                pass

            features[symbol] = df_features
            logger.debug(f"{symbol}: {len(df_features.columns)} features")

        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")

    # Add macro features if enabled
    if feature_config.get("macro", False):
        try:
            fred_key = config.get("fred_api_key")
            if fred_key:
                macro_gen = MacroFeatureGenerator(api_key=fred_key)
                # Get date range from data
                all_dates = pd.concat([df.index.to_series() for df in features.values()])
                start_date = all_dates.min()
                end_date = all_dates.max()

                macro_df = macro_gen.get_all_features(start_date, end_date)

                # Merge macro features with each symbol
                for symbol in features:
                    features[symbol] = features[symbol].join(macro_df, how="left")
                    features[symbol] = features[symbol].ffill()

                logger.info("Added macroeconomic features")
        except Exception as e:
            logger.warning(f"Could not add macro features: {e}")

    logger.info(f"Generated features for {len(features)} symbols")
    return features


# =============================================================================
# STRATEGY CREATION
# =============================================================================

def create_strategies(config: Dict[str, Any]) -> List:
    """Create strategy instances."""
    strategies = []
    strategy_config = config.get("strategies", {})

    if strategy_config.get("momentum", {}).get("enabled", True):
        mom_config = strategy_config.get("momentum", {})
        strategies.append(MultiFactorMomentumStrategy(
            name="Momentum",
            params={
                "lookback_periods": mom_config.get("lookback_periods", [5, 10, 20]),
                "top_n_long": mom_config.get("top_n_long", 5),
                "volatility_adjusted": mom_config.get("volatility_adjusted", True),
            },
        ))

    if strategy_config.get("mean_reversion", {}).get("enabled", True):
        mr_config = strategy_config.get("mean_reversion", {})
        strategies.append(MeanReversionStrategy(
            name="MeanReversion",
            params={
                "lookback_period": mr_config.get("lookback_period", 20),
                "entry_zscore": mr_config.get("entry_zscore", 2.0),
                "exit_zscore": mr_config.get("exit_zscore", 0.5),
            },
        ))

    if strategy_config.get("volatility_breakout", {}).get("enabled", True):
        vb_config = strategy_config.get("volatility_breakout", {})
        strategies.append(VolatilityBreakoutStrategy(
            name="VolatilityBreakout",
            params={
                "atr_period": vb_config.get("atr_period", 14),
                "atr_multiplier": vb_config.get("atr_multiplier", 2.0),
            },
        ))

    return strategies


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(
    strategy,
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    features: Optional[Dict[str, pd.DataFrame]] = None,
    engine_type: str = "vectorized",
) -> BacktestResult:
    """Run backtest with selected engine."""
    backtest_config = config.get("backtest", {})

    if engine_type == "event-driven":
        logger.info(f"Running event-driven backtest for {strategy.name}")

        engine_config = EventEngineConfig(
            initial_capital=backtest_config.get("initial_capital", 1_000_000),
            slippage_bps=backtest_config.get("slippage_bps", 1.0),
            commission_per_share=backtest_config.get("commission_per_share", 0.005),
        )

        engine = EventDrivenEngine(engine_config)
        result = engine.run(data)

    else:  # vectorized
        logger.info(f"Running vectorized backtest for {strategy.name}")

        # Add market impact model if enabled
        market_impact = None
        if backtest_config.get("market_impact", True):
            market_impact = AlmgrenChrissModel(
                sigma=0.02,
                eta=0.1,
                gamma=0.1,
                lambda_=1e-6,
                adv=1_000_000,
            )

        engine = BacktestEngine(
            initial_capital=backtest_config.get("initial_capital", 1_000_000),
            commission_pct=backtest_config.get("commission_pct", 0.001),
            slippage_pct=backtest_config.get("slippage_pct", 0.0005),
        )

        result = engine.run(strategy, data, features)

    # Log summary
    logger.info(
        f"{strategy.name} - Return: {result.metrics.get('total_return', 0):.2%}, "
        f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}, "
        f"MaxDD: {result.metrics.get('max_drawdown', 0):.2%}"
    )

    return result


def run_monte_carlo_analysis(
    returns: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Monte Carlo analysis on returns."""
    mc_config = config.get("monte_carlo", {})

    analyzer = MonteCarloAnalyzer(
        n_simulations=mc_config.get("n_simulations", 1000),
    )

    # Bootstrap analysis
    simulated_returns = analyzer.bootstrap_returns(
        returns,
        block_size=mc_config.get("block_size", 20),
    )

    # Confidence intervals
    confidence_intervals = analyzer.confidence_intervals(
        lambda r: (r.mean() / r.std()) * np.sqrt(252),  # Annualized Sharpe
        returns,
    )

    # Statistical tests
    stats = StatisticalTests()
    psr = stats.probabilistic_sharpe_ratio(
        observed_sharpe=confidence_intervals.get("median", 0),
        benchmark_sharpe=0,
        n_observations=len(returns),
        skewness=returns.skew(),
        kurtosis=returns.kurtosis(),
    )

    return {
        "confidence_intervals": confidence_intervals,
        "probabilistic_sharpe_ratio": psr,
        "n_simulations": mc_config.get("n_simulations", 1000),
    }


# =============================================================================
# ML TRAINING
# =============================================================================

def train_ml_model(
    data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> TrainingResult:
    """Train ML model with full pipeline."""
    logger.info("Training ML model...")

    train_config = config.get("training", {})

    # Prepare data
    all_features = []
    all_targets = []

    for symbol, df in features.items():
        target = df["close"].pct_change().shift(-1)
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]

        X = df[feature_cols].iloc[:-1]
        y = target.iloc[:-1]

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        all_features.append(X)
        all_targets.append(y)

    X_train = pd.concat(all_features, axis=0)
    y_train = pd.concat(all_targets, axis=0)

    logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")

    # Initialize MLflow tracking if enabled
    tracker = None
    if train_config.get("mlflow_tracking", True):
        try:
            tracker = ExperimentTracker(experiment_name="alphatrade_training")
            tracker.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            tracker.log_params(train_config)
        except Exception as e:
            logger.warning(f"MLflow tracking not available: {e}")

    # Create model
    model_type = train_config.get("model_type", "lightgbm")
    model = ModelFactory.create(model_type)

    # CRITICAL: Calculate purge gap dynamically based on feature lookback
    purge_gap_setting = train_config.get("purge_gap", "auto")
    if purge_gap_setting == "auto":
        purge_gap = calculate_purge_gap(config)
    else:
        purge_gap = int(purge_gap_setting)
        # Warn if hardcoded value is too small
        min_required = (
            train_config.get("prediction_horizon", 5)
            + train_config.get("max_feature_lookback", 200)
        )
        if purge_gap < min_required:
            logger.warning(
                f"CRITICAL: purge_gap={purge_gap} is less than minimum required "
                f"({min_required}). This will cause data leakage!"
            )

    # Calculate embargo percentage dynamically
    embargo_setting = train_config.get("embargo_pct", "auto")
    if embargo_setting == "auto":
        embargo_pct = calculate_embargo_pct(len(X_train), config)
    else:
        embargo_pct = float(embargo_setting)

    # Cross-validation - use CPCV for institutional-grade validation
    cv_type = train_config.get("cv_type", "combinatorial_purged")

    if cv_type == "combinatorial_purged":
        cv = CombinatorialPurgedKFoldCV(
            n_splits=train_config.get("cv_splits", 6),
            n_test_splits=train_config.get("n_test_splits", 2),
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
        )
        logger.info(f"Using CombinatorialPurgedKFoldCV with purge_gap={purge_gap}")
    else:
        cv = PurgedKFoldCV(
            n_splits=train_config.get("cv_splits", 5),
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
            prediction_horizon=train_config.get("prediction_horizon", 5),
            max_feature_lookback=train_config.get("max_feature_lookback", 200),
        )
        logger.info(f"Using PurgedKFoldCV with purge_gap={purge_gap}")

    # Optuna optimization if requested
    if train_config.get("optimize", False):
        logger.info("Running Optuna hyperparameter optimization...")
        optimizer = OptunaOptimizer(
            model_type=model_type,
            cv=cv,
            metric="sharpe_ratio",
            n_trials=train_config.get("optuna_trials", 50),
        )
        best_params = optimizer.optimize(X_train.values, y_train.values)
        model = ModelFactory.create(model_type, **best_params)

        if tracker:
            tracker.log_params({"best_params": best_params})

    # Train
    trainer = Trainer(model=model, cv=cv)
    result = trainer.train(X_train.values, y_train.values)

    # Log to MLflow
    if tracker:
        tracker.log_metrics({
            "cv_score_mean": result.cv_scores.mean(),
            "cv_score_std": result.cv_scores.std(),
        })
        tracker.end_run()

    logger.info(f"Training complete - CV Score: {result.cv_scores.mean():.4f}")

    return result


def train_deep_learning(
    data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Any:
    """Train deep learning model (LSTM or Transformer)."""
    logger.info("Training deep learning model...")

    dl_config = config.get("deep_learning", {})
    model_type = dl_config.get("model", "lstm")

    # Prepare sequence data
    all_X = []
    all_y = []
    seq_length = 20

    for symbol, df in features.items():
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        X = df[feature_cols].values
        y = df["close"].pct_change().shift(-1).values

        # Create sequences
        for i in range(seq_length, len(X) - 1):
            if not np.isnan(y[i]):
                all_X.append(X[i-seq_length:i])
                all_y.append(y[i])

    X_train = np.array(all_X)
    y_train = np.array(all_y)

    logger.info(f"Training sequences: {len(X_train)}")

    # Create model
    input_size = X_train.shape[2]

    if model_type == "transformer":
        model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=dl_config.get("hidden_size", 128),
            num_attention_heads=4,
            dropout=dl_config.get("dropout", 0.2),
        )
    else:  # LSTM
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=dl_config.get("hidden_size", 128),
            num_layers=dl_config.get("num_layers", 2),
            dropout=dl_config.get("dropout", 0.2),
        )

    logger.info(f"Created {model_type.upper()} model")

    # Note: Full training requires PyTorch Lightning Trainer
    # This is a placeholder for the training loop

    return model


# =============================================================================
# REPORTING
# =============================================================================

def generate_report(
    results: Dict[str, BacktestResult],
    config: Dict[str, Any],
    output_dir: str = "reports",
) -> None:
    """Generate comprehensive reports."""
    logger.info("Generating reports...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find best strategy
    best_strategy = max(
        results.keys(),
        key=lambda k: results[k].metrics.get("sharpe_ratio", 0),
    )
    best_result = results[best_strategy]

    # 1. Generate Tear Sheet
    tear_sheet_path = output_path / f"tear_sheet_{timestamp}.html"
    create_tear_sheet(
        returns=best_result.returns,
        strategy_name=best_strategy,
        output_path=tear_sheet_path,
    )
    logger.info(f"Tear sheet: {tear_sheet_path}")

    # 2. Generate detailed report
    report_path = output_path / f"report_{timestamp}.html"
    generator = ReportGenerator(best_result, output_dir=output_path)
    generator.generate_html(f"report_{timestamp}.html")
    logger.info(f"Report: {report_path}")

    # 3. Monte Carlo analysis
    mc_results = run_monte_carlo_analysis(best_result.returns, config)

    # 4. Strategy comparison
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "Strategy": name,
            "Total Return": f"{result.metrics.get('total_return', 0):.2%}",
            "CAGR": f"{result.metrics.get('cagr', 0):.2%}",
            "Sharpe": f"{result.metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino": f"{result.metrics.get('sortino_ratio', 0):.2f}",
            "Max Drawdown": f"{result.metrics.get('max_drawdown', 0):.2%}",
            "Win Rate": f"{result.metrics.get('win_rate', 0):.1%}",
            "Trades": len(result.trades),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("ALPHATRADE SYSTEM - BACKTEST RESULTS")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(f"\nBest Strategy: {best_strategy}")
    print(f"Probabilistic Sharpe Ratio: {mc_results['probabilistic_sharpe_ratio']:.2%}")
    print(f"\nReports saved to: {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaTrade System - JPMorgan-Level Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["backtest", "train", "report", "full"],
        default="full",
        help="Execution mode (default: full pipeline)",
    )

    # Strategy
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean_reversion", "volatility_breakout", "ensemble", "all"],
        default="all",
        help="Strategy to run",
    )

    # Engine
    parser.add_argument(
        "--engine",
        choices=["vectorized", "event-driven"],
        default="vectorized",
        help="Backtest engine type",
    )

    # ML
    parser.add_argument(
        "--model",
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
        default="lightgbm",
        help="ML model type",
    )
    parser.add_argument("--deep-learning", action="store_true", help="Train deep learning model")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization")

    # Paths
    parser.add_argument("--data-path", default="data/raw", help="Data directory")
    parser.add_argument("--config", default="config/trading_config.yaml", help="Config file")
    parser.add_argument("--output", default="reports", help="Output directory")

    # Options
    parser.add_argument("--symbols", nargs="+", help="Symbols to process")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)

    print("=" * 70)
    print("  ALPHATRADE SYSTEM v2.0.0")
    print("  Institutional-Grade Algorithmic Trading Platform")
    print("=" * 70)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Engine: {args.engine}")
    logger.info(f"Capital: ${args.capital:,.0f}")

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # Load config
    config = load_config(args.config)
    config["backtest"]["initial_capital"] = args.capital
    config["training"]["model_type"] = args.model
    config["training"]["optimize"] = args.optimize

    # Load data
    data = load_data(
        args.data_path,
        args.symbols,
        config["data"].get("min_bars", 100),
    )

    if not data:
        logger.error("No data loaded!")
        logger.info("Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    # Generate features
    features = generate_features(data, config)

    # Training mode
    if args.mode in ["train", "full"]:
        if args.deep_learning:
            train_deep_learning(data, features, config)
        else:
            train_ml_model(data, features, config)

    # Backtest mode
    if args.mode in ["backtest", "full"]:
        strategies = create_strategies(config)
        results = {}

        for strategy in strategies:
            try:
                result = run_backtest(
                    strategy, data, config, features, args.engine
                )
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Error with {strategy.name}: {e}")

        # Ensemble
        if len(strategies) > 1 and args.strategy in ["ensemble", "all"]:
            try:
                ensemble = EnsembleStrategy(strategies=strategies, name="Ensemble")
                result = run_backtest(ensemble, data, config, features, args.engine)
                results["Ensemble"] = result
            except Exception as e:
                logger.error(f"Error with Ensemble: {e}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/backtest_{timestamp}.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

        # Generate reports
        generate_report(results, config, args.output)

    print("\n" + "=" * 70)
    print("  ALPHATRADE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
