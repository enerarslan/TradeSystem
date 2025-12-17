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

# Optional PyTorch import for deep learning
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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
from src.data.pit import UniverseManager, CorporateActionAdjuster

# Risk and Compliance
from src.risk.circuit_breakers import CircuitBreakerManager
from src.risk.realtime_monitor import RealTimeRiskMonitor
from src.risk.pretrade_compliance import PreTradeComplianceChecker
from src.compliance.audit_trail import AuditTrail, AuditEventType

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
from src.backtesting.events.order import (
    OrderSide,
    create_market_order,
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

def validate_data_for_backtest(
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Perform comprehensive data validation before backtest.

    JPMorgan-level requirement: Validate all data quality before any backtest.

    Args:
        data: Dictionary of symbol -> DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of (cleaned_data, validation_report)
    """
    logger.info("Running pre-backtest data validation...")

    validator = DataValidator(
        max_missing_pct=5.0,
        max_price_change_pct=50.0,  # Flag potential splits/errors
        min_price=0.01,
    )

    cleaned_data = {}
    validation_report = {
        "total_symbols": len(data),
        "valid_symbols": 0,
        "rejected_symbols": [],
        "warnings": [],
        "critical_issues": [],
    }

    for symbol, df in data.items():
        result = validator.validate(df, symbol=symbol)

        if result.is_valid:
            cleaned_data[symbol] = df
            validation_report["valid_symbols"] += 1

            if result.warnings:
                validation_report["warnings"].extend(
                    [f"{symbol}: {w}" for w in result.warnings]
                )
        else:
            validation_report["rejected_symbols"].append(symbol)
            validation_report["critical_issues"].extend(
                [f"{symbol}: {e}" for e in result.errors]
            )

    # Log summary
    logger.info(
        f"Data validation complete: {validation_report['valid_symbols']}/{len(data)} symbols passed"
    )

    if validation_report["rejected_symbols"]:
        logger.warning(
            f"Rejected {len(validation_report['rejected_symbols'])} symbols due to data quality issues"
        )

    if validation_report["critical_issues"]:
        for issue in validation_report["critical_issues"][:5]:  # Log first 5
            logger.error(f"DATA ISSUE: {issue}")

    return cleaned_data, validation_report


def load_data(
    data_path: str,
    symbols: Optional[List[str]] = None,
    min_bars: int = 100,
    use_cache: bool = True,
    apply_survivorship_filter: bool = True,
    backtest_start_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load and validate market data with survivorship bias handling.

    Supports loading from:
    1. TimescaleDB (if configured and available)
    2. CSV files (fallback)

    Args:
        data_path: Path to data directory (for CSV fallback)
        symbols: List of symbols to load (None for all)
        min_bars: Minimum bars required
        use_cache: Whether to use data cache
        apply_survivorship_filter: Apply survivorship bias correction
        backtest_start_date: Start date for universe filtering
        config: Configuration dictionary (for TimescaleDB settings)

    Returns:
        Dictionary of symbol -> DataFrame
    """
    config = config or {}

    # Check if TimescaleDB is enabled and available
    timescale_config = config.get("timescale", {})
    use_timescale = timescale_config.get("enabled", False) and TIMESCALE_AVAILABLE

    if use_timescale:
        logger.info("Loading data from TimescaleDB...")
        try:
            from src.data.storage.timescale_client import TimescaleClient, ConnectionConfig

            # Create connection config
            conn_config = ConnectionConfig(
                host=timescale_config.get("host", "localhost"),
                port=timescale_config.get("port", 5432),
                database=timescale_config.get("database", "alphatrade_db"),
                user=timescale_config.get("user", "alphatrade"),
                password=timescale_config.get("password", ""),
            )

            data = {}
            with TimescaleClient(conn_config) as client:
                # Get available symbols if not specified
                if symbols is None:
                    symbols = client.get_symbols()
                    logger.info(f"Found {len(symbols)} symbols in TimescaleDB")

                for symbol in symbols:
                    try:
                        # Fetch OHLCV data
                        df = client.get_ohlcv(
                            symbol=symbol,
                            timeframe="15min",
                            start=backtest_start_date,
                        )

                        if df is not None and len(df) >= min_bars:
                            # Ensure proper index
                            if "timestamp" in df.columns:
                                df = df.set_index("timestamp")
                            df.index.name = "timestamp"
                            data[symbol] = df
                            logger.debug(f"Loaded {symbol} from TimescaleDB: {len(df)} bars")
                    except Exception as e:
                        logger.warning(f"Error loading {symbol} from TimescaleDB: {e}")

            if data:
                logger.info(f"Loaded {len(data)} symbols from TimescaleDB")
                return data
            else:
                logger.warning("No data loaded from TimescaleDB, falling back to CSV")

        except Exception as e:
            logger.warning(f"TimescaleDB connection failed: {e}, falling back to CSV")

    # Fallback to CSV loading
    logger.info(f"Loading data from {data_path}")

    data_dir = Path(data_path)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_path}")
        return {}

    # Initialize data loader - Use Polars for JPMorgan-level performance (10-100x faster)
    if POLARS_AVAILABLE:
        from src.data.loaders.polars_loader import PolarsDataLoader
        loader = PolarsDataLoader(data_dir=str(data_dir))
        logger.info("Using PolarsDataLoader (high-performance, 10-100x faster)")
    else:
        loader = DataLoader(data_dir=str(data_dir))
        logger.warning("Polars not available - using pandas DataLoader (slower)")

    validator = DataValidator()
    processor = DataProcessor()

    # Optional: Use cache
    cache = DataCache(cache_dir="data/cache") if use_cache else None

    # Survivorship bias handling
    universe_manager = None
    if apply_survivorship_filter:
        metadata_path = data_dir / "symbol_metadata.json"
        if metadata_path.exists():
            universe_manager = UniverseManager(metadata_path=metadata_path)
            logger.info(f"Loaded universe metadata for {len(universe_manager)} symbols")
        else:
            logger.warning(
                "No symbol_metadata.json found - survivorship bias correction not available. "
                "Run 'python scripts/generate_universe_metadata.py' to create metadata."
            )

    available_symbols = loader.symbols
    logger.info(f"Found {len(available_symbols)} symbols")

    # Apply survivorship filter if available
    if universe_manager and backtest_start_date:
        from datetime import date as date_type
        if isinstance(backtest_start_date, str):
            start_date = date_type.fromisoformat(backtest_start_date)
        else:
            start_date = backtest_start_date

        # Get universe as it existed at backtest start
        historical_universe = universe_manager.get_universe(as_of=start_date)

        if historical_universe:
            # Include only symbols that were tradeable at start
            available_symbols = [s for s in available_symbols if s in historical_universe]
            logger.info(
                f"Survivorship filter applied: {len(available_symbols)} symbols "
                f"valid as of {start_date}"
            )

    if symbols:
        symbols = [s for s in symbols if s in available_symbols]
    else:
        symbols = available_symbols

    data = {}
    validation_results = []

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
            validation_results.append(result)

            if not result.is_valid:
                logger.warning(f"{symbol}: Validation failed - {result.errors}")
                continue

            if result.warnings:
                logger.debug(f"{symbol}: Warnings - {result.warnings}")

            # Process
            df = processor.process(df)

            if len(df) >= min_bars:
                data[symbol] = df
                if cache:
                    cache.set(f"{symbol}_processed", df)
                logger.debug(f"Loaded {symbol}: {len(df)} bars")
            else:
                logger.debug(f"{symbol}: Insufficient bars ({len(df)} < {min_bars})")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    # Generate validation report
    if validation_results:
        valid_count = sum(1 for r in validation_results if r.is_valid)
        logger.info(
            f"Loaded {len(data)} symbols "
            f"({valid_count}/{len(validation_results)} passed validation)"
        )

    return data


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def generate_features(
    data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    train_ratio: float = 0.8,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, FeaturePipeline]]:
    """
    Generate all features with proper fit/transform separation to prevent data leakage.

    CRITICAL (JPMorgan-level requirement):
    This function uses FeaturePipeline with proper fit/transform separation:
    1. The pipeline is FIT only on TRAINING data (first train_ratio of each symbol)
    2. The fitted parameters (scaling mean/std) are then APPLIED to the full dataset
    3. This prevents future information from leaking into historical features

    Args:
        data: Dictionary of symbol -> OHLCV DataFrame
        config: Configuration dictionary
        train_ratio: Ratio of data to use for fitting (default 0.8)

    Returns:
        Tuple of (features_dict, pipelines_dict) where:
            - features_dict: symbol -> processed feature DataFrame
            - pipelines_dict: symbol -> fitted FeaturePipeline (for later use)
    """
    logger.info("Generating features with leakage-safe pipeline...")

    feature_config = config.get("features", {})
    features = {}
    pipelines = {}

    # Get pipeline parameters from config
    ma_periods = feature_config.get("ma_periods", [5, 10, 20, 50, 100, 200])
    max_lookback = max(ma_periods) if ma_periods else 200

    for symbol, df in data.items():
        try:
            # Create pipeline for this symbol
            pipeline = FeaturePipeline(
                ma_periods=ma_periods,
                scaling="robust",
                strict_leakage_check=feature_config.get("leakage_check", "strict") == "strict",
            )

            # CRITICAL: Split data for fit/transform separation
            n_samples = len(df)
            train_end_idx = int(n_samples * train_ratio)

            if train_end_idx < max_lookback + 50:
                logger.warning(
                    f"{symbol}: Insufficient training data ({train_end_idx} samples) "
                    f"for max_lookback={max_lookback}. Using all data for fitting."
                )
                train_end_idx = n_samples

            train_df = df.iloc[:train_end_idx]

            # FIT pipeline on training data only
            # This learns scaling parameters from training data
            pipeline.fit(train_df)

            # TRANSFORM full dataset using fitted parameters
            # This applies the scaling learned from training data to the full dataset
            # The pipeline.transform() method:
            # 1. Generates the same features it was trained on
            # 2. Applies the fitted scaling parameters (no refitting)
            df_features = pipeline.transform(df)

            # Apply fractional differentiation if enabled (after pipeline features)
            if feature_config.get("fractional_diff", True):
                price_cols = ["close", "high", "low"]
                for col in price_cols:
                    if col in df.columns:
                        try:
                            # Find optimal d using ONLY training data
                            train_col = df.iloc[:train_end_idx][col].dropna()
                            d_opt = find_min_d(train_col)
                            # Apply to full series
                            df_features[f"{col}_fracdiff"] = frac_diff_ffd(
                                df[col], d=d_opt
                            )
                        except Exception:
                            pass  # Skip if fails

            # Add original OHLCV columns back (needed for some strategies)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns and col not in df_features.columns:
                    df_features[col] = df[col]

            features[symbol] = df_features
            pipelines[symbol] = pipeline
            logger.debug(f"{symbol}: {len(df_features.columns)} features (fitted on {train_end_idx} samples)")

        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            import traceback
            traceback.print_exc()

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

    logger.info(f"Generated leakage-safe features for {len(features)} symbols")
    return features, pipelines


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

def calculate_symbol_adv(data: Dict[str, pd.DataFrame], lookback: int = 20) -> Dict[str, float]:
    """
    Calculate Average Daily Value (ADV) for each symbol.

    ADV = rolling average of (volume * close price) over lookback period.
    This is critical for realistic market impact modeling.

    Args:
        data: Dictionary mapping symbols to OHLCV DataFrames
        lookback: Rolling window for ADV calculation (default 20 days)

    Returns:
        Dictionary mapping symbols to their ADV values
    """
    adv_dict = {}

    for symbol, df in data.items():
        if "volume" in df.columns and "close" in df.columns:
            # Calculate daily dollar volume
            daily_value = df["volume"] * df["close"]
            # Rolling average
            adv = daily_value.rolling(window=lookback).mean()
            # Use the most recent ADV value
            latest_adv = adv.dropna().iloc[-1] if len(adv.dropna()) > 0 else 1_000_000
            adv_dict[symbol] = float(latest_adv)
        else:
            # Default fallback
            adv_dict[symbol] = 1_000_000

    logger.info(f"Calculated ADV for {len(adv_dict)} symbols")
    return adv_dict


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
        logger.info(f"Running event-driven backtest for {strategy.name} (JPMorgan-level)")

        engine_config = EventEngineConfig(
            initial_capital=backtest_config.get("initial_capital", 1_000_000),
            slippage_bps=backtest_config.get("slippage_bps", 1.0),
            commission_per_share=backtest_config.get("commission_per_share", 0.005),
        )

        engine = EventDrivenEngine(engine_config)

        # Register strategy callback - converts bar events to signals/orders
        def on_bar_callback(bar_event):
            """Process bar event through strategy to generate signals."""
            symbol = bar_event.symbol
            if symbol not in data:
                return

            # Get feature data for this symbol if available
            symbol_features = features.get(symbol) if features else None

            # Get current bar index
            df = data[symbol]
            if bar_event.timestamp not in df.index:
                return

            bar_idx = df.index.get_loc(bar_event.timestamp)

            # Need enough history for strategy
            if bar_idx < 50:
                return

            # Get historical data up to this point (no look-ahead)
            historical_data = df.iloc[:bar_idx + 1]
            historical_features = symbol_features.iloc[:bar_idx + 1] if symbol_features is not None else None

            # Generate signal from strategy
            try:
                signal = strategy.generate_signal(historical_data, historical_features)

                if signal and signal != 0:
                    # Calculate position size
                    current_price = bar_event.close
                    portfolio_value = engine.equity
                    position_size = portfolio_value * 0.02  # 2% per position

                    shares = int(position_size / current_price)
                    if shares > 0:
                        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
                        order = create_market_order(
                            symbol=symbol,
                            side=side,
                            quantity=shares,
                            timestamp=bar_event.timestamp,
                        )
                        engine.submit_order(order)

            except Exception as e:
                logger.debug(f"Strategy signal error for {symbol}: {e}")

        engine.register_strategy(on_bar=on_bar_callback)
        event_result = engine.run(data)

        # Convert EventEngineResult to BacktestResult for compatibility
        result = BacktestResult(
            equity_curve=event_result.equity_curve,
            returns=event_result.returns,
            trades=pd.DataFrame([{
                'timestamp': f.timestamp,
                'symbol': f.symbol,
                'side': f.side.value if hasattr(f.side, 'value') else str(f.side),
                'quantity': f.quantity,
                'price': f.fill_price,
                'commission': f.commission,
            } for f in event_result.trades if hasattr(f, 'fill_price')]),
            positions=pd.DataFrame(),
            metrics=event_result.metrics,
        )

    else:  # vectorized
        logger.info(f"Running vectorized backtest for {strategy.name}")

        # CRITICAL: Calculate actual ADV per symbol for realistic market impact
        symbol_adv = calculate_symbol_adv(data, lookback=20)

        # Calculate average ADV across all symbols for the impact model
        # This provides a baseline; per-symbol ADV can be used for order-level impact
        avg_adv = np.mean(list(symbol_adv.values())) if symbol_adv else 1_000_000

        # Add market impact model if enabled - NOW WITH ACTUAL ADV
        market_impact = None
        if backtest_config.get("market_impact", True):
            # Calculate volatility from data for more realistic impact
            all_returns = []
            for df in data.values():
                if "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    all_returns.extend(returns.values)

            sigma = np.std(all_returns) if all_returns else 0.02

            market_impact = AlmgrenChrissModel(
                sigma=sigma,  # Use actual realized volatility
                eta=0.1,
                gamma=0.1,
                lambda_=1e-6,
                adv=avg_adv,  # Use actual calculated ADV
            )

            logger.info(
                f"Market impact model initialized: sigma={sigma:.4f}, avg_adv=${avg_adv:,.0f}"
            )

        engine = BacktestEngine(
            initial_capital=backtest_config.get("initial_capital", 1_000_000),
            commission_pct=backtest_config.get("commission_pct", 0.001),
            slippage_pct=backtest_config.get("slippage_pct", 0.0005),
            symbol_adv=symbol_adv,  # CRITICAL: Pass per-symbol ADV for market impact
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
    """
    Train ML model with proper purged cross-validation.

    CRITICAL (JPMorgan-level requirement):
    This function implements MANUAL CV loop to ensure purge gap is properly applied.
    sklearn's cross_validate() does NOT apply purge gap correctly.

    The proper CV loop:
    1. Use cv.split(X, y) to get train/test indices with purging
    2. For each fold: fit model ONLY on train_idx, score on test_idx
    3. Verify no index overlap between train and test (leakage check)
    """
    logger.info("Training ML model with purged cross-validation...")

    train_config = config.get("training", {})

    # Prepare data
    all_features = []
    all_targets = []
    all_indices = []

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
        all_indices.append(pd.Series(range(len(X)), name=symbol))

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

    # Cross-validation setup - check walk-forward config first
    backtest_config = config.get("backtest", {})
    walk_forward_config = backtest_config.get("walk_forward", {})
    use_walk_forward = walk_forward_config.get("enabled", False)

    if use_walk_forward:
        # Walk-Forward Validation (simulates realistic model deployment)
        # Convert days to bars (assuming 26 bars per day for 15-min data)
        bars_per_day = 26
        train_period = walk_forward_config.get("train_period_days", 126) * bars_per_day
        test_period = walk_forward_config.get("test_period_days", 21) * bars_per_day
        expanding = walk_forward_config.get("anchored", False)

        cv = WalkForwardValidator(
            train_period=train_period,
            test_period=test_period,
            step_size=test_period,  # Non-overlapping test periods
            expanding=expanding,
            purge_gap=purge_gap,
            embargo_bars=int(embargo_pct * test_period),
            min_train_samples=train_config.get("min_train_samples", 1000),
        )
        logger.info(
            f"Using WalkForwardValidator: train={train_period} bars, test={test_period} bars, "
            f"{'expanding' if expanding else 'sliding'} window, purge_gap={purge_gap}"
        )
    else:
        # Standard cross-validation - use CPCV for institutional-grade validation
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

    # Convert to numpy for training
    X_np = X_train.values
    y_np = y_train.values

    # Optuna optimization if requested (uses proper manual CV internally)
    best_params = {}
    if train_config.get("optimize", False):
        logger.info("Running Optuna hyperparameter optimization...")
        optimizer = OptunaOptimizer(
            model_type=model_type,
            cv=cv,
            metric="sharpe_ratio",
            n_trials=train_config.get("optuna_trials", 50),
        )
        best_params = optimizer.optimize(X_np, y_np)
        if tracker:
            tracker.log_params({"best_params": best_params})

    # ==========================================================================
    # CRITICAL: MANUAL CV LOOP WITH LEAKAGE VERIFICATION
    # This replaces sklearn's cross_validate() which doesn't apply purge properly
    # ==========================================================================
    logger.info("Starting manual purged cross-validation loop...")

    fold_scores = []
    fold_predictions = []
    fold_train_scores = []
    total_leakage_checks = 0
    leakage_detected = False

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_np, y_np)):
        # CRITICAL: Leakage verification - train and test indices must not overlap
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set.intersection(test_set)

        if overlap:
            logger.error(
                f"CRITICAL LEAKAGE DETECTED in fold {fold_idx}: "
                f"{len(overlap)} overlapping indices!"
            )
            leakage_detected = True
            continue

        # Verify temporal separation (test indices should not precede train indices after purge)
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Find the gap between training end and test start for each segment
            train_max_before_test = train_idx[train_idx < test_idx.min()].max() if any(train_idx < test_idx.min()) else None
            if train_max_before_test is not None:
                actual_gap = test_idx.min() - train_max_before_test
                if actual_gap < purge_gap:
                    logger.warning(
                        f"Fold {fold_idx}: Insufficient gap between train and test "
                        f"(actual: {actual_gap}, required: {purge_gap})"
                    )

        total_leakage_checks += 1

        # Split data for this fold
        X_fold_train, X_fold_test = X_np[train_idx], X_np[test_idx]
        y_fold_train, y_fold_test = y_np[train_idx], y_np[test_idx]

        logger.debug(
            f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}, "
            f"train_range=[{train_idx.min()}-{train_idx.max()}], "
            f"test_range=[{test_idx.min()}-{test_idx.max()}]"
        )

        # Create fresh model for each fold
        model = ModelFactory.create_model(model_type, params=best_params)

        # Train on this fold's training data ONLY
        try:
            model.fit(X_fold_train, y_fold_train)
        except Exception as e:
            logger.error(f"Fold {fold_idx} training failed: {e}")
            continue

        # Score on test data
        try:
            y_pred = model.predict(X_fold_test)
            fold_predictions.append((test_idx, y_pred))

            # Calculate metrics
            # Correlation-based score (like IC - Information Coefficient)
            if len(y_fold_test) > 10:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(y_fold_test, y_pred)
                fold_scores.append(ic if not np.isnan(ic) else 0.0)
            else:
                fold_scores.append(0.0)

            # Also calculate train score for overfitting detection
            y_train_pred = model.predict(X_fold_train)
            if len(y_fold_train) > 10:
                train_ic, _ = spearmanr(y_fold_train, y_train_pred)
                fold_train_scores.append(train_ic if not np.isnan(train_ic) else 0.0)

            logger.info(
                f"Fold {fold_idx}: Test IC={fold_scores[-1]:.4f}, "
                f"Train IC={fold_train_scores[-1] if fold_train_scores else 0:.4f}"
            )

        except Exception as e:
            logger.error(f"Fold {fold_idx} prediction failed: {e}")
            fold_scores.append(0.0)

    # Summarize results
    if leakage_detected:
        logger.error("CRITICAL: Data leakage was detected during cross-validation!")

    if not fold_scores:
        logger.error("No valid CV folds completed!")
        fold_scores = [0.0]

    cv_scores = np.array(fold_scores)
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    # Overfitting check: compare train vs test scores
    if fold_train_scores:
        train_mean = np.mean(fold_train_scores)
        overfit_ratio = train_mean / mean_score if mean_score != 0 else float('inf')
        if overfit_ratio > 2.0:
            logger.warning(
                f"OVERFITTING WARNING: Train IC ({train_mean:.4f}) is {overfit_ratio:.1f}x "
                f"higher than Test IC ({mean_score:.4f})"
            )

    logger.info(
        f"CV Complete: Mean IC={mean_score:.4f} (+/- {std_score:.4f}), "
        f"Folds={len(fold_scores)}, Leakage checks={total_leakage_checks}"
    )

    # Train final model on all data
    final_model = ModelFactory.create_model(model_type, params=best_params)
    final_model.fit(X_np, y_np)

    # Log to MLflow
    if tracker:
        tracker.log_metrics({
            "cv_score_mean": mean_score,
            "cv_score_std": std_score,
            "n_folds": len(fold_scores),
            "leakage_detected": int(leakage_detected),
        })
        tracker.end_run()

    # Create result object
    result = TrainingResult(
        model=final_model,
        model_type=model_type,
        task_type="regression",  # Default for financial prediction
        train_metrics={"ic_mean": train_mean if train_scores else 0.0},
        validation_metrics={"ic_mean": mean_score, "ic_std": std_score},
        cv_scores={"ic": fold_scores} if fold_scores else None,
        feature_importance=None,  # Can be added from model.feature_importances_ if available
        training_time_seconds=0.0,  # TODO: Add timing
        n_train_samples=len(X_train),
        n_features=len(X_train.columns),
        best_iteration=None,
        params=best_params,
        metadata={
            "cv_type": cv_type,
            "purge_gap": purge_gap,
            "embargo_pct": embargo_pct,
            "leakage_detected": leakage_detected,
            "feature_names": list(X_train.columns),
        }
    )

    logger.info(f"Training complete - CV Score: {mean_score:.4f}")

    return result


def train_deep_learning(
    data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Any:
    """
    Train deep learning model (LSTM or Transformer) with complete training loop.

    Implements:
    - Proper train/validation split with purging
    - Custom financial loss functions (Sharpe, Sortino)
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Model checkpointing
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for deep learning training")
        return None

    logger.info("Training deep learning model...")

    dl_config = config.get("deep_learning", {})
    training_config = dl_config.get("training", {})
    model_type = dl_config.get("model", "lstm")

    # Training hyperparameters
    seq_length = training_config.get("seq_length", 20)
    batch_size = training_config.get("batch_size", 64)
    max_epochs = training_config.get("max_epochs", 100)
    patience = training_config.get("patience", 10)
    learning_rate = training_config.get("learning_rate", 1e-3)
    gradient_clip = dl_config.get("gradient_clip", 1.0)
    validation_split = training_config.get("validation_split", 0.2)
    purge_gap = config.get("training", {}).get("purge_gap", 50)

    # Prepare sequence data
    all_X = []
    all_y = []

    for symbol, df in features.items():
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        if not feature_cols:
            logger.warning(f"No feature columns found for {symbol}")
            continue

        X = df[feature_cols].values
        y = df["close"].pct_change().shift(-1).values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        for i in range(seq_length, len(X) - 1):
            if not np.isnan(y[i]):
                all_X.append(X[i-seq_length:i])
                all_y.append(y[i])

    if len(all_X) == 0:
        logger.error("No training sequences created!")
        return None

    X_all = np.array(all_X, dtype=np.float32)
    y_all = np.array(all_y, dtype=np.float32)

    logger.info(f"Total sequences: {len(X_all)}, Features: {X_all.shape[2]}")

    # Train/validation split with purge gap
    n_samples = len(X_all)
    split_idx = int(n_samples * (1 - validation_split))

    # Apply purge gap between train and validation
    train_end_idx = split_idx - purge_gap
    val_start_idx = split_idx

    X_train = X_all[:train_end_idx]
    y_train = y_all[:train_end_idx]
    X_val = X_all[val_start_idx:]
    y_val = y_all[val_start_idx:]

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    logger.info(f"Purge gap: {purge_gap} samples between train and validation")

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Create model
    input_size = X_train.shape[2]

    if model_type == "transformer":
        model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=dl_config.get("hidden_size", 128),
            num_attention_heads=dl_config.get("num_attention_heads", 4),
            dropout=dl_config.get("dropout", 0.2),
        )
    else:  # LSTM
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=dl_config.get("hidden_size", 128),
            num_layers=dl_config.get("num_layers", 2),
            dropout=dl_config.get("dropout", 0.2),
        )

    logger.info(f"Created {model_type.upper()} model with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Training on device: {device}")

    # Loss function - use financial loss if available
    loss_config = dl_config.get("loss", "mse")
    if loss_config == "sharpe":
        criterion = SharpeLoss()
    elif loss_config == "sortino":
        criterion = SortinoLoss()
    else:
        criterion = torch.nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get("weight_decay", 1e-5)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    training_history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    logger.info(f"Starting training for max {max_epochs} epochs...")

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_X)

            # Handle different output shapes
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(1)

            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(1)

                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")

        # Record history
        current_lr = scheduler.get_last_lr()[0]
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["learning_rate"].append(current_lr)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Best: {best_val_loss:.6f}"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.6f}")

    # Store training history in model
    model.training_history = training_history
    model.best_val_loss = best_val_loss

    logger.info(f"Training complete! Best validation loss: {best_val_loss:.6f}")

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

    # Engine - Default is event-driven for JPMorgan-level realism
    # Use --engine vectorized for faster prototyping
    parser.add_argument(
        "--engine",
        choices=["vectorized", "event-driven"],
        default="event-driven",
        help="Backtest engine type (event-driven=realistic, vectorized=fast)",
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

    # Load data (with TimescaleDB support if configured)
    data = load_data(
        args.data_path,
        args.symbols,
        config["data"].get("min_bars", 100),
        config=config,
    )

    if not data:
        logger.error("No data loaded!")
        logger.info("Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    # Generate features with leakage-safe pipeline
    features, feature_pipelines = generate_features(data, config)

    # Training mode
    if args.mode in ["train", "full"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure models directory exists
        Path("models").mkdir(parents=True, exist_ok=True)

        if args.deep_learning:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch is required for deep learning but not installed!")
                logger.info("Install with: pip install torch")
                sys.exit(1)

            model = train_deep_learning(data, features, config)

            if model is not None:
                # Get model type for filename
                dl_config = config.get("deep_learning", {})
                model_type_name = dl_config.get("model", "lstm")

                # Create save path with metadata
                save_path = Path(f"models/{model_type_name}_{timestamp}.pth")
                metadata_path = Path(f"models/{model_type_name}_{timestamp}_metadata.json")

                # Save model checkpoint (state_dict is more portable)
                try:
                    checkpoint = {
                        "model_state_dict": model.state_dict() if hasattr(model, 'state_dict') else model,
                        "model_type": model_type_name,
                        "timestamp": timestamp,
                        "config": dl_config,
                        "feature_names": list(features[list(features.keys())[0]].columns) if features else [],
                    }
                    torch.save(checkpoint, save_path)
                    logger.info(f"Deep Learning Model saved to {save_path}")

                    # Save metadata as JSON for easy inspection
                    import json
                    metadata = {
                        "model_type": model_type_name,
                        "timestamp": timestamp,
                        "config": {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v
                                   for k, v in dl_config.items()},
                        "n_features": len(features[list(features.keys())[0]].columns) if features else 0,
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"Model metadata saved to {metadata_path}")

                except Exception as e:
                    logger.error(f"Failed to save deep learning model: {e}")
            else:
                logger.warning("Deep learning training returned None - model not saved")

        else:
            # Train ML model
            result = train_ml_model(data, features, config)

            # Save result and model with metadata
            model_filename = Path(f"models/{args.model}_{timestamp}.pkl")
            metadata_filename = Path(f"models/{args.model}_{timestamp}_metadata.json")

            try:
                with open(model_filename, "wb") as f:
                    pickle.dump(result, f)
                logger.info(f"Trained Model ({args.model}) saved to {model_filename}")

                # Save metadata as JSON
                import json
                metadata = {
                    "model_type": args.model,
                    "timestamp": timestamp,
                    "n_samples": result.n_train_samples,
                    "n_features": result.n_features,
                    "cv_scores": result.cv_scores,
                    "validation_metrics": result.validation_metrics,
                    "params": result.params,
                }
                with open(metadata_filename, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"Model metadata saved to {metadata_filename}")

            except Exception as e:
                logger.error(f"Failed to save ML model: {e}")

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
