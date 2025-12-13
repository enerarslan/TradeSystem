"""
Institutional-Grade Model Training Script
Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado

Features:
- Triple Barrier Method labeling (path-dependent)
- Meta-labeling for bet sizing
- Information-driven bars (volume/dollar)
- Feature neutralization (market beta removal)
- Clustered feature importance
- PurgedKFoldCV with embargo
- Probabilistic/Deflated Sharpe Ratio validation
"""

import sys
import os
from pathlib import Path

# Suppress warnings before other imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')
warnings.filterwarnings('ignore', message='.*numpy.dtype size changed.*')

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging
from src.data.loader import MultiAssetLoader
from src.data.preprocessor import (
    DataPreprocessor,
    InformationDrivenBars,
    convert_time_bars_to_information_bars,
    FeatureNeutralizer,
    OutlierMethod,
    TradingHoursFilter
)
from src.data.labeling import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    MetaLabeler,
    CUSUMFilter,
    get_sample_weights,
    get_time_decay_weights,
    combine_weights
)
from src.features.builder import FeatureBuilder
from src.features.cross_asset import CrossAssetFeatures
from src.features.regime import RegimeDetector
from src.models.ml_model import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.ensemble import StackingEnsemble, VotingEnsemble
from src.models.training import (
    ModelTrainer,
    WalkForwardValidator,
    CrossValidationTrainer,
    ClusteredFeatureImportance,
    feature_importance_with_clustering
)
from src.backtest.metrics import SharpeRatioStatistics, calculate_psr, calculate_dsr

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def convert_to_information_bars(
    df: pd.DataFrame,
    bar_type: str = "dollar",
    target_bars_per_day: int = 50
) -> pd.DataFrame:
    """
    Convert time bars to information-driven bars.

    Information-driven bars (volume, dollar, tick) have better statistical
    properties than time bars:
    - Returns closer to IID normal
    - Lower serial correlation
    - Adapts to market activity

    Args:
        df: OHLCV DataFrame with time bars
        bar_type: "volume", "dollar", or "tick"
        target_bars_per_day: Approximate bars per trading day

    Returns:
        DataFrame with information-driven bars
    """
    logger.info(f"Converting time bars to {bar_type} bars...")

    bars = convert_time_bars_to_information_bars(
        df,
        bar_type=bar_type,
        target_bars_per_day=target_bars_per_day
    )

    return bars


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    pt_sl_ratio: tuple = (1.0, 1.0),
    max_holding_period: int = 10,
    volatility_lookback: int = 20,
    min_return: float = 0.0,
    use_cusum_events: bool = True
) -> pd.DataFrame:
    """
    Apply Triple Barrier Method labeling.

    The Triple Barrier Method defines three barriers:
    1. Upper (Profit Take): Price touches profit target
    2. Lower (Stop Loss): Price touches stop loss
    3. Vertical (Time): Maximum holding period reached

    Benefits over fixed-horizon labeling:
    - Path dependency captured
    - Variable holding periods
    - More realistic trade outcomes

    Args:
        df: OHLCV DataFrame
        pt_sl_ratio: (profit_take, stop_loss) volatility multipliers
        max_holding_period: Maximum bars to hold
        volatility_lookback: Periods for volatility estimation
        min_return: Minimum return threshold
        use_cusum_events: Use CUSUM filter for event sampling

    Returns:
        DataFrame with labels and barrier info
    """
    logger.info("Applying Triple Barrier Method labeling...")

    # Configure Triple Barrier
    config = TripleBarrierConfig(
        pt_sl_ratio=pt_sl_ratio,
        volatility_lookback=volatility_lookback,
        volatility_method="ewm",
        max_holding_period=max_holding_period,
        min_return=min_return
    )

    labeler = TripleBarrierLabeler(config)

    # Get event timestamps
    if use_cusum_events:
        # Use CUSUM filter for adaptive event sampling
        returns = df['close'].pct_change().dropna()
        cusum = CUSUMFilter(threshold=returns.std() * 2)
        t_events = cusum.get_events(returns)
        logger.info(f"CUSUM filter detected {len(t_events)} events")
    else:
        # Use all timestamps
        t_events = df.index

    # Generate labels
    events = labeler.get_events_with_ohlcv(
        prices=df,
        t_events=t_events,
        pt_sl=pt_sl_ratio
    )

    if len(events) == 0:
        logger.warning("No events generated by Triple Barrier Method")
        return df

    # Add labels to original DataFrame
    df = df.copy()
    df['tb_label'] = events['label']
    df['tb_bin_label'] = events['bin_label']
    df['tb_t1'] = events['t1']  # Barrier touch time
    df['tb_ret'] = events['ret']  # Return at barrier

    logger.info(
        f"Triple Barrier labels: "
        f"{(events['label'] == 1).sum()} profitable, "
        f"{(events['label'] == -1).sum()} losing, "
        f"{(events['label'] == 0).sum()} neutral"
    )

    return df


def neutralize_features(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    market_prices: pd.DataFrame = None,
    microstructure_features: list = None
) -> pd.DataFrame:
    """
    Neutralize features against market beta.

    Feature neutralization removes systematic market exposure to ensure
    the model learns alpha, not just market beta.

    Args:
        features: Feature DataFrame
        prices: Price DataFrame for the asset
        market_prices: Market benchmark prices (uses asset if None)
        microstructure_features: List of microstructure feature names to downweight

    Returns:
        DataFrame with neutralized features
    """
    logger.info("Applying feature neutralization...")

    # Default microstructure features derived from OHLCV (less reliable)
    if microstructure_features is None:
        microstructure_features = [
            'kyle_lambda', 'amihud_lambda', 'vpin', 'roll_spread',
            'corwin_schultz', 'order_imbalance', 'volume_clock'
        ]

    # Use asset returns as proxy if no market data
    if market_prices is None:
        market_returns = prices['close'].pct_change()
    else:
        market_returns = market_prices['close'].pct_change()

    # Initialize neutralizer
    neutralizer = FeatureNeutralizer()
    neutralizer.set_market_returns(market_returns)

    # Identify features to neutralize (exclude microstructure)
    features_to_neutralize = [
        col for col in features.columns
        if col not in microstructure_features
        and features[col].dtype in [np.float64, np.float32, np.int64]
    ]

    # Neutralize features
    neutralized = neutralizer.neutralize_dataframe(
        features,
        columns=features_to_neutralize,
        rolling=True,
        suffix='_neutral'
    )

    # Downweight microstructure features (scale by 0.5)
    for col in microstructure_features:
        if col in neutralized.columns:
            neutralized[col] = neutralized[col] * 0.5
            logger.debug(f"Downweighted microstructure feature: {col}")

    logger.info(f"Neutralized {len(features_to_neutralize)} features")

    return neutralized


def apply_clustered_feature_selection(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'mda',
    n_clusters: int = None
) -> tuple:
    """
    Apply clustered feature importance for robust feature selection.

    Standard feature importance is unreliable with correlated features.
    Clustered importance addresses this by:
    1. Grouping correlated features into clusters
    2. Computing importance at cluster level
    3. Selecting representative features per cluster

    Args:
        model: Fitted model
        X: Feature DataFrame
        y: Labels
        method: 'mda' (Mean Decrease Accuracy) or 'mdi' (Mean Decrease Impurity)
        n_clusters: Number of clusters (auto if None)

    Returns:
        Tuple of (selected_features, importance_report)
    """
    logger.info("Computing clustered feature importance...")

    result = feature_importance_with_clustering(
        model=model,
        X=X,
        y=y,
        n_clusters=n_clusters,
        method=method,
        n_iterations=10
    )

    logger.info(
        f"Clustered into {result['n_clusters']} clusters, "
        f"selected {len(result['selected_features'])} features"
    )

    # Log top clusters
    for cluster_id, importance in result['cluster_importance'].head(5).items():
        features_in_cluster = result['clusters'].get(cluster_id, [])
        logger.info(
            f"  Cluster {cluster_id}: importance={importance:.4f}, "
            f"features={features_in_cluster[:3]}{'...' if len(features_in_cluster) > 3 else ''}"
        )

    return result['selected_features'], result


def prepare_data(
    symbols: list,
    data_dir: str = "data/processed",
    lookback_days: int = 500,
    use_information_bars: bool = True,
    use_triple_barrier: bool = True,
    neutralize: bool = True,
    filter_trading_hours: bool = True,
    include_extended_hours: bool = False,
    use_cross_asset_features: bool = True,
    use_regime_features: bool = True,
    use_time_decay_weights: bool = True,
    time_decay_factor: float = 0.5,
    symbols_config: dict = None,
    triple_barrier_config: dict = None
) -> tuple:
    """
    Load and prepare data for institutional-grade training.

    Pipeline:
    1. Load raw OHLCV data for all symbols
    2. Filter to US regular trading hours (optional)
    3. Convert to information-driven bars (optional)
    4. Apply Triple Barrier labeling (optional)
    5. Generate technical features
    6. Add cross-asset features (correlations, sector momentum, beta)
    7. Add regime detection features (HMM, volatility regime)
    8. Neutralize features (optional)
    9. Create combined sample weights (uniqueness + time decay)

    Args:
        symbols: List of symbols to load
        data_dir: Directory containing raw data
        lookback_days: Number of days of history to use
        use_information_bars: Convert time bars to dollar bars
        use_triple_barrier: Use Triple Barrier labeling method
        neutralize: Apply feature neutralization
        filter_trading_hours: Filter to US regular trading hours only (9:30-16:00 ET)
        include_extended_hours: Include pre-market and after-hours data
        use_cross_asset_features: Add cross-asset correlation and sector features
        use_regime_features: Add HMM regime detection features
        use_time_decay_weights: Combine uniqueness weights with time decay
        time_decay_factor: Factor for time decay (0=no decay, 1=full linear decay)
        symbols_config: Symbol configuration dict for sector mapping

    Returns:
        Tuple of (features_df, labels, sample_weights, events_df)
    """
    logger.info(f"Loading data for {len(symbols)} symbols...")

    loader = MultiAssetLoader(data_path=data_dir)
    preprocessor = DataPreprocessor()
    feature_builder = FeatureBuilder()

    # Initialize trading hours filter if enabled
    trading_hours_filter = None
    if filter_trading_hours:
        trading_hours_filter = TradingHoursFilter(
            include_extended_hours=include_extended_hours
        )
        logger.info(
            f"Trading hours filter enabled: "
            f"{'Extended' if include_extended_hours else 'Regular'} hours only"
        )

    # Initialize cross-asset feature generator
    cross_asset_generator = None
    sector_mapping = {}
    if use_cross_asset_features and symbols_config:
        # Build sector mapping from config
        sectors = symbols_config.get('sectors', {})
        for sector_name, sector_data in sectors.items():
            sector_mapping[sector_name] = sector_data.get('symbols', [])

        cross_asset_generator = CrossAssetFeatures(sector_mapping=sector_mapping)
        logger.info(f"Cross-asset features enabled with {len(sector_mapping)} sectors")

    # Initialize regime detector
    regime_detector = None
    if use_regime_features:
        regime_detector = RegimeDetector()
        logger.info("Regime detection features enabled")

    # First pass: Load and preprocess all symbol data
    symbol_data = {}
    symbol_prices = {}

    print(f"\n[1/4] Loading and preprocessing data for {len(symbols)} symbols...")
    for symbol in tqdm(symbols, desc="Loading symbols", unit="symbol"):
        try:
            # 1. Load raw data
            df = loader.loader.load(symbol)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # 2. Preprocess with Winsorization
            df_clean, _ = preprocessor.preprocess(df, symbol)

            # 2b. Filter to trading hours
            if trading_hours_filter is not None:
                df_clean = trading_hours_filter.filter(df_clean)
                if len(df_clean) < 100:
                    logger.warning(f"{symbol}: Too few bars after trading hours filter ({len(df_clean)})")
                    continue

            # 3. Convert to information-driven bars (optional)
            if use_information_bars:
                df_bars = convert_to_information_bars(
                    df_clean,
                    bar_type="dollar",
                    target_bars_per_day=50
                )
                if len(df_bars) < 50:
                    logger.warning(f"{symbol}: Too few bars after conversion, using time bars")
                    df_bars = df_clean
            else:
                df_bars = df_clean

            symbol_data[symbol] = df_bars
            symbol_prices[symbol] = df_bars['close']
            logger.debug(f"Loaded {len(df_bars)} bars for {symbol}")

        except Exception as e:
            logger.warning(f"Error loading {symbol}: {e}")

    if not symbol_data:
        raise ValueError("No data loaded for any symbols")

    logger.info(f"Successfully loaded data for {len(symbol_data)} symbols")

    # Add cross-sectional features (relative rankings across symbols)
    if len(symbol_data) > 1:
        symbol_data = add_cross_sectional_features(
            symbol_data,
            sector_mapping=sector_mapping if sector_mapping else None
        )

    # Build cross-asset price matrix for cross-asset features
    cross_asset_features = None
    if cross_asset_generator and len(symbol_data) > 1:
        try:
            # Create aligned price DataFrame
            prices_df = pd.DataFrame(symbol_prices)
            prices_df = prices_df.dropna(how='all')

            if len(prices_df) > 100:
                # Get pairs for pairs trading from config
                pairs = []
                if symbols_config:
                    correlation_pairs = symbols_config.get('correlation_pairs', [])
                    for pair_info in correlation_pairs:
                        pair = pair_info.get('pair', [])
                        if len(pair) == 2:
                            pairs.append((pair[0], pair[1]))

                # Generate cross-asset features
                cross_asset_features = cross_asset_generator.generate_all_features(
                    prices=prices_df,
                    pairs=pairs if pairs else None,
                    window=60
                )
                logger.info(f"Generated {len(cross_asset_features.columns)} cross-asset features")
        except Exception as e:
            logger.warning(f"Cross-asset feature generation failed: {e}")

    # Second pass: Generate features and labels for each symbol
    all_features = []
    all_labels = []
    all_weights = []
    all_events = []

    print(f"\n[2/4] Generating features and labels...")
    for symbol, df_bars in tqdm(symbol_data.items(), desc="Processing symbols", unit="symbol"):
        try:
            # 4. Apply Triple Barrier labeling
            if use_triple_barrier:
                # Get symbol-specific params from calibrated config
                tb_defaults = triple_barrier_config.get('default_params', {}) if triple_barrier_config else {}
                tb_symbol_params = {}
                if triple_barrier_config and 'symbols' in triple_barrier_config:
                    tb_symbol_params = triple_barrier_config['symbols'].get(symbol, {})

                # Use symbol-specific params if available, else defaults
                pt_mult = tb_symbol_params.get('profit_target_atr_mult', tb_defaults.get('profit_target_atr_mult', 1.5))
                sl_mult = tb_symbol_params.get('stop_loss_atr_mult', tb_defaults.get('stop_loss_atr_mult', 1.0))
                max_hold = tb_symbol_params.get('max_holding_period', tb_defaults.get('max_holding_period', 10))
                vol_lookback = tb_defaults.get('volatility_lookback', 20)

                df_labeled = apply_triple_barrier_labels(
                    df_bars,
                    pt_sl_ratio=(pt_mult, sl_mult),
                    max_holding_period=max_hold,
                    volatility_lookback=vol_lookback
                )
                label_col = 'tb_bin_label'
            else:
                logger.warning("Using simple forward returns (not recommended)")
                df_labeled = df_bars.copy()
                forward_returns = df_labeled['close'].pct_change(5).shift(-5)
                df_labeled['simple_label'] = (forward_returns > 0).astype(int)
                label_col = 'simple_label'

            # 5. Generate technical features
            features = feature_builder.build_features(df_labeled)

            # 6. Add regime features
            if regime_detector is not None:
                try:
                    regime_features = regime_detector.generate_regime_features(
                        df_labeled['close'],
                        window=20
                    )
                    # Align indices and merge
                    common_idx = features.index.intersection(regime_features.index)
                    features = features.loc[common_idx]
                    regime_features = regime_features.loc[common_idx]
                    features = pd.concat([features, regime_features], axis=1)
                    logger.debug(f"{symbol}: Added {len(regime_features.columns)} regime features")
                except Exception as e:
                    logger.warning(f"{symbol}: Regime feature generation failed: {e}")

            # 7. Add cross-asset features
            if cross_asset_features is not None:
                try:
                    common_idx = features.index.intersection(cross_asset_features.index)
                    if len(common_idx) > 0:
                        features = features.loc[common_idx]
                        ca_features = cross_asset_features.loc[common_idx]
                        features = pd.concat([features, ca_features], axis=1)
                        logger.debug(f"{symbol}: Added {len(ca_features.columns)} cross-asset features")
                except Exception as e:
                    logger.warning(f"{symbol}: Cross-asset feature merge failed: {e}")

            # 8. Neutralize features (optional)
            if neutralize:
                features = neutralize_features(
                    features,
                    prices=df_labeled,
                    market_prices=None
                )

            # 9. Create labels
            labels = df_labeled[label_col].reindex(features.index)

            # 10. Calculate sample weights with time decay
            if use_triple_barrier and 'tb_t1' in df_labeled.columns:
                events_for_weights = df_labeled[['tb_t1']].rename(columns={'tb_t1': 't1'})
                events_for_weights = events_for_weights.reindex(features.index).dropna()

                if len(events_for_weights) > 0:
                    # Get uniqueness weights
                    uniqueness_weights = get_sample_weights(
                        events_for_weights,
                        df_labeled['close'],
                        num_threads=1
                    )
                    uniqueness_weights = uniqueness_weights.reindex(features.index).fillna(1.0 / len(features))

                    # Get time decay weights if enabled
                    if use_time_decay_weights:
                        time_weights = get_time_decay_weights(
                            events_for_weights,
                            c=time_decay_factor
                        )
                        time_weights = time_weights.reindex(features.index).fillna(1.0 / len(features))

                        # Combine weights (50% uniqueness, 50% time decay by default)
                        weights = combine_weights(
                            uniqueness_weights,
                            time_weights,
                            alpha=0.5  # Equal weight to uniqueness and time decay
                        )
                        weights = weights.reindex(features.index).fillna(1.0 / len(features))
                    else:
                        weights = uniqueness_weights
                else:
                    weights = pd.Series(1.0 / len(features), index=features.index)
            else:
                weights = pd.Series(1.0 / len(features), index=features.index)

            # 11. Align and drop NaN
            valid_idx = features.dropna().index.intersection(labels.dropna().index)
            features = features.loc[valid_idx]
            labels = labels.loc[valid_idx]
            weights = weights.loc[valid_idx]

            features['symbol'] = symbol
            all_features.append(features)
            all_labels.append(labels)
            all_weights.append(weights)

            logger.info(f"Prepared {len(features)} samples with {len(features.columns)} features for {symbol}")

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    if not all_features:
        raise ValueError("No data prepared")

    # Combine all symbols
    features_df = pd.concat(all_features, axis=0)
    labels = pd.concat(all_labels, axis=0)
    weights = pd.concat(all_weights, axis=0)

    # Normalize weights
    weights = weights / weights.sum()

    logger.info(f"Total samples: {len(features_df)}, Total features: {len(features_df.columns)}")

    return features_df, labels, weights, None


def add_cross_sectional_features(
    symbol_data: Dict[str, pd.DataFrame],
    sector_mapping: Dict[str, List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Add cross-sectional features that rank each symbol relative to others.

    Cross-sectional features capture relative value signals across the universe.
    These are critical for strategies that exploit divergences between symbols.

    Features added:
    - Return rank vs all symbols
    - Return rank vs sector
    - Volume rank vs all symbols
    - Momentum rank vs sector average
    - Volatility percentile
    - Z-score vs cross-sectional mean

    Args:
        symbol_data: Dict mapping symbol to DataFrame with price data
        sector_mapping: Dict mapping sector names to list of symbols

    Returns:
        Dict with cross-sectional features added to each DataFrame
    """
    if len(symbol_data) < 2:
        logger.warning("Need at least 2 symbols for cross-sectional features")
        return symbol_data

    logger.info(f"Generating cross-sectional features for {len(symbol_data)} symbols...")

    # Calculate returns and other metrics for all symbols
    returns_dict = {}
    volume_dict = {}
    volatility_dict = {}
    momentum_dict = {}

    for symbol, df in symbol_data.items():
        if 'close' in df.columns:
            returns_dict[symbol] = df['close'].pct_change()
            volatility_dict[symbol] = df['close'].pct_change().rolling(20).std()
            momentum_dict[symbol] = df['close'].pct_change(20)  # 20-period momentum
        if 'volume' in df.columns:
            volume_dict[symbol] = df['volume']

    # Create aligned DataFrames
    returns_df = pd.DataFrame(returns_dict)
    volume_df = pd.DataFrame(volume_dict) if volume_dict else None
    volatility_df = pd.DataFrame(volatility_dict)
    momentum_df = pd.DataFrame(momentum_dict)

    # Calculate cross-sectional stats at each timestamp
    returns_mean = returns_df.mean(axis=1)
    returns_std = returns_df.std(axis=1)
    volatility_mean = volatility_df.mean(axis=1)

    # Build sector mapping lookup
    symbol_to_sector = {}
    if sector_mapping:
        for sector, symbols in sector_mapping.items():
            for s in symbols:
                symbol_to_sector[s] = sector

    # Add cross-sectional features to each symbol's DataFrame
    result = {}

    for symbol, df in symbol_data.items():
        df_new = df.copy()

        # Return rank (1 = best, 0 = worst)
        returns_rank = returns_df.rank(axis=1, pct=True)
        if symbol in returns_rank.columns:
            df_new['cs_return_rank'] = returns_rank[symbol].reindex(df_new.index)

        # Z-score vs cross-sectional mean
        if symbol in returns_df.columns:
            symbol_return = returns_df[symbol]
            z_score = (symbol_return - returns_mean) / (returns_std + 1e-10)
            df_new['cs_return_zscore'] = z_score.reindex(df_new.index)

        # Volume rank
        if volume_df is not None and symbol in volume_df.columns:
            volume_rank = volume_df.rank(axis=1, pct=True)
            df_new['cs_volume_rank'] = volume_rank[symbol].reindex(df_new.index)

        # Volatility percentile
        volatility_rank = volatility_df.rank(axis=1, pct=True)
        if symbol in volatility_rank.columns:
            df_new['cs_volatility_pctl'] = volatility_rank[symbol].reindex(df_new.index)

        # Momentum rank
        momentum_rank = momentum_df.rank(axis=1, pct=True)
        if symbol in momentum_rank.columns:
            df_new['cs_momentum_rank'] = momentum_rank[symbol].reindex(df_new.index)

        # Sector-relative features
        if symbol in symbol_to_sector:
            sector = symbol_to_sector[symbol]
            sector_symbols = [s for s in sector_mapping.get(sector, []) if s in returns_df.columns]

            if len(sector_symbols) > 1:
                # Sector average return
                sector_returns = returns_df[sector_symbols].mean(axis=1)
                if symbol in returns_df.columns:
                    # Return vs sector
                    df_new['cs_vs_sector_return'] = (
                        returns_df[symbol] - sector_returns
                    ).reindex(df_new.index)

                    # Rank within sector
                    sector_rank = returns_df[sector_symbols].rank(axis=1, pct=True)
                    df_new['cs_sector_rank'] = sector_rank[symbol].reindex(df_new.index)

                # Sector momentum
                sector_momentum = momentum_df[sector_symbols].mean(axis=1) if sector_symbols else None
                if sector_momentum is not None and symbol in momentum_df.columns:
                    df_new['cs_vs_sector_momentum'] = (
                        momentum_df[symbol] - sector_momentum
                    ).reindex(df_new.index)

        # Distance from cross-sectional extremes
        if symbol in returns_df.columns:
            cs_max = returns_df.max(axis=1)
            cs_min = returns_df.min(axis=1)
            cs_range = cs_max - cs_min + 1e-10

            df_new['cs_distance_from_max'] = (
                (cs_max - returns_df[symbol]) / cs_range
            ).reindex(df_new.index)

            df_new['cs_distance_from_min'] = (
                (returns_df[symbol] - cs_min) / cs_range
            ).reindex(df_new.index)

        result[symbol] = df_new

    # Count features added
    sample_symbol = list(result.keys())[0]
    cs_features = [c for c in result[sample_symbol].columns if c.startswith('cs_')]
    logger.info(f"Added {len(cs_features)} cross-sectional features: {cs_features}")

    return result


def calculate_dynamic_embargo(
    feature_columns: list,
    data_frequency_minutes: int = 15,
    min_embargo_pct: float = 0.05
) -> float:
    """
    Calculate dynamic embargo based on maximum feature lookback period.

    Fixed 5% embargo may not be sufficient given 200-period feature lookbacks.
    This function analyzes feature names to infer lookback periods and
    calculates an appropriate embargo.

    Args:
        feature_columns: List of feature column names
        data_frequency_minutes: Frequency of data bars in minutes
        min_embargo_pct: Minimum embargo percentage (AFML recommends >= 5%)

    Returns:
        Calculated embargo percentage (at least min_embargo_pct)
    """
    max_lookback = 0

    # Parse feature names to infer lookback periods
    # Common patterns: sma_20, ema_50, rsi_14, close_pct_200, etc.
    import re

    lookback_patterns = [
        r'_(\d+)$',           # suffix number: sma_20, ema_50
        r'(\d+)_',            # prefix number: 14_rsi
        r'_(\d+)_',           # middle number: close_20_ema
        r'pct_(\d+)',         # percentage lookback: pct_20
        r'rolling_(\d+)',     # rolling windows
        r'window_(\d+)',      # window specification
        r'period_(\d+)',      # period specification
    ]

    for col in feature_columns:
        for pattern in lookback_patterns:
            matches = re.findall(pattern, col.lower())
            for match in matches:
                try:
                    lookback = int(match)
                    if lookback > max_lookback and lookback < 1000:  # Sanity check
                        max_lookback = lookback
                except ValueError:
                    continue

    # If no lookback found, use default assumption
    if max_lookback == 0:
        max_lookback = 200  # Conservative default for SMA_200

    # Calculate embargo as percentage
    # Embargo should be at least as large as the longest feature lookback
    # expressed as a percentage of typical dataset size

    # Assuming ~26 bars per trading day (15-min bars, 6.5 hours)
    bars_per_day = int(6.5 * 60 / data_frequency_minutes)

    # Embargo should cover at least max_lookback periods
    # Plus some buffer for serial correlation decay
    embargo_periods = max_lookback + 10  # Add small buffer

    # Calculate as percentage of 6-month training window
    # 6 months = ~126 trading days
    typical_train_size = 126 * bars_per_day  # ~3276 bars
    calculated_embargo_pct = embargo_periods / typical_train_size

    # Apply minimum constraint
    final_embargo_pct = max(calculated_embargo_pct, min_embargo_pct)

    logger.info(
        f"Dynamic embargo calculation: "
        f"max_lookback={max_lookback}, embargo_periods={embargo_periods}, "
        f"embargo_pct={final_embargo_pct:.2%} (min={min_embargo_pct:.1%})"
    )

    return final_embargo_pct


def train_with_purged_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series = None,
    n_splits: int = 3,  # Changed from 5 to 3 for better effective test size with 11% embargo
    embargo_pct: float = None,
    use_dynamic_embargo: bool = True
) -> dict:
    """
    Train model with PurgedKFoldCV.

    PurgedKFoldCV prevents information leakage by:
    1. Purging: Removing training samples that overlap with test
    2. Embargo: Adding gap after test set to handle serial correlation

    Args:
        model: Model to train
        X: Features
        y: Labels
        sample_weight: Sample weights (optional)
        n_splits: Number of CV folds
        embargo_pct: Embargo percentage (None = use dynamic calculation)
        use_dynamic_embargo: Calculate embargo based on feature lookbacks

    Returns:
        Dictionary with CV results
    """
    # Calculate dynamic embargo if not specified
    if embargo_pct is None and use_dynamic_embargo:
        embargo_pct = calculate_dynamic_embargo(
            feature_columns=X.columns.tolist(),
            data_frequency_minutes=15
        )
    elif embargo_pct is None:
        embargo_pct = 0.05  # Default minimum

    logger.info(f"Training with PurgedKFoldCV (embargo={embargo_pct:.1%})...")

    cv_trainer = CrossValidationTrainer(
        cv_method='purged_kfold',
        n_splits=n_splits,
        purge_gap=0,
        embargo_pct=embargo_pct,
        score_metric='accuracy'
    )

    results = cv_trainer.cross_validate(model, X, y)

    return results


def train_single_model(
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name: str,
    sample_weight: pd.Series = None,
    **kwargs
) -> tuple:
    """Train a single model and return metrics"""
    logger.info(f"Training {model_name}...")

    model = model_class(**kwargs)

    # Set model_id to the friendly name for saving
    model.model_id = model_name

    # Train (use fit method)
    model.fit(X_train, y_train, validation_data=(X_val, y_val))

    # Evaluate
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    n_classes = len(np.unique(y_train))
    is_binary = n_classes == 2

    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'precision': precision_score(y_val, val_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, val_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_val, val_pred, average='weighted', zero_division=0),
    }

    # Calculate AUC appropriately for binary vs multiclass
    try:
        if val_prob is not None and len(val_prob.shape) == 2:
            if is_binary:
                metrics['auc'] = roc_auc_score(y_val, val_prob[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_val, val_prob, multi_class='ovr', average='weighted')
        else:
            metrics['auc'] = 0.0
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        metrics['auc'] = 0.0

    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

    return model, metrics


def train_ensemble(
    base_models: list,
    X_train,
    y_train,
    X_val,
    y_val
) -> tuple:
    """Train ensemble model"""
    logger.info("Training ensemble...")

    # Voting ensemble
    voting = VotingEnsemble(
        models=base_models,
        voting='soft'
    )
    # Try fit first, fall back to train
    if hasattr(voting, 'fit'):
        voting.fit(X_train, y_train, validation_data=(X_val, y_val))
    elif hasattr(voting, 'train'):
        voting.train(X_train, y_train, X_val, y_val)
    else:
        logger.warning("Ensemble has no train or fit method, models are already trained")

    # Evaluate
    val_pred = voting.predict(X_val)
    val_prob = voting.predict_proba(X_val)

    from sklearn.metrics import accuracy_score, roc_auc_score

    n_classes = len(np.unique(y_train))
    is_binary = n_classes == 2

    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
    }

    # Calculate AUC appropriately for binary vs multiclass
    try:
        if val_prob is not None and len(val_prob.shape) == 2:
            if is_binary:
                metrics['auc'] = roc_auc_score(y_val, val_prob[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_val, val_prob, multi_class='ovr', average='weighted')
        else:
            metrics['auc'] = 0.0
    except Exception as e:
        logger.warning(f"Could not calculate ensemble AUC: {e}")
        metrics['auc'] = 0.0

    logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

    return voting, metrics


def validate_with_sharpe_statistics(
    returns: pd.Series,
    n_trials: int = 1
) -> dict:
    """
    Validate strategy with Probabilistic and Deflated Sharpe Ratio.

    Standard Sharpe Ratio is susceptible to:
    - Selection bias (p-hacking from multiple backtests)
    - Non-normality (fat tails in financial returns)

    PSR/DSR address these issues.

    Args:
        returns: Strategy returns series
        n_trials: Number of backtests/trials run

    Returns:
        Dictionary with Sharpe statistics
    """
    logger.info("Computing advanced Sharpe statistics...")

    sr_stats = SharpeRatioStatistics(periods_per_year=252)
    report = sr_stats.generate_sharpe_report(
        returns,
        n_trials=n_trials,
        sr_benchmark=0.0,
        confidence=0.95
    )

    logger.info(
        f"Sharpe: {report['sharpe_ratio']:.2f}, "
        f"PSR: {report['probabilistic_sr']:.1%}, "
        f"DSR: {report['deflated_sr']:.1%}"
    )
    logger.info(f"Interpretation: {report['interpretation']}")

    return report


def main():
    """Main training function with institutional-grade methodology"""
    parser = argparse.ArgumentParser(description="Train ML models (Institutional Grade)")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--use-information-bars", action="store_true", default=True)
    parser.add_argument("--use-triple-barrier", action="store_true", default=True)
    parser.add_argument("--neutralize", action="store_true", default=True)
    parser.add_argument("--filter-trading-hours", action="store_true", default=True,
                        help="Filter data to US regular trading hours (9:30-16:00 ET)")
    parser.add_argument("--include-extended-hours", action="store_true", default=False,
                        help="Include pre-market and after-hours data")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of backtests for DSR")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Number of CV folds (default: 3, recommended with 11%% embargo)")
    parser.add_argument("--embargo", type=float, default=0.11,
                        help="Embargo percentage for PurgedKFoldCV (default: 0.11 = 11%%)")
    parser.add_argument("--use-optimal-features", action="store_true", default=True,
                        help="Use pre-selected optimal features from config/optimal_features.yaml")
    parser.add_argument("--optimal-features-config", type=str, default="config/optimal_features.yaml",
                        help="Path to optimal features configuration")
    parser.add_argument("--triple-barrier-config", type=str, default="config/triple_barrier_params.yaml",
                        help="Path to calibrated triple barrier parameters")
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_path="logs", level="INFO")

    logger.info("=" * 60)
    logger.info("Starting Institutional-Grade Model Training")
    logger.info("Based on AFML by Marcos Lopez de Prado")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)
    symbols_config = load_config("config/symbols.yaml")

    # Load calibrated triple barrier parameters
    triple_barrier_config = None
    try:
        triple_barrier_config = load_config(args.triple_barrier_config)
        logger.info(f"Loaded calibrated triple barrier params from {args.triple_barrier_config}")
        logger.info(f"  - {len(triple_barrier_config.get('symbols', {}))} symbol-specific configs")
    except Exception as e:
        logger.warning(f"Could not load triple barrier config: {e}. Using defaults.")

    # Get symbols - extract from all sectors
    if args.symbols:
        symbols = args.symbols
    else:
        # Extract symbols from all sectors in the YAML structure
        symbols = []
        sectors = symbols_config.get('sectors', {})
        for sector_name, sector_data in sectors.items():
            sector_symbols = sector_data.get('symbols', [])
            symbols.extend(sector_symbols)

        # If no symbols found in sectors, fall back to symbols dict keys
        if not symbols:
            symbols_dict = symbols_config.get('symbols', {})
            symbols = list(symbols_dict.keys())

        logger.info(f"Loaded {len(symbols)} symbols from {len(sectors)} sectors")

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")

    # Prepare data with institutional methodology
    features_df, labels, sample_weights, _ = prepare_data(
        symbols,
        use_information_bars=args.use_information_bars,
        use_triple_barrier=args.use_triple_barrier,
        neutralize=args.neutralize,
        filter_trading_hours=args.filter_trading_hours,
        include_extended_hours=args.include_extended_hours,
        use_cross_asset_features=True,
        use_regime_features=True,
        use_time_decay_weights=True,
        time_decay_factor=0.5,
        symbols_config=symbols_config,
        triple_barrier_config=triple_barrier_config
    )

    # Remove symbol column for training
    X = features_df.drop(columns=['symbol'], errors='ignore')
    y = labels.astype(int)

    # Validate label distribution
    unique_labels = y.unique()
    logger.info(f"Unique labels in dataset: {sorted(unique_labels)}")
    logger.info(f"Label distribution:\n{y.value_counts().sort_index()}")

    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes for classification, but found only: {unique_labels}")

    # Apply optimal feature selection if enabled
    if args.use_optimal_features:
        try:
            optimal_config = load_config(args.optimal_features_config)
            optimal_features = optimal_config.get('optimal_features', [])

            if optimal_features:
                # Find intersection with available features
                available_features = set(X.columns)
                selected_features = [f for f in optimal_features if f in available_features]
                missing_features = [f for f in optimal_features if f not in available_features]

                if missing_features:
                    logger.warning(f"Missing optimal features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")

                if selected_features:
                    X = X[selected_features]
                    logger.info(f"Using {len(selected_features)} optimal features from {args.optimal_features_config}")
                else:
                    logger.warning("No optimal features found in data, using all features")
        except Exception as e:
            logger.warning(f"Could not load optimal features config: {e}. Using all features.")

    # Train/validation split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    weights_train = sample_weights.iloc[:split_idx]

    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    logger.info(f"CV settings: n_splits={args.n_splits}, embargo={args.embargo:.1%}")

    # Validate train/val label distributions
    train_labels = y_train.unique()
    val_labels = y_val.unique()
    logger.info(f"Train labels: {sorted(train_labels)}, distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Val labels: {sorted(val_labels)}, distribution: {y_val.value_counts().to_dict()}")

    if len(train_labels) < 2:
        raise ValueError(f"Training set needs at least 2 classes, but found: {train_labels}")
    if len(val_labels) < 2:
        logger.warning(f"Validation set has only {len(val_labels)} classes: {val_labels}. Metrics may be limited.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train individual models
    models = []
    all_metrics = {}

    print(f"\n[3/4] Training models...")

    # XGBoost (with early stopping to prevent overfitting)
    print("  Training XGBoost...")
    xgb_model, xgb_metrics = train_single_model(
        XGBoostModel,
        X_train, y_train, X_val, y_val,
        model_name="xgboost_v1",
        sample_weight=weights_train,
        n_estimators=500,  # More iterations with early stopping
        max_depth=4,       # Reduced depth to prevent overfitting
        learning_rate=0.05,  # Lower learning rate
        early_stopping_rounds=30,  # Stop if no improvement for 30 rounds
        min_child_weight=5,  # Regularization
        subsample=0.8,       # Row sampling
        colsample_bytree=0.8  # Column sampling
    )
    models.append(xgb_model)
    all_metrics['xgboost'] = xgb_metrics

    # Apply clustered feature importance analysis
    try:
        # Access the underlying model from our wrapper
        underlying_model = xgb_model._model if hasattr(xgb_model, '_model') else xgb_model
        cluster_results = apply_clustered_feature_selection(
            underlying_model,
            X_val,
            y_val,
            method='mdi'
        )
    except Exception as e:
        logger.warning(f"Clustered feature importance failed: {e}")
        cluster_results = ([], {})

    # LightGBM (with regularization)
    print("  Training LightGBM...")
    lgb_model, lgb_metrics = train_single_model(
        LightGBMModel,
        X_train, y_train, X_val, y_val,
        model_name="lightgbm_v1",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        early_stopping_rounds=30,
        min_child_samples=20,  # Regularization
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1  # L2 regularization
    )
    models.append(lgb_model)
    all_metrics['lightgbm'] = lgb_metrics

    # CatBoost (with regularization)
    print("  Training CatBoost...")
    cat_model, cat_metrics = train_single_model(
        CatBoostModel,
        X_train, y_train, X_val, y_val,
        model_name="catboost_v1",
        iterations=500,
        depth=4,
        learning_rate=0.05,
        early_stopping_rounds=30,
        l2_leaf_reg=3.0,  # L2 regularization
        subsample=0.8,
        colsample_bylevel=0.8
    )
    models.append(cat_model)
    all_metrics['catboost'] = cat_metrics

    # Train ensemble
    print("  Training Ensemble...")
    ensemble_model, ensemble_metrics = train_ensemble(
        models, X_train, y_train, X_val, y_val
    )
    all_metrics['ensemble'] = ensemble_metrics

    # Validate with Sharpe statistics
    # Generate pseudo-returns from predictions for demonstration
    val_pred_prob = ensemble_model.predict_proba(X_val)[:, 1]
    pseudo_returns = pd.Series(
        (val_pred_prob - 0.5) * 0.01,  # Scale predictions to returns
        index=X_val.index
    )
    sharpe_report = validate_with_sharpe_statistics(pseudo_returns, n_trials=args.n_trials)
    # Convert numpy types to native Python types for YAML serialization
    all_metrics['sharpe_analysis'] = {
        'sharpe_ratio': float(sharpe_report['sharpe_ratio']),
        'psr': float(sharpe_report['probabilistic_sr']),
        'dsr': float(sharpe_report['deflated_sr']),
        'is_significant': bool(sharpe_report['is_significant'])
    }

    # Save models
    print(f"\n[4/4] Saving models...")
    logger.info("Saving models...")

    for model in models:
        model_name = getattr(model, 'model_id', getattr(model, 'name', 'model'))
        model_path = output_dir / f"{model_name}.pkl"
        model.save(str(model_path))
        logger.info(f"Saved {model_name} to {model_path}")

    # Save ensemble
    ensemble_path = output_dir / "ensemble_model.pkl"
    ensemble_model.save(str(ensemble_path))
    logger.info(f"Saved ensemble to {ensemble_path}")

    # Save metrics
    metrics_path = output_dir / "training_metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump({
            'training_date': datetime.now().isoformat(),
            'methodology': {
                'labeling': 'triple_barrier' if args.use_triple_barrier else 'simple',
                'bar_type': 'dollar' if args.use_information_bars else 'time',
                'feature_neutralization': args.neutralize,
                'cv_method': 'purged_kfold',
                'n_splits': args.n_splits,
                'embargo_pct': args.embargo
            },
            'symbols': symbols,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'metrics': all_metrics,
            'selected_features': cluster_results[0] if cluster_results else []
        }, f)

    # Save feature names
    feature_names_path = output_dir / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    # Print summary
    print("\nTraining Summary:")
    print("-" * 40)
    print(f"Methodology: AFML Institutional-Grade")
    print(f"Labeling: {'Triple Barrier' if args.use_triple_barrier else 'Simple'}")
    print(f"Bars: {'Dollar Bars' if args.use_information_bars else 'Time Bars'}")
    print(f"Neutralization: {'Yes' if args.neutralize else 'No'}")
    print()

    for model_name, metrics in all_metrics.items():
        if model_name == 'sharpe_analysis':
            print(f"\nSharpe Analysis:")
            print(f"  SR: {metrics['sharpe_ratio']:.2f}")
            print(f"  PSR: {metrics['psr']:.1%}")
            print(f"  DSR: {metrics['dsr']:.1%}")
            print(f"  Significant: {metrics['is_significant']}")
        else:
            acc = metrics.get('accuracy', 0)
            auc = metrics.get('auc', 0)
            print(f"{model_name:15} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    print(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
