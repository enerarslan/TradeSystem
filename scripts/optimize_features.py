"""
Institutional Feature Optimization Pipeline
============================================

Based on AFML (Advances in Financial Machine Learning) by Marcos Lopez de Prado.

This script implements institutional-grade feature engineering:

1. FRACTIONAL DIFFERENTIATION
   - Finds optimal d that achieves stationarity while preserving memory
   - Tests d in [0.1, 0.9] range, selecting minimum d where ADF p-value < 0.05
   - Preserves correlation with original series (target: > 0.7)

2. MICROSTRUCTURE FEATURES
   - VPIN: Volume-Synchronized Probability of Informed Trading
   - Kyle's Lambda: Price impact coefficient
   - Amihud Illiquidity: |return| / dollar volume
   - Order Flow Imbalance: Net buying/selling pressure

3. REGIME DETECTION
   - HMM-based market regime (bull/bear/neutral)
   - Volatility regime (low/normal/high/extreme)
   - Trend regime (based on MA relationships)

4. FEATURE CLUSTERING
   - Hierarchical clustering of correlated features
   - Select representative from each cluster
   - Apply PCA to microstructure features

Usage:
    python scripts/optimize_features.py --analyze          # Analyze correlations
    python scripts/optimize_features.py --reduce           # Reduce feature set
    python scripts/optimize_features.py --add-regime       # Add regime features
    python scripts/optimize_features.py --institutional    # Full institutional pipeline
    python scripts/optimize_features.py --fracdiff-search  # Optimize FracDiff d

Author: AlphaTrade Institutional System
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
import yaml
import json
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FeatureOptimizationConfig:
    """Configuration for feature optimization"""
    # Correlation thresholds
    max_correlation: float = 0.95        # Features with higher correlation are redundant
    cluster_threshold: float = 0.92      # Threshold for hierarchical clustering (less aggressive)

    # Target feature count
    min_features: int = 40               # Adjusted to reflect realistic technical feature set
    max_features: int = 60

    # Feature importance
    min_importance_pct: float = 0.01     # Drop features with < 1% importance

    # Data paths
    processed_data_dir: str = "data/processed"
    output_dir: str = "config"


@dataclass
class FeatureCluster:
    """A cluster of correlated features"""
    cluster_id: int
    features: List[str]
    representative: str                   # Keep this one
    avg_correlation: float
    max_correlation: float
    avg_importance: float


@dataclass
class FeatureSelectionReport:
    """Report on feature selection results"""
    original_feature_count: int
    final_feature_count: int
    removed_features: List[str]
    kept_features: List[str]
    clusters: List[FeatureCluster]
    correlation_matrix_rank: int
    redundancy_removed_pct: float


# ============================================================================
# TECHNICAL FEATURE GENERATOR (Standalone)
# ============================================================================

def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical features from OHLCV data.
    Returns DataFrame with ~100 features for correlation analysis.
    """
    features = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Price-based features
    for period in [5, 10, 20, 50, 100, 200]:
        features[f'sma_{period}'] = close.rolling(period).mean()
        features[f'ema_{period}'] = close.ewm(span=period).mean()
        features[f'price_vs_sma_{period}'] = close / features[f'sma_{period}'] - 1

    # Returns at different horizons
    for period in [1, 5, 10, 20, 60]:
        features[f'return_{period}'] = close.pct_change(period)

    # Volatility
    for period in [5, 10, 20, 60]:
        features[f'volatility_{period}'] = close.pct_change().rolling(period).std()

    # RSI
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    for period in [10, 20]:
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        features[f'bb_upper_{period}'] = sma + 2 * std
        features[f'bb_lower_{period}'] = sma - 2 * std
        features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / sma
        features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-10)

    # ATR
    for period in [7, 14, 21]:
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        features[f'atr_{period}'] = tr.rolling(period).mean()
        features[f'atr_pct_{period}'] = features[f'atr_{period}'] / close

    # Volume features
    for period in [5, 10, 20]:
        features[f'volume_sma_{period}'] = volume.rolling(period).mean()
        features[f'volume_ratio_{period}'] = volume / features[f'volume_sma_{period}']

    # Price momentum
    for period in [5, 10, 20, 60]:
        features[f'momentum_{period}'] = close / close.shift(period) - 1

    # High-Low range
    for period in [5, 10, 20]:
        features[f'range_{period}'] = (high.rolling(period).max() - low.rolling(period).min()) / close

    # Stochastic
    for period in [14, 21]:
        low_min = low.rolling(period).min()
        high_max = high.rolling(period).max()
        features[f'stoch_k_{period}'] = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()

    # Williams %R
    for period in [14, 21]:
        high_max = high.rolling(period).max()
        low_min = low.rolling(period).min()
        features[f'williams_r_{period}'] = -100 * (high_max - close) / (high_max - low_min + 1e-10)

    # CCI
    for period in [14, 20]:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        features[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Drop NaN rows and return
    features = features.dropna()

    return features


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

class CorrelationAnalyzer:
    """
    Analyze feature correlations and identify redundant features.

    High correlation between features means:
    1. They carry similar information
    2. One can be removed without information loss
    3. Multicollinearity issues in linear models
    4. Increased training time without benefit
    """

    def __init__(self, config: FeatureOptimizationConfig):
        self.config = config

    def compute_correlation_matrix(
        self,
        features: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Compute correlation matrix between all features.

        Args:
            features: Feature DataFrame
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Correlation matrix as DataFrame
        """
        return features.corr(method=method)

    def find_high_correlation_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find all pairs of features with correlation above threshold.

        Returns list of (feature1, feature2, correlation) tuples.
        """
        threshold = threshold or self.config.max_correlation

        pairs = []
        n = len(corr_matrix)
        cols = corr_matrix.columns

        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(corr_matrix.iloc[i, j])
                if corr >= threshold:
                    pairs.append((cols[i], cols[j], corr))

        # Sort by correlation descending
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs

    def compute_feature_redundancy_score(
        self,
        corr_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Compute redundancy score for each feature.

        Score = average of absolute correlations with other features.
        Higher score = more redundant.
        """
        abs_corr = corr_matrix.abs()

        # Exclude self-correlation (diagonal)
        np.fill_diagonal(abs_corr.values, 0)

        redundancy = abs_corr.mean(axis=1)
        return redundancy.sort_values(ascending=False)

    def rank_matrix_analysis(
        self,
        corr_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze the rank of the correlation matrix.

        A matrix with rank < n indicates linear dependencies.
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Count significant eigenvalues (> 0.01)
        significant = (eigenvalues > 0.01).sum()

        # Compute condition number
        cond_number = eigenvalues[0] / max(eigenvalues[-1], 1e-10)

        return {
            'matrix_size': len(corr_matrix),
            'effective_rank': significant,
            'redundant_dimensions': len(corr_matrix) - significant,
            'condition_number': cond_number,
            'largest_eigenvalue': eigenvalues[0],
            'smallest_eigenvalue': eigenvalues[-1],
            'eigenvalue_ratio': eigenvalues[0] / max(eigenvalues[significant - 1], 1e-10) if significant > 0 else np.inf
        }


# ============================================================================
# FEATURE CLUSTERING
# ============================================================================

class FeatureClusterer:
    """
    Cluster correlated features using hierarchical clustering.

    Instead of removing one feature at a time, we:
    1. Group features into clusters based on correlation
    2. Keep one representative from each cluster
    3. Select representative based on importance or variance
    """

    def __init__(self, config: FeatureOptimizationConfig):
        self.config = config

    def cluster_features(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = None,
        method: str = 'average'
    ) -> Dict[int, List[str]]:
        """
        Cluster features using hierarchical clustering.

        Args:
            corr_matrix: Correlation matrix
            threshold: Distance threshold for cluster formation
            method: Linkage method ('average', 'complete', 'ward')

        Returns:
            Dict mapping cluster_id to list of feature names
        """
        threshold = threshold or (1 - self.config.cluster_threshold)

        # Convert correlation to distance (1 - |correlation|)
        distance_matrix = 1 - corr_matrix.abs()

        # Ensure it's symmetric and has zeros on diagonal
        np.fill_diagonal(distance_matrix.values, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        # Convert to condensed distance matrix for linkage
        condensed = squareform(distance_matrix.values, checks=False)

        # Handle any negative values from numerical issues
        condensed = np.clip(condensed, 0, 2)

        # Perform hierarchical clustering
        Z = linkage(condensed, method=method)

        # Form clusters
        cluster_labels = fcluster(Z, t=threshold, criterion='distance')

        # Group features by cluster
        clusters = defaultdict(list)
        for feature, cluster_id in zip(corr_matrix.columns, cluster_labels):
            clusters[cluster_id].append(feature)

        return dict(clusters)

    def select_cluster_representative(
        self,
        features: pd.DataFrame,
        cluster_features: List[str],
        importance: pd.Series = None
    ) -> str:
        """
        Select the best representative feature from a cluster.

        Selection criteria (in order):
        1. Highest importance (if provided)
        2. Highest variance
        3. First alphabetically (fallback)
        """
        if len(cluster_features) == 1:
            return cluster_features[0]

        # If importance scores provided, use highest importance
        if importance is not None:
            cluster_importance = importance.reindex(cluster_features).dropna()
            if len(cluster_importance) > 0:
                return cluster_importance.idxmax()

        # Otherwise use variance
        available = [f for f in cluster_features if f in features.columns]
        if available:
            variances = features[available].var()
            return variances.idxmax()

        # Fallback
        return sorted(cluster_features)[0]

    def reduce_features_by_clustering(
        self,
        features: pd.DataFrame,
        importance: pd.Series = None,
        corr_matrix: pd.DataFrame = None
    ) -> Tuple[List[str], List[FeatureCluster]]:
        """
        Reduce features to cluster representatives.

        Returns:
            Tuple of (selected_features, cluster_info)
        """
        if corr_matrix is None:
            corr_matrix = features.corr()

        # Cluster features
        clusters = self.cluster_features(corr_matrix)

        selected_features = []
        cluster_info = []

        for cluster_id, cluster_features in clusters.items():
            # Select representative
            representative = self.select_cluster_representative(
                features, cluster_features, importance
            )
            selected_features.append(representative)

            # Calculate cluster statistics
            if len(cluster_features) > 1:
                cluster_corr = corr_matrix.loc[cluster_features, cluster_features]
                upper_tri = cluster_corr.where(
                    np.triu(np.ones(cluster_corr.shape), k=1).astype(bool)
                )
                avg_corr = upper_tri.abs().stack().mean()
                max_corr = upper_tri.abs().stack().max()
            else:
                avg_corr = 1.0
                max_corr = 1.0

            avg_imp = 0.0
            if importance is not None:
                cluster_imp = importance.reindex(cluster_features).dropna()
                avg_imp = cluster_imp.mean() if len(cluster_imp) > 0 else 0.0

            cluster_info.append(FeatureCluster(
                cluster_id=cluster_id,
                features=cluster_features,
                representative=representative,
                avg_correlation=avg_corr,
                max_correlation=max_corr,
                avg_importance=avg_imp
            ))

        return selected_features, cluster_info


# ============================================================================
# REGIME FEATURES
# ============================================================================

class RegimeFeatureGenerator:
    """
    Generate regime awareness features.

    Regime features are CRITICAL because:
    - Same technical signal means different things in different regimes
    - Bull market dips are buying opportunities, bear market rallies are traps
    - High VIX requires wider stops and smaller positions
    - Model should adapt behavior to regime

    Features must be NOT forward-looking.
    """

    def __init__(self):
        pass

    def add_vix_regime_features(
        self,
        df: pd.DataFrame,
        vix_data: pd.Series = None
    ) -> pd.DataFrame:
        """
        Add VIX-based regime features.

        VIX Regimes:
        - Low (<15): Complacency, potential for spike
        - Normal (15-25): Typical market conditions
        - High (25-35): Elevated fear, wider ranges
        - Extreme (>35): Crisis mode, different dynamics

        If VIX data not provided, estimate from price volatility.
        """
        df = df.copy()

        if vix_data is not None:
            # Use actual VIX data
            vix = vix_data.reindex(df.index).ffill()
        else:
            # Estimate VIX from price volatility
            # VIX ≈ annualized vol * 100
            returns = df['close'].pct_change()
            rolling_vol = returns.rolling(20).std()
            # Annualize (assuming 15-min bars, ~26 per day, ~252 days/year)
            annual_factor = np.sqrt(26 * 252)
            vix = rolling_vol * annual_factor * 100

        # VIX level
        df['vix_level'] = vix

        # VIX regime (categorical)
        df['vix_regime'] = 0  # Normal
        df.loc[vix < 15, 'vix_regime'] = -1  # Low
        df.loc[vix >= 25, 'vix_regime'] = 1  # High
        df.loc[vix >= 35, 'vix_regime'] = 2  # Extreme

        # VIX change
        df['vix_change_1d'] = vix.diff()
        df['vix_change_5d'] = vix.diff(5)

        # VIX percentile (rolling 60-day)
        df['vix_percentile'] = vix.rolling(60 * 4).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )

        # VIX term structure proxy (using vol of vol)
        df['vix_of_vix'] = vix.rolling(20).std()

        # VIX spike indicator
        vix_ma = vix.rolling(10).mean()
        df['vix_spike'] = (vix > vix_ma * 1.3).astype(int)

        return df

    def add_trend_regime_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add trend regime features.

        Trend Regimes:
        - Strong Bull: Price > SMA50 > SMA200
        - Bull: Price > SMA50, SMA50 > SMA200
        - Sideways: No clear trend
        - Bear: Price < SMA50, SMA50 < SMA200
        - Strong Bear: Price < SMA50 < SMA200
        """
        df = df.copy()
        close = df['close']

        # Moving averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        # Trend regime
        df['trend_regime'] = 0  # Sideways

        # Bull conditions
        bull_mask = (close > sma_50) & (sma_50 > sma_200)
        strong_bull_mask = bull_mask & (close > close.shift(20))

        # Bear conditions
        bear_mask = (close < sma_50) & (sma_50 < sma_200)
        strong_bear_mask = bear_mask & (close < close.shift(20))

        df.loc[bull_mask, 'trend_regime'] = 1
        df.loc[strong_bull_mask, 'trend_regime'] = 2
        df.loc[bear_mask, 'trend_regime'] = -1
        df.loc[strong_bear_mask, 'trend_regime'] = -2

        # Price position relative to MAs
        df['price_vs_sma20'] = (close / sma_20 - 1) * 100
        df['price_vs_sma50'] = (close / sma_50 - 1) * 100
        df['price_vs_sma200'] = (close / sma_200 - 1) * 100

        # MA slope (trend strength)
        df['sma50_slope'] = (sma_50 / sma_50.shift(20) - 1) * 100
        df['sma200_slope'] = (sma_200 / sma_200.shift(20) - 1) * 100

        # Days since regime change
        regime_changed = df['trend_regime'].diff() != 0
        df['days_since_regime_change'] = regime_changed.groupby(regime_changed.cumsum()).cumcount()

        # Trend alignment
        df['trend_alignment'] = (
            np.sign(df['price_vs_sma20']) +
            np.sign(df['price_vs_sma50']) +
            np.sign(df['price_vs_sma200'])
        ) / 3

        return df

    def add_volatility_regime_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add volatility regime features.

        Volatility Regimes:
        - Low: Vol < 20th percentile of rolling history
        - Normal: 20th-80th percentile
        - High: > 80th percentile
        - Extreme: > 95th percentile
        """
        df = df.copy()

        # Calculate realized volatility
        returns = df['close'].pct_change()

        # Different volatility measures
        df['vol_20'] = returns.rolling(20).std()
        df['vol_60'] = returns.rolling(60).std()

        # Volatility percentile
        vol = df['vol_20']
        df['vol_percentile'] = vol.rolling(200).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )

        # Volatility regime
        df['vol_regime'] = 1  # Normal
        df.loc[df['vol_percentile'] < 0.2, 'vol_regime'] = 0  # Low
        df.loc[df['vol_percentile'] > 0.8, 'vol_regime'] = 2  # High
        df.loc[df['vol_percentile'] > 0.95, 'vol_regime'] = 3  # Extreme

        # Volatility ratio (short-term vs long-term)
        df['vol_ratio'] = df['vol_20'] / df['vol_60']

        # Volatility trend
        df['vol_trend'] = df['vol_20'] / df['vol_20'].shift(20) - 1

        # Range expansion/contraction
        df['range_pct'] = (df['high'] - df['low']) / df['close']
        df['range_vs_avg'] = df['range_pct'] / df['range_pct'].rolling(20).mean()

        return df

    def add_all_regime_features(
        self,
        df: pd.DataFrame,
        vix_data: pd.Series = None
    ) -> pd.DataFrame:
        """
        Add all regime features to DataFrame.
        """
        df = self.add_vix_regime_features(df, vix_data)
        df = self.add_trend_regime_features(df)
        df = self.add_volatility_regime_features(df)

        # Combined regime indicator
        # Higher = more favorable conditions, Lower = more risky
        df['combined_regime'] = (
            (df['vix_regime'] == -1).astype(int) * 1 +      # Low VIX good
            (df['vix_regime'] >= 1).astype(int) * -1 +      # High VIX bad
            (df['trend_regime'] >= 1).astype(int) * 1 +     # Bull good
            (df['trend_regime'] <= -1).astype(int) * -1 +   # Bear bad
            (df['vol_regime'] <= 1).astype(int) * 1 +       # Low/normal vol good
            (df['vol_regime'] >= 2).astype(int) * -1        # High vol bad
        )

        return df


# ============================================================================
# FEATURE SELECTOR
# ============================================================================

class OptimalFeatureSelector:
    """
    Select optimal feature set combining correlation reduction and importance.
    """

    def __init__(self, config: FeatureOptimizationConfig):
        self.config = config
        self.correlation_analyzer = CorrelationAnalyzer(config)
        self.clusterer = FeatureClusterer(config)

    def select_optimal_features(
        self,
        features: pd.DataFrame,
        importance: pd.Series = None,
        target_count: int = None
    ) -> FeatureSelectionReport:
        """
        Select optimal feature subset.

        Steps:
        1. Compute correlation matrix
        2. Cluster correlated features
        3. Select representatives based on importance
        4. Ensure target feature count achieved
        """
        target_count = target_count or self.config.max_features
        original_count = len(features.columns)

        logger.info(f"Selecting optimal features from {original_count} candidates...")

        # Step 1: Compute correlations
        corr_matrix = self.correlation_analyzer.compute_correlation_matrix(features)
        rank_analysis = self.correlation_analyzer.rank_matrix_analysis(corr_matrix)

        # Step 2: Cluster features
        selected, cluster_info = self.clusterer.reduce_features_by_clustering(
            features, importance, corr_matrix
        )

        logger.info(f"Clustering reduced to {len(selected)} features ({len(cluster_info)} clusters)")

        # Step 3: Further reduce if needed
        if len(selected) > target_count and importance is not None:
            # Keep top features by importance
            selected_importance = importance.reindex(selected).dropna()
            selected_importance = selected_importance.sort_values(ascending=False)
            selected = list(selected_importance.head(target_count).index)

            logger.info(f"Importance filtering reduced to {len(selected)} features")

        # Step 4: Ensure minimum features
        if len(selected) < self.config.min_features:
            logger.warning(
                f"Only {len(selected)} features selected, below minimum {self.config.min_features}"
            )

        # Generate report
        removed = [f for f in features.columns if f not in selected]

        report = FeatureSelectionReport(
            original_feature_count=original_count,
            final_feature_count=len(selected),
            removed_features=removed,
            kept_features=selected,
            clusters=cluster_info,
            correlation_matrix_rank=rank_analysis['effective_rank'],
            redundancy_removed_pct=(original_count - len(selected)) / original_count * 100
        )

        return report

    def export_feature_list(
        self,
        report: FeatureSelectionReport,
        output_path: str = "config/optimal_features.yaml"
    ):
        """
        Export optimal feature list to YAML.
        """
        config = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'selection_summary': {
                'original_count': int(report.original_feature_count),
                'final_count': int(report.final_feature_count),
                'redundancy_removed_pct': round(float(report.redundancy_removed_pct), 2)
            },
            'optimal_features': list(report.kept_features),
            'removed_features': list(report.removed_features),
            'clusters': [
                {
                    'cluster_id': int(c.cluster_id),
                    'representative': str(c.representative),
                    'members': list(c.features),
                    'avg_correlation': round(float(c.avg_correlation), 4)
                }
                for c in report.clusters
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported feature list to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_institutional_pipeline(df: pd.DataFrame, symbol: str, output_path: str):
    """
    Run the full institutional feature engineering pipeline.

    This implements the AFML best practices:
    1. Optimal FracDiff d search
    2. Microstructure features (VPIN, Kyle's Lambda, etc.)
    3. HMM regime detection
    4. Feature clustering and selection
    """
    from src.features.institutional import (
        InstitutionalFeatureEngineer, InstitutionalFeatureConfig,
        ClusteredFeatureSelector, MicrostructurePCA
    )

    print("\n" + "=" * 70)
    print("INSTITUTIONAL FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    config = InstitutionalFeatureConfig()
    engineer = InstitutionalFeatureEngineer(config)

    # 1. Generate all institutional features
    print("\n[1/4] Generating institutional features...")
    features = engineer.build_features(df)
    print(f"      Generated {len(features.columns)} features")

    # Show optimal d values
    optimal_d = engineer.get_optimal_d()
    if optimal_d:
        print("\n      Optimal FracDiff d values:")
        for col, d in optimal_d.items():
            print(f"        {col}: d = {d:.2f}")

    # 2. Feature statistics
    print("\n[2/4] Computing feature statistics...")
    stats = engineer.get_feature_statistics(features)

    # Show top features by variance
    if not stats.empty:
        print("\n      Top features by variance:")
        top_var = stats.nlargest(10, 'std')
        for feat in top_var.index[:5]:
            print(f"        {feat}: std={stats.loc[feat, 'std']:.4f}")

    # 3. Apply PCA to microstructure features
    print("\n[3/4] Applying PCA to microstructure features...")
    pca = MicrostructurePCA(config)
    pca_features = pca.fit_transform(features)

    if not pca_features.empty:
        print(f"      Extracted {len(pca_features.columns)} PCA components")
        features = pd.concat([features, pca_features], axis=1)

    # 4. Cluster and select features
    print("\n[4/4] Clustering and selecting features...")

    # Remove non-numeric columns
    numeric_features = features.select_dtypes(include=[np.number])
    numeric_features = numeric_features.dropna(axis=1, how='all')

    selector = ClusteredFeatureSelector(config)
    selected = selector.fit_select(numeric_features)

    clusters = selector.get_clusters()
    print(f"      Created {len(clusters)} clusters")
    print(f"      Selected {len(selected)} features")

    # Export results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    # Feature categories
    fracdiff_features = [f for f in selected if 'ffd' in f]
    micro_features = [f for f in selected if 'micro_' in f]
    regime_features = [f for f in selected if 'regime_' in f]
    return_features = [f for f in selected if f.startswith('return_') or f.startswith('log_return_')]
    vol_features = [f for f in selected if f.startswith('vol_')]
    other_features = [f for f in selected if f not in
                     fracdiff_features + micro_features + regime_features + return_features + vol_features]

    print(f"\n  Feature Categories:")
    print(f"    Fractional Differentiation: {len(fracdiff_features)}")
    print(f"    Microstructure:             {len(micro_features)}")
    print(f"    Regime:                     {len(regime_features)}")
    print(f"    Returns:                    {len(return_features)}")
    print(f"    Volatility:                 {len(vol_features)}")
    print(f"    Other:                      {len(other_features)}")
    print(f"    --------------------------------")
    print(f"    TOTAL:                      {len(selected)}")

    # Export to YAML
    export_institutional_features(
        selected_features=selected,
        clusters=clusters,
        optimal_d=optimal_d,
        symbol=symbol,
        output_path=output_path
    )

    print(f"\n  Features exported to: {output_path}")

    return features, selected


def export_institutional_features(
    selected_features: List[str],
    clusters: Dict[int, List[str]],
    optimal_d: Dict[str, float],
    symbol: str,
    output_path: str
):
    """Export institutional features to YAML."""

    config = {
        'version': '2.0',
        'type': 'institutional',
        'created_at': datetime.now().isoformat(),
        'symbol': symbol,
        'selection_summary': {
            'total_features': len(selected_features),
            'n_clusters': len(clusters)
        },
        'optimal_fracdiff_d': {k: float(v) for k, v in optimal_d.items()},
        'feature_categories': {
            'fracdiff': [f for f in selected_features if 'ffd' in f],
            'microstructure': [f for f in selected_features if 'micro_' in f],
            'regime': [f for f in selected_features if 'regime_' in f],
            'returns': [f for f in selected_features if f.startswith('return_') or f.startswith('log_return_')],
            'volatility': [f for f in selected_features if f.startswith('vol_')],
        },
        'optimal_features': list(selected_features),
        'clusters': [
            {
                'cluster_id': int(cid),
                'features': list(feats)
            }
            for cid, feats in clusters.items()
        ]
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_fracdiff_search(df: pd.DataFrame, symbol: str):
    """
    Run fractional differentiation parameter search.

    This finds the optimal d value for each price series that:
    1. Achieves stationarity (ADF p-value < 0.05)
    2. Maximizes correlation with original series
    """
    from src.features.institutional import OptimalFracDiff, InstitutionalFeatureConfig

    print("\n" + "=" * 70)
    print("FRACTIONAL DIFFERENTIATION PARAMETER SEARCH")
    print("=" * 70)
    print(f"\nAnalyzing {symbol}...")

    config = InstitutionalFeatureConfig()
    fracdiff = OptimalFracDiff(config)

    close = df['close']
    volume = df['volume']

    print("\n" + "-" * 70)
    print("Close Price Analysis")
    print("-" * 70)

    optimal_d, diagnostics = fracdiff.find_optimal_d(close, 'close')

    print(f"\n  Optimal d: {optimal_d:.2f}")
    print(f"\n  Parameter Search Results:")
    print(f"  {'d':>6} {'ADF Stat':>12} {'p-value':>12} {'Stationary':>12} {'Correlation':>12}")
    print("  " + "-" * 60)

    for r in diagnostics['all_results']:
        stat = '*' if r['is_stationary'] else ''
        print(f"  {r['d']:>6.2f} {r['adf_stat']:>12.4f} {r['p_value']:>12.4f} {stat:>12} {r['correlation']:>12.4f}")

    print("\n  * = Stationary (ADF p-value < 0.05)")

    # Volume analysis
    print("\n" + "-" * 70)
    print("Log Volume Analysis")
    print("-" * 70)

    log_volume = np.log1p(volume)
    optimal_d_vol, diagnostics_vol = fracdiff.find_optimal_d(log_volume, 'log_volume')

    print(f"\n  Optimal d: {optimal_d_vol:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Series              Optimal d    Correlation    Stationary")
    print("  " + "-" * 55)
    print(f"  Close               {optimal_d:>8.2f}    {diagnostics['selected_row']['correlation']:>11.4f}    {'Yes' if diagnostics['selected_row']['is_stationary'] else 'No'}")
    print(f"  Log Volume          {optimal_d_vol:>8.2f}    {diagnostics_vol['selected_row']['correlation']:>11.4f}    {'Yes' if diagnostics_vol['selected_row']['is_stationary'] else 'No'}")

    print("\n  Interpretation:")
    print(f"    - d = 0 means no differencing (non-stationary, full memory)")
    print(f"    - d = 1 means full differencing (stationary, no memory)")
    print(f"    - d = {optimal_d:.2f} preserves {diagnostics['selected_row']['correlation']:.0%} correlation while achieving stationarity")


def main():
    parser = argparse.ArgumentParser(
        description="Institutional Feature Optimization for AlphaTrade System"
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze feature correlations'
    )
    parser.add_argument(
        '--reduce',
        action='store_true',
        help='Reduce to optimal feature set'
    )
    parser.add_argument(
        '--add-regime',
        action='store_true',
        help='Add regime awareness features'
    )
    parser.add_argument(
        '--institutional',
        action='store_true',
        help='Run full institutional feature pipeline (FracDiff + Microstructure + Regime)'
    )
    parser.add_argument(
        '--fracdiff-search',
        action='store_true',
        help='Search for optimal fractional differentiation d parameter'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='ALL',
        help='Symbol to analyze (default: ALL for all symbols, or specific like AAPL)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config/optimal_features.yaml',
        help='Output path for feature list'
    )

    args = parser.parse_args()
    config = FeatureOptimizationConfig()

    # New institutional pipeline options
    if args.institutional or args.fracdiff_search:
        analysis_symbol = args.symbol if args.symbol != 'ALL' else 'AAPL'
        data_path = Path("data/raw") / f"{analysis_symbol}_15min.csv"

        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return

        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        df.columns = df.columns.str.lower()

        if args.fracdiff_search:
            run_fracdiff_search(df, analysis_symbol)
        elif args.institutional:
            run_institutional_pipeline(df, analysis_symbol, args.output)

        return

    if args.analyze or args.reduce:
        logger.info("=" * 60)
        logger.info("FEATURE CORRELATION ANALYSIS" if args.analyze else "FEATURE REDUCTION")
        logger.info("=" * 60)

        # For correlation analysis, use AAPL as representative sample (or specified symbol)
        analysis_symbol = args.symbol if args.symbol != 'ALL' else 'AAPL'
        print(f"\nGenerating technical features for {analysis_symbol}...")

        data_path = Path("data/raw") / f"{analysis_symbol}_15min.csv"
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return

        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

        # Generate basic technical features
        features = generate_technical_features(df)
        print(f"Generated {len(features.columns)} features")

        # Analyze correlations
        analyzer = CorrelationAnalyzer(config)
        corr_matrix = analyzer.compute_correlation_matrix(features)

        # Find high correlation pairs
        high_corr_pairs = analyzer.find_high_correlation_pairs(corr_matrix, threshold=0.95)
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (>0.95):")
        for f1, f2, corr in high_corr_pairs[:10]:
            print(f"  {f1} <-> {f2}: {corr:.3f}")

        # Rank analysis
        rank_info = analyzer.rank_matrix_analysis(corr_matrix)
        print(f"\nMatrix Analysis:")
        print(f"  Features: {rank_info['matrix_size']}")
        print(f"  Effective rank: {rank_info['effective_rank']}")
        print(f"  Redundant dimensions: {rank_info['redundant_dimensions']}")

        if args.reduce:
            # Perform clustering and reduction
            selector = OptimalFeatureSelector(config)
            report = selector.select_optimal_features(features, target_count=config.max_features)

            print(f"\nFeature Reduction Results:")
            print(f"  Original: {report.original_feature_count}")
            print(f"  Final: {report.final_feature_count}")
            print(f"  Removed: {report.redundancy_removed_pct:.1f}%")

            # Export to YAML
            selector.export_feature_list(report, args.output)
            print(f"\nOptimal features exported to: {args.output}")

    elif args.add_regime:
        logger.info("=" * 60)
        logger.info("REGIME FEATURE GENERATION")
        logger.info("=" * 60)

        # Get all symbols from config or data/raw directory
        symbols_to_process = []

        if args.symbol and args.symbol != 'ALL':
            # Single symbol mode
            symbols_to_process = [args.symbol]
        else:
            # ALL symbols mode - get from data/raw directory
            raw_dir = Path("data/raw")
            if raw_dir.exists():
                symbols_to_process = sorted([
                    f.stem.replace('_15min', '')
                    for f in raw_dir.glob("*_15min.csv")
                ])

            if not symbols_to_process:
                print("No data files found in data/raw/")
                return

            print(f"\nProcessing {len(symbols_to_process)} symbols...")

        generator = RegimeFeatureGenerator()
        all_regime_stats = {}

        for symbol in symbols_to_process:
            data_path = Path("data/raw") / f"{symbol}_15min.csv"
            if not data_path.exists():
                print(f"  {symbol}: Data file not found, skipping")
                continue

            df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

            # Generate regime features
            df_with_regime = generator.add_all_regime_features(df)

            # Get regime columns
            regime_cols = [c for c in df_with_regime.columns if c not in df.columns]

            # Collect stats
            if 'vix_regime' in df_with_regime.columns:
                vix_dist = df_with_regime['vix_regime'].value_counts()
                trend_dist = df_with_regime['trend_regime'].value_counts()
                vol_dist = df_with_regime['vol_regime'].value_counts()

                all_regime_stats[symbol] = {
                    'vix_high_pct': (vix_dist.get(1, 0) + vix_dist.get(2, 0)) / len(df_with_regime) * 100,
                    'trend_bull_pct': (trend_dist.get(1, 0) + trend_dist.get(2, 0)) / len(df_with_regime) * 100,
                    'vol_high_pct': (vol_dist.get(2, 0) + vol_dist.get(3, 0)) / len(df_with_regime) * 100,
                    'bars': len(df_with_regime)
                }

            if len(symbols_to_process) == 1:
                # Single symbol - show detailed output
                print(f"\nAdded {len(regime_cols)} regime features to {symbol}:")
                print("-" * 60)

                for col in regime_cols:
                    sample = df_with_regime[col].dropna()
                    print(f"  {col}:")
                    print(f"    Range: [{sample.min():.4f}, {sample.max():.4f}]")
                    print(f"    Mean: {sample.mean():.4f}")
                    if df_with_regime[col].dtype in ['int64', 'int32']:
                        print(f"    Distribution: {df_with_regime[col].value_counts().to_dict()}")
            else:
                # Multi symbol - show progress
                print(f"  {symbol}: {len(regime_cols)} regime features added")

        # Show summary for multi-symbol mode
        if len(symbols_to_process) > 1 and all_regime_stats:
            print("\n" + "=" * 60)
            print("REGIME SUMMARY (ALL SYMBOLS)")
            print("=" * 60)

            print(f"\n{'Symbol':<8} {'Bars':>10} {'VIX High%':>10} {'Bull%':>10} {'VolHigh%':>10}")
            print("-" * 50)

            total_bars = 0
            total_vix_high = 0
            total_bull = 0
            total_vol_high = 0

            for symbol, stats in sorted(all_regime_stats.items()):
                print(f"{symbol:<8} {stats['bars']:>10,} {stats['vix_high_pct']:>10.1f} {stats['trend_bull_pct']:>10.1f} {stats['vol_high_pct']:>10.1f}")
                total_bars += stats['bars']
                total_vix_high += stats['vix_high_pct'] * stats['bars']
                total_bull += stats['trend_bull_pct'] * stats['bars']
                total_vol_high += stats['vol_high_pct'] * stats['bars']

            print("-" * 50)
            print(f"{'TOTAL':<8} {total_bars:>10,} {total_vix_high/total_bars:>10.1f} {total_bull/total_bars:>10.1f} {total_vol_high/total_bars:>10.1f}")

            print(f"\nTotal regime features per symbol: 24")
            print(f"Symbols processed: {len(all_regime_stats)}")

        # Show regime distribution for single symbol
        if len(symbols_to_process) == 1:
            print("\n" + "=" * 60)
            print("REGIME DISTRIBUTION")
            print("=" * 60)

            if 'vix_regime' in df_with_regime.columns:
                vix_dist = df_with_regime['vix_regime'].value_counts().sort_index()
                print("\nVIX Regime:")
                print(f"  Low (-1):     {vix_dist.get(-1, 0):,} bars")
                print(f"  Normal (0):   {vix_dist.get(0, 0):,} bars")
                print(f"  High (1):     {vix_dist.get(1, 0):,} bars")
                print(f"  Extreme (2):  {vix_dist.get(2, 0):,} bars")

            if 'trend_regime' in df_with_regime.columns:
                trend_dist = df_with_regime['trend_regime'].value_counts().sort_index()
                print("\nTrend Regime:")
                print(f"  Strong Bear (-2): {trend_dist.get(-2, 0):,} bars")
                print(f"  Bear (-1):        {trend_dist.get(-1, 0):,} bars")
                print(f"  Sideways (0):     {trend_dist.get(0, 0):,} bars")
                print(f"  Bull (1):         {trend_dist.get(1, 0):,} bars")
                print(f"  Strong Bull (2):  {trend_dist.get(2, 0):,} bars")

            if 'vol_regime' in df_with_regime.columns:
                vol_dist = df_with_regime['vol_regime'].value_counts().sort_index()
                print("\nVolatility Regime:")
                print(f"  Low (0):      {vol_dist.get(0, 0):,} bars")
                print(f"  Normal (1):   {vol_dist.get(1, 0):,} bars")
                print(f"  High (2):     {vol_dist.get(2, 0):,} bars")
                print(f"  Extreme (3):  {vol_dist.get(3, 0):,} bars")

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("FEATURE OPTIMIZATION OVERVIEW")
        print("=" * 60)
        print("\nThis script helps reduce features from 200+ to 60-80 by:")
        print("  1. Identifying highly correlated features (>0.95)")
        print("  2. Clustering similar features")
        print("  3. Keeping representative from each cluster")
        print("  4. Adding regime awareness features")
        print("\nRun with --add-regime to see regime feature generation.")


if __name__ == "__main__":
    main()
