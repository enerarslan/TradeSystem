"""
Feature Optimization Pipeline
=============================

This script implements PRIORITY 4 tasks from AI_AGENT_INSTRUCTIONS.md:
- Task 8: Remove Redundant Features (correlation analysis, clustering)
- Task 9: Add Regime Awareness (VIX, trend, volatility regimes)

Reduces features from 200+ to 60-80 high-quality, low-correlation features
while adding critical regime awareness signals.

Usage:
    python scripts/optimize_features.py --analyze       # Analyze feature correlations
    python scripts/optimize_features.py --reduce        # Reduce to optimal feature set
    python scripts/optimize_features.py --add-regime    # Add regime features

Author: AlphaTrade System
Based on AFML (Advances in Financial Machine Learning) best practices
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
    cluster_threshold: float = 0.80      # Threshold for hierarchical clustering

    # Target feature count
    min_features: int = 60
    max_features: int = 80

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
                'original_count': report.original_feature_count,
                'final_count': report.final_feature_count,
                'redundancy_removed_pct': round(report.redundancy_removed_pct, 2)
            },
            'optimal_features': report.kept_features,
            'removed_features': report.removed_features,
            'clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'representative': c.representative,
                    'members': c.features,
                    'avg_correlation': round(c.avg_correlation, 4)
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

def main():
    parser = argparse.ArgumentParser(
        description="Feature Optimization for AlphaTrade System"
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
        '--symbol',
        type=str,
        default='AAPL',
        help='Symbol to analyze (default: AAPL)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config/optimal_features.yaml',
        help='Output path for feature list'
    )

    args = parser.parse_args()
    config = FeatureOptimizationConfig()

    if args.analyze:
        logger.info("=" * 60)
        logger.info("FEATURE CORRELATION ANALYSIS")
        logger.info("=" * 60)

        # This would normally use real feature data
        # For demo, create sample correlation analysis
        print("\nTo analyze features, you need generated feature data.")
        print("Run the training pipeline first to generate features,")
        print("then re-run this script.")
        print("\nExample workflow:")
        print("  1. python scripts/train_models.py --generate-features-only")
        print("  2. python scripts/optimize_features.py --analyze")

    elif args.reduce:
        logger.info("=" * 60)
        logger.info("FEATURE REDUCTION")
        logger.info("=" * 60)

        print("\nFeature reduction requires generated feature data.")
        print("This will:")
        print("  1. Compute correlation matrix")
        print("  2. Cluster correlated features")
        print("  3. Select 60-80 optimal features")
        print("  4. Export to config/optimal_features.yaml")

    elif args.add_regime:
        logger.info("=" * 60)
        logger.info("REGIME FEATURE GENERATION")
        logger.info("=" * 60)

        # Load sample data
        data_path = Path("data/raw") / f"{args.symbol}_15min.csv"
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return

        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

        # Generate regime features
        generator = RegimeFeatureGenerator()
        df_with_regime = generator.add_all_regime_features(df)

        # Show new features
        regime_cols = [c for c in df_with_regime.columns if c not in df.columns]

        print(f"\nAdded {len(regime_cols)} regime features to {args.symbol}:")
        print("-" * 60)

        for col in regime_cols:
            sample = df_with_regime[col].dropna()
            print(f"  {col}:")
            print(f"    Range: [{sample.min():.4f}, {sample.max():.4f}]")
            print(f"    Mean: {sample.mean():.4f}")
            if df_with_regime[col].dtype in ['int64', 'int32']:
                print(f"    Distribution: {df_with_regime[col].value_counts().to_dict()}")

        # Show regime distribution
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
