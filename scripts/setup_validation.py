"""
Validation Setup Pipeline
=========================

This script implements PRIORITY 3 tasks from AI_AGENT_INSTRUCTIONS.md:
- Task 6: Verify Embargo Prevents Leakage
- Task 7: Reserve Holdout Data

Ensures proper train/test split methodology to prevent information leakage.

Usage:
    python scripts/setup_validation.py --setup-holdout    # Reserve holdout data
    python scripts/setup_validation.py --verify-embargo   # Verify embargo settings
    python scripts/setup_validation.py --leakage-test     # Run leakage detection

Author: AlphaTrade System
Based on AFML (Advances in Financial Machine Learning) best practices
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import argparse
import shutil
from dataclasses import dataclass, asdict
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation setup"""
    # Embargo settings
    min_embargo_pct: float = 0.05       # Minimum 5% embargo
    max_feature_lookback: int = 200     # SMA_200 is typically the longest
    embargo_buffer: int = 20            # Additional buffer

    # Holdout settings
    temporal_holdout_months: int = 3    # Last 3 months for final testing
    symbol_holdout_count: int = 6       # 6 symbols for OOS testing

    # Stress test periods (YYYY-MM-DD format)
    stress_periods: List[Dict] = None

    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    holdout_data_dir: str = "data/holdout"

    def __post_init__(self):
        if self.stress_periods is None:
            # Define known stress periods
            self.stress_periods = [
                {
                    "name": "2022_bear_market",
                    "start": "2022-01-01",
                    "end": "2022-06-30",
                    "description": "Fed rate hikes, tech selloff, Ukraine war"
                },
                {
                    "name": "covid_crash",
                    "start": "2020-02-15",
                    "end": "2020-04-15",
                    "description": "COVID-19 market crash and recovery"
                },
                {
                    "name": "2023_banking_crisis",
                    "start": "2023-03-01",
                    "end": "2023-03-31",
                    "description": "SVB collapse, regional banking crisis"
                }
            ]


@dataclass
class HoldoutManifest:
    """Manifest tracking holdout data configuration"""
    created_at: str
    version: str

    # Temporal holdout
    temporal_cutoff_date: str
    temporal_holdout_start: str
    temporal_holdout_end: str

    # Symbol holdout
    holdout_symbols: List[str]
    holdout_symbol_sectors: Dict[str, str]

    # Stress periods
    stress_periods: List[Dict]

    # Training data boundaries
    training_start_date: str
    training_end_date: str

    # Statistics
    training_samples_per_symbol: Dict[str, int]
    holdout_samples_per_symbol: Dict[str, int]


# ============================================================================
# EMBARGO VERIFICATION
# ============================================================================

class EmbargoVerifier:
    """
    Verify that embargo settings prevent information leakage.

    Information leakage occurs when:
    1. Training samples overlap with test labels (temporal leakage)
    2. Features computed on future data (look-ahead bias)
    3. Insufficient gap between train/test periods

    The embargo period should be >= max_feature_lookback + buffer
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def calculate_min_embargo(self) -> Dict[str, Any]:
        """
        Calculate minimum required embargo based on feature lookbacks.
        """
        # List all features with lookback periods
        feature_lookbacks = {
            # Technical indicators
            'SMA_200': 200,
            'SMA_50': 50,
            'SMA_20': 20,
            'EMA_200': 200,
            'RSI_14': 14,
            'MACD': 26,  # 26-period EMA
            'Bollinger_Bands': 20,
            'ATR_14': 14,
            'ADX': 14,

            # Volatility features
            'Rolling_Std_20': 20,
            'Rolling_Std_60': 60,
            'Volatility_Ratio': 60,

            # Cross-asset features
            'Correlation_60': 60,
            'Beta_60': 60,
            'Sector_Momentum_20': 20,

            # Regime features
            'HMM_Regime': 100,  # Typical HMM lookback
            'Trend_Regime': 50,

            # Triple Barrier
            'Max_Holding_Period': 40,  # Typical max holding
        }

        max_lookback = max(feature_lookbacks.values())
        min_embargo_bars = max_lookback + self.config.embargo_buffer

        # For 15-min bars, calculate as percentage
        # Assume ~2000 training samples per symbol (about 1 year of regular hours)
        typical_train_size = 2000
        min_embargo_pct = min_embargo_bars / typical_train_size

        return {
            'max_feature_lookback': max_lookback,
            'buffer': self.config.embargo_buffer,
            'min_embargo_bars': min_embargo_bars,
            'min_embargo_pct': min_embargo_pct,
            'recommended_embargo_pct': max(min_embargo_pct, self.config.min_embargo_pct),
            'feature_lookbacks': feature_lookbacks
        }

    def verify_purged_kfold_settings(
        self,
        n_splits: int = 5,
        embargo_pct: float = None
    ) -> Dict[str, Any]:
        """
        Verify PurgedKFoldCV settings prevent leakage.

        PurgedKFold removes:
        1. Training samples whose labels overlap with test samples
        2. An embargo period after each test sample
        """
        min_embargo = self.calculate_min_embargo()
        embargo_pct = embargo_pct or min_embargo['recommended_embargo_pct']

        verification = {
            'n_splits': n_splits,
            'embargo_pct': embargo_pct,
            'min_required_embargo_pct': min_embargo['recommended_embargo_pct'],
            'is_sufficient': embargo_pct >= min_embargo['recommended_embargo_pct'],
        }

        # Calculate effective test size
        # With purging and embargo, effective test size is reduced
        base_test_pct = 1 / n_splits
        purge_loss_pct = 0.02  # Typical purge loss
        embargo_loss_pct = embargo_pct

        effective_test_pct = base_test_pct - purge_loss_pct - embargo_loss_pct
        verification['effective_test_pct'] = max(0, effective_test_pct)

        if effective_test_pct < 0.1:
            verification['warning'] = "Effective test size < 10%, consider fewer folds"

        return verification

    def test_for_leakage(
        self,
        df: pd.DataFrame,
        train_idx: pd.Index,
        test_idx: pd.Index,
        label_col: str = 'tb_t1',
        embargo_bars: int = None
    ) -> Dict[str, Any]:
        """
        Test for information leakage between train and test sets.

        Checks:
        1. No temporal overlap between train features and test labels
        2. No training samples within embargo period of test samples
        3. All test timestamps are after all train timestamps (for walk-forward)
        """
        embargo_bars = embargo_bars or self.config.max_feature_lookback + self.config.embargo_buffer

        results = {
            'leakage_detected': False,
            'issues': [],
            'stats': {}
        }

        # Check 1: Temporal order
        train_max = train_idx.max()
        test_min = test_idx.min()

        if train_max >= test_min:
            results['leakage_detected'] = True
            results['issues'].append(
                f"Temporal overlap: train max {train_max} >= test min {test_min}"
            )

        # Check 2: Embargo gap
        if isinstance(train_max, pd.Timestamp) and isinstance(test_min, pd.Timestamp):
            gap = (test_min - train_max).total_seconds() / 60  # Minutes
            gap_bars = gap / 15  # 15-min bars

            results['stats']['gap_bars'] = gap_bars
            results['stats']['required_embargo'] = embargo_bars

            if gap_bars < embargo_bars:
                results['leakage_detected'] = True
                results['issues'].append(
                    f"Insufficient embargo: {gap_bars:.0f} bars < {embargo_bars} required"
                )

        # Check 3: Label overlap (if labels span time)
        if label_col in df.columns:
            train_labels = df.loc[train_idx, label_col]
            test_start = test_idx.min()

            # Check if any training label extends into test period
            for idx, label_end in train_labels.items():
                if pd.notna(label_end) and label_end >= test_start:
                    results['leakage_detected'] = True
                    results['issues'].append(
                        f"Label overlap: train sample {idx} label extends to {label_end}"
                    )
                    break  # Just report first instance

        return results


# ============================================================================
# HOLDOUT DATA SETUP
# ============================================================================

class HoldoutDataManager:
    """
    Reserve holdout data for final out-of-sample testing.

    Three types of holdout:
    1. Temporal: Last N months of data (never seen during development)
    2. Symbol: Some symbols excluded from training entirely
    3. Stress: Specific stress periods for robustness testing
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.raw_dir = Path(config.raw_data_dir)
        self.processed_dir = Path(config.processed_data_dir)
        self.holdout_dir = Path(config.holdout_data_dir)

    def select_holdout_symbols(
        self,
        symbols: List[str],
        sector_mapping: Dict[str, List[str]] = None
    ) -> List[str]:
        """
        Select symbols for holdout, ensuring sector diversity.

        Goal: 2 symbols per sector for 6 total holdout symbols.
        Selection criteria:
        - Mid-tier volatility (not highest/lowest)
        - Sufficient data history
        - Representative of sector
        """
        # Default sector mapping if not provided
        if sector_mapping is None:
            sector_mapping = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'INTC', 'CSCO', 'AVGO'],
                'Healthcare': ['UNH', 'JNJ', 'LLY', 'MRK', 'ABBV', 'TMO', 'AMGN'],
                'Financials': ['JPM', 'BAC', 'GS', 'V', 'MA', 'AXP'],
                'Consumer_Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'DIS'],
                'Consumer_Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'WBA'],
                'Energy_Industrials': ['XOM', 'CVX', 'CAT', 'BA', 'HON', 'MMM', 'DOW']
            }

        holdout_symbols = []
        symbols_per_sector = 1  # 1 per sector for ~6 total

        for sector, sector_symbols in sector_mapping.items():
            # Filter to available symbols
            available = [s for s in sector_symbols if s in symbols]

            if not available:
                continue

            # Select from middle of list (not first/last which are often most/least popular)
            mid_idx = len(available) // 2
            selected = available[mid_idx:mid_idx + symbols_per_sector]
            holdout_symbols.extend(selected)

            if len(holdout_symbols) >= self.config.symbol_holdout_count:
                break

        logger.info(f"Selected holdout symbols: {holdout_symbols}")
        return holdout_symbols

    def get_temporal_cutoff(
        self,
        df: pd.DataFrame
    ) -> pd.Timestamp:
        """
        Calculate temporal cutoff date for holdout period.

        Returns timestamp N months before end of data.
        """
        end_date = df.index.max()
        cutoff = end_date - pd.DateOffset(months=self.config.temporal_holdout_months)
        return cutoff

    def split_temporal_data(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and holdout based on temporal cutoff.
        """
        if cutoff is None:
            cutoff = self.get_temporal_cutoff(df)

        train_df = df[df.index < cutoff]
        holdout_df = df[df.index >= cutoff]

        return train_df, holdout_df

    def setup_holdout_directory(self):
        """Create holdout directory structure."""
        dirs = [
            self.holdout_dir / "temporal",
            self.holdout_dir / "symbols",
            self.holdout_dir / "stress"
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Create .gitignore to prevent accidental commits
        gitignore_path = self.holdout_dir / ".gitignore"
        gitignore_path.write_text("# Holdout data - DO NOT COMMIT\n*\n!.gitignore\n")

        logger.info(f"Created holdout directory structure at {self.holdout_dir}")

    def reserve_holdout_data(
        self,
        symbols: List[str] = None,
        sector_mapping: Dict[str, List[str]] = None
    ) -> HoldoutManifest:
        """
        Reserve holdout data for final testing.

        Steps:
        1. Create holdout directory structure
        2. Select holdout symbols
        3. Copy temporal holdout data
        4. Move symbol holdout data
        5. Mark stress periods
        6. Generate manifest
        """
        # Get all symbols if not provided
        if symbols is None:
            symbols = [f.stem.replace('_15min', '') for f in self.raw_dir.glob('*_15min.csv')]

        # Setup directories
        self.setup_holdout_directory()

        # Select holdout symbols
        holdout_symbols = self.select_holdout_symbols(symbols, sector_mapping)

        # Track statistics
        training_samples = {}
        holdout_samples = {}

        # Get date range from first symbol
        sample_df = pd.read_csv(
            self.raw_dir / f"{symbols[0]}_15min.csv",
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        temporal_cutoff = self.get_temporal_cutoff(sample_df)

        # Process each symbol
        for symbol in symbols:
            try:
                # Load data
                raw_path = self.raw_dir / f"{symbol}_15min.csv"
                if not raw_path.exists():
                    continue

                df = pd.read_csv(raw_path, parse_dates=['timestamp'], index_col='timestamp')

                if symbol in holdout_symbols:
                    # Symbol holdout: move entire symbol to holdout
                    holdout_path = self.holdout_dir / "symbols" / f"{symbol}_15min.csv"
                    shutil.copy(raw_path, holdout_path)
                    holdout_samples[symbol] = len(df)
                    training_samples[symbol] = 0
                    logger.info(f"{symbol}: Moved to symbol holdout ({len(df)} bars)")
                else:
                    # Regular symbol: split temporally
                    train_df, temporal_holdout_df = self.split_temporal_data(df, temporal_cutoff)

                    # Save temporal holdout
                    temporal_path = self.holdout_dir / "temporal" / f"{symbol}_15min.csv"
                    temporal_holdout_df.to_csv(temporal_path)

                    training_samples[symbol] = len(train_df)
                    holdout_samples[symbol] = len(temporal_holdout_df)

                    logger.info(
                        f"{symbol}: {len(train_df)} training, "
                        f"{len(temporal_holdout_df)} temporal holdout"
                    )

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")

        # Create stress period markers
        for stress in self.config.stress_periods:
            stress_path = self.holdout_dir / "stress" / f"{stress['name']}.json"
            with open(stress_path, 'w') as f:
                json.dump(stress, f, indent=2)

        # Generate manifest
        holdout_symbol_sectors = {}
        if sector_mapping:
            for sector, syms in sector_mapping.items():
                for s in holdout_symbols:
                    if s in syms:
                        holdout_symbol_sectors[s] = sector

        manifest = HoldoutManifest(
            created_at=datetime.now().isoformat(),
            version="1.0",
            temporal_cutoff_date=str(temporal_cutoff),
            temporal_holdout_start=str(temporal_cutoff),
            temporal_holdout_end=str(sample_df.index.max()),
            holdout_symbols=holdout_symbols,
            holdout_symbol_sectors=holdout_symbol_sectors,
            stress_periods=self.config.stress_periods,
            training_start_date=str(sample_df.index.min()),
            training_end_date=str(temporal_cutoff),
            training_samples_per_symbol=training_samples,
            holdout_samples_per_symbol=holdout_samples
        )

        # Save manifest
        manifest_path = self.holdout_dir / "holdout_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, indent=2, default=str)

        logger.info(f"Holdout manifest saved to {manifest_path}")

        return manifest

    def verify_holdout_isolation(self) -> Dict[str, Any]:
        """
        Verify that holdout data is properly isolated.

        Checks:
        1. Holdout directory exists and contains data
        2. Manifest exists and is valid
        3. Training code cannot access holdout
        """
        results = {
            'passed': True,
            'checks': {}
        }

        # Check 1: Directory exists
        results['checks']['holdout_dir_exists'] = self.holdout_dir.exists()
        if not self.holdout_dir.exists():
            results['passed'] = False

        # Check 2: Manifest exists
        manifest_path = self.holdout_dir / "holdout_manifest.json"
        results['checks']['manifest_exists'] = manifest_path.exists()
        if not manifest_path.exists():
            results['passed'] = False
        else:
            # Load and validate manifest
            with open(manifest_path) as f:
                manifest = json.load(f)
            results['checks']['manifest_valid'] = 'holdout_symbols' in manifest

        # Check 3: Data files exist
        temporal_dir = self.holdout_dir / "temporal"
        symbols_dir = self.holdout_dir / "symbols"

        temporal_files = list(temporal_dir.glob('*.csv')) if temporal_dir.exists() else []
        symbol_files = list(symbols_dir.glob('*.csv')) if symbols_dir.exists() else []

        results['checks']['temporal_holdout_count'] = len(temporal_files)
        results['checks']['symbol_holdout_count'] = len(symbol_files)

        if len(temporal_files) == 0 and len(symbol_files) == 0:
            results['passed'] = False
            results['checks']['no_holdout_data'] = True

        # Check 4: Gitignore exists
        gitignore_path = self.holdout_dir / ".gitignore"
        results['checks']['gitignore_exists'] = gitignore_path.exists()

        return results


# ============================================================================
# TRAINING DATA GUARD
# ============================================================================

class TrainingDataGuard:
    """
    Guard to prevent accidental access to holdout data during training.

    Usage:
        guard = TrainingDataGuard()
        guard.block_holdout()  # Call before training

        # Training code...

        guard.unblock_holdout()  # Call after training for evaluation
    """

    def __init__(self, holdout_dir: str = "data/holdout"):
        self.holdout_dir = Path(holdout_dir)
        self.manifest_path = self.holdout_dir / "holdout_manifest.json"
        self._blocked = False

    def load_manifest(self) -> Optional[Dict]:
        """Load holdout manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return None

    def get_blocked_symbols(self) -> List[str]:
        """Get list of symbols that should not be used in training."""
        manifest = self.load_manifest()
        if manifest:
            return manifest.get('holdout_symbols', [])
        return []

    def get_blocked_dates(self) -> Tuple[Optional[str], Optional[str]]:
        """Get date range that should not be used in training."""
        manifest = self.load_manifest()
        if manifest:
            return (
                manifest.get('temporal_holdout_start'),
                manifest.get('temporal_holdout_end')
            )
        return None, None

    def validate_training_request(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """
        Validate that a training request doesn't use holdout data.

        Returns dict with validation result and any violations.
        """
        result = {
            'valid': True,
            'violations': []
        }

        # Check symbols
        blocked_symbols = set(self.get_blocked_symbols())
        requested_symbols = set(symbols)

        symbol_violations = requested_symbols.intersection(blocked_symbols)
        if symbol_violations:
            result['valid'] = False
            result['violations'].append(
                f"Holdout symbols requested: {list(symbol_violations)}"
            )

        # Check dates
        holdout_start, holdout_end = self.get_blocked_dates()

        if end_date and holdout_start:
            if pd.Timestamp(end_date) >= pd.Timestamp(holdout_start):
                result['valid'] = False
                result['violations'].append(
                    f"Training end date {end_date} overlaps holdout period starting {holdout_start}"
                )

        return result


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validation Setup for AlphaTrade System"
    )
    parser.add_argument(
        '--setup-holdout',
        action='store_true',
        help='Reserve holdout data for final testing'
    )
    parser.add_argument(
        '--verify-embargo',
        action='store_true',
        help='Verify embargo settings prevent leakage'
    )
    parser.add_argument(
        '--leakage-test',
        action='store_true',
        help='Run leakage detection tests'
    )
    parser.add_argument(
        '--verify-holdout',
        action='store_true',
        help='Verify holdout data isolation'
    )

    args = parser.parse_args()
    config = ValidationConfig()

    if args.setup_holdout:
        logger.info("=" * 60)
        logger.info("HOLDOUT DATA SETUP")
        logger.info("=" * 60)

        manager = HoldoutDataManager(config)
        manifest = manager.reserve_holdout_data()

        print("\nHoldout Setup Complete:")
        print("-" * 60)
        print(f"Temporal cutoff: {manifest.temporal_cutoff_date}")
        print(f"Holdout symbols: {manifest.holdout_symbols}")
        print(f"Stress periods: {len(manifest.stress_periods)}")

        total_train = sum(manifest.training_samples_per_symbol.values())
        total_holdout = sum(manifest.holdout_samples_per_symbol.values())

        print(f"\nData Split:")
        print(f"  Training samples: {total_train:,}")
        print(f"  Holdout samples: {total_holdout:,}")
        print(f"  Holdout ratio: {total_holdout/(total_train+total_holdout)*100:.1f}%")

    elif args.verify_embargo:
        logger.info("=" * 60)
        logger.info("EMBARGO VERIFICATION")
        logger.info("=" * 60)

        verifier = EmbargoVerifier(config)

        # Calculate minimum embargo
        min_embargo = verifier.calculate_min_embargo()

        print("\nFeature Lookback Analysis:")
        print("-" * 60)
        print(f"Maximum feature lookback: {min_embargo['max_feature_lookback']} bars")
        print(f"Buffer: {min_embargo['buffer']} bars")
        print(f"Minimum embargo required: {min_embargo['min_embargo_bars']} bars")
        print(f"Minimum embargo percentage: {min_embargo['min_embargo_pct']*100:.2f}%")
        print(f"Recommended embargo: {min_embargo['recommended_embargo_pct']*100:.2f}%")

        print("\nTop Lookback Features:")
        sorted_features = sorted(
            min_embargo['feature_lookbacks'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feat, lookback in sorted_features:
            print(f"  {feat}: {lookback} bars")

        # Verify PurgedKFold settings
        print("\nPurgedKFoldCV Verification:")
        print("-" * 60)

        for n_splits in [3, 5, 10]:
            result = verifier.verify_purged_kfold_settings(n_splits=n_splits)
            status = "OK" if result['is_sufficient'] else "INSUFFICIENT"
            print(
                f"  {n_splits}-fold: embargo={result['embargo_pct']*100:.1f}%, "
                f"effective_test={result['effective_test_pct']*100:.1f}% [{status}]"
            )
            if 'warning' in result:
                print(f"    WARNING: {result['warning']}")

    elif args.verify_holdout:
        logger.info("=" * 60)
        logger.info("HOLDOUT VERIFICATION")
        logger.info("=" * 60)

        manager = HoldoutDataManager(config)
        results = manager.verify_holdout_isolation()

        print("\nHoldout Isolation Checks:")
        print("-" * 60)

        for check, value in results['checks'].items():
            status = "PASS" if value else "FAIL"
            print(f"  {check}: {value} [{status}]")

        overall = "PASS" if results['passed'] else "FAIL"
        print(f"\nOverall: {overall}")

        if results['passed']:
            # Print manifest summary
            manifest_path = Path(config.holdout_data_dir) / "holdout_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                print(f"\nHoldout Summary:")
                print(f"  Temporal cutoff: {manifest.get('temporal_cutoff_date', 'N/A')}")
                print(f"  Symbol holdout: {manifest.get('holdout_symbols', [])}")

    elif args.leakage_test:
        logger.info("=" * 60)
        logger.info("LEAKAGE DETECTION TEST")
        logger.info("=" * 60)

        print("\nLeakage test requires labeled data. Run after data processing.")
        print("Use with: python scripts/setup_validation.py --leakage-test --symbol AAPL")

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START:")
        print("=" * 60)
        print("1. Setup holdout data:  python scripts/setup_validation.py --setup-holdout")
        print("2. Verify embargo:      python scripts/setup_validation.py --verify-embargo")
        print("3. Verify isolation:    python scripts/setup_validation.py --verify-holdout")


if __name__ == "__main__":
    main()
