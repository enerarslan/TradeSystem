#!/usr/bin/env python3
"""
Migration script to transfer CSV data to TimescaleDB.

This script reads all existing CSV files from data/raw/ and data/processed/,
validates the data, and inserts it into TimescaleDB with progress tracking.

Usage:
    python scripts/migrate_csv_to_timescale.py --host localhost --database alphatrade_db
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage.timescale_client import (
    Bar,
    ConnectionConfig,
    TimescaleClient,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationTracker:
    """Track migration progress and verify data integrity."""

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.migrated_files: dict[str, dict] = {}
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            import json
            with open(self.checkpoint_file) as f:
                self.migrated_files = json.load(f)

    def _save_checkpoint(self) -> None:
        """Save checkpoint to file."""
        import json
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.migrated_files, f, indent=2, default=str)

    def is_migrated(self, file_path: Path) -> bool:
        """Check if file has been migrated."""
        return str(file_path) in self.migrated_files

    def mark_migrated(
        self,
        file_path: Path,
        row_count: int,
        checksum: str,
    ) -> None:
        """Mark file as migrated."""
        self.migrated_files[str(file_path)] = {
            'row_count': row_count,
            'checksum': checksum,
            'migrated_at': datetime.now().isoformat(),
        }
        self._save_checkpoint()

    def get_stats(self) -> dict:
        """Get migration statistics."""
        total_rows = sum(
            f['row_count'] for f in self.migrated_files.values()
        )
        return {
            'files_migrated': len(self.migrated_files),
            'total_rows': total_rows,
        }


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate MD5 checksum of file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def extract_symbol_from_filename(file_path: Path) -> str:
    """Extract symbol from filename."""
    # Handle formats like: AAPL_15min.csv, AAPL.csv, AAPL_processed.csv
    name = file_path.stem
    parts = name.split('_')
    return parts[0].upper()


def load_csv_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and validate CSV file."""
    try:
        df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Find timestamp column
        time_cols = ['timestamp', 'time', 'datetime', 'date']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            # Try index
            if df.index.name and df.index.name.lower() in time_cols:
                df = df.reset_index()
                time_col = df.columns[0]
            else:
                logger.warning(f"No timestamp column found in {file_path}")
                return None

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df[time_col])

        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing} in {file_path}")
            return None

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Select and order columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])

        # Sort by time
        df = df.sort_values('timestamp')

        return df

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def convert_df_to_bars(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "15min",
) -> list[Bar]:
    """Convert DataFrame to list of Bar objects."""
    bars = []
    for _, row in df.iterrows():
        bars.append(Bar(
            timestamp=row['timestamp'],
            symbol=symbol,
            timeframe=timeframe,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
        ))
    return bars


def migrate_file(
    client: TimescaleClient,
    file_path: Path,
    timeframe: str = "15min",
) -> tuple[int, str]:
    """
    Migrate a single CSV file to TimescaleDB.

    Returns:
        Tuple of (rows_inserted, checksum)
    """
    # Load data
    df = load_csv_file(file_path)
    if df is None or df.empty:
        return 0, ""

    # Extract symbol
    symbol = extract_symbol_from_filename(file_path)

    # Calculate checksum
    checksum = calculate_file_checksum(file_path)

    # Convert to bars
    bars = convert_df_to_bars(df, symbol, timeframe)

    # Insert
    rows_inserted = client.insert_ohlcv(bars)

    return rows_inserted, checksum


def run_migration(
    config: ConnectionConfig,
    data_dirs: list[Path],
    timeframe: str = "15min",
    force: bool = False,
) -> dict:
    """
    Run full migration of all CSV files.

    Args:
        config: Database connection config
        data_dirs: Directories to scan for CSV files
        timeframe: Timeframe for the data
        force: If True, re-migrate already migrated files

    Returns:
        Migration statistics
    """
    # Initialize tracker
    checkpoint_file = PROJECT_ROOT / "data" / "migration_checkpoint.json"
    tracker = MigrationTracker(checkpoint_file)

    # Collect all CSV files
    csv_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            csv_files.extend(list(data_dir.glob("*.csv")))

    logger.info(f"Found {len(csv_files)} CSV files to migrate")

    # Connect to database
    client = TimescaleClient(config)
    client.connect()

    try:
        # Initialize schema
        logger.info("Initializing TimescaleDB schema...")
        client.initialize_schema()

        # Track statistics
        stats = {
            'total_files': len(csv_files),
            'files_migrated': 0,
            'files_skipped': 0,
            'total_rows': 0,
            'errors': [],
        }

        # Process files with progress bar
        for file_path in tqdm(csv_files, desc="Migrating"):
            # Check if already migrated
            if not force and tracker.is_migrated(file_path):
                stats['files_skipped'] += 1
                continue

            try:
                rows, checksum = migrate_file(client, file_path, timeframe)

                if rows > 0:
                    tracker.mark_migrated(file_path, rows, checksum)
                    stats['files_migrated'] += 1
                    stats['total_rows'] += rows
                else:
                    stats['files_skipped'] += 1

            except Exception as e:
                logger.error(f"Error migrating {file_path}: {e}")
                stats['errors'].append({
                    'file': str(file_path),
                    'error': str(e),
                })

        # Set up additional configurations
        logger.info("Setting up continuous aggregates...")
        try:
            client.setup_continuous_aggregates()
        except Exception as e:
            logger.warning(f"Could not set up continuous aggregates: {e}")

        logger.info("Setting up retention policies...")
        try:
            client.setup_retention_policies()
        except Exception as e:
            logger.warning(f"Could not set up retention policies: {e}")

        logger.info("Setting up compression...")
        try:
            client.setup_compression()
        except Exception as e:
            logger.warning(f"Could not set up compression: {e}")

        return stats

    finally:
        client.disconnect()


def verify_migration(
    config: ConnectionConfig,
    data_dirs: list[Path],
) -> dict:
    """
    Verify migration by comparing row counts.

    Returns:
        Verification results
    """
    # Count CSV rows
    csv_rows = {}
    for data_dir in data_dirs:
        if data_dir.exists():
            for file_path in data_dir.glob("*.csv"):
                df = load_csv_file(file_path)
                if df is not None:
                    symbol = extract_symbol_from_filename(file_path)
                    csv_rows[symbol] = csv_rows.get(symbol, 0) + len(df)

    # Count database rows
    client = TimescaleClient(config)
    client.connect()

    try:
        db_counts = client.execute_raw("""
            SELECT symbol, COUNT(*) as count
            FROM ohlcv_bars
            GROUP BY symbol
        """)

        db_rows = dict(zip(db_counts['symbol'], db_counts['count']))

        # Compare
        results = {
            'matches': [],
            'mismatches': [],
            'missing_in_db': [],
        }

        for symbol, csv_count in csv_rows.items():
            db_count = db_rows.get(symbol, 0)
            if csv_count == db_count:
                results['matches'].append(symbol)
            elif db_count == 0:
                results['missing_in_db'].append(symbol)
            else:
                results['mismatches'].append({
                    'symbol': symbol,
                    'csv_rows': csv_count,
                    'db_rows': db_count,
                    'diff': csv_count - db_count,
                })

        return results

    finally:
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CSV data to TimescaleDB"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Database host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5432,
        help="Database port"
    )
    parser.add_argument(
        "--database",
        default="alphatrade_db",
        help="Database name"
    )
    parser.add_argument(
        "--user",
        default="alphatrade",
        help="Database user"
    )
    parser.add_argument(
        "--password",
        default="",
        help="Database password"
    )
    parser.add_argument(
        "--timeframe",
        default="15min",
        help="Data timeframe"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-migration of all files"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing migration"
    )

    args = parser.parse_args()

    # Build config
    config = ConnectionConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )

    # Data directories
    data_dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
    ]

    if args.verify_only:
        logger.info("Verifying migration...")
        results = verify_migration(config, data_dirs)

        print("\n" + "=" * 50)
        print("MIGRATION VERIFICATION RESULTS")
        print("=" * 50)
        print(f"Symbols with matching counts: {len(results['matches'])}")
        print(f"Symbols missing in database: {len(results['missing_in_db'])}")
        print(f"Symbols with mismatches: {len(results['mismatches'])}")

        if results['mismatches']:
            print("\nMismatches:")
            for m in results['mismatches']:
                print(f"  {m['symbol']}: CSV={m['csv_rows']}, DB={m['db_rows']}, Diff={m['diff']}")

    else:
        logger.info("Starting migration...")
        stats = run_migration(config, data_dirs, args.timeframe, args.force)

        print("\n" + "=" * 50)
        print("MIGRATION COMPLETE")
        print("=" * 50)
        print(f"Total files found: {stats['total_files']}")
        print(f"Files migrated: {stats['files_migrated']}")
        print(f"Files skipped: {stats['files_skipped']}")
        print(f"Total rows inserted: {stats['total_rows']:,}")

        if stats['errors']:
            print(f"\nErrors ({len(stats['errors'])}):")
            for err in stats['errors'][:10]:
                print(f"  {err['file']}: {err['error']}")


if __name__ == "__main__":
    main()
