#!/usr/bin/env python3
"""
Generate Universe Metadata for Survivorship Bias Correction.

This script creates symbol_metadata.json which is critical for preventing
survivorship bias in backtests. Without this metadata, backtests only
include currently surviving symbols, leading to overly optimistic results.

JPMorgan-level requirement:
- All symbols must have listing/delisting dates
- Universe must be reconstructed as it existed at any point in time
- Delisted symbols must be included in historical backtests

Usage:
    python scripts/generate_universe_metadata.py
    python scripts/generate_universe_metadata.py --data-path data/raw
    python scripts/generate_universe_metadata.py --include-delisted
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def detect_column_name(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Detect column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_symbol_metadata(
    file_path: Path,
    symbol: str,
) -> Dict[str, Any]:
    """
    Extract metadata for a single symbol from its data file.

    Args:
        file_path: Path to the symbol's data file
        symbol: Symbol name

    Returns:
        Dictionary containing symbol metadata
    """
    try:
        # Read the file to get date range
        df = pd.read_csv(file_path, nrows=0)
        columns = df.columns.tolist()

        # Detect date column
        date_col = detect_column_name(df, ["timestamp", "date", "Date", "Timestamp"])

        if date_col:
            df = pd.read_csv(file_path, parse_dates=[date_col])
            df[date_col] = pd.to_datetime(df[date_col])
        else:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)

        # Get date range
        if date_col:
            first_date = df[date_col].min()
            last_date = df[date_col].max()
        else:
            first_date = df.index.min()
            last_date = df.index.max()

        # Calculate basic statistics
        n_bars = len(df)

        # Determine if symbol is likely delisted
        # If last date is more than 30 days ago, assume delisted
        days_since_last = (datetime.now() - last_date).days
        is_active = days_since_last < 30

        # Get average daily volume if available
        avg_volume = None
        volume_col = detect_column_name(df, ["volume", "Volume", "vol"])
        close_col = detect_column_name(df, ["close", "Close", "adj_close"])

        if volume_col and close_col:
            # Calculate ADV (Average Daily Value)
            daily_value = df[volume_col] * df[close_col]
            avg_volume = float(daily_value.mean())

        return {
            "symbol": symbol,
            "listing_date": first_date.strftime("%Y-%m-%d"),
            "delisting_date": None if is_active else last_date.strftime("%Y-%m-%d"),
            "is_active": is_active,
            "first_data_date": first_date.strftime("%Y-%m-%d"),
            "last_data_date": last_date.strftime("%Y-%m-%d"),
            "n_bars": n_bars,
            "avg_daily_value": avg_volume,
            "sector": infer_sector(symbol),
            "data_quality": {
                "days_since_update": days_since_last,
                "has_volume": volume_col is not None,
            }
        }

    except Exception as e:
        print(f"  Error processing {symbol}: {e}")
        return {
            "symbol": symbol,
            "listing_date": None,
            "delisting_date": None,
            "is_active": False,
            "error": str(e),
        }


def infer_sector(symbol: str) -> Optional[str]:
    """
    Infer sector from symbol name.

    In production, this would be replaced with proper sector data from
    a reference data provider (Bloomberg, Reuters, etc.)
    """
    # Simple heuristic mapping for common symbols
    sector_map = {
        # Technology
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "META": "technology", "NVDA": "technology", "ADBE": "technology",
        "CRM": "technology", "INTC": "technology", "AMD": "technology",
        "CSCO": "technology", "IBM": "technology", "ORCL": "technology",
        # Healthcare
        "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
        "MRK": "healthcare", "ABBV": "healthcare", "LLY": "healthcare",
        "AMGN": "healthcare", "BMY": "healthcare", "TMO": "healthcare",
        # Financials
        "JPM": "financials", "BAC": "financials", "WFC": "financials",
        "GS": "financials", "MS": "financials", "V": "financials",
        "MA": "financials", "AXP": "financials", "BRK.B": "financials",
        # Consumer
        "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
        "HD": "consumer_discretionary", "NKE": "consumer_discretionary",
        "MCD": "consumer_discretionary", "DIS": "consumer_discretionary",
        # Consumer Staples
        "PG": "consumer_staples", "KO": "consumer_staples",
        "PEP": "consumer_staples", "WMT": "consumer_staples",
        "COST": "consumer_staples",
        # Energy
        "XOM": "energy", "CVX": "energy", "COP": "energy",
        # Industrials
        "BA": "industrials", "CAT": "industrials", "HON": "industrials",
        "UPS": "industrials", "GE": "industrials",
        # Communication
        "VZ": "communication", "T": "communication", "CMCSA": "communication",
    }

    return sector_map.get(symbol)


def generate_universe_metadata(
    data_path: str = "data/raw",
    output_path: Optional[str] = None,
    file_format: str = "csv",
) -> Dict[str, Any]:
    """
    Generate universe metadata for all symbols in data directory.

    Args:
        data_path: Path to data directory
        output_path: Path for output JSON file
        file_format: File format to look for

    Returns:
        Dictionary containing all metadata
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Find all data files
    # Try _15min format first, then simple format
    files = list(data_dir.glob(f"*_15min.{file_format}"))
    if not files:
        files = list(data_dir.glob(f"*.{file_format}"))

    if not files:
        raise ValueError(f"No {file_format} files found in {data_path}")

    print(f"Found {len(files)} data files in {data_path}")
    print()

    metadata = {
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "data_source": str(data_dir.absolute()),
        "symbols": {},
        "summary": {
            "total_symbols": 0,
            "active_symbols": 0,
            "delisted_symbols": 0,
            "sectors": {},
        }
    }

    for file_path in sorted(files):
        # Extract symbol from filename
        stem = file_path.stem
        symbol = stem.replace("_15min", "")

        # Skip non-data files
        if symbol.lower() in ["symbol_metadata", ".gitkeep"]:
            continue

        print(f"  Processing {symbol}...", end=" ")

        symbol_meta = get_symbol_metadata(file_path, symbol)
        metadata["symbols"][symbol] = symbol_meta

        # Update summary
        metadata["summary"]["total_symbols"] += 1
        if symbol_meta.get("is_active"):
            metadata["summary"]["active_symbols"] += 1
        else:
            metadata["summary"]["delisted_symbols"] += 1

        sector = symbol_meta.get("sector")
        if sector:
            metadata["summary"]["sectors"][sector] = \
                metadata["summary"]["sectors"].get(sector, 0) + 1

        print("OK" if "error" not in symbol_meta else f"WARN: {symbol_meta.get('error', '')[:30]}")

    # Save to JSON
    if output_path is None:
        output_path = data_dir / "symbol_metadata.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print("UNIVERSE METADATA GENERATED")
    print("=" * 60)
    print(f"Total symbols:    {metadata['summary']['total_symbols']}")
    print(f"Active symbols:   {metadata['summary']['active_symbols']}")
    print(f"Delisted symbols: {metadata['summary']['delisted_symbols']}")
    print()
    print("Sectors:")
    for sector, count in sorted(metadata["summary"]["sectors"].items()):
        print(f"  {sector}: {count}")
    print()
    print(f"Output saved to: {output_path}")
    print()
    print("IMPORTANT: This metadata enables survivorship bias correction.")
    print("Backtest will now use point-in-time universe reconstruction.")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate universe metadata for survivorship bias correction"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw",
        help="Path to data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/raw/symbol_metadata.json)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "parquet", "feather"],
        help="Data file format (default: csv)",
    )

    args = parser.parse_args()

    generate_universe_metadata(
        data_path=args.data_path,
        output_path=args.output,
        file_format=args.format,
    )


if __name__ == "__main__":
    main()
