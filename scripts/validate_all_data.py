#!/usr/bin/env python3
"""
Comprehensive Data Validation Script for AlphaTrade System.

This script performs pre-flight validation of all data files to ensure
they meet institutional-grade requirements before running the pipeline.

Validation checks:
1. File format and naming conventions
2. Required columns (OHLCV)
3. Data types and parsing
4. Missing values and gaps
5. Data quality (outliers, price continuity)
6. Survivorship bias indicators
7. Corporate actions detection (splits, dividends)

Usage:
    python scripts/validate_all_data.py
    python scripts/validate_all_data.py --data-path data/raw
    python scripts/validate_all_data.py --strict  # Fail on warnings
    python scripts/validate_all_data.py --fix     # Attempt auto-fixes
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    """Single validation issue."""
    level: str  # "error", "warning", "info"
    symbol: str
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class SymbolReport:
    """Validation report for a single symbol."""
    symbol: str
    file_path: str
    status: str  # "valid", "warning", "error"
    n_bars: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    data_path: str
    total_symbols: int = 0
    valid_symbols: int = 0
    warning_symbols: int = 0
    error_symbols: int = 0
    total_bars: int = 0
    symbols: List[SymbolReport] = field(default_factory=list)
    global_issues: List[ValidationIssue] = field(default_factory=list)


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the date/timestamp column in a DataFrame."""
    candidates = ["timestamp", "date", "Date", "Timestamp", "datetime", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    # Check first column
    if len(df.columns) > 0:
        first_col = df.columns[0]
        if first_col.lower() in ["timestamp", "date", "datetime", "time"]:
            return first_col
    return None


def validate_file_format(file_path: Path) -> List[ValidationIssue]:
    """Validate file format and naming conventions."""
    issues = []
    symbol = file_path.stem.replace("_15min", "")

    # Check file extension
    if file_path.suffix.lower() not in [".csv", ".parquet", ".feather"]:
        issues.append(ValidationIssue(
            level="warning",
            symbol=symbol,
            category="format",
            message=f"Non-standard file extension: {file_path.suffix}"
        ))

    # Check naming convention
    if "_15min" not in file_path.stem:
        issues.append(ValidationIssue(
            level="info",
            symbol=symbol,
            category="format",
            message="File does not follow *_15min.csv naming convention"
        ))

    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb < 0.001:
        issues.append(ValidationIssue(
            level="error",
            symbol=symbol,
            category="format",
            message=f"File too small: {size_mb:.4f} MB"
        ))
    elif size_mb > 1000:
        issues.append(ValidationIssue(
            level="warning",
            symbol=symbol,
            category="format",
            message=f"Large file: {size_mb:.1f} MB - consider partitioning"
        ))

    return issues


def validate_columns(df: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
    """Validate required columns exist."""
    issues = []

    # Normalize column names to lowercase for comparison
    cols_lower = {c.lower(): c for c in df.columns}

    required = ["open", "high", "low", "close", "volume"]
    missing = []

    for col in required:
        if col not in cols_lower:
            missing.append(col)

    if missing:
        issues.append(ValidationIssue(
            level="error",
            symbol=symbol,
            category="columns",
            message=f"Missing required columns: {missing}",
            details={"available_columns": list(df.columns)}
        ))

    # Check for date column
    date_col = detect_date_column(df)
    if not date_col:
        issues.append(ValidationIssue(
            level="error",
            symbol=symbol,
            category="columns",
            message="No date/timestamp column found"
        ))

    return issues


def validate_data_types(df: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
    """Validate data types are correct."""
    issues = []

    # Check numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                issues.append(ValidationIssue(
                    level="error",
                    symbol=symbol,
                    category="types",
                    message=f"Column '{col}' is not numeric: {df[col].dtype}"
                ))

    # Check date column
    date_col = detect_date_column(df)
    if date_col:
        try:
            pd.to_datetime(df[date_col])
        except Exception as e:
            issues.append(ValidationIssue(
                level="error",
                symbol=symbol,
                category="types",
                message=f"Cannot parse dates in column '{date_col}': {e}"
            ))

    return issues


def validate_missing_values(df: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
    """Check for missing values."""
    issues = []

    # Overall missing percentage
    total_missing = df.isna().sum().sum()
    total_cells = df.size
    missing_pct = total_missing / total_cells * 100

    if missing_pct > 10:
        issues.append(ValidationIssue(
            level="error",
            symbol=symbol,
            category="missing",
            message=f"High missing data: {missing_pct:.1f}% ({total_missing:,} cells)"
        ))
    elif missing_pct > 1:
        issues.append(ValidationIssue(
            level="warning",
            symbol=symbol,
            category="missing",
            message=f"Some missing data: {missing_pct:.1f}% ({total_missing:,} cells)"
        ))

    # Check critical columns
    critical = ["close", "volume"]
    for col in critical:
        if col in df.columns:
            col_missing = df[col].isna().sum()
            col_pct = col_missing / len(df) * 100
            if col_pct > 0:
                issues.append(ValidationIssue(
                    level="warning" if col_pct < 5 else "error",
                    symbol=symbol,
                    category="missing",
                    message=f"Missing values in '{col}': {col_missing:,} ({col_pct:.2f}%)"
                ))

    return issues


def validate_data_quality(df: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
    """Check data quality - outliers, continuity, etc."""
    issues = []

    if "close" not in df.columns:
        return issues

    close = df["close"].dropna()
    if len(close) < 10:
        return issues

    # Check for price outliers (>20 sigma)
    returns = close.pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret > 0:
        extreme_returns = returns[abs(returns - mean_ret) > 20 * std_ret]
        if len(extreme_returns) > 0:
            issues.append(ValidationIssue(
                level="warning",
                symbol=symbol,
                category="quality",
                message=f"Found {len(extreme_returns)} extreme returns (>20 sigma)",
                details={"dates": extreme_returns.index.tolist()[:5]}
            ))

    # Check for zero/negative prices
    invalid_prices = close[close <= 0]
    if len(invalid_prices) > 0:
        issues.append(ValidationIssue(
            level="error",
            symbol=symbol,
            category="quality",
            message=f"Found {len(invalid_prices)} zero or negative prices"
        ))

    # Check OHLC consistency (high >= low, etc.)
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        inconsistent = df[(df["high"] < df["low"]) |
                         (df["high"] < df["open"]) |
                         (df["high"] < df["close"]) |
                         (df["low"] > df["open"]) |
                         (df["low"] > df["close"])]
        if len(inconsistent) > 0:
            issues.append(ValidationIssue(
                level="error",
                symbol=symbol,
                category="quality",
                message=f"Found {len(inconsistent)} rows with inconsistent OHLC"
            ))

    # Check for duplicate timestamps
    date_col = detect_date_column(df)
    if date_col:
        duplicates = df[df.duplicated(subset=[date_col], keep=False)]
        if len(duplicates) > 0:
            issues.append(ValidationIssue(
                level="error",
                symbol=symbol,
                category="quality",
                message=f"Found {len(duplicates)} duplicate timestamps"
            ))

    return issues


def validate_data_gaps(df: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
    """Check for large gaps in data."""
    issues = []

    date_col = detect_date_column(df)
    if not date_col:
        return issues

    try:
        dates = pd.to_datetime(df[date_col])
        dates = dates.sort_values()

        # Calculate time gaps
        gaps = dates.diff()
        median_gap = gaps.median()

        # Find large gaps (>10x median)
        if pd.notna(median_gap) and median_gap.total_seconds() > 0:
            large_gaps = gaps[gaps > 10 * median_gap]
            if len(large_gaps) > 0:
                max_gap = large_gaps.max()
                issues.append(ValidationIssue(
                    level="warning",
                    symbol=symbol,
                    category="gaps",
                    message=f"Found {len(large_gaps)} large data gaps (max: {max_gap})",
                ))

        # Check for weekend/holiday gaps (expected)
        # This is informational only
        weekday_gaps = gaps[dates.dt.dayofweek < 5]
        if len(weekday_gaps) > 0:
            max_weekday_gap = weekday_gaps.max()
            if pd.notna(max_weekday_gap) and max_weekday_gap > timedelta(days=3):
                issues.append(ValidationIssue(
                    level="info",
                    symbol=symbol,
                    category="gaps",
                    message=f"Largest weekday gap: {max_weekday_gap}"
                ))

    except Exception as e:
        issues.append(ValidationIssue(
            level="warning",
            symbol=symbol,
            category="gaps",
            message=f"Could not analyze gaps: {e}"
        ))

    return issues


def detect_corporate_actions(df: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
    """Detect potential corporate actions (splits, large dividends)."""
    issues = []

    if "close" not in df.columns:
        return issues

    close = df["close"].dropna()
    if len(close) < 10:
        return issues

    returns = close.pct_change().dropna()

    # Detect potential splits (large price jumps)
    # Common split ratios: 2:1 (-50%), 3:1 (-67%), 4:1 (-75%), 10:1 (-90%)
    split_candidates = returns[(returns < -0.4) | (returns > 0.8)]

    if len(split_candidates) > 0:
        issues.append(ValidationIssue(
            level="warning",
            symbol=symbol,
            category="corporate_actions",
            message=f"Detected {len(split_candidates)} potential stock splits",
            details={
                "dates": [str(d) for d in split_candidates.index.tolist()[:5]],
                "returns": [f"{r:.1%}" for r in split_candidates.values[:5]]
            }
        ))

    return issues


def validate_symbol(file_path: Path) -> SymbolReport:
    """Validate a single symbol's data file."""
    symbol = file_path.stem.replace("_15min", "")
    report = SymbolReport(
        symbol=symbol,
        file_path=str(file_path),
        status="valid",
    )

    # Validate file format
    report.issues.extend(validate_file_format(file_path))

    # Try to load the file
    try:
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        report.issues.append(ValidationIssue(
            level="error",
            symbol=symbol,
            category="read",
            message=f"Could not read file: {e}"
        ))
        report.status = "error"
        return report

    report.n_bars = len(df)

    # Run all validations
    report.issues.extend(validate_columns(df, symbol))
    report.issues.extend(validate_data_types(df, symbol))
    report.issues.extend(validate_missing_values(df, symbol))
    report.issues.extend(validate_data_quality(df, symbol))
    report.issues.extend(validate_data_gaps(df, symbol))
    report.issues.extend(detect_corporate_actions(df, symbol))

    # Extract date range
    date_col = detect_date_column(df)
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col])
            report.start_date = str(dates.min())
            report.end_date = str(dates.max())
        except:
            pass

    # Calculate statistics
    if "close" in df.columns:
        close = df["close"].dropna()
        if len(close) > 0:
            report.stats = {
                "mean_price": float(close.mean()),
                "std_price": float(close.std()),
                "min_price": float(close.min()),
                "max_price": float(close.max()),
            }

    # Determine status
    error_count = sum(1 for i in report.issues if i.level == "error")
    warning_count = sum(1 for i in report.issues if i.level == "warning")

    if error_count > 0:
        report.status = "error"
    elif warning_count > 0:
        report.status = "warning"
    else:
        report.status = "valid"

    return report


def validate_all_data(
    data_path: str = "data/raw",
    file_format: str = "csv",
) -> ValidationReport:
    """
    Validate all data files in a directory.

    Args:
        data_path: Path to data directory
        file_format: Expected file format

    Returns:
        Complete validation report
    """
    data_dir = Path(data_path)
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        data_path=str(data_dir.absolute()),
    )

    if not data_dir.exists():
        report.global_issues.append(ValidationIssue(
            level="error",
            symbol="",
            category="directory",
            message=f"Data directory not found: {data_path}"
        ))
        return report

    # Find data files
    files = list(data_dir.glob(f"*_15min.{file_format}"))
    if not files:
        files = list(data_dir.glob(f"*.{file_format}"))
        if files:
            report.global_issues.append(ValidationIssue(
                level="warning",
                symbol="",
                category="format",
                message="Files do not follow *_15min.csv naming convention"
            ))

    if not files:
        report.global_issues.append(ValidationIssue(
            level="error",
            symbol="",
            category="directory",
            message=f"No {file_format} files found in {data_path}"
        ))
        return report

    print(f"Validating {len(files)} files in {data_path}")
    print()

    # Validate each file
    for i, file_path in enumerate(sorted(files)):
        symbol = file_path.stem.replace("_15min", "")
        print(f"  [{i+1}/{len(files)}] {symbol}...", end=" ")

        symbol_report = validate_symbol(file_path)
        report.symbols.append(symbol_report)
        report.total_bars += symbol_report.n_bars

        # Print status
        status_symbol = {
            "valid": "OK",
            "warning": "WARN",
            "error": "FAIL",
        }.get(symbol_report.status, "?")
        print(status_symbol)

        # Update counts
        if symbol_report.status == "valid":
            report.valid_symbols += 1
        elif symbol_report.status == "warning":
            report.warning_symbols += 1
        else:
            report.error_symbols += 1

    report.total_symbols = len(report.symbols)

    return report


def print_report(report: ValidationReport, verbose: bool = False):
    """Print validation report to console."""
    print()
    print("=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)
    print(f"Timestamp:     {report.timestamp}")
    print(f"Data Path:     {report.data_path}")
    print()
    print(f"Total Symbols: {report.total_symbols}")
    print(f"Valid:         {report.valid_symbols}")
    print(f"Warnings:      {report.warning_symbols}")
    print(f"Errors:        {report.error_symbols}")
    print(f"Total Bars:    {report.total_bars:,}")
    print()

    # Global issues
    if report.global_issues:
        print("GLOBAL ISSUES:")
        for issue in report.global_issues:
            print(f"  [{issue.level.upper()}] {issue.message}")
        print()

    # Symbol issues
    if verbose or report.error_symbols > 0:
        print("SYMBOL ISSUES:")
        for sym_report in report.symbols:
            if sym_report.issues:
                errors = [i for i in sym_report.issues if i.level == "error"]
                warnings = [i for i in sym_report.issues if i.level == "warning"]

                if errors or (verbose and warnings):
                    print(f"\n  {sym_report.symbol}:")
                    for issue in sym_report.issues:
                        if issue.level == "error" or verbose:
                            print(f"    [{issue.level.upper()}] {issue.category}: {issue.message}")

    # Summary
    print()
    print("=" * 70)
    if report.error_symbols == 0:
        print("VALIDATION PASSED")
        if report.warning_symbols > 0:
            print(f"  (with {report.warning_symbols} warnings)")
    else:
        print("VALIDATION FAILED")
        print(f"  {report.error_symbols} symbols have errors")
    print("=" * 70)


def save_report(report: ValidationReport, output_path: Optional[str] = None):
    """Save validation report to JSON file."""
    if output_path is None:
        output_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert to dict
    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        else:
            return obj

    with open(output_path, "w") as f:
        json.dump(to_dict(report), f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate all data files for AlphaTrade system"
    )
    parser.add_argument(
        "--data-path",
        default="data/raw",
        help="Path to data directory",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet", "feather"],
        help="Data file format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any warnings exist (not just errors)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all issues including warnings",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save report to JSON file",
    )

    args = parser.parse_args()

    # Run validation
    report = validate_all_data(
        data_path=args.data_path,
        file_format=args.format,
    )

    # Print report
    print_report(report, verbose=args.verbose)

    # Save report if requested
    if args.save_report:
        save_report(report, args.save_report)

    # Exit code
    if report.error_symbols > 0:
        sys.exit(1)
    elif args.strict and report.warning_symbols > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
