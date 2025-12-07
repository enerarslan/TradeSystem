#!/usr/bin/env python3
"""
Symbol Coverage Checker
=======================

Utility script to diagnose symbol data coverage issues.
Identifies missing symbols, file naming issues, and data quality problems.

Usage:
    python scripts/check_symbol_coverage.py
    python scripts/check_symbol_coverage.py --data-path /path/to/data
    python scripts/check_symbol_coverage.py --fix  # Attempt to fix naming issues

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging
from config.symbols import ALL_SYMBOLS, CORE_SYMBOLS, SYMBOL_INFO, get_symbol_info

logger = get_logger(__name__)


# =============================================================================
# EXPECTED FILE PATTERNS
# =============================================================================

# Common file naming patterns
FILE_PATTERNS = [
    "{symbol}_15min.csv",
    "{symbol}_1h.csv",
    "{symbol}_1hour.csv",
    "{symbol}_4h.csv",
    "{symbol}_1d.csv",
    "{symbol}_daily.csv",
    "{symbol}.csv",
    "{SYMBOL}_15min.csv",  # Uppercase
    "{SYMBOL}_1h.csv",
    "{symbol}USD_15min.csv",  # Crypto-style
    "{symbol}_USDT_15min.csv",
]


# =============================================================================
# COVERAGE CHECKER
# =============================================================================

class SymbolCoverageChecker:
    """Check and diagnose symbol data coverage."""
    
    def __init__(self, data_path: Path | None = None):
        """Initialize checker."""
        settings = get_settings()
        self.data_path = data_path or settings.data.storage_path
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
    
    def check_coverage(self) -> dict[str, Any]:
        """
        Comprehensive coverage check.
        
        Returns:
            Dictionary with detailed coverage analysis
        """
        logger.info(f"\nChecking data coverage in: {self.data_path}")
        logger.info("="*60)
        
        # List all files
        all_files = list(self.data_path.glob("*"))
        csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
        
        logger.info(f"Total files: {len(all_files)}")
        logger.info(f"CSV files: {len(csv_files)}")
        
        # Extract symbols from filenames
        found_symbols = self._extract_symbols_from_files(csv_files)
        
        # Compare with expected
        expected = set(ALL_SYMBOLS)
        found = set(found_symbols.keys())
        
        missing = expected - found
        extra = found - expected
        matched = expected & found
        
        # Analyze issues
        issues = self._analyze_issues(csv_files, found_symbols)
        
        # Build report
        report = {
            "data_path": str(self.data_path),
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "expected_symbols": len(expected),
                "found_symbols": len(found),
                "matched_symbols": len(matched),
                "missing_symbols": len(missing),
                "extra_symbols": len(extra),
                "coverage_pct": len(matched) / len(expected) if expected else 0,
            },
            "symbols": {
                "expected": sorted(expected),
                "found": sorted(found),
                "matched": sorted(matched),
                "missing": sorted(missing),
                "extra": sorted(extra),
            },
            "files": {
                "total_csv": len(csv_files),
                "by_symbol": {s: files for s, files in found_symbols.items()},
            },
            "issues": issues,
        }
        
        return report
    
    def print_report(self, report: dict[str, Any]) -> None:
        """Print human-readable report."""
        summary = report["summary"]
        symbols = report["symbols"]
        
        print("\n" + "="*70)
        print("SYMBOL COVERAGE REPORT")
        print("="*70)
        print(f"Data Path: {report['data_path']}")
        print(f"Timestamp: {report['timestamp']}")
        print()
        
        print("SUMMARY")
        print(f"  Expected Symbols:  {summary['expected_symbols']}")
        print(f"  Found Symbols:     {summary['found_symbols']}")
        print(f"  Matched:           {summary['matched_symbols']}")
        print(f"  Missing:           {summary['missing_symbols']}")
        print(f"  Extra:             {summary['extra_symbols']}")
        print(f"  Coverage:          {summary['coverage_pct']:.1%}")
        print()
        
        if symbols["missing"]:
            print(f"MISSING SYMBOLS ({len(symbols['missing'])}):")
            for s in symbols["missing"]:
                info = get_symbol_info(s)
                name = info.name if info else "Unknown"
                print(f"  ✗ {s:6} - {name}")
            print()
        
        if symbols["extra"]:
            print(f"EXTRA SYMBOLS ({len(symbols['extra'])}):")
            for s in symbols["extra"]:
                print(f"  + {s}")
            print()
        
        if report["issues"]:
            print("ISSUES DETECTED:")
            for issue in report["issues"]:
                print(f"  ⚠ {issue['type']}: {issue['description']}")
                if issue.get("files"):
                    for f in issue["files"][:3]:
                        print(f"      - {f}")
                    if len(issue["files"]) > 3:
                        print(f"      ... and {len(issue['files']) - 3} more")
            print()
        
        print("="*70)
    
    def suggest_fixes(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Suggest fixes for coverage issues.
        
        Returns:
            List of suggested fixes
        """
        fixes = []
        
        # Check for case sensitivity issues
        found_lower = {s.lower(): s for s in report["symbols"]["found"]}
        for missing in report["symbols"]["missing"]:
            if missing.lower() in found_lower:
                actual = found_lower[missing.lower()]
                fixes.append({
                    "type": "rename",
                    "symbol": missing,
                    "issue": f"Case mismatch: found '{actual}' instead of '{missing}'",
                    "action": f"Rename {actual} files to {missing}",
                })
        
        # Check for alternate naming
        for missing in report["symbols"]["missing"]:
            if missing not in [f["symbol"] for f in fixes]:
                fixes.append({
                    "type": "download",
                    "symbol": missing,
                    "issue": f"No data found for {missing}",
                    "action": f"Download historical data for {missing}",
                })
        
        return fixes
    
    def _extract_symbols_from_files(
        self,
        files: list[Path],
    ) -> dict[str, list[str]]:
        """Extract symbol names from file list."""
        symbol_files: dict[str, list[str]] = {}
        
        for f in files:
            filename = f.stem  # Remove extension
            
            # Try different extraction patterns
            symbol = None
            
            # Pattern: SYMBOL_timeframe.csv
            if "_" in filename:
                parts = filename.split("_")
                potential_symbol = parts[0].upper()
                
                # Check if it looks like a valid symbol
                if 1 <= len(potential_symbol) <= 5 and potential_symbol.isalpha():
                    symbol = potential_symbol
            
            # Pattern: SYMBOL.csv
            elif filename.upper().isalpha() and len(filename) <= 5:
                symbol = filename.upper()
            
            if symbol:
                if symbol not in symbol_files:
                    symbol_files[symbol] = []
                symbol_files[symbol].append(f.name)
        
        return symbol_files
    
    def _analyze_issues(
        self,
        files: list[Path],
        found_symbols: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """Analyze potential issues."""
        issues = []
        
        # Check for inconsistent naming
        for symbol, file_list in found_symbols.items():
            if len(file_list) > 1:
                issues.append({
                    "type": "multiple_files",
                    "description": f"Symbol {symbol} has multiple data files",
                    "files": file_list,
                })
        
        # Check for unrecognized files
        unrecognized = []
        for f in files:
            filename = f.stem.upper()
            if not any(filename.startswith(s) for s in found_symbols.keys()):
                unrecognized.append(f.name)
        
        if unrecognized:
            issues.append({
                "type": "unrecognized",
                "description": f"Found {len(unrecognized)} unrecognized files",
                "files": unrecognized,
            })
        
        # Check for empty files
        empty_files = []
        for f in files:
            try:
                if f.stat().st_size == 0:
                    empty_files.append(f.name)
            except Exception:
                pass
        
        if empty_files:
            issues.append({
                "type": "empty_files",
                "description": f"Found {len(empty_files)} empty files",
                "files": empty_files,
            })
        
        return issues


# =============================================================================
# DOWNLOAD SUGGESTIONS
# =============================================================================

def get_download_commands(missing_symbols: list[str]) -> list[str]:
    """
    Generate download commands for missing symbols.
    
    Returns:
        List of suggested download commands
    """
    commands = []
    
    # Alpaca download command
    alpaca_cmd = (
        f"python scripts/download_data.py "
        f"--symbols {' '.join(missing_symbols)} "
        f"--start 2020-01-01 "
        f"--timeframe 15min"
    )
    commands.append(("Alpaca", alpaca_cmd))
    
    # Alternative: Yahoo Finance
    yfinance_cmd = f"# Using yfinance (install with: pip install yfinance)\n"
    yfinance_cmd += "import yfinance as yf\n"
    yfinance_cmd += f"symbols = {missing_symbols}\n"
    yfinance_cmd += "for symbol in symbols:\n"
    yfinance_cmd += "    df = yf.download(symbol, start='2020-01-01', interval='15m')\n"
    yfinance_cmd += "    df.to_csv(f'data/storage/{symbol}_15min.csv')\n"
    commands.append(("yfinance (Python)", yfinance_cmd))
    
    return commands


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check symbol data coverage")
    parser.add_argument("--data-path", "-d", type=str, help="Path to data directory")
    parser.add_argument("--output", "-o", type=str, help="Output JSON report path")
    parser.add_argument("--fix", action="store_true", help="Show fix suggestions")
    parser.add_argument("--download-commands", action="store_true", help="Show download commands")
    
    args = parser.parse_args()
    
    # Initialize
    data_path = Path(args.data_path) if args.data_path else None
    checker = SymbolCoverageChecker(data_path)
    
    # Run check
    report = checker.check_coverage()
    
    # Print report
    checker.print_report(report)
    
    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")
    
    # Show fixes if requested
    if args.fix:
        fixes = checker.suggest_fixes(report)
        if fixes:
            print("\nSUGGESTED FIXES:")
            print("-"*60)
            for fix in fixes:
                print(f"  [{fix['type']}] {fix['symbol']}")
                print(f"    Issue: {fix['issue']}")
                print(f"    Action: {fix['action']}")
            print()
    
    # Show download commands if requested
    if args.download_commands and report["symbols"]["missing"]:
        print("\nDOWNLOAD COMMANDS FOR MISSING SYMBOLS:")
        print("-"*60)
        for name, cmd in get_download_commands(report["symbols"]["missing"]):
            print(f"\n{name}:")
            print(cmd)
    
    # Return exit code based on coverage
    coverage = report["summary"]["coverage_pct"]
    if coverage < 0.9:
        sys.exit(1)  # Less than 90% coverage
    return 0


if __name__ == "__main__":
    configure_logging(get_settings())
    sys.exit(main() or 0)