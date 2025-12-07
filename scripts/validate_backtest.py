"""
Backtest Metrics Validation Script
==================================

This script validates backtest reports for common calculation errors.
Run it after any backtest to catch issues before they go to production.

Usage:
    python validate_backtest.py --report backtest_report.json
    python validate_backtest.py --check-code  # Validates code for datetime.now() usage
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import re


class BacktestValidator:
    """Validates backtest reports for sanity and correctness."""
    
    REALISTIC_SHARPE_RANGE = (-3, 5)
    REALISTIC_SORTINO_RANGE = (-5, 10)
    REALISTIC_CALMAR_RANGE = (-10, 50)
    REALISTIC_MAX_DD_RANGE = (-0.95, 0.0)  # -95% to 0%
    MIN_EXPECTED_DD_PCT = 0.001  # At least 0.1% drawdown expected
    
    def __init__(self, report_path: str | Path | None = None, report_data: dict | None = None):
        """Initialize validator with report path or data."""
        if report_path:
            with open(report_path, 'r') as f:
                self.report = json.load(f)
        elif report_data:
            self.report = report_data
        else:
            raise ValueError("Either report_path or report_data must be provided")
        
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> tuple[list[str], list[str]]:
        """Run all validations."""
        self._validate_timestamps()
        self._validate_sharpe_ratio()
        self._validate_sortino_ratio()
        self._validate_calmar_ratio()
        self._validate_max_drawdown()
        self._validate_return_consistency()
        self._validate_trade_stats()
        
        return self.errors, self.warnings
    
    def _validate_timestamps(self) -> None:
        """Validate trade timestamps make sense."""
        trades = self.report.get("trades", [])
        
        for i, trade in enumerate(trades[:10]):  # Check first 10
            entry_time = trade.get("entry_time", "")
            exit_time = trade.get("exit_time", "")
            
            if entry_time and exit_time:
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                    
                    if entry_dt > exit_dt:
                        self.errors.append(
                            f"🚨 CRITICAL: Trade {i+1} has entry_time ({entry_time}) "
                            f"AFTER exit_time ({exit_time})! "
                            f"This indicates datetime.now() bug in backtesting code."
                        )
                except Exception as e:
                    self.warnings.append(f"Could not parse timestamps for trade {i+1}: {e}")
    
    def _validate_sharpe_ratio(self) -> None:
        """Validate Sharpe ratio is realistic."""
        sharpe = self.report.get("sharpe_ratio", 0)
        
        if sharpe < self.REALISTIC_SHARPE_RANGE[0] or sharpe > self.REALISTIC_SHARPE_RANGE[1]:
            self.errors.append(
                f"🚨 UNREALISTIC: Sharpe ratio = {sharpe:.2f} "
                f"(expected: {self.REALISTIC_SHARPE_RANGE[0]} to {self.REALISTIC_SHARPE_RANGE[1]}). "
                f"Check periods_per_year setting!"
            )
        elif abs(sharpe) > 3:
            self.warnings.append(
                f"⚠️ HIGH: Sharpe ratio = {sharpe:.2f} is unusually high. "
                f"Verify this is accurate."
            )
    
    def _validate_sortino_ratio(self) -> None:
        """Validate Sortino ratio is realistic."""
        sortino = self.report.get("sortino_ratio", 0)
        
        if sortino < self.REALISTIC_SORTINO_RANGE[0] or sortino > self.REALISTIC_SORTINO_RANGE[1]:
            self.errors.append(
                f"🚨 UNREALISTIC: Sortino ratio = {sortino:.2f} "
                f"(expected: {self.REALISTIC_SORTINO_RANGE[0]} to {self.REALISTIC_SORTINO_RANGE[1]}). "
                f"Check periods_per_year or downside_volatility calculation!"
            )
    
    def _validate_calmar_ratio(self) -> None:
        """Validate Calmar ratio is realistic."""
        calmar = self.report.get("calmar_ratio", 0)
        
        if calmar < self.REALISTIC_CALMAR_RANGE[0] or calmar > self.REALISTIC_CALMAR_RANGE[1]:
            self.errors.append(
                f"🚨 UNREALISTIC: Calmar ratio = {calmar:.2f} "
                f"(expected: {self.REALISTIC_CALMAR_RANGE[0]} to {self.REALISTIC_CALMAR_RANGE[1]}). "
                f"Check max_drawdown calculation!"
            )
    
    def _validate_max_drawdown(self) -> None:
        """Validate max drawdown is realistic."""
        max_dd = self.report.get("max_drawdown", 0)
        total_trades = self.report.get("trade_stats", {}).get("total_trades", 0)
        
        if max_dd < self.REALISTIC_MAX_DD_RANGE[0] or max_dd > self.REALISTIC_MAX_DD_RANGE[1]:
            self.errors.append(
                f"🚨 INVALID: Max drawdown = {max_dd:.4%} is outside valid range (-95% to 0%)."
            )
        
        # If many trades but tiny drawdown, something is wrong
        if total_trades > 100 and abs(max_dd) < self.MIN_EXPECTED_DD_PCT:
            self.errors.append(
                f"🚨 SUSPICIOUS: Max drawdown = {max_dd:.4%} is unrealistically low "
                f"for {total_trades} trades. Expected at least {self.MIN_EXPECTED_DD_PCT:.2%}."
            )
    
    def _validate_return_consistency(self) -> None:
        """Validate return metrics are internally consistent."""
        initial = self.report.get("initial_capital", 100000)
        final = self.report.get("final_capital", 0)
        total_return = self.report.get("total_return", 0)
        total_return_pct = self.report.get("total_return_pct", 0)
        
        # Check total return calculation
        expected_return = final - initial
        if abs(total_return - expected_return) > 1:  # Allow $1 rounding
            self.warnings.append(
                f"⚠️ total_return ({total_return:.2f}) != final - initial ({expected_return:.2f})"
            )
        
        # Check percentage calculation
        if initial > 0:
            expected_pct = (final - initial) / initial
            if abs(total_return_pct - expected_pct) > 0.001:  # Allow 0.1% tolerance
                self.warnings.append(
                    f"⚠️ total_return_pct ({total_return_pct:.4f}) != calculated ({expected_pct:.4f})"
                )
    
    def _validate_trade_stats(self) -> None:
        """Validate trade statistics."""
        stats = self.report.get("trade_stats", {})
        total = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0)
        
        if total > 0:
            if win_rate < 0 or win_rate > 1:
                self.errors.append(
                    f"🚨 INVALID: Win rate = {win_rate:.2%} is outside valid range (0% to 100%)"
                )
            
            # Check for unrealistic profit factors
            pf = stats.get("profit_factor", 0)
            if pf > 100 and total > 50:
                self.warnings.append(
                    f"⚠️ Profit factor = {pf:.2f} is extremely high. Verify calculations."
                )
    
    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 70)
        print("BACKTEST VALIDATION REPORT")
        print("=" * 70)
        
        if self.errors:
            print(f"\n❌ ERRORS FOUND: {len(self.errors)}")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")
        
        print("\n" + "=" * 70)


def check_code_for_datetime_now(directory: str = ".") -> list[str]:
    """
    Scan code files for problematic datetime.now() usage.
    
    Returns list of issues found.
    """
    issues = []
    
    # Patterns that indicate problematic usage
    patterns = [
        (r'datetime\.now\(\)', 'datetime.now()'),
        (r'time\.time\(\)', 'time.time()'),
        (r'entry_time\s*=\s*None', 'entry_time default to None'),
        (r'exit_time\s*=\s*None', 'exit_time default to None'),
    ]
    
    for path in Path(directory).rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        
        try:
            content = path.read_text()
            
            for pattern, desc in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    issues.append(f"{path}: Found {len(matches)} instances of {desc}")
        except Exception as e:
            pass
    
    return issues


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate backtest reports")
    parser.add_argument("--report", "-r", help="Path to backtest report JSON")
    parser.add_argument("--check-code", "-c", action="store_true", help="Check code for datetime.now() issues")
    parser.add_argument("--directory", "-d", default=".", help="Directory to check for code issues")
    
    args = parser.parse_args()
    
    if args.check_code:
        print("\n🔍 Checking code for datetime.now() issues...")
        issues = check_code_for_datetime_now(args.directory)
        
        if issues:
            print(f"\n❌ Found {len(issues)} potential issues:\n")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("\n✅ No datetime.now() issues found!")
        return
    
    if args.report:
        validator = BacktestValidator(report_path=args.report)
        errors, warnings = validator.validate_all()
        validator.print_report()
        
        # Exit with error code if errors found
        sys.exit(1 if errors else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()