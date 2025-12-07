#!/usr/bin/env python3
"""
Phase 1 Integration Module
==========================

Unified entry point for all Phase 1 tasks:
1. Backtesting Integration - Validate model performance
2. Walk-Forward Validation - Prevent overfitting
3. Risk Management Integration - Position sizing, stop-loss
4. Symbol Coverage Check - Identify missing data

This module provides a complete pipeline for model validation
and production readiness assessment.

Usage:
    # Run all Phase 1 tasks for a symbol
    python -m phase1 --symbol AAPL --all
    
    # Run specific tasks
    python -m phase1 --symbol AAPL --backtest
    python -m phase1 --symbol AAPL --walk-forward
    python -m phase1 --symbols AAPL GOOGL MSFT --portfolio

    # Check symbol coverage
    python -m phase1 --check-symbols

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, get_logger, configure_logging
from config.symbols import ALL_SYMBOLS, CORE_SYMBOLS, discover_symbols_from_data

# Phase 1 imports
from phase1.backtesting_integration import (
    BacktestRunner,
    BacktestRunConfig,
    BacktestResult,
    run_symbol_backtest,
    run_portfolio_backtest,
)
from phase1.walk_forward_validation import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward_validation,
)
from phase1.risk_integration import (
    RiskIntegrator,
    RiskIntegrationConfig,
    RiskAssessment,
)

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Phase1Config:
    """Configuration for Phase 1 pipeline."""
    # General
    output_dir: Path = field(default_factory=lambda: Path("reports/phase1"))
    
    # Backtesting
    backtest_capital: float = 100_000.0
    backtest_commission: float = 0.001
    backtest_slippage: float = 0.0005
    
    # Walk-forward
    wf_n_splits: int = 5
    wf_min_train_samples: int = 5000
    wf_scheme: str = "expanding"
    
    # Risk management
    max_position_size: float = 0.10
    max_drawdown: float = 0.15
    use_trailing_stop: bool = True
    
    # Validation thresholds
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.45
    max_overfit_ratio: float = 0.85
    min_stability_score: float = 0.70


@dataclass
class Phase1Result:
    """Combined result of Phase 1 validation."""
    symbol: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Overall status
    passed: bool = False
    status_message: str = ""
    
    # Backtesting results
    backtest_completed: bool = False
    backtest_sharpe: float = 0.0
    backtest_return: float = 0.0
    backtest_max_drawdown: float = 0.0
    backtest_win_rate: float = 0.0
    backtest_result: BacktestResult | None = None
    
    # Walk-forward results
    wf_completed: bool = False
    wf_test_accuracy: float = 0.0
    wf_overfitting_ratio: float = 0.0
    wf_stability_score: float = 0.0
    wf_significant: bool = False
    wf_result: WalkForwardResult | None = None
    
    # Risk assessment
    risk_config_valid: bool = False
    risk_warnings: list[str] = field(default_factory=list)
    
    # Production readiness score (0-100)
    readiness_score: float = 0.0
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["backtest_result"] = d["backtest_result"].to_dict() if d["backtest_result"] else None
        d["wf_result"] = d["wf_result"].to_dict() if d["wf_result"] else None
        return d
    
    def to_json(self, path: Path | str | None = None) -> str:
        """Convert to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        status_icon = "✓" if self.passed else "✗"
        
        print("\n" + "="*70)
        print(f"PHASE 1 VALIDATION RESULTS: {self.symbol}")
        print("="*70)
        print(f"Status: {status_icon} {self.status_message}")
        print(f"Readiness Score: {self.readiness_score:.0f}/100")
        print()
        
        print("BACKTESTING")
        if self.backtest_completed:
            print(f"  Sharpe Ratio:   {self.backtest_sharpe:.2f}")
            print(f"  Total Return:   {self.backtest_return:.2%}")
            print(f"  Max Drawdown:   {self.backtest_max_drawdown:.2%}")
            print(f"  Win Rate:       {self.backtest_win_rate:.2%}")
        else:
            print("  Not completed")
        print()
        
        print("WALK-FORWARD VALIDATION")
        if self.wf_completed:
            print(f"  Test Accuracy:     {self.wf_test_accuracy:.4f}")
            print(f"  Overfitting Ratio: {self.wf_overfitting_ratio:.4f}")
            print(f"  Stability Score:   {self.wf_stability_score:.4f}")
            print(f"  Significant:       {'Yes' if self.wf_significant else 'No'}")
        else:
            print("  Not completed")
        print()
        
        print("RISK MANAGEMENT")
        print(f"  Config Valid: {'Yes' if self.risk_config_valid else 'No'}")
        if self.risk_warnings:
            print("  Warnings:")
            for w in self.risk_warnings:
                print(f"    - {w}")
        print()
        
        if self.recommendations:
            print("RECOMMENDATIONS")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*70)


# =============================================================================
# PHASE 1 RUNNER
# =============================================================================

class Phase1Runner:
    """
    Orchestrates all Phase 1 validation tasks.
    
    Example:
        runner = Phase1Runner()
        
        # Run full validation
        result = runner.run_full_validation("AAPL")
        
        # Check if model is production-ready
        if result.passed:
            print("Model ready for deployment!")
        else:
            print("Model needs improvements:")
            for rec in result.recommendations:
                print(f"  - {rec}")
    """
    
    def __init__(self, config: Phase1Config | None = None):
        """Initialize Phase 1 runner."""
        self.config = config or Phase1Config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        settings = get_settings()
        
        # Results storage
        self._results: dict[str, Phase1Result] = {}
        
        logger.info("Phase1Runner initialized")
    
    def run_full_validation(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Phase1Result:
        """
        Run complete Phase 1 validation for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            Phase1Result with comprehensive validation
        """
        symbol = symbol.upper()
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"# PHASE 1 FULL VALIDATION: {symbol}")
        logger.info(f"{'#'*70}")
        
        result = Phase1Result(symbol=symbol)
        
        # 1. Backtesting
        logger.info("\n[1/3] Running Backtesting...")
        try:
            backtest_result = self._run_backtest(symbol, start_date, end_date)
            result.backtest_completed = True
            result.backtest_sharpe = backtest_result.sharpe_ratio
            result.backtest_return = backtest_result.total_return_pct
            result.backtest_max_drawdown = backtest_result.max_drawdown
            result.backtest_win_rate = backtest_result.win_rate
            result.backtest_result = backtest_result
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            result.recommendations.append(f"Fix backtesting error: {str(e)}")
        
        # 2. Walk-Forward Validation
        logger.info("\n[2/3] Running Walk-Forward Validation...")
        try:
            wf_result = self._run_walk_forward(symbol, start_date, end_date)
            result.wf_completed = True
            result.wf_test_accuracy = wf_result.mean_test_accuracy
            result.wf_overfitting_ratio = wf_result.overfitting_ratio
            result.wf_stability_score = wf_result.stability_score
            result.wf_significant = wf_result.significant
            result.wf_result = wf_result
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            result.recommendations.append(f"Fix walk-forward error: {str(e)}")
        
        # 3. Risk Management Validation
        logger.info("\n[3/3] Validating Risk Management...")
        try:
            risk_valid, risk_warnings = self._validate_risk_management()
            result.risk_config_valid = risk_valid
            result.risk_warnings = risk_warnings
        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            result.risk_warnings.append(f"Error: {str(e)}")
        
        # Calculate overall result
        self._evaluate_result(result)
        
        # Save result
        self._save_result(result)
        self._results[symbol] = result
        
        # Print summary
        result.print_summary()
        
        return result
    
    def run_backtest_only(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> BacktestResult:
        """Run only backtesting."""
        return self._run_backtest(symbol, start_date, end_date)
    
    def run_walk_forward_only(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> WalkForwardResult:
        """Run only walk-forward validation."""
        return self._run_walk_forward(symbol, start_date, end_date)
    
    def validate_multiple(
        self,
        symbols: list[str],
    ) -> dict[str, Phase1Result]:
        """
        Validate multiple symbols.
        
        Args:
            symbols: List of symbols
        
        Returns:
            Dictionary of results
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Validating {symbol}...")
            
            try:
                result = self.run_full_validation(symbol)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Validation failed for {symbol}: {e}")
                results[symbol] = Phase1Result(
                    symbol=symbol,
                    passed=False,
                    status_message=f"Error: {str(e)}",
                )
        
        # Generate aggregate report
        self._generate_aggregate_report(results)
        
        return results
    
    def check_symbol_coverage(self) -> dict[str, Any]:
        """
        Check data coverage for all symbols.
        
        Returns:
            Dictionary with coverage analysis
        """
        settings = get_settings()
        data_path = settings.data.storage_path
        
        # Discover available symbols
        available = discover_symbols_from_data(data_path)
        
        # Compare with expected
        expected = set(ALL_SYMBOLS)
        found = set(available)
        
        missing = expected - found
        extra = found - expected
        
        coverage = {
            "expected_count": len(expected),
            "found_count": len(found),
            "coverage_pct": len(found & expected) / len(expected) if expected else 0,
            "missing_symbols": sorted(missing),
            "extra_symbols": sorted(extra),
            "available_symbols": sorted(found),
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("SYMBOL COVERAGE CHECK")
        logger.info("="*60)
        logger.info(f"Expected: {coverage['expected_count']} symbols")
        logger.info(f"Found: {coverage['found_count']} symbols")
        logger.info(f"Coverage: {coverage['coverage_pct']:.1%}")
        
        if missing:
            logger.warning(f"\nMISSING SYMBOLS ({len(missing)}):")
            for s in sorted(missing):
                logger.warning(f"  - {s}")
        
        if extra:
            logger.info(f"\nEXTRA SYMBOLS ({len(extra)}):")
            for s in sorted(extra):
                logger.info(f"  + {s}")
        
        logger.info("="*60)
        
        # Save report
        report_path = self.config.output_dir / "symbol_coverage.json"
        with open(report_path, "w") as f:
            json.dump(coverage, f, indent=2)
        
        return coverage
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _run_backtest(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> BacktestResult:
        """Run backtesting for a symbol."""
        from strategies.alpha_ml_v2 import AlphaMLConfigV2
        
        backtest_config = BacktestRunConfig(
            initial_capital=self.config.backtest_capital,
            commission_pct=self.config.backtest_commission,
            slippage_pct=self.config.backtest_slippage,
            max_position_size=self.config.max_position_size,
            output_dir=self.config.output_dir / "backtests",
        )
        
        strategy_config = AlphaMLConfigV2()
        
        return run_symbol_backtest(
            symbol=symbol,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            start_date=start_date,
            end_date=end_date,
        )
    
    def _run_walk_forward(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> WalkForwardResult:
        """Run walk-forward validation."""
        wf_config = WalkForwardConfig(
            n_splits=self.config.wf_n_splits,
            min_train_samples=self.config.wf_min_train_samples,
            scheme=self.config.wf_scheme,
            output_dir=self.config.output_dir / "walk_forward",
        )
        
        return run_walk_forward_validation(
            symbol=symbol,
            config=wf_config,
            n_splits=self.config.wf_n_splits,
        )
    
    def _validate_risk_management(self) -> tuple[bool, list[str]]:
        """Validate risk management configuration."""
        warnings = []
        
        # Check configuration sanity
        if self.config.max_position_size > 0.20:
            warnings.append(f"Max position size ({self.config.max_position_size:.0%}) > 20% is aggressive")
        
        if self.config.max_drawdown > 0.20:
            warnings.append(f"Max drawdown ({self.config.max_drawdown:.0%}) > 20% is aggressive")
        
        if not self.config.use_trailing_stop:
            warnings.append("Trailing stops disabled - consider enabling for profit protection")
        
        # Test risk integrator
        risk_config = RiskIntegrationConfig(
            max_position_size=self.config.max_position_size,
            max_drawdown=self.config.max_drawdown,
            use_trailing_stop=self.config.use_trailing_stop,
        )
        
        try:
            integrator = RiskIntegrator(risk_config)
            is_valid = True
        except Exception as e:
            warnings.append(f"Risk integrator initialization failed: {e}")
            is_valid = False
        
        return is_valid, warnings
    
    def _evaluate_result(self, result: Phase1Result) -> None:
        """Evaluate overall result and set status."""
        score = 0
        passed_checks = 0
        total_checks = 0
        recommendations = []
        
        # Backtest evaluation
        if result.backtest_completed:
            total_checks += 4
            
            # Sharpe ratio check
            if result.backtest_sharpe >= self.config.min_sharpe_ratio:
                passed_checks += 1
                score += 20
            else:
                recommendations.append(
                    f"Improve Sharpe ratio ({result.backtest_sharpe:.2f} < {self.config.min_sharpe_ratio})"
                )
            
            # Return check
            if result.backtest_return > 0:
                passed_checks += 1
                score += 15
            else:
                recommendations.append("Strategy has negative returns")
            
            # Drawdown check
            if result.backtest_max_drawdown <= self.config.max_drawdown:
                passed_checks += 1
                score += 15
            else:
                recommendations.append(
                    f"Reduce max drawdown ({result.backtest_max_drawdown:.1%} > {self.config.max_drawdown:.1%})"
                )
            
            # Win rate check
            if result.backtest_win_rate >= self.config.min_win_rate:
                passed_checks += 1
                score += 10
            else:
                recommendations.append(
                    f"Improve win rate ({result.backtest_win_rate:.1%} < {self.config.min_win_rate:.1%})"
                )
        else:
            recommendations.append("Complete backtesting validation")
        
        # Walk-forward evaluation
        if result.wf_completed:
            total_checks += 3
            
            # Overfitting check
            if result.wf_overfitting_ratio >= self.config.max_overfit_ratio:
                passed_checks += 1
                score += 20
            else:
                recommendations.append(
                    f"Reduce overfitting (ratio {result.wf_overfitting_ratio:.2f} < {self.config.max_overfit_ratio})"
                )
            
            # Stability check
            if result.wf_stability_score >= self.config.min_stability_score:
                passed_checks += 1
                score += 10
            else:
                recommendations.append(
                    f"Improve model stability ({result.wf_stability_score:.2f} < {self.config.min_stability_score})"
                )
            
            # Statistical significance
            if result.wf_significant:
                passed_checks += 1
                score += 10
            else:
                recommendations.append("Model performance not statistically significant")
        else:
            recommendations.append("Complete walk-forward validation")
        
        # Risk management
        if result.risk_config_valid:
            score += 10
        else:
            recommendations.append("Fix risk management configuration")
        
        # Set final result
        result.readiness_score = min(100, score)
        result.recommendations = recommendations
        
        # Determine pass/fail
        # Requirements: Backtest Sharpe >= 0.5, WF overfitting >= 0.85, no major issues
        major_issues = [
            not result.backtest_completed,
            not result.wf_completed,
            result.backtest_sharpe < self.config.min_sharpe_ratio * 0.8,
            result.wf_overfitting_ratio < self.config.max_overfit_ratio * 0.9,
            result.backtest_return < -0.05,  # More than 5% loss
        ]
        
        if any(major_issues):
            result.passed = False
            result.status_message = "FAILED - Major issues detected"
        elif score >= 70 and len(recommendations) <= 2:
            result.passed = True
            result.status_message = "PASSED - Ready for production"
        else:
            result.passed = False
            result.status_message = "NEEDS IMPROVEMENT - See recommendations"
    
    def _save_result(self, result: Phase1Result) -> None:
        """Save result to disk."""
        output_dir = self.config.output_dir / result.symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"phase1_result_{timestamp}.json"
        result.to_json(result_path)
        
        logger.info(f"Result saved to {result_path}")
    
    def _generate_aggregate_report(
        self,
        results: dict[str, Phase1Result],
    ) -> None:
        """Generate aggregate report for multiple symbols."""
        if not results:
            return
        
        passed = [s for s, r in results.items() if r.passed]
        failed = [s for s, r in results.items() if not r.passed]
        
        avg_score = sum(r.readiness_score for r in results.values()) / len(results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(results),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "pass_rate": len(passed) / len(results),
            "avg_readiness_score": avg_score,
            "passed_symbols": passed,
            "failed_symbols": failed,
            "results": {s: r.to_dict() for s, r in results.items()},
        }
        
        # Save
        report_path = self.config.output_dir / f"aggregate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("AGGREGATE VALIDATION REPORT")
        logger.info("="*70)
        logger.info(f"Total Symbols: {len(results)}")
        logger.info(f"Passed: {len(passed)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Pass Rate: {len(passed)/len(results):.1%}")
        logger.info(f"Avg Readiness Score: {avg_score:.0f}/100")
        
        if passed:
            logger.info(f"\nPASSED: {', '.join(passed[:10])}" + ("..." if len(passed) > 10 else ""))
        if failed:
            logger.info(f"\nFAILED: {', '.join(failed[:10])}" + ("..." if len(failed) > 10 else ""))
        
        logger.info("="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point for Phase 1 CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 1 Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full validation for a symbol
    python -m phase1 --symbol AAPL --all
    
    # Run only backtesting
    python -m phase1 --symbol AAPL --backtest
    
    # Run only walk-forward validation
    python -m phase1 --symbol AAPL --walk-forward
    
    # Validate multiple symbols
    python -m phase1 --symbols AAPL GOOGL MSFT
    
    # Check symbol data coverage
    python -m phase1 --check-symbols
    
    # Validate core symbols (most liquid)
    python -m phase1 --core --all
        """
    )
    
    # Symbol selection
    parser.add_argument("--symbol", "-s", type=str, help="Single symbol to validate")
    parser.add_argument("--symbols", "-S", type=str, nargs="+", help="Multiple symbols")
    parser.add_argument("--core", "-c", action="store_true", help="Validate core symbols")
    parser.add_argument("--all-symbols", "-A", action="store_true", help="Validate all symbols")
    
    # Task selection
    parser.add_argument("--all", "-a", action="store_true", help="Run all validation tasks")
    parser.add_argument("--backtest", "-b", action="store_true", help="Run only backtesting")
    parser.add_argument("--walk-forward", "-w", action="store_true", help="Run only walk-forward")
    parser.add_argument("--check-symbols", action="store_true", help="Check symbol coverage")
    
    # Configuration
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward splits")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    # Create config
    config = Phase1Config(
        backtest_capital=args.capital,
        wf_n_splits=args.splits,
    )
    if args.output:
        config.output_dir = Path(args.output)
    
    # Create runner
    runner = Phase1Runner(config)
    
    # Execute tasks
    if args.check_symbols:
        runner.check_symbol_coverage()
        return
    
    # Determine symbols
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    elif args.core:
        symbols = CORE_SYMBOLS
    elif args.all_symbols:
        symbols = ALL_SYMBOLS
    
    if not symbols:
        parser.print_help()
        return
    
    # Execute validation
    if args.all or (not args.backtest and not args.walk_forward):
        # Full validation
        if len(symbols) == 1:
            result = runner.run_full_validation(symbols[0], start_date, end_date)
            print(f"\nFinal Result: {'PASSED' if result.passed else 'FAILED'}")
            print(f"Readiness Score: {result.readiness_score:.0f}/100")
        else:
            results = runner.validate_multiple(symbols)
            passed = sum(1 for r in results.values() if r.passed)
            print(f"\nValidated {len(results)} symbols: {passed} passed, {len(results)-passed} failed")
    
    elif args.backtest:
        for symbol in symbols:
            try:
                result = runner.run_backtest_only(symbol, start_date, end_date)
                print(f"\n{symbol}: Sharpe={result.sharpe_ratio:.2f}, Return={result.total_return_pct:.2%}")
            except Exception as e:
                print(f"\n{symbol}: Error - {e}")
    
    elif args.walk_forward:
        for symbol in symbols:
            try:
                result = runner.run_walk_forward_only(symbol, start_date, end_date)
                print(f"\n{symbol}: OOS Acc={result.mean_test_accuracy:.4f}, Overfit={result.overfitting_ratio:.4f}")
            except Exception as e:
                print(f"\n{symbol}: Error - {e}")


if __name__ == "__main__":
    configure_logging(get_settings())
    main()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "Phase1Config",
    "Phase1Result",
    # Main class
    "Phase1Runner",
    # Re-exports from submodules
    "BacktestRunner",
    "BacktestRunConfig",
    "BacktestResult",
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardResult",
    "RiskIntegrator",
    "RiskIntegrationConfig",
    "RiskAssessment",
]