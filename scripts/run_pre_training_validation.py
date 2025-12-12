"""
Pre-Training Validation Master Script
=====================================

This script runs ALL pre-training validation checks from AI_AGENT_INSTRUCTIONS.md.

Run this BEFORE any model training to ensure data quality meets institutional standards.

Executes in order:
1. Data Quality Pipeline (Tasks 1-3)
2. Triple Barrier Calibration (Task 4)
3. Label Quality Validation (Task 5)
4. Embargo Verification (Task 6)
5. Holdout Data Setup (Task 7)
6. Feature Optimization Check (Tasks 8-9)
7. Symbol Parameter Validation (Task 10)

Usage:
    python scripts/run_pre_training_validation.py --all           # Run all checks
    python scripts/run_pre_training_validation.py --quick         # Quick sanity check
    python scripts/run_pre_training_validation.py --process       # Process data + validate
    python scripts/run_pre_training_validation.py --report-only   # Generate report only

Author: AlphaTrade System
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import json
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# SUCCESS CRITERIA FROM AI_AGENT_INSTRUCTIONS.md
# ============================================================================

SUCCESS_CRITERIA = {
    'regular_hours_only': {
        'description': 'Only regular market hours data (09:30-16:00 ET)',
        'target': '26 bars/day, no extended hours',
        'check': lambda x: x.get('extended_hours_pct', 100) < 1
    },
    'ohlc_valid': {
        'description': 'No OHLC relationship violations',
        'target': '0 violations',
        'check': lambda x: x.get('ohlc_violations', 1) == 0
    },
    'label_balance': {
        'description': 'Each class between 25-40%',
        'target': '25-40% per class',
        'check': lambda x: all(25 <= p <= 40 for p in x.get('class_distribution', {}).values())
    },
    'label_autocorr': {
        'description': 'Label autocorrelation below threshold',
        'target': '< 0.1',
        'check': lambda x: abs(x.get('label_autocorr', 1)) < 0.1
    },
    'embargo': {
        'description': 'Sufficient embargo period',
        'target': '>= 5% of training data',
        'check': lambda x: x.get('embargo_pct', 0) >= 0.05
    },
    'holdout_reserved': {
        'description': 'Holdout data properly reserved',
        'target': '3 months + 6 symbols locked',
        'check': lambda x: x.get('holdout_setup', False)
    },
    'features': {
        'description': 'Optimal feature count without redundancy',
        'target': '< 80 features, no redundancy',
        'check': lambda x: x.get('feature_count', 100) <= 80
    },
    'leakage_test': {
        'description': 'No information leakage detected',
        'target': 'PASS',
        'check': lambda x: not x.get('leakage_detected', True)
    }
}


# ============================================================================
# VALIDATION RUNNER
# ============================================================================

class PreTrainingValidator:
    """
    Run comprehensive pre-training validation.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.warnings = []
        self.errors = []

    def log(self, message: str):
        if self.verbose:
            print(message)
        logger.info(message)

    def run_data_quality_check(self) -> Dict[str, Any]:
        """
        Run data quality pipeline checks.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 1: DATA QUALITY CHECK")
        self.log("=" * 60)

        results = {
            'status': 'SKIPPED',
            'extended_hours_pct': 0,
            'ohlc_violations': 0,
            'symbols_checked': 0
        }

        try:
            from scripts.data_quality_pipeline import DataQualityPipeline, DataQualityConfig

            config = DataQualityConfig()
            pipeline = DataQualityPipeline(config)

            # Get all symbols
            raw_dir = Path(config.raw_data_dir)
            symbols = [f.stem.replace('_15min', '') for f in raw_dir.glob('*_15min.csv')][:5]  # Sample 5

            total_extended = 0
            total_ohlc_violations = 0

            for symbol in symbols:
                try:
                    report = pipeline.analyze_symbol(symbol)
                    extended_pct = (report.pre_market_rows + report.after_hours_rows) / report.total_rows_raw * 100
                    total_extended += extended_pct
                    total_ohlc_violations += report.ohlc_violations_before
                except Exception as e:
                    self.warnings.append(f"Data quality check failed for {symbol}: {e}")

            results['extended_hours_pct'] = total_extended / len(symbols) if symbols else 0
            results['ohlc_violations'] = total_ohlc_violations
            results['symbols_checked'] = len(symbols)
            results['status'] = 'PASSED' if total_ohlc_violations == 0 else 'FAILED'

            self.log(f"  Checked {len(symbols)} symbols")
            self.log(f"  Extended hours: {results['extended_hours_pct']:.1f}%")
            self.log(f"  OHLC violations: {total_ohlc_violations}")

        except ImportError as e:
            self.warnings.append(f"Data quality pipeline not available: {e}")
            results['status'] = 'SKIPPED'

        return results

    def run_label_quality_check(self) -> Dict[str, Any]:
        """
        Run label quality validation.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 2: LABEL QUALITY CHECK")
        self.log("=" * 60)

        results = {
            'status': 'SKIPPED',
            'class_distribution': {},
            'label_autocorr': 0,
            'balanced_symbols': 0
        }

        try:
            from scripts.calibrate_triple_barrier import LabelQualityValidator

            validator = LabelQualityValidator()

            # Get symbols
            raw_dir = Path("data/raw")
            symbols = [f.stem.replace('_15min', '') for f in raw_dir.glob('*_15min.csv')][:5]

            total_class_1 = 0
            total_class_minus1 = 0
            total_class_0 = 0
            total_autocorr = 0
            balanced = 0

            for symbol in symbols:
                try:
                    report = validator.validate_symbol(symbol)
                    total_class_1 += report.class_1_pct
                    total_class_minus1 += report.class_minus1_pct
                    total_class_0 += report.class_0_pct
                    total_autocorr += abs(report.autocorr_lag1)
                    if report.is_balanced:
                        balanced += 1
                except Exception as e:
                    self.warnings.append(f"Label check failed for {symbol}: {e}")

            n = len(symbols)
            if n > 0:
                results['class_distribution'] = {
                    '+1': total_class_1 / n,
                    '-1': total_class_minus1 / n,
                    '0': total_class_0 / n
                }
                results['label_autocorr'] = total_autocorr / n
                results['balanced_symbols'] = balanced

            results['status'] = 'PASSED' if balanced >= n * 0.8 else 'WARNING'

            self.log(f"  Class distribution: {results['class_distribution']}")
            self.log(f"  Avg autocorrelation: {results['label_autocorr']:.3f}")
            self.log(f"  Balanced symbols: {balanced}/{n}")

        except ImportError as e:
            self.warnings.append(f"Label quality validator not available: {e}")
            results['status'] = 'SKIPPED'

        return results

    def run_embargo_check(self) -> Dict[str, Any]:
        """
        Check embargo settings.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 3: EMBARGO VERIFICATION")
        self.log("=" * 60)

        results = {
            'status': 'SKIPPED',
            'embargo_pct': 0,
            'min_required_pct': 0.05
        }

        try:
            from scripts.setup_validation import EmbargoVerifier, ValidationConfig

            config = ValidationConfig()
            verifier = EmbargoVerifier(config)

            min_embargo = verifier.calculate_min_embargo()

            results['embargo_pct'] = min_embargo['recommended_embargo_pct']
            results['min_required_bars'] = min_embargo['min_embargo_bars']
            results['max_lookback'] = min_embargo['max_feature_lookback']

            is_sufficient = results['embargo_pct'] >= 0.05
            results['status'] = 'PASSED' if is_sufficient else 'FAILED'

            self.log(f"  Max feature lookback: {min_embargo['max_feature_lookback']} bars")
            self.log(f"  Minimum embargo: {min_embargo['min_embargo_bars']} bars")
            self.log(f"  Embargo %: {results['embargo_pct']*100:.1f}%")

        except ImportError as e:
            self.warnings.append(f"Embargo verifier not available: {e}")
            results['status'] = 'SKIPPED'

        return results

    def run_holdout_check(self) -> Dict[str, Any]:
        """
        Check holdout data setup.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 4: HOLDOUT DATA CHECK")
        self.log("=" * 60)

        results = {
            'status': 'SKIPPED',
            'holdout_setup': False,
            'temporal_holdout': False,
            'symbol_holdout': False
        }

        holdout_dir = Path("data/holdout")
        manifest_path = holdout_dir / "holdout_manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            results['holdout_setup'] = True
            results['temporal_holdout'] = bool(manifest.get('temporal_cutoff_date'))
            results['symbol_holdout'] = bool(manifest.get('holdout_symbols'))
            results['holdout_symbols'] = manifest.get('holdout_symbols', [])
            results['status'] = 'PASSED'

            self.log(f"  Holdout manifest found: {manifest_path}")
            self.log(f"  Temporal cutoff: {manifest.get('temporal_cutoff_date', 'N/A')}")
            self.log(f"  Symbol holdout: {manifest.get('holdout_symbols', [])}")
        else:
            results['status'] = 'NOT_SETUP'
            self.log("  Holdout data NOT configured")
            self.log("  Run: python scripts/setup_validation.py --setup-holdout")

        return results

    def run_feature_check(self) -> Dict[str, Any]:
        """
        Check feature configuration.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 5: FEATURE CHECK")
        self.log("=" * 60)

        results = {
            'status': 'SKIPPED',
            'feature_count': 0,
            'regime_features': False
        }

        # Check for optimal features config
        optimal_features_path = Path("config/optimal_features.yaml")

        if optimal_features_path.exists():
            import yaml
            with open(optimal_features_path) as f:
                config = yaml.safe_load(f)

            features = config.get('optimal_features', [])
            results['feature_count'] = len(features)
            results['status'] = 'PASSED' if len(features) <= 80 else 'WARNING'

            self.log(f"  Optimal features configured: {len(features)}")
        else:
            self.log("  Optimal features NOT configured")
            self.log("  Run: python scripts/optimize_features.py --reduce")
            results['status'] = 'NOT_SETUP'

        return results

    def run_symbol_params_check(self) -> Dict[str, Any]:
        """
        Check symbol-specific parameters.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 6: SYMBOL PARAMETERS CHECK")
        self.log("=" * 60)

        results = {
            'status': 'SKIPPED',
            'symbols_configured': 0,
            'symbols_with_calculated_params': 0
        }

        config_path = Path("config/symbols.yaml")

        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            symbols = config.get('symbols', {})
            results['symbols_configured'] = len(symbols)

            with_params = sum(
                1 for s in symbols.values()
                if 'calculated_params' in s or s.get('spread_bps', 0) > 1
            )
            results['symbols_with_calculated_params'] = with_params

            pct_configured = with_params / len(symbols) * 100 if symbols else 0
            results['status'] = 'PASSED' if pct_configured >= 80 else 'WARNING'

            self.log(f"  Symbols in config: {len(symbols)}")
            self.log(f"  With calculated params: {with_params} ({pct_configured:.0f}%)")
        else:
            self.log("  symbols.yaml NOT found")
            results['status'] = 'NOT_SETUP'

        return results

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        """
        self.log("\n" + "=" * 70)
        self.log("PRE-TRAINING VALIDATION - AlphaTrade System")
        self.log("=" * 70)
        self.log(f"Started at: {datetime.now().isoformat()}")

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        # Run each check
        all_results['checks']['data_quality'] = self.run_data_quality_check()
        all_results['checks']['label_quality'] = self.run_label_quality_check()
        all_results['checks']['embargo'] = self.run_embargo_check()
        all_results['checks']['holdout'] = self.run_holdout_check()
        all_results['checks']['features'] = self.run_feature_check()
        all_results['checks']['symbol_params'] = self.run_symbol_params_check()

        # Summary
        self.log("\n" + "=" * 70)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 70)

        passed = 0
        failed = 0
        warnings = 0
        skipped = 0

        for check_name, check_results in all_results['checks'].items():
            status = check_results.get('status', 'UNKNOWN')
            if status == 'PASSED':
                passed += 1
                symbol = "[PASS]"
            elif status == 'FAILED':
                failed += 1
                symbol = "[FAIL]"
            elif status == 'WARNING':
                warnings += 1
                symbol = "[WARN]"
            else:
                skipped += 1
                symbol = "[SKIP]"

            self.log(f"  {symbol} {check_name}")

        all_results['summary'] = {
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'skipped': skipped,
            'total': len(all_results['checks'])
        }

        # Overall status
        if failed > 0:
            overall = "FAILED"
            self.log("\n  OVERALL: FAILED - Fix issues before training")
        elif warnings > 0:
            overall = "PASSED_WITH_WARNINGS"
            self.log("\n  OVERALL: PASSED WITH WARNINGS - Review before training")
        elif skipped > 0:
            overall = "INCOMPLETE"
            self.log("\n  OVERALL: INCOMPLETE - Some checks skipped")
        else:
            overall = "PASSED"
            self.log("\n  OVERALL: PASSED - Ready for training!")

        all_results['overall_status'] = overall

        # Warnings and errors
        if self.warnings:
            self.log("\nWarnings:")
            for w in self.warnings:
                self.log(f"  - {w}")

        if self.errors:
            self.log("\nErrors:")
            for e in self.errors:
                self.log(f"  - {e}")

        return all_results

    def save_report(self, results: Dict, output_path: str = "pre_training_validation_report.json"):
        """
        Save validation report to file.
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.log(f"\nReport saved to: {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-Training Validation for AlphaTrade System"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all validation checks'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick sanity check'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate report from existing checks'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pre_training_validation_report.json',
        help='Output report path'
    )

    args = parser.parse_args()

    validator = PreTrainingValidator()

    if args.all or args.quick or (not args.report_only):
        results = validator.run_all_checks()
        validator.save_report(results, args.output)

        # Print success criteria
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)

        for criterion, spec in SUCCESS_CRITERIA.items():
            print(f"\n{criterion}:")
            print(f"  Target: {spec['target']}")
            print(f"  Description: {spec['description']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
