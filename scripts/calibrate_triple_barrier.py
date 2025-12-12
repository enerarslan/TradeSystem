"""
Triple Barrier Parameter Calibration Pipeline
==============================================

This script implements PRIORITY 2 tasks from AI_AGENT_INSTRUCTIONS.md:
- Task 4: Calibrate Triple Barrier Per Symbol
- Task 5: Validate Label Quality

Calibrates barrier parameters for each symbol based on:
- 20-day ATR (Average True Range) for barrier width
- Historical analysis for optimal holding period
- VIX-based regime adjustment

Usage:
    python scripts/calibrate_triple_barrier.py --calibrate     # Calibrate all symbols
    python scripts/calibrate_triple_barrier.py --validate      # Validate label quality
    python scripts/calibrate_triple_barrier.py --analyze AAPL  # Analyze specific symbol

Author: AlphaTrade System
Based on AFML (Advances in Financial Machine Learning) best practices
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import argparse
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from src, fallback to standalone implementation
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

try:
    from src.data.labeling import (
        TripleBarrierLabeler,
        TripleBarrierConfig,
        VolatilityEstimator
    )
    USE_SRC_LABELING = True
except ImportError:
    USE_SRC_LABELING = False
    logger.warning("Could not import src.data.labeling, using standalone implementation")

# Standalone implementation if imports fail
if not USE_SRC_LABELING:
    @dataclass
    class TripleBarrierConfig:
        """Configuration for Triple Barrier Method"""
        pt_sl_ratio: Tuple[float, float] = (1.0, 1.0)
        volatility_lookback: int = 20
        volatility_method: str = "ewm"
        max_holding_period: int = 10
        min_return: float = 0.0
        n_jobs: int = 1
        use_side: bool = False

    class VolatilityEstimator:
        """Volatility estimation for barrier widths"""
        def __init__(self, method: str = "ewm", lookback: int = 20, min_periods: int = 5):
            self.method = method
            self.lookback = lookback
            self.min_periods = min_periods

        def estimate(self, prices: pd.DataFrame, column: str = "close") -> pd.Series:
            if self.method == "ewm":
                returns = prices[column].pct_change()
                return returns.ewm(span=self.lookback, min_periods=self.min_periods).std()
            else:
                returns = prices[column].pct_change()
                return returns.rolling(window=self.lookback, min_periods=self.min_periods).std()

    class TripleBarrierLabeler:
        """Triple Barrier Method for labeling (standalone)"""
        def __init__(self, config: TripleBarrierConfig = None):
            self.config = config or TripleBarrierConfig()
            self.volatility_estimator = VolatilityEstimator(
                method=self.config.volatility_method,
                lookback=self.config.volatility_lookback
            )

        def get_events_with_ohlcv(self, prices: pd.DataFrame, pt_sl: Tuple[float, float] = None) -> pd.DataFrame:
            """Generate labels using Triple Barrier Method"""
            pt_sl = pt_sl or self.config.pt_sl_ratio
            close = prices['close']
            target = self.volatility_estimator.estimate(prices, 'close')

            events = []
            for i in range(len(close) - self.config.max_holding_period):
                t0 = close.index[i]
                t1_idx = min(i + self.config.max_holding_period, len(close) - 1)
                t1 = close.index[t1_idx]

                vol = target.iloc[i]
                if pd.isna(vol) or vol <= 0:
                    continue

                path = close.iloc[i:t1_idx + 1]
                entry = path.iloc[0]
                returns = (path / entry - 1)

                upper = vol * pt_sl[0]
                lower = -vol * pt_sl[1]

                # Determine label
                upper_touch = returns[returns >= upper]
                lower_touch = returns[returns <= lower]

                if len(upper_touch) > 0 and (len(lower_touch) == 0 or upper_touch.index[0] <= lower_touch.index[0]):
                    label = 1
                    touch = 'upper'
                elif len(lower_touch) > 0:
                    label = -1
                    touch = 'lower'
                else:
                    label = np.sign(returns.iloc[-1]) if returns.iloc[-1] != 0 else 0
                    touch = 'vertical'

                events.append({
                    't0': t0,
                    't1': t1,
                    'label': label,
                    'touch_type': touch,
                    'ret': returns.iloc[-1]
                })

            df = pd.DataFrame(events)
            if len(df) > 0:
                df = df.set_index('t0')
                df['bin_label'] = (df['label'] > 0).astype(int)
            return df


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BarrierCalibrationResult:
    """Results from barrier calibration for a single symbol"""
    symbol: str

    # ATR-based parameters
    atr_20: float                      # 20-day Average True Range
    atr_pct: float                     # ATR as percentage of price
    volatility_group: str              # "low", "medium", "high"

    # Calibrated barrier parameters
    profit_target_atr_mult: float      # PT = mult * ATR
    stop_loss_atr_mult: float          # SL = mult * ATR
    max_holding_period: int            # Optimal holding period in bars

    # Barrier touch analysis
    avg_time_to_upper: float           # Average bars to hit PT
    avg_time_to_lower: float           # Average bars to hit SL
    avg_time_to_vertical: float        # Average bars to hit vertical

    # Label distribution
    upper_touch_pct: float             # % hitting profit target
    lower_touch_pct: float             # % hitting stop loss
    vertical_touch_pct: float          # % hitting time barrier

    # Quality metrics
    label_autocorrelation: float       # Should be < 0.1
    class_balance: Dict[str, float]    # Class distribution

    # VIX adjustment factors
    vix_adjustment_high: float = 1.5   # Multiply barriers when VIX > 25
    vix_adjustment_extreme: float = 2.0 # Multiply barriers when VIX > 35


@dataclass
class LabelQualityReport:
    """Label quality metrics for validation"""
    symbol: str

    # Class distribution
    class_1_pct: float                 # Profitable class
    class_minus1_pct: float            # Loss class
    class_0_pct: float                 # Neutral class
    is_balanced: bool                  # Each class 25-40%?

    # Autocorrelation
    autocorr_lag1: float
    autocorr_lag2: float
    autocorr_lag5: float
    is_low_autocorr: bool              # < 0.1?

    # Barrier touch analysis
    upper_barrier_touch_rate: float
    lower_barrier_touch_rate: float
    vertical_barrier_touch_rate: float

    # Recommendations
    issues: List[str]
    recommendations: List[str]


# ============================================================================
# ATR CALCULATION
# ============================================================================

class ATRCalculator:
    """
    Calculate Average True Range (ATR) for barrier calibration.

    ATR is the gold standard for volatility-based barrier sizing because:
    - Accounts for gaps (High vs previous Close)
    - More stable than standard deviation
    - Industry standard for stop placement
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range for each bar.

        True Range = max(
            High - Low,
            |High - Previous Close|,
            |Low - Previous Close|
        )
        """
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR using exponential moving average.
        """
        true_range = self.calculate_true_range(df)
        atr = true_range.ewm(span=self.period, min_periods=self.period).mean()
        return atr

    def get_current_atr(self, df: pd.DataFrame) -> float:
        """Get the most recent ATR value."""
        atr = self.calculate_atr(df)
        return atr.iloc[-1]

    def get_atr_percentile(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR as percentage of price.
        Useful for comparing volatility across different price levels.
        """
        atr = self.calculate_atr(df)
        atr_pct = atr / df['close'] * 100
        return atr_pct


# ============================================================================
# BARRIER CALIBRATION
# ============================================================================

class TripleBarrierCalibrator:
    """
    Calibrate Triple Barrier parameters per symbol.

    The goal is to find barrier parameters that produce:
    1. Balanced class distribution (each class 25-40%)
    2. Low label autocorrelation (< 0.1)
    3. Reasonable barrier touch distribution (vertical < 40%)
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        default_pt_mult: float = 1.5,
        default_sl_mult: float = 1.0,
        default_holding: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.default_pt_mult = default_pt_mult
        self.default_sl_mult = default_sl_mult
        self.default_holding = default_holding
        self.atr_calculator = ATRCalculator(period=20)

    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load processed data for a symbol."""
        # Try processed first, then raw
        processed_path = self.data_dir / f"{symbol}_15min_clean.csv"
        raw_path = Path("data/raw") / f"{symbol}_15min.csv"

        if processed_path.exists():
            df = pd.read_csv(processed_path, parse_dates=['timestamp'], index_col='timestamp')
        elif raw_path.exists():
            df = pd.read_csv(raw_path, parse_dates=['timestamp'], index_col='timestamp')
        else:
            raise FileNotFoundError(f"No data found for {symbol}")

        return df

    def classify_volatility(self, atr_pct: float) -> str:
        """
        Classify symbol into volatility bucket.

        Thresholds adjusted for 15-minute intraday bars:
        - Low: ATR < 0.4% (very stable stocks)
        - Medium: 0.4% <= ATR < 0.6% (typical stocks)
        - High: ATR >= 0.6% (volatile stocks like TSLA, NVDA)

        Note: Intraday ATR is much lower than daily ATR
        """
        if atr_pct < 0.4:
            return "low"
        elif atr_pct < 0.6:
            return "medium"
        else:
            return "high"

    def find_optimal_holding_period(
        self,
        df: pd.DataFrame,
        atr: float,
        pt_mult: float = 1.5,
        sl_mult: float = 1.0,
        test_periods: List[int] = None
    ) -> Tuple[int, Dict[int, Dict]]:
        """
        Find optimal max_holding_period by testing different values.

        Criteria:
        - Minimize vertical barrier touches (should be < 40%)
        - Maintain reasonable class balance
        - Keep average hold time practical

        Returns:
            Tuple of (optimal_period, results_dict)
        """
        if test_periods is None:
            test_periods = [5, 10, 15, 20, 30, 40]

        results = {}
        best_period = self.default_holding
        best_score = float('inf')

        close = df['close']

        for period in test_periods:
            # Create labeler with test parameters
            config = TripleBarrierConfig(
                pt_sl_ratio=(pt_mult, sl_mult),
                volatility_lookback=20,
                max_holding_period=period
            )
            labeler = TripleBarrierLabeler(config)

            try:
                # Generate labels
                events = labeler.get_events_with_ohlcv(df)

                if len(events) < 100:
                    continue

                # Calculate barrier touch distribution
                touch_types = events['touch_type'] if 'touch_type' in events else None

                if touch_types is None:
                    # Infer from barrier times
                    vertical_pct = 0.5  # Default estimate
                else:
                    vertical_pct = (touch_types == 'vertical').mean()

                # Calculate class distribution
                labels = events['label'].dropna()
                class_dist = {
                    1: (labels == 1).mean(),
                    -1: (labels == -1).mean(),
                    0: (labels == 0).mean()
                }

                # Score: prefer low vertical touch and balanced classes
                imbalance = max(class_dist.values()) - min(class_dist.values())
                score = vertical_pct + imbalance

                results[period] = {
                    'vertical_pct': vertical_pct,
                    'class_dist': class_dist,
                    'score': score,
                    'n_samples': len(events)
                }

                if score < best_score and vertical_pct < 0.4:
                    best_score = score
                    best_period = period

            except Exception as e:
                logger.warning(f"Period {period} failed: {e}")
                continue

        return best_period, results

    def analyze_barrier_touches(
        self,
        df: pd.DataFrame,
        config: TripleBarrierConfig
    ) -> Dict[str, Any]:
        """
        Analyze how barriers are being touched.

        Returns statistics on:
        - Time to touch each barrier type
        - Percentage touching each barrier
        - Return distribution at touch
        """
        labeler = TripleBarrierLabeler(config)
        events = labeler.get_events_with_ohlcv(df)

        if len(events) == 0:
            return {}

        analysis = {
            'n_events': len(events),
        }

        # Count barrier touches
        labels = events['label'].dropna()
        analysis['upper_touch_count'] = (labels == 1).sum()
        analysis['lower_touch_count'] = (labels == -1).sum()
        analysis['vertical_touch_count'] = (labels == 0).sum()

        total = len(labels)
        analysis['upper_touch_pct'] = analysis['upper_touch_count'] / total if total > 0 else 0
        analysis['lower_touch_pct'] = analysis['lower_touch_count'] / total if total > 0 else 0
        analysis['vertical_touch_pct'] = analysis['vertical_touch_count'] / total if total > 0 else 0

        # Calculate time to touch (in bars)
        if 't1' in events.columns:
            # t1 is the barrier touch time
            for idx, row in events.iterrows():
                t1 = row['t1']
                if pd.notna(t1):
                    try:
                        start_loc = df.index.get_loc(idx)
                        end_loc = df.index.get_loc(t1)
                        bars_to_touch = end_loc - start_loc

                        if 'bars_to_touch' not in analysis:
                            analysis['bars_to_touch'] = []
                        analysis['bars_to_touch'].append(bars_to_touch)
                    except:
                        pass

        if 'bars_to_touch' in analysis and len(analysis['bars_to_touch']) > 0:
            analysis['avg_bars_to_touch'] = np.mean(analysis['bars_to_touch'])
            analysis['std_bars_to_touch'] = np.std(analysis['bars_to_touch'])

        # Return distribution
        if 'ret' in events.columns:
            returns = events['ret'].dropna()
            analysis['avg_return'] = returns.mean()
            analysis['return_std'] = returns.std()
            analysis['return_skew'] = returns.skew()

        return analysis

    def calibrate_symbol(
        self,
        symbol: str,
        optimize_holding: bool = True
    ) -> BarrierCalibrationResult:
        """
        Calibrate Triple Barrier parameters for a single symbol.

        Steps:
        1. Calculate 20-day ATR
        2. Set barriers based on ATR (PT = 1.5*ATR, SL = 1.0*ATR)
        3. Find optimal holding period
        4. Analyze barrier touch distribution
        5. Validate label quality
        """
        logger.info(f"Calibrating {symbol}...")

        # Load data
        df = self.load_symbol_data(symbol)

        # Calculate ATR
        atr = self.atr_calculator.get_current_atr(df)
        atr_pct = (atr / df['close'].iloc[-1]) * 100
        volatility_group = self.classify_volatility(atr_pct)

        logger.info(f"{symbol}: ATR={atr:.4f} ({atr_pct:.2f}%), Group={volatility_group}")

        # Set barrier multipliers based on volatility group
        if volatility_group == "low":
            pt_mult = 1.8  # Wider targets for low vol
            sl_mult = 1.2
        elif volatility_group == "high":
            pt_mult = 1.2  # Tighter targets for high vol
            sl_mult = 0.8
        else:
            pt_mult = self.default_pt_mult
            sl_mult = self.default_sl_mult

        # Find optimal holding period
        if optimize_holding:
            optimal_holding, holding_results = self.find_optimal_holding_period(
                df, atr, pt_mult, sl_mult
            )
        else:
            optimal_holding = self.default_holding
            holding_results = {}

        # Create final config and analyze
        final_config = TripleBarrierConfig(
            pt_sl_ratio=(pt_mult, sl_mult),
            volatility_lookback=20,
            max_holding_period=optimal_holding
        )

        touch_analysis = self.analyze_barrier_touches(df, final_config)

        # Calculate label autocorrelation
        labeler = TripleBarrierLabeler(final_config)
        events = labeler.get_events_with_ohlcv(df)
        labels = events['label'].dropna()

        autocorr = labels.autocorr(lag=1) if len(labels) > 10 else 0

        # Class distribution
        class_balance = {
            '1': (labels == 1).mean(),
            '-1': (labels == -1).mean(),
            '0': (labels == 0).mean()
        }

        result = BarrierCalibrationResult(
            symbol=symbol,
            atr_20=atr,
            atr_pct=atr_pct,
            volatility_group=volatility_group,
            profit_target_atr_mult=pt_mult,
            stop_loss_atr_mult=sl_mult,
            max_holding_period=optimal_holding,
            avg_time_to_upper=touch_analysis.get('avg_bars_to_touch', 0),
            avg_time_to_lower=touch_analysis.get('avg_bars_to_touch', 0),
            avg_time_to_vertical=optimal_holding,
            upper_touch_pct=touch_analysis.get('upper_touch_pct', 0),
            lower_touch_pct=touch_analysis.get('lower_touch_pct', 0),
            vertical_touch_pct=touch_analysis.get('vertical_touch_pct', 0),
            label_autocorrelation=autocorr,
            class_balance=class_balance
        )

        return result

    def calibrate_all_symbols(
        self,
        symbols: List[str] = None
    ) -> Dict[str, BarrierCalibrationResult]:
        """
        Calibrate all symbols in the universe.
        """
        if symbols is None:
            # Get all symbols from data directory
            raw_dir = Path("data/raw")
            symbols = [f.stem.replace('_15min', '') for f in raw_dir.glob('*_15min.csv')]

        results = {}
        for symbol in symbols:
            try:
                result = self.calibrate_symbol(symbol)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to calibrate {symbol}: {e}")

        return results

    def export_config(
        self,
        results: Dict[str, BarrierCalibrationResult],
        output_path: str = "config/triple_barrier_params.yaml"
    ):
        """
        Export calibration results to YAML config file.
        """
        config = {
            'version': '1.0',
            'calibration_date': datetime.now().isoformat(),
            'default_params': {
                'profit_target_atr_mult': self.default_pt_mult,
                'stop_loss_atr_mult': self.default_sl_mult,
                'max_holding_period': self.default_holding,
                'volatility_lookback': 20
            },
            'vix_adjustments': {
                'low': {'threshold': 15, 'multiplier': 0.8},
                'normal': {'threshold': 25, 'multiplier': 1.0},
                'high': {'threshold': 35, 'multiplier': 1.5},
                'extreme': {'threshold': 50, 'multiplier': 2.0}
            },
            'symbols': {}
        }

        for symbol, result in results.items():
            config['symbols'][symbol] = {
                'volatility_group': result.volatility_group,
                'atr_pct': float(round(result.atr_pct, 4)),
                'profit_target_atr_mult': float(round(result.profit_target_atr_mult, 2)),
                'stop_loss_atr_mult': float(round(result.stop_loss_atr_mult, 2)),
                'max_holding_period': int(result.max_holding_period),
                'quality_metrics': {
                    'label_autocorr': float(round(result.label_autocorrelation, 4)),
                    'upper_touch_pct': float(round(result.upper_touch_pct, 4)),
                    'lower_touch_pct': float(round(result.lower_touch_pct, 4)),
                    'vertical_touch_pct': float(round(result.vertical_touch_pct, 4))
                }
            }

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported config to {output_path}")


# ============================================================================
# LABEL QUALITY VALIDATION
# ============================================================================

class LabelQualityValidator:
    """
    Validate label quality before training.

    Checks:
    1. Class distribution (target: each class 25-40%)
    2. Label autocorrelation (target: < 0.1)
    3. Barrier touch distribution (target: vertical < 40%)
    """

    def __init__(self, config_path: str = "config/triple_barrier_params.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load barrier configuration."""
        if Path(self.config_path).exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def validate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame = None
    ) -> LabelQualityReport:
        """
        Validate label quality for a single symbol.
        """
        # Load data if not provided
        if df is None:
            processed_path = Path("data/processed") / f"{symbol}_15min_clean.csv"
            raw_path = Path("data/raw") / f"{symbol}_15min.csv"

            if processed_path.exists():
                df = pd.read_csv(processed_path, parse_dates=['timestamp'], index_col='timestamp')
            elif raw_path.exists():
                df = pd.read_csv(raw_path, parse_dates=['timestamp'], index_col='timestamp')
            else:
                raise FileNotFoundError(f"No data found for {symbol}")

        # Get symbol-specific config
        symbol_config = self.config.get('symbols', {}).get(symbol, {})
        default_config = self.config.get('default_params', {})

        pt_mult = symbol_config.get('profit_target_atr_mult', default_config.get('profit_target_atr_mult', 1.5))
        sl_mult = symbol_config.get('stop_loss_atr_mult', default_config.get('stop_loss_atr_mult', 1.0))
        holding = symbol_config.get('max_holding_period', default_config.get('max_holding_period', 10))

        # Create labeler and generate labels
        config = TripleBarrierConfig(
            pt_sl_ratio=(pt_mult, sl_mult),
            max_holding_period=holding
        )
        labeler = TripleBarrierLabeler(config)
        events = labeler.get_events_with_ohlcv(df)

        if len(events) == 0:
            return LabelQualityReport(
                symbol=symbol,
                class_1_pct=0, class_minus1_pct=0, class_0_pct=0,
                is_balanced=False,
                autocorr_lag1=0, autocorr_lag2=0, autocorr_lag5=0,
                is_low_autocorr=False,
                upper_barrier_touch_rate=0, lower_barrier_touch_rate=0, vertical_barrier_touch_rate=0,
                issues=["No events generated"],
                recommendations=["Check data quality and barrier parameters"]
            )

        labels = events['label'].dropna()

        # Class distribution
        class_1_pct = (labels == 1).mean() * 100
        class_minus1_pct = (labels == -1).mean() * 100
        class_0_pct = (labels == 0).mean() * 100

        # Check balance (each class 25-40%)
        is_balanced = all(25 <= pct <= 40 for pct in [class_1_pct, class_minus1_pct, class_0_pct])

        # Autocorrelation
        autocorr_lag1 = labels.autocorr(lag=1) if len(labels) > 10 else 0
        autocorr_lag2 = labels.autocorr(lag=2) if len(labels) > 10 else 0
        autocorr_lag5 = labels.autocorr(lag=5) if len(labels) > 10 else 0

        is_low_autocorr = abs(autocorr_lag1) < 0.1

        # Barrier touch rates
        upper_rate = (labels == 1).mean()
        lower_rate = (labels == -1).mean()
        vertical_rate = (labels == 0).mean()

        # Generate issues and recommendations
        issues = []
        recommendations = []

        if class_1_pct > 40:
            issues.append(f"High positive class: {class_1_pct:.1f}%")
            recommendations.append("Increase profit target multiplier")
        elif class_1_pct < 25:
            issues.append(f"Low positive class: {class_1_pct:.1f}%")
            recommendations.append("Decrease profit target multiplier")

        if class_minus1_pct > 40:
            issues.append(f"High negative class: {class_minus1_pct:.1f}%")
            recommendations.append("Decrease stop loss multiplier (widen stops)")
        elif class_minus1_pct < 25:
            issues.append(f"Low negative class: {class_minus1_pct:.1f}%")
            recommendations.append("Increase stop loss multiplier (tighten stops)")

        if class_0_pct > 40:
            issues.append(f"High vertical barrier touches: {class_0_pct:.1f}%")
            recommendations.append("Increase max_holding_period")

        if abs(autocorr_lag1) >= 0.1:
            issues.append(f"High label autocorrelation: {autocorr_lag1:.3f}")
            recommendations.append("Consider using CUSUM event sampling")

        return LabelQualityReport(
            symbol=symbol,
            class_1_pct=class_1_pct,
            class_minus1_pct=class_minus1_pct,
            class_0_pct=class_0_pct,
            is_balanced=is_balanced,
            autocorr_lag1=autocorr_lag1,
            autocorr_lag2=autocorr_lag2,
            autocorr_lag5=autocorr_lag5,
            is_low_autocorr=is_low_autocorr,
            upper_barrier_touch_rate=upper_rate,
            lower_barrier_touch_rate=lower_rate,
            vertical_barrier_touch_rate=vertical_rate,
            issues=issues,
            recommendations=recommendations
        )

    def validate_all_symbols(
        self,
        symbols: List[str] = None
    ) -> Dict[str, LabelQualityReport]:
        """Validate all symbols."""
        if symbols is None:
            raw_dir = Path("data/raw")
            symbols = [f.stem.replace('_15min', '') for f in raw_dir.glob('*_15min.csv')]

        reports = {}
        for symbol in symbols:
            try:
                report = self.validate_symbol(symbol)
                reports[symbol] = report
            except Exception as e:
                logger.error(f"Failed to validate {symbol}: {e}")

        return reports

    def generate_summary(
        self,
        reports: Dict[str, LabelQualityReport]
    ) -> Dict[str, Any]:
        """Generate summary statistics across all symbols."""
        if not reports:
            return {}

        summary = {
            'total_symbols': len(reports),
            'balanced_symbols': sum(1 for r in reports.values() if r.is_balanced),
            'low_autocorr_symbols': sum(1 for r in reports.values() if r.is_low_autocorr),
            'avg_class_1_pct': np.mean([r.class_1_pct for r in reports.values()]),
            'avg_class_minus1_pct': np.mean([r.class_minus1_pct for r in reports.values()]),
            'avg_class_0_pct': np.mean([r.class_0_pct for r in reports.values()]),
            'avg_autocorr': np.mean([r.autocorr_lag1 for r in reports.values()]),
            'symbols_with_issues': [s for s, r in reports.items() if r.issues],
            'common_issues': defaultdict(int)
        }

        for report in reports.values():
            for issue in report.issues:
                # Extract issue type
                if 'positive class' in issue:
                    summary['common_issues']['class_imbalance_positive'] += 1
                elif 'negative class' in issue:
                    summary['common_issues']['class_imbalance_negative'] += 1
                elif 'vertical' in issue:
                    summary['common_issues']['high_vertical_touches'] += 1
                elif 'autocorrelation' in issue:
                    summary['common_issues']['high_autocorrelation'] += 1

        summary['common_issues'] = dict(summary['common_issues'])

        return summary


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Triple Barrier Parameter Calibration"
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Calibrate barrier parameters for all symbols'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate label quality for all symbols'
    )
    parser.add_argument(
        '--analyze',
        type=str,
        help='Analyze specific symbol'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config/triple_barrier_params.yaml',
        help='Output config path'
    )

    args = parser.parse_args()

    if args.calibrate:
        logger.info("=" * 60)
        logger.info("TRIPLE BARRIER CALIBRATION")
        logger.info("=" * 60)

        calibrator = TripleBarrierCalibrator()
        results = calibrator.calibrate_all_symbols()

        # Print summary
        print("\nCalibration Results:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'ATR%':<8} {'Group':<8} {'PT Mult':<8} {'SL Mult':<8} {'Hold':<6} {'Vert%':<8}")
        print("-" * 80)

        for symbol, result in sorted(results.items()):
            print(
                f"{symbol:<10} "
                f"{result.atr_pct:<8.2f} "
                f"{result.volatility_group:<8} "
                f"{result.profit_target_atr_mult:<8.2f} "
                f"{result.stop_loss_atr_mult:<8.2f} "
                f"{result.max_holding_period:<6} "
                f"{result.vertical_touch_pct*100:<8.1f}"
            )

        # Export config
        calibrator.export_config(results, args.output)
        print(f"\nConfig exported to {args.output}")

    elif args.validate:
        logger.info("=" * 60)
        logger.info("LABEL QUALITY VALIDATION")
        logger.info("=" * 60)

        validator = LabelQualityValidator()
        reports = validator.validate_all_symbols()

        # Print summary
        print("\nLabel Quality Report:")
        print("-" * 100)
        print(f"{'Symbol':<10} {'Class+1':<10} {'Class-1':<10} {'Class0':<10} {'Autocorr':<10} {'Status':<10}")
        print("-" * 100)

        for symbol, report in sorted(reports.items()):
            status = "OK" if report.is_balanced and report.is_low_autocorr else "WARN"
            print(
                f"{symbol:<10} "
                f"{report.class_1_pct:<10.1f} "
                f"{report.class_minus1_pct:<10.1f} "
                f"{report.class_0_pct:<10.1f} "
                f"{report.autocorr_lag1:<10.3f} "
                f"{status:<10}"
            )

        # Summary
        summary = validator.generate_summary(reports)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total symbols: {summary['total_symbols']}")
        print(f"Balanced labels: {summary['balanced_symbols']}/{summary['total_symbols']}")
        print(f"Low autocorrelation: {summary['low_autocorr_symbols']}/{summary['total_symbols']}")
        print(f"Average class distribution: +1={summary['avg_class_1_pct']:.1f}%, -1={summary['avg_class_minus1_pct']:.1f}%, 0={summary['avg_class_0_pct']:.1f}%")

        if summary['common_issues']:
            print("\nCommon Issues:")
            for issue, count in summary['common_issues'].items():
                print(f"  {issue}: {count} symbols")

    elif args.analyze:
        symbol = args.analyze
        logger.info(f"Analyzing {symbol}...")

        calibrator = TripleBarrierCalibrator()
        result = calibrator.calibrate_symbol(symbol, optimize_holding=True)

        validator = LabelQualityValidator()
        report = validator.validate_symbol(symbol)

        print(f"\n{symbol} Analysis")
        print("=" * 60)
        print(f"ATR (20-day): {result.atr_20:.4f} ({result.atr_pct:.2f}%)")
        print(f"Volatility Group: {result.volatility_group}")
        print(f"Recommended PT Multiplier: {result.profit_target_atr_mult:.2f}")
        print(f"Recommended SL Multiplier: {result.stop_loss_atr_mult:.2f}")
        print(f"Optimal Holding Period: {result.max_holding_period} bars")
        print()
        print("Label Distribution:")
        print(f"  Positive (Upper Touch): {report.class_1_pct:.1f}%")
        print(f"  Negative (Lower Touch): {report.class_minus1_pct:.1f}%")
        print(f"  Neutral (Vertical):     {report.class_0_pct:.1f}%")
        print()
        print(f"Label Autocorrelation: {report.autocorr_lag1:.4f}")
        print(f"  Status: {'OK' if report.is_low_autocorr else 'HIGH (should be < 0.1)'}")

        if report.issues:
            print("\nIssues:")
            for issue in report.issues:
                print(f"  - {issue}")

        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
