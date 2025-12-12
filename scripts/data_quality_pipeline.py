"""
Data Quality Pipeline for Pre-Training Validation
==================================================

This script implements PRIORITY 1 tasks from AI_AGENT_INSTRUCTIONS.md:
- Task 1: Fix Trading Hours Contamination
- Task 2: Handle Volume Anomalies
- Task 3: Validate OHLC Data Integrity

Run this BEFORE training to ensure data quality meets institutional standards.

Usage:
    python scripts/data_quality_pipeline.py --analyze    # Analyze data quality
    python scripts/data_quality_pipeline.py --process    # Process and save clean data
    python scripts/data_quality_pipeline.py --validate   # Validate processed data

Author: AlphaTrade System
Based on AFML (Advances in Financial Machine Learning) best practices
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import argparse
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DataQualityConfig:
    """Configuration for data quality pipeline"""
    # Trading Hours (US Eastern Time)
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "America/New_York"
    expected_bars_per_day: int = 26  # 6.5 hours * 4 bars/hour for 15-min data

    # Volume Anomaly Detection
    volume_spike_threshold: float = 5.0  # Flag if volume > 5x rolling average
    volume_rolling_window: int = 20  # Bars for rolling statistics

    # OHLC Validation
    price_decimal_places: int = 2  # Standardize to 2 decimal places

    # Event Calendars (dates in YYYY-MM-DD format)
    # These should be loaded from external calendar files in production
    fomc_dates: List[str] = None

    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    def __post_init__(self):
        if self.fomc_dates is None:
            # Major FOMC dates in the data range (2021-2024)
            self.fomc_dates = [
                "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
                "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
                "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
                "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
                "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
                "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
                "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
                "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
            ]


@dataclass
class DataQualityReport:
    """Report for a single symbol's data quality"""
    symbol: str
    total_rows_raw: int
    total_rows_filtered: int
    date_range: Tuple[str, str]

    # Trading Hours Analysis
    regular_hours_rows: int
    pre_market_rows: int
    after_hours_rows: int
    overnight_rows: int
    trading_days: int
    bars_per_day_avg: float
    bars_per_day_std: float
    days_with_missing_bars: int

    # Volume Analysis
    volume_spikes_detected: int
    volume_spike_dates: List[str]
    earnings_window_bars: int
    fomc_day_bars: int
    opex_day_bars: int

    # OHLC Validation
    ohlc_violations_before: int
    ohlc_violations_after: int
    price_precision_issues: int

    # Quality Score
    quality_score: float
    issues: List[str]


# ============================================================================
# TASK 1: TRADING HOURS FILTERING
# ============================================================================

class TradingHoursAnalyzer:
    """
    Analyze and filter data to US regular trading hours.

    Critical because extended hours data has:
    - 10-100x less volume
    - Wider spreads
    - More noise
    - Different market participants

    Training on mixed data will degrade model performance.
    """

    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.market_open = pd.to_datetime(config.market_open).time()
        self.market_close = pd.to_datetime(config.market_close).time()

    def analyze_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze timestamp distribution to identify timezone and session types.

        Returns detailed breakdown of:
        - Pre-market (04:00-09:29 ET)
        - Regular hours (09:30-16:00 ET)
        - After-hours (16:01-20:00 ET)
        - Overnight (20:01-03:59 ET)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Get time distribution
        times = df.index.time

        # Define session boundaries
        pre_market_start = time(4, 0)
        pre_market_end = time(9, 29)
        regular_start = time(9, 30)
        regular_end = time(16, 0)
        after_hours_end = time(20, 0)

        # Count by session
        pre_market_mask = (times >= pre_market_start) & (times <= pre_market_end)
        regular_mask = (times >= regular_start) & (times <= regular_end)
        after_hours_mask = (times > regular_end) & (times <= after_hours_end)
        overnight_mask = ~(pre_market_mask | regular_mask | after_hours_mask)

        analysis = {
            'total_rows': len(df),
            'pre_market_rows': pre_market_mask.sum(),
            'regular_hours_rows': regular_mask.sum(),
            'after_hours_rows': after_hours_mask.sum(),
            'overnight_rows': overnight_mask.sum(),
            'pre_market_pct': pre_market_mask.mean() * 100,
            'regular_hours_pct': regular_mask.mean() * 100,
            'after_hours_pct': after_hours_mask.mean() * 100,
            'overnight_pct': overnight_mask.mean() * 100,
            'first_timestamp': str(df.index[0]),
            'last_timestamp': str(df.index[-1]),
            'unique_times': sorted(set(times))[:20],  # First 20 unique times
            'timestamp_range_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600
        }

        # Analyze bars per day for regular hours
        df_regular = df[regular_mask]
        if len(df_regular) > 0:
            bars_by_day = df_regular.groupby(df_regular.index.date).size()
            analysis['trading_days'] = len(bars_by_day)
            analysis['bars_per_day_avg'] = bars_by_day.mean()
            analysis['bars_per_day_std'] = bars_by_day.std()
            analysis['bars_per_day_min'] = bars_by_day.min()
            analysis['bars_per_day_max'] = bars_by_day.max()
            analysis['days_with_expected_bars'] = (bars_by_day == self.config.expected_bars_per_day).sum()
            analysis['days_with_missing_bars'] = (bars_by_day < self.config.expected_bars_per_day).sum()

        return analysis

    def detect_timezone(self, df: pd.DataFrame) -> str:
        """
        Detect the likely timezone of the data based on trading patterns.

        US markets:
        - Regular session starts at 09:30 ET
        - If first bar of day is 09:00, data is likely already in ET
        - If first bar of day is 14:30, data is likely in UTC
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Get first bar of each day
        first_bars = df.groupby(df.index.date).first()
        first_times = [df.loc[df.index.date == date].index[0].time()
                       for date in first_bars.index[:20]]

        # Check most common first bar time
        from collections import Counter
        time_counts = Counter(first_times)
        most_common_first = time_counts.most_common(1)[0][0]

        # Heuristic: if trading starts around 9:00 AM, it's likely ET
        # If trading starts around 14:30, it's likely UTC
        if time(8, 30) <= most_common_first <= time(10, 0):
            return "America/New_York"  # Already in ET
        elif time(13, 30) <= most_common_first <= time(15, 0):
            return "UTC"  # Need to convert to ET
        else:
            logger.warning(f"Unusual first bar time: {most_common_first}, assuming ET")
            return "America/New_York"

    def filter_trading_hours(
        self,
        df: pd.DataFrame,
        assume_timezone: str = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame to regular US market hours (09:30-16:00 ET).

        Args:
            df: DataFrame with DatetimeIndex
            assume_timezone: Override timezone detection ('ET' or 'UTC')

        Returns:
            Filtered DataFrame with only regular hours data
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Detect or use provided timezone
        if assume_timezone:
            source_tz = assume_timezone
        else:
            source_tz = self.detect_timezone(df)

        logger.info(f"Detected/assumed source timezone: {source_tz}")

        # Handle timezone conversion
        if df.index.tz is None:
            if source_tz == "America/New_York":
                # Data is already in ET, just localize
                df.index = df.index.tz_localize(self.config.timezone, ambiguous='NaT', nonexistent='NaT')
            else:
                # Data is in UTC, convert to ET
                df.index = df.index.tz_localize('UTC').tz_convert(self.config.timezone)
        else:
            df.index = df.index.tz_convert(self.config.timezone)

        # Drop any NaT indices (from DST transitions)
        df = df[df.index.notna()]

        # Filter to regular hours
        time_filter = (df.index.time >= self.market_open) & (df.index.time <= self.market_close)
        df_filtered = df[time_filter]

        logger.info(
            f"Trading hours filter: {len(df)} -> {len(df_filtered)} rows "
            f"({len(df) - len(df_filtered)} extended hours rows removed)"
        )

        return df_filtered

    def validate_daily_bar_count(
        self,
        df: pd.DataFrame,
        expected_bars: int = None
    ) -> pd.DataFrame:
        """
        Validate that each trading day has the expected number of bars.

        For 15-min bars: 26 bars per day (6.5 hours * 4 bars/hour)

        Returns DataFrame with daily bar counts and flags for incomplete days.
        """
        expected = expected_bars or self.config.expected_bars_per_day

        bars_by_day = df.groupby(df.index.date).size()

        validation_df = pd.DataFrame({
            'bar_count': bars_by_day,
            'expected': expected,
            'is_complete': bars_by_day == expected,
            'missing_bars': expected - bars_by_day
        })

        # Flag potentially problematic days
        validation_df['status'] = 'OK'
        validation_df.loc[validation_df['bar_count'] < expected * 0.8, 'status'] = 'INCOMPLETE'
        validation_df.loc[validation_df['bar_count'] > expected * 1.2, 'status'] = 'EXCESS_BARS'

        return validation_df


# ============================================================================
# TASK 2: VOLUME ANOMALY DETECTION
# ============================================================================

class VolumeAnomalyDetector:
    """
    Detect and flag volume anomalies.

    Volume spikes often occur during:
    - Earnings announcements
    - FOMC decisions
    - Options expiration (OpEx)
    - Index rebalancing
    - Breaking news

    These bars should NOT be removed but FLAGGED for the model to learn from.
    """

    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.fomc_dates = set(pd.to_datetime(config.fomc_dates).date)

    def calculate_volume_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volume statistics for anomaly detection.
        """
        df = df.copy()
        window = self.config.volume_rolling_window

        # Rolling statistics
        df['volume_rolling_mean'] = df['volume'].rolling(window, min_periods=5).mean()
        df['volume_rolling_std'] = df['volume'].rolling(window, min_periods=5).std()
        df['volume_rolling_median'] = df['volume'].rolling(window, min_periods=5).median()

        # Volume ratio (current / rolling average)
        df['volume_ratio'] = df['volume'] / df['volume_rolling_mean']

        # Z-score
        df['volume_zscore'] = (df['volume'] - df['volume_rolling_mean']) / (df['volume_rolling_std'] + 1e-10)

        return df

    def detect_volume_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag bars where volume exceeds threshold * rolling average.
        """
        df = df.copy()
        threshold = self.config.volume_spike_threshold

        if 'volume_ratio' not in df.columns:
            df = self.calculate_volume_statistics(df)

        # Flag spikes
        df['is_volume_spike'] = (df['volume_ratio'] > threshold).astype(int)

        # Severity levels
        df['volume_spike_severity'] = 0
        df.loc[df['volume_ratio'] > threshold, 'volume_spike_severity'] = 1  # Moderate
        df.loc[df['volume_ratio'] > threshold * 2, 'volume_spike_severity'] = 2  # High
        df.loc[df['volume_ratio'] > threshold * 5, 'volume_spike_severity'] = 3  # Extreme

        spike_count = df['is_volume_spike'].sum()
        logger.info(f"Detected {spike_count} volume spikes ({spike_count/len(df)*100:.2f}%)")

        return df

    def add_event_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary flags for known market events.

        Flags:
        - is_fomc_day: Federal Reserve meeting day
        - is_opex_day: Options expiration (3rd Friday of month)
        - is_earnings_window: +/- 1 day around earnings (if calendar provided)
        - is_month_end: Last trading day of month
        - is_quarter_end: Last trading day of quarter
        """
        df = df.copy()

        dates = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index).date
        dates_series = pd.Series(dates, index=df.index)

        # FOMC days
        df['is_fomc_day'] = dates_series.isin(self.fomc_dates).astype(int)

        # Options Expiration (3rd Friday of each month)
        def is_third_friday(date):
            return date.weekday() == 4 and 15 <= date.day <= 21

        df['is_opex_day'] = dates_series.apply(is_third_friday).astype(int)

        # Month-end
        df['is_month_end'] = dates_series.apply(
            lambda d: d == pd.Timestamp(d).to_period('M').to_timestamp('M').date()
        ).astype(int)

        # Quarter-end
        df['is_quarter_end'] = dates_series.apply(
            lambda d: d == pd.Timestamp(d).to_period('Q').to_timestamp('Q').date()
        ).astype(int)

        # Log event distribution
        logger.info(
            f"Event flags: FOMC={df['is_fomc_day'].sum()}, "
            f"OpEx={df['is_opex_day'].sum()}, "
            f"MonthEnd={df['is_month_end'].sum()}"
        )

        return df

    def create_earnings_calendar(
        self,
        symbols: List[str],
        earnings_file: str = None
    ) -> Dict[str, List[str]]:
        """
        Load or create earnings calendar for symbols.

        In production, this would load from:
        - SEC EDGAR filings
        - Bloomberg/Reuters calendar
        - Earnings Whispers API

        Returns dict mapping symbol to list of earnings dates.
        """
        # Placeholder - in production, load from external source
        # For now, return empty calendar
        return {symbol: [] for symbol in symbols}


# ============================================================================
# TASK 3: OHLC DATA INTEGRITY
# ============================================================================

class OHLCValidator:
    """
    Validate and fix OHLC data integrity.

    Rules:
    1. High >= max(Open, Close)
    2. Low <= min(Open, Close)
    3. High >= Low
    4. All prices > 0
    5. Consistent decimal precision
    """

    def __init__(self, config: DataQualityConfig):
        self.config = config

    def validate_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for OHLC relationship violations.

        Returns DataFrame with violation flags.
        """
        violations = pd.DataFrame(index=df.index)

        # Rule 1: High >= max(Open, Close)
        max_oc = df[['open', 'close']].max(axis=1)
        violations['high_too_low'] = (df['high'] < max_oc)

        # Rule 2: Low <= min(Open, Close)
        min_oc = df[['open', 'close']].min(axis=1)
        violations['low_too_high'] = (df['low'] > min_oc)

        # Rule 3: High >= Low
        violations['high_below_low'] = (df['high'] < df['low'])

        # Rule 4: Positive prices
        violations['negative_open'] = (df['open'] <= 0)
        violations['negative_high'] = (df['high'] <= 0)
        violations['negative_low'] = (df['low'] <= 0)
        violations['negative_close'] = (df['close'] <= 0)

        # Count violations
        violations['any_violation'] = violations.any(axis=1)

        violation_count = violations['any_violation'].sum()
        logger.info(f"Found {violation_count} OHLC violations ({violation_count/len(df)*100:.4f}%)")

        return violations

    def fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix OHLC relationship violations by adjusting High/Low.

        Strategy:
        - If High < max(O,C): Set High = max(O,C)
        - If Low > min(O,C): Set Low = min(O,C)
        - If High < Low: Swap them
        """
        df = df.copy()

        # Fix High
        max_oc = df[['open', 'close']].max(axis=1)
        high_violations = df['high'] < max_oc
        df.loc[high_violations, 'high'] = max_oc[high_violations]

        # Fix Low
        min_oc = df[['open', 'close']].min(axis=1)
        low_violations = df['low'] > min_oc
        df.loc[low_violations, 'low'] = min_oc[low_violations]

        # Fix High < Low (swap)
        swap_mask = df['high'] < df['low']
        if swap_mask.any():
            df.loc[swap_mask, ['high', 'low']] = df.loc[swap_mask, ['low', 'high']].values

        logger.info(
            f"Fixed OHLC: {high_violations.sum()} high, "
            f"{low_violations.sum()} low, {swap_mask.sum()} swaps"
        )

        return df

    def standardize_price_precision(
        self,
        df: pd.DataFrame,
        decimals: int = None
    ) -> pd.DataFrame:
        """
        Standardize all prices to consistent decimal precision.

        Inconsistent precision (2 vs 4 decimals) often indicates
        merged data from different sources.
        """
        decimals = decimals or self.config.price_decimal_places
        df = df.copy()

        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].round(decimals)

        logger.info(f"Standardized prices to {decimals} decimal places")

        return df

    def detect_price_precision_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price precision across the dataset.

        Returns statistics about decimal precision distribution.
        """
        def count_decimals(series):
            # Convert to string and count decimal places
            str_series = series.astype(str)
            decimal_counts = str_series.apply(
                lambda x: len(x.split('.')[-1]) if '.' in x else 0
            )
            return decimal_counts

        precision_stats = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                decimals = count_decimals(df[col])
                precision_stats[col] = {
                    'min_decimals': decimals.min(),
                    'max_decimals': decimals.max(),
                    'mode_decimals': decimals.mode().iloc[0] if len(decimals.mode()) > 0 else 0,
                    'inconsistent_count': (decimals != decimals.mode().iloc[0]).sum() if len(decimals.mode()) > 0 else 0
                }

        return precision_stats

    def detect_stock_splits(
        self,
        df: pd.DataFrame,
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Detect potential stock splits based on price jumps.

        A stock split causes:
        - Sudden price drop (or rise for reverse split)
        - Price change > threshold (e.g., 40%)
        - Volume adjustment in opposite direction

        Returns list of potential split dates with estimated ratios.
        """
        splits = []

        # Calculate daily returns
        daily_close = df.groupby(df.index.date)['close'].last()
        returns = daily_close.pct_change()

        # Find large price changes
        large_drops = returns[returns < -threshold]
        large_rises = returns[returns > threshold / (1 - threshold)]  # Reverse split

        for date, ret in large_drops.items():
            # Estimate split ratio
            ratio = 1 / (1 + ret)  # e.g., -50% return suggests 2:1 split
            ratio_rounded = round(ratio)

            if ratio_rounded >= 2:
                splits.append({
                    'date': str(date),
                    'return': ret,
                    'estimated_ratio': f"{ratio_rounded}:1",
                    'type': 'forward_split'
                })

        for date, ret in large_rises.items():
            ratio = 1 + ret
            ratio_rounded = round(ratio)

            if ratio_rounded >= 2:
                splits.append({
                    'date': str(date),
                    'return': ret,
                    'estimated_ratio': f"1:{ratio_rounded}",
                    'type': 'reverse_split'
                })

        if splits:
            logger.warning(f"Detected {len(splits)} potential stock splits")

        return splits


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DataQualityPipeline:
    """
    Complete data quality pipeline integrating all validators.
    """

    def __init__(self, config: DataQualityConfig = None):
        self.config = config or DataQualityConfig()
        self.hours_analyzer = TradingHoursAnalyzer(self.config)
        self.volume_detector = VolumeAnomalyDetector(self.config)
        self.ohlc_validator = OHLCValidator(self.config)
        self.reports: Dict[str, DataQualityReport] = {}

    def load_raw_data(self, symbol: str) -> pd.DataFrame:
        """Load raw CSV data for a symbol."""
        raw_dir = Path(self.config.raw_data_dir)
        file_path = raw_dir / f"{symbol}_15min.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        return df

    def analyze_symbol(self, symbol: str) -> DataQualityReport:
        """
        Run full analysis on a single symbol.

        Returns detailed quality report without modifying data.
        """
        logger.info(f"Analyzing {symbol}...")

        # Load data
        df = self.load_raw_data(symbol)
        total_rows_raw = len(df)

        # Trading hours analysis
        hours_analysis = self.hours_analyzer.analyze_timestamps(df)

        # Filter to regular hours for further analysis
        df_filtered = self.hours_analyzer.filter_trading_hours(df, assume_timezone="America/New_York")
        total_rows_filtered = len(df_filtered)

        # Volume analysis
        df_vol = self.volume_detector.calculate_volume_statistics(df_filtered)
        df_vol = self.volume_detector.detect_volume_spikes(df_vol)
        df_vol = self.volume_detector.add_event_flags(df_vol)

        volume_spike_dates = df_vol[df_vol['is_volume_spike'] == 1].index.date

        # OHLC validation
        violations_before = self.ohlc_validator.validate_ohlc_relationships(df_filtered)
        precision_issues = self.ohlc_validator.detect_price_precision_issues(df_filtered)
        splits = self.ohlc_validator.detect_stock_splits(df_filtered)

        # Build report
        issues = []

        # Check trading hours issues
        if hours_analysis.get('pre_market_pct', 0) > 5:
            issues.append(f"High pre-market data: {hours_analysis['pre_market_pct']:.1f}%")
        if hours_analysis.get('after_hours_pct', 0) > 10:
            issues.append(f"High after-hours data: {hours_analysis['after_hours_pct']:.1f}%")
        if hours_analysis.get('days_with_missing_bars', 0) > 0:
            issues.append(f"{hours_analysis['days_with_missing_bars']} days with missing bars")

        # Check volume issues
        spike_count = df_vol['is_volume_spike'].sum()
        if spike_count > len(df_filtered) * 0.05:
            issues.append(f"High volume spike rate: {spike_count/len(df_filtered)*100:.1f}%")

        # Check OHLC issues
        violation_count = violations_before['any_violation'].sum()
        if violation_count > 0:
            issues.append(f"{violation_count} OHLC violations")

        # Check splits
        if splits:
            issues.append(f"{len(splits)} potential stock splits detected")

        # Calculate quality score (100 = perfect, lower = more issues)
        quality_score = 100.0
        quality_score -= hours_analysis.get('pre_market_pct', 0) * 0.5
        quality_score -= hours_analysis.get('after_hours_pct', 0) * 0.3
        quality_score -= (violation_count / max(len(df_filtered), 1)) * 1000
        quality_score -= len(splits) * 5
        quality_score = max(0, min(100, quality_score))

        report = DataQualityReport(
            symbol=symbol,
            total_rows_raw=total_rows_raw,
            total_rows_filtered=total_rows_filtered,
            date_range=(str(df.index[0]), str(df.index[-1])),
            regular_hours_rows=hours_analysis.get('regular_hours_rows', 0),
            pre_market_rows=hours_analysis.get('pre_market_rows', 0),
            after_hours_rows=hours_analysis.get('after_hours_rows', 0),
            overnight_rows=hours_analysis.get('overnight_rows', 0),
            trading_days=hours_analysis.get('trading_days', 0),
            bars_per_day_avg=hours_analysis.get('bars_per_day_avg', 0),
            bars_per_day_std=hours_analysis.get('bars_per_day_std', 0),
            days_with_missing_bars=hours_analysis.get('days_with_missing_bars', 0),
            volume_spikes_detected=spike_count,
            volume_spike_dates=[str(d) for d in list(set(volume_spike_dates))[:10]],  # First 10
            earnings_window_bars=0,  # Placeholder until earnings calendar available
            fomc_day_bars=df_vol['is_fomc_day'].sum(),
            opex_day_bars=df_vol['is_opex_day'].sum(),
            ohlc_violations_before=violation_count,
            ohlc_violations_after=0,  # Set after processing
            price_precision_issues=sum(
                v.get('inconsistent_count', 0) for v in precision_issues.values()
            ),
            quality_score=quality_score,
            issues=issues
        )

        self.reports[symbol] = report
        return report

    def process_symbol(
        self,
        symbol: str,
        save: bool = True
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Process a single symbol through the full quality pipeline.

        Steps:
        1. Load raw data
        2. Filter to trading hours
        3. Fix OHLC relationships
        4. Standardize price precision
        5. Add volume statistics and event flags
        6. Save processed data

        Returns processed DataFrame and quality report.
        """
        logger.info(f"Processing {symbol}...")

        # Load raw data
        df = self.load_raw_data(symbol)

        # Step 1: Filter trading hours
        df = self.hours_analyzer.filter_trading_hours(df, assume_timezone="America/New_York")

        # Step 2: Fix OHLC relationships
        violations_before = self.ohlc_validator.validate_ohlc_relationships(df)
        df = self.ohlc_validator.fix_ohlc_relationships(df)
        violations_after = self.ohlc_validator.validate_ohlc_relationships(df)

        # Step 3: Standardize precision
        df = self.ohlc_validator.standardize_price_precision(df)

        # Step 4: Add volume statistics
        df = self.volume_detector.calculate_volume_statistics(df)
        df = self.volume_detector.detect_volume_spikes(df)

        # Step 5: Add event flags
        df = self.volume_detector.add_event_flags(df)

        # Generate report
        report = self.analyze_symbol(symbol)
        report.ohlc_violations_after = violations_after['any_violation'].sum()

        # Save processed data
        if save:
            output_dir = Path(self.config.processed_data_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{symbol}_15min_clean.csv"
            df.to_csv(output_path)
            logger.info(f"Saved processed data to {output_path}")

        return df, report

    def process_all_symbols(
        self,
        symbols: List[str] = None,
        save: bool = True
    ) -> Dict[str, DataQualityReport]:
        """
        Process all symbols through the quality pipeline.
        """
        if symbols is None:
            # Get all symbols from raw data directory
            raw_dir = Path(self.config.raw_data_dir)
            symbols = [f.stem.replace('_15min', '') for f in raw_dir.glob('*_15min.csv')]

        reports = {}
        for symbol in symbols:
            try:
                _, report = self.process_symbol(symbol, save=save)
                reports[symbol] = report
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")

        return reports

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate aggregate summary across all symbols.
        """
        if not self.reports:
            return {}

        summary = {
            'total_symbols': len(self.reports),
            'total_rows_raw': sum(r.total_rows_raw for r in self.reports.values()),
            'total_rows_filtered': sum(r.total_rows_filtered for r in self.reports.values()),
            'rows_removed_pct': 1 - sum(r.total_rows_filtered for r in self.reports.values()) /
                                max(sum(r.total_rows_raw for r in self.reports.values()), 1),
            'avg_quality_score': np.mean([r.quality_score for r in self.reports.values()]),
            'symbols_with_issues': [s for s, r in self.reports.items() if r.issues],
            'total_ohlc_violations_fixed': sum(
                r.ohlc_violations_before - r.ohlc_violations_after
                for r in self.reports.values()
            ),
            'total_volume_spikes': sum(r.volume_spikes_detected for r in self.reports.values()),
        }

        return summary


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Data Quality Pipeline for AlphaTrade System"
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze data quality without modifying'
    )
    parser.add_argument(
        '--process',
        action='store_true',
        help='Process and save clean data'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate processed data'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Process single symbol (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_quality_report.json',
        help='Output report path'
    )

    args = parser.parse_args()

    # Initialize pipeline
    config = DataQualityConfig()
    pipeline = DataQualityPipeline(config)

    if args.analyze or (not args.process and not args.validate):
        # Default to analyze mode
        logger.info("=" * 60)
        logger.info("DATA QUALITY ANALYSIS")
        logger.info("=" * 60)

        if args.symbol:
            report = pipeline.analyze_symbol(args.symbol)
            print(f"\n{args.symbol} Report:")
            print(f"  Raw rows: {report.total_rows_raw}")
            print(f"  Regular hours rows: {report.regular_hours_rows}")
            print(f"  Pre-market rows: {report.pre_market_rows}")
            print(f"  After-hours rows: {report.after_hours_rows}")
            print(f"  Trading days: {report.trading_days}")
            print(f"  Bars/day avg: {report.bars_per_day_avg:.1f}")
            print(f"  Volume spikes: {report.volume_spikes_detected}")
            print(f"  OHLC violations: {report.ohlc_violations_before}")
            print(f"  Quality score: {report.quality_score:.1f}")
            print(f"  Issues: {report.issues}")
        else:
            # Analyze all symbols
            raw_dir = Path(config.raw_data_dir)
            symbols = [f.stem.replace('_15min', '') for f in raw_dir.glob('*_15min.csv')]

            for symbol in symbols:
                try:
                    report = pipeline.analyze_symbol(symbol)
                    status = "OK" if report.quality_score >= 80 else "WARNING" if report.quality_score >= 60 else "CRITICAL"
                    print(f"{symbol}: Score={report.quality_score:.1f} [{status}] - {len(report.issues)} issues")
                except Exception as e:
                    print(f"{symbol}: ERROR - {e}")

            # Summary
            summary = pipeline.generate_summary_report()
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Symbols analyzed: {summary['total_symbols']}")
            print(f"Total raw rows: {summary['total_rows_raw']:,}")
            print(f"Rows to remove: {summary['rows_removed_pct']*100:.1f}%")
            print(f"Average quality score: {summary['avg_quality_score']:.1f}")

        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(config),
            'reports': {s: asdict(r) for s, r in pipeline.reports.items()},
            'summary': pipeline.generate_summary_report()
        }

        with open(args.output, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nReport saved to {args.output}")

    elif args.process:
        logger.info("=" * 60)
        logger.info("DATA PROCESSING")
        logger.info("=" * 60)

        if args.symbol:
            df, report = pipeline.process_symbol(args.symbol, save=True)
            print(f"Processed {args.symbol}: {len(df)} rows saved")
        else:
            reports = pipeline.process_all_symbols(save=True)
            print(f"Processed {len(reports)} symbols")

            summary = pipeline.generate_summary_report()
            print(f"Fixed {summary['total_ohlc_violations_fixed']} OHLC violations")

    elif args.validate:
        logger.info("=" * 60)
        logger.info("VALIDATION MODE")
        logger.info("=" * 60)

        # Validate processed data meets quality standards
        processed_dir = Path(config.processed_data_dir)

        if not processed_dir.exists():
            print("ERROR: No processed data found. Run --process first.")
            return

        validation_results = {
            'regular_hours_only': True,
            'ohlc_valid': True,
            'bars_per_day': True,
            'all_passed': True
        }

        for csv_file in processed_dir.glob('*_clean.csv'):
            symbol = csv_file.stem.replace('_15min_clean', '')
            df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')

            # Check 1: Regular hours only
            times = df.index.time
            regular_start = time(9, 30)
            regular_end = time(16, 0)
            outside_hours = ((times < regular_start) | (times > regular_end)).sum()
            if outside_hours > 0:
                validation_results['regular_hours_only'] = False
                print(f"{symbol}: FAIL - {outside_hours} bars outside regular hours")

            # Check 2: OHLC valid
            max_oc = df[['open', 'close']].max(axis=1)
            min_oc = df[['open', 'close']].min(axis=1)
            violations = (df['high'] < max_oc).sum() + (df['low'] > min_oc).sum()
            if violations > 0:
                validation_results['ohlc_valid'] = False
                print(f"{symbol}: FAIL - {violations} OHLC violations")

            # Check 3: Bars per day
            bars_by_day = df.groupby(df.index.date).size()
            expected = config.expected_bars_per_day
            incomplete_days = (bars_by_day < expected * 0.9).sum()
            if incomplete_days > bars_by_day.count() * 0.1:  # Allow 10% incomplete
                validation_results['bars_per_day'] = False
                print(f"{symbol}: WARNING - {incomplete_days} incomplete trading days")

        validation_results['all_passed'] = all([
            validation_results['regular_hours_only'],
            validation_results['ohlc_valid']
        ])

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        for check, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")


if __name__ == "__main__":
    main()
