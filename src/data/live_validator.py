"""
Live Data Validation Gate
=========================
JPMorgan-Level Data Quality Assurance for Live Trading

This module validates incoming live data before feeding to the model.
Catches data anomalies that could cause:
1. Model mispredictions (garbage in, garbage out)
2. Position sizing errors
3. Risk calculation failures

Key Validations:
1. Price sanity (no unrealistic moves)
2. Volume sanity (no impossible spikes)
3. OHLC integrity (high >= low, etc.)
4. Timestamp validation (no stale data)
5. Missing value detection

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - MISSING-3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"  # No issues
    GOOD = "good"  # Minor warnings, safe to use
    SUSPICIOUS = "suspicious"  # Some anomalies, use with caution
    POOR = "poor"  # Significant issues, may affect signals
    INVALID = "invalid"  # Do not use for trading


class ValidationCategory(Enum):
    """Categories of validation checks"""
    PRICE = "price"
    VOLUME = "volume"
    OHLC = "ohlc"
    TIMESTAMP = "timestamp"
    MISSING = "missing"
    SEQUENCE = "sequence"
    SPREAD = "spread"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    passed: bool
    category: ValidationCategory
    message: str
    severity: int  # 1-5, 5 being most severe
    value: Optional[Any] = None
    threshold: Optional[Any] = None


@dataclass
class DataQualityReport:
    """Complete data quality report for a bar or batch"""
    timestamp: datetime
    symbol: str
    quality_level: DataQualityLevel
    quality_score: float  # 0-100
    validations_passed: int
    validations_failed: int
    warnings: List[ValidationResult]
    errors: List[ValidationResult]
    data_usable: bool  # Final verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'quality_level': self.quality_level.value,
            'quality_score': self.quality_score,
            'validations_passed': self.validations_passed,
            'validations_failed': self.validations_failed,
            'warnings': [w.message for w in self.warnings],
            'errors': [e.message for e in self.errors],
            'data_usable': self.data_usable
        }


@dataclass
class HistoricalStats:
    """Historical statistics for a symbol"""
    avg_price: float = 0.0
    price_std: float = 0.0
    avg_volume: float = 0.0
    volume_std: float = 0.0
    avg_spread: float = 0.0
    avg_high_low_range: float = 0.0
    last_price: float = 0.0
    last_volume: float = 0.0
    last_timestamp: Optional[datetime] = None
    sample_count: int = 0


class LiveDataValidator:
    """
    Validates incoming live data before processing.

    This is a critical gate that prevents bad data from affecting:
    - Feature calculations
    - Model predictions
    - Position sizing
    - Risk metrics

    All data must pass validation before being used.
    """

    def __init__(
        self,
        # Price validation
        max_price_change_pct: float = 0.20,  # 20% max single-bar move
        max_price_z_score: float = 5.0,  # 5 std devs
        min_price: float = 0.01,  # Minimum valid price

        # Volume validation
        max_volume_spike: float = 50.0,  # 50x average volume
        max_volume_z_score: float = 10.0,  # 10 std devs
        min_volume: int = 0,  # Minimum volume (can be 0 for illiquid)

        # Timestamp validation
        max_staleness_seconds: int = 300,  # 5 minutes max stale
        min_tick_interval_ms: int = 100,  # Minimum between ticks

        # Historical window
        historical_window: int = 1000,  # Bars to keep for statistics
    ):
        self.thresholds = {
            'max_price_change_pct': max_price_change_pct,
            'max_price_z_score': max_price_z_score,
            'min_price': min_price,
            'max_volume_spike': max_volume_spike,
            'max_volume_z_score': max_volume_z_score,
            'min_volume': min_volume,
            'max_staleness_seconds': max_staleness_seconds,
            'min_tick_interval_ms': min_tick_interval_ms,
        }

        self.historical_window = historical_window

        # Per-symbol historical data
        self._symbol_history: Dict[str, deque] = {}
        self._symbol_stats: Dict[str, HistoricalStats] = {}

        # Validation statistics
        self._stats = {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'by_category': {cat.value: {'passed': 0, 'failed': 0}
                          for cat in ValidationCategory}
        }

    def validate(
        self,
        symbol: str,
        bar: Dict[str, Any]
    ) -> DataQualityReport:
        """
        Validate a single bar of data.

        Args:
            symbol: Trading symbol
            bar: Dictionary with keys: open, high, low, close, volume, timestamp

        Returns:
            DataQualityReport with validation results
        """
        results: List[ValidationResult] = []
        timestamp = bar.get('timestamp', datetime.now())

        # Get or create historical stats
        stats = self._get_or_create_stats(symbol)

        # Run all validations
        results.extend(self._validate_price(bar, stats))
        results.extend(self._validate_volume(bar, stats))
        results.extend(self._validate_ohlc(bar))
        results.extend(self._validate_timestamp(bar, stats))
        results.extend(self._validate_missing(bar))
        results.extend(self._validate_sequence(bar, stats))

        # Categorize results
        warnings = [r for r in results if not r.passed and r.severity <= 3]
        errors = [r for r in results if not r.passed and r.severity > 3]
        passed = [r for r in results if r.passed]

        # Calculate quality score
        total_checks = len(results)
        if total_checks == 0:
            quality_score = 100.0
        else:
            # Weight errors more heavily
            penalty = sum(r.severity * 5 for r in warnings)
            penalty += sum(r.severity * 15 for r in errors)
            quality_score = max(0, 100 - penalty)

        # Determine quality level
        quality_level = self._determine_quality_level(quality_score, errors)

        # Determine if data is usable
        data_usable = quality_level not in [DataQualityLevel.INVALID]

        # Update statistics
        self._update_stats(results)

        # Update historical stats if data is good enough
        if quality_score >= 50:
            self._update_historical_stats(symbol, bar)

        report = DataQualityReport(
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
            symbol=symbol,
            quality_level=quality_level,
            quality_score=quality_score,
            validations_passed=len(passed),
            validations_failed=len(warnings) + len(errors),
            warnings=warnings,
            errors=errors,
            data_usable=data_usable
        )

        # Log issues
        if errors:
            logger.warning(
                f"Data quality ERROR for {symbol}: {[e.message for e in errors]}"
            )
        elif warnings:
            logger.debug(
                f"Data quality warning for {symbol}: {[w.message for w in warnings]}"
            )

        return report

    def validate_batch(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, DataQualityReport]:
        """
        Validate a batch of data for multiple symbols.

        Args:
            data: Dictionary of symbol -> DataFrame

        Returns:
            Dictionary of symbol -> DataQualityReport (for most recent bar)
        """
        reports = {}

        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue

            # Validate the most recent bar
            latest = df.iloc[-1]
            bar = {
                'open': latest.get('open', latest.get('Open', 0)),
                'high': latest.get('high', latest.get('High', 0)),
                'low': latest.get('low', latest.get('Low', 0)),
                'close': latest.get('close', latest.get('Close', 0)),
                'volume': latest.get('volume', latest.get('Volume', 0)),
                'timestamp': latest.name if isinstance(latest.name, datetime) else datetime.now()
            }

            reports[symbol] = self.validate(symbol, bar)

            # Also update history with full dataframe
            self._update_history_from_df(symbol, df)

        return reports

    def _validate_price(
        self,
        bar: Dict,
        stats: HistoricalStats
    ) -> List[ValidationResult]:
        """Validate price data"""
        results = []
        close = bar.get('close', 0)
        high = bar.get('high', 0)
        low = bar.get('low', 0)

        # Minimum price check
        if close < self.thresholds['min_price']:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.PRICE,
                message=f"Price {close} below minimum {self.thresholds['min_price']}",
                severity=5,
                value=close,
                threshold=self.thresholds['min_price']
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                category=ValidationCategory.PRICE,
                message="Price above minimum",
                severity=0
            ))

        # Price change check (if we have history)
        if stats.last_price > 0:
            price_change = abs(close - stats.last_price) / stats.last_price

            if price_change > self.thresholds['max_price_change_pct']:
                results.append(ValidationResult(
                    passed=False,
                    category=ValidationCategory.PRICE,
                    message=f"Abnormal price move: {price_change:.1%} "
                            f"(max: {self.thresholds['max_price_change_pct']:.1%})",
                    severity=4,
                    value=price_change,
                    threshold=self.thresholds['max_price_change_pct']
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    category=ValidationCategory.PRICE,
                    message="Price change within normal range",
                    severity=0
                ))

        # Z-score check (if we have enough history)
        if stats.sample_count >= 20 and stats.price_std > 0:
            z_score = abs(close - stats.avg_price) / stats.price_std

            if z_score > self.thresholds['max_price_z_score']:
                results.append(ValidationResult(
                    passed=False,
                    category=ValidationCategory.PRICE,
                    message=f"Price z-score {z_score:.1f} exceeds threshold "
                            f"{self.thresholds['max_price_z_score']}",
                    severity=3,
                    value=z_score,
                    threshold=self.thresholds['max_price_z_score']
                ))

        # Negative price check
        if close < 0 or high < 0 or low < 0:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.PRICE,
                message="Negative price detected",
                severity=5,
                value=min(close, high, low)
            ))

        return results

    def _validate_volume(
        self,
        bar: Dict,
        stats: HistoricalStats
    ) -> List[ValidationResult]:
        """Validate volume data"""
        results = []
        volume = bar.get('volume', 0)

        # Negative volume check
        if volume < 0:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.VOLUME,
                message="Negative volume detected",
                severity=5,
                value=volume
            ))
            return results

        # Volume spike check
        if stats.avg_volume > 0:
            volume_ratio = volume / stats.avg_volume

            if volume_ratio > self.thresholds['max_volume_spike']:
                results.append(ValidationResult(
                    passed=False,
                    category=ValidationCategory.VOLUME,
                    message=f"Volume spike: {volume_ratio:.1f}x average "
                            f"(max: {self.thresholds['max_volume_spike']}x)",
                    severity=2,  # Warning, not error
                    value=volume_ratio,
                    threshold=self.thresholds['max_volume_spike']
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    category=ValidationCategory.VOLUME,
                    message="Volume within normal range",
                    severity=0
                ))

        # Z-score check
        if stats.sample_count >= 20 and stats.volume_std > 0:
            z_score = abs(volume - stats.avg_volume) / stats.volume_std

            if z_score > self.thresholds['max_volume_z_score']:
                results.append(ValidationResult(
                    passed=False,
                    category=ValidationCategory.VOLUME,
                    message=f"Volume z-score {z_score:.1f} exceeds threshold",
                    severity=2,
                    value=z_score,
                    threshold=self.thresholds['max_volume_z_score']
                ))

        return results

    def _validate_ohlc(self, bar: Dict) -> List[ValidationResult]:
        """Validate OHLC integrity"""
        results = []
        o = bar.get('open', 0)
        h = bar.get('high', 0)
        l = bar.get('low', 0)
        c = bar.get('close', 0)

        # High must be highest
        if h < max(o, c, l):
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.OHLC,
                message=f"Invalid OHLC: high ({h}) is not the highest value",
                severity=4,
                value={'o': o, 'h': h, 'l': l, 'c': c}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                category=ValidationCategory.OHLC,
                message="High is valid",
                severity=0
            ))

        # Low must be lowest
        if l > min(o, c, h):
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.OHLC,
                message=f"Invalid OHLC: low ({l}) is not the lowest value",
                severity=4,
                value={'o': o, 'h': h, 'l': l, 'c': c}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                category=ValidationCategory.OHLC,
                message="Low is valid",
                severity=0
            ))

        # Open must be between high and low
        if not (l <= o <= h):
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.OHLC,
                message="Invalid OHLC: open outside high-low range",
                severity=4,
                value={'o': o, 'h': h, 'l': l}
            ))

        # Close must be between high and low
        if not (l <= c <= h):
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.OHLC,
                message="Invalid OHLC: close outside high-low range",
                severity=4,
                value={'c': c, 'h': h, 'l': l}
            ))

        # Check for zero values
        if 0 in [o, h, l, c]:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.OHLC,
                message="Zero value in OHLC data",
                severity=3,
                value={'o': o, 'h': h, 'l': l, 'c': c}
            ))

        return results

    def _validate_timestamp(
        self,
        bar: Dict,
        stats: HistoricalStats
    ) -> List[ValidationResult]:
        """Validate timestamp"""
        results = []
        timestamp = bar.get('timestamp')

        if timestamp is None:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.TIMESTAMP,
                message="Missing timestamp",
                severity=3
            ))
            return results

        # Convert to datetime if needed
        if not isinstance(timestamp, datetime):
            try:
                timestamp = pd.to_datetime(timestamp)
            except Exception:
                results.append(ValidationResult(
                    passed=False,
                    category=ValidationCategory.TIMESTAMP,
                    message="Invalid timestamp format",
                    severity=4
                ))
                return results

        # Future timestamp check
        now = datetime.now()
        if timestamp > now + timedelta(minutes=5):
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.TIMESTAMP,
                message="Timestamp is in the future",
                severity=4,
                value=timestamp
            ))

        # Staleness check
        staleness = (now - timestamp).total_seconds()
        if staleness > self.thresholds['max_staleness_seconds']:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.TIMESTAMP,
                message=f"Data is stale: {staleness:.0f}s old "
                        f"(max: {self.thresholds['max_staleness_seconds']}s)",
                severity=3,
                value=staleness,
                threshold=self.thresholds['max_staleness_seconds']
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                category=ValidationCategory.TIMESTAMP,
                message="Timestamp is fresh",
                severity=0
            ))

        # Out-of-sequence check
        if stats.last_timestamp and timestamp < stats.last_timestamp:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.TIMESTAMP,
                message="Timestamp is out of sequence (before last)",
                severity=3,
                value=timestamp
            ))

        return results

    def _validate_missing(self, bar: Dict) -> List[ValidationResult]:
        """Check for missing required fields"""
        results = []
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        missing = []

        for field in required_fields:
            value = bar.get(field)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing.append(field)

        if missing:
            results.append(ValidationResult(
                passed=False,
                category=ValidationCategory.MISSING,
                message=f"Missing fields: {missing}",
                severity=4,
                value=missing
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                category=ValidationCategory.MISSING,
                message="All required fields present",
                severity=0
            ))

        return results

    def _validate_sequence(
        self,
        bar: Dict,
        stats: HistoricalStats
    ) -> List[ValidationResult]:
        """Validate data sequence consistency"""
        results = []
        close = bar.get('close', 0)

        # Gap detection (if we have last price)
        if stats.last_price > 0:
            gap_pct = abs(bar.get('open', close) - stats.last_price) / stats.last_price

            # Large gap (might be valid, but worth noting)
            if gap_pct > 0.05:  # 5% gap
                results.append(ValidationResult(
                    passed=True,  # Not an error, just a note
                    category=ValidationCategory.SEQUENCE,
                    message=f"Large gap detected: {gap_pct:.1%}",
                    severity=1,
                    value=gap_pct
                ))

        return results

    def _determine_quality_level(
        self,
        score: float,
        errors: List[ValidationResult]
    ) -> DataQualityLevel:
        """Determine overall quality level"""
        # Any critical errors = invalid
        critical_errors = [e for e in errors if e.severity >= 5]
        if critical_errors:
            return DataQualityLevel.INVALID

        if score >= 90:
            return DataQualityLevel.EXCELLENT
        elif score >= 70:
            return DataQualityLevel.GOOD
        elif score >= 50:
            return DataQualityLevel.SUSPICIOUS
        elif score >= 30:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.INVALID

    def _get_or_create_stats(self, symbol: str) -> HistoricalStats:
        """Get or create historical stats for symbol"""
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = HistoricalStats()
            self._symbol_history[symbol] = deque(maxlen=self.historical_window)
        return self._symbol_stats[symbol]

    def _update_historical_stats(self, symbol: str, bar: Dict) -> None:
        """Update historical statistics with new bar"""
        stats = self._symbol_stats[symbol]
        history = self._symbol_history[symbol]

        close = bar.get('close', 0)
        volume = bar.get('volume', 0)
        timestamp = bar.get('timestamp')

        # Add to history
        history.append({
            'close': close,
            'volume': volume,
            'high': bar.get('high', close),
            'low': bar.get('low', close),
            'timestamp': timestamp
        })

        # Update stats
        prices = [h['close'] for h in history]
        volumes = [h['volume'] for h in history]

        stats.avg_price = np.mean(prices)
        stats.price_std = np.std(prices) if len(prices) > 1 else 0
        stats.avg_volume = np.mean(volumes)
        stats.volume_std = np.std(volumes) if len(volumes) > 1 else 0
        stats.last_price = close
        stats.last_volume = volume
        stats.last_timestamp = timestamp if isinstance(timestamp, datetime) else None
        stats.sample_count = len(history)

        # Calculate average high-low range
        hl_ranges = [h['high'] - h['low'] for h in history]
        stats.avg_high_low_range = np.mean(hl_ranges) if hl_ranges else 0

    def _update_history_from_df(self, symbol: str, df: pd.DataFrame) -> None:
        """Update history from a full DataFrame"""
        if symbol not in self._symbol_history:
            self._symbol_history[symbol] = deque(maxlen=self.historical_window)
            self._symbol_stats[symbol] = HistoricalStats()

        # Take last N rows
        recent = df.tail(self.historical_window)

        # Clear and repopulate
        self._symbol_history[symbol].clear()

        for idx, row in recent.iterrows():
            bar = {
                'close': row.get('close', row.get('Close', 0)),
                'volume': row.get('volume', row.get('Volume', 0)),
                'high': row.get('high', row.get('High', 0)),
                'low': row.get('low', row.get('Low', 0)),
                'timestamp': idx if isinstance(idx, datetime) else None
            }
            self._symbol_history[symbol].append(bar)

        # Update stats
        if len(self._symbol_history[symbol]) > 0:
            prices = [h['close'] for h in self._symbol_history[symbol]]
            volumes = [h['volume'] for h in self._symbol_history[symbol]]

            stats = self._symbol_stats[symbol]
            stats.avg_price = np.mean(prices)
            stats.price_std = np.std(prices) if len(prices) > 1 else 0
            stats.avg_volume = np.mean(volumes)
            stats.volume_std = np.std(volumes) if len(volumes) > 1 else 0
            stats.last_price = prices[-1]
            stats.last_volume = volumes[-1]
            stats.sample_count = len(prices)

    def _update_stats(self, results: List[ValidationResult]) -> None:
        """Update validation statistics"""
        self._stats['total_validations'] += 1

        for r in results:
            cat = r.category.value
            if r.passed:
                self._stats['passed'] += 1
                self._stats['by_category'][cat]['passed'] += 1
            else:
                if r.severity <= 3:
                    self._stats['warnings'] += 1
                else:
                    self._stats['failed'] += 1
                self._stats['by_category'][cat]['failed'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            **self._stats,
            'symbols_tracked': len(self._symbol_stats),
            'pass_rate': self._stats['passed'] / max(1, self._stats['total_validations'])
        }

    def get_symbol_stats(self, symbol: str) -> Optional[Dict]:
        """Get statistics for a specific symbol"""
        stats = self._symbol_stats.get(symbol)
        if stats:
            return {
                'avg_price': stats.avg_price,
                'price_std': stats.price_std,
                'avg_volume': stats.avg_volume,
                'volume_std': stats.volume_std,
                'last_price': stats.last_price,
                'sample_count': stats.sample_count
            }
        return None

    def reset_symbol(self, symbol: str) -> None:
        """Reset statistics for a symbol"""
        if symbol in self._symbol_stats:
            del self._symbol_stats[symbol]
        if symbol in self._symbol_history:
            del self._symbol_history[symbol]

    def reset_all(self) -> None:
        """Reset all statistics"""
        self._symbol_stats.clear()
        self._symbol_history.clear()
        self._stats = {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'by_category': {cat.value: {'passed': 0, 'failed': 0}
                          for cat in ValidationCategory}
        }
