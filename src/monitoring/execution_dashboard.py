"""
Execution Monitoring Dashboard
==============================
JPMorgan-Level Real-Time Execution Quality Monitoring

This module provides comprehensive execution analytics:
1. Real-time slippage tracking
2. Fill rate monitoring
3. VWAP/TWAP benchmark comparison
4. Execution venue analysis
5. Latency metrics
6. Prometheus metrics export for Grafana

Key Metrics Tracked:
- Slippage (bps): Actual vs expected fill price
- Fill Rate: Percentage of order filled
- Implementation Shortfall: Total cost of execution
- Market Impact: Price movement caused by our trading
- Timing Cost: Cost of delayed execution

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 4
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import statistics

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionState:
    """State of an execution"""
    parent_order_id: str
    symbol: str
    side: ExecutionSide
    target_quantity: int
    executed_quantity: int
    target_price: float  # Decision price
    avg_fill_price: float
    vwap_benchmark: float  # Market VWAP during execution
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    child_orders: List[str] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled"""
        if self.target_quantity == 0:
            return 0.0
        return self.executed_quantity / self.target_quantity

    @property
    def slippage_bps(self) -> float:
        """Slippage in basis points vs decision price"""
        if self.target_price == 0:
            return 0.0
        diff = self.avg_fill_price - self.target_price
        if self.side == ExecutionSide.SELL:
            diff = -diff
        return (diff / self.target_price) * 10000

    @property
    def vwap_slippage_bps(self) -> float:
        """Slippage vs VWAP benchmark"""
        if self.vwap_benchmark == 0:
            return 0.0
        diff = self.avg_fill_price - self.vwap_benchmark
        if self.side == ExecutionSide.SELL:
            diff = -diff
        return (diff / self.vwap_benchmark) * 10000

    @property
    def execution_time_seconds(self) -> float:
        """Time taken to execute"""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parent_order_id': self.parent_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'target_quantity': self.target_quantity,
            'executed_quantity': self.executed_quantity,
            'target_price': self.target_price,
            'avg_fill_price': self.avg_fill_price,
            'vwap_benchmark': self.vwap_benchmark,
            'fill_rate': self.fill_rate,
            'slippage_bps': self.slippage_bps,
            'vwap_slippage_bps': self.vwap_slippage_bps,
            'execution_time_seconds': self.execution_time_seconds,
            'status': self.status.value,
            'n_child_orders': len(self.child_orders),
            'n_fills': len(self.fills)
        }


@dataclass
class ExecutionQualityReport:
    """Aggregated execution quality metrics"""
    period_start: datetime
    period_end: datetime
    total_executions: int
    completed_executions: int
    partial_executions: int
    failed_executions: int

    # Slippage metrics
    avg_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    max_slippage_bps: float

    # VWAP performance
    avg_vwap_slippage_bps: float
    pct_beat_vwap: float

    # Fill metrics
    avg_fill_rate: float
    avg_execution_time_seconds: float

    # By symbol breakdown
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Cost in dollars
    total_slippage_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat()
            },
            'counts': {
                'total': self.total_executions,
                'completed': self.completed_executions,
                'partial': self.partial_executions,
                'failed': self.failed_executions
            },
            'slippage': {
                'avg_bps': self.avg_slippage_bps,
                'median_bps': self.median_slippage_bps,
                'p95_bps': self.p95_slippage_bps,
                'max_bps': self.max_slippage_bps,
                'total_cost_usd': self.total_slippage_cost
            },
            'vwap': {
                'avg_slippage_bps': self.avg_vwap_slippage_bps,
                'pct_beat': self.pct_beat_vwap
            },
            'fill': {
                'avg_rate': self.avg_fill_rate,
                'avg_time_seconds': self.avg_execution_time_seconds
            },
            'by_symbol': self.by_symbol
        }


@dataclass
class MetricSample:
    """Single metric sample"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class ExecutionMetricsCollector:
    """
    Collects and exports execution metrics.

    Supports:
    - In-memory storage
    - Prometheus format export
    - JSON export for Grafana
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        export_interval_seconds: float = 10.0
    ):
        self.buffer_size = buffer_size
        self.export_interval = export_interval_seconds

        # Metric buffers
        self._metrics: Deque[MetricSample] = deque(maxlen=buffer_size)
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

        # Callbacks for metric export
        self._exporters: List[Callable[[List[MetricSample]], None]] = []

        # Background export task
        self._export_task: Optional[asyncio.Task] = None

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric sample"""
        sample = MetricSample(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self._metrics.append(sample)

        # Update gauge
        label_str = json.dumps(labels or {}, sort_keys=True)
        gauge_key = f"{name}:{label_str}"
        self._gauges[gauge_key] = value

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter"""
        label_str = json.dumps(labels or {}, sort_keys=True)
        key = f"{name}:{label_str}"

        if key not in self._counters:
            self._counters[key] = 0.0

        self._counters[key] += value
        self.record(name, self._counters[key], labels)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram observation"""
        label_str = json.dumps(labels or {}, sort_keys=True)
        key = f"{name}:{label_str}"

        if key not in self._histograms:
            self._histograms[key] = []

        self._histograms[key].append(value)

        # Keep only recent values
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

        self.record(name, value, labels)

    def add_exporter(self, exporter: Callable[[List[MetricSample]], None]) -> None:
        """Add a metric exporter callback"""
        self._exporters.append(exporter)

    async def start_export_loop(self) -> None:
        """Start background export loop"""
        self._export_task = asyncio.create_task(self._export_loop())

    async def stop_export_loop(self) -> None:
        """Stop background export loop"""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

    async def _export_loop(self) -> None:
        """Background loop for metric export"""
        while True:
            await asyncio.sleep(self.export_interval)
            await self._export_metrics()

    async def _export_metrics(self) -> None:
        """Export metrics to all exporters"""
        samples = list(self._metrics)

        for exporter in self._exporters:
            try:
                if asyncio.iscoroutinefunction(exporter):
                    await exporter(samples)
                else:
                    exporter(samples)
            except Exception as e:
                logger.error(f"Metric export failed: {e}")

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format"""
        lines = []

        # Counters
        for key, value in self._counters.items():
            name, labels_json = key.split(':', 1)
            labels = json.loads(labels_json)
            label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())

            if label_str:
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Gauges
        for key, value in self._gauges.items():
            name, labels_json = key.split(':', 1)
            labels = json.loads(labels_json)
            label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())

            if label_str:
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        return '\n'.join(lines)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        label_str = json.dumps(labels or {}, sort_keys=True)
        key = f"{name}:{label_str}"

        values = self._histograms.get(key, [])

        if not values:
            return {
                'count': 0,
                'mean': 0,
                'median': 0,
                'p95': 0,
                'max': 0
            }

        sorted_values = sorted(values)
        p95_idx = int(len(sorted_values) * 0.95)

        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': sorted_values[p95_idx] if p95_idx < len(sorted_values) else sorted_values[-1],
            'max': max(values)
        }


class SlippageAnalyzer:
    """
    Analyzes slippage patterns to identify issues.

    Detects:
    - Systematic slippage by time of day
    - Slippage by symbol characteristics
    - Slippage by order size
    - Adverse selection (slippage worse when model is right)
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self._executions: List[ExecutionState] = []

    def add_execution(self, execution: ExecutionState) -> None:
        """Add execution for analysis"""
        self._executions.append(execution)

        # Trim old executions
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        self._executions = [e for e in self._executions if e.start_time > cutoff]

    def analyze_by_time_of_day(self) -> Dict[int, Dict[str, float]]:
        """Analyze slippage by hour of day"""
        by_hour: Dict[int, List[float]] = {}

        for exec in self._executions:
            hour = exec.start_time.hour
            if hour not in by_hour:
                by_hour[hour] = []
            by_hour[hour].append(exec.slippage_bps)

        result = {}
        for hour, slippages in by_hour.items():
            if slippages:
                result[hour] = {
                    'avg_slippage_bps': statistics.mean(slippages),
                    'count': len(slippages)
                }

        return result

    def analyze_by_symbol(self) -> Dict[str, Dict[str, float]]:
        """Analyze slippage by symbol"""
        by_symbol: Dict[str, List[float]] = {}

        for exec in self._executions:
            if exec.symbol not in by_symbol:
                by_symbol[exec.symbol] = []
            by_symbol[exec.symbol].append(exec.slippage_bps)

        result = {}
        for symbol, slippages in by_symbol.items():
            if slippages:
                result[symbol] = {
                    'avg_slippage_bps': statistics.mean(slippages),
                    'median_slippage_bps': statistics.median(slippages),
                    'count': len(slippages),
                    'total_cost_bps': sum(slippages)
                }

        return result

    def analyze_by_size_bucket(
        self,
        buckets: List[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze slippage by order size bucket"""
        if buckets is None:
            buckets = [100, 500, 1000, 5000, 10000]

        by_bucket: Dict[str, List[float]] = {}

        for exec in self._executions:
            bucket_name = self._get_bucket_name(exec.target_quantity, buckets)
            if bucket_name not in by_bucket:
                by_bucket[bucket_name] = []
            by_bucket[bucket_name].append(exec.slippage_bps)

        result = {}
        for bucket, slippages in by_bucket.items():
            if slippages:
                result[bucket] = {
                    'avg_slippage_bps': statistics.mean(slippages),
                    'count': len(slippages)
                }

        return result

    def _get_bucket_name(self, quantity: int, buckets: List[int]) -> str:
        """Get bucket name for quantity"""
        for i, threshold in enumerate(buckets):
            if quantity <= threshold:
                if i == 0:
                    return f"0-{threshold}"
                return f"{buckets[i-1]+1}-{threshold}"
        return f">{buckets[-1]}"

    def detect_adverse_selection(
        self,
        model_predictions: Optional[Dict[str, bool]] = None
    ) -> Dict[str, float]:
        """
        Detect if slippage is worse when model is correct.

        This indicates information leakage or market anticipation.
        """
        if model_predictions is None:
            return {'error': 'No model predictions provided'}

        correct_slippages = []
        incorrect_slippages = []

        for exec in self._executions:
            was_correct = model_predictions.get(exec.parent_order_id)
            if was_correct is None:
                continue

            if was_correct:
                correct_slippages.append(exec.slippage_bps)
            else:
                incorrect_slippages.append(exec.slippage_bps)

        if not correct_slippages or not incorrect_slippages:
            return {'error': 'Insufficient data'}

        avg_correct = statistics.mean(correct_slippages)
        avg_incorrect = statistics.mean(incorrect_slippages)

        # If slippage is worse when model is correct, there's adverse selection
        adverse_selection_indicator = avg_correct - avg_incorrect

        return {
            'avg_slippage_when_correct_bps': avg_correct,
            'avg_slippage_when_incorrect_bps': avg_incorrect,
            'adverse_selection_bps': adverse_selection_indicator,
            'has_adverse_selection': adverse_selection_indicator > 2.0  # 2 bps threshold
        }


class ExecutionMonitor:
    """
    Real-time execution quality monitoring.

    This is the main class that:
    1. Tracks all ongoing executions
    2. Computes real-time metrics
    3. Publishes to metrics collector
    4. Generates alerts on poor execution
    """

    def __init__(
        self,
        metrics_collector: Optional[ExecutionMetricsCollector] = None,
        slippage_alert_threshold_bps: float = 10.0,
        fill_rate_alert_threshold: float = 0.8
    ):
        self.metrics = metrics_collector or ExecutionMetricsCollector()
        self.slippage_alert_threshold = slippage_alert_threshold_bps
        self.fill_rate_alert_threshold = fill_rate_alert_threshold

        # Active executions
        self._active: Dict[str, ExecutionState] = {}

        # Completed executions (for reporting)
        self._completed: Deque[ExecutionState] = deque(maxlen=10000)

        # Slippage analyzer
        self._analyzer = SlippageAnalyzer()

        # Alert callbacks
        self._alert_handlers: List[Callable[[str, Dict], None]] = []

    def start_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        target_quantity: int,
        target_price: float,
        vwap_benchmark: float
    ) -> ExecutionState:
        """
        Start tracking an execution.

        Call this when a parent order is submitted.
        """
        execution = ExecutionState(
            parent_order_id=order_id,
            symbol=symbol,
            side=ExecutionSide(side.lower()),
            target_quantity=target_quantity,
            executed_quantity=0,
            target_price=target_price,
            avg_fill_price=0.0,
            vwap_benchmark=vwap_benchmark,
            start_time=datetime.now(),
            status=ExecutionStatus.IN_PROGRESS
        )

        self._active[order_id] = execution

        # Record start metric
        self.metrics.increment(
            'execution_started_total',
            labels={'symbol': symbol, 'side': side}
        )

        logger.info(f"Started tracking execution {order_id} for {symbol}")

        return execution

    def record_fill(
        self,
        order_id: str,
        fill_quantity: int,
        fill_price: float,
        child_order_id: Optional[str] = None
    ) -> None:
        """
        Record a fill for an execution.

        Call this for each fill received.
        """
        execution = self._active.get(order_id)
        if not execution:
            logger.warning(f"Fill for unknown execution: {order_id}")
            return

        # Update execution state
        old_qty = execution.executed_quantity
        old_notional = old_qty * execution.avg_fill_price

        new_notional = old_notional + fill_quantity * fill_price
        new_qty = old_qty + fill_quantity

        execution.executed_quantity = new_qty
        execution.avg_fill_price = new_notional / new_qty if new_qty > 0 else 0

        execution.fills.append({
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': datetime.now().isoformat(),
            'child_order_id': child_order_id
        })

        if child_order_id and child_order_id not in execution.child_orders:
            execution.child_orders.append(child_order_id)

        # Record metrics
        self.metrics.histogram(
            'fill_price',
            fill_price,
            labels={'symbol': execution.symbol}
        )

        self.metrics.record(
            'execution_fill_rate',
            execution.fill_rate,
            labels={'order_id': order_id, 'symbol': execution.symbol}
        )

        # Check completion
        if execution.fill_rate >= 0.9999:
            self._complete_execution(order_id, ExecutionStatus.COMPLETED)
        else:
            # Check slippage alert
            self._check_slippage_alert(execution)

    def cancel_execution(self, order_id: str) -> None:
        """Mark execution as cancelled"""
        execution = self._active.get(order_id)
        if not execution:
            return

        if execution.fill_rate > 0:
            self._complete_execution(order_id, ExecutionStatus.PARTIAL)
        else:
            self._complete_execution(order_id, ExecutionStatus.CANCELLED)

    def fail_execution(self, order_id: str, reason: str = "") -> None:
        """Mark execution as failed"""
        self._complete_execution(order_id, ExecutionStatus.FAILED)

        # Alert on failure
        self._send_alert('execution_failed', {
            'order_id': order_id,
            'reason': reason
        })

    def _complete_execution(
        self,
        order_id: str,
        status: ExecutionStatus
    ) -> None:
        """Complete an execution"""
        execution = self._active.pop(order_id, None)
        if not execution:
            return

        execution.end_time = datetime.now()
        execution.status = status

        # Add to completed
        self._completed.append(execution)
        self._analyzer.add_execution(execution)

        # Record final metrics
        self.metrics.histogram(
            'execution_slippage_bps',
            execution.slippage_bps,
            labels={'symbol': execution.symbol, 'status': status.value}
        )

        self.metrics.histogram(
            'execution_time_seconds',
            execution.execution_time_seconds,
            labels={'symbol': execution.symbol}
        )

        self.metrics.increment(
            'execution_completed_total',
            labels={'symbol': execution.symbol, 'status': status.value}
        )

        # Log completion
        logger.info(
            f"Execution {order_id} {status.value}: "
            f"fill_rate={execution.fill_rate:.2%}, "
            f"slippage={execution.slippage_bps:.2f}bps, "
            f"time={execution.execution_time_seconds:.1f}s"
        )

        # Check for alerts
        self._check_slippage_alert(execution)
        self._check_fill_rate_alert(execution)

    def _check_slippage_alert(self, execution: ExecutionState) -> None:
        """Check if slippage exceeds threshold"""
        if abs(execution.slippage_bps) > self.slippage_alert_threshold:
            self._send_alert('high_slippage', {
                'order_id': execution.parent_order_id,
                'symbol': execution.symbol,
                'slippage_bps': execution.slippage_bps,
                'threshold': self.slippage_alert_threshold
            })

    def _check_fill_rate_alert(self, execution: ExecutionState) -> None:
        """Check if fill rate is below threshold"""
        if execution.fill_rate < self.fill_rate_alert_threshold:
            self._send_alert('low_fill_rate', {
                'order_id': execution.parent_order_id,
                'symbol': execution.symbol,
                'fill_rate': execution.fill_rate,
                'threshold': self.fill_rate_alert_threshold
            })

    def _send_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Send alert to handlers"""
        logger.warning(f"ALERT [{alert_type}]: {details}")

        for handler in self._alert_handlers:
            try:
                handler(alert_type, details)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(
        self,
        handler: Callable[[str, Dict], None]
    ) -> None:
        """Add an alert handler"""
        self._alert_handlers.append(handler)

    def get_active_executions(self) -> List[ExecutionState]:
        """Get all active executions"""
        return list(self._active.values())

    def get_execution_state(self, order_id: str) -> Optional[ExecutionState]:
        """Get state of a specific execution"""
        return self._active.get(order_id)

    def generate_report(
        self,
        period_hours: int = 24
    ) -> ExecutionQualityReport:
        """Generate execution quality report for period"""
        cutoff = datetime.now() - timedelta(hours=period_hours)

        # Filter executions in period
        executions = [
            e for e in self._completed
            if e.start_time > cutoff
        ]

        if not executions:
            return ExecutionQualityReport(
                period_start=cutoff,
                period_end=datetime.now(),
                total_executions=0,
                completed_executions=0,
                partial_executions=0,
                failed_executions=0,
                avg_slippage_bps=0,
                median_slippage_bps=0,
                p95_slippage_bps=0,
                max_slippage_bps=0,
                avg_vwap_slippage_bps=0,
                pct_beat_vwap=0,
                avg_fill_rate=0,
                avg_execution_time_seconds=0
            )

        # Compute metrics
        slippages = [e.slippage_bps for e in executions]
        vwap_slippages = [e.vwap_slippage_bps for e in executions]
        fill_rates = [e.fill_rate for e in executions]
        exec_times = [e.execution_time_seconds for e in executions]

        sorted_slippages = sorted(slippages)
        p95_idx = int(len(sorted_slippages) * 0.95)

        # By symbol breakdown
        by_symbol: Dict[str, Dict[str, float]] = {}
        for e in executions:
            if e.symbol not in by_symbol:
                by_symbol[e.symbol] = {'slippages': [], 'fill_rates': []}
            by_symbol[e.symbol]['slippages'].append(e.slippage_bps)
            by_symbol[e.symbol]['fill_rates'].append(e.fill_rate)

        symbol_summary = {}
        for symbol, data in by_symbol.items():
            symbol_summary[symbol] = {
                'avg_slippage_bps': statistics.mean(data['slippages']),
                'avg_fill_rate': statistics.mean(data['fill_rates']),
                'count': len(data['slippages'])
            }

        # Estimate slippage cost
        total_cost = sum(
            e.slippage_bps / 10000 * e.executed_quantity * e.avg_fill_price
            for e in executions
        )

        return ExecutionQualityReport(
            period_start=cutoff,
            period_end=datetime.now(),
            total_executions=len(executions),
            completed_executions=len([e for e in executions if e.status == ExecutionStatus.COMPLETED]),
            partial_executions=len([e for e in executions if e.status == ExecutionStatus.PARTIAL]),
            failed_executions=len([e for e in executions if e.status == ExecutionStatus.FAILED]),
            avg_slippage_bps=statistics.mean(slippages),
            median_slippage_bps=statistics.median(slippages),
            p95_slippage_bps=sorted_slippages[p95_idx] if p95_idx < len(sorted_slippages) else max(slippages),
            max_slippage_bps=max(slippages),
            avg_vwap_slippage_bps=statistics.mean(vwap_slippages),
            pct_beat_vwap=len([s for s in vwap_slippages if s < 0]) / len(vwap_slippages),
            avg_fill_rate=statistics.mean(fill_rates),
            avg_execution_time_seconds=statistics.mean(exec_times),
            by_symbol=symbol_summary,
            total_slippage_cost=total_cost
        )

    async def publish_to_grafana(self, execution: ExecutionState) -> None:
        """Publish execution state to Grafana (via metrics)"""
        self.metrics.record(
            'execution_fill_rate_gauge',
            execution.fill_rate,
            labels={
                'order_id': execution.parent_order_id,
                'symbol': execution.symbol
            }
        )

        self.metrics.record(
            'execution_slippage_gauge',
            execution.slippage_bps,
            labels={
                'order_id': execution.parent_order_id,
                'symbol': execution.symbol
            }
        )

        self.metrics.record(
            'execution_elapsed_time_gauge',
            execution.execution_time_seconds,
            labels={
                'order_id': execution.parent_order_id,
                'symbol': execution.symbol
            }
        )

    def get_slippage_analysis(self) -> Dict[str, Any]:
        """Get comprehensive slippage analysis"""
        return {
            'by_time_of_day': self._analyzer.analyze_by_time_of_day(),
            'by_symbol': self._analyzer.analyze_by_symbol(),
            'by_size_bucket': self._analyzer.analyze_by_size_bucket()
        }


# =============================================================================
# PROMETHEUS EXPORTER
# =============================================================================

class PrometheusExporter:
    """
    Export metrics in Prometheus format.

    Can be used with:
    - Prometheus pull (HTTP endpoint)
    - Prometheus push gateway
    """

    def __init__(
        self,
        metrics_collector: ExecutionMetricsCollector,
        job_name: str = "alphatrade_execution"
    ):
        self.metrics = metrics_collector
        self.job_name = job_name

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return self.metrics.to_prometheus_format()

    async def push_to_gateway(
        self,
        gateway_url: str
    ) -> bool:
        """Push metrics to Prometheus push gateway"""
        try:
            import aiohttp

            metrics_text = self.get_metrics_text()
            url = f"{gateway_url}/metrics/job/{self.job_name}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=metrics_text,
                    headers={'Content-Type': 'text/plain'}
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to push to Prometheus gateway: {e}")
            return False


# =============================================================================
# HTTP SERVER FOR DASHBOARD
# =============================================================================

class ExecutionDashboardServer:
    """
    HTTP server for execution monitoring dashboard.

    Endpoints:
    - GET /executions - List active executions
    - GET /executions/{id} - Get execution details
    - GET /report - Get execution quality report
    - GET /metrics - Prometheus metrics endpoint
    - GET /analysis - Slippage analysis
    """

    def __init__(
        self,
        monitor: ExecutionMonitor,
        host: str = "0.0.0.0",
        port: int = 8081
    ):
        self.monitor = monitor
        self.host = host
        self.port = port

        try:
            from aiohttp import web
            self._web = web
            self._app = web.Application()
            self._runner: Optional[web.AppRunner] = None

            # Setup routes
            self._app.router.add_get('/executions', self._handle_list_executions)
            self._app.router.add_get('/executions/{order_id}', self._handle_get_execution)
            self._app.router.add_get('/report', self._handle_report)
            self._app.router.add_get('/metrics', self._handle_metrics)
            self._app.router.add_get('/analysis', self._handle_analysis)
        except ImportError:
            logger.warning("aiohttp not installed, dashboard server disabled")
            self._web = None

    async def start(self) -> None:
        """Start dashboard server"""
        if self._web is None:
            return

        self._runner = self._web.AppRunner(self._app)
        await self._runner.setup()

        site = self._web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        logger.info(f"Execution dashboard started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop dashboard server"""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Execution dashboard stopped")

    async def _handle_list_executions(self, request) -> 'web.Response':
        """Handle GET /executions"""
        executions = self.monitor.get_active_executions()
        return self._web.json_response([e.to_dict() for e in executions])

    async def _handle_get_execution(self, request) -> 'web.Response':
        """Handle GET /executions/{order_id}"""
        order_id = request.match_info['order_id']
        execution = self.monitor.get_execution_state(order_id)

        if execution:
            return self._web.json_response(execution.to_dict())
        else:
            return self._web.json_response(
                {'error': 'Execution not found'},
                status=404
            )

    async def _handle_report(self, request) -> 'web.Response':
        """Handle GET /report"""
        hours = int(request.query.get('hours', 24))
        report = self.monitor.generate_report(period_hours=hours)
        return self._web.json_response(report.to_dict())

    async def _handle_metrics(self, request) -> 'web.Response':
        """Handle GET /metrics (Prometheus format)"""
        metrics_text = self.monitor.metrics.to_prometheus_format()
        return self._web.Response(
            text=metrics_text,
            content_type='text/plain'
        )

    async def _handle_analysis(self, request) -> 'web.Response':
        """Handle GET /analysis"""
        analysis = self.monitor.get_slippage_analysis()
        return self._web.json_response(analysis)
