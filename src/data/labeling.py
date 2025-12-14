"""
Institutional-Grade Data Labeling Module
Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado

Features:
- Triple Barrier Method for path-dependent labeling
- Meta-Labeling for secondary model training
- Dynamic volatility-based barrier widths
- Multiprocessing support for large datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BarrierType(Enum):
    """Types of barrier touches"""
    UPPER = "upper"      # Profit take barrier
    LOWER = "lower"      # Stop loss barrier
    VERTICAL = "vertical"  # Time expiration


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier Method"""
    # Barrier multipliers (applied to volatility target)
    pt_sl_ratio: Tuple[float, float] = (1.0, 1.0)  # (profit_take, stop_loss) multipliers

    # Volatility estimation
    volatility_lookback: int = 20  # Periods for volatility estimation
    volatility_method: str = "ewm"  # "ewm", "rolling", or "garman_klass"

    # Vertical barrier
    max_holding_period: int = 10  # Maximum bars to hold position

    # Minimum return threshold
    min_return: float = 0.0  # Minimum return to consider

    # Multiprocessing
    n_jobs: int = 1  # Number of parallel jobs

    # Side prediction
    use_side: bool = False  # Whether to use predetermined side


@dataclass
class MetaLabelingConfig:
    """Configuration for Meta-Labeling"""
    # Primary model threshold
    primary_threshold: float = 0.5  # Threshold for primary model signal

    # Whether to use probability or binary prediction
    use_probability: bool = True

    # Minimum confidence for primary signal
    min_confidence: float = 0.0


class VolatilityEstimator:
    """
    Volatility estimation for dynamic barrier widths.

    Multiple methods supported:
    - EWM (Exponentially Weighted Moving) standard deviation
    - Rolling standard deviation
    - Garman-Klass volatility (uses OHLC)
    - Parkinson volatility (uses High-Low)
    """

    def __init__(
        self,
        method: str = "ewm",
        lookback: int = 20,
        min_periods: int = 5
    ):
        """
        Initialize VolatilityEstimator.

        Args:
            method: "ewm", "rolling", "garman_klass", or "parkinson"
            lookback: Number of periods for estimation
            min_periods: Minimum periods required
        """
        self.method = method
        self.lookback = lookback
        self.min_periods = min_periods

    def estimate(
        self,
        prices: pd.DataFrame,
        column: str = "close"
    ) -> pd.Series:
        """
        Estimate volatility.

        Args:
            prices: DataFrame with price data (OHLC optional)
            column: Price column to use for returns-based methods

        Returns:
            Series of volatility estimates
        """
        if self.method == "ewm":
            return self._ewm_volatility(prices[column])
        elif self.method == "rolling":
            return self._rolling_volatility(prices[column])
        elif self.method == "garman_klass":
            return self._garman_klass_volatility(prices)
        elif self.method == "parkinson":
            return self._parkinson_volatility(prices)
        else:
            raise ValueError(f"Unknown volatility method: {self.method}")

    def _ewm_volatility(self, close: pd.Series) -> pd.Series:
        """EWM standard deviation of returns"""
        returns = close.pct_change()
        return returns.ewm(span=self.lookback, min_periods=self.min_periods).std()

    def _rolling_volatility(self, close: pd.Series) -> pd.Series:
        """Rolling standard deviation of returns"""
        returns = close.pct_change()
        return returns.rolling(window=self.lookback, min_periods=self.min_periods).std()

    def _garman_klass_volatility(self, prices: pd.DataFrame) -> pd.Series:
        """
        Garman-Klass volatility estimator.

        Uses OHLC prices for more efficient estimation.
        """
        if not all(col in prices.columns for col in ['open', 'high', 'low', 'close']):
            logger.warning("OHLC columns not found, falling back to EWM")
            return self._ewm_volatility(prices['close'])

        log_hl = np.log(prices['high'] / prices['low'])
        log_co = np.log(prices['close'] / prices['open'])

        # Garman-Klass formula
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

        return gk.rolling(window=self.lookback, min_periods=self.min_periods).mean().apply(np.sqrt)

    def _parkinson_volatility(self, prices: pd.DataFrame) -> pd.Series:
        """
        Parkinson volatility estimator.

        Uses High-Low range for estimation.
        """
        if not all(col in prices.columns for col in ['high', 'low']):
            logger.warning("High-Low columns not found, falling back to EWM")
            return self._ewm_volatility(prices['close'])

        log_hl = np.log(prices['high'] / prices['low'])

        # Parkinson formula
        parkinson = log_hl ** 2 / (4 * np.log(2))

        return parkinson.rolling(window=self.lookback, min_periods=self.min_periods).mean().apply(np.sqrt)


class TripleBarrierLabeler:
    """
    Triple Barrier Method for path-dependent labeling.

    Based on AFML Chapter 3: Labeling.

    The Triple Barrier Method defines three barriers:
    1. Upper Barrier (Profit Take): Price touches upper profit target
    2. Lower Barrier (Stop Loss): Price touches lower stop loss
    3. Vertical Barrier (Time Expiration): Maximum holding period reached

    The label is determined by which barrier is touched first:
    - Upper first -> Label = 1 (profitable trade)
    - Lower first -> Label = -1 (losing trade)
    - Vertical first -> Label depends on position at expiration

    This approach captures:
    - Path dependency of returns
    - Variable holding periods
    - Asymmetric profit/loss profiles
    """

    def __init__(
        self,
        config: Optional[TripleBarrierConfig] = None
    ):
        """
        Initialize TripleBarrierLabeler.

        Args:
            config: Triple barrier configuration
        """
        self.config = config or TripleBarrierConfig()
        self.volatility_estimator = VolatilityEstimator(
            method=self.config.volatility_method,
            lookback=self.config.volatility_lookback
        )

        logger.info(
            f"TripleBarrierLabeler initialized: "
            f"pt_sl_ratio={self.config.pt_sl_ratio}, "
            f"max_holding={self.config.max_holding_period}"
        )

    def get_daily_volatility(
        self,
        prices: pd.DataFrame,
        column: str = "close"
    ) -> pd.Series:
        """
        Get daily volatility target for barrier widths.

        Args:
            prices: Price DataFrame
            column: Column to use

        Returns:
            Series of volatility targets
        """
        return self.volatility_estimator.estimate(prices, column)

    def get_vertical_barriers(
        self,
        t_events: pd.DatetimeIndex,
        prices: pd.DataFrame,
        num_bars: Optional[int] = None,
        training_end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Get vertical barrier timestamps.

        FIXED: Added training_end_date parameter to prevent look-ahead bias.
        When labeling for training, events near the end of the training period
        should be excluded if their vertical barrier extends into the holdout period.

        Args:
            t_events: Event timestamps
            prices: Price DataFrame with DatetimeIndex
            num_bars: Number of bars for vertical barrier (default: config value)
            training_end_date: If provided, exclude events whose vertical barrier
                               extends past this date (prevents label leakage)

        Returns:
            Series mapping event time to vertical barrier time
        """
        num_bars = num_bars or self.config.max_holding_period

        # Get all price indices
        price_idx = prices.index

        vertical_barriers = {}

        for t in t_events:
            # Find position of event in price index
            try:
                loc = price_idx.get_loc(t)
            except KeyError:
                # Find nearest index
                loc = price_idx.searchsorted(t)
                if loc >= len(price_idx):
                    vertical_barriers[t] = pd.NaT
                    continue

            # Get vertical barrier time
            barrier_loc = min(loc + num_bars, len(price_idx) - 1)
            barrier_time = price_idx[barrier_loc]

            # FIXED: Check for look-ahead bias
            if training_end_date is not None:
                if barrier_time > training_end_date:
                    # This event's outcome extends into holdout period
                    # Mark as NaT to exclude from training
                    vertical_barriers[t] = pd.NaT
                    logger.debug(
                        f"Excluding event at {t}: vertical barrier {barrier_time} "
                        f"extends past training end {training_end_date}"
                    )
                    continue

            vertical_barriers[t] = barrier_time

        valid_count = sum(1 for v in vertical_barriers.values() if pd.notna(v))
        logger.info(
            f"Vertical barriers: {valid_count}/{len(t_events)} valid "
            f"({len(t_events) - valid_count} excluded for look-ahead)"
        )

        return pd.Series(vertical_barriers)

    def apply_pt_sl_on_t1(
        self,
        close: pd.Series,
        events: pd.DataFrame,
        pt_sl: Tuple[float, float]
    ) -> pd.DataFrame:
        """
        Apply profit-taking and stop-loss barriers.

        This is the core barrier-touching logic.

        Args:
            close: Close price series
            events: DataFrame with columns ['t1', 'trgt', 'side']
                   t1: Vertical barrier time
                   trgt: Target volatility (barrier width)
                   side: Position side (+1 long, -1 short)
            pt_sl: Tuple of (profit_take, stop_loss) multipliers

        Returns:
            DataFrame with barrier touch times and returns
        """
        out = events[['t1']].copy(deep=True)
        out['sl'] = pd.NaT
        out['pt'] = pd.NaT
        out['ret'] = np.nan

        pt_mult, sl_mult = pt_sl

        for loc, (t0, row) in enumerate(events.iterrows()):
            t1 = row['t1']
            trgt = row['trgt']
            side = row.get('side', 1)

            if pd.isna(t1) or pd.isna(trgt) or trgt <= 0:
                continue

            # Get price path from t0 to t1
            # Handle timezone issues by using positional indexing
            try:
                # Find positions to avoid timezone offset issues with DST
                t0_pos = close.index.get_loc(t0)
                t1_pos = close.index.get_loc(t1) if t1 in close.index else close.index.searchsorted(t1)
                path = close.iloc[t0_pos:t1_pos + 1]
            except (KeyError, TypeError):
                # Fallback to direct slicing if positional fails
                try:
                    path = close.loc[t0:t1]
                except Exception:
                    continue

            if len(path) < 2:
                continue

            # Calculate returns from entry
            entry_price = path.iloc[0]
            returns = (path / entry_price - 1) * side  # Adjust for side

            # Define barriers
            upper_barrier = trgt * pt_mult if pt_mult > 0 else np.inf
            lower_barrier = -trgt * sl_mult if sl_mult > 0 else -np.inf

            # Check for barrier touches
            # Upper barrier (profit take)
            pt_touches = returns[returns >= upper_barrier]
            if len(pt_touches) > 0:
                out.loc[t0, 'pt'] = pt_touches.index[0]

            # Lower barrier (stop loss)
            sl_touches = returns[returns <= lower_barrier]
            if len(sl_touches) > 0:
                out.loc[t0, 'sl'] = sl_touches.index[0]

            # Store return at first touch (not at t1)
            # Determine which barrier was touched first
            first_touch_time = None
            first_touch_return = returns.iloc[-1]  # Default to final return

            pt_time = out.loc[t0, 'pt']
            sl_time = out.loc[t0, 'sl']
            t1_time = out.loc[t0, 't1']

            # Find first touch
            touch_times = []
            if pd.notna(pt_time):
                touch_times.append(('pt', pt_time, upper_barrier))
            if pd.notna(sl_time):
                touch_times.append(('sl', sl_time, lower_barrier))
            if pd.notna(t1_time):
                touch_times.append(('t1', t1_time, returns.iloc[-1]))

            if touch_times:
                # Sort by time and get first
                touch_times.sort(key=lambda x: x[1])
                first_type, first_time, barrier_val = touch_times[0]

                # Get return at first touch
                if first_type == 'pt':
                    first_touch_return = barrier_val  # Upper barrier value
                elif first_type == 'sl':
                    first_touch_return = barrier_val  # Lower barrier value
                else:  # vertical
                    first_touch_return = returns.iloc[-1]

            out.loc[t0, 'ret'] = first_touch_return

        return out

    def get_first_touch(
        self,
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Determine which barrier was touched first.

        Args:
            events: DataFrame from apply_pt_sl_on_t1
            close: Close price series

        Returns:
            DataFrame with first touch info and labels
        """
        out = events.copy()

        # Determine first touch - handle NaN values by converting to datetime
        # and using pd.NaT for missing values
        touch_cols = out[['t1', 'sl', 'pt']].copy()
        for col in touch_cols.columns:
            touch_cols[col] = pd.to_datetime(touch_cols[col], errors='coerce')
        out['first_touch'] = touch_cols.min(axis=1, skipna=True)

        # Determine touch type
        out['touch_type'] = 'vertical'  # Default

        for idx in out.index:
            row = out.loc[idx]
            first = row['first_touch']

            if pd.notna(row['pt']) and row['pt'] == first:
                out.loc[idx, 'touch_type'] = 'upper'
            elif pd.notna(row['sl']) and row['sl'] == first:
                out.loc[idx, 'touch_type'] = 'lower'

        return out

    def get_labels(
        self,
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.Series:
        """
        Get labels based on barrier touches.

        Label scheme:
        - 1: Upper barrier touched first (profitable)
        - -1: Lower barrier touched first (loss)
        - 0: Vertical barrier (depends on sign of return)

        For binary classification:
        - 1: Profitable (upper or positive at vertical)
        - 0: Loss (lower or negative at vertical)

        Args:
            events: DataFrame with barrier info
            close: Close price series

        Returns:
            Series of labels
        """
        events_with_touch = self.get_first_touch(events, close)

        labels = pd.Series(index=events_with_touch.index, dtype=float)

        for idx in events_with_touch.index:
            row = events_with_touch.loc[idx]
            touch_type = row['touch_type']
            ret = row.get('ret', 0)

            if touch_type == 'upper':
                labels[idx] = 1
            elif touch_type == 'lower':
                labels[idx] = -1
            else:  # vertical
                labels[idx] = np.sign(ret) if ret != 0 else 0

        return labels

    def get_events(
        self,
        close: pd.Series,
        t_events: pd.DatetimeIndex,
        pt_sl: Optional[Tuple[float, float]] = None,
        target: Optional[pd.Series] = None,
        min_ret: Optional[float] = None,
        vertical_barrier_times: Optional[pd.Series] = None,
        side: Optional[pd.Series] = None,
        training_end_date: Optional[pd.Timestamp] = None,
        require_training_end_date: bool = True
    ) -> pd.DataFrame:
        """
        Main entry point for Triple Barrier labeling.

        This function:
        1. Gets target volatility for each event
        2. Gets vertical barrier times
        3. Forms events DataFrame
        4. Applies barrier logic

        ISSUE 1.5 FIX: training_end_date is now REQUIRED by default to prevent
        label leakage. Events whose vertical barrier extends past training_end_date
        are excluded. Set require_training_end_date=False only for live trading
        or when you explicitly want to disable this protection.

        Args:
            close: Close price series
            t_events: Event timestamps (signal times)
            pt_sl: (profit_take, stop_loss) multipliers
            target: Pre-computed volatility targets (optional)
            min_ret: Minimum return threshold
            vertical_barrier_times: Pre-computed vertical barriers (optional)
            side: Pre-determined side for each event (optional)
            training_end_date: REQUIRED for training. End date of training period.
                               Events whose outcome extends past this date are excluded
                               to prevent look-ahead bias.
            require_training_end_date: If True (default), raises error when
                               training_end_date is not provided. Set to False
                               only for live trading or deliberate bypass.

        Returns:
            DataFrame with columns:
            - t1: Vertical barrier time
            - trgt: Target volatility
            - side: Position side
            - sl: Stop loss touch time
            - pt: Profit take touch time
            - ret: Return at first touch
            - label: Final label

        Raises:
            ValueError: If require_training_end_date=True and training_end_date is None
        """
        # ISSUE 1.5 FIX: Enforce training_end_date requirement
        if require_training_end_date and training_end_date is None:
            raise ValueError(
                "training_end_date is REQUIRED to prevent label leakage. "
                "Events near the end of the training period can have labels "
                "that look into the future (holdout period). Provide training_end_date "
                "to automatically filter these events. "
                "If you are doing live trading or explicitly want to bypass this check, "
                "set require_training_end_date=False."
            )

        if training_end_date is not None:
            logger.info(
                f"Label generation with training_end_date={training_end_date}: "
                f"events with barriers extending past this date will be excluded"
            )

        # Get configuration values
        pt_sl = pt_sl or self.config.pt_sl_ratio
        min_ret = min_ret if min_ret is not None else self.config.min_return

        # 1. Get target volatility
        if target is None:
            target = self.get_daily_volatility(
                pd.DataFrame({'close': close}),
                column='close'
            )

        # Align target to events
        target = target.reindex(t_events)

        # Filter by minimum return
        target = target[target > min_ret]

        if len(target) == 0:
            logger.warning("No events meet minimum return threshold")
            return pd.DataFrame()

        # 2. Get vertical barriers (FIXED: now supports training_end_date)
        if vertical_barrier_times is None:
            vertical_barrier_times = self.get_vertical_barriers(
                target.index,
                pd.DataFrame({'close': close}),
                training_end_date=training_end_date  # FIXED: Pass through
            )
        else:
            vertical_barrier_times = vertical_barrier_times.reindex(target.index)
            # Also filter pre-computed barriers if training_end_date provided
            if training_end_date is not None:
                mask = vertical_barrier_times > training_end_date
                vertical_barrier_times[mask] = pd.NaT
                if mask.sum() > 0:
                    logger.info(
                        f"Filtered {mask.sum()} pre-computed vertical barriers "
                        f"that extend past training end date"
                    )

        # 3. Form events object
        if side is None:
            side_ = pd.Series(1.0, index=target.index)
        else:
            side_ = side.reindex(target.index)

        events = pd.concat({
            't1': vertical_barrier_times,
            'trgt': target,
            'side': side_
        }, axis=1).dropna(subset=['trgt'])

        logger.info(f"Processing {len(events)} events")

        # 4. Apply triple barrier logic
        out = self.apply_pt_sl_on_t1(close, events, pt_sl)

        # 5. Get labels
        out['label'] = self.get_labels(out, close)

        # Add binary label for classification
        out['bin_label'] = (out['label'] > 0).astype(int)

        logger.info(
            f"Labeling complete: "
            f"{(out['label'] == 1).sum()} positive, "
            f"{(out['label'] == -1).sum()} negative, "
            f"{(out['label'] == 0).sum()} neutral"
        )

        return out

    def get_events_with_ohlcv(
        self,
        prices: pd.DataFrame,
        t_events: Optional[pd.DatetimeIndex] = None,
        pt_sl: Optional[Tuple[float, float]] = None,
        side: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Convenience method for OHLCV DataFrames.

        Args:
            prices: DataFrame with OHLCV columns
            t_events: Event timestamps (default: all timestamps)
            pt_sl: Barrier multipliers
            side: Position sides

        Returns:
            Events DataFrame with labels
        """
        if t_events is None:
            t_events = prices.index

        # Use close prices
        close = prices['close']

        # Get volatility using available data
        target = self.volatility_estimator.estimate(prices, 'close')

        return self.get_events(
            close=close,
            t_events=t_events,
            pt_sl=pt_sl,
            target=target,
            side=side
        )


class MetaLabeler:
    """
    Meta-Labeling for secondary model training.

    Based on AFML Chapter 3.6: Meta-Labeling.

    Meta-labeling is a two-stage approach:
    1. Primary Model: Determines trade direction (side)
       - Can be any model or signal (technical, fundamental, ML)
       - Outputs: Long (+1), Short (-1), or Flat (0)

    2. Secondary Model (Meta-Model): Determines bet size
       - Binary classifier: Trade (1) or No-Trade (0)
       - Predicts whether primary model's signal will be profitable

    Benefits:
    - Separates direction prediction from bet sizing
    - Reduces false positives from primary model
    - Easier to interpret model behavior
    - Allows different ML approaches for each stage

    The meta-label is:
    - 1: If the trade was profitable (primary model correct)
    - 0: If the trade was unprofitable (primary model wrong)

    FIXED: Added auto-calibration for optimal threshold.
    """

    def __init__(
        self,
        config: Optional[MetaLabelingConfig] = None
    ):
        """
        Initialize MetaLabeler.

        Args:
            config: Meta-labeling configuration
        """
        self.config = config or MetaLabelingConfig()
        self._calibrated_threshold = None

        logger.info(
            f"MetaLabeler initialized: "
            f"threshold={self.config.primary_threshold}"
        )

    def calibrate_threshold(
        self,
        primary_predictions: pd.Series,
        actual_outcomes: pd.Series,
        metric: str = 'f1',
        n_thresholds: int = 20
    ) -> float:
        """
        Auto-calibrate the optimal threshold for meta-labeling.

        FIXES the issue where a fixed 0.5 threshold may not be optimal.
        This method finds the threshold that maximizes the chosen metric.

        Args:
            primary_predictions: Probability predictions from primary model
            actual_outcomes: Actual labels (1 for profitable, -1 for loss)
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            n_thresholds: Number of thresholds to test

        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        # Align indices
        common_idx = primary_predictions.index.intersection(actual_outcomes.index)
        if len(common_idx) == 0:
            logger.warning("No common indices for threshold calibration")
            return self.config.primary_threshold

        preds = primary_predictions.loc[common_idx].values
        actuals = (actual_outcomes.loc[common_idx] > 0).astype(int).values

        # Test different thresholds
        thresholds = np.linspace(0.3, 0.7, n_thresholds)
        best_threshold = 0.5
        best_score = 0

        metric_funcs = {
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score,
            'accuracy': accuracy_score
        }
        metric_func = metric_funcs.get(metric, f1_score)

        for threshold in thresholds:
            binary_preds = (preds > threshold).astype(int)

            try:
                score = metric_func(actuals, binary_preds, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except Exception:
                continue

        self._calibrated_threshold = best_threshold

        logger.info(
            f"Meta-labeling threshold calibrated: {best_threshold:.3f} "
            f"(optimized for {metric}, score={best_score:.4f})"
        )

        return best_threshold

    def get_threshold(self) -> float:
        """Get current threshold (calibrated or default)."""
        if self._calibrated_threshold is not None:
            return self._calibrated_threshold
        return self.config.primary_threshold

    def get_primary_side(
        self,
        primary_predictions: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Convert primary model predictions to side (direction).

        FIXED: Uses calibrated threshold if available.

        Args:
            primary_predictions: Predictions from primary model
                - If probabilities: convert using threshold
                - If binary: use directly
            threshold: Classification threshold (uses calibrated if None)

        Returns:
            Series of sides: +1 (long), -1 (short), 0 (no position)
        """
        # FIXED: Use calibrated threshold if available
        if threshold is None:
            threshold = self.get_threshold()

        if isinstance(primary_predictions, np.ndarray):
            primary_predictions = pd.Series(primary_predictions)

        # Check if predictions are probabilities
        if self.config.use_probability and primary_predictions.max() <= 1:
            # Convert probabilities to sides
            # Assume: P > threshold -> Long, P < (1-threshold) -> Short
            side = pd.Series(0.0, index=primary_predictions.index)
            side[primary_predictions > threshold] = 1.0
            side[primary_predictions < (1 - threshold)] = -1.0
        else:
            # Use predictions directly as sides
            side = primary_predictions.map({1: 1.0, 0: -1.0, -1: -1.0})

        return side

    def create_meta_labels(
        self,
        triple_barrier_events: pd.DataFrame,
        primary_side: pd.Series
    ) -> pd.DataFrame:
        """
        Create meta-labels from triple barrier events and primary model side.

        The meta-label indicates whether the primary model's prediction
        was correct (i.e., the trade was profitable).

        Args:
            triple_barrier_events: Output from TripleBarrierLabeler.get_events()
                Must contain 'label' or 'bin_label' column
            primary_side: Side predictions from primary model (+1, -1)

        Returns:
            DataFrame with meta-labels and aligned data
        """
        # Align indices
        common_idx = triple_barrier_events.index.intersection(primary_side.index)

        if len(common_idx) == 0:
            logger.warning("No common indices between events and primary predictions")
            return pd.DataFrame()

        events = triple_barrier_events.loc[common_idx].copy()
        side = primary_side.loc[common_idx]

        # Get the actual trade outcome
        # label: 1 (profitable), -1 (loss), 0 (flat)
        actual_outcome = events['label'].fillna(0)

        # Meta-label: Was the primary model correct?
        # If side and outcome have same sign -> correct prediction
        meta_label = (side * actual_outcome > 0).astype(int)

        events['primary_side'] = side
        events['meta_label'] = meta_label

        # Statistics
        correct = meta_label.sum()
        total = len(meta_label)
        accuracy = correct / total if total > 0 else 0

        logger.info(
            f"Meta-labeling complete: "
            f"{correct}/{total} correct ({accuracy:.2%})"
        )

        return events

    def get_meta_training_data(
        self,
        features: pd.DataFrame,
        meta_events: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for meta-model training.

        Args:
            features: Feature DataFrame
            meta_events: DataFrame from create_meta_labels()

        Returns:
            Tuple of (aligned features, meta-labels)
        """
        # Align indices
        common_idx = features.index.intersection(meta_events.index)

        X = features.loc[common_idx]
        y = meta_events.loc[common_idx, 'meta_label']

        # Filter out samples where primary model gave no signal
        valid_mask = meta_events.loc[common_idx, 'primary_side'] != 0

        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"Meta-training data: {len(X)} samples")

        return X, y


class CUSUMFilter:
    """
    CUSUM Filter for event sampling.

    Based on AFML Chapter 2.5.2.

    CUSUM (Cumulative Sum) detects structural breaks in time series.
    Used to sample events when the series has drifted significantly.

    Benefits over fixed-time sampling:
    - Adapts to market conditions
    - More events during high volatility
    - Fewer events during consolidation
    """

    def __init__(
        self,
        threshold: float = 0.01,
        reset_on_signal: bool = True
    ):
        """
        Initialize CUSUMFilter.

        Args:
            threshold: CUSUM threshold for event detection
            reset_on_signal: Reset CUSUM after signal
        """
        self.threshold = threshold
        self.reset_on_signal = reset_on_signal

    def get_events(
        self,
        returns: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.DatetimeIndex:
        """
        Get event timestamps using CUSUM filter.

        Args:
            returns: Returns series
            threshold: CUSUM threshold (default: instance value)

        Returns:
            DatetimeIndex of event timestamps
        """
        threshold = threshold or self.threshold

        events = []
        s_pos = 0
        s_neg = 0

        returns = returns.dropna()

        for t, r in returns.items():
            # Update CUSUM values
            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)

            # Check for threshold breach
            if s_pos > threshold:
                events.append(t)
                if self.reset_on_signal:
                    s_pos = 0

            elif s_neg < -threshold:
                events.append(t)
                if self.reset_on_signal:
                    s_neg = 0

        logger.info(f"CUSUM filter detected {len(events)} events")

        return pd.DatetimeIndex(events)


def get_sample_weights(
    events: pd.DataFrame,
    close: pd.Series,
    num_threads: int = 1
) -> pd.Series:
    """
    Calculate sample weights based on concurrent labels.

    Based on AFML Chapter 4.5.

    Events that overlap with many other events should have lower weight
    to avoid oversampling certain time periods.

    Args:
        events: DataFrame with 't1' column (barrier touch time)
        close: Close price series
        num_threads: Number of parallel threads

    Returns:
        Series of sample weights
    """
    # Count concurrent labels at each time point
    concurrent = pd.Series(0, index=close.index)

    for t0, row in events.iterrows():
        t1 = row['t1']
        if pd.isna(t1):
            continue

        # Increment count for all times this label spans
        concurrent.loc[t0:t1] += 1

    # Calculate average uniqueness for each sample
    weights = pd.Series(index=events.index, dtype=float)

    for t0, row in events.iterrows():
        t1 = row['t1']
        if pd.isna(t1):
            weights[t0] = 0
            continue

        # Get concurrent labels during this sample's lifetime
        concurrent_during_sample = concurrent.loc[t0:t1]

        if len(concurrent_during_sample) > 0:
            # Average uniqueness = 1 / average_concurrency
            avg_concurrency = concurrent_during_sample.mean()
            weights[t0] = 1 / avg_concurrency if avg_concurrency > 0 else 0
        else:
            weights[t0] = 0

    # Normalize weights
    weights = weights / weights.sum() if weights.sum() > 0 else weights

    return weights


def get_time_decay_weights(
    events: pd.DataFrame,
    c: float = 1.0
) -> pd.Series:
    """
    Calculate time-decay sample weights.

    Based on AFML Chapter 4.6.

    More recent samples get higher weights to adapt to
    changing market conditions.

    Args:
        events: DataFrame with event timestamps as index
        c: Decay factor (0: no decay, 1: linear decay)

    Returns:
        Series of time-decay weights
    """
    n = len(events)

    if c == 0:
        # No decay - equal weights
        weights = pd.Series(1, index=events.index)
    else:
        # Linear decay with factor c
        # Most recent gets weight 1, oldest gets weight (1-c)
        weights = pd.Series(
            np.linspace(1 - c, 1, n),
            index=events.index
        )

    # Normalize
    weights = weights / weights.sum()

    return weights


def combine_weights(
    uniqueness_weights: pd.Series,
    time_decay_weights: pd.Series,
    alpha: float = 0.5
) -> pd.Series:
    """
    Combine uniqueness and time-decay weights.

    Args:
        uniqueness_weights: Weights from concurrent label analysis
        time_decay_weights: Weights from time decay
        alpha: Weight for uniqueness (1-alpha for time decay)

    Returns:
        Combined and normalized weights
    """
    # Align indices
    common_idx = uniqueness_weights.index.intersection(time_decay_weights.index)

    u = uniqueness_weights.loc[common_idx]
    t = time_decay_weights.loc[common_idx]

    # Combine
    combined = alpha * u + (1 - alpha) * t

    # Normalize
    combined = combined / combined.sum() if combined.sum() > 0 else combined

    return combined


# =============================================================================
# ISSUE 1.1 FIX: Label Decorrelation Functions
# =============================================================================

def calculate_label_autocorrelation(
    labels: pd.Series,
    max_lags: int = 10
) -> Dict[int, float]:
    """
    Calculate autocorrelation of labels at different lags.

    High autocorrelation (> 0.1) indicates serial correlation that can
    inflate cross-validation scores and cause overfitting.

    Args:
        labels: Series of labels (typically -1, 0, 1)
        max_lags: Maximum number of lags to compute

    Returns:
        Dictionary mapping lag -> autocorrelation value
    """
    autocorrs = {}
    labels_clean = labels.dropna()

    for lag in range(1, max_lags + 1):
        try:
            autocorrs[lag] = labels_clean.autocorr(lag=lag)
        except Exception:
            autocorrs[lag] = 0.0

    logger.info(
        f"Label autocorrelation: lag1={autocorrs.get(1, 0):.3f}, "
        f"lag5={autocorrs.get(5, 0):.3f}"
    )

    return autocorrs


def decorrelate_labels(
    labels: pd.Series,
    max_autocorr: float = 0.1,
    method: str = 'subsample',
    min_samples: int = 100
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Reduce label autocorrelation to target level.

    FIXES Issue 1.1: Label autocorrelation between 0.65-0.71 is extremely high
    and indicates strong serial correlation that will cause CV score inflation.
    Target autocorrelation should be < 0.1.

    Methods:
    - 'subsample': Skip samples to reduce correlation (simple, effective)
    - 'random': Randomly drop correlated samples
    - 'clustered': Use time-based clustering to select representative samples

    Args:
        labels: Series of labels with DatetimeIndex
        max_autocorr: Target maximum autocorrelation (default 0.1)
        method: Decorrelation method ('subsample', 'random', 'clustered')
        min_samples: Minimum samples to retain

    Returns:
        Tuple of (decorrelated labels, diagnostic info dict)
    """
    labels_clean = labels.dropna()
    original_count = len(labels_clean)

    if original_count < min_samples:
        logger.warning(f"Too few samples ({original_count}) for decorrelation")
        return labels, {'skipped': True, 'reason': 'insufficient_samples'}

    # Calculate initial autocorrelation
    initial_autocorr = labels_clean.autocorr(lag=1)

    if initial_autocorr <= max_autocorr:
        logger.info(
            f"Labels already decorrelated: autocorr={initial_autocorr:.3f} "
            f"<= target={max_autocorr:.3f}"
        )
        return labels, {
            'initial_autocorr': initial_autocorr,
            'final_autocorr': initial_autocorr,
            'samples_removed': 0,
            'skip_factor': 1
        }

    logger.info(
        f"Decorrelating labels: initial autocorr={initial_autocorr:.3f}, "
        f"target={max_autocorr:.3f}"
    )

    if method == 'subsample':
        return _decorrelate_subsample(labels_clean, max_autocorr, min_samples, initial_autocorr)
    elif method == 'random':
        return _decorrelate_random(labels_clean, max_autocorr, min_samples, initial_autocorr)
    elif method == 'clustered':
        return _decorrelate_clustered(labels_clean, max_autocorr, min_samples, initial_autocorr)
    else:
        raise ValueError(f"Unknown decorrelation method: {method}")


def _decorrelate_subsample(
    labels: pd.Series,
    max_autocorr: float,
    min_samples: int,
    initial_autocorr: float
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Decorrelate by subsampling every N-th sample.

    This is the simplest and most effective method for reducing
    autocorrelation while maintaining temporal structure.
    """
    # Binary search for optimal skip factor
    best_skip = 1
    best_autocorr = initial_autocorr

    max_skip = min(len(labels) // min_samples, 20)  # Don't skip more than 1 in 20

    for skip_factor in range(2, max_skip + 1):
        subsampled = labels.iloc[::skip_factor]

        if len(subsampled) < min_samples:
            break

        current_autocorr = subsampled.autocorr(lag=1)

        if pd.isna(current_autocorr):
            continue

        if current_autocorr <= max_autocorr:
            best_skip = skip_factor
            best_autocorr = current_autocorr
            break
        elif current_autocorr < best_autocorr:
            best_skip = skip_factor
            best_autocorr = current_autocorr

    # Apply best skip factor
    result = labels.iloc[::best_skip]
    final_autocorr = result.autocorr(lag=1) if len(result) > 1 else 0

    samples_removed = len(labels) - len(result)

    logger.info(
        f"Label decorrelation complete: autocorr {initial_autocorr:.3f} -> {final_autocorr:.3f}, "
        f"samples {len(labels)} -> {len(result)} (skip_factor={best_skip})"
    )

    return result, {
        'method': 'subsample',
        'initial_autocorr': initial_autocorr,
        'final_autocorr': final_autocorr,
        'samples_removed': samples_removed,
        'samples_retained': len(result),
        'skip_factor': best_skip,
        'target_met': final_autocorr <= max_autocorr
    }


def _decorrelate_random(
    labels: pd.Series,
    max_autocorr: float,
    min_samples: int,
    initial_autocorr: float
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Decorrelate by randomly dropping samples until target is met.
    """
    result = labels.copy()
    iterations = 0
    max_iterations = 100

    while len(result) > min_samples and iterations < max_iterations:
        current_autocorr = result.autocorr(lag=1)

        if pd.isna(current_autocorr) or current_autocorr <= max_autocorr:
            break

        # Drop 10% of samples randomly
        n_drop = max(1, len(result) // 10)
        drop_idx = np.random.choice(result.index, n_drop, replace=False)
        result = result.drop(drop_idx)
        iterations += 1

    final_autocorr = result.autocorr(lag=1) if len(result) > 1 else 0

    logger.info(
        f"Random decorrelation: autocorr {initial_autocorr:.3f} -> {final_autocorr:.3f}, "
        f"samples {len(labels)} -> {len(result)}"
    )

    return result, {
        'method': 'random',
        'initial_autocorr': initial_autocorr,
        'final_autocorr': final_autocorr,
        'samples_removed': len(labels) - len(result),
        'samples_retained': len(result),
        'iterations': iterations,
        'target_met': final_autocorr <= max_autocorr
    }


def _decorrelate_clustered(
    labels: pd.Series,
    max_autocorr: float,
    min_samples: int,
    initial_autocorr: float
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Decorrelate using time-based clustering.

    Groups labels by time clusters and selects one representative
    from each cluster, reducing temporal autocorrelation.
    """
    # Calculate cluster size based on autocorrelation
    # Higher autocorr = larger clusters needed
    base_cluster_size = max(2, int(initial_autocorr * 10))

    # Create time-based clusters
    n_clusters = max(min_samples, len(labels) // base_cluster_size)

    # Assign cluster labels
    cluster_labels = pd.Series(
        np.repeat(range(n_clusters), len(labels) // n_clusters + 1)[:len(labels)],
        index=labels.index
    )

    # Select one sample from each cluster (middle sample)
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = labels.index[cluster_mask]

        if len(cluster_indices) > 0:
            # Select middle sample from cluster
            mid_idx = len(cluster_indices) // 2
            selected_indices.append(cluster_indices[mid_idx])

    result = labels.loc[selected_indices]
    final_autocorr = result.autocorr(lag=1) if len(result) > 1 else 0

    logger.info(
        f"Clustered decorrelation: autocorr {initial_autocorr:.3f} -> {final_autocorr:.3f}, "
        f"samples {len(labels)} -> {len(result)}, clusters={n_clusters}"
    )

    return result, {
        'method': 'clustered',
        'initial_autocorr': initial_autocorr,
        'final_autocorr': final_autocorr,
        'samples_removed': len(labels) - len(result),
        'samples_retained': len(result),
        'n_clusters': n_clusters,
        'cluster_size': base_cluster_size,
        'target_met': final_autocorr <= max_autocorr
    }


def apply_label_decorrelation_to_events(
    events: pd.DataFrame,
    max_autocorr: float = 0.1,
    method: str = 'subsample',
    min_samples: int = 100
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply decorrelation to events DataFrame (full pipeline integration).

    This is the main entry point for applying decorrelation to the output
    of TripleBarrierLabeler.get_events().

    Args:
        events: Events DataFrame from TripleBarrierLabeler with 'label' column
        max_autocorr: Target maximum autocorrelation
        method: Decorrelation method
        min_samples: Minimum samples to retain

    Returns:
        Tuple of (decorrelated events DataFrame, diagnostic info)
    """
    if 'label' not in events.columns and 'bin_label' not in events.columns:
        raise ValueError("Events DataFrame must have 'label' or 'bin_label' column")

    label_col = 'label' if 'label' in events.columns else 'bin_label'
    labels = events[label_col]

    # Perform decorrelation
    decorrelated_labels, info = decorrelate_labels(
        labels,
        max_autocorr=max_autocorr,
        method=method,
        min_samples=min_samples
    )

    # Filter events to match decorrelated labels
    decorrelated_events = events.loc[decorrelated_labels.index]

    logger.info(
        f"Events decorrelated: {len(events)} -> {len(decorrelated_events)} events"
    )

    return decorrelated_events, info


def validate_label_quality(
    labels: pd.Series,
    max_autocorr: float = 0.1,
    min_class_ratio: float = 0.20,
    max_class_ratio: float = 0.80
) -> Dict[str, Any]:
    """
    Validate label quality for training.

    Checks:
    1. Autocorrelation (should be < max_autocorr)
    2. Class balance (should be between min_class_ratio and max_class_ratio)
    3. Sufficient samples
    4. No missing values

    Args:
        labels: Series of labels
        max_autocorr: Maximum acceptable autocorrelation
        min_class_ratio: Minimum acceptable ratio for minority class
        max_class_ratio: Maximum acceptable ratio for majority class

    Returns:
        Dictionary with validation results
    """
    labels_clean = labels.dropna()

    results = {
        'total_samples': len(labels),
        'valid_samples': len(labels_clean),
        'missing_samples': len(labels) - len(labels_clean),
        'passed': True,
        'warnings': [],
        'errors': []
    }

    # Check autocorrelation
    if len(labels_clean) > 10:
        autocorr = labels_clean.autocorr(lag=1)
        results['autocorrelation'] = autocorr

        if autocorr > max_autocorr:
            results['errors'].append(
                f"Autocorrelation {autocorr:.3f} exceeds max {max_autocorr:.3f}"
            )
            results['passed'] = False

    # Check class balance
    value_counts = labels_clean.value_counts(normalize=True)
    results['class_distribution'] = value_counts.to_dict()

    for class_val, ratio in value_counts.items():
        if ratio < min_class_ratio:
            results['warnings'].append(
                f"Class {class_val} ratio {ratio:.3f} below minimum {min_class_ratio:.3f}"
            )
        if ratio > max_class_ratio:
            results['warnings'].append(
                f"Class {class_val} ratio {ratio:.3f} above maximum {max_class_ratio:.3f}"
            )

    # Check sample count
    if len(labels_clean) < 100:
        results['warnings'].append(
            f"Low sample count: {len(labels_clean)} (recommend >100)"
        )

    # Log results
    if results['passed']:
        logger.info(f"Label quality validation PASSED: {len(results['warnings'])} warnings")
    else:
        logger.error(f"Label quality validation FAILED: {results['errors']}")

    return results
