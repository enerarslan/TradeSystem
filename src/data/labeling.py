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
        num_bars: Optional[int] = None
    ) -> pd.Series:
        """
        Get vertical barrier timestamps.

        Args:
            t_events: Event timestamps
            prices: Price DataFrame with DatetimeIndex
            num_bars: Number of bars for vertical barrier (default: config value)

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
            vertical_barriers[t] = price_idx[barrier_loc]

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

            # Store final return
            out.loc[t0, 'ret'] = returns.iloc[-1]

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
        side: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Main entry point for Triple Barrier labeling.

        This function:
        1. Gets target volatility for each event
        2. Gets vertical barrier times
        3. Forms events DataFrame
        4. Applies barrier logic

        Args:
            close: Close price series
            t_events: Event timestamps (signal times)
            pt_sl: (profit_take, stop_loss) multipliers
            target: Pre-computed volatility targets (optional)
            min_ret: Minimum return threshold
            vertical_barrier_times: Pre-computed vertical barriers (optional)
            side: Pre-determined side for each event (optional)

        Returns:
            DataFrame with columns:
            - t1: Vertical barrier time
            - trgt: Target volatility
            - side: Position side
            - sl: Stop loss touch time
            - pt: Profit take touch time
            - ret: Return at first touch
            - label: Final label
        """
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

        # 2. Get vertical barriers
        if vertical_barrier_times is None:
            vertical_barrier_times = self.get_vertical_barriers(
                target.index,
                pd.DataFrame({'close': close})
            )
        else:
            vertical_barrier_times = vertical_barrier_times.reindex(target.index)

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

        logger.info(
            f"MetaLabeler initialized: "
            f"threshold={self.config.primary_threshold}"
        )

    def get_primary_side(
        self,
        primary_predictions: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Convert primary model predictions to side (direction).

        Args:
            primary_predictions: Predictions from primary model
                - If probabilities: convert using threshold
                - If binary: use directly
            threshold: Classification threshold

        Returns:
            Series of sides: +1 (long), -1 (short), 0 (no position)
        """
        threshold = threshold or self.config.primary_threshold

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
