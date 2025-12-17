"""
VPIN (Volume-Synchronized Probability of Informed Trading).

VPIN estimates the probability of informed trading in a market.
High VPIN indicates elevated informed trading and potential price
dislocations.

Reference:
    Easley, López de Prado, O'Hara (2011) - "The Microstructure of the
    'Flash Crash': Flow Toxicity, Liquidity Crashes and the Probability
    of Informed Trading"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VPINResult:
    """Container for VPIN calculation results."""

    vpin: pd.Series                    # VPIN values
    bucket_buys: pd.Series             # Buy volume per bucket
    bucket_sells: pd.Series            # Sell volume per bucket
    bucket_timestamps: pd.Series       # Timestamp when bucket completed
    n_buckets: int                     # Number of buckets created


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.

    VPIN aggregates trades into volume buckets (rather than time intervals)
    and measures the imbalance between buyer- and seller-initiated trades.

    High VPIN values (> 0.6) indicate:
    - Elevated informed trading
    - Higher probability of adverse price movements
    - Increased market maker losses

    Example usage:
        vpin = VPIN(bucket_size=50000, n_buckets=50)
        result = vpin.calculate(prices, volume)

        # Use as risk indicator
        if result.vpin.iloc[-1] > 0.7:
            reduce_exposure()
    """

    def __init__(
        self,
        bucket_size: Optional[int] = None,
        bucket_size_method: str = "percentile",
        bucket_percentile: float = 0.01,
        n_buckets: int = 50,
        classification_method: str = "bulk",
    ) -> None:
        """
        Initialize VPIN calculator.

        Args:
            bucket_size: Fixed volume bucket size (if None, auto-calculated)
            bucket_size_method: How to calculate bucket size ("percentile", "adv")
            bucket_percentile: Percentile of daily volume for bucket size
            n_buckets: Number of buckets for rolling VPIN
            classification_method: Trade classification ("bulk", "tick")
        """
        self.bucket_size = bucket_size
        self.bucket_size_method = bucket_size_method
        self.bucket_percentile = bucket_percentile
        self.n_buckets = n_buckets
        self.classification_method = classification_method

    def calculate(
        self,
        close: pd.Series,
        volume: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> VPINResult:
        """
        Calculate VPIN.

        Args:
            close: Close prices
            volume: Volume
            high: High prices (for bulk classification)
            low: Low prices (for bulk classification)

        Returns:
            VPINResult with VPIN values and bucket data
        """
        # Determine bucket size
        if self.bucket_size is None:
            bucket_size = self._auto_bucket_size(volume)
        else:
            bucket_size = self.bucket_size

        logger.debug(f"Using bucket size: {bucket_size:,.0f}")

        # Classify volume
        buy_vol, sell_vol = self._classify_volume(close, volume, high, low)

        # Create volume buckets
        bucket_data = self._create_buckets(
            buy_vol, sell_vol, volume, close.index, bucket_size
        )

        if len(bucket_data) < self.n_buckets:
            logger.warning(
                f"Insufficient buckets: {len(bucket_data)} < {self.n_buckets}"
            )
            # Return with available data
            return VPINResult(
                vpin=pd.Series(dtype=float),
                bucket_buys=pd.Series(dtype=float),
                bucket_sells=pd.Series(dtype=float),
                bucket_timestamps=pd.Series(dtype='datetime64[ns]'),
                n_buckets=len(bucket_data),
            )

        # Calculate rolling VPIN
        vpin_values = self._calculate_vpin(bucket_data)

        return VPINResult(
            vpin=vpin_values,
            bucket_buys=pd.Series([b["buy"] for b in bucket_data]),
            bucket_sells=pd.Series([b["sell"] for b in bucket_data]),
            bucket_timestamps=pd.Series([b["end_time"] for b in bucket_data]),
            n_buckets=len(bucket_data),
        )

    def _auto_bucket_size(self, volume: pd.Series) -> float:
        """Auto-calculate bucket size."""
        if self.bucket_size_method == "percentile":
            # Use percentile of daily volume
            daily_volume = volume.resample("D").sum()
            return daily_volume.quantile(self.bucket_percentile)
        else:
            # Use fraction of average daily volume
            adv = volume.resample("D").sum().mean()
            return adv * self.bucket_percentile

    def _classify_volume(
        self,
        close: pd.Series,
        volume: pd.Series,
        high: Optional[pd.Series],
        low: Optional[pd.Series],
    ) -> Tuple[pd.Series, pd.Series]:
        """Classify volume into buy/sell."""
        if self.classification_method == "bulk" and high is not None and low is not None:
            # Bulk Volume Classification
            bar_range = high - low
            normalized_position = (close - low) / bar_range.replace(0, np.nan)
            buy_vol = volume * normalized_position.fillna(0.5)
            sell_vol = volume * (1 - normalized_position.fillna(0.5))
        else:
            # Tick rule
            price_change = close.diff()
            tick_direction = np.sign(price_change).replace(0, np.nan).ffill().fillna(0)
            buy_mask = tick_direction > 0
            sell_mask = tick_direction < 0

            buy_vol = volume * buy_mask.astype(float)
            sell_vol = volume * sell_mask.astype(float)

            # Handle ties
            tie_mask = tick_direction == 0
            buy_vol[tie_mask] = volume[tie_mask] * 0.5
            sell_vol[tie_mask] = volume[tie_mask] * 0.5

        return buy_vol, sell_vol

    def _create_buckets(
        self,
        buy_vol: pd.Series,
        sell_vol: pd.Series,
        total_vol: pd.Series,
        index: pd.Index,
        bucket_size: float,
    ) -> List[Dict]:
        """Create volume buckets."""
        buckets = []

        cumulative_vol = 0
        bucket_buy = 0
        bucket_sell = 0
        bucket_start_idx = 0

        for i, (buy, sell, vol) in enumerate(zip(buy_vol, sell_vol, total_vol)):
            if pd.isna(vol):
                continue

            # Check if this bar completes one or more buckets
            remaining_vol = vol
            remaining_buy = buy
            remaining_sell = sell

            while cumulative_vol + remaining_vol >= bucket_size:
                # Volume needed to complete bucket
                vol_to_bucket = bucket_size - cumulative_vol

                # Proportionally allocate
                if remaining_vol > 0:
                    ratio = vol_to_bucket / remaining_vol
                else:
                    ratio = 0

                bucket_buy += remaining_buy * ratio
                bucket_sell += remaining_sell * ratio

                # Save bucket
                buckets.append({
                    "buy": bucket_buy,
                    "sell": bucket_sell,
                    "start_idx": bucket_start_idx,
                    "end_idx": i,
                    "end_time": index[i],
                })

                # Reset for next bucket
                remaining_vol -= vol_to_bucket
                remaining_buy -= remaining_buy * ratio
                remaining_sell -= remaining_sell * ratio

                cumulative_vol = 0
                bucket_buy = 0
                bucket_sell = 0
                bucket_start_idx = i + 1

            # Add remaining to current bucket
            cumulative_vol += remaining_vol
            bucket_buy += remaining_buy
            bucket_sell += remaining_sell

        return buckets

    def _calculate_vpin(self, bucket_data: List[Dict]) -> pd.Series:
        """Calculate rolling VPIN from buckets."""
        n = len(bucket_data)
        vpin_values = []
        timestamps = []

        for i in range(self.n_buckets - 1, n):
            # Get last n_buckets
            window = bucket_data[i - self.n_buckets + 1:i + 1]

            # Calculate order imbalance
            total_buy = sum(b["buy"] for b in window)
            total_sell = sum(b["sell"] for b in window)
            total_volume = total_buy + total_sell

            if total_volume > 0:
                vpin = abs(total_buy - total_sell) / total_volume
            else:
                vpin = 0.5

            vpin_values.append(vpin)
            timestamps.append(window[-1]["end_time"])

        return pd.Series(vpin_values, index=timestamps, name="VPIN")


def calculate_vpin(
    close: pd.Series,
    volume: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    n_buckets: int = 50,
) -> pd.Series:
    """
    Convenience function to calculate VPIN.

    Args:
        close: Close prices
        volume: Volume
        high: High prices
        low: Low prices
        n_buckets: Number of buckets for rolling calculation

    Returns:
        VPIN series
    """
    calculator = VPIN(n_buckets=n_buckets)
    result = calculator.calculate(close, volume, high, low)
    return result.vpin


def detect_toxicity_alert(
    vpin: pd.Series,
    threshold: float = 0.7,
    lookback: int = 10,
) -> pd.Series:
    """
    Generate toxicity alerts based on VPIN.

    Args:
        vpin: VPIN series
        threshold: Alert threshold
        lookback: Lookback for rolling max

    Returns:
        Boolean series indicating alerts
    """
    rolling_max = vpin.rolling(lookback).max()
    return rolling_max > threshold
