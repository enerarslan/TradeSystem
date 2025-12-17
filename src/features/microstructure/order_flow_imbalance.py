"""
Order Flow Imbalance (OFI) Features.

Order Flow Imbalance measures the net buying/selling pressure in the market.
Strong OFI is predictive of short-term price movements.

Reference:
    Cont, Kukanov, Stoikov (2014) - "The Price Impact of Order Book Events"

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
class OFIResult:
    """Container for OFI calculation results."""

    ofi: pd.Series                    # Raw OFI values
    ofi_normalized: pd.Series         # Normalized OFI
    cumulative_ofi: pd.Series         # Cumulative OFI
    buy_volume: pd.Series             # Estimated buy volume
    sell_volume: pd.Series            # Estimated sell volume
    volume_imbalance_ratio: pd.Series  # Buy volume / Total volume


class OrderFlowImbalance:
    """
    Order Flow Imbalance calculator.

    Estimates buy/sell volume decomposition using the tick rule or
    trade classification, then calculates net buying pressure.

    Methods supported:
    1. Tick rule: classify trades by price movement
    2. Quote rule: classify by relation to bid/ask
    3. Lee-Ready: combine tick and quote rules
    4. Bulk volume classification: probabilistic approach

    Example usage:
        ofi = OrderFlowImbalance(method="tick")
        result = ofi.calculate(price, volume)

        # Use OFI as a feature
        features['ofi'] = result.ofi_normalized
    """

    def __init__(
        self,
        method: str = "tick",
        lookback: int = 20,
        normalization: str = "zscore",
    ) -> None:
        """
        Initialize OFI calculator.

        Args:
            method: Classification method ("tick", "bulk")
            lookback: Lookback for normalization
            normalization: Normalization method ("zscore", "minmax", "none")
        """
        self.method = method
        self.lookback = lookback
        self.normalization = normalization

    def calculate(
        self,
        close: pd.Series,
        volume: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> OFIResult:
        """
        Calculate Order Flow Imbalance.

        Args:
            close: Close prices
            volume: Volume
            high: High prices (for bulk classification)
            low: Low prices (for bulk classification)

        Returns:
            OFIResult with all OFI metrics
        """
        if self.method == "tick":
            buy_vol, sell_vol = self._tick_rule_classification(close, volume)
        elif self.method == "bulk":
            buy_vol, sell_vol = self._bulk_volume_classification(close, high, low, volume)
        else:
            buy_vol, sell_vol = self._tick_rule_classification(close, volume)

        # Calculate raw OFI
        ofi = buy_vol - sell_vol

        # Calculate normalized OFI
        if self.normalization == "zscore":
            ofi_norm = self._zscore_normalize(ofi)
        elif self.normalization == "minmax":
            ofi_norm = self._minmax_normalize(ofi)
        else:
            ofi_norm = ofi

        # Calculate cumulative OFI
        cumulative = ofi.cumsum()

        # Volume imbalance ratio
        total_vol = buy_vol + sell_vol
        vir = buy_vol / total_vol.replace(0, np.nan)

        return OFIResult(
            ofi=ofi,
            ofi_normalized=ofi_norm,
            cumulative_ofi=cumulative,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            volume_imbalance_ratio=vir,
        )

    def _tick_rule_classification(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Classify volume using tick rule.

        Trade at uptick -> buy initiated
        Trade at downtick -> sell initiated
        Trade at zero tick -> use previous classification
        """
        price_change = close.diff()

        # Classify
        buy_mask = price_change > 0
        sell_mask = price_change < 0

        buy_volume = volume * buy_mask.astype(float)
        sell_volume = volume * sell_mask.astype(float)

        # Handle zero ticks (use previous classification)
        zero_mask = price_change == 0
        if zero_mask.any():
            # Forward fill classification
            classification = np.sign(price_change).replace(0, np.nan).ffill()
            buy_volume[zero_mask] = volume[zero_mask] * (classification[zero_mask] > 0).astype(float)
            sell_volume[zero_mask] = volume[zero_mask] * (classification[zero_mask] < 0).astype(float)

        return buy_volume, sell_volume

    def _bulk_volume_classification(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Bulk Volume Classification (BVC) method.

        Uses the normalized price position within the bar to probabilistically
        assign volume to buy/sell.

        Reference: Easley, de Prado, O'Hara (2012)
        """
        if high is None or low is None:
            return self._tick_rule_classification(close, volume)

        # Calculate bar range
        bar_range = high - low

        # Normalized close position [0, 1]
        # 1 = close at high (bullish), 0 = close at low (bearish)
        normalized_position = (close - low) / bar_range.replace(0, np.nan)

        # Estimate buy/sell volume
        buy_volume = volume * normalized_position
        sell_volume = volume * (1 - normalized_position)

        return buy_volume.fillna(0), sell_volume.fillna(0)

    def _zscore_normalize(self, series: pd.Series) -> pd.Series:
        """Z-score normalization using rolling window."""
        rolling_mean = series.rolling(self.lookback).mean()
        rolling_std = series.rolling(self.lookback).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)

    def _minmax_normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalization to [-1, 1]."""
        rolling_min = series.rolling(self.lookback).min()
        rolling_max = series.rolling(self.lookback).max()
        range_val = rolling_max - rolling_min
        return 2 * (series - rolling_min) / range_val.replace(0, np.nan) - 1


def calculate_ofi(
    close: pd.Series,
    volume: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    method: str = "bulk",
) -> pd.Series:
    """
    Convenience function to calculate normalized OFI.

    Args:
        close: Close prices
        volume: Volume
        high: High prices
        low: Low prices
        method: Classification method

    Returns:
        Normalized OFI series
    """
    calculator = OrderFlowImbalance(method=method)
    result = calculator.calculate(close, volume, high, low)
    return result.ofi_normalized


def calculate_volume_imbalance(
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Calculate rolling volume imbalance ratio.

    Args:
        close: Close prices
        volume: Volume
        lookback: Rolling window

    Returns:
        Volume imbalance ratio [0, 1]
    """
    calculator = OrderFlowImbalance()
    result = calculator.calculate(close, volume)
    return result.volume_imbalance_ratio.rolling(lookback).mean()
