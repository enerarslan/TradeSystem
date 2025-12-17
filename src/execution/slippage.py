"""
Slippage modeling for AlphaTrade system.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class SlippageModel:
    """
    Slippage model for realistic execution simulation.

    Supports:
    - Fixed percentage slippage
    - Volatility-based slippage
    - Volume-based market impact
    """

    def __init__(
        self,
        model_type: Literal["fixed", "volatility", "volume"] = "fixed",
        fixed_pct: float = 0.0005,
        vol_multiplier: float = 0.1,
        impact_power: float = 0.5,
    ) -> None:
        """
        Initialize slippage model.

        Args:
            model_type: Type of slippage model
            fixed_pct: Fixed slippage percentage
            vol_multiplier: Volatility multiplier
            impact_power: Power for market impact model
        """
        self.model_type = model_type
        self.fixed_pct = fixed_pct
        self.vol_multiplier = vol_multiplier
        self.impact_power = impact_power

    def calculate(
        self,
        price: float,
        quantity: float,
        side: str,
        volatility: float | None = None,
        avg_volume: float | None = None,
    ) -> float:
        """
        Calculate slippage.

        Args:
            price: Current price
            quantity: Order quantity
            side: BUY or SELL
            volatility: Recent volatility
            avg_volume: Average daily volume

        Returns:
            Slippage amount (positive = cost)
        """
        direction = 1 if side.upper() == "BUY" else -1

        if self.model_type == "fixed":
            slippage_pct = self.fixed_pct

        elif self.model_type == "volatility":
            if volatility is None:
                volatility = 0.02  # Default 2% daily vol
            slippage_pct = volatility * self.vol_multiplier

        elif self.model_type == "volume":
            if avg_volume is None or avg_volume == 0:
                slippage_pct = self.fixed_pct
            else:
                participation = (quantity * price) / (avg_volume * price)
                slippage_pct = 0.001 * (participation ** self.impact_power)

        else:
            slippage_pct = self.fixed_pct

        return price * slippage_pct * direction

    def get_execution_price(
        self,
        price: float,
        quantity: float,
        side: str,
        **kwargs,
    ) -> float:
        """
        Get expected execution price after slippage.

        Args:
            price: Reference price
            quantity: Order quantity
            side: BUY or SELL
            **kwargs: Additional parameters

        Returns:
            Expected execution price
        """
        slippage = self.calculate(price, quantity, side, **kwargs)
        return price + slippage


def calculate_slippage(
    price: float,
    quantity: float,
    side: str,
    pct: float = 0.0005,
) -> float:
    """
    Convenience function to calculate slippage.

    Args:
        price: Current price
        quantity: Order quantity
        side: BUY or SELL
        pct: Slippage percentage

    Returns:
        Slippage amount
    """
    model = SlippageModel(model_type="fixed", fixed_pct=pct)
    return model.calculate(price, quantity, side)
