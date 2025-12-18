"""
ML-Based Position Sizing Module.

JPMorgan Institutional-Level Position Sizing using ML Predictions.

This module provides sophisticated position sizing strategies that
combine ML model predictions with risk management principles.

Key Features:
- Kelly Criterion sizing
- Confidence-based sizing
- Meta-labeling integration
- Dynamic leverage adjustment
- Risk-adjusted position scaling

Reference:
    "Advances in Financial Machine Learning" by de Prado (2018)
    Chapter 10: Bet Sizing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    CONFIDENCE = "confidence"
    META_LABEL = "meta_label"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    size: float  # Position size (fraction of capital)
    leverage: float  # Effective leverage
    confidence: float  # Sizing confidence
    method: SizingMethod
    details: Dict[str, Any]


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    The Kelly Criterion determines the optimal fraction of capital
    to risk on a trade given the probability of winning and the
    payoff ratio.

    Formula: f* = (p * b - q) / b
    Where:
        f* = optimal fraction
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (average win / average loss)
    """

    def __init__(
        self,
        max_position: float = 0.25,
        kelly_fraction: float = 0.5,
        min_probability: float = 0.5,
    ):
        """
        Initialize Kelly calculator.

        Args:
            max_position: Maximum position size (fraction of capital)
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
            min_probability: Minimum win probability to take position
        """
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
        self.min_probability = min_probability

    def calculate(
        self,
        win_probability: float,
        win_loss_ratio: float = 1.0,
        side: int = 1,
    ) -> PositionSizeResult:
        """
        Calculate Kelly-optimal position size.

        Args:
            win_probability: Probability of trade being profitable
            win_loss_ratio: Average win / average loss
            side: Trade side (1=long, -1=short)

        Returns:
            PositionSizeResult with optimal size
        """
        if win_probability < self.min_probability:
            return PositionSizeResult(
                size=0.0,
                leverage=0.0,
                confidence=win_probability,
                method=SizingMethod.KELLY,
                details={"reason": "probability_below_threshold"}
            )

        # Kelly formula
        p = win_probability
        q = 1 - p
        b = win_loss_ratio

        kelly_full = (p * b - q) / b if b > 0 else 0

        # Apply Kelly fraction (half Kelly is common)
        kelly_adjusted = kelly_full * self.kelly_fraction

        # Clip to max position
        position_size = np.clip(kelly_adjusted, -self.max_position, self.max_position)

        # Apply side
        position_size = position_size * side

        return PositionSizeResult(
            size=position_size,
            leverage=abs(position_size),
            confidence=win_probability,
            method=SizingMethod.KELLY,
            details={
                "kelly_full": kelly_full,
                "kelly_adjusted": kelly_adjusted,
                "win_probability": p,
                "win_loss_ratio": b,
            }
        )


class ConfidenceSizer:
    """
    Position sizing based on model prediction confidence.

    Scales position size linearly or non-linearly with
    prediction probability/confidence.
    """

    def __init__(
        self,
        max_position: float = 0.25,
        min_confidence: float = 0.5,
        scaling: str = "linear",
    ):
        """
        Initialize confidence-based sizer.

        Args:
            max_position: Maximum position size
            min_confidence: Minimum confidence to take position
            scaling: "linear" or "sigmoid" scaling
        """
        self.max_position = max_position
        self.min_confidence = min_confidence
        self.scaling = scaling

    def calculate(
        self,
        probability: float,
        side: int = 1,
    ) -> PositionSizeResult:
        """
        Calculate position size based on confidence.

        Args:
            probability: Model prediction probability
            side: Trade side (1=long, -1=short)

        Returns:
            PositionSizeResult with confidence-scaled size
        """
        if probability < self.min_confidence:
            return PositionSizeResult(
                size=0.0,
                leverage=0.0,
                confidence=probability,
                method=SizingMethod.CONFIDENCE,
                details={"reason": "confidence_below_threshold"}
            )

        # Calculate confidence factor (0 to 1)
        confidence_range = 1.0 - self.min_confidence
        confidence_factor = (probability - self.min_confidence) / confidence_range

        if self.scaling == "sigmoid":
            # S-curve scaling for smoother transition
            confidence_factor = 1 / (1 + np.exp(-6 * (confidence_factor - 0.5)))
        elif self.scaling == "quadratic":
            # Quadratic scaling (more conservative)
            confidence_factor = confidence_factor ** 2

        position_size = confidence_factor * self.max_position * side

        return PositionSizeResult(
            size=position_size,
            leverage=abs(position_size),
            confidence=probability,
            method=SizingMethod.CONFIDENCE,
            details={
                "confidence_factor": confidence_factor,
                "scaling": self.scaling,
            }
        )


class MetaLabelSizer:
    """
    Position sizing using meta-labeling predictions.

    Uses a secondary model's prediction of whether the primary
    model will be correct to size positions.
    """

    def __init__(
        self,
        max_position: float = 0.25,
        min_meta_probability: float = 0.5,
        use_kelly: bool = True,
    ):
        """
        Initialize meta-label sizer.

        Args:
            max_position: Maximum position size
            min_meta_probability: Minimum meta-model probability
            use_kelly: Use Kelly criterion for final sizing
        """
        self.max_position = max_position
        self.min_meta_probability = min_meta_probability
        self.use_kelly = use_kelly
        self.kelly = KellyCriterion(max_position=max_position)

    def calculate(
        self,
        primary_side: int,
        meta_probability: float,
        primary_probability: Optional[float] = None,
        win_loss_ratio: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate position size using meta-labeling.

        Args:
            primary_side: Primary model's predicted side (1=long, -1=short)
            meta_probability: Meta-model's probability of primary being correct
            primary_probability: Primary model's probability (optional)
            win_loss_ratio: Historical win/loss ratio

        Returns:
            PositionSizeResult
        """
        if meta_probability < self.min_meta_probability:
            return PositionSizeResult(
                size=0.0,
                leverage=0.0,
                confidence=meta_probability,
                method=SizingMethod.META_LABEL,
                details={"reason": "meta_probability_below_threshold"}
            )

        if self.use_kelly:
            result = self.kelly.calculate(
                win_probability=meta_probability,
                win_loss_ratio=win_loss_ratio,
                side=primary_side,
            )
            result.method = SizingMethod.META_LABEL
            result.details["primary_side"] = primary_side
            result.details["meta_probability"] = meta_probability
            return result

        # Simple linear scaling
        size = (meta_probability - 0.5) * 2 * self.max_position * primary_side

        return PositionSizeResult(
            size=size,
            leverage=abs(size),
            confidence=meta_probability,
            method=SizingMethod.META_LABEL,
            details={
                "primary_side": primary_side,
                "meta_probability": meta_probability,
            }
        )


class VolatilityAdjustedSizer:
    """
    Position sizing adjusted for current market volatility.

    Reduces position size during high volatility periods
    and increases during low volatility.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        max_position: float = 0.25,
        volatility_lookback: int = 20,
    ):
        """
        Initialize volatility-adjusted sizer.

        Args:
            target_volatility: Target annualized volatility
            max_position: Maximum position size
            volatility_lookback: Lookback period for volatility calculation
        """
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.volatility_lookback = volatility_lookback

    def calculate(
        self,
        base_size: float,
        current_volatility: float,
        side: int = 1,
    ) -> PositionSizeResult:
        """
        Adjust position size for volatility.

        Args:
            base_size: Base position size from primary sizing method
            current_volatility: Current annualized volatility
            side: Trade side

        Returns:
            Volatility-adjusted position size
        """
        if current_volatility <= 0:
            return PositionSizeResult(
                size=base_size * side,
                leverage=abs(base_size),
                confidence=1.0,
                method=SizingMethod.VOLATILITY_ADJUSTED,
                details={"error": "invalid_volatility"}
            )

        # Volatility scalar
        vol_scalar = self.target_volatility / current_volatility
        vol_scalar = np.clip(vol_scalar, 0.25, 2.0)  # Limit scaling

        adjusted_size = base_size * vol_scalar * side
        adjusted_size = np.clip(adjusted_size, -self.max_position, self.max_position)

        return PositionSizeResult(
            size=adjusted_size,
            leverage=abs(adjusted_size),
            confidence=vol_scalar,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            details={
                "base_size": base_size,
                "vol_scalar": vol_scalar,
                "current_volatility": current_volatility,
                "target_volatility": self.target_volatility,
            }
        )


class MLPositionSizer:
    """
    Comprehensive ML-based position sizing.

    Combines multiple sizing strategies with ML predictions
    for optimal position sizing.

    Example:
        sizer = MLPositionSizer(max_position=0.25)

        # Using model predictions
        size = sizer.size_position(
            probability=0.65,
            side=1,
            meta_probability=0.7,
            volatility=0.20,
        )
    """

    def __init__(
        self,
        max_position: float = 0.25,
        default_method: SizingMethod = SizingMethod.META_LABEL,
        kelly_fraction: float = 0.5,
        min_confidence: float = 0.5,
        target_volatility: float = 0.15,
    ):
        """
        Initialize the ML position sizer.

        Args:
            max_position: Maximum position size (fraction of capital)
            default_method: Default sizing method
            kelly_fraction: Fraction of Kelly to use
            min_confidence: Minimum confidence for trading
            target_volatility: Target portfolio volatility
        """
        self.max_position = max_position
        self.default_method = default_method
        self.min_confidence = min_confidence

        # Initialize sub-sizers
        self.kelly = KellyCriterion(
            max_position=max_position,
            kelly_fraction=kelly_fraction,
        )
        self.confidence_sizer = ConfidenceSizer(
            max_position=max_position,
            min_confidence=min_confidence,
        )
        self.meta_sizer = MetaLabelSizer(
            max_position=max_position,
            min_meta_probability=min_confidence,
        )
        self.vol_sizer = VolatilityAdjustedSizer(
            target_volatility=target_volatility,
            max_position=max_position,
        )

    def size_position(
        self,
        probability: float,
        side: int = 1,
        meta_probability: Optional[float] = None,
        volatility: Optional[float] = None,
        win_loss_ratio: float = 1.0,
        method: Optional[SizingMethod] = None,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            probability: Primary model prediction probability
            side: Trade side (1=long, -1=short)
            meta_probability: Meta-model probability (if using meta-labeling)
            volatility: Current market volatility (annualized)
            win_loss_ratio: Historical win/loss ratio
            method: Specific sizing method to use

        Returns:
            PositionSizeResult with recommended size
        """
        method = method or self.default_method

        # Get base size from primary method
        if method == SizingMethod.KELLY:
            result = self.kelly.calculate(probability, win_loss_ratio, side)

        elif method == SizingMethod.CONFIDENCE:
            result = self.confidence_sizer.calculate(probability, side)

        elif method == SizingMethod.META_LABEL and meta_probability is not None:
            result = self.meta_sizer.calculate(
                primary_side=side,
                meta_probability=meta_probability,
                primary_probability=probability,
                win_loss_ratio=win_loss_ratio,
            )

        else:
            # Default to confidence-based
            result = self.confidence_sizer.calculate(probability, side)

        # Apply volatility adjustment if available
        if volatility is not None and volatility > 0:
            vol_result = self.vol_sizer.calculate(
                base_size=abs(result.size),
                current_volatility=volatility,
                side=np.sign(result.size) if result.size != 0 else side,
            )
            result.size = vol_result.size
            result.leverage = vol_result.leverage
            result.details["volatility_adjusted"] = True
            result.details["vol_scalar"] = vol_result.details.get("vol_scalar", 1.0)

        return result

    def size_portfolio(
        self,
        signals: pd.DataFrame,
        volatilities: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate position sizes for a portfolio of signals.

        Args:
            signals: DataFrame with columns: symbol, side, probability, [meta_probability]
            volatilities: Series of volatilities by symbol

        Returns:
            DataFrame with position sizes added
        """
        results = signals.copy()
        sizes = []

        for idx, row in signals.iterrows():
            vol = volatilities.get(row.get("symbol")) if volatilities is not None else None
            meta_prob = row.get("meta_probability")

            result = self.size_position(
                probability=row["probability"],
                side=row["side"],
                meta_probability=meta_prob,
                volatility=vol,
            )
            sizes.append(result.size)

        results["position_size"] = sizes
        return results
