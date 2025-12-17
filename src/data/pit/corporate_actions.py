"""
Corporate Action Adjustment Engine.

This module handles price adjustments for splits, dividends, spinoffs,
and other corporate actions to ensure data consistency.

Without proper corporate action handling:
- Stock splits cause artificial price jumps
- Dividends distort total return calculations
- Spinoffs create phantom gains/losses

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of corporate actions."""

    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    RIGHTS_ISSUE = "rights_issue"
    STOCK_DIVIDEND = "stock_dividend"
    NAME_CHANGE = "name_change"


@dataclass
class CorporateAction:
    """
    Represents a corporate action event.

    Attributes:
        symbol: Affected symbol
        action_type: Type of corporate action
        ex_date: Ex-date (when adjustment takes effect)
        record_date: Record date
        announcement_date: When action was announced
        ratio: Adjustment ratio (for splits)
        amount: Cash amount (for dividends)
        new_symbol: New symbol (for name changes/spinoffs)
        notes: Additional information
    """

    symbol: str
    action_type: ActionType
    ex_date: date
    record_date: Optional[date] = None
    announcement_date: Optional[date] = None
    ratio: float = 1.0
    amount: float = 0.0
    new_symbol: Optional[str] = None
    notes: str = ""

    @property
    def adjustment_factor(self) -> float:
        """
        Calculate the price adjustment factor.

        For splits: prices before ex-date are divided by this factor
        For dividends: factor adjusts for cash distribution
        """
        if self.action_type in (ActionType.SPLIT, ActionType.STOCK_DIVIDEND):
            return self.ratio
        elif self.action_type == ActionType.REVERSE_SPLIT:
            return 1.0 / self.ratio
        elif self.action_type in (ActionType.DIVIDEND, ActionType.SPECIAL_DIVIDEND):
            # Will be calculated based on price
            return 1.0
        else:
            return 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "action_type": self.action_type.value,
            "ex_date": self.ex_date.isoformat(),
            "record_date": self.record_date.isoformat() if self.record_date else None,
            "announcement_date": self.announcement_date.isoformat() if self.announcement_date else None,
            "ratio": self.ratio,
            "amount": self.amount,
            "new_symbol": self.new_symbol,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CorporateAction":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            action_type=ActionType(data["action_type"]),
            ex_date=date.fromisoformat(data["ex_date"]),
            record_date=date.fromisoformat(data["record_date"]) if data.get("record_date") else None,
            announcement_date=date.fromisoformat(data["announcement_date"]) if data.get("announcement_date") else None,
            ratio=data.get("ratio", 1.0),
            amount=data.get("amount", 0.0),
            new_symbol=data.get("new_symbol"),
            notes=data.get("notes", ""),
        )


class CorporateActionAdjuster:
    """
    Adjusts price data for corporate actions.

    This class provides methods to:
    1. Apply adjustments to historical data
    2. Track adjustment history for audit trails
    3. Support point-in-time correct adjustments

    Example usage:
        adjuster = CorporateActionAdjuster()

        # Register a stock split
        adjuster.register_action(CorporateAction(
            symbol="AAPL",
            action_type=ActionType.SPLIT,
            ex_date=date(2020, 8, 31),
            ratio=4.0  # 4-for-1 split
        ))

        # Apply adjustments to data
        adjusted_df = adjuster.adjust(df, symbol="AAPL")
    """

    def __init__(self) -> None:
        """Initialize the adjuster."""
        self._actions: Dict[str, List[CorporateAction]] = {}
        self._adjustment_log: List[Dict] = []

    def register_action(self, action: CorporateAction) -> None:
        """
        Register a corporate action.

        Args:
            action: Corporate action to register
        """
        if action.symbol not in self._actions:
            self._actions[action.symbol] = []

        self._actions[action.symbol].append(action)

        # Keep sorted by ex_date
        self._actions[action.symbol].sort(key=lambda a: a.ex_date)

        logger.debug(
            f"Registered {action.action_type.value} for {action.symbol} "
            f"ex-date {action.ex_date}"
        )

    def get_actions(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        action_types: Optional[List[ActionType]] = None,
    ) -> List[CorporateAction]:
        """
        Get corporate actions for a symbol.

        Args:
            symbol: Symbol to query
            start_date: Filter by start date
            end_date: Filter by end date
            action_types: Filter by action types

        Returns:
            List of matching corporate actions
        """
        actions = self._actions.get(symbol, [])

        if start_date:
            actions = [a for a in actions if a.ex_date >= start_date]
        if end_date:
            actions = [a for a in actions if a.ex_date <= end_date]
        if action_types:
            actions = [a for a in actions if a.action_type in action_types]

        return actions

    def adjust(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of: Optional[date] = None,
        method: str = "backward",
        price_columns: List[str] = ["open", "high", "low", "close"],
        volume_column: str = "volume",
    ) -> pd.DataFrame:
        """
        Apply corporate action adjustments to data.

        Args:
            df: OHLCV DataFrame to adjust
            symbol: Symbol for adjustment lookup
            as_of: Only apply adjustments known at this date (PIT)
            method: "backward" (adjust historical) or "forward" (adjust future)
            price_columns: Columns to adjust for price
            volume_column: Column to adjust for volume

        Returns:
            Adjusted DataFrame
        """
        if df.empty:
            return df

        df = df.copy()
        actions = self.get_actions(symbol)

        if not actions:
            return df

        # Filter for PIT correctness
        if as_of:
            actions = [
                a for a in actions
                if a.announcement_date is None or a.announcement_date <= as_of
            ]

        # Apply each action
        for action in actions:
            df = self._apply_single_action(
                df=df,
                action=action,
                method=method,
                price_columns=price_columns,
                volume_column=volume_column,
            )

        return df

    def _apply_single_action(
        self,
        df: pd.DataFrame,
        action: CorporateAction,
        method: str,
        price_columns: List[str],
        volume_column: str,
    ) -> pd.DataFrame:
        """Apply a single corporate action adjustment."""
        ex_date = pd.Timestamp(action.ex_date)

        # Determine which rows to adjust
        if method == "backward":
            # Adjust prices BEFORE ex-date
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index < ex_date
            else:
                mask = pd.to_datetime(df.index) < ex_date
        else:
            # Adjust prices ON OR AFTER ex-date
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index >= ex_date
            else:
                mask = pd.to_datetime(df.index) >= ex_date

        # Calculate adjustment factor
        if action.action_type in (ActionType.SPLIT, ActionType.STOCK_DIVIDEND):
            price_factor = 1.0 / action.ratio if method == "backward" else action.ratio
            volume_factor = action.ratio if method == "backward" else 1.0 / action.ratio

        elif action.action_type == ActionType.REVERSE_SPLIT:
            price_factor = action.ratio if method == "backward" else 1.0 / action.ratio
            volume_factor = 1.0 / action.ratio if method == "backward" else action.ratio

        elif action.action_type in (ActionType.DIVIDEND, ActionType.SPECIAL_DIVIDEND):
            # For dividends, adjust based on dividend yield
            # Find price just before ex-date
            pre_ex_data = df.loc[~mask]
            if not pre_ex_data.empty and "close" in df.columns:
                close_before_ex = pre_ex_data["close"].iloc[-1]
                # Adjustment factor = (price - dividend) / price
                price_factor = (close_before_ex - action.amount) / close_before_ex
            else:
                price_factor = 1.0
            volume_factor = 1.0

        else:
            price_factor = 1.0
            volume_factor = 1.0

        # Apply adjustments
        for col in price_columns:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] * price_factor

        if volume_column in df.columns:
            df.loc[mask, volume_column] = df.loc[mask, volume_column] * volume_factor

        # Log adjustment
        self._adjustment_log.append({
            "symbol": action.symbol,
            "action_type": action.action_type.value,
            "ex_date": action.ex_date,
            "price_factor": price_factor,
            "volume_factor": volume_factor,
            "rows_adjusted": mask.sum(),
            "method": method,
        })

        logger.debug(
            f"Applied {action.action_type.value} adjustment for {action.symbol}: "
            f"price_factor={price_factor:.6f}, rows={mask.sum()}"
        )

        return df

    def calculate_total_adjustment(
        self,
        symbol: str,
        as_of: date,
        reference_date: Optional[date] = None,
    ) -> float:
        """
        Calculate cumulative adjustment factor from reference_date to as_of.

        This is useful for converting between adjusted and unadjusted prices.

        Args:
            symbol: Symbol to calculate for
            as_of: Target date
            reference_date: Starting date (default: earliest action)

        Returns:
            Cumulative adjustment factor
        """
        actions = self.get_actions(symbol)

        if not actions:
            return 1.0

        cumulative_factor = 1.0

        for action in actions:
            if reference_date and action.ex_date < reference_date:
                continue
            if action.ex_date > as_of:
                continue

            cumulative_factor *= action.adjustment_factor

        return cumulative_factor

    def get_adjustment_log(self) -> pd.DataFrame:
        """Get log of all adjustments applied."""
        return pd.DataFrame(self._adjustment_log)

    def clear_log(self) -> None:
        """Clear adjustment log."""
        self._adjustment_log.clear()

    def load_from_csv(self, path: str) -> None:
        """
        Load corporate actions from CSV file.

        Expected columns:
        - symbol
        - action_type (split, dividend, etc.)
        - ex_date
        - ratio (for splits)
        - amount (for dividends)
        """
        df = pd.read_csv(path, parse_dates=["ex_date", "record_date", "announcement_date"])

        for _, row in df.iterrows():
            action = CorporateAction(
                symbol=row["symbol"],
                action_type=ActionType(row["action_type"]),
                ex_date=row["ex_date"].date() if pd.notna(row["ex_date"]) else None,
                record_date=row["record_date"].date() if pd.notna(row.get("record_date")) else None,
                announcement_date=row["announcement_date"].date() if pd.notna(row.get("announcement_date")) else None,
                ratio=row.get("ratio", 1.0),
                amount=row.get("amount", 0.0),
                new_symbol=row.get("new_symbol"),
                notes=row.get("notes", ""),
            )
            self.register_action(action)

        logger.info(f"Loaded corporate actions from {path}")

    @property
    def symbols_with_actions(self) -> List[str]:
        """Get list of symbols with registered actions."""
        return list(self._actions.keys())


def create_sample_corporate_actions() -> CorporateActionAdjuster:
    """Create adjuster with sample corporate actions for testing."""
    adjuster = CorporateActionAdjuster()

    # AAPL 4-for-1 split (2020)
    adjuster.register_action(
        CorporateAction(
            symbol="AAPL",
            action_type=ActionType.SPLIT,
            ex_date=date(2020, 8, 31),
            announcement_date=date(2020, 7, 30),
            ratio=4.0,
            notes="4-for-1 stock split",
        )
    )

    # AAPL 7-for-1 split (2014)
    adjuster.register_action(
        CorporateAction(
            symbol="AAPL",
            action_type=ActionType.SPLIT,
            ex_date=date(2014, 6, 9),
            announcement_date=date(2014, 4, 23),
            ratio=7.0,
            notes="7-for-1 stock split",
        )
    )

    # TSLA 5-for-1 split (2020)
    adjuster.register_action(
        CorporateAction(
            symbol="TSLA",
            action_type=ActionType.SPLIT,
            ex_date=date(2020, 8, 31),
            announcement_date=date(2020, 8, 11),
            ratio=5.0,
            notes="5-for-1 stock split",
        )
    )

    # TSLA 3-for-1 split (2022)
    adjuster.register_action(
        CorporateAction(
            symbol="TSLA",
            action_type=ActionType.SPLIT,
            ex_date=date(2022, 8, 25),
            announcement_date=date(2022, 8, 4),
            ratio=3.0,
            notes="3-for-1 stock split",
        )
    )

    # Sample dividend
    adjuster.register_action(
        CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            ex_date=date(2023, 11, 10),
            announcement_date=date(2023, 11, 2),
            amount=0.24,
            notes="Quarterly dividend",
        )
    )

    return adjuster
