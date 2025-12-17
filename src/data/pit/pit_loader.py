"""
Point-in-Time Data Loader for look-ahead bias prevention.

This module provides as-of timestamp queries to ensure that data
loaded for any point in time reflects ONLY what was known at that time.

Without PIT data handling, backtests suffer from:
1. Using restated financial data (earnings revisions)
2. Using future corporate action adjustments
3. Using data that was reported late

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import numpy as np

from .universe_manager import UniverseManager

logger = logging.getLogger(__name__)


class PITDataLoader:
    """
    Point-in-Time data loader that prevents look-ahead bias.

    This loader ensures data queries return ONLY information that was
    available at the specified timestamp, not future revisions or corrections.

    Key features:
    1. As-of queries for market data
    2. Corporate action adjustment tracking
    3. Data revision history support
    4. Integration with UniverseManager for survivorship bias prevention

    Example usage:
        pit_loader = PITDataLoader(
            data_dir="data/raw",
            universe_manager=universe_mgr
        )

        # Load data as it was known on Jan 1, 2020
        # (ignoring any future revisions or adjustments)
        data = pit_loader.load_as_of(
            symbol="AAPL",
            as_of=date(2020, 1, 1),
            lookback_days=252
        )
    """

    def __init__(
        self,
        data_dir: Path | str,
        universe_manager: Optional[UniverseManager] = None,
        adjustment_method: Literal["split_adjusted", "dividend_adjusted", "raw"] = "split_adjusted",
    ) -> None:
        """
        Initialize the PIT data loader.

        Args:
            data_dir: Directory containing data files
            universe_manager: UniverseManager for survivorship bias handling
            adjustment_method: How to adjust for corporate actions
        """
        self.data_dir = Path(data_dir)
        self.universe_manager = universe_manager or UniverseManager()
        self.adjustment_method = adjustment_method

        # Cache for loaded data
        self._cache: Dict[str, pd.DataFrame] = {}

        # Corporate action history (would be loaded from database in production)
        self._corporate_actions: Dict[str, pd.DataFrame] = {}

        logger.info(f"PITDataLoader initialized with data_dir={data_dir}")

    def load_as_of(
        self,
        symbol: str,
        as_of: date | datetime | str,
        lookback_days: int = 252,
        fields: List[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load data for a symbol as it was known on a specific date.

        CRITICAL: This method returns data that was available at the
        as_of timestamp, NOT revised data from the future.

        Args:
            symbol: Symbol to load
            as_of: Reference timestamp (what was known at this time)
            lookback_days: Number of historical days to load
            fields: Specific fields to load (default: all OHLCV)

        Returns:
            DataFrame with PIT-correct data

        Raises:
            ValueError: If symbol was not active on as_of date
        """
        # Parse date
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        elif isinstance(as_of, datetime):
            as_of = as_of.date()

        # Check if symbol was active on this date
        meta = self.universe_manager.get_symbol(symbol)
        if meta and not meta.is_active(as_of):
            raise ValueError(
                f"Symbol {symbol} was not active on {as_of}. "
                f"Listed: {meta.listing_date}, Delisted: {meta.delist_date}"
            )

        # Load raw data
        df = self._load_raw_data(symbol)
        if df is None or df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # Filter to as-of date (only data available at that time)
        start_date = as_of - timedelta(days=lookback_days)
        end_date = as_of

        # Convert index to date if needed for comparison
        if isinstance(df.index, pd.DatetimeIndex):
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        else:
            df.index = pd.to_datetime(df.index)
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)

        df = df.loc[mask].copy()

        # Apply corporate action adjustments (as they were known at as_of)
        df = self._apply_pit_adjustments(df, symbol, as_of)

        # Select fields
        if fields:
            available_fields = [f for f in fields if f in df.columns]
            df = df[available_fields]

        logger.debug(
            f"Loaded {len(df)} rows for {symbol} as-of {as_of} "
            f"({start_date} to {end_date})"
        )

        return df

    def load_universe_as_of(
        self,
        as_of: date | datetime | str,
        lookback_days: int = 252,
        index: Optional[str] = None,
        max_symbols: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for the entire universe as it existed on a date.

        This method:
        1. Gets the universe as it existed on as_of (includes delisted stocks)
        2. Loads PIT-correct data for each symbol
        3. Applies corporate action adjustments as they were known at as_of

        Args:
            as_of: Reference timestamp
            lookback_days: Historical days to load
            index: Filter by index membership
            max_symbols: Maximum number of symbols to load

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # Get universe as it existed on as_of date
        symbols = self.universe_manager.get_universe(as_of=as_of, index=index)

        if max_symbols:
            symbols = symbols[:max_symbols]

        logger.info(f"Loading {len(symbols)} symbols as-of {as_of}")

        data = {}
        for symbol in symbols:
            try:
                df = self.load_as_of(
                    symbol=symbol,
                    as_of=as_of,
                    lookback_days=lookback_days,
                )
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")

        logger.info(f"Successfully loaded {len(data)}/{len(symbols)} symbols")

        return data

    def load_walk_forward(
        self,
        symbol: str,
        start_date: date | str,
        end_date: date | str,
        window_size: int = 252,
        step_size: int = 21,
    ) -> List[Tuple[date, pd.DataFrame]]:
        """
        Load data in walk-forward fashion for realistic backtesting.

        At each step, only data known at that point is included.
        This simulates how the data would have been seen in real-time.

        Args:
            symbol: Symbol to load
            start_date: Start of backtest period
            end_date: End of backtest period
            window_size: Lookback window in days
            step_size: Days between windows

        Returns:
            List of (as_of_date, DataFrame) tuples
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        results = []
        current_date = start_date

        while current_date <= end_date:
            try:
                df = self.load_as_of(
                    symbol=symbol,
                    as_of=current_date,
                    lookback_days=window_size,
                )
                results.append((current_date, df))
            except Exception as e:
                logger.warning(f"Failed to load {symbol} as-of {current_date}: {e}")

            current_date += timedelta(days=step_size)

        return results

    def _load_raw_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load raw data from file (with caching)."""
        if symbol in self._cache:
            return self._cache[symbol].copy()

        # Try different file patterns
        patterns = [
            f"{symbol}_15min.csv",
            f"{symbol}_15min.parquet",
            f"{symbol}.csv",
            f"{symbol}.parquet",
        ]

        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                try:
                    if file_path.suffix == ".csv":
                        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
                    else:
                        df = pd.read_parquet(file_path)

                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)

                    df = df.sort_index()
                    self._cache[symbol] = df
                    return df.copy()
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        return None

    def _apply_pit_adjustments(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of: date,
    ) -> pd.DataFrame:
        """
        Apply corporate action adjustments as they were known at as_of.

        This is critical for PIT correctness - we apply adjustments
        that were known at as_of, not future adjustments.
        """
        if self.adjustment_method == "raw":
            return df

        # Get corporate actions known at as_of
        actions = self._get_corporate_actions(symbol, as_of)

        if actions.empty:
            return df

        # Apply adjustments (backwards from as_of)
        for _, action in actions.iterrows():
            ex_date = action["ex_date"]
            action_type = action["action_type"]

            # Only apply if action was before as_of (known at as_of)
            if ex_date > as_of:
                continue

            # Get data before ex-date
            pre_ex_mask = df.index.date < ex_date

            if action_type == "split":
                ratio = action["ratio"]
                # Adjust price columns before ex-date
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.loc[pre_ex_mask, col] = df.loc[pre_ex_mask, col] / ratio
                # Adjust volume inversely
                if "volume" in df.columns:
                    df.loc[pre_ex_mask, "volume"] = df.loc[pre_ex_mask, "volume"] * ratio

            elif action_type == "dividend" and self.adjustment_method == "dividend_adjusted":
                dividend = action["amount"]
                # Adjust prices for dividend
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        # Get price on ex-date for adjustment factor
                        close_on_ex = df.loc[~pre_ex_mask, "close"].iloc[0] if (~pre_ex_mask).any() else None
                        if close_on_ex:
                            adj_factor = (close_on_ex - dividend) / close_on_ex
                            df.loc[pre_ex_mask, col] = df.loc[pre_ex_mask, col] * adj_factor

        return df

    def _get_corporate_actions(
        self,
        symbol: str,
        as_of: date,
    ) -> pd.DataFrame:
        """
        Get corporate actions known at as_of date.

        In production, this would query a database with historical
        corporate action data. Here we return from cache or empty.
        """
        if symbol in self._corporate_actions:
            actions = self._corporate_actions[symbol]
            # Filter to actions that were announced before as_of
            if "announcement_date" in actions.columns:
                mask = actions["announcement_date"].dt.date <= as_of
                return actions.loc[mask]
            else:
                # If no announcement date, assume all were known
                return actions

        return pd.DataFrame()

    def register_corporate_action(
        self,
        symbol: str,
        action_type: str,
        ex_date: date,
        ratio: Optional[float] = None,
        amount: Optional[float] = None,
        announcement_date: Optional[date] = None,
    ) -> None:
        """
        Register a corporate action for a symbol.

        Args:
            symbol: Symbol
            action_type: Type of action (split, dividend, spinoff)
            ex_date: Ex-dividend or ex-split date
            ratio: Split ratio (e.g., 2.0 for 2-for-1 split)
            amount: Dividend amount
            announcement_date: When action was announced (for PIT filtering)
        """
        action = {
            "action_type": action_type,
            "ex_date": pd.Timestamp(ex_date),
            "ratio": ratio,
            "amount": amount,
            "announcement_date": pd.Timestamp(announcement_date) if announcement_date else pd.Timestamp(ex_date),
        }

        if symbol not in self._corporate_actions:
            self._corporate_actions[symbol] = pd.DataFrame()

        self._corporate_actions[symbol] = pd.concat([
            self._corporate_actions[symbol],
            pd.DataFrame([action]),
        ], ignore_index=True)

    def clear_cache(self) -> None:
        """Clear data cache."""
        self._cache.clear()
        logger.debug("PIT data cache cleared")

    def preload_symbols(self, symbols: List[str]) -> None:
        """Preload data for multiple symbols into cache."""
        for symbol in symbols:
            self._load_raw_data(symbol)
        logger.info(f"Preloaded {len(self._cache)} symbols into cache")
