"""
As-Of Query Engine for point-in-time data access.

This module provides a query interface that ensures all data retrieval
reflects only what was known at a specific timestamp.

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .universe_manager import UniverseManager
from .pit_loader import PITDataLoader
from .corporate_actions import CorporateActionAdjuster

logger = logging.getLogger(__name__)


class AsOfQueryEngine:
    """
    Query engine that enforces point-in-time data access.

    This engine coordinates all data components to ensure queries
    return only information that was available at the specified time.

    Key features:
    1. Unified query interface for all data types
    2. Automatic survivorship bias prevention
    3. Corporate action adjustment
    4. Data revision tracking

    Example usage:
        engine = AsOfQueryEngine(data_dir="data/raw")

        # Get everything known on Jan 1, 2020
        result = engine.query(
            symbols=["AAPL", "GOOGL"],
            as_of=date(2020, 1, 1),
            lookback=252
        )

        # result.prices -> PIT-correct price data
        # result.universe -> Universe that existed on that date
    """

    def __init__(
        self,
        data_dir: Path | str,
        universe_manager: Optional[UniverseManager] = None,
        corporate_adjuster: Optional[CorporateActionAdjuster] = None,
    ) -> None:
        """
        Initialize the query engine.

        Args:
            data_dir: Directory containing data files
            universe_manager: Optional pre-configured universe manager
            corporate_adjuster: Optional pre-configured corporate action adjuster
        """
        self.data_dir = Path(data_dir)
        self.universe_manager = universe_manager or UniverseManager()
        self.corporate_adjuster = corporate_adjuster or CorporateActionAdjuster()

        self._pit_loader = PITDataLoader(
            data_dir=data_dir,
            universe_manager=self.universe_manager,
        )

        logger.info(f"AsOfQueryEngine initialized with data_dir={data_dir}")

    def query(
        self,
        as_of: date | datetime | str,
        symbols: Optional[List[str]] = None,
        index: Optional[str] = None,
        lookback_days: int = 252,
        fields: List[str] = ["open", "high", "low", "close", "volume"],
    ) -> "QueryResult":
        """
        Execute point-in-time query.

        Args:
            as_of: Reference timestamp (return only data known at this time)
            symbols: Specific symbols to query (None for full universe)
            index: Filter by index membership
            lookback_days: Number of historical days to include
            fields: Data fields to return

        Returns:
            QueryResult containing PIT-correct data
        """
        # Parse date
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        elif isinstance(as_of, datetime):
            as_of = as_of.date()

        # Get universe as it existed on as_of
        if symbols is None:
            symbols = self.universe_manager.get_universe(as_of=as_of, index=index)

        # Load data for each symbol
        data: Dict[str, pd.DataFrame] = {}
        metadata: Dict[str, Dict[str, Any]] = {}

        for symbol in symbols:
            try:
                # Load PIT-correct data
                df = self._pit_loader.load_as_of(
                    symbol=symbol,
                    as_of=as_of,
                    lookback_days=lookback_days,
                    fields=fields,
                )

                if not df.empty:
                    # Apply corporate action adjustments (as known at as_of)
                    df = self.corporate_adjuster.adjust(
                        df=df,
                        symbol=symbol,
                        as_of=as_of,
                    )

                    data[symbol] = df

                    # Collect metadata
                    meta = self.universe_manager.get_symbol(symbol)
                    if meta:
                        metadata[symbol] = {
                            "sector": meta.sector,
                            "industry": meta.industry,
                            "is_active": meta.is_active(as_of),
                            "listing_date": meta.listing_date,
                            "delist_date": meta.delist_date,
                        }

            except Exception as e:
                logger.warning(f"Failed to query {symbol} as-of {as_of}: {e}")

        return QueryResult(
            as_of=as_of,
            data=data,
            metadata=metadata,
            universe=symbols,
            lookback_days=lookback_days,
        )

    def query_panel(
        self,
        as_of: date | datetime | str,
        field: str = "close",
        symbols: Optional[List[str]] = None,
        index: Optional[str] = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Query single field as panel (timestamps x symbols).

        Args:
            as_of: Reference timestamp
            field: Field to query
            symbols: Specific symbols
            index: Index filter
            lookback_days: Historical lookback

        Returns:
            DataFrame with timestamps as index and symbols as columns
        """
        result = self.query(
            as_of=as_of,
            symbols=symbols,
            index=index,
            lookback_days=lookback_days,
            fields=[field],
        )

        return result.to_panel(field)

    def query_time_series(
        self,
        symbol: str,
        start_date: date | str,
        end_date: date | str,
        step_days: int = 1,
        lookback_days: int = 252,
    ) -> List["QueryResult"]:
        """
        Query data over time in walk-forward fashion.

        At each step, returns only what was known at that time.

        Args:
            symbol: Symbol to query
            start_date: Start of query period
            end_date: End of query period
            step_days: Days between queries
            lookback_days: Historical lookback at each query

        Returns:
            List of QueryResult objects
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        results = []
        current = start_date

        while current <= end_date:
            result = self.query(
                as_of=current,
                symbols=[symbol],
                lookback_days=lookback_days,
            )
            results.append(result)
            current += timedelta(days=step_days)

        return results

    def validate_no_lookahead(
        self,
        symbol: str,
        test_date: date,
    ) -> bool:
        """
        Validate that queries do not leak future information.

        This is a diagnostic method to verify PIT correctness.

        Args:
            symbol: Symbol to test
            test_date: Date to test

        Returns:
            True if no lookahead detected
        """
        # Query as of test_date
        result = self.query(
            as_of=test_date,
            symbols=[symbol],
            lookback_days=30,
        )

        if symbol not in result.data:
            return True

        df = result.data[symbol]

        # Check that no data is after test_date
        if isinstance(df.index, pd.DatetimeIndex):
            max_date = df.index.max().date()
        else:
            max_date = pd.to_datetime(df.index.max()).date()

        if max_date > test_date:
            logger.error(
                f"LOOKAHEAD BIAS DETECTED: {symbol} has data from {max_date} "
                f"in query as-of {test_date}"
            )
            return False

        return True


class QueryResult:
    """
    Container for point-in-time query results.

    Provides convenient access to PIT-correct data with
    various output formats.
    """

    def __init__(
        self,
        as_of: date,
        data: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict[str, Any]],
        universe: List[str],
        lookback_days: int,
    ) -> None:
        """
        Initialize query result.

        Args:
            as_of: Query reference date
            data: Symbol -> DataFrame mapping
            metadata: Symbol -> metadata mapping
            universe: List of symbols in universe
            lookback_days: Lookback used in query
        """
        self.as_of = as_of
        self.data = data
        self.metadata = metadata
        self.universe = universe
        self.lookback_days = lookback_days

    def to_panel(self, field: str = "close") -> pd.DataFrame:
        """
        Convert to panel format (timestamps x symbols).

        Args:
            field: Field to extract

        Returns:
            Panel DataFrame
        """
        panels = {}
        for symbol, df in self.data.items():
            if field in df.columns:
                panels[symbol] = df[field]

        if not panels:
            return pd.DataFrame()

        panel = pd.DataFrame(panels)
        return panel.sort_index()

    def to_multiindex(self) -> pd.DataFrame:
        """
        Convert to MultiIndex DataFrame.

        Returns:
            DataFrame with (timestamp, symbol) MultiIndex
        """
        dfs = []
        for symbol, df in self.data.items():
            temp = df.copy()
            temp["symbol"] = symbol
            temp = temp.reset_index()
            dfs.append(temp)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Determine index column name
        index_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
        combined = combined.set_index([index_col, "symbol"])
        return combined.sort_index()

    def get_returns(
        self,
        periods: int = 1,
        field: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            periods: Return periods
            field: Price field to use

        Returns:
            Returns panel
        """
        prices = self.to_panel(field)
        return prices.pct_change(periods)

    def get_sectors(self) -> Dict[str, List[str]]:
        """
        Get symbols grouped by sector.

        Returns:
            Sector -> symbols mapping
        """
        sectors: Dict[str, List[str]] = {}
        for symbol, meta in self.metadata.items():
            sector = meta.get("sector", "Unknown")
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)
        return sectors

    @property
    def symbols(self) -> List[str]:
        """Get symbols with data."""
        return list(self.data.keys())

    @property
    def start_date(self) -> Optional[date]:
        """Get earliest date in data."""
        dates = []
        for df in self.data.values():
            if not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    dates.append(df.index.min().date())
                else:
                    dates.append(pd.to_datetime(df.index.min()).date())
        return min(dates) if dates else None

    @property
    def end_date(self) -> Optional[date]:
        """Get latest date in data."""
        dates = []
        for df in self.data.values():
            if not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    dates.append(df.index.max().date())
                else:
                    dates.append(pd.to_datetime(df.index.max()).date())
        return max(dates) if dates else None

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"QueryResult(as_of={self.as_of}, symbols={len(self.data)}, "
            f"date_range={self.start_date} to {self.end_date})"
        )
