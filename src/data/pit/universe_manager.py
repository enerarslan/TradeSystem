"""
Universe Manager for survivorship bias prevention.

This module maintains historical universe membership to prevent survivorship bias
in backtesting. Without proper universe management, backtests exclude:
- Delisted companies (bankruptcies, M&A targets)
- Removed index constituents
- Suspended symbols

Research shows survivorship bias inflates returns by 0.9-1.4% annually.
(Elton, Gruber, Blake 1996)

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolMetadata:
    """
    Metadata for a tradeable symbol including lifecycle information.

    This tracks when a symbol was valid for trading, enabling
    point-in-time universe reconstruction.
    """

    symbol: str
    name: str
    exchange: str = ""
    asset_class: str = "equity"
    sector: str = ""
    industry: str = ""

    # Lifecycle dates (critical for survivorship bias prevention)
    listing_date: Optional[date] = None
    delist_date: Optional[date] = None
    delist_reason: Optional[str] = None  # bankruptcy, merger, acquisition, etc.

    # Index membership history
    index_memberships: Dict[str, List[tuple]] = field(default_factory=dict)
    # Format: {"SP500": [(start_date, end_date), ...], "NASDAQ100": [...]}

    # Corporate structure
    cusip: Optional[str] = None
    isin: Optional[str] = None
    sedol: Optional[str] = None

    # Trading characteristics
    lot_size: int = 100
    tick_size: float = 0.01
    currency: str = "USD"

    def is_active(self, as_of: date) -> bool:
        """Check if symbol was actively traded on given date."""
        # Check listing date
        if self.listing_date and as_of < self.listing_date:
            return False

        # Check delist date
        if self.delist_date and as_of > self.delist_date:
            return False

        return True

    def is_index_member(self, index: str, as_of: date) -> bool:
        """Check if symbol was member of index on given date."""
        if index not in self.index_memberships:
            return False

        for start, end in self.index_memberships[index]:
            if start <= as_of <= (end or date.max):
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "asset_class": self.asset_class,
            "sector": self.sector,
            "industry": self.industry,
            "listing_date": self.listing_date.isoformat() if self.listing_date else None,
            "delist_date": self.delist_date.isoformat() if self.delist_date else None,
            "delist_reason": self.delist_reason,
            "index_memberships": {
                idx: [(s.isoformat(), e.isoformat() if e else None) for s, e in periods]
                for idx, periods in self.index_memberships.items()
            },
            "cusip": self.cusip,
            "isin": self.isin,
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolMetadata":
        """Create from dictionary."""
        # Parse dates
        listing_date = None
        if data.get("listing_date"):
            listing_date = date.fromisoformat(data["listing_date"])

        delist_date = None
        if data.get("delist_date"):
            delist_date = date.fromisoformat(data["delist_date"])

        # Parse index memberships
        index_memberships = {}
        for idx, periods in data.get("index_memberships", {}).items():
            index_memberships[idx] = [
                (
                    date.fromisoformat(s),
                    date.fromisoformat(e) if e else None,
                )
                for s, e in periods
            ]

        return cls(
            symbol=data["symbol"],
            name=data.get("name", ""),
            exchange=data.get("exchange", ""),
            asset_class=data.get("asset_class", "equity"),
            sector=data.get("sector", ""),
            industry=data.get("industry", ""),
            listing_date=listing_date,
            delist_date=delist_date,
            delist_reason=data.get("delist_reason"),
            index_memberships=index_memberships,
            cusip=data.get("cusip"),
            isin=data.get("isin"),
            lot_size=data.get("lot_size", 100),
            tick_size=data.get("tick_size", 0.01),
            currency=data.get("currency", "USD"),
        )


class UniverseManager:
    """
    Manages point-in-time trading universe to prevent survivorship bias.

    This class is CRITICAL for institutional-grade backtesting:
    1. Tracks historical symbol lifecycle (listing/delisting)
    2. Maintains index membership history
    3. Provides as-of queries for universe reconstruction

    Example usage:
        universe_mgr = UniverseManager()
        universe_mgr.load_metadata("symbol_metadata.json")

        # Get universe as it existed on Jan 1, 2020
        symbols = universe_mgr.get_universe(
            as_of=date(2020, 1, 1),
            index="SP500"
        )

        # This will INCLUDE companies that later went bankrupt or were acquired

    Reference:
        Elton, Gruber, Blake (1996) - "Survivorship Bias and Mutual Fund Performance"
    """

    def __init__(
        self,
        metadata_path: Optional[Path | str] = None,
    ) -> None:
        """
        Initialize the universe manager.

        Args:
            metadata_path: Path to symbol metadata file
        """
        self._symbols: Dict[str, SymbolMetadata] = {}
        self._metadata_path = Path(metadata_path) if metadata_path else None

        if self._metadata_path and self._metadata_path.exists():
            self.load_metadata(self._metadata_path)

    def add_symbol(self, metadata: SymbolMetadata) -> None:
        """Add or update symbol metadata."""
        self._symbols[metadata.symbol] = metadata
        logger.debug(f"Added symbol metadata: {metadata.symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from universe."""
        if symbol in self._symbols:
            del self._symbols[symbol]

    def get_symbol(self, symbol: str) -> Optional[SymbolMetadata]:
        """Get metadata for a symbol."""
        return self._symbols.get(symbol)

    def get_universe(
        self,
        as_of: date | datetime | str,
        index: Optional[str] = None,
        sector: Optional[str] = None,
        min_listing_days: int = 0,
        exclude_delisted: bool = False,
    ) -> List[str]:
        """
        Get the trading universe as it existed on a specific date.

        CRITICAL: This method returns symbols that were VALID on the given date,
        including companies that may have later been delisted.

        Args:
            as_of: Reference date for universe construction
            index: Filter by index membership (e.g., "SP500")
            sector: Filter by sector
            min_listing_days: Minimum days since listing
            exclude_delisted: If True, exclude symbols delisted before as_of
                              (Default: False to include for historical backtest)

        Returns:
            List of symbols that were valid on the given date
        """
        # Parse date
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        elif isinstance(as_of, datetime):
            as_of = as_of.date()

        universe = []

        for symbol, meta in self._symbols.items():
            # Check if active on date
            if not meta.is_active(as_of):
                continue

            # Check index membership
            if index and not meta.is_index_member(index, as_of):
                continue

            # Check sector
            if sector and meta.sector != sector:
                continue

            # Check minimum listing days
            if min_listing_days > 0 and meta.listing_date:
                days_listed = (as_of - meta.listing_date).days
                if days_listed < min_listing_days:
                    continue

            # Optionally exclude delisted symbols
            if exclude_delisted and meta.delist_date and meta.delist_date <= as_of:
                continue

            universe.append(symbol)

        logger.info(
            f"Universe as of {as_of}: {len(universe)} symbols "
            f"(index={index}, sector={sector})"
        )

        return sorted(universe)

    def get_historical_universe(
        self,
        start_date: date | str,
        end_date: date | str,
        index: Optional[str] = None,
        frequency: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> Dict[date, List[str]]:
        """
        Get universe for each date in a range.

        This is useful for backtests that need to adjust the universe
        at each rebalancing point.

        Args:
            start_date: Start of date range
            end_date: End of date range
            index: Filter by index
            frequency: How often to compute universe

        Returns:
            Dictionary mapping dates to symbol lists
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D").date.tolist()

        if frequency == "weekly":
            dates = [d for d in dates if d.weekday() == 0]  # Mondays
        elif frequency == "monthly":
            dates = [d for d in dates if d.day == 1]

        # Get universe for each date
        historical = {}
        for d in dates:
            historical[d] = self.get_universe(as_of=d, index=index)

        return historical

    def get_delistings(
        self,
        start_date: date | str,
        end_date: date | str,
        reason: Optional[str] = None,
    ) -> List[SymbolMetadata]:
        """
        Get symbols that were delisted in a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            reason: Filter by delist reason (bankruptcy, merger, etc.)

        Returns:
            List of delisted symbols with metadata
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        delistings = []

        for symbol, meta in self._symbols.items():
            if meta.delist_date:
                if start_date <= meta.delist_date <= end_date:
                    if reason is None or meta.delist_reason == reason:
                        delistings.append(meta)

        logger.info(
            f"Found {len(delistings)} delistings between {start_date} and {end_date}"
        )

        return delistings

    def get_index_changes(
        self,
        index: str,
        start_date: date | str,
        end_date: date | str,
    ) -> Dict[str, List[tuple]]:
        """
        Get index additions and removals in a date range.

        Args:
            index: Index name (e.g., "SP500")
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with "additions" and "removals" lists
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        additions = []
        removals = []

        for symbol, meta in self._symbols.items():
            if index in meta.index_memberships:
                for period_start, period_end in meta.index_memberships[index]:
                    # Check for addition
                    if start_date <= period_start <= end_date:
                        additions.append((period_start, symbol))

                    # Check for removal
                    if period_end and start_date <= period_end <= end_date:
                        removals.append((period_end, symbol))

        return {
            "additions": sorted(additions),
            "removals": sorted(removals),
        }

    def load_metadata(self, path: Path | str) -> None:
        """Load symbol metadata from JSON file."""
        path = Path(path)

        if not path.exists():
            logger.warning(f"Metadata file not found: {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        symbols_data = data.get("symbols", {})
        # Support both list and dict format
        if isinstance(symbols_data, dict):
            for symbol, symbol_data in symbols_data.items():
                meta = SymbolMetadata.from_dict(symbol_data)
                self._symbols[meta.symbol] = meta
        else:
            for symbol_data in symbols_data:
                meta = SymbolMetadata.from_dict(symbol_data)
                self._symbols[meta.symbol] = meta

        logger.info(f"Loaded metadata for {len(self._symbols)} symbols from {path}")

    def save_metadata(self, path: Path | str) -> None:
        """Save symbol metadata to JSON file."""
        path = Path(path)

        data = {
            "symbols": [meta.to_dict() for meta in self._symbols.values()],
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved metadata for {len(self._symbols)} symbols to {path}")

    def create_from_current_data(
        self,
        data_dir: Path | str,
        default_listing_date: Optional[date] = None,
    ) -> None:
        """
        Create basic metadata from existing data files.

        WARNING: This creates metadata without historical delist information.
        For production use, supplement with external data sources.

        Args:
            data_dir: Directory containing data files
            default_listing_date: Default listing date if unknown
        """
        data_dir = Path(data_dir)

        for file_path in data_dir.glob("*_15min.csv"):
            symbol = file_path.stem.replace("_15min", "")

            if symbol not in self._symbols:
                self._symbols[symbol] = SymbolMetadata(
                    symbol=symbol,
                    name=symbol,
                    listing_date=default_listing_date,
                )

        logger.info(f"Created metadata for {len(self._symbols)} symbols from {data_dir}")

    @property
    def symbols(self) -> List[str]:
        """Get all known symbols."""
        return list(self._symbols.keys())

    @property
    def active_symbols(self) -> List[str]:
        """Get symbols that are currently active (not delisted)."""
        return [s for s, m in self._symbols.items() if m.delist_date is None]

    @property
    def delisted_symbols(self) -> List[str]:
        """Get symbols that have been delisted."""
        return [s for s, m in self._symbols.items() if m.delist_date is not None]

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._symbols


def create_sample_universe_metadata() -> UniverseManager:
    """
    Create sample universe with realistic lifecycle data.

    This demonstrates the metadata structure needed for proper
    survivorship bias prevention.
    """
    manager = UniverseManager()

    # Sample active stock
    manager.add_symbol(
        SymbolMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            listing_date=date(1980, 12, 12),
            index_memberships={
                "SP500": [(date(1982, 1, 1), None)],  # Still member
                "NASDAQ100": [(date(1985, 1, 1), None)],
            },
        )
    )

    # Sample delisted stock (acquisition)
    manager.add_symbol(
        SymbolMetadata(
            symbol="CELG",
            name="Celgene Corporation",
            exchange="NASDAQ",
            sector="Healthcare",
            industry="Biotechnology",
            listing_date=date(1987, 1, 1),
            delist_date=date(2019, 11, 21),
            delist_reason="merger",  # Acquired by Bristol-Myers
            index_memberships={
                "SP500": [(date(2008, 1, 1), date(2019, 11, 21))],
            },
        )
    )

    # Sample delisted stock (bankruptcy)
    manager.add_symbol(
        SymbolMetadata(
            symbol="LEH",
            name="Lehman Brothers Holdings",
            exchange="NYSE",
            sector="Financials",
            industry="Investment Banking",
            listing_date=date(1994, 5, 1),
            delist_date=date(2008, 9, 17),
            delist_reason="bankruptcy",
            index_memberships={
                "SP500": [(date(1998, 1, 1), date(2008, 9, 17))],
            },
        )
    )

    return manager
