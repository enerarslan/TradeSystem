"""
Data loader module for AlphaTrade system.

This module provides robust data loading capabilities:
- Support for multiple file formats (CSV, Parquet, Feather)
- Lazy loading for memory efficiency
- Automatic symbol detection
- MultiIndex DataFrame creation
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from config.settings import settings, RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:
    """
    Robust data loader for OHLCV stock data.

    Supports loading from multiple file formats and creates
    a unified DataFrame with proper indexing.

    Attributes:
        data_dir: Directory containing data files
        file_format: Expected file format
        symbols: List of available symbols
    """

    SUPPORTED_FORMATS = ("csv", "parquet", "feather")

    def __init__(
        self,
        data_dir: Path | str | None = None,
        file_format: Literal["csv", "parquet", "feather"] = "csv",
        date_column: str = "timestamp",
        ohlcv_columns: list[str] | None = None,
    ) -> None:
        """
        Initialize the DataLoader.

        Args:
            data_dir: Directory containing data files
            file_format: File format to load
            date_column: Name of the datetime column
            ohlcv_columns: List of OHLCV column names
        """
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.file_format = file_format
        self.date_column = date_column
        self.ohlcv_columns = ohlcv_columns or ["open", "high", "low", "close", "volume"]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._symbols: list[str] | None = None
        self._data_cache: dict[str, pd.DataFrame] = {}

        logger.info(f"DataLoader initialized with data_dir={self.data_dir}")

    @property
    def symbols(self) -> list[str]:
        """Get list of available symbols."""
        if self._symbols is None:
            self._symbols = self._discover_symbols()
        return self._symbols

    def _discover_symbols(self) -> list[str]:
        """Discover available symbols from data directory."""
        pattern = f"*_15min.{self.file_format}"
        files = list(self.data_dir.glob(pattern))

        symbols = []
        for f in files:
            # Extract symbol from filename (e.g., AAPL_15min.csv -> AAPL)
            symbol = f.stem.replace("_15min", "")
            if symbol and not symbol.startswith("."):
                symbols.append(symbol)

        symbols.sort()
        logger.info(f"Discovered {len(symbols)} symbols")
        return symbols

    def _get_file_path(self, symbol: str) -> Path:
        """Get file path for a symbol."""
        return self.data_dir / f"{symbol}_15min.{self.file_format}"

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single data file.

        Args:
            file_path: Path to the data file

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        suffix = file_path.suffix.lower().lstrip(".")

        if suffix == "csv":
            df = pd.read_csv(
                file_path,
                parse_dates=[self.date_column],
                index_col=None,
            )
        elif suffix == "parquet":
            df = pd.read_parquet(file_path)
        elif suffix == "feather":
            df = pd.read_feather(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return df

    def load_symbol(
        self,
        symbol: str,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load data for a single symbol.

        Args:
            symbol: Stock symbol to load
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        # Check cache
        if use_cache and symbol in self._data_cache:
            df = self._data_cache[symbol].copy()
        else:
            file_path = self._get_file_path(symbol)
            df = self._load_file(file_path)

            # Set datetime index
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df = df.set_index(self.date_column)
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df.index.name = "timestamp"

            # Sort by index
            df = df.sort_index()

            # Cache if enabled
            if use_cache:
                self._data_cache[symbol] = df.copy()

        # Apply date filters
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            df = df[df.index >= start_date]
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            df = df[df.index <= end_date]

        logger.debug(f"Loaded {symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        return df

    def load_symbols(
        self,
        symbols: list[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        as_multiindex: bool = False,
        show_progress: bool = True,
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of symbols to load (None for all)
            start_date: Start date filter
            end_date: End date filter
            as_multiindex: Return as MultiIndex DataFrame
            show_progress: Show loading progress

        Returns:
            Dictionary of DataFrames or MultiIndex DataFrame
        """
        if symbols is None:
            symbols = self.symbols

        data: dict[str, pd.DataFrame] = {}

        iterator = symbols
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(symbols, desc="Loading data")
            except ImportError:
                pass

        for symbol in iterator:
            try:
                df = self.load_symbol(symbol, start_date, end_date)
                data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")

        logger.info(f"Loaded data for {len(data)}/{len(symbols)} symbols")

        if as_multiindex:
            return self._create_multiindex(data)
        return data

    def _create_multiindex(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Create a MultiIndex DataFrame from symbol data.

        Args:
            data: Dictionary mapping symbols to DataFrames

        Returns:
            MultiIndex DataFrame with (timestamp, symbol) index
        """
        dfs = []
        for symbol, df in data.items():
            df = df.copy()
            df["symbol"] = symbol
            df = df.reset_index()
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.set_index(["timestamp", "symbol"])
        combined = combined.sort_index()

        return combined

    def load_panel(
        self,
        column: str = "close",
        symbols: list[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Load a panel of single-column data for all symbols.

        Args:
            column: Column to extract (e.g., 'close')
            symbols: Symbols to load
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with timestamps as rows and symbols as columns
        """
        data = self.load_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            as_multiindex=False,
            show_progress=True,
        )

        panels = {}
        for symbol, df in data.items():
            if column in df.columns:
                panels[symbol] = df[column]

        panel = pd.DataFrame(panels)
        panel = panel.sort_index()

        logger.info(f"Created panel: {panel.shape[0]} timestamps x {panel.shape[1]} symbols")
        return panel

    def get_date_range(
        self,
        symbols: list[str] | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the common date range across all symbols.

        Args:
            symbols: Symbols to check (None for all)

        Returns:
            Tuple of (earliest_date, latest_date)
        """
        if symbols is None:
            symbols = self.symbols

        min_dates = []
        max_dates = []

        for symbol in symbols:
            try:
                df = self.load_symbol(symbol)
                min_dates.append(df.index.min())
                max_dates.append(df.index.max())
            except Exception:
                continue

        if not min_dates:
            raise ValueError("No valid data found")

        return max(min_dates), min(max_dates)

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        logger.debug("Data cache cleared")

    def get_data_info(self, symbol: str | None = None) -> pd.DataFrame:
        """
        Get information about loaded data.

        Args:
            symbol: Specific symbol (None for all)

        Returns:
            DataFrame with data information
        """
        symbols = [symbol] if symbol else self.symbols

        info = []
        for sym in symbols:
            try:
                df = self.load_symbol(sym)
                info.append(
                    {
                        "symbol": sym,
                        "rows": len(df),
                        "start_date": df.index.min(),
                        "end_date": df.index.max(),
                        "columns": list(df.columns),
                        "missing_pct": df.isnull().sum().sum() / df.size * 100,
                    }
                )
            except Exception as e:
                info.append(
                    {
                        "symbol": sym,
                        "rows": 0,
                        "start_date": None,
                        "end_date": None,
                        "columns": [],
                        "missing_pct": 100.0,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(info)


# Convenience functions
def load_single_stock(
    symbol: str,
    data_dir: Path | str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load data for a single stock.

    Args:
        symbol: Stock symbol
        data_dir: Data directory
        start_date: Start date filter
        end_date: End date filter

    Returns:
        DataFrame with OHLCV data
    """
    loader = DataLoader(data_dir=data_dir)
    return loader.load_symbol(symbol, start_date, end_date)


def load_all_stocks(
    data_dir: Path | str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    as_multiindex: bool = False,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Load data for all available stocks.

    Args:
        data_dir: Data directory
        start_date: Start date filter
        end_date: End date filter
        as_multiindex: Return as MultiIndex DataFrame

    Returns:
        Dictionary of DataFrames or MultiIndex DataFrame
    """
    loader = DataLoader(data_dir=data_dir)
    return loader.load_symbols(
        start_date=start_date,
        end_date=end_date,
        as_multiindex=as_multiindex,
    )


def get_available_symbols(data_dir: Path | str | None = None) -> list[str]:
    """
    Get list of available symbols.

    Args:
        data_dir: Data directory

    Returns:
        List of symbol strings
    """
    loader = DataLoader(data_dir=data_dir)
    return loader.symbols
