"""
Symbols Configuration Module
============================

Centralized symbol management for the algorithmic trading platform.
Contains all 46 supported symbols (Dow Jones + NASDAQ 100 subset).

Features:
- Symbol categorization by sector/index
- Symbol metadata (sector, market cap tier, etc.)
- Symbol discovery from data directory
- Symbol validation utilities

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from config.settings import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class Sector(str, Enum):
    """Stock sectors."""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


class MarketCapTier(str, Enum):
    """Market capitalization tier."""
    MEGA = "mega"      # > $200B
    LARGE = "large"    # $10B - $200B
    MID = "mid"        # $2B - $10B
    SMALL = "small"    # < $2B


class Index(str, Enum):
    """Stock index membership."""
    DOW_JONES = "dow_jones"
    SP500 = "sp500"
    NASDAQ100 = "nasdaq100"
    RUSSELL2000 = "russell2000"


# =============================================================================
# SYMBOL METADATA
# =============================================================================

@dataclass
class SymbolInfo:
    """Metadata for a trading symbol."""
    symbol: str
    name: str
    sector: Sector
    market_cap_tier: MarketCapTier = MarketCapTier.LARGE
    indices: list[Index] = field(default_factory=list)
    is_active: bool = True
    average_volume: float = 0.0  # Average daily volume
    beta: float = 1.0  # Market beta
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector.value,
            "market_cap_tier": self.market_cap_tier.value,
            "indices": [i.value for i in self.indices],
            "is_active": self.is_active,
            "average_volume": self.average_volume,
            "beta": self.beta,
        }


# =============================================================================
# SYMBOL REGISTRY - ALL 46 SYMBOLS
# =============================================================================

# Complete symbol information for all 46 supported symbols
SYMBOL_INFO: dict[str, SymbolInfo] = {
    # === TECHNOLOGY ===
    "AAPL": SymbolInfo(
        symbol="AAPL", name="Apple Inc.", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500, Index.NASDAQ100]
    ),
    "MSFT": SymbolInfo(
        symbol="MSFT", name="Microsoft Corporation", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500, Index.NASDAQ100]
    ),
    "GOOGL": SymbolInfo(
        symbol="GOOGL", name="Alphabet Inc. Class A", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "AMZN": SymbolInfo(
        symbol="AMZN", name="Amazon.com Inc.", sector=Sector.CONSUMER_DISCRETIONARY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "META": SymbolInfo(
        symbol="META", name="Meta Platforms Inc.", sector=Sector.COMMUNICATION,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "NVDA": SymbolInfo(
        symbol="NVDA", name="NVIDIA Corporation", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "TSLA": SymbolInfo(
        symbol="TSLA", name="Tesla Inc.", sector=Sector.CONSUMER_DISCRETIONARY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "AMD": SymbolInfo(
        symbol="AMD", name="Advanced Micro Devices", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "INTC": SymbolInfo(
        symbol="INTC", name="Intel Corporation", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500, Index.NASDAQ100]
    ),
    "CRM": SymbolInfo(
        symbol="CRM", name="Salesforce Inc.", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "CSCO": SymbolInfo(
        symbol="CSCO", name="Cisco Systems Inc.", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500, Index.NASDAQ100]
    ),
    "ORCL": SymbolInfo(
        symbol="ORCL", name="Oracle Corporation", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500]
    ),
    "ADBE": SymbolInfo(
        symbol="ADBE", name="Adobe Inc.", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "NFLX": SymbolInfo(
        symbol="NFLX", name="Netflix Inc.", sector=Sector.COMMUNICATION,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "QCOM": SymbolInfo(
        symbol="QCOM", name="Qualcomm Inc.", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "AVGO": SymbolInfo(
        symbol="AVGO", name="Broadcom Inc.", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "TXN": SymbolInfo(
        symbol="TXN", name="Texas Instruments", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "IBM": SymbolInfo(
        symbol="IBM", name="IBM Corporation", sector=Sector.TECHNOLOGY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    
    # === HEALTHCARE ===
    "JNJ": SymbolInfo(
        symbol="JNJ", name="Johnson & Johnson", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "UNH": SymbolInfo(
        symbol="UNH", name="UnitedHealth Group", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "PFE": SymbolInfo(
        symbol="PFE", name="Pfizer Inc.", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500]
    ),
    "ABBV": SymbolInfo(
        symbol="ABBV", name="AbbVie Inc.", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500]
    ),
    "MRK": SymbolInfo(
        symbol="MRK", name="Merck & Co.", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "LLY": SymbolInfo(
        symbol="LLY", name="Eli Lilly and Company", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500]
    ),
    "AMGN": SymbolInfo(
        symbol="AMGN", name="Amgen Inc.", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500, Index.NASDAQ100]
    ),
    "TMO": SymbolInfo(
        symbol="TMO", name="Thermo Fisher Scientific", sector=Sector.HEALTHCARE,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500]
    ),

    # === FINANCIALS ===
    "JPM": SymbolInfo(
        symbol="JPM", name="JPMorgan Chase & Co.", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "V": SymbolInfo(
        symbol="V", name="Visa Inc.", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "MA": SymbolInfo(
        symbol="MA", name="Mastercard Inc.", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500]
    ),
    "BAC": SymbolInfo(
        symbol="BAC", name="Bank of America", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500]
    ),
    "GS": SymbolInfo(
        symbol="GS", name="Goldman Sachs", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "AXP": SymbolInfo(
        symbol="AXP", name="American Express", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "TRV": SymbolInfo(
        symbol="TRV", name="The Travelers Companies", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "BRK.B": SymbolInfo(
        symbol="BRK.B", name="Berkshire Hathaway Class B", sector=Sector.FINANCIALS,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500]
    ),

    # === CONSUMER ===
    "WMT": SymbolInfo(
        symbol="WMT", name="Walmart Inc.", sector=Sector.CONSUMER_STAPLES,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "HD": SymbolInfo(
        symbol="HD", name="The Home Depot", sector=Sector.CONSUMER_DISCRETIONARY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "PG": SymbolInfo(
        symbol="PG", name="Procter & Gamble", sector=Sector.CONSUMER_STAPLES,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "KO": SymbolInfo(
        symbol="KO", name="The Coca-Cola Company", sector=Sector.CONSUMER_STAPLES,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "PEP": SymbolInfo(
        symbol="PEP", name="PepsiCo Inc.", sector=Sector.CONSUMER_STAPLES,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "MCD": SymbolInfo(
        symbol="MCD", name="McDonald's Corporation", sector=Sector.CONSUMER_DISCRETIONARY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "NKE": SymbolInfo(
        symbol="NKE", name="Nike Inc.", sector=Sector.CONSUMER_DISCRETIONARY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "COST": SymbolInfo(
        symbol="COST", name="Costco Wholesale", sector=Sector.CONSUMER_STAPLES,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "SBUX": SymbolInfo(
        symbol="SBUX", name="Starbucks Corporation", sector=Sector.CONSUMER_DISCRETIONARY,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.SP500, Index.NASDAQ100]
    ),
    "WBA": SymbolInfo(
        symbol="WBA", name="Walgreens Boots Alliance", sector=Sector.CONSUMER_STAPLES,
        market_cap_tier=MarketCapTier.MID, indices=[Index.SP500, Index.NASDAQ100]
    ),

    # === INDUSTRIALS & ENERGY ===
    "BA": SymbolInfo(
        symbol="BA", name="The Boeing Company", sector=Sector.INDUSTRIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "CAT": SymbolInfo(
        symbol="CAT", name="Caterpillar Inc.", sector=Sector.INDUSTRIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "HON": SymbolInfo(
        symbol="HON", name="Honeywell International", sector=Sector.INDUSTRIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "MMM": SymbolInfo(
        symbol="MMM", name="3M Company", sector=Sector.INDUSTRIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "DOW": SymbolInfo(
        symbol="DOW", name="Dow Inc.", sector=Sector.MATERIALS,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "XOM": SymbolInfo(
        symbol="XOM", name="Exxon Mobil Corporation", sector=Sector.ENERGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.SP500]
    ),
    "CVX": SymbolInfo(
        symbol="CVX", name="Chevron Corporation", sector=Sector.ENERGY,
        market_cap_tier=MarketCapTier.MEGA, indices=[Index.DOW_JONES, Index.SP500]
    ),
    
    # === COMMUNICATION ===
    "DIS": SymbolInfo(
        symbol="DIS", name="The Walt Disney Company", sector=Sector.COMMUNICATION,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
    "VZ": SymbolInfo(
        symbol="VZ", name="Verizon Communications", sector=Sector.COMMUNICATION,
        market_cap_tier=MarketCapTier.LARGE, indices=[Index.DOW_JONES, Index.SP500]
    ),
}

# =============================================================================
# SYMBOL LISTS
# =============================================================================

# All 46 symbols in alphabetical order
ALL_SYMBOLS: list[str] = sorted(SYMBOL_INFO.keys())

# Symbols by index
DOW_JONES_SYMBOLS: list[str] = [
    s for s, info in SYMBOL_INFO.items() 
    if Index.DOW_JONES in info.indices
]

NASDAQ100_SYMBOLS: list[str] = [
    s for s, info in SYMBOL_INFO.items() 
    if Index.NASDAQ100 in info.indices
]

# Symbols by sector
TECH_SYMBOLS: list[str] = [
    s for s, info in SYMBOL_INFO.items() 
    if info.sector == Sector.TECHNOLOGY
]

HEALTHCARE_SYMBOLS: list[str] = [
    s for s, info in SYMBOL_INFO.items() 
    if info.sector == Sector.HEALTHCARE
]

FINANCIAL_SYMBOLS: list[str] = [
    s for s, info in SYMBOL_INFO.items() 
    if info.sector == Sector.FINANCIALS
]

# Mega-cap symbols (most liquid)
MEGA_CAP_SYMBOLS: list[str] = [
    s for s, info in SYMBOL_INFO.items() 
    if info.market_cap_tier == MarketCapTier.MEGA
]

# Core trading symbols (most recommended for ML training)
CORE_SYMBOLS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "JNJ", "UNH", "WMT", "PG", "XOM"
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_symbol_info(symbol: str) -> SymbolInfo | None:
    """
    Get symbol information.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        SymbolInfo or None if not found
    """
    return SYMBOL_INFO.get(symbol.upper())


def get_symbols_by_sector(sector: Sector) -> list[str]:
    """
    Get symbols by sector.
    
    Args:
        sector: Sector enum
    
    Returns:
        List of symbols in that sector
    """
    return [s for s, info in SYMBOL_INFO.items() if info.sector == sector]


def get_symbols_by_index(index: Index) -> list[str]:
    """
    Get symbols by index membership.
    
    Args:
        index: Index enum
    
    Returns:
        List of symbols in that index
    """
    return [s for s, info in SYMBOL_INFO.items() if index in info.indices]


def validate_symbol(symbol: str) -> bool:
    """
    Check if a symbol is valid.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        True if valid
    """
    return symbol.upper() in SYMBOL_INFO


def validate_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate a list of symbols.
    
    Args:
        symbols: List of symbols to validate
    
    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    valid = []
    invalid = []
    for symbol in symbols:
        if validate_symbol(symbol):
            valid.append(symbol.upper())
        else:
            invalid.append(symbol)
    return valid, invalid


def discover_symbols_from_data(data_path: Path) -> list[str]:
    """
    Discover available symbols from data directory.
    
    Args:
        data_path: Path to data storage directory
    
    Returns:
        List of discovered symbols
    """
    if not data_path.exists():
        return []
    
    csv_files = list(data_path.glob("*_15min.csv")) + list(data_path.glob("*_1h.csv"))
    symbols = set()
    
    for f in csv_files:
        # Extract symbol from filename (e.g., AAPL_15min.csv -> AAPL)
        symbol = f.stem.split("_")[0].upper()
        if validate_symbol(symbol):
            symbols.add(symbol)
    
    return sorted(symbols)


def get_sector_allocation(symbols: list[str]) -> dict[str, int]:
    """
    Get sector allocation for a list of symbols.
    
    Args:
        symbols: List of symbols
    
    Returns:
        Dictionary of sector -> count
    """
    allocation = {}
    for symbol in symbols:
        info = get_symbol_info(symbol)
        if info:
            sector = info.sector.value
            allocation[sector] = allocation.get(sector, 0) + 1
    return allocation


# =============================================================================
# MODEL NAMING UTILITIES
# =============================================================================

def get_model_filename(
    symbol: str,
    model_type: str,
    version: str = "v1",
    extension: str = "pkl",
) -> str:
    """
    Generate standardized model filename.
    
    Args:
        symbol: Trading symbol
        model_type: Model type (lightgbm, xgboost, ensemble, etc.)
        version: Model version string
        extension: File extension
    
    Returns:
        Standardized filename
    
    Example:
        get_model_filename("AAPL", "lightgbm", "v1")
        # Returns: "AAPL_lightgbm_v1.pkl"
    """
    return f"{symbol.upper()}_{model_type}_{version}.{extension}"


def parse_model_filename(filename: str) -> dict[str, str] | None:
    """
    Parse a model filename into components.
    
    Args:
        filename: Model filename
    
    Returns:
        Dictionary with symbol, model_type, version, or None if invalid
    
    Example:
        parse_model_filename("AAPL_lightgbm_v1.pkl")
        # Returns: {"symbol": "AAPL", "model_type": "lightgbm", "version": "v1"}
    """
    try:
        name = Path(filename).stem  # Remove extension
        parts = name.split("_")
        if len(parts) >= 3:
            return {
                "symbol": parts[0],
                "model_type": parts[1],
                "version": "_".join(parts[2:]),  # Handle versions like v1_optimized
            }
    except Exception:
        pass
    return None


def get_model_directory(
    base_dir: Path,
    symbol: str,
    create: bool = True,
) -> Path:
    """
    Get the model directory for a symbol.
    
    Args:
        base_dir: Base models directory
        symbol: Trading symbol
        create: Create directory if it doesn't exist
    
    Returns:
        Path to symbol's model directory
    
    Example:
        models/artifacts/AAPL/
    """
    model_dir = base_dir / symbol.upper()
    if create:
        model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "Sector",
    "MarketCapTier",
    "Index",
    # Data classes
    "SymbolInfo",
    # Symbol data
    "SYMBOL_INFO",
    "ALL_SYMBOLS",
    "DOW_JONES_SYMBOLS",
    "NASDAQ100_SYMBOLS",
    "TECH_SYMBOLS",
    "HEALTHCARE_SYMBOLS",
    "FINANCIAL_SYMBOLS",
    "MEGA_CAP_SYMBOLS",
    "CORE_SYMBOLS",
    # Functions
    "get_symbol_info",
    "get_symbols_by_sector",
    "get_symbols_by_index",
    "validate_symbol",
    "validate_symbols",
    "discover_symbols_from_data",
    "get_sector_allocation",
    # Model naming
    "get_model_filename",
    "parse_model_filename",
    "get_model_directory",
]