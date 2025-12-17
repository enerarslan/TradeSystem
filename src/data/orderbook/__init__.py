"""
Order Book Data Module for AlphaTrade System.

This module provides institutional-grade order book data handling:
1. L2 Order Book (top-of-book with depth)
2. L3 Order Book (full order-by-order data)
3. Order book reconstruction and simulation
4. Market microstructure feature extraction

Reference:
    - "Algorithmic and High-Frequency Trading" by Cartea et al. (2015)
    - "Market Microstructure in Practice" by Lehalle & Laruelle (2018)

Designed for JPMorgan-level institutional requirements.
"""

from .order_book import (
    OrderBook,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from .book_builder import (
    OrderBookBuilder,
    L2BookBuilder,
    L3BookBuilder,
)
from .book_features import (
    OrderBookFeatures,
    calculate_book_imbalance,
    calculate_weighted_mid,
    calculate_depth_profile,
)

__all__ = [
    # Core order book
    "OrderBook",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "Side",
    # Book builders
    "OrderBookBuilder",
    "L2BookBuilder",
    "L3BookBuilder",
    # Features
    "OrderBookFeatures",
    "calculate_book_imbalance",
    "calculate_weighted_mid",
    "calculate_depth_profile",
]
