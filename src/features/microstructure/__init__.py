"""
Market Microstructure Features Module.

This module provides institutional-grade microstructure features:
1. Order Flow Imbalance (OFI)
2. VPIN (Volume-Synchronized Probability of Informed Trading)
3. Kyle's Lambda (Price Impact)
4. Roll Spread (Effective Spread)
5. Amihud Illiquidity
6. Advanced LOB Features (multi-depth OBI, spread analytics, toxicity, smart money)

These features capture market quality and informed trading activity,
providing alpha signals unavailable from standard technical indicators.

Reference:
    "Market Microstructure Theory" by O'Hara (1995)
    "Flow Toxicity and Liquidity" by Easley, de Prado, O'Hara (2012)
    "Algorithmic and High-Frequency Trading" by Cartea, Jaimungal, Penalva (2015)
"""

from .order_flow_imbalance import OrderFlowImbalance, calculate_ofi
from .vpin import VPIN, calculate_vpin
from .kyle_lambda import KyleLambda, calculate_kyle_lambda
from .roll_spread import RollSpread, AmihudIlliquidity
from .order_book_dynamics import OrderBookDynamics
from .advanced_lob_features import (
    AdvancedLOBFeatures,
    MultiDepthImbalance,
    SpreadAnalytics,
    TradeFlowToxicity,
    SmartMoneyIndicator,
    LOBSnapshot,
    LOBFeatures,
    calculate_multi_depth_obi,
    calculate_spread_percentile,
    calculate_toxicity,
    calculate_smart_money,
    calculate_institutional_activity,
)

__all__ = [
    # Original features
    "OrderFlowImbalance",
    "calculate_ofi",
    "VPIN",
    "calculate_vpin",
    "KyleLambda",
    "calculate_kyle_lambda",
    "RollSpread",
    "AmihudIlliquidity",
    "OrderBookDynamics",
    # Advanced LOB features
    "AdvancedLOBFeatures",
    "MultiDepthImbalance",
    "SpreadAnalytics",
    "TradeFlowToxicity",
    "SmartMoneyIndicator",
    "LOBSnapshot",
    "LOBFeatures",
    "calculate_multi_depth_obi",
    "calculate_spread_percentile",
    "calculate_toxicity",
    "calculate_smart_money",
    "calculate_institutional_activity",
]
