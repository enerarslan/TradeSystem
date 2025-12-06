"""
Market Regime Enum
==================

Unified market regime classification for the algorithmic trading platform.
This file consolidates the MarketRegime enum that was previously duplicated
in config/settings.py and features/statistical.py.

Usage:
    from core.enums import MarketRegime
    
    # or import from config
    from config.settings import MarketRegime

Author: Algo Trading Platform
License: MIT
"""

from enum import Enum


class MarketRegime(str, Enum):
    """
    Market regime classification.
    
    Unified enum combining all regime classifications needed across the platform.
    
    Categories:
        - Volatility + Trend combinations (detailed classification)
        - Simple trend states
        - Simple volatility states
    
    Usage in different contexts:
        - Backtesting: Use simple states (TRENDING_UP, TRENDING_DOWN, RANGING)
        - Feature engineering: Use detailed states (HIGH_VOL_UPTREND, etc.)
        - Risk management: Use volatility states (HIGH_VOLATILITY, LOW_VOLATILITY)
    """
    
    # === Detailed Regimes (Volatility + Trend) ===
    HIGH_VOL_UPTREND = "high_vol_uptrend"
    HIGH_VOL_DOWNTREND = "high_vol_downtrend"
    LOW_VOL_UPTREND = "low_vol_uptrend"
    LOW_VOL_DOWNTREND = "low_vol_downtrend"
    
    # === Simple Trend Regimes ===
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    
    # === Simple Volatility Regimes ===
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    
    # === Special States ===
    UNKNOWN = "unknown"
    
    @property
    def is_trending(self) -> bool:
        """Check if regime indicates a trend."""
        return self in (
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.HIGH_VOL_UPTREND,
            MarketRegime.HIGH_VOL_DOWNTREND,
            MarketRegime.LOW_VOL_UPTREND,
            MarketRegime.LOW_VOL_DOWNTREND,
        )
    
    @property
    def is_bullish(self) -> bool:
        """Check if regime is bullish."""
        return self in (
            MarketRegime.TRENDING_UP,
            MarketRegime.HIGH_VOL_UPTREND,
            MarketRegime.LOW_VOL_UPTREND,
        )
    
    @property
    def is_bearish(self) -> bool:
        """Check if regime is bearish."""
        return self in (
            MarketRegime.TRENDING_DOWN,
            MarketRegime.HIGH_VOL_DOWNTREND,
            MarketRegime.LOW_VOL_DOWNTREND,
        )
    
    @property
    def is_high_volatility(self) -> bool:
        """Check if regime has high volatility."""
        return self in (
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.HIGH_VOL_UPTREND,
            MarketRegime.HIGH_VOL_DOWNTREND,
        )
    
    @property
    def is_low_volatility(self) -> bool:
        """Check if regime has low volatility."""
        return self in (
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.LOW_VOL_UPTREND,
            MarketRegime.LOW_VOL_DOWNTREND,
        )
    
    @classmethod
    def from_trend_and_volatility(
        cls,
        trend: str,
        volatility: str,
    ) -> "MarketRegime":
        """
        Create regime from trend and volatility states.
        
        Args:
            trend: "up", "down", or "ranging"
            volatility: "high", "low", or "normal"
        
        Returns:
            Appropriate MarketRegime
        """
        if volatility == "high":
            if trend == "up":
                return cls.HIGH_VOL_UPTREND
            elif trend == "down":
                return cls.HIGH_VOL_DOWNTREND
            else:
                return cls.HIGH_VOLATILITY
        elif volatility == "low":
            if trend == "up":
                return cls.LOW_VOL_UPTREND
            elif trend == "down":
                return cls.LOW_VOL_DOWNTREND
            else:
                return cls.LOW_VOLATILITY
        else:
            if trend == "up":
                return cls.TRENDING_UP
            elif trend == "down":
                return cls.TRENDING_DOWN
            else:
                return cls.RANGING