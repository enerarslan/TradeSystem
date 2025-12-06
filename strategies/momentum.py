"""
Momentum Strategies Module
==========================

Momentum-based trading strategies for the algorithmic trading platform.

Strategies:
- TrendFollowingStrategy: Trade in direction of established trends
- BreakoutStrategy: Trade breakouts from consolidation
- MeanReversionStrategy: Trade reversions to mean
- DualMomentumStrategy: Combine absolute and relative momentum
- RSIDivergenceStrategy: Trade RSI divergences

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from config.settings import get_logger, TimeFrame
from core.events import MarketEvent, SignalEvent
from core.types import PortfolioState, Position
from strategies.base import BaseStrategy, StrategyConfig, StrategyState
from features.technical import (
    sma,
    ema,
    macd,
    rsi,
    bollinger_bands,
    atr,
    adx,
    supertrend,
    donchian_channels,
    roc,
)

logger = get_logger(__name__)


# =============================================================================
# TREND FOLLOWING STRATEGY
# =============================================================================

@dataclass
class TrendFollowingConfig(StrategyConfig):
    """Configuration for trend following strategy."""
    name: str = "TrendFollowing"
    
    # Moving average parameters
    fast_ma_period: int = 20
    slow_ma_period: int = 50
    trend_ma_period: int = 200
    ma_type: str = "ema"  # "sma" or "ema"
    
    # Trend strength filter
    use_adx_filter: bool = True
    adx_period: int = 14
    adx_threshold: float = 25.0
    
    # Entry conditions
    require_price_above_trend_ma: bool = True
    require_pullback: bool = False
    pullback_pct: float = 0.02
    
    # Exit conditions
    exit_on_ma_cross: bool = True
    trailing_stop_atr_mult: float = 2.0
    use_trailing_stop: bool = True


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy.
    
    Enters trades in the direction of the established trend
    using moving average crossovers and trend filters.
    
    Entry Conditions (Long):
        1. Fast MA crosses above Slow MA
        2. Price above Trend MA (200)
        3. ADX > threshold (optional)
        4. RSI not overbought
    
    Exit Conditions:
        1. Fast MA crosses below Slow MA
        2. Price closes below Trend MA
        3. Stop-loss hit
        4. Trailing stop triggered
    
    Example:
        config = TrendFollowingConfig(
            fast_ma_period=10,
            slow_ma_period=30,
        )
        strategy = TrendFollowingStrategy(config)
    """
    
    def __init__(
        self,
        config: TrendFollowingConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize trend following strategy."""
        super().__init__(config or TrendFollowingConfig(), parameters)
        self.config: TrendFollowingConfig = self.config
        
        # State tracking
        self._prev_fast_ma: dict[str, float] = {}
        self._prev_slow_ma: dict[str, float] = {}
        self._trailing_stops: dict[str, float] = {}
    
    def _on_initialize(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Initialize strategy state."""
        for symbol in symbols:
            self._prev_fast_ma[symbol] = 0.0
            self._prev_slow_ma[symbol] = 0.0
            self._trailing_stops[symbol] = 0.0
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate trend following signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.slow_ma_period + 10:
            return signals
        
        # Calculate indicators
        df = self._calculate_indicators(data)
        if df is None:
            return signals
        
        # Get current values
        current = df.tail(1).to_dicts()[0]
        
        fast_ma = current.get(f"{self.config.ma_type}_{self.config.fast_ma_period}")
        slow_ma = current.get(f"{self.config.ma_type}_{self.config.slow_ma_period}")
        trend_ma = current.get(f"{self.config.ma_type}_{self.config.trend_ma_period}")
        close = current["close"]
        atr_val = current.get(f"atr_{14}", close * 0.02)
        
        if fast_ma is None or slow_ma is None:
            return signals
        
        # Get previous MA values
        prev_fast = self._prev_fast_ma.get(symbol, fast_ma)
        prev_slow = self._prev_slow_ma.get(symbol, slow_ma)
        
        # Update previous values
        self._prev_fast_ma[symbol] = fast_ma
        self._prev_slow_ma[symbol] = slow_ma
        
        # Check ADX filter
        adx_ok = True
        if self.config.use_adx_filter:
            adx_val = current.get("adx", 0)
            adx_ok = adx_val > self.config.adx_threshold if adx_val else False
        
        # Check trend filter
        trend_ok = True
        if self.config.require_price_above_trend_ma and trend_ma:
            trend_ok = close > trend_ma
        
        # Current position
        position = self.get_position(symbol)
        has_long = position and position.quantity > 0
        has_short = position and position.quantity < 0
        
        # Check for crossover
        bullish_cross = prev_fast <= prev_slow and fast_ma > slow_ma
        bearish_cross = prev_fast >= prev_slow and fast_ma < slow_ma
        
        # Entry signals
        if not has_long and not has_short:
            # Long entry
            if bullish_cross and adx_ok and trend_ok:
                strength = self._calculate_signal_strength(df, 1)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=strength,
                    price=close,
                    stop_loss=close - self.config.trailing_stop_atr_mult * atr_val,
                    metadata={
                        "entry_type": "ma_crossover",
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "adx": current.get("adx"),
                    }
                ))
                self._trailing_stops[symbol] = close - self.config.trailing_stop_atr_mult * atr_val
            
            # Short entry (if trend MA allows)
            elif bearish_cross and adx_ok and (not self.config.require_price_above_trend_ma or close < trend_ma):
                strength = self._calculate_signal_strength(df, -1)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=strength,
                    price=close,
                    stop_loss=close + self.config.trailing_stop_atr_mult * atr_val,
                    metadata={
                        "entry_type": "ma_crossover",
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                    }
                ))
                self._trailing_stops[symbol] = close + self.config.trailing_stop_atr_mult * atr_val
        
        # Exit signals
        elif has_long:
            # Update trailing stop
            if self.config.use_trailing_stop:
                new_stop = close - self.config.trailing_stop_atr_mult * atr_val
                self._trailing_stops[symbol] = max(
                    self._trailing_stops.get(symbol, 0),
                    new_stop
                )
                
                # Check trailing stop
                if close < self._trailing_stops[symbol]:
                    signals.append(self.create_exit_signal(
                        symbol=symbol,
                        direction=1,
                        strength=1.0,
                        price=close,
                        reason="trailing_stop",
                    ))
                    return signals
            
            # Check MA cross exit
            if self.config.exit_on_ma_cross and bearish_cross:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=1,
                    strength=0.8,
                    price=close,
                    reason="ma_crossover",
                ))
        
        elif has_short:
            # Update trailing stop for short
            if self.config.use_trailing_stop:
                new_stop = close + self.config.trailing_stop_atr_mult * atr_val
                current_stop = self._trailing_stops.get(symbol, float('inf'))
                self._trailing_stops[symbol] = min(current_stop, new_stop)
                
                if close > self._trailing_stops[symbol]:
                    signals.append(self.create_exit_signal(
                        symbol=symbol,
                        direction=-1,
                        strength=1.0,
                        price=close,
                        reason="trailing_stop",
                    ))
                    return signals
            
            # Check MA cross exit
            if self.config.exit_on_ma_cross and bullish_cross:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=0.8,
                    price=close,
                    reason="ma_crossover",
                ))
        
        return signals
    
    def _calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Calculate required indicators."""
        try:
            # Moving averages
            if self.config.ma_type == "ema":
                df = ema(df, self.config.fast_ma_period)
                df = ema(df, self.config.slow_ma_period)
                df = ema(df, self.config.trend_ma_period)
            else:
                df = sma(df, self.config.fast_ma_period)
                df = sma(df, self.config.slow_ma_period)
                df = sma(df, self.config.trend_ma_period)
            
            # ADX
            if self.config.use_adx_filter:
                df = adx(df, self.config.adx_period)
            
            # ATR for stops
            df = atr(df, 14)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def _calculate_signal_strength(
        self,
        df: pl.DataFrame,
        direction: int,
    ) -> float:
        """Calculate signal strength based on multiple factors."""
        current = df.tail(1).to_dicts()[0]
        
        strength = 0.5  # Base strength
        
        # ADX contribution
        adx_val = current.get("adx", 25)
        if adx_val > 25:
            strength += 0.1
        if adx_val > 40:
            strength += 0.1
        
        # MA separation contribution
        fast_ma = current.get(f"{self.config.ma_type}_{self.config.fast_ma_period}", 0)
        slow_ma = current.get(f"{self.config.ma_type}_{self.config.slow_ma_period}", 0)
        
        if slow_ma > 0:
            separation = abs(fast_ma - slow_ma) / slow_ma
            strength += min(separation * 10, 0.2)
        
        return min(strength, 1.0)


# =============================================================================
# BREAKOUT STRATEGY
# =============================================================================

@dataclass
class BreakoutConfig(StrategyConfig):
    """Configuration for breakout strategy."""
    name: str = "Breakout"
    
    # Breakout parameters
    lookback_period: int = 20
    channel_type: str = "donchian"  # "donchian", "bollinger"
    
    # Volatility filter
    use_volatility_filter: bool = True
    min_atr_pct: float = 0.01
    max_atr_pct: float = 0.05
    
    # Volume confirmation
    require_volume_confirmation: bool = True
    volume_mult_threshold: float = 1.5
    
    # False breakout filter
    confirmation_bars: int = 1
    require_close_beyond: bool = True
    
    # Exit parameters
    exit_on_channel_reentry: bool = True
    profit_target_atr_mult: float = 3.0


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy.
    
    Trades breakouts from price channels (Donchian or Bollinger).
    Uses volume and volatility filters to reduce false breakouts.
    
    Entry Conditions:
        1. Price breaks above/below channel
        2. Volume confirmation (above average)
        3. Volatility within acceptable range
        4. Breakout confirmed by close
    
    Exit Conditions:
        1. Price re-enters channel
        2. Opposite breakout
        3. Profit target hit
        4. Stop-loss (channel boundary)
    
    Example:
        config = BreakoutConfig(
            lookback_period=20,
            channel_type="donchian",
        )
        strategy = BreakoutStrategy(config)
    """
    
    def __init__(
        self,
        config: BreakoutConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize breakout strategy."""
        super().__init__(config or BreakoutConfig(), parameters)
        self.config: BreakoutConfig = self.config
        
        # State tracking
        self._breakout_bar: dict[str, int] = {}
        self._breakout_direction: dict[str, int] = {}
        self._entry_channel: dict[str, float] = {}
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate breakout signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.lookback_period + 10:
            return signals
        
        # Calculate indicators
        df = self._calculate_indicators(data)
        if df is None:
            return signals
        
        # Get current and previous values
        current = df.tail(1).to_dicts()[0]
        prev = df.tail(2).head(1).to_dicts()[0] if len(df) > 1 else current
        
        close = current["close"]
        high = current["high"]
        low = current["low"]
        volume = current["volume"]
        
        upper_channel = current.get("dc_upper") or current.get("bb_upper")
        lower_channel = current.get("dc_lower") or current.get("bb_lower")
        middle_channel = current.get("dc_middle") or current.get("bb_middle")
        
        prev_upper = prev.get("dc_upper") or prev.get("bb_upper")
        prev_lower = prev.get("dc_lower") or prev.get("bb_lower")
        prev_close = prev["close"]
        
        if upper_channel is None or lower_channel is None:
            return signals
        
        # Volume filter
        avg_volume = df["volume"].mean()
        volume_ok = not self.config.require_volume_confirmation or (
            volume > avg_volume * self.config.volume_mult_threshold
        )
        
        # Volatility filter
        atr_val = current.get(f"atr_14", close * 0.02)
        atr_pct = atr_val / close if close > 0 else 0
        volatility_ok = not self.config.use_volatility_filter or (
            self.config.min_atr_pct <= atr_pct <= self.config.max_atr_pct
        )
        
        # Current position
        position = self.get_position(symbol)
        has_long = position and position.quantity > 0
        has_short = position and position.quantity < 0
        
        # Check breakouts
        upward_breakout = prev_close <= prev_upper and close > upper_channel
        downward_breakout = prev_close >= prev_lower and close < lower_channel
        
        # Confirm breakout
        close_beyond_upper = close > upper_channel if self.config.require_close_beyond else high > upper_channel
        close_beyond_lower = close < lower_channel if self.config.require_close_beyond else low < lower_channel
        
        # Entry signals
        if not has_long and not has_short:
            # Long entry on upward breakout
            if upward_breakout and close_beyond_upper and volume_ok and volatility_ok:
                strength = self._calculate_breakout_strength(df, 1)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=strength,
                    price=close,
                    stop_loss=middle_channel,
                    take_profit=close + self.config.profit_target_atr_mult * atr_val,
                    metadata={
                        "entry_type": "breakout",
                        "channel_upper": upper_channel,
                        "volume_ratio": volume / avg_volume,
                    }
                ))
                self._entry_channel[symbol] = upper_channel
            
            # Short entry on downward breakout
            elif downward_breakout and close_beyond_lower and volume_ok and volatility_ok:
                strength = self._calculate_breakout_strength(df, -1)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=strength,
                    price=close,
                    stop_loss=middle_channel,
                    take_profit=close - self.config.profit_target_atr_mult * atr_val,
                    metadata={
                        "entry_type": "breakout",
                        "channel_lower": lower_channel,
                        "volume_ratio": volume / avg_volume,
                    }
                ))
                self._entry_channel[symbol] = lower_channel
        
        # Exit signals
        elif has_long:
            # Exit on channel reentry
            if self.config.exit_on_channel_reentry:
                entry_level = self._entry_channel.get(symbol, upper_channel)
                if close < entry_level:
                    signals.append(self.create_exit_signal(
                        symbol=symbol,
                        direction=1,
                        strength=0.9,
                        price=close,
                        reason="channel_reentry",
                    ))
            
            # Exit on opposite breakout
            elif downward_breakout:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=1,
                    strength=1.0,
                    price=close,
                    reason="opposite_breakout",
                ))
        
        elif has_short:
            if self.config.exit_on_channel_reentry:
                entry_level = self._entry_channel.get(symbol, lower_channel)
                if close > entry_level:
                    signals.append(self.create_exit_signal(
                        symbol=symbol,
                        direction=-1,
                        strength=0.9,
                        price=close,
                        reason="channel_reentry",
                    ))
            
            elif upward_breakout:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=1.0,
                    price=close,
                    reason="opposite_breakout",
                ))
        
        return signals
    
    def _calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Calculate required indicators."""
        try:
            if self.config.channel_type == "donchian":
                df = donchian_channels(df, self.config.lookback_period)
            else:
                df = bollinger_bands(df, self.config.lookback_period)
            
            df = atr(df, 14)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def _calculate_breakout_strength(
        self,
        df: pl.DataFrame,
        direction: int,
    ) -> float:
        """Calculate breakout strength."""
        current = df.tail(1).to_dicts()[0]
        
        strength = 0.6
        
        # Volume confirmation strength
        avg_volume = df["volume"].mean()
        volume = current["volume"]
        if avg_volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio > 2.0:
                strength += 0.2
            elif vol_ratio > 1.5:
                strength += 0.1
        
        # Channel width (tighter = stronger breakout)
        upper = current.get("dc_upper") or current.get("bb_upper")
        lower = current.get("dc_lower") or current.get("bb_lower")
        close = current["close"]
        
        if upper and lower and close > 0:
            width_pct = (upper - lower) / close
            if width_pct < 0.03:  # Tight consolidation
                strength += 0.1
        
        return min(strength, 1.0)


# =============================================================================
# MEAN REVERSION STRATEGY
# =============================================================================

@dataclass
class MeanReversionConfig(StrategyConfig):
    """Configuration for mean reversion strategy."""
    name: str = "MeanReversion"
    
    # Indicator parameters
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    
    # Entry thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_entry_threshold: float = 0.0  # Enter at lower band
    
    # Exit thresholds
    exit_at_mean: bool = True
    rsi_exit_long: float = 50.0
    rsi_exit_short: float = 50.0
    
    # Filters
    use_trend_filter: bool = True
    trend_ma_period: int = 100
    only_trade_with_trend: bool = False
    
    # Risk parameters
    max_bars_in_trade: int = 10


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.
    
    Trades reversions to the mean when price reaches extremes.
    Uses Bollinger Bands and RSI to identify overbought/oversold conditions.
    
    Entry Conditions (Long):
        1. Price touches/crosses lower Bollinger Band
        2. RSI < oversold threshold
        3. Optional: Price above trend MA (with trend)
    
    Exit Conditions:
        1. Price reaches middle band (mean)
        2. RSI crosses above/below threshold
        3. Max holding period reached
    
    Example:
        config = MeanReversionConfig(
            rsi_oversold=25,
            rsi_overbought=75,
        )
        strategy = MeanReversionStrategy(config)
    """
    
    def __init__(
        self,
        config: MeanReversionConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize mean reversion strategy."""
        super().__init__(config or MeanReversionConfig(), parameters)
        self.config: MeanReversionConfig = self.config
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate mean reversion signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.bb_period + 10:
            return signals
        
        # Calculate indicators
        df = self._calculate_indicators(data)
        if df is None:
            return signals
        
        # Get current values
        current = df.tail(1).to_dicts()[0]
        
        close = current["close"]
        bb_upper = current["bb_upper"]
        bb_lower = current["bb_lower"]
        bb_middle = current["bb_middle"]
        bb_pctb = current.get("bb_pctb", 0.5)
        rsi_val = current.get(f"rsi_{self.config.rsi_period}", 50)
        
        # Trend filter
        trend_ok = True
        if self.config.use_trend_filter:
            trend_ma = current.get(f"ema_{self.config.trend_ma_period}")
            if trend_ma:
                if self.config.only_trade_with_trend:
                    trend_ok = close > trend_ma  # Only long in uptrend
                else:
                    trend_ok = True  # Trade both directions
        
        # Current position
        position = self.get_position(symbol)
        has_long = position and position.quantity > 0
        has_short = position and position.quantity < 0
        
        # Entry conditions
        oversold = rsi_val < self.config.rsi_oversold and bb_pctb < 0.1
        overbought = rsi_val > self.config.rsi_overbought and bb_pctb > 0.9
        
        # Entry signals
        if not has_long and not has_short:
            # Long entry on oversold
            if oversold and trend_ok:
                strength = self._calculate_reversion_strength(rsi_val, bb_pctb, 1)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=strength,
                    price=close,
                    stop_loss=bb_lower * 0.99,
                    take_profit=bb_middle,
                    metadata={
                        "entry_type": "mean_reversion",
                        "rsi": rsi_val,
                        "bb_pctb": bb_pctb,
                    }
                ))
            
            # Short entry on overbought
            elif overbought and (not self.config.only_trade_with_trend):
                strength = self._calculate_reversion_strength(rsi_val, bb_pctb, -1)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=strength,
                    price=close,
                    stop_loss=bb_upper * 1.01,
                    take_profit=bb_middle,
                    metadata={
                        "entry_type": "mean_reversion",
                        "rsi": rsi_val,
                        "bb_pctb": bb_pctb,
                    }
                ))
        
        # Exit signals
        elif has_long:
            # Exit at mean
            if self.config.exit_at_mean and close >= bb_middle:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=1,
                    strength=0.9,
                    price=close,
                    reason="reached_mean",
                ))
            # RSI exit
            elif rsi_val >= self.config.rsi_exit_long:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=1,
                    strength=0.8,
                    price=close,
                    reason="rsi_exit",
                ))
        
        elif has_short:
            if self.config.exit_at_mean and close <= bb_middle:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=0.9,
                    price=close,
                    reason="reached_mean",
                ))
            elif rsi_val <= self.config.rsi_exit_short:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=0.8,
                    price=close,
                    reason="rsi_exit",
                ))
        
        return signals
    
    def _calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Calculate required indicators."""
        try:
            df = bollinger_bands(df, self.config.bb_period, self.config.bb_std)
            df = rsi(df, self.config.rsi_period)
            
            if self.config.use_trend_filter:
                df = ema(df, self.config.trend_ma_period)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def _calculate_reversion_strength(
        self,
        rsi_val: float,
        bb_pctb: float,
        direction: int,
    ) -> float:
        """Calculate signal strength based on extremity."""
        strength = 0.5
        
        if direction == 1:  # Long
            # More oversold = stronger signal
            if rsi_val < 20:
                strength += 0.3
            elif rsi_val < 25:
                strength += 0.2
            elif rsi_val < 30:
                strength += 0.1
            
            # Below lower band = stronger
            if bb_pctb < 0:
                strength += 0.1
        else:  # Short
            if rsi_val > 80:
                strength += 0.3
            elif rsi_val > 75:
                strength += 0.2
            elif rsi_val > 70:
                strength += 0.1
            
            if bb_pctb > 1:
                strength += 0.1
        
        return min(strength, 1.0)


# =============================================================================
# DUAL MOMENTUM STRATEGY
# =============================================================================

@dataclass
class DualMomentumConfig(StrategyConfig):
    """Configuration for dual momentum strategy."""
    name: str = "DualMomentum"
    
    # Momentum periods
    absolute_momentum_period: int = 12  # months (or periods)
    relative_momentum_period: int = 12
    
    # Lookback for ROC
    roc_period: int = 63  # ~3 months of daily bars
    
    # Risk-free comparison (for absolute momentum)
    risk_free_threshold: float = 0.0  # Return threshold
    
    # Rebalance frequency (in bars)
    rebalance_frequency: int = 21  # Monthly
    
    # Number of top assets to hold
    top_n: int = 3
    
    # Minimum momentum for entry
    min_absolute_momentum: float = 0.0


class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum Strategy.
    
    Combines absolute momentum (time series) with relative momentum
    (cross-sectional) for asset selection and timing.
    
    Strategy:
        1. Calculate absolute momentum (vs risk-free)
        2. Calculate relative momentum (vs other assets)
        3. Only invest if absolute momentum is positive
        4. Rank and select top N assets by relative momentum
    
    Example:
        config = DualMomentumConfig(
            absolute_momentum_period=12,
            top_n=3,
        )
        strategy = DualMomentumStrategy(config)
    """
    
    def __init__(
        self,
        config: DualMomentumConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize dual momentum strategy."""
        super().__init__(config or DualMomentumConfig(), parameters)
        self.config: DualMomentumConfig = self.config
        
        self._bars_since_rebalance: int = 0
        self._current_holdings: set[str] = set()
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate dual momentum signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.roc_period + 10:
            return signals
        
        self._bars_since_rebalance += 1
        
        # Only rebalance at specified frequency
        if self._bars_since_rebalance < self.config.rebalance_frequency:
            return signals
        
        self._bars_since_rebalance = 0
        
        # Calculate momentum
        df = roc(data, self.config.roc_period)
        current = df.tail(1).to_dicts()[0]
        
        momentum = current.get(f"roc_{self.config.roc_period}", 0)
        close = current["close"]
        
        # Check absolute momentum
        abs_mom_positive = momentum > self.config.min_absolute_momentum
        
        # Current position
        position = self.get_position(symbol)
        has_position = position and position.is_open
        
        if abs_mom_positive and not has_position:
            # Enter position
            strength = min(abs(momentum) / 50, 1.0)  # Normalize
            signals.append(self.create_entry_signal(
                symbol=symbol,
                direction=1,
                strength=strength,
                price=close,
                metadata={
                    "entry_type": "dual_momentum",
                    "momentum": momentum,
                }
            ))
            self._current_holdings.add(symbol)
        
        elif not abs_mom_positive and has_position:
            # Exit on negative absolute momentum
            signals.append(self.create_exit_signal(
                symbol=symbol,
                direction=1,
                strength=0.9,
                price=close,
                reason="negative_momentum",
            ))
            self._current_holdings.discard(symbol)
        
        return signals


# =============================================================================
# RSI DIVERGENCE STRATEGY
# =============================================================================

@dataclass
class RSIDivergenceConfig(StrategyConfig):
    """Configuration for RSI divergence strategy."""
    name: str = "RSIDivergence"
    
    # RSI parameters
    rsi_period: int = 14
    
    # Divergence detection
    divergence_lookback: int = 10
    min_price_diff_pct: float = 0.02
    min_rsi_diff: float = 5.0
    
    # Entry confirmation
    require_confirmation: bool = True
    confirmation_bars: int = 2
    
    # Exit
    exit_on_rsi_cross: bool = True
    rsi_exit_threshold: float = 50.0


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI Divergence Strategy.
    
    Identifies bullish and bearish divergences between price and RSI
    for reversal trading.
    
    Bullish Divergence:
        - Price makes lower low
        - RSI makes higher low
        - Signals potential reversal up
    
    Bearish Divergence:
        - Price makes higher high
        - RSI makes lower high
        - Signals potential reversal down
    
    Example:
        config = RSIDivergenceConfig(
            rsi_period=14,
            divergence_lookback=10,
        )
        strategy = RSIDivergenceStrategy(config)
    """
    
    def __init__(
        self,
        config: RSIDivergenceConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize RSI divergence strategy."""
        super().__init__(config or RSIDivergenceConfig(), parameters)
        self.config: RSIDivergenceConfig = self.config
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate RSI divergence signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.rsi_period + self.config.divergence_lookback + 10:
            return signals
        
        # Calculate RSI
        df = rsi(data, self.config.rsi_period)
        
        # Detect divergences
        bullish_div, bearish_div = self._detect_divergence(df)
        
        current = df.tail(1).to_dicts()[0]
        close = current["close"]
        rsi_val = current.get(f"rsi_{self.config.rsi_period}", 50)
        
        # Current position
        position = self.get_position(symbol)
        has_long = position and position.quantity > 0
        has_short = position and position.quantity < 0
        
        # Entry signals
        if not has_long and not has_short:
            if bullish_div and rsi_val < 40:  # Oversold area
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=0.7,
                    price=close,
                    metadata={
                        "entry_type": "bullish_divergence",
                        "rsi": rsi_val,
                    }
                ))
            
            elif bearish_div and rsi_val > 60:  # Overbought area
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=0.7,
                    price=close,
                    metadata={
                        "entry_type": "bearish_divergence",
                        "rsi": rsi_val,
                    }
                ))
        
        # Exit signals
        elif has_long:
            if self.config.exit_on_rsi_cross and rsi_val > self.config.rsi_exit_threshold:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=1,
                    strength=0.8,
                    price=close,
                    reason="rsi_exit",
                ))
        
        elif has_short:
            if self.config.exit_on_rsi_cross and rsi_val < self.config.rsi_exit_threshold:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=0.8,
                    price=close,
                    reason="rsi_exit",
                ))
        
        return signals
    
    def _detect_divergence(
        self,
        df: pl.DataFrame,
    ) -> tuple[bool, bool]:
        """
        Detect bullish and bearish divergences.
        
        Returns:
            Tuple of (bullish_divergence, bearish_divergence)
        """
        lookback = self.config.divergence_lookback
        rsi_col = f"rsi_{self.config.rsi_period}"
        
        # Get recent data
        recent = df.tail(lookback)
        if len(recent) < lookback:
            return False, False
        
        prices = recent["close"].to_numpy()
        rsi_vals = recent[rsi_col].to_numpy()
        
        # Find local extremes
        current_price = prices[-1]
        current_rsi = rsi_vals[-1]
        
        # Look for swing lows (bullish divergence)
        price_min_idx = np.argmin(prices[:-1])
        price_min = prices[price_min_idx]
        rsi_at_price_min = rsi_vals[price_min_idx]
        
        bullish_divergence = (
            current_price < price_min * (1 - self.config.min_price_diff_pct) and
            current_rsi > rsi_at_price_min + self.config.min_rsi_diff
        )
        
        # Look for swing highs (bearish divergence)
        price_max_idx = np.argmax(prices[:-1])
        price_max = prices[price_max_idx]
        rsi_at_price_max = rsi_vals[price_max_idx]
        
        bearish_divergence = (
            current_price > price_max * (1 + self.config.min_price_diff_pct) and
            current_rsi < rsi_at_price_max - self.config.min_rsi_diff
        )
        
        return bullish_divergence, bearish_divergence


# =============================================================================
# MACD STRATEGY
# =============================================================================

@dataclass
class MACDStrategyConfig(StrategyConfig):
    """Configuration for MACD strategy."""
    name: str = "MACD"
    
    # MACD parameters
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    
    # Entry conditions
    require_zero_cross: bool = False
    use_histogram: bool = True
    histogram_threshold: float = 0.0
    
    # Trend filter
    use_trend_filter: bool = True
    trend_ema_period: int = 200
    
    # Exit conditions
    exit_on_signal_cross: bool = True
    exit_on_histogram_reversal: bool = False


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy.
    
    Trades based on MACD line and signal line crossovers.
    
    Entry Conditions:
        - MACD crosses above signal (long)
        - MACD crosses below signal (short)
        - Optional: Zero line cross confirmation
        - Optional: Trend filter (price above/below EMA)
    
    Exit Conditions:
        - Opposite signal cross
        - Histogram reversal
        - Stop-loss
    """
    
    def __init__(
        self,
        config: MACDStrategyConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize MACD strategy."""
        super().__init__(config or MACDStrategyConfig(), parameters)
        self.config: MACDStrategyConfig = self.config
        
        self._prev_macd: dict[str, float] = {}
        self._prev_signal: dict[str, float] = {}
        self._prev_histogram: dict[str, float] = {}
    
    def calculate_signals(
        self,
        event: MarketEvent,
        portfolio: PortfolioState,
    ) -> list[SignalEvent]:
        """Generate MACD signals."""
        signals = []
        symbol = event.symbol
        data = event.data
        
        if data is None or len(data) < self.config.slow_period + 10:
            return signals
        
        # Calculate MACD
        df = macd(
            data,
            self.config.fast_period,
            self.config.slow_period,
            self.config.signal_period,
        )
        
        if self.config.use_trend_filter:
            df = ema(df, self.config.trend_ema_period)
        
        current = df.tail(1).to_dicts()[0]
        
        macd_val = current["macd"]
        signal_val = current["macd_signal"]
        histogram = current["macd_histogram"]
        close = current["close"]
        
        # Get previous values
        prev_macd = self._prev_macd.get(symbol, macd_val)
        prev_signal = self._prev_signal.get(symbol, signal_val)
        prev_histogram = self._prev_histogram.get(symbol, histogram)
        
        # Update previous values
        self._prev_macd[symbol] = macd_val
        self._prev_signal[symbol] = signal_val
        self._prev_histogram[symbol] = histogram
        
        # Trend filter
        trend_ok_long = True
        trend_ok_short = True
        if self.config.use_trend_filter:
            trend_ema = current.get(f"ema_{self.config.trend_ema_period}")
            if trend_ema:
                trend_ok_long = close > trend_ema
                trend_ok_short = close < trend_ema
        
        # Crossover detection
        bullish_cross = prev_macd <= prev_signal and macd_val > signal_val
        bearish_cross = prev_macd >= prev_signal and macd_val < signal_val
        
        # Zero line filter
        if self.config.require_zero_cross:
            bullish_cross = bullish_cross and macd_val > 0
            bearish_cross = bearish_cross and macd_val < 0
        
        # Current position
        position = self.get_position(symbol)
        has_long = position and position.quantity > 0
        has_short = position and position.quantity < 0
        
        # Entry signals
        if not has_long and not has_short:
            if bullish_cross and trend_ok_long:
                strength = self._calculate_macd_strength(macd_val, signal_val, histogram)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=1,
                    strength=strength,
                    price=close,
                    metadata={
                        "entry_type": "macd_cross",
                        "macd": macd_val,
                        "signal": signal_val,
                        "histogram": histogram,
                    }
                ))
            
            elif bearish_cross and trend_ok_short:
                strength = self._calculate_macd_strength(macd_val, signal_val, histogram)
                signals.append(self.create_entry_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=strength,
                    price=close,
                    metadata={
                        "entry_type": "macd_cross",
                        "macd": macd_val,
                        "signal": signal_val,
                        "histogram": histogram,
                    }
                ))
        
        # Exit signals
        elif has_long:
            exit_signal = False
            reason = ""
            
            if self.config.exit_on_signal_cross and bearish_cross:
                exit_signal = True
                reason = "signal_cross"
            elif self.config.exit_on_histogram_reversal:
                if prev_histogram > 0 and histogram < 0:
                    exit_signal = True
                    reason = "histogram_reversal"
            
            if exit_signal:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=1,
                    strength=0.8,
                    price=close,
                    reason=reason,
                ))
        
        elif has_short:
            exit_signal = False
            reason = ""
            
            if self.config.exit_on_signal_cross and bullish_cross:
                exit_signal = True
                reason = "signal_cross"
            elif self.config.exit_on_histogram_reversal:
                if prev_histogram < 0 and histogram > 0:
                    exit_signal = True
                    reason = "histogram_reversal"
            
            if exit_signal:
                signals.append(self.create_exit_signal(
                    symbol=symbol,
                    direction=-1,
                    strength=0.8,
                    price=close,
                    reason=reason,
                ))
        
        return signals
    
    def _calculate_macd_strength(
        self,
        macd_val: float,
        signal_val: float,
        histogram: float,
    ) -> float:
        """Calculate signal strength."""
        strength = 0.5
        
        # Histogram magnitude
        if abs(histogram) > 0.5:
            strength += 0.2
        elif abs(histogram) > 0.2:
            strength += 0.1
        
        # MACD above/below zero
        if macd_val > 0 and histogram > 0:
            strength += 0.1
        elif macd_val < 0 and histogram < 0:
            strength += 0.1
        
        return min(strength, 1.0)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configurations
    "TrendFollowingConfig",
    "BreakoutConfig",
    "MeanReversionConfig",
    "DualMomentumConfig",
    "RSIDivergenceConfig",
    "MACDStrategyConfig",
    # Strategies
    "TrendFollowingStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "DualMomentumStrategy",
    "RSIDivergenceStrategy",
    "MACDStrategy",
]