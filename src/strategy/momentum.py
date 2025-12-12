"""
Momentum Strategies
JPMorgan-Level Trend Following Systems

Features:
- Classic momentum
- Trend following with multiple timeframes
- Breakout strategies
- Momentum with regime filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .base_strategy import (
    BaseStrategy, Signal, SignalType, StrategyConfig
)
from ..features.technical import TechnicalIndicators, AdvancedTechnicals
from ..utils.logger import get_logger


logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Classic momentum strategy.

    Buys assets with strong recent performance,
    sells assets with weak recent performance.
    """

    def __init__(
        self,
        lookback_periods: List[int] = None,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        use_volume_confirmation: bool = True,
        **kwargs
    ):
        """
        Initialize MomentumStrategy.

        Args:
            lookback_periods: Momentum lookback periods
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            use_volume_confirmation: Require volume confirmation
        """
        config = StrategyConfig(name="momentum", **kwargs)
        super().__init__(config)

        self.lookback_periods = lookback_periods or [5, 10, 20]
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_volume_confirmation = use_volume_confirmation

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate momentum signals for all symbols"""
        signals = {}

        for symbol, df in data.items():
            try:
                signal_type, strength = self.calculate_signal_strength(df, symbol)

                if signal_type != SignalType.FLAT:
                    signal = self._create_signal(df, symbol, signal_type, strength)

                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {e}")

        return signals

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Calculate momentum signal and strength"""
        if len(df) < max(self.lookback_periods) + 10:
            return SignalType.FLAT, 0.0

        close = df['close']
        volume = df['volume']

        # Calculate momentum scores for each lookback
        momentum_scores = []
        for period in self.lookback_periods:
            ret = close.pct_change(period).iloc[-1]
            # Normalize return to roughly -1 to 1
            normalized = np.tanh(ret * 10)
            momentum_scores.append(normalized)

        avg_momentum = np.mean(momentum_scores)

        # RSI component
        rsi = TechnicalIndicators.rsi(close, self.rsi_period).iloc[-1]
        rsi_score = (rsi - 50) / 50  # -1 to 1

        # Volume confirmation
        vol_score = 1.0
        if self.use_volume_confirmation:
            vol_ma = volume.rolling(20).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
            vol_score = min(vol_ratio, 2) / 2  # Cap at 2x average

        # Combine scores
        combined_score = (
            0.5 * avg_momentum +
            0.3 * rsi_score +
            0.2 * (vol_score - 0.5) * 2
        )

        # Determine signal type
        if combined_score > 0.3 and rsi < self.rsi_overbought:
            signal_type = SignalType.LONG
        elif combined_score < -0.3 and rsi > self.rsi_oversold:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT

        strength = min(abs(combined_score), 1.0)

        return signal_type, strength

    def _create_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        strength: float
    ) -> Signal:
        """Create signal with stop loss and take profit"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type == SignalType.LONG else -1

        # ATR-based stop loss
        atr = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'], 14
        ).iloc[-1]

        stop_loss = self.calculate_stop_loss(current_price, direction, atr)
        take_profit = self.calculate_take_profit(current_price, direction, stop_loss)

        # Calculate confidence based on trend strength
        adx, _, _ = TechnicalIndicators.adx(
            df['high'], df['low'], df['close'], 14
        )
        trend_strength = adx.iloc[-1] / 100 if not pd.isna(adx.iloc[-1]) else 0.5

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=trend_strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'atr': atr,
                'rsi': TechnicalIndicators.rsi(df['close'], 14).iloc[-1]
            }
        )


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using multiple indicators.

    Features:
    - Moving average crossovers
    - ADX trend strength filter
    - Supertrend confirmation
    - Multi-timeframe analysis
    """

    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        trend_ma: int = 200,
        adx_threshold: float = 25,
        use_supertrend: bool = True,
        **kwargs
    ):
        """
        Initialize TrendFollowingStrategy.

        Args:
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period
            trend_ma: Long-term trend MA period
            adx_threshold: Minimum ADX for trend trades
            use_supertrend: Use Supertrend confirmation
        """
        config = StrategyConfig(name="trend_following", **kwargs)
        super().__init__(config)

        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.trend_ma = trend_ma
        self.adx_threshold = adx_threshold
        self.use_supertrend = use_supertrend

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate trend following signals"""
        signals = {}

        for symbol, df in data.items():
            try:
                signal_type, strength = self.calculate_signal_strength(df, symbol)

                if signal_type != SignalType.FLAT:
                    signal = self._create_signal(df, symbol, signal_type, strength)

                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error in trend following for {symbol}: {e}")

        return signals

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Calculate trend following signal"""
        if len(df) < self.trend_ma + 10:
            return SignalType.FLAT, 0.0

        close = df['close']

        # Moving averages
        ema_fast = TechnicalIndicators.ema(close, self.fast_ma)
        ema_slow = TechnicalIndicators.ema(close, self.slow_ma)
        sma_trend = TechnicalIndicators.sma(close, self.trend_ma)

        # Current values
        price = close.iloc[-1]
        fast = ema_fast.iloc[-1]
        slow = ema_slow.iloc[-1]
        trend = sma_trend.iloc[-1]

        # ADX trend strength
        adx, plus_di, minus_di = TechnicalIndicators.adx(
            df['high'], df['low'], df['close'], 14
        )
        adx_val = adx.iloc[-1]

        # Supertrend
        if self.use_supertrend:
            supertrend, st_direction = TechnicalIndicators.supertrend(
                df['high'], df['low'], df['close']
            )
            st_dir = st_direction.iloc[-1]
        else:
            st_dir = 0

        # Scoring components
        ma_score = 0
        if fast > slow > trend:
            ma_score = 1
        elif fast < slow < trend:
            ma_score = -1
        elif fast > slow:
            ma_score = 0.5
        elif fast < slow:
            ma_score = -0.5

        # Price position
        price_score = 0
        if price > trend:
            price_score = 0.5
        elif price < trend:
            price_score = -0.5

        # Trend strength from ADX
        trend_score = min(adx_val / 50, 1) if not pd.isna(adx_val) else 0.5

        # DI difference
        di_score = 0
        if not pd.isna(plus_di.iloc[-1]) and not pd.isna(minus_di.iloc[-1]):
            di_diff = plus_di.iloc[-1] - minus_di.iloc[-1]
            di_score = np.tanh(di_diff / 20)

        # Combine scores
        combined = (
            0.3 * ma_score +
            0.2 * price_score +
            0.2 * di_score +
            0.3 * (st_dir if self.use_supertrend else ma_score)
        )

        # Require minimum ADX
        if adx_val < self.adx_threshold:
            combined *= 0.5  # Reduce signal in low trend environment

        # Determine signal
        if combined > 0.4:
            signal_type = SignalType.LONG
        elif combined < -0.4:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT

        strength = min(abs(combined) * trend_score, 1.0)

        return signal_type, strength

    def _create_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        strength: float
    ) -> Signal:
        """Create trend following signal"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type == SignalType.LONG else -1

        # Use Supertrend for stop loss
        supertrend, _ = TechnicalIndicators.supertrend(
            df['high'], df['low'], df['close']
        )

        if direction == 1:
            stop_loss = supertrend.iloc[-1]
        else:
            stop_loss = supertrend.iloc[-1]

        take_profit = self.calculate_take_profit(current_price, direction, stop_loss)

        # Confidence from ADX
        adx, _, _ = TechnicalIndicators.adx(
            df['high'], df['low'], df['close'], 14
        )

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=min(adx.iloc[-1] / 50, 1.0) if not pd.isna(adx.iloc[-1]) else 0.5,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'adx': adx.iloc[-1],
                'supertrend': supertrend.iloc[-1]
            }
        )


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy based on price channels.

    Features:
    - Donchian channel breakouts
    - Bollinger band breakouts
    - Volume confirmation
    - False breakout filtering
    """

    def __init__(
        self,
        channel_period: int = 20,
        breakout_confirmation: int = 2,
        volume_multiplier: float = 1.5,
        **kwargs
    ):
        config = StrategyConfig(name="breakout", **kwargs)
        super().__init__(config)

        self.channel_period = channel_period
        self.breakout_confirmation = breakout_confirmation
        self.volume_multiplier = volume_multiplier

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate breakout signals"""
        signals = {}

        for symbol, df in data.items():
            try:
                signal_type, strength = self.calculate_signal_strength(df, symbol)

                if signal_type != SignalType.FLAT:
                    signal = self._create_signal(df, symbol, signal_type, strength)

                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error in breakout strategy for {symbol}: {e}")

        return signals

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Calculate breakout signal"""
        if len(df) < self.channel_period + 10:
            return SignalType.FLAT, 0.0

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Donchian channels
        dc_upper, dc_middle, dc_lower = TechnicalIndicators.donchian_channel(
            high, low, self.channel_period
        )

        # Current values
        current_close = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]

        # Previous channel values
        prev_upper = dc_upper.iloc[-2]
        prev_lower = dc_lower.iloc[-2]

        # Volume check
        vol_ma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
        volume_confirmed = vol_ratio >= self.volume_multiplier

        # Breakout detection
        signal_type = SignalType.FLAT
        strength = 0.0

        # Upside breakout
        if current_high > prev_upper:
            # Check for confirmation
            recent_highs = high.iloc[-self.breakout_confirmation:]
            if all(h > dc_upper.iloc[-self.breakout_confirmation:].values[i]
                   for i, h in enumerate(recent_highs)):
                signal_type = SignalType.LONG
                breakout_strength = (current_close - prev_upper) / prev_upper
                strength = min(abs(breakout_strength) * 10, 1.0)

        # Downside breakout
        elif current_low < prev_lower:
            recent_lows = low.iloc[-self.breakout_confirmation:]
            if all(l < dc_lower.iloc[-self.breakout_confirmation:].values[i]
                   for i, l in enumerate(recent_lows)):
                signal_type = SignalType.SHORT
                breakout_strength = (prev_lower - current_close) / prev_lower
                strength = min(abs(breakout_strength) * 10, 1.0)

        # Reduce strength if volume not confirmed
        if not volume_confirmed:
            strength *= 0.5

        return signal_type, strength

    def _create_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        strength: float
    ) -> Signal:
        """Create breakout signal"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type == SignalType.LONG else -1

        dc_upper, dc_middle, dc_lower = TechnicalIndicators.donchian_channel(
            df['high'], df['low'], self.channel_period
        )

        # Stop at channel middle
        if direction == 1:
            stop_loss = dc_middle.iloc[-1]
        else:
            stop_loss = dc_middle.iloc[-1]

        take_profit = self.calculate_take_profit(current_price, direction, stop_loss)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'channel_upper': dc_upper.iloc[-1],
                'channel_lower': dc_lower.iloc[-1]
            }
        )
