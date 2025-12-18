"""
Technical indicators for AlphaTrade system.

This module provides 50+ technical indicators across categories:
- Trend indicators (MA, MACD, ADX, etc.)
- Momentum indicators (RSI, Stochastic, etc.)
- Volatility indicators (Bollinger, ATR, etc.)
- Volume indicators (OBV, VWAP, MFI, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import jit
from loguru import logger


class TechnicalIndicators:
    """
    Comprehensive technical indicator calculator.

    Provides vectorized implementations of 50+ technical indicators
    with configurable parameters.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        """
        self.df = df.copy()
        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validate required columns exist."""
        required = ["open", "high", "low", "close"]
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================================
    # TREND INDICATORS
    # =========================================================================

    def sma(self, period: int = 20, column: str = "close") -> pd.Series:
        """Simple Moving Average."""
        return self.df[column].rolling(window=period).mean()

    def ema(self, period: int = 20, column: str = "close") -> pd.Series:
        """Exponential Moving Average."""
        return self.df[column].ewm(span=period, adjust=False).mean()

    def wma(self, period: int = 20, column: str = "close") -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return self.df[column].rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def dema(self, period: int = 20, column: str = "close") -> pd.Series:
        """Double Exponential Moving Average."""
        ema1 = self.ema(period, column)
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    def tema(self, period: int = 20, column: str = "close") -> pd.Series:
        """Triple Exponential Moving Average."""
        ema1 = self.ema(period, column)
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = self.ema(fast)
        ema_slow = self.ema(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def adx(self, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index.

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    def parabolic_sar(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.2,
    ) -> pd.Series:
        """Parabolic SAR indicator."""
        high = self.df["high"].values
        low = self.df["low"].values
        close = self.df["close"].values
        n = len(close)

        sar = np.zeros(n)
        trend = np.zeros(n)
        af = af_start
        ep = low[0]

        sar[0] = high[0]
        trend[0] = -1

        for i in range(1, n):
            if trend[i - 1] == 1:  # Uptrend
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])

                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, af_max)
            else:  # Downtrend
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])

                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, af_max)

        return pd.Series(sar, index=self.df.index)

    def ichimoku(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
    ) -> dict[str, pd.Series]:
        """
        Ichimoku Cloud components.

        Returns:
            Dictionary with tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span
        """
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (
            high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()
        ) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (
            high.rolling(window=kijun).max() + low.rolling(window=kijun).min()
        ) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

        # Senkou Span B (Leading Span B)
        senkou_b_line = (
            (high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2
        ).shift(kijun)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b_line,
            "chikou_span": chikou_span,
        }

    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================

    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = self.df["close"].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Returns:
            Tuple of (%K, %D)
        """
        low_min = self.df["low"].rolling(window=k_period).min()
        high_max = self.df["high"].rolling(window=k_period).max()

        stoch_k = 100 * (self.df["close"] - low_min) / (high_max - low_min)
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R."""
        high_max = self.df["high"].rolling(window=period).max()
        low_min = self.df["low"].rolling(window=period).min()

        wr = -100 * (high_max - self.df["close"]) / (high_max - low_min)
        return wr

    def cci(self, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        return cci

    def roc(self, period: int = 10) -> pd.Series:
        """Rate of Change."""
        return ((self.df["close"] - self.df["close"].shift(period)) /
                self.df["close"].shift(period)) * 100

    def momentum(self, period: int = 10) -> pd.Series:
        """Momentum indicator."""
        return self.df["close"] - self.df["close"].shift(period)

    def tsi(self, long_period: int = 25, short_period: int = 13) -> pd.Series:
        """True Strength Index."""
        price_change = self.df["close"].diff()

        double_smoothed = price_change.ewm(span=long_period, adjust=False).mean()
        double_smoothed = double_smoothed.ewm(span=short_period, adjust=False).mean()

        abs_double_smoothed = price_change.abs().ewm(span=long_period, adjust=False).mean()
        abs_double_smoothed = abs_double_smoothed.ewm(span=short_period, adjust=False).mean()

        tsi = 100 * double_smoothed / abs_double_smoothed
        return tsi

    def ultimate_oscillator(
        self,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator."""
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - \
             pd.concat([low, close.shift(1)], axis=1).min(axis=1)

        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()

        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo

    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================

    def bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict[str, pd.Series]:
        """
        Bollinger Bands.

        Returns:
            Dictionary with upper, middle, lower, bandwidth, percent_b
        """
        middle = self.sma(period)
        std = self.df["close"].rolling(window=period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        bandwidth = (upper - lower) / middle * 100
        percent_b = (self.df["close"] - lower) / (upper - lower)

        return {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def keltner_channels(
        self,
        period: int = 20,
        atr_mult: float = 2.0,
    ) -> dict[str, pd.Series]:
        """
        Keltner Channels.

        Returns:
            Dictionary with upper, middle, lower
        """
        middle = self.ema(period)
        atr = self.atr(period)

        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)

        return {
            "kc_upper": upper,
            "kc_middle": middle,
            "kc_lower": lower,
        }

    def donchian_channels(self, period: int = 20) -> dict[str, pd.Series]:
        """
        Donchian Channels.

        Returns:
            Dictionary with upper, middle, lower
        """
        upper = self.df["high"].rolling(window=period).max()
        lower = self.df["low"].rolling(window=period).min()
        middle = (upper + lower) / 2

        return {
            "dc_upper": upper,
            "dc_middle": middle,
            "dc_lower": lower,
        }

    def historical_volatility(self, period: int = 20) -> pd.Series:
        """Historical volatility (annualized)."""
        log_returns = np.log(self.df["close"] / self.df["close"].shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252 * 26)  # 26 bars per day

    def garman_klass_volatility(self, period: int = 20) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(self.df["high"] / self.df["low"]) ** 2
        log_co = np.log(self.df["close"] / self.df["open"]) ** 2

        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return gk.rolling(window=period).mean().apply(np.sqrt) * np.sqrt(252 * 26)

    def parkinson_volatility(self, period: int = 20) -> pd.Series:
        """Parkinson volatility estimator."""
        log_hl = np.log(self.df["high"] / self.df["low"]) ** 2
        factor = 1 / (4 * np.log(2))

        return (factor * log_hl).rolling(window=period).mean().apply(np.sqrt) * np.sqrt(252 * 26)

    def yang_zhang_volatility(self, period: int = 20) -> pd.Series:
        """Yang-Zhang volatility estimator."""
        log_ho = np.log(self.df["high"] / self.df["open"])
        log_lo = np.log(self.df["low"] / self.df["open"])
        log_co = np.log(self.df["close"] / self.df["open"])
        log_oc = np.log(self.df["open"] / self.df["close"].shift(1))
        log_cc = np.log(self.df["close"] / self.df["close"].shift(1))

        # Overnight volatility
        overnight_var = log_oc.rolling(window=period).var()

        # Open-to-close volatility
        open_close_var = log_co.rolling(window=period).var()

        # Rogers-Satchell volatility
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        rs_var = rs.rolling(window=period).mean()

        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var

        return yz_var.apply(np.sqrt) * np.sqrt(252 * 26)

    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================

    def obv(self) -> pd.Series:
        """On-Balance Volume."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        sign = np.sign(self.df["close"].diff())
        return (sign * self.df["volume"]).cumsum()

    def vwap(self) -> pd.Series:
        """Volume Weighted Average Price."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        return (typical_price * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def mfi(self, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        raw_money_flow = typical_price * self.df["volume"]

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    def accumulation_distribution(self) -> pd.Series:
        """Accumulation/Distribution Line."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        clv = ((self.df["close"] - self.df["low"]) -
               (self.df["high"] - self.df["close"])) / (self.df["high"] - self.df["low"])
        clv = clv.fillna(0)

        return (clv * self.df["volume"]).cumsum()

    def chaikin_money_flow(self, period: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        clv = ((self.df["close"] - self.df["low"]) -
               (self.df["high"] - self.df["close"])) / (self.df["high"] - self.df["low"])
        clv = clv.fillna(0)

        mf_volume = clv * self.df["volume"]
        cmf = mf_volume.rolling(window=period).sum() / self.df["volume"].rolling(window=period).sum()

        return cmf

    def volume_roc(self, period: int = 10) -> pd.Series:
        """Volume Rate of Change."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        return ((self.df["volume"] - self.df["volume"].shift(period)) /
                self.df["volume"].shift(period)) * 100

    def force_index(self, period: int = 13) -> pd.Series:
        """Force Index."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        fi = self.df["close"].diff() * self.df["volume"]
        return fi.ewm(span=period, adjust=False).mean()

    def ease_of_movement(self, period: int = 14) -> pd.Series:
        """Ease of Movement."""
        if "volume" not in self.df.columns:
            return pd.Series(index=self.df.index, dtype=float)

        dm = ((self.df["high"] + self.df["low"]) / 2 -
              (self.df["high"].shift(1) + self.df["low"].shift(1)) / 2)
        br = self.df["volume"] / (self.df["high"] - self.df["low"])

        eom = dm / br
        return eom.rolling(window=period).mean()

    # =========================================================================
    # COMPREHENSIVE FEATURE GENERATION
    # =========================================================================

    def generate_all_features(
        self,
        ma_periods: list[int] | None = None,
        rsi_periods: list[int] | None = None,
        adaptive_periods: list[int] | None = None,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Generate all technical indicators as features.

        Args:
            ma_periods: Periods for moving averages
            rsi_periods: Periods for RSI
            adaptive_periods: Periods for adaptive indicators (KAMA, FRAMA, VIDYA, ALMA)
            prefix: Prefix for feature names

        Returns:
            DataFrame with all technical features
        """
        if ma_periods is None:
            ma_periods = [5, 10, 20, 50, 100, 200]
        if rsi_periods is None:
            rsi_periods = [7, 14, 21]
        if adaptive_periods is None:
            adaptive_periods = [10, 20, 50]

        features = pd.DataFrame(index=self.df.index)
        p = prefix

        logger.info("Generating technical indicators...")

        # Moving averages
        for period in ma_periods:
            features[f"{p}sma_{period}"] = self.sma(period)
            features[f"{p}ema_{period}"] = self.ema(period)

            # Price relative to MA
            features[f"{p}price_sma_{period}_ratio"] = self.df["close"] / self.sma(period)
            features[f"{p}price_ema_{period}_ratio"] = self.df["close"] / self.ema(period)

        # MA crossovers
        features[f"{p}sma_5_20_cross"] = (self.sma(5) > self.sma(20)).astype(int)
        features[f"{p}ema_12_26_cross"] = (self.ema(12) > self.ema(26)).astype(int)

        # MACD
        macd_line, signal_line, histogram = self.macd()
        features[f"{p}macd_line"] = macd_line
        features[f"{p}macd_signal"] = signal_line
        features[f"{p}macd_hist"] = histogram

        # ADX
        adx_val, plus_di, minus_di = self.adx()
        features[f"{p}adx"] = adx_val
        features[f"{p}plus_di"] = plus_di
        features[f"{p}minus_di"] = minus_di
        features[f"{p}di_diff"] = plus_di - minus_di

        # Parabolic SAR
        features[f"{p}psar"] = self.parabolic_sar()
        features[f"{p}psar_trend"] = (self.df["close"] > features[f"{p}psar"]).astype(int)

        # Ichimoku
        ichi = self.ichimoku()
        for name, series in ichi.items():
            features[f"{p}{name}"] = series

        # RSI
        for period in rsi_periods:
            features[f"{p}rsi_{period}"] = self.rsi(period)

        # Stochastic
        stoch_k, stoch_d = self.stochastic()
        features[f"{p}stoch_k"] = stoch_k
        features[f"{p}stoch_d"] = stoch_d

        # Williams %R
        features[f"{p}williams_r"] = self.williams_r()

        # CCI
        features[f"{p}cci"] = self.cci()

        # ROC
        for period in [5, 10, 20]:
            features[f"{p}roc_{period}"] = self.roc(period)

        # Momentum
        for period in [5, 10, 20]:
            features[f"{p}momentum_{period}"] = self.momentum(period)

        # TSI
        features[f"{p}tsi"] = self.tsi()

        # Ultimate Oscillator
        features[f"{p}ultimate_osc"] = self.ultimate_oscillator()

        # Bollinger Bands
        bb = self.bollinger_bands()
        for name, series in bb.items():
            features[f"{p}{name}"] = series

        # ATR
        features[f"{p}atr"] = self.atr()
        features[f"{p}atr_pct"] = self.atr() / self.df["close"] * 100

        # Keltner Channels
        kc = self.keltner_channels()
        for name, series in kc.items():
            features[f"{p}{name}"] = series

        # Donchian Channels
        dc = self.donchian_channels()
        for name, series in dc.items():
            features[f"{p}{name}"] = series

        # Volatility measures
        features[f"{p}hist_vol"] = self.historical_volatility()
        features[f"{p}gk_vol"] = self.garman_klass_volatility()
        features[f"{p}parkinson_vol"] = self.parkinson_volatility()

        # Volume indicators
        if "volume" in self.df.columns:
            features[f"{p}obv"] = self.obv()
            features[f"{p}vwap"] = self.vwap()
            features[f"{p}mfi"] = self.mfi()
            features[f"{p}ad_line"] = self.accumulation_distribution()
            features[f"{p}cmf"] = self.chaikin_money_flow()
            features[f"{p}volume_roc"] = self.volume_roc()
            features[f"{p}force_index"] = self.force_index()

            # Volume ratios
            vol_ma = self.df["volume"].rolling(20).mean()
            features[f"{p}volume_ratio"] = self.df["volume"] / vol_ma

        # Adaptive Indicators
        for period in adaptive_periods:
            features[f"{p}kama_{period}"] = self.kama(period)
            features[f"{p}frama_{period}"] = self.frama(period)
            features[f"{p}vidya_{period}"] = self.vidya(period)
            features[f"{p}alma_{period}"] = self.alma(period)

        # Adaptive ATR
        features[f"{p}adaptive_atr"] = self.adaptive_atr()

        # McGinley Dynamic
        features[f"{p}mcginley_dynamic"] = self.mcginley_dynamic()

        logger.info(f"Generated {len(features.columns)} technical features")
        return features

    # =========================================================================
    # ADAPTIVE INDICATORS - JPMorgan Institutional Level
    # =========================================================================

    def kama(
        self,
        period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        column: str = "close",
    ) -> pd.Series:
        """
        Kaufman's Adaptive Moving Average (KAMA).

        KAMA adapts to volatility by using an Efficiency Ratio (ER) to adjust
        the smoothing constant. In trending markets, KAMA moves faster; in
        ranging markets, it smooths more heavily.

        Args:
            period: Lookback period for efficiency ratio calculation
            fast_period: Fast EMA period (used in trending markets)
            slow_period: Slow EMA period (used in ranging markets)
            column: Column to use

        Returns:
            KAMA series
        """
        price = self.df[column].values
        n = len(price)

        # Calculate efficiency ratio
        # ER = abs(price_change over period) / sum(abs(individual changes))
        change = np.abs(price[period:] - price[:-period])
        volatility = np.zeros(n)

        for i in range(period, n):
            volatility[i] = np.sum(np.abs(np.diff(price[i - period:i + 1])))

        # Avoid division by zero
        volatility = np.where(volatility == 0, 1, volatility)

        er = np.zeros(n)
        er[period:] = change / volatility[period:]

        # Calculate smoothing constants
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)

        # Smoothing constant = (ER * (fast_sc - slow_sc) + slow_sc)^2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Calculate KAMA
        kama = np.zeros(n)
        kama[period - 1] = price[period - 1]

        for i in range(period, n):
            kama[i] = kama[i - 1] + sc[i] * (price[i] - kama[i - 1])

        # Set initial values to NaN
        kama[:period - 1] = np.nan

        return pd.Series(kama, index=self.df.index, name=f"kama_{period}")

    def frama(self, period: int = 16, column: str = "close") -> pd.Series:
        """
        Fractal Adaptive Moving Average (FRAMA).

        Uses fractal dimension to adjust smoothing. In trending markets (low
        fractal dimension), FRAMA follows price closely. In ranging markets
        (high fractal dimension), it smooths more.

        Args:
            period: Lookback period (must be even)
            column: Column to use

        Returns:
            FRAMA series
        """
        price = self.df[column].values
        n = len(price)

        # Ensure period is even
        period = period if period % 2 == 0 else period + 1
        half_period = period // 2

        frama = np.zeros(n)
        alpha = np.zeros(n)

        for i in range(period, n):
            # Get high and low over the period
            high1 = np.max(price[i - period:i - half_period])
            low1 = np.min(price[i - period:i - half_period])
            high2 = np.max(price[i - half_period:i])
            low2 = np.min(price[i - half_period:i])
            high3 = np.max(price[i - period:i])
            low3 = np.min(price[i - period:i])

            n1 = (high1 - low1) / half_period if half_period > 0 else 0
            n2 = (high2 - low2) / half_period if half_period > 0 else 0
            n3 = (high3 - low3) / period if period > 0 else 0

            if n1 + n2 > 0 and n3 > 0:
                d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
                alpha[i] = np.exp(-4.6 * (d - 1))
                alpha[i] = max(0.01, min(alpha[i], 1))
            else:
                alpha[i] = 0.5

        # Calculate FRAMA
        frama[period - 1] = price[period - 1]
        for i in range(period, n):
            frama[i] = alpha[i] * price[i] + (1 - alpha[i]) * frama[i - 1]

        frama[:period - 1] = np.nan

        return pd.Series(frama, index=self.df.index, name=f"frama_{period}")

    def vidya(
        self,
        period: int = 14,
        cmo_period: int = 9,
        column: str = "close",
    ) -> pd.Series:
        """
        Variable Index Dynamic Average (VIDYA).

        Uses Chande Momentum Oscillator (CMO) as the volatility index to
        adaptively adjust smoothing. Tushar Chande's improvement on
        traditional moving averages.

        Args:
            period: EMA period
            cmo_period: CMO lookback period
            column: Column to use

        Returns:
            VIDYA series
        """
        price = self.df[column].values
        n = len(price)

        # Calculate CMO (Chande Momentum Oscillator)
        diff = np.diff(price)
        diff = np.concatenate([[0], diff])

        up_sum = np.zeros(n)
        down_sum = np.zeros(n)

        for i in range(cmo_period, n):
            window_diff = diff[i - cmo_period + 1:i + 1]
            up_sum[i] = np.sum(np.where(window_diff > 0, window_diff, 0))
            down_sum[i] = np.sum(np.where(window_diff < 0, -window_diff, 0))

        total = up_sum + down_sum
        cmo = np.where(total != 0, (up_sum - down_sum) / total, 0)

        # Smoothing constant
        sc = 2 / (period + 1)

        # Calculate VIDYA
        vidya = np.zeros(n)
        vidya[cmo_period - 1] = price[cmo_period - 1]

        for i in range(cmo_period, n):
            vidya[i] = sc * np.abs(cmo[i]) * price[i] + (1 - sc * np.abs(cmo[i])) * vidya[i - 1]

        vidya[:cmo_period - 1] = np.nan

        return pd.Series(vidya, index=self.df.index, name=f"vidya_{period}")

    def alma(
        self,
        period: int = 9,
        offset: float = 0.85,
        sigma: float = 6,
        column: str = "close",
    ) -> pd.Series:
        """
        Arnaud Legoux Moving Average (ALMA).

        Uses Gaussian distribution for weighting, with configurable
        offset (controls timing) and sigma (controls smoothness).
        Very smooth with minimal lag.

        Args:
            period: Lookback period
            offset: Weight offset (0-1, higher = more weight to recent data)
            sigma: Gaussian sigma (higher = smoother)
            column: Column to use

        Returns:
            ALMA series
        """
        price = self.df[column].values
        n = len(price)

        # Calculate weights
        m = np.floor(offset * (period - 1))
        s = period / sigma

        weights = np.zeros(period)
        for i in range(period):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s ** 2))

        weights = weights / weights.sum()

        # Calculate ALMA
        alma = np.full(n, np.nan)
        for i in range(period - 1, n):
            alma[i] = np.sum(weights * price[i - period + 1:i + 1])

        return pd.Series(alma, index=self.df.index, name=f"alma_{period}")

    def adaptive_atr(
        self,
        period: int = 14,
        fast_period: int = 5,
        slow_period: int = 50,
    ) -> pd.Series:
        """
        Adaptive ATR - ATR that adapts to market conditions.

        Uses efficiency ratio to blend between fast and slow ATR.
        More responsive in trending markets, smoother in ranging markets.

        Args:
            period: Base ATR period
            fast_period: Fast ATR period
            slow_period: Slow ATR period

        Returns:
            Adaptive ATR series
        """
        # Calculate standard ATR
        high = self.df["high"].values
        low = self.df["low"].values
        close = self.df["close"].values
        n = len(close)

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # Calculate efficiency ratio on close
        change = np.abs(close[period:] - close[:-period])
        volatility = np.zeros(n)
        for i in range(period, n):
            volatility[i] = np.sum(np.abs(np.diff(close[i - period:i + 1])))

        volatility = np.where(volatility == 0, 1, volatility)
        er = np.zeros(n)
        er[period:] = change / volatility[period:]

        # Fast and slow ATR
        fast_atr = pd.Series(tr).ewm(span=fast_period, adjust=False).mean().values
        slow_atr = pd.Series(tr).ewm(span=slow_period, adjust=False).mean().values

        # Blend based on ER
        adaptive_atr = er * fast_atr + (1 - er) * slow_atr

        return pd.Series(adaptive_atr, index=self.df.index, name="adaptive_atr")

    def mcginley_dynamic(
        self,
        period: int = 14,
        column: str = "close",
    ) -> pd.Series:
        """
        McGinley Dynamic Indicator.

        A moving average that automatically adjusts for market speed.
        Developed by John McGinley to avoid whipsaws while staying
        responsive to price changes.

        Args:
            period: Lookback period
            column: Column to use

        Returns:
            McGinley Dynamic series
        """
        price = self.df[column].values
        n = len(price)

        md = np.zeros(n)
        md[0] = price[0]

        for i in range(1, n):
            if md[i - 1] != 0:
                ratio = price[i] / md[i - 1]
                md[i] = md[i - 1] + (price[i] - md[i - 1]) / (period * (ratio ** 4))
            else:
                md[i] = price[i]

        return pd.Series(md, index=self.df.index, name=f"mcginley_dynamic_{period}")
