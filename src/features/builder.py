"""
Institutional-Grade Feature Builder
JPMorgan-Level Feature Engineering Pipeline

Features:
- Automated feature generation from OHLCV data
- Multi-timeframe feature aggregation
- Feature selection and importance
- Rolling and expanding window features
- Target variable generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import warnings

from .technical import TechnicalIndicators, AdvancedTechnicals
from .fracdiff import FractionalDifferentiation, FracDiffConfig, FracDiffFeatureTransformer
from ..utils.logger import get_logger, get_performance_logger
from ..utils.helpers import safe_divide, timer


logger = get_logger(__name__)
perf_logger = get_performance_logger()


@dataclass
class FeatureConfig:
    """Configuration for feature generation"""
    # Technical indicator periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    bb_periods: List[int] = field(default_factory=lambda: [10, 20, 30])

    # Target configuration
    target_lookahead: int = 4  # Periods to look ahead for target
    target_threshold: float = 0.005  # 0.5% threshold for classification

    # Feature selection
    max_features: int = 100
    min_importance: float = 0.001
    correlation_threshold: float = 0.95

    # Fractional differentiation
    use_frac_diff: bool = True
    frac_diff_auto_optimize: bool = True
    frac_diff_default_d: float = 0.5
    frac_diff_threshold: float = 1e-5


class FeatureBuilder:
    """
    Comprehensive feature builder for ML models.

    Generates 200+ features from OHLCV data including:
    - Technical indicators
    - Price action features
    - Volume features
    - Volatility features
    - Statistical features
    - Time-based features
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize FeatureBuilder.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self._feature_functions: Dict[str, Callable] = {}
        self._register_features()

        # Initialize fractional differentiation
        if self.config.use_frac_diff:
            frac_config = FracDiffConfig(
                threshold=self.config.frac_diff_threshold
            )
            self._frac_diff = FractionalDifferentiation(frac_config)
            self._frac_diff_transformer = FracDiffFeatureTransformer(
                d=None if self.config.frac_diff_auto_optimize else self.config.frac_diff_default_d,
                auto_optimize=self.config.frac_diff_auto_optimize,
                threshold=self.config.frac_diff_threshold
            )
        else:
            self._frac_diff = None
            self._frac_diff_transformer = None

        self._fitted_frac_diff_d: Dict[str, float] = {}

        logger.info("FeatureBuilder initialized")

    def _register_features(self) -> None:
        """Register all feature generation functions"""
        # Price-based features
        self._feature_functions['price'] = self._generate_price_features
        self._feature_functions['returns'] = self._generate_return_features
        self._feature_functions['trend'] = self._generate_trend_features
        self._feature_functions['momentum'] = self._generate_momentum_features
        self._feature_functions['volatility'] = self._generate_volatility_features
        self._feature_functions['volume'] = self._generate_volume_features
        self._feature_functions['statistical'] = self._generate_statistical_features
        self._feature_functions['time'] = self._generate_time_features
        self._feature_functions['pattern'] = self._generate_pattern_features
        self._feature_functions['fracdiff'] = self._generate_fracdiff_features

    def _generate_fracdiff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fractionally differentiated features.

        Uses fractional differentiation to achieve stationarity while
        preserving memory in the time series.
        """
        features = pd.DataFrame(index=df.index)

        if self._frac_diff is None:
            return features

        # Apply FFD to close price
        try:
            result = self._frac_diff.auto_frac_diff(
                df['close'],
                find_optimal=self.config.frac_diff_auto_optimize
            )
            features['close_ffd'] = result.series
            self._fitted_frac_diff_d['close'] = result.d

            # Apply FFD to volume (log volume first)
            log_volume = np.log1p(df['volume'])
            vol_result = self._frac_diff.auto_frac_diff(
                log_volume,
                find_optimal=self.config.frac_diff_auto_optimize
            )
            features['log_volume_ffd'] = vol_result.series
            self._fitted_frac_diff_d['log_volume'] = vol_result.d

            # Apply to high-low range
            hl_range = np.log(df['high'] / df['low'])
            range_result = self._frac_diff.auto_frac_diff(hl_range, find_optimal=False)
            features['hl_range_ffd'] = range_result.series

            # Generate features using FFD close
            if 'close_ffd' in features.columns:
                ffd_close = features['close_ffd'].dropna()
                if len(ffd_close) > 20:
                    # Moving averages of FFD close
                    features['ffd_close_sma_5'] = TechnicalIndicators.sma(features['close_ffd'], 5)
                    features['ffd_close_sma_20'] = TechnicalIndicators.sma(features['close_ffd'], 20)

                    # Z-score of FFD close
                    ffd_mean = features['close_ffd'].rolling(20).mean()
                    ffd_std = features['close_ffd'].rolling(20).std()
                    features['ffd_close_zscore'] = (features['close_ffd'] - ffd_mean) / ffd_std

            logger.info(f"Generated FFD features with d={result.d:.4f}")

        except Exception as e:
            logger.warning(f"FFD feature generation failed: {e}")

        return features

    def get_frac_diff_d(self) -> Dict[str, float]:
        """Get fitted fractional differentiation d values"""
        return self._fitted_frac_diff_d.copy()

    def build_features(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Build all features for a single symbol.

        Args:
            df: OHLCV DataFrame
            categories: Feature categories to generate (None = all)

        Returns:
            DataFrame with all features
        """
        with perf_logger.measure_time("build_features"):
            features = df.copy()

            # Ensure column names are lowercase
            features.columns = features.columns.str.lower()

            categories = categories or list(self._feature_functions.keys())

            for category in categories:
                if category in self._feature_functions:
                    try:
                        category_features = self._feature_functions[category](features)
                        features = pd.concat([features, category_features], axis=1)
                    except Exception as e:
                        logger.warning(f"Failed to generate {category} features: {e}")

            # Remove duplicate columns
            features = features.loc[:, ~features.columns.duplicated()]

            # Handle infinite values
            features = features.replace([np.inf, -np.inf], np.nan)

            logger.info(f"Generated {len(features.columns)} features")

            return features

    def _generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features"""
        features = pd.DataFrame(index=df.index)

        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        features['high_close_ratio'] = df['high'] / df['close']
        features['low_close_ratio'] = df['low'] / df['close']

        # Price position
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

        # Typical price variations
        features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        features['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        features['median_price'] = (df['high'] + df['low']) / 2

        # Price gaps
        features['gap'] = df['open'] - df['close'].shift(1)
        features['gap_pct'] = features['gap'] / df['close'].shift(1)

        # Moving average distances
        for period in self.config.sma_periods:
            sma = TechnicalIndicators.sma(df['close'], period)
            features[f'dist_sma_{period}'] = (df['close'] - sma) / sma

        for period in self.config.ema_periods:
            ema = TechnicalIndicators.ema(df['close'], period)
            features[f'dist_ema_{period}'] = (df['close'] - ema) / ema

        return features

    def _generate_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate return-based features"""
        features = pd.DataFrame(index=df.index)

        # Simple returns at various periods
        for period in [1, 2, 3, 5, 10, 20, 50]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # Log returns
        for period in [1, 5, 10, 20]:
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # Cumulative returns
        for period in [5, 10, 20, 50]:
            features[f'cum_return_{period}'] = (
                (1 + df['close'].pct_change()).rolling(period).apply(np.prod, raw=True) - 1
            )

        # Return momentum
        features['return_momentum'] = features['return_5'] - features['return_20']

        # Acceleration
        features['return_acceleration'] = features['return_1'] - features['return_1'].shift(1)

        # Higher moments of returns
        for period in [10, 20, 50]:
            returns = df['close'].pct_change()
            features[f'return_skew_{period}'] = returns.rolling(period).skew()
            features[f'return_kurt_{period}'] = returns.rolling(period).kurt()

        return features

    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features"""
        features = pd.DataFrame(index=df.index)

        # Moving average crossovers
        for fast, slow in [(5, 20), (10, 50), (20, 100), (50, 200)]:
            sma_fast = TechnicalIndicators.sma(df['close'], fast)
            sma_slow = TechnicalIndicators.sma(df['close'], slow)
            features[f'ma_cross_{fast}_{slow}'] = (sma_fast - sma_slow) / sma_slow

        # EMA crossovers
        for fast, slow in [(12, 26), (5, 20)]:
            ema_fast = TechnicalIndicators.ema(df['close'], fast)
            ema_slow = TechnicalIndicators.ema(df['close'], slow)
            features[f'ema_cross_{fast}_{slow}'] = (ema_fast - ema_slow) / ema_slow

        # MACD features
        macd, signal, hist = TechnicalIndicators.macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_hist_change'] = hist.diff()

        # ADX and directional movement
        adx, plus_di, minus_di = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['di_diff'] = plus_di - minus_di
        features['di_ratio'] = safe_divide(plus_di, minus_di)

        # Ichimoku features
        ichimoku = TechnicalIndicators.ichimoku(df['high'], df['low'], df['close'])
        features['ichimoku_tenkan'] = ichimoku['tenkan_sen']
        features['ichimoku_kijun'] = ichimoku['kijun_sen']
        features['ichimoku_diff'] = ichimoku['tenkan_sen'] - ichimoku['kijun_sen']

        # Supertrend
        supertrend, direction = TechnicalIndicators.supertrend(
            df['high'], df['low'], df['close']
        )
        features['supertrend'] = supertrend
        features['supertrend_dir'] = direction

        # Trend strength
        features['trend_strength'] = AdvancedTechnicals.calculate_trend_strength(
            df['close'], df['high'], df['low']
        )

        # Linear regression slope
        for period in [10, 20, 50]:
            features[f'linreg_slope_{period}'] = self._linear_regression_slope(
                df['close'], period
            )

        return features

    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based features"""
        features = pd.DataFrame(index=df.index)

        # RSI at multiple periods
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = TechnicalIndicators.rsi(df['close'], period)

        # RSI divergence (RSI - RSI smoothed)
        rsi_14 = features['rsi_14']
        features['rsi_divergence'] = rsi_14 - TechnicalIndicators.sma(rsi_14, 5)

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            df['high'], df['low'], df['close']
        )
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_diff'] = stoch_k - stoch_d

        # Williams %R
        features['williams_r'] = TechnicalIndicators.williams_r(
            df['high'], df['low'], df['close']
        )

        # CCI
        features['cci'] = TechnicalIndicators.cci(
            df['high'], df['low'], df['close']
        )

        # ROC at multiple periods
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = TechnicalIndicators.roc(df['close'], period)

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = TechnicalIndicators.momentum(df['close'], period)

        # TSI
        features['tsi'] = TechnicalIndicators.tsi(df['close'])

        # Ultimate Oscillator
        features['ultimate_osc'] = TechnicalIndicators.ultimate_oscillator(
            df['high'], df['low'], df['close']
        )

        # Awesome Oscillator
        features['awesome_osc'] = TechnicalIndicators.awesome_oscillator(
            df['high'], df['low']
        )

        # Relative momentum
        features['rel_momentum'] = (
            features['momentum_5'] / features['momentum_20'].abs().rolling(20).mean()
        )

        return features

    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features"""
        features = pd.DataFrame(index=df.index)

        # ATR at multiple periods
        for period in self.config.atr_periods:
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period)
            features[f'atr_{period}'] = atr
            features[f'atr_pct_{period}'] = atr / df['close']

        # ATR ratio (short-term vs long-term)
        features['atr_ratio'] = features['atr_7'] / features['atr_21']

        # Bollinger Bands features
        for period in self.config.bb_periods:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'], period)
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = (upper - lower) / middle
            features[f'bb_pct_b_{period}'] = (df['close'] - lower) / (upper - lower)

        # Keltner Channel
        kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channel(
            df['high'], df['low'], df['close']
        )
        features['kc_upper'] = kc_upper
        features['kc_lower'] = kc_lower
        features['kc_width'] = (kc_upper - kc_lower) / kc_middle

        # Donchian Channel
        dc_upper, dc_middle, dc_lower = TechnicalIndicators.donchian_channel(
            df['high'], df['low']
        )
        features['dc_upper'] = dc_upper
        features['dc_lower'] = dc_lower
        features['dc_width'] = (dc_upper - dc_lower) / dc_middle
        features['dc_position'] = (df['close'] - dc_lower) / (dc_upper - dc_lower)

        # Historical volatility
        for period in [10, 20, 50]:
            features[f'hist_vol_{period}'] = TechnicalIndicators.historical_volatility(
                df['close'], period, annualize=False
            )

        # Parkinson volatility
        features['parkinson_vol'] = TechnicalIndicators.parkinson_volatility(
            df['high'], df['low']
        )

        # Garman-Klass volatility
        features['gk_vol'] = TechnicalIndicators.garman_klass_volatility(
            df['open'], df['high'], df['low'], df['close']
        )

        # Volatility ratio
        features['vol_ratio'] = features['hist_vol_10'] / features['hist_vol_50']

        # Range features
        features['range'] = (df['high'] - df['low']) / df['close']
        features['range_ma'] = features['range'].rolling(20).mean()
        features['range_ratio'] = features['range'] / features['range_ma']

        return features

    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features"""
        features = pd.DataFrame(index=df.index)

        # Volume moving averages
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_{period}'] = TechnicalIndicators.sma(df['volume'], period)

        # Volume ratios
        features['volume_ratio_5'] = df['volume'] / features['volume_ma_5']
        features['volume_ratio_20'] = df['volume'] / features['volume_ma_20']

        # Volume trend
        features['volume_trend'] = (
            features['volume_ma_5'] / features['volume_ma_20']
        )

        # OBV
        features['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        features['obv_ma'] = TechnicalIndicators.sma(features['obv'], 20)
        features['obv_diff'] = features['obv'] - features['obv_ma']

        # VWAP
        features['vwap'] = TechnicalIndicators.vwap(
            df['high'], df['low'], df['close'], df['volume']
        )
        features['vwap_diff'] = (df['close'] - features['vwap']) / features['vwap']

        # MFI
        features['mfi'] = TechnicalIndicators.mfi(
            df['high'], df['low'], df['close'], df['volume']
        )

        # A/D Line
        features['ad_line'] = TechnicalIndicators.ad_line(
            df['high'], df['low'], df['close'], df['volume']
        )
        features['ad_line_ma'] = TechnicalIndicators.sma(features['ad_line'], 20)

        # CMF
        features['cmf'] = TechnicalIndicators.cmf(
            df['high'], df['low'], df['close'], df['volume']
        )

        # Force Index
        features['force_index'] = TechnicalIndicators.force_index(
            df['close'], df['volume']
        )

        # Volume-price confirmation
        price_up = df['close'] > df['close'].shift(1)
        volume_up = df['volume'] > df['volume'].shift(1)
        features['vol_price_confirm'] = (price_up == volume_up).astype(int)

        # Relative volume
        features['rel_volume'] = df['volume'] / df['volume'].rolling(50).mean()

        return features

    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        features = pd.DataFrame(index=df.index)

        # Z-score of price
        for period in [20, 50]:
            rolling_mean = df['close'].rolling(period).mean()
            rolling_std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - rolling_mean) / rolling_std

        # Percentile rank
        for period in [20, 50, 100]:
            features[f'percentile_{period}'] = df['close'].rolling(period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False
            )

        # Rolling correlation with market proxy (using returns)
        returns = df['close'].pct_change()
        for period in [20, 50]:
            features[f'autocorr_{period}'] = returns.rolling(period).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            )

        # Hurst exponent estimation
        features['hurst_20'] = self._calculate_hurst(df['close'], 20)

        # Rolling beta (simplified - using self-correlation as proxy)
        features['rolling_beta'] = returns.rolling(50).std() / returns.std()

        # Distance from recent high/low
        for period in [20, 50, 100]:
            features[f'dist_high_{period}'] = (
                df['close'] / df['high'].rolling(period).max() - 1
            )
            features[f'dist_low_{period}'] = (
                df['close'] / df['low'].rolling(period).min() - 1
            )

        # Number of up/down days
        for period in [5, 10, 20]:
            up_days = (df['close'] > df['close'].shift(1)).rolling(period).sum()
            features[f'up_days_ratio_{period}'] = up_days / period

        return features

    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features"""
        features = pd.DataFrame(index=df.index)

        # Extract time components
        features['hour'] = df.index.hour
        features['minute'] = df.index.minute
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter

        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

        # Cyclical encoding for day of week
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # Cyclical encoding for month
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        # Market session indicators (for 15-min US market data)
        features['is_market_open'] = ((features['hour'] == 9) & (features['minute'] >= 30)).astype(int)
        features['is_market_close'] = ((features['hour'] == 15) & (features['minute'] >= 45)).astype(int)
        features['is_lunch'] = ((features['hour'] >= 12) & (features['hour'] < 13)).astype(int)

        # Time since market open (in 15-min bars)
        market_open_minutes = 9 * 60 + 30
        current_minutes = features['hour'] * 60 + features['minute']
        features['bars_since_open'] = (current_minutes - market_open_minutes) / 15

        # Time until market close
        market_close_minutes = 16 * 60
        features['bars_until_close'] = (market_close_minutes - current_minutes) / 15

        return features

    def _generate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate candlestick pattern features"""
        features = pd.DataFrame(index=df.index)

        # Body analysis
        body = df['close'] - df['open']
        body_abs = abs(body)
        range_hl = df['high'] - df['low']

        features['body_pct'] = body_abs / range_hl
        features['is_bullish'] = (body > 0).astype(int)

        # Doji detection (small body)
        features['is_doji'] = (body_abs / range_hl < 0.1).astype(int)

        # Hammer/Shooting star
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

        features['is_hammer'] = (
            (lower_shadow > 2 * body_abs) &
            (upper_shadow < body_abs * 0.5) &
            (body > 0)
        ).astype(int)

        features['is_shooting_star'] = (
            (upper_shadow > 2 * body_abs) &
            (lower_shadow < body_abs * 0.5) &
            (body < 0)
        ).astype(int)

        # Engulfing patterns
        prev_body = body.shift(1)
        features['bullish_engulf'] = (
            (prev_body < 0) &
            (body > 0) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)

        features['bearish_engulf'] = (
            (prev_body > 0) &
            (body < 0) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)

        # Inside bar
        features['inside_bar'] = (
            (df['high'] < df['high'].shift(1)) &
            (df['low'] > df['low'].shift(1))
        ).astype(int)

        # Outside bar
        features['outside_bar'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['low'] < df['low'].shift(1))
        ).astype(int)

        # Consecutive candles
        for n in [2, 3, 4, 5]:
            features[f'consec_up_{n}'] = (
                (body > 0).rolling(n).sum() == n
            ).astype(int)
            features[f'consec_down_{n}'] = (
                (body < 0).rolling(n).sum() == n
            ).astype(int)

        return features

    def _linear_regression_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate rolling linear regression slope"""
        def slope(x):
            if len(x) < 2:
                return 0
            y = np.arange(len(x))
            A = np.vstack([y, np.ones(len(y))]).T
            m, c = np.linalg.lstsq(A, x, rcond=None)[0]
            return m

        return series.rolling(window=period).apply(slope, raw=True)

    def _calculate_hurst(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate rolling Hurst exponent"""
        def hurst(x):
            if len(x) < 20:
                return 0.5

            lags = range(2, min(20, len(x) // 2))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]

            if not tau or min(tau) <= 0:
                return 0.5

            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] / 2.0
            except:
                return 0.5

        return series.rolling(window=period).apply(hurst, raw=True)

    def generate_target(
        self,
        df: pd.DataFrame,
        target_type: str = 'classification',
        lookahead: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Generate target variable for ML training.

        Args:
            df: OHLCV DataFrame
            target_type: 'classification' or 'regression'
            lookahead: Periods to look ahead
            threshold: Classification threshold

        Returns:
            Target series
        """
        lookahead = lookahead or self.config.target_lookahead
        threshold = threshold or self.config.target_threshold

        # Forward returns
        future_return = df['close'].shift(-lookahead) / df['close'] - 1

        if target_type == 'classification':
            # 3-class: -1 (down), 0 (neutral), 1 (up)
            target = pd.Series(0, index=df.index)
            target[future_return > threshold] = 1
            target[future_return < -threshold] = -1

        elif target_type == 'binary':
            # 2-class: 0 (down/neutral), 1 (up)
            target = (future_return > 0).astype(int)

        elif target_type == 'regression':
            target = future_return

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        return target


class FeaturePipeline:
    """
    Complete feature engineering pipeline for multiple symbols.

    Features:
    - Parallel processing
    - Feature caching
    - Feature selection
    - Cross-validation aware splitting
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        n_jobs: int = -1
    ):
        """
        Initialize FeaturePipeline.

        Args:
            config: Feature configuration
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.config = config or FeatureConfig()
        self.n_jobs = n_jobs if n_jobs > 0 else None  # None = use default
        self.builder = FeatureBuilder(config)

        self._feature_importance: Dict[str, float] = {}
        self._selected_features: List[str] = []

        logger.info("FeaturePipeline initialized")

    def fit_transform(
        self,
        data: Dict[str, pd.DataFrame],
        target_type: str = 'classification'
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Build features for all symbols and generate targets.

        Args:
            data: Dictionary of symbol DataFrames
            target_type: Type of target variable

        Returns:
            Tuple of (features dict, targets dict)
        """
        features_dict = {}
        targets_dict = {}

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(self._process_symbol, symbol, df, target_type): symbol
                for symbol, df in data.items()
            }

            for future in futures:
                symbol = futures[future]
                try:
                    features, target = future.result()
                    features_dict[symbol] = features
                    targets_dict[symbol] = target
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")

        logger.info(f"Processed {len(features_dict)} symbols")

        return features_dict, targets_dict

    def _process_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_type: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Process single symbol"""
        features = self.builder.build_features(df)
        target = self.builder.generate_target(df, target_type)

        # Align features and target
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        return features, target

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'importance'
    ) -> List[str]:
        """
        Select most important features.

        Args:
            X: Feature DataFrame
            y: Target series
            method: Selection method ('importance', 'correlation', 'both')

        Returns:
            List of selected feature names
        """
        selected = set(X.columns)

        if method in ['correlation', 'both']:
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = set()
            for col in upper.columns:
                if any(upper[col] > self.config.correlation_threshold):
                    to_drop.add(col)

            selected -= to_drop
            logger.info(f"Removed {len(to_drop)} highly correlated features")

        if method in ['importance', 'both']:
            # Calculate feature importance using Random Forest
            try:
                from sklearn.ensemble import RandomForestClassifier

                X_clean = X[list(selected)].dropna()
                y_clean = y.loc[X_clean.index]

                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_clean, y_clean)

                importance = pd.Series(
                    rf.feature_importances_,
                    index=X_clean.columns
                ).sort_values(ascending=False)

                self._feature_importance = importance.to_dict()

                # Keep features above importance threshold
                important_features = importance[
                    importance > self.config.min_importance
                ].index.tolist()

                selected &= set(important_features)

            except ImportError:
                logger.warning("sklearn not available for feature importance")

        # Limit to max features
        self._selected_features = list(selected)[:self.config.max_features]

        logger.info(f"Selected {len(self._selected_features)} features")

        return self._selected_features

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self._feature_importance

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """Create lagged versions of features"""
        lagged = pd.DataFrame(index=df.index)

        for col in columns:
            if col in df.columns:
                for lag in lags:
                    lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return lagged

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        funcs: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Create rolling window features"""
        rolling = pd.DataFrame(index=df.index)

        for col in columns:
            if col in df.columns:
                for window in windows:
                    roll = df[col].rolling(window)
                    for func in funcs:
                        rolling[f'{col}_roll_{window}_{func}'] = getattr(roll, func)()

        return rolling
