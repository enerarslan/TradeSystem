"""
Market Regime Detection
JPMorgan-Level Regime Analysis and Detection

Features:
- Hidden Markov Model regime detection
- Volatility regime classification
- Trend regime identification
- Correlation regime analysis
- Dynamic regime switching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger
from ..utils.helpers import safe_divide


logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Trend regime classification"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class RegimeState:
    """Current regime state"""
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    confidence: float
    transition_probability: float


class RegimeDetector:
    """
    Market regime detection and analysis.

    Uses multiple methods:
    - Statistical regime detection
    - Hidden Markov Models
    - Volatility clustering
    - Trend analysis
    """

    def __init__(
        self,
        vol_threshold_low: float = 0.10,
        vol_threshold_high: float = 0.25,
        vol_threshold_extreme: float = 0.40,
        trend_threshold: float = 0.02
    ):
        """
        Initialize RegimeDetector.

        Args:
            vol_threshold_low: Annualized vol threshold for low regime
            vol_threshold_high: Annualized vol threshold for high regime
            vol_threshold_extreme: Annualized vol threshold for extreme regime
            trend_threshold: Return threshold for trend detection
        """
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_extreme = vol_threshold_extreme
        self.trend_threshold = trend_threshold

        self._hmm_model = None
        self._regime_history: List[RegimeState] = []

    def detect_regime(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Detect market regime using multiple methods.

        Args:
            prices: Price series
            returns: Return series (optional)
            window: Lookback window

        Returns:
            DataFrame with regime classifications
        """
        if returns is None:
            returns = prices.pct_change()

        features = pd.DataFrame(index=prices.index)

        # Volatility regime
        vol_regime, vol_value = self._detect_volatility_regime(returns, window)
        features['vol_regime'] = vol_regime
        features['vol_value'] = vol_value

        # Trend regime
        trend_regime, trend_value = self._detect_trend_regime(prices, returns, window)
        features['trend_regime'] = trend_regime
        features['trend_value'] = trend_value

        # Combined market regime
        features['market_regime'] = self._combine_regimes(
            features['vol_regime'],
            features['trend_regime']
        )

        # Regime probabilities (simplified)
        features['regime_confidence'] = self._calculate_regime_confidence(
            features['vol_value'],
            features['trend_value']
        )

        return features

    def _detect_volatility_regime(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect volatility regime.

        Returns:
            Tuple of (regime series, volatility value series)
        """
        # Calculate annualized volatility
        vol = returns.rolling(window).std() * np.sqrt(252 * 26)  # 15-min bars

        # Classify regime
        regime = pd.Series(index=returns.index, dtype=str)
        regime[vol <= self.vol_threshold_low] = VolatilityRegime.LOW.value
        regime[(vol > self.vol_threshold_low) & (vol <= self.vol_threshold_high)] = VolatilityRegime.NORMAL.value
        regime[(vol > self.vol_threshold_high) & (vol <= self.vol_threshold_extreme)] = VolatilityRegime.HIGH.value
        regime[vol > self.vol_threshold_extreme] = VolatilityRegime.EXTREME.value

        return regime, vol

    def _detect_trend_regime(
        self,
        prices: pd.Series,
        returns: pd.Series,
        window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect trend regime.

        Returns:
            Tuple of (regime series, trend value series)
        """
        # Calculate multiple trend indicators
        sma_short = prices.rolling(window).mean()
        sma_long = prices.rolling(window * 3).mean()

        # Trend strength: normalized distance from moving average
        trend_value = (prices - sma_long) / sma_long

        # Cumulative return
        cum_return = returns.rolling(window).sum()

        # ADX-like trend strength
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        range_pct = (high - low) / low

        # Classify regime based on multiple factors
        regime = pd.Series(TrendRegime.SIDEWAYS.value, index=prices.index)

        strong_up = (trend_value > self.trend_threshold * 2) & (cum_return > self.trend_threshold)
        up = (trend_value > self.trend_threshold) & (cum_return > 0)
        strong_down = (trend_value < -self.trend_threshold * 2) & (cum_return < -self.trend_threshold)
        down = (trend_value < -self.trend_threshold) & (cum_return < 0)

        regime[strong_up] = TrendRegime.STRONG_UPTREND.value
        regime[up & ~strong_up] = TrendRegime.UPTREND.value
        regime[strong_down] = TrendRegime.STRONG_DOWNTREND.value
        regime[down & ~strong_down] = TrendRegime.DOWNTREND.value

        return regime, trend_value

    def _combine_regimes(
        self,
        vol_regime: pd.Series,
        trend_regime: pd.Series
    ) -> pd.Series:
        """
        Combine volatility and trend regimes into market regime.
        """
        market_regime = pd.Series(index=vol_regime.index, dtype=str)

        # Bull markets
        bull_mask = trend_regime.isin([
            TrendRegime.STRONG_UPTREND.value,
            TrendRegime.UPTREND.value
        ])

        # Bear markets
        bear_mask = trend_regime.isin([
            TrendRegime.STRONG_DOWNTREND.value,
            TrendRegime.DOWNTREND.value
        ])

        # Low volatility
        low_vol = vol_regime.isin([
            VolatilityRegime.LOW.value,
            VolatilityRegime.NORMAL.value
        ])

        # High volatility
        high_vol = vol_regime.isin([
            VolatilityRegime.HIGH.value,
            VolatilityRegime.EXTREME.value
        ])

        # Combine
        market_regime[bull_mask & low_vol] = MarketRegime.BULL_LOW_VOL.value
        market_regime[bull_mask & high_vol] = MarketRegime.BULL_HIGH_VOL.value
        market_regime[bear_mask & low_vol] = MarketRegime.BEAR_LOW_VOL.value
        market_regime[bear_mask & high_vol] = MarketRegime.BEAR_HIGH_VOL.value
        market_regime[~bull_mask & ~bear_mask] = MarketRegime.SIDEWAYS.value

        # Crisis detection (extreme vol + strong downtrend)
        crisis_mask = (vol_regime == VolatilityRegime.EXTREME.value) & \
                      (trend_regime == TrendRegime.STRONG_DOWNTREND.value)
        market_regime[crisis_mask] = MarketRegime.CRISIS.value

        return market_regime

    def _calculate_regime_confidence(
        self,
        vol_value: pd.Series,
        trend_value: pd.Series
    ) -> pd.Series:
        """
        Calculate confidence in regime classification.
        """
        # Normalize values
        vol_zscore = (vol_value - vol_value.rolling(100).mean()) / vol_value.rolling(100).std()
        trend_zscore = (trend_value - trend_value.rolling(100).mean()) / trend_value.rolling(100).std()

        # Confidence based on how extreme the values are
        confidence = (abs(vol_zscore) + abs(trend_zscore)) / 2
        confidence = confidence.clip(0, 1)

        return confidence

    def fit_hmm(
        self,
        returns: pd.Series,
        n_states: int = 3,
        n_iterations: int = 100
    ) -> 'RegimeDetector':
        """
        Fit Hidden Markov Model for regime detection.

        Args:
            returns: Return series
            n_states: Number of hidden states
            n_iterations: Number of EM iterations

        Returns:
            Self for chaining
        """
        try:
            from hmmlearn import hmm

            # Prepare data
            X = returns.dropna().values.reshape(-1, 1)

            # Fit Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iterations,
                random_state=42
            )

            model.fit(X)
            self._hmm_model = model

            logger.info(f"HMM fitted with {n_states} states")

        except ImportError:
            logger.warning("hmmlearn not installed, using fallback method")

        return self

    def predict_hmm_regime(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Predict regime using fitted HMM.

        Args:
            returns: Return series

        Returns:
            DataFrame with HMM predictions
        """
        if self._hmm_model is None:
            logger.warning("HMM not fitted, fitting now")
            self.fit_hmm(returns)

        if self._hmm_model is None:
            # Fallback if hmmlearn not available
            return self._fallback_hmm(returns)

        X = returns.dropna().values.reshape(-1, 1)

        # Predict states
        states = self._hmm_model.predict(X)
        probs = self._hmm_model.predict_proba(X)

        # Create DataFrame
        valid_idx = returns.dropna().index
        result = pd.DataFrame(index=returns.index)
        result.loc[valid_idx, 'hmm_state'] = states
        result.loc[valid_idx, 'hmm_confidence'] = np.max(probs, axis=1)

        # Map states to regime labels based on mean return
        state_means = self._hmm_model.means_.flatten()
        state_order = np.argsort(state_means)

        regime_map = {
            state_order[0]: 'bear',
            state_order[-1]: 'bull',
        }
        if len(state_order) > 2:
            for i in state_order[1:-1]:
                regime_map[i] = 'neutral'

        result['hmm_regime'] = result['hmm_state'].map(regime_map)

        return result

    def _fallback_hmm(self, returns: pd.Series) -> pd.DataFrame:
        """
        Fallback regime detection when hmmlearn not available.

        Uses simple volatility clustering.
        """
        result = pd.DataFrame(index=returns.index)

        # Simple regime based on rolling volatility quantiles
        vol = returns.rolling(20).std()
        vol_percentile = vol.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )

        result['hmm_state'] = pd.cut(
            vol_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2]
        ).astype(float)

        result['hmm_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['low_vol', 'normal', 'high_vol']
        )

        result['hmm_confidence'] = abs(vol_percentile - 0.5) * 2

        return result

    def regime_transition_matrix(
        self,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix.

        Args:
            regimes: Series of regime labels

        Returns:
            Transition probability matrix
        """
        # Get unique regimes
        unique_regimes = regimes.dropna().unique()

        # Count transitions
        transitions = pd.DataFrame(
            0,
            index=unique_regimes,
            columns=unique_regimes
        )

        prev_regime = regimes.shift(1)
        for curr, prev in zip(regimes, prev_regime):
            if pd.notna(curr) and pd.notna(prev):
                transitions.loc[prev, curr] += 1

        # Convert to probabilities
        row_sums = transitions.sum(axis=1)
        transition_probs = transitions.div(row_sums, axis=0)

        return transition_probs

    def regime_statistics(
        self,
        returns: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            returns: Return series
            regimes: Regime labels

        Returns:
            DataFrame with regime statistics
        """
        stats = []

        for regime in regimes.dropna().unique():
            mask = regimes == regime
            regime_returns = returns[mask]

            stat = {
                'regime': regime,
                'count': mask.sum(),
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'max_drawdown': self._max_drawdown(regime_returns),
                'avg_duration': self._avg_regime_duration(regimes, regime)
            }
            stats.append(stat)

        return pd.DataFrame(stats).set_index('regime')

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()

    def _avg_regime_duration(self, regimes: pd.Series, target_regime: str) -> float:
        """Calculate average regime duration"""
        in_regime = (regimes == target_regime).astype(int)
        regime_changes = in_regime.diff().abs()

        # Count periods in regime
        regime_lengths = []
        current_length = 0

        for i, (val, change) in enumerate(zip(in_regime, regime_changes)):
            if val == 1:
                current_length += 1
            elif current_length > 0:
                regime_lengths.append(current_length)
                current_length = 0

        if current_length > 0:
            regime_lengths.append(current_length)

        return np.mean(regime_lengths) if regime_lengths else 0

    def generate_regime_features(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Generate all regime-related features.

        Args:
            prices: Price series
            window: Lookback window

        Returns:
            DataFrame with all regime features
        """
        returns = prices.pct_change()
        features = pd.DataFrame(index=prices.index)

        # Basic regime detection
        basic_regime = self.detect_regime(prices, returns, window)
        features = pd.concat([features, basic_regime], axis=1)

        # HMM regime (if available)
        try:
            hmm_regime = self.predict_hmm_regime(returns)
            features = pd.concat([features, hmm_regime], axis=1)
        except Exception as e:
            logger.debug(f"HMM regime detection failed: {e}")

        # Regime-based adjustments
        features['regime_vol_adjustment'] = self._calculate_vol_adjustment(
            features.get('vol_regime', pd.Series(index=prices.index))
        )

        features['regime_position_multiplier'] = self._calculate_position_multiplier(
            features.get('market_regime', pd.Series(index=prices.index))
        )

        return features

    def _calculate_vol_adjustment(self, vol_regime: pd.Series) -> pd.Series:
        """Calculate volatility-based position adjustment"""
        adjustment_map = {
            VolatilityRegime.LOW.value: 1.2,
            VolatilityRegime.NORMAL.value: 1.0,
            VolatilityRegime.HIGH.value: 0.7,
            VolatilityRegime.EXTREME.value: 0.3
        }

        return vol_regime.map(adjustment_map).fillna(1.0)

    def _calculate_position_multiplier(self, market_regime: pd.Series) -> pd.Series:
        """Calculate regime-based position multiplier"""
        multiplier_map = {
            MarketRegime.BULL_LOW_VOL.value: 1.2,
            MarketRegime.BULL_HIGH_VOL.value: 0.8,
            MarketRegime.BEAR_LOW_VOL.value: 0.6,
            MarketRegime.BEAR_HIGH_VOL.value: 0.3,
            MarketRegime.SIDEWAYS.value: 0.7,
            MarketRegime.CRISIS.value: 0.0
        }

        return market_regime.map(multiplier_map).fillna(0.5)

    # =========================================================================
    # ISSUE 1.3 FIX: Point-in-Time HMM Regime Detection
    # =========================================================================

    def compute_hmm_regime_pit(
        self,
        returns: pd.Series,
        n_states: int = 3,
        min_samples: int = 252,
        refit_frequency: int = 21
    ) -> pd.DataFrame:
        """
        Point-in-Time HMM regime detection.

        FIXES Issue 1.3: HMM models trained on entire dataset cause look-ahead
        bias. This method uses expanding window - at each point, HMM is fit
        using only past data.

        Args:
            returns: Return series
            n_states: Number of hidden states
            min_samples: Minimum samples needed before fitting HMM
            refit_frequency: How often to refit the HMM (bars)

        Returns:
            DataFrame with point-in-time HMM predictions
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed, using fallback PIT method")
            return self._fallback_hmm_pit(returns, min_samples)

        returns_clean = returns.dropna()
        n_samples = len(returns_clean)

        if n_samples < min_samples:
            logger.warning(
                f"Insufficient samples ({n_samples}) for PIT HMM. "
                f"Need at least {min_samples}."
            )
            return self._fallback_hmm_pit(returns, min_samples)

        # Initialize result arrays
        result = pd.DataFrame(index=returns.index)
        result['hmm_state_pit'] = np.nan
        result['hmm_regime_pit'] = np.nan
        result['hmm_confidence_pit'] = np.nan

        # Track last fit index to avoid refitting every bar
        last_fit_idx = 0
        current_model = None
        current_regime_map = None

        for i in range(min_samples, n_samples):
            current_idx = returns_clean.index[i]

            # Check if we need to refit
            should_refit = (
                current_model is None or
                (i - last_fit_idx) >= refit_frequency
            )

            if should_refit:
                # Fit HMM using only data up to current point (not including current)
                train_data = returns_clean.iloc[:i].values.reshape(-1, 1)

                try:
                    model = hmm.GaussianHMM(
                        n_components=n_states,
                        covariance_type="full",
                        n_iter=50,  # Fewer iterations for speed
                        random_state=42
                    )
                    model.fit(train_data)
                    current_model = model
                    last_fit_idx = i

                    # Create regime mapping based on state means
                    state_means = model.means_.flatten()
                    state_order = np.argsort(state_means)

                    current_regime_map = {
                        state_order[0]: 'bear',
                        state_order[-1]: 'bull',
                    }
                    if len(state_order) > 2:
                        for j in state_order[1:-1]:
                            current_regime_map[j] = 'neutral'

                except Exception as e:
                    logger.debug(f"HMM fit failed at index {i}: {e}")
                    continue

            # Predict current state using past-only model
            if current_model is not None:
                try:
                    current_return = returns_clean.iloc[[i]].values.reshape(-1, 1)
                    state = current_model.predict(current_return)[0]
                    proba = current_model.predict_proba(current_return)[0]

                    result.loc[current_idx, 'hmm_state_pit'] = state
                    result.loc[current_idx, 'hmm_regime_pit'] = current_regime_map.get(state, 'unknown')
                    result.loc[current_idx, 'hmm_confidence_pit'] = np.max(proba)

                except Exception as e:
                    logger.debug(f"HMM predict failed at index {i}: {e}")

        # Log statistics
        valid_count = result['hmm_state_pit'].notna().sum()
        logger.info(
            f"PIT HMM regime detection complete: {valid_count}/{len(returns)} "
            f"predictions, refit every {refit_frequency} bars"
        )

        return result

    def _fallback_hmm_pit(
        self,
        returns: pd.Series,
        min_samples: int = 252
    ) -> pd.DataFrame:
        """
        Point-in-time fallback when hmmlearn not available.

        Uses expanding window volatility percentile ranking.
        """
        result = pd.DataFrame(index=returns.index)
        result['hmm_state_pit'] = np.nan
        result['hmm_regime_pit'] = np.nan
        result['hmm_confidence_pit'] = np.nan

        returns_clean = returns.dropna()

        for i in range(min_samples, len(returns_clean)):
            current_idx = returns_clean.index[i]

            # Use only past data for percentile calculation
            past_returns = returns_clean.iloc[:i]

            # Calculate rolling volatility
            vol = past_returns.rolling(20).std()
            current_vol = vol.iloc[-1]

            # Percentile rank within past data only
            vol_percentile = (vol < current_vol).mean()

            # Classify
            if vol_percentile < 0.33:
                state = 0
                regime = 'low_vol'
            elif vol_percentile < 0.67:
                state = 1
                regime = 'normal'
            else:
                state = 2
                regime = 'high_vol'

            result.loc[current_idx, 'hmm_state_pit'] = state
            result.loc[current_idx, 'hmm_regime_pit'] = regime
            result.loc[current_idx, 'hmm_confidence_pit'] = abs(vol_percentile - 0.5) * 2

        return result

    def generate_regime_features_pit(
        self,
        prices: pd.Series,
        window: int = 20,
        min_samples: int = 252
    ) -> pd.DataFrame:
        """
        Generate point-in-time regime features (no look-ahead bias).

        FIXES Issue 1.3: This is the preferred method for generating regime
        features for model training. Uses expanding window approach where
        each feature only uses data available at that point in time.

        Args:
            prices: Price series
            window: Lookback window for basic regime detection
            min_samples: Minimum samples for HMM fitting

        Returns:
            DataFrame with point-in-time regime features
        """
        returns = prices.pct_change()
        features = pd.DataFrame(index=prices.index)

        # Basic regime detection (already point-in-time as it uses rolling windows)
        basic_regime = self.detect_regime(prices, returns, window)
        features = pd.concat([features, basic_regime], axis=1)

        # Point-in-time HMM regime (ISSUE 1.3 FIX)
        try:
            hmm_regime_pit = self.compute_hmm_regime_pit(returns, min_samples=min_samples)
            features = pd.concat([features, hmm_regime_pit], axis=1)
        except Exception as e:
            logger.warning(f"PIT HMM regime detection failed: {e}")

        # Regime-based adjustments
        features['regime_vol_adjustment'] = self._calculate_vol_adjustment(
            features.get('vol_regime', pd.Series(index=prices.index))
        )

        features['regime_position_multiplier'] = self._calculate_position_multiplier(
            features.get('market_regime', pd.Series(index=prices.index))
        )

        logger.info("Generated point-in-time regime features (no look-ahead bias)")

        return features
