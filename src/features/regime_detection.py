"""
Market Regime Detection Module.

JPMorgan Institutional-Level Market Regime Classification.

Implements multiple approaches for detecting market regimes:
1. Hidden Markov Models (HMM) for regime switching
2. Volatility-based regime classification
3. Trend-based regime detection
4. Clustering-based regime identification

Reference:
    "Advances in Financial Machine Learning" by de Prado (2018)
    Chapter 10: Regime Detection

Market regimes typically identified:
- Bull/Bear markets (trend-based)
- High/Low volatility regimes
- Mean-reverting vs Trending
- Risk-on vs Risk-off
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    GMM_AVAILABLE = True
except ImportError:
    GMM_AVAILABLE = False


logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Types of market regimes."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regimes: pd.Series  # Regime labels
    regime_probabilities: Optional[pd.DataFrame] = None  # Probability of each regime
    transition_matrix: Optional[np.ndarray] = None  # Regime transition probabilities
    regime_stats: Optional[Dict[int, Dict[str, float]]] = None  # Stats per regime


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detection.

    Uses HMM to identify latent market states based on
    returns and volatility features.

    This is the gold standard for regime detection in institutional
    quantitative finance.

    Example:
        detector = HMMRegimeDetector(n_regimes=3)
        detector.fit(df)
        regimes = detector.predict(df)
    """

    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        features: List[str] = None,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: Type of covariance parameters
            n_iter: Number of EM iterations
            random_state: Random seed for reproducibility
            features: List of features to use for regime detection
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for HMM regime detection")

        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.features = features or ["returns", "volatility"]

        self.model_ = None
        self.scaler_ = StandardScaler()
        self.fitted_ = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for HMM."""
        feature_data = []

        if "returns" in self.features:
            if "returns" in df.columns:
                feature_data.append(df["returns"].values)
            else:
                returns = df["close"].pct_change().fillna(0)
                feature_data.append(returns.values)

        if "volatility" in self.features:
            if "volatility" in df.columns:
                feature_data.append(df["volatility"].values)
            else:
                returns = df["close"].pct_change().fillna(0)
                vol = returns.rolling(20).std().fillna(method="bfill").fillna(0.01)
                feature_data.append(vol.values)

        if "volume_change" in self.features:
            if "volume" in df.columns:
                vol_change = df["volume"].pct_change().fillna(0).clip(-5, 5)
                feature_data.append(vol_change.values)

        if "momentum" in self.features:
            momentum = df["close"].pct_change(20).fillna(0)
            feature_data.append(momentum.values)

        X = np.column_stack(feature_data)
        return X

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the HMM to the data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Self
        """
        X = self._prepare_features(df)
        X_scaled = self.scaler_.fit_transform(X)

        # Remove any NaN/Inf
        mask = np.isfinite(X_scaled).all(axis=1)
        X_clean = X_scaled[mask]

        self.model_ = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.model_.fit(X_clean)
        self.fitted_ = True

        logger.info(
            f"HMM fitted with {self.n_regimes} regimes. "
            f"Log-likelihood: {self.model_.score(X_clean):.2f}"
        )

        return self

    def predict(self, df: pd.DataFrame) -> RegimeResult:
        """
        Predict regimes for the data.

        Args:
            df: OHLCV DataFrame

        Returns:
            RegimeResult with regime labels and probabilities
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = self._prepare_features(df)
        X_scaled = self.scaler_.transform(X)

        # Handle NaN/Inf
        mask = np.isfinite(X_scaled).all(axis=1)
        X_clean = X_scaled.copy()
        X_clean[~mask] = 0  # Replace with neutral values

        # Predict regimes
        regimes = self.model_.predict(X_clean)

        # Get probabilities
        proba = self.model_.predict_proba(X_clean)

        # Create result
        regime_series = pd.Series(regimes, index=df.index, name="regime")
        proba_df = pd.DataFrame(
            proba,
            index=df.index,
            columns=[f"regime_{i}_prob" for i in range(self.n_regimes)]
        )

        # Calculate regime statistics
        returns = df["close"].pct_change().fillna(0)
        regime_stats = {}
        for i in range(self.n_regimes):
            mask = regimes == i
            regime_stats[i] = {
                "mean_return": returns[mask].mean() if mask.any() else 0,
                "volatility": returns[mask].std() if mask.any() else 0,
                "sharpe": (returns[mask].mean() / returns[mask].std() * np.sqrt(252))
                         if mask.any() and returns[mask].std() > 0 else 0,
                "frequency": mask.mean(),
                "avg_duration": self._calculate_avg_duration(regime_series, i),
            }

        return RegimeResult(
            regimes=regime_series,
            regime_probabilities=proba_df,
            transition_matrix=self.model_.transmat_,
            regime_stats=regime_stats,
        )

    def _calculate_avg_duration(self, regimes: pd.Series, regime: int) -> float:
        """Calculate average duration of a regime."""
        in_regime = (regimes == regime).astype(int)
        transitions = in_regime.diff().abs()
        n_transitions = transitions.sum()
        if n_transitions == 0:
            return len(regimes) if in_regime.iloc[0] == 1 else 0
        return in_regime.sum() / (n_transitions / 2)

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        """
        Get the current (most recent) regime and its probabilities.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (regime_label, probabilities)
        """
        result = self.predict(df)
        current_regime = result.regimes.iloc[-1]
        current_proba = result.regime_probabilities.iloc[-1].values
        return current_regime, current_proba


class VolatilityRegimeDetector:
    """
    Volatility-based regime detection.

    Classifies market into volatility regimes using percentile-based
    thresholds or clustering.

    This is a simpler, more interpretable approach than HMM.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        method: str = "percentile",
        lookback: int = 20,
        percentiles: List[float] = None,
    ):
        """
        Initialize volatility regime detector.

        Args:
            n_regimes: Number of volatility regimes
            method: "percentile" or "kmeans"
            lookback: Rolling window for volatility calculation
            percentiles: Percentile thresholds for regime boundaries
        """
        self.n_regimes = n_regimes
        self.method = method
        self.lookback = lookback
        self.percentiles = percentiles or self._default_percentiles()

        self.thresholds_ = None
        self.kmeans_ = None
        self.fitted_ = False

    def _default_percentiles(self) -> List[float]:
        """Default percentile boundaries."""
        if self.n_regimes == 2:
            return [50]
        elif self.n_regimes == 3:
            return [33, 67]
        else:
            step = 100 / self.n_regimes
            return [step * i for i in range(1, self.n_regimes)]

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rolling volatility."""
        returns = df["close"].pct_change()
        volatility = returns.rolling(self.lookback).std() * np.sqrt(252)
        return volatility.fillna(method="bfill").fillna(0.15)

    def fit(self, df: pd.DataFrame) -> "VolatilityRegimeDetector":
        """
        Fit the regime detector.

        Args:
            df: OHLCV DataFrame

        Returns:
            Self
        """
        volatility = self._calculate_volatility(df)

        if self.method == "percentile":
            self.thresholds_ = [
                np.percentile(volatility.dropna(), p)
                for p in self.percentiles
            ]
        elif self.method == "kmeans":
            self.kmeans_ = KMeans(n_clusters=self.n_regimes, random_state=42)
            self.kmeans_.fit(volatility.values.reshape(-1, 1))

        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> RegimeResult:
        """
        Predict volatility regimes.

        Args:
            df: OHLCV DataFrame

        Returns:
            RegimeResult
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")

        volatility = self._calculate_volatility(df)

        if self.method == "percentile":
            regimes = np.zeros(len(volatility), dtype=int)
            for i, threshold in enumerate(self.thresholds_):
                regimes[volatility > threshold] = i + 1
        elif self.method == "kmeans":
            regimes = self.kmeans_.predict(volatility.values.reshape(-1, 1))
            # Sort regimes by volatility level
            centers = self.kmeans_.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            regime_map = {old: new for new, old in enumerate(sorted_indices)}
            regimes = np.array([regime_map[r] for r in regimes])

        regime_series = pd.Series(regimes, index=df.index, name="vol_regime")

        # Calculate regime statistics
        returns = df["close"].pct_change().fillna(0)
        regime_stats = {}
        for i in range(self.n_regimes):
            mask = regimes == i
            regime_stats[i] = {
                "mean_return": returns[mask].mean() if mask.any() else 0,
                "mean_volatility": volatility[mask].mean() if mask.any() else 0,
                "frequency": mask.mean(),
            }

        return RegimeResult(
            regimes=regime_series,
            regime_stats=regime_stats,
        )


class TrendRegimeDetector:
    """
    Trend-based regime detection.

    Classifies market into trending vs mean-reverting regimes
    using various trend indicators.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        atr_window: int = 14,
        adx_threshold: float = 25.0,
    ):
        """
        Initialize trend regime detector.

        Args:
            short_window: Short MA window
            long_window: Long MA window
            atr_window: ATR calculation window
            adx_threshold: ADX threshold for trending classification
        """
        self.short_window = short_window
        self.long_window = long_window
        self.atr_window = atr_window
        self.adx_threshold = adx_threshold

    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        plus_di = 100 * (plus_dm.rolling(self.atr_window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.atr_window).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.atr_window).mean()

        return adx.fillna(0)

    def predict(self, df: pd.DataFrame) -> RegimeResult:
        """
        Predict trend regimes.

        Regimes:
        - 0: Mean-reverting (low ADX)
        - 1: Trending up (high ADX, price > MA)
        - 2: Trending down (high ADX, price < MA)

        Args:
            df: OHLCV DataFrame

        Returns:
            RegimeResult
        """
        adx = self._calculate_adx(df)

        ma_short = df["close"].rolling(self.short_window).mean()
        ma_long = df["close"].rolling(self.long_window).mean()

        # Classify regimes
        regimes = np.zeros(len(df), dtype=int)

        # Trending up: ADX > threshold and short MA > long MA
        trending_up = (adx > self.adx_threshold) & (ma_short > ma_long)
        regimes[trending_up] = 1

        # Trending down: ADX > threshold and short MA < long MA
        trending_down = (adx > self.adx_threshold) & (ma_short < ma_long)
        regimes[trending_down] = 2

        # Mean-reverting: ADX <= threshold (remains 0)

        regime_series = pd.Series(regimes, index=df.index, name="trend_regime")

        # Calculate statistics
        returns = df["close"].pct_change().fillna(0)
        regime_stats = {
            0: {"description": "mean_reverting", "frequency": (regimes == 0).mean()},
            1: {"description": "trending_up", "frequency": (regimes == 1).mean()},
            2: {"description": "trending_down", "frequency": (regimes == 2).mean()},
        }

        for i in range(3):
            mask = regimes == i
            if mask.any():
                regime_stats[i]["mean_return"] = returns[mask].mean()
                regime_stats[i]["volatility"] = returns[mask].std()

        return RegimeResult(
            regimes=regime_series,
            regime_stats=regime_stats,
        )


class CompositeRegimeDetector:
    """
    Composite regime detector combining multiple methods.

    Combines HMM, volatility, and trend regime detection
    for a more robust regime classification.
    """

    def __init__(
        self,
        use_hmm: bool = True,
        use_volatility: bool = True,
        use_trend: bool = True,
        hmm_n_regimes: int = 3,
        vol_n_regimes: int = 3,
    ):
        """
        Initialize composite regime detector.

        Args:
            use_hmm: Include HMM-based detection
            use_volatility: Include volatility-based detection
            use_trend: Include trend-based detection
            hmm_n_regimes: Number of HMM regimes
            vol_n_regimes: Number of volatility regimes
        """
        self.use_hmm = use_hmm and HMM_AVAILABLE
        self.use_volatility = use_volatility
        self.use_trend = use_trend

        self.hmm_detector = None
        self.vol_detector = None
        self.trend_detector = None

        if self.use_hmm:
            self.hmm_detector = HMMRegimeDetector(n_regimes=hmm_n_regimes)
        if self.use_volatility:
            self.vol_detector = VolatilityRegimeDetector(n_regimes=vol_n_regimes)
        if self.use_trend:
            self.trend_detector = TrendRegimeDetector()

    def fit(self, df: pd.DataFrame) -> "CompositeRegimeDetector":
        """
        Fit all regime detectors.

        Args:
            df: OHLCV DataFrame

        Returns:
            Self
        """
        if self.hmm_detector:
            self.hmm_detector.fit(df)
        if self.vol_detector:
            self.vol_detector.fit(df)
        # Trend detector doesn't need fitting

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes using all methods.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with regime columns from each method
        """
        results = pd.DataFrame(index=df.index)

        if self.hmm_detector and self.hmm_detector.fitted_:
            hmm_result = self.hmm_detector.predict(df)
            results["hmm_regime"] = hmm_result.regimes
            for col in hmm_result.regime_probabilities.columns:
                results[col] = hmm_result.regime_probabilities[col]

        if self.vol_detector and self.vol_detector.fitted_:
            vol_result = self.vol_detector.predict(df)
            results["vol_regime"] = vol_result.regimes

        if self.trend_detector:
            trend_result = self.trend_detector.predict(df)
            results["trend_regime"] = trend_result.regimes

        # Create composite regime (optional encoding)
        if "hmm_regime" in results and "vol_regime" in results:
            # Encode as combined regime
            results["composite_regime"] = (
                results.get("hmm_regime", 0) * 10 +
                results.get("vol_regime", 0)
            )

        return results


class RegimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate regime-based features for ML models.

    Transforms regime predictions into features suitable
    for machine learning pipelines.

    Example:
        generator = RegimeFeatureGenerator()
        generator.fit(df)
        features = generator.transform(df)
    """

    def __init__(
        self,
        n_hmm_regimes: int = 3,
        include_probabilities: bool = True,
        include_durations: bool = True,
        include_transitions: bool = True,
    ):
        """
        Initialize regime feature generator.

        Args:
            n_hmm_regimes: Number of HMM regimes
            include_probabilities: Include regime probabilities
            include_durations: Include regime duration features
            include_transitions: Include regime transition features
        """
        self.n_hmm_regimes = n_hmm_regimes
        self.include_probabilities = include_probabilities
        self.include_durations = include_durations
        self.include_transitions = include_transitions

        self.detector_ = None

    def fit(self, X: pd.DataFrame, y=None) -> "RegimeFeatureGenerator":
        """
        Fit the regime detector.

        Args:
            X: OHLCV DataFrame
            y: Ignored

        Returns:
            Self
        """
        self.detector_ = CompositeRegimeDetector(
            hmm_n_regimes=self.n_hmm_regimes,
            use_hmm=HMM_AVAILABLE,
        )
        self.detector_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime features.

        Args:
            X: OHLCV DataFrame

        Returns:
            DataFrame with regime features
        """
        if self.detector_ is None:
            raise ValueError("Must fit before transform")

        regimes = self.detector_.predict(X)
        features = regimes.copy()

        # Add regime duration features
        if self.include_durations and "hmm_regime" in regimes:
            features["regime_duration"] = self._calculate_duration(regimes["hmm_regime"])

        # Add regime transition features
        if self.include_transitions and "hmm_regime" in regimes:
            features["regime_changed"] = (regimes["hmm_regime"].diff() != 0).astype(int)
            features["bars_since_change"] = self._bars_since_change(regimes["hmm_regime"])

        return features

    def _calculate_duration(self, regimes: pd.Series) -> pd.Series:
        """Calculate how long we've been in current regime."""
        duration = pd.Series(0, index=regimes.index)
        current_duration = 0
        prev_regime = None

        for i, regime in enumerate(regimes):
            if regime == prev_regime:
                current_duration += 1
            else:
                current_duration = 1
            duration.iloc[i] = current_duration
            prev_regime = regime

        return duration

    def _bars_since_change(self, regimes: pd.Series) -> pd.Series:
        """Calculate bars since last regime change."""
        changed = regimes.diff() != 0
        groups = changed.cumsum()
        return groups.groupby(groups).cumcount() + 1

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
