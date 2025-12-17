"""
Hidden Markov Model Regime Detection.

This module implements HMM-based regime classification for identifying
distinct market states (e.g., bull, bear, sideways markets).

Reference:
    Hamilton, J.D. (1989) - "A New Approach to the Economic Analysis
    of Nonstationary Time Series and the Business Cycle"

Designed for JPMorgan-level institutional requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegimeState(Enum):
    """Market regime states."""

    BULL = "bull"               # High returns, low volatility
    BEAR = "bear"               # Negative returns, high volatility
    SIDEWAYS = "sideways"       # Low returns, low volatility
    HIGH_VOL = "high_volatility"  # Any returns, very high volatility
    CRISIS = "crisis"           # Large negative returns, extreme volatility


@dataclass
class RegimeResult:
    """Container for regime detection results."""

    states: pd.Series           # Regime state at each timestamp
    probabilities: pd.DataFrame  # Probability of each state
    transition_matrix: np.ndarray  # State transition probabilities
    state_means: np.ndarray     # Mean return in each state
    state_variances: np.ndarray  # Variance in each state
    log_likelihood: float       # Model fit quality
    n_states: int
    converged: bool

    def get_current_regime(self) -> RegimeState:
        """Get the most recent regime."""
        return self.states.iloc[-1]

    def get_regime_durations(self) -> Dict[RegimeState, float]:
        """Calculate average duration of each regime."""
        durations = {}
        for state in RegimeState:
            if state in self.states.values:
                # Find consecutive runs of this state
                runs = (self.states != state).cumsum()
                run_lengths = self.states.groupby(runs).size()
                state_runs = run_lengths[self.states.groupby(runs).first() == state]
                durations[state] = state_runs.mean() if len(state_runs) > 0 else 0
        return durations

    def get_regime_frequency(self) -> Dict[RegimeState, float]:
        """Calculate frequency of each regime."""
        return self.states.value_counts(normalize=True).to_dict()


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detector.

    Uses Gaussian HMM to identify distinct market regimes from
    observable features (returns, volatility, etc.).

    Example usage:
        detector = HMMRegimeDetector(n_states=3)

        # Fit on historical data
        detector.fit(returns, volatility)

        # Get regime probabilities
        result = detector.predict(returns, volatility)

        # Use in strategy
        if result.get_current_regime() == RegimeState.BEAR:
            reduce_exposure()
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        min_samples: int = 252,
    ) -> None:
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ("full", "diag", "spherical")
            n_iter: Maximum iterations for EM algorithm
            tol: Convergence tolerance
            random_state: Random seed
            min_samples: Minimum samples required for fitting
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMMRegimeDetector. "
                "Install with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.min_samples = min_samples

        self._model: Optional[GaussianHMM] = None
        self._state_labels: Dict[int, RegimeState] = {}
        self._is_fitted = False

    def fit(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume_ratio: Optional[pd.Series] = None,
    ) -> "HMMRegimeDetector":
        """
        Fit HMM to market data.

        Args:
            returns: Return series
            volatility: Optional volatility series
            volume_ratio: Optional volume ratio series

        Returns:
            self
        """
        # Prepare features
        features = self._prepare_features(returns, volatility, volume_ratio)

        if len(features) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(features)} < {self.min_samples}"
            )

        # Initialize and fit HMM
        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self._model.fit(features)

        # Label states based on characteristics
        self._label_states(features)

        self._is_fitted = True

        logger.info(
            f"HMM fitted with {self.n_states} states, "
            f"log_likelihood={self._model.score(features):.2f}"
        )

        return self

    def predict(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume_ratio: Optional[pd.Series] = None,
    ) -> RegimeResult:
        """
        Predict regime states.

        Args:
            returns: Return series
            volatility: Optional volatility series
            volume_ratio: Optional volume ratio series

        Returns:
            RegimeResult with states and probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        features = self._prepare_features(returns, volatility, volume_ratio)

        # Predict states
        state_indices = self._model.predict(features)

        # Get state probabilities
        probs = self._model.predict_proba(features)

        # Convert to labeled states
        states = pd.Series(
            [self._state_labels.get(s, RegimeState.SIDEWAYS) for s in state_indices],
            index=returns.index[-len(state_indices):],
            name="regime",
        )

        # Create probability DataFrame
        prob_df = pd.DataFrame(
            probs,
            index=states.index,
            columns=[self._state_labels.get(i, RegimeState.SIDEWAYS).value for i in range(self.n_states)],
        )

        return RegimeResult(
            states=states,
            probabilities=prob_df,
            transition_matrix=self._model.transmat_,
            state_means=self._model.means_.flatten(),
            state_variances=np.diag(self._model.covars_[0]) if self.covariance_type == "full"
                           else self._model.covars_.flatten(),
            log_likelihood=self._model.score(features),
            n_states=self.n_states,
            converged=self._model.monitor_.converged,
        )

    def predict_proba(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Get probability of each regime."""
        result = self.predict(returns, volatility)
        return result.probabilities

    def get_transition_probabilities(self) -> pd.DataFrame:
        """Get regime transition probability matrix."""
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        labels = [self._state_labels.get(i, RegimeState.SIDEWAYS).value
                  for i in range(self.n_states)]

        return pd.DataFrame(
            self._model.transmat_,
            index=labels,
            columns=labels,
        )

    def expected_regime_duration(self, state: RegimeState) -> float:
        """
        Calculate expected duration (in periods) of a regime.

        Based on transition matrix: E[duration] = 1 / (1 - P_ii)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        state_idx = None
        for idx, s in self._state_labels.items():
            if s == state:
                state_idx = idx
                break

        if state_idx is None:
            return 0.0

        stay_prob = self._model.transmat_[state_idx, state_idx]
        return 1 / (1 - stay_prob) if stay_prob < 1 else float('inf')

    def _prepare_features(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume_ratio: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Prepare feature matrix for HMM."""
        features = [returns.values.reshape(-1, 1)]

        if volatility is not None:
            features.append(volatility.values.reshape(-1, 1))

        if volume_ratio is not None:
            features.append(volume_ratio.values.reshape(-1, 1))

        X = np.hstack(features)

        # Handle NaN/inf
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]

        return X

    def _label_states(self, features: np.ndarray) -> None:
        """
        Label HMM states with interpretable regime names.

        States are labeled based on their mean returns and variances.
        """
        # Get state characteristics
        state_indices = self._model.predict(features)

        # Calculate mean return for each state
        returns = features[:, 0]
        state_returns = {}
        state_volatility = {}

        for s in range(self.n_states):
            mask = state_indices == s
            if mask.sum() > 0:
                state_returns[s] = returns[mask].mean()
                state_volatility[s] = returns[mask].std()

        # Sort states by return
        sorted_states = sorted(state_returns.keys(), key=lambda x: state_returns[x])

        # Assign labels
        if self.n_states == 2:
            self._state_labels[sorted_states[0]] = RegimeState.BEAR
            self._state_labels[sorted_states[1]] = RegimeState.BULL

        elif self.n_states == 3:
            self._state_labels[sorted_states[0]] = RegimeState.BEAR
            self._state_labels[sorted_states[1]] = RegimeState.SIDEWAYS
            self._state_labels[sorted_states[2]] = RegimeState.BULL

        elif self.n_states == 4:
            # Check for high volatility states
            vol_sorted = sorted(state_volatility.keys(), key=lambda x: state_volatility[x])
            high_vol_state = vol_sorted[-1]

            self._state_labels[sorted_states[0]] = RegimeState.BEAR
            self._state_labels[sorted_states[1]] = RegimeState.SIDEWAYS
            self._state_labels[sorted_states[2]] = RegimeState.BULL
            self._state_labels[high_vol_state] = RegimeState.HIGH_VOL

        else:
            # Generic labeling for more states
            for i, s in enumerate(sorted_states):
                if i < len(sorted_states) // 3:
                    self._state_labels[s] = RegimeState.BEAR
                elif i < 2 * len(sorted_states) // 3:
                    self._state_labels[s] = RegimeState.SIDEWAYS
                else:
                    self._state_labels[s] = RegimeState.BULL

        logger.debug(f"State labels: {self._state_labels}")


def detect_regimes(
    returns: pd.Series,
    volatility: Optional[pd.Series] = None,
    n_states: int = 3,
) -> RegimeResult:
    """
    Convenience function for regime detection.

    Args:
        returns: Return series
        volatility: Optional volatility series
        n_states: Number of regimes

    Returns:
        RegimeResult
    """
    detector = HMMRegimeDetector(n_states=n_states)
    detector.fit(returns, volatility)
    return detector.predict(returns, volatility)
