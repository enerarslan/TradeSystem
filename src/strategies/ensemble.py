"""
Ensemble Strategy for AlphaTrade system.

Combines signals from multiple strategies using
various combination methods.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy, SignalCombiner, SignalFilter


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy combining multiple sub-strategies.

    Combination methods:
    - Weighted average
    - Majority voting
    - Meta-model learning
    """

    DEFAULT_PARAMS = {
        "combination_method": "weighted_average",
        "min_agreement": 0.6,
        "signal_smoothing": 3,
        "dynamic_weights": False,
        "lookback_for_weights": 60,
    }

    def __init__(
        self,
        strategies: list[BaseStrategy],
        weights: list[float] | None = None,
        name: str = "Ensemble",
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ensemble strategy.

        Args:
            strategies: List of sub-strategies
            weights: Weights for each strategy (equal if None)
            name: Strategy name
            params: Strategy parameters
        """
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)

        super().__init__(name, merged_params)

        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)

        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")

        logger.info(
            f"Ensemble initialized with {len(strategies)} strategies: "
            f"{[s.name for s in strategies]}"
        )

    def _calculate_dynamic_weights(
        self,
        strategy_signals: list[pd.DataFrame],
        returns: pd.DataFrame,
        lookback: int,
    ) -> list[float]:
        """
        Calculate dynamic weights based on recent performance.

        Args:
            strategy_signals: List of signal DataFrames
            returns: Actual returns
            lookback: Lookback period for evaluation

        Returns:
            Dynamic weights
        """
        performances = []

        for signals in strategy_signals:
            # Calculate strategy returns
            strategy_returns = (signals.shift(1) * returns).sum(axis=1)

            # Recent Sharpe ratio
            recent_returns = strategy_returns.tail(lookback)
            if len(recent_returns) > 10:
                sharpe = (
                    recent_returns.mean() / (recent_returns.std() + 1e-10)
                ) * np.sqrt(252 * 26)
            else:
                sharpe = 0

            performances.append(max(sharpe, 0))  # Only positive contributions

        # Normalize to weights
        total = sum(performances) + 1e-10
        weights = [p / total for p in performances]

        # Apply minimum weight constraint
        min_weight = 0.1 / len(self.strategies)
        weights = [max(w, min_weight) for w in weights]

        # Renormalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return weights

    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate ensemble signals.

        Args:
            data: OHLCV data
            features: Pre-computed features

        Returns:
            Combined signal DataFrame
        """
        # Generate signals from each strategy
        strategy_signals = []

        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data, features)
                strategy_signals.append(signals)
                logger.debug(f"Generated signals from {strategy.name}")
            except Exception as e:
                logger.error(f"Error generating signals from {strategy.name}: {e}")
                # Create neutral signals as fallback
                if isinstance(data, dict):
                    symbols = list(data.keys())
                    idx = list(data.values())[0].index
                else:
                    symbols = data.columns if hasattr(data, "columns") else ["signal"]
                    idx = data.index

                neutral = pd.DataFrame(0, index=idx, columns=symbols)
                strategy_signals.append(neutral)

        if not strategy_signals:
            raise ValueError("No strategy signals generated")

        # Align all signals to same index/columns
        common_idx = strategy_signals[0].index
        common_cols = strategy_signals[0].columns

        for signals in strategy_signals[1:]:
            common_idx = common_idx.intersection(signals.index)
            common_cols = common_cols.intersection(signals.columns)

        strategy_signals = [
            s.loc[common_idx, common_cols] for s in strategy_signals
        ]

        # Calculate weights
        weights = self.weights
        if self.params["dynamic_weights"]:
            if isinstance(data, dict):
                returns = pd.DataFrame(
                    {sym: df["close"].pct_change() for sym, df in data.items()}
                ).loc[common_idx, common_cols]
            else:
                returns = data["close"].pct_change()

            weights = self._calculate_dynamic_weights(
                strategy_signals, returns, self.params["lookback_for_weights"]
            )
            logger.debug(f"Dynamic weights: {weights}")

        # Combine signals
        method = self.params["combination_method"]

        if method == "weighted_average":
            combined = SignalCombiner.weighted_average(strategy_signals, weights)

        elif method == "voting":
            combined = SignalCombiner.voting(
                strategy_signals, self.params["min_agreement"]
            )

        elif method == "rank_average":
            combined = SignalCombiner.rank_average(strategy_signals)

        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Apply signal smoothing
        if self.params["signal_smoothing"] > 1:
            combined = SignalFilter.smooth(
                combined, self.params["signal_smoothing"]
            )

        # Discretize signals
        combined = self._discretize_signals(combined)

        logger.info(f"Generated ensemble signals from {len(self.strategies)} strategies")
        return combined

    def _discretize_signals(
        self,
        signals: pd.DataFrame,
        threshold: float = 0.3,
    ) -> pd.DataFrame:
        """
        Convert continuous signals to discrete (-1, 0, 1).

        Args:
            signals: Continuous signal DataFrame
            threshold: Threshold for non-zero signal

        Returns:
            Discretized signals
        """
        discrete = pd.DataFrame(0, index=signals.index, columns=signals.columns)
        discrete[signals > threshold] = 1
        discrete[signals < -threshold] = -1

        return discrete

    def calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Convert signals to position weights.

        Args:
            signals: Signal DataFrame
            prices: Current prices
            capital: Available capital

        Returns:
            Position weights
        """
        positions = signals.copy().astype(float)

        for idx in positions.index:
            row = positions.loc[idx]
            n_positions = (row != 0).sum()

            if n_positions > 0:
                weight = 1.0 / n_positions
                positions.loc[idx] = row * weight

        return positions

    def get_strategy_contributions(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Analyze contribution of each strategy to final signals.

        Args:
            data: OHLCV data

        Returns:
            DataFrame with strategy contribution statistics
        """
        strategy_signals = []

        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            strategy_signals.append(signals)

        contributions = []
        for i, (strategy, signals) in enumerate(zip(self.strategies, strategy_signals)):
            flat = signals.values.flatten()
            flat = flat[~np.isnan(flat)]

            contributions.append({
                "strategy": strategy.name,
                "weight": self.weights[i],
                "avg_signal": np.mean(flat),
                "signal_std": np.std(flat),
                "long_pct": (flat > 0).mean() * 100,
                "short_pct": (flat < 0).mean() * 100,
                "neutral_pct": (flat == 0).mean() * 100,
            })

        return pd.DataFrame(contributions)

    def add_strategy(
        self,
        strategy: BaseStrategy,
        weight: float = None,
    ) -> None:
        """
        Add a strategy to the ensemble.

        Args:
            strategy: Strategy to add
            weight: Weight for the strategy
        """
        self.strategies.append(strategy)

        if weight is None:
            # Equal weights
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies)
        else:
            self.weights.append(weight)
            # Renormalize
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        logger.info(f"Added strategy {strategy.name} to ensemble")

    def remove_strategy(self, name: str) -> bool:
        """
        Remove a strategy from the ensemble.

        Args:
            name: Strategy name to remove

        Returns:
            True if removed successfully
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == name:
                self.strategies.pop(i)
                self.weights.pop(i)

                # Renormalize weights
                if self.weights:
                    total = sum(self.weights)
                    self.weights = [w / total for w in self.weights]

                logger.info(f"Removed strategy {name} from ensemble")
                return True

        return False

    def set_weights(self, weights: list[float]) -> None:
        """
        Set strategy weights.

        Args:
            weights: New weights
        """
        if len(weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")

        # Normalize
        total = sum(weights)
        self.weights = [w / total for w in weights]

        logger.info(f"Updated ensemble weights: {self.weights}")
