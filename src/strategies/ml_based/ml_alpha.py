"""
ML-Based Alpha Strategy for AlphaTrade system.

This strategy uses machine learning models to predict
forward returns and generate trading signals.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.strategies.base import BaseStrategy
from src.features.pipeline import FeaturePipeline


class MLAlphaStrategy(BaseStrategy):
    """
    Machine learning based alpha strategy.

    Uses gradient boosting (LightGBM/XGBoost) to predict
    forward returns and generate trading signals.
    """

    DEFAULT_PARAMS = {
        "model_type": "lightgbm",  # lightgbm, xgboost, random_forest
        "prediction_horizon": 5,
        "classification": True,
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_samples_train": 1000,
        "feature_selection": True,
        "n_features": 50,
        "top_n_long": 5,
        "bottom_n_short": 0,
        "signal_threshold": 0.5,
    }

    def __init__(
        self,
        name: str = "MLAlpha",
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)

        super().__init__(name, merged_params)

        self._model = None
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._feature_importance: pd.Series | None = None
        self._is_fitted = False

    def _create_model(self):
        """Create the ML model based on parameters."""
        model_type = self.params["model_type"]
        is_classification = self.params["classification"]

        common_params = {
            "n_estimators": self.params["n_estimators"],
            "max_depth": self.params["max_depth"],
            "random_state": 42,
        }

        if model_type == "lightgbm" and HAS_LIGHTGBM:
            if is_classification:
                return LGBMClassifier(
                    **common_params,
                    learning_rate=self.params["learning_rate"],
                    verbose=-1,
                )
            else:
                return LGBMRegressor(
                    **common_params,
                    learning_rate=self.params["learning_rate"],
                    verbose=-1,
                )

        elif model_type == "xgboost" and HAS_XGBOOST:
            if is_classification:
                return XGBClassifier(
                    **common_params,
                    learning_rate=self.params["learning_rate"],
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            else:
                return XGBRegressor(
                    **common_params,
                    learning_rate=self.params["learning_rate"],
                )

        else:
            # Fallback to Random Forest
            if is_classification:
                return RandomForestClassifier(**common_params)
            else:
                return RandomForestRegressor(**common_params)

    def _prepare_target(
        self,
        prices: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Prepare target variable (forward returns).

        Args:
            prices: Price panel
            horizon: Forward prediction horizon

        Returns:
            Target DataFrame
        """
        forward_returns = prices.pct_change(horizon).shift(-horizon)

        if self.params["classification"]:
            # Binary classification: 1 if positive return, 0 otherwise
            return (forward_returns > 0).astype(int)
        else:
            return forward_returns

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """
        Fit the ML model.

        Args:
            features: Feature DataFrame
            target: Target variable
        """
        # Remove rows with NaN
        mask = features.notna().all(axis=1) & target.notna()
        X = features.loc[mask]
        y = target.loc[mask]

        if len(X) < self.params["min_samples_train"]:
            logger.warning(f"Insufficient training samples: {len(X)}")
            return

        # Feature selection if enabled
        if self.params["feature_selection"]:
            X = self._select_features(X, y)

        self._feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Create and fit model
        self._model = self._create_model()
        self._model.fit(X_scaled, y)

        # Store feature importance
        if hasattr(self._model, "feature_importances_"):
            self._feature_importance = pd.Series(
                self._model.feature_importances_,
                index=self._feature_names,
            ).sort_values(ascending=False)

        self._is_fitted = True
        logger.info(f"ML model fitted with {len(X)} samples, {len(self._feature_names)} features")

    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Select top features using feature importance.

        Args:
            X: Feature DataFrame
            y: Target variable

        Returns:
            Selected features DataFrame
        """
        n_features = min(self.params["n_features"], X.shape[1])

        # Train a quick model for feature importance
        if self.params["classification"]:
            selector_model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42
            )
        else:
            selector_model = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42
            )

        # Handle NaN for selector
        X_temp = X.fillna(0)
        selector_model.fit(X_temp, y)

        importance = pd.Series(
            selector_model.feature_importances_,
            index=X.columns,
        )

        top_features = importance.nlargest(n_features).index.tolist()

        return X[top_features]

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        """
        Generate predictions.

        Args:
            features: Feature DataFrame

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Align features
        X = features.reindex(columns=self._feature_names, fill_value=0)

        # Handle NaN
        X = X.fillna(0)

        # Scale
        X_scaled = self._scaler.transform(X)

        # Predict
        if self.params["classification"]:
            # Get probability of class 1 (positive return)
            proba = self._model.predict_proba(X_scaled)[:, 1]
            return pd.Series(proba, index=features.index)
        else:
            pred = self._model.predict(X_scaled)
            return pd.Series(pred, index=features.index)

    def generate_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        features: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Generate ML-based signals.

        Args:
            data: OHLCV data
            features: Pre-computed features

        Returns:
            Signal DataFrame
        """
        # Convert data to price panel
        if isinstance(data, dict):
            prices = pd.DataFrame({sym: df["close"] for sym, df in data.items()})
        else:
            prices = data["close"].unstack()

        # Generate features if not provided
        if features is None:
            logger.info("Generating features...")
            pipeline = FeaturePipeline()
            features = {}
            for sym, df in data.items():
                features[sym] = pipeline.generate_features(df)

        # Prepare predictions for each symbol
        predictions = pd.DataFrame(index=prices.index, columns=prices.columns)

        for symbol in prices.columns:
            if symbol not in features:
                continue

            sym_features = features[symbol]
            sym_prices = prices[symbol].to_frame("close")

            # Prepare target
            target = self._prepare_target(sym_prices, self.params["prediction_horizon"])
            target = target["close"]

            # Align features and target
            common_idx = sym_features.index.intersection(target.index)
            sym_features = sym_features.loc[common_idx]
            target = target.loc[common_idx]

            # Split into train/predict periods
            train_end = int(len(common_idx) * 0.7)
            train_idx = common_idx[:train_end]
            predict_idx = common_idx[train_end:]

            # Fit model
            self.fit(sym_features.loc[train_idx], target.loc[train_idx])

            if self._is_fitted:
                # Predict for all periods
                pred = self.predict(sym_features)
                predictions[symbol] = pred

        # Convert predictions to signals
        signals = self._predictions_to_signals(predictions)

        logger.debug(f"Generated ML signals: shape={signals.shape}")
        return signals

    def _predictions_to_signals(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert predictions to trading signals.

        Args:
            predictions: Model predictions

        Returns:
            Signal DataFrame
        """
        signals = pd.DataFrame(0, index=predictions.index, columns=predictions.columns)

        top_n = self.params["top_n_long"]
        bottom_n = self.params["bottom_n_short"]
        threshold = self.params["signal_threshold"]

        for idx in predictions.index:
            row = predictions.loc[idx].dropna()

            if len(row) == 0:
                continue

            # Rank predictions
            ranked = row.rank(ascending=False)

            # Long top predictions above threshold
            for symbol in ranked.nsmallest(top_n).index:
                if row[symbol] > threshold:
                    signals.loc[idx, symbol] = 1

            # Short bottom predictions below threshold
            if bottom_n > 0:
                for symbol in ranked.nlargest(bottom_n).index:
                    if row[symbol] < (1 - threshold):
                        signals.loc[idx, symbol] = -1

        return signals

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

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature importance scores."""
        return self._feature_importance
