"""
Model Explainability Module (SHAP Integration)
Interpretable Machine Learning for Trading Models

Features:
- SHAP value computation for predictions
- Feature importance analysis
- Local and global explanations
- Interactive visualizations
- Prediction attribution reports

Usage:
    explainer = ModelExplainer(model, X_train)
    shap_values = explainer.explain(X_test)
    explainer.plot_summary()
    report = explainer.generate_report(X_test, predictions)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import warnings
import io
import base64

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Model explainability limited.")


@dataclass
class ExplanationResult:
    """Container for SHAP explanation results"""
    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    data: np.ndarray

    # Computed summaries
    mean_abs_shap: Dict[str, float] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PredictionExplanation:
    """Explanation for a single prediction"""
    prediction: float
    base_value: float
    contributions: Dict[str, float]
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]


class ModelExplainer:
    """
    SHAP-based model explainer for trading models.

    Supports:
    - Tree-based models (LightGBM, XGBoost, CatBoost, RandomForest)
    - Linear models
    - Deep learning models (via DeepExplainer)
    - Any model via KernelExplainer (slower)
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = 'auto'
    ):
        """
        Initialize ModelExplainer.

        Args:
            model: Trained model to explain
            background_data: Background dataset for SHAP (for KernelExplainer)
            feature_names: Feature names
            explainer_type: 'auto', 'tree', 'linear', 'deep', 'kernel'
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or []
        self.explainer_type = explainer_type

        self._explainer = None
        self._shap_values = None
        self._expected_value = None

        if SHAP_AVAILABLE:
            self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            if self.explainer_type == 'auto':
                self._explainer = self._create_auto_explainer()
            elif self.explainer_type == 'tree':
                self._explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == 'linear':
                self._explainer = shap.LinearExplainer(
                    self.model,
                    self.background_data
                )
            elif self.explainer_type == 'kernel':
                self._explainer = shap.KernelExplainer(
                    self.model.predict,
                    self.background_data
                )
            else:
                self._explainer = self._create_auto_explainer()

            logger.info(f"SHAP explainer initialized: {type(self._explainer).__name__}")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._explainer = None

    def _create_auto_explainer(self):
        """Auto-detect model type and create appropriate explainer."""
        model = self.model

        # Handle wrapped models
        if hasattr(model, '_model'):
            model = model._model

        # Check for tree-based models
        model_type = type(model).__name__

        if model_type in ['LGBMClassifier', 'LGBMRegressor', 'Booster']:
            return shap.TreeExplainer(model)

        elif model_type in ['XGBClassifier', 'XGBRegressor', 'Booster']:
            return shap.TreeExplainer(model)

        elif model_type in ['CatBoostClassifier', 'CatBoostRegressor']:
            return shap.TreeExplainer(model)

        elif model_type in ['RandomForestClassifier', 'RandomForestRegressor',
                           'GradientBoostingClassifier', 'GradientBoostingRegressor']:
            return shap.TreeExplainer(model)

        elif model_type in ['LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso']:
            if self.background_data is not None:
                return shap.LinearExplainer(model, self.background_data)
            else:
                logger.warning("Linear explainer requires background data")
                return None

        else:
            # Fallback to KernelExplainer
            if self.background_data is not None:
                logger.info("Using KernelExplainer (may be slow)")
                # Sample background data for speed
                bg_sample = self.background_data
                if len(bg_sample) > 100:
                    bg_sample = shap.sample(self.background_data, 100)
                return shap.KernelExplainer(model.predict, bg_sample)
            else:
                logger.warning("KernelExplainer requires background data")
                return None

    def explain(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> Optional[ExplanationResult]:
        """
        Compute SHAP values for data.

        Args:
            X: Data to explain
            check_additivity: Verify SHAP additivity property

        Returns:
            ExplanationResult with SHAP values
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            logger.warning("SHAP not available for explanation")
            return None

        try:
            # Get feature names
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
                X_array = X.values
            else:
                feature_names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
                X_array = X

            # Compute SHAP values
            shap_values = self._explainer.shap_values(X_array, check_additivity=check_additivity)

            # Handle multi-class output
            if isinstance(shap_values, list):
                # For classification, use positive class
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[-1]

            # Get expected value
            expected_value = self._explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[-1]

            self._shap_values = shap_values
            self._expected_value = expected_value

            # Compute summaries
            mean_abs_shap = {}
            for i, name in enumerate(feature_names):
                mean_abs_shap[name] = np.abs(shap_values[:, i]).mean()

            # Feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': [mean_abs_shap[f] for f in feature_names]
            }).sort_values('importance', ascending=False)

            result = ExplanationResult(
                shap_values=shap_values,
                base_value=float(expected_value),
                feature_names=feature_names,
                data=X_array,
                mean_abs_shap=mean_abs_shap,
                feature_importance=importance_df
            )

            logger.info(f"SHAP values computed for {len(X)} samples")
            return result

        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            return None

    def explain_prediction(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        top_n: int = 10
    ) -> Optional[PredictionExplanation]:
        """
        Explain a single prediction.

        Args:
            X: Data
            idx: Index of sample to explain
            top_n: Number of top features to include

        Returns:
            PredictionExplanation for the sample
        """
        result = self.explain(X.iloc[[idx]])

        if result is None:
            return None

        try:
            shap_vals = result.shap_values[0]

            # Build contributions dict
            contributions = {
                result.feature_names[i]: shap_vals[i]
                for i in range(len(result.feature_names))
            }

            # Sort by absolute contribution
            sorted_contribs = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Get top positive and negative
            top_positive = [
                (k, v) for k, v in sorted_contribs if v > 0
            ][:top_n]

            top_negative = [
                (k, v) for k, v in sorted_contribs if v < 0
            ][:top_n]

            # Calculate prediction
            prediction = result.base_value + sum(shap_vals)

            return PredictionExplanation(
                prediction=prediction,
                base_value=result.base_value,
                contributions=contributions,
                top_positive=top_positive,
                top_negative=top_negative
            )

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return None

    def get_feature_importance(
        self,
        X: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.

        Args:
            X: Data to compute importance on (uses cached if None)

        Returns:
            DataFrame with feature importance
        """
        if X is not None:
            result = self.explain(X)
            if result:
                return result.feature_importance

        if self._shap_values is not None:
            mean_abs = np.abs(self._shap_values).mean(axis=0)
            feature_names = self.feature_names or [f"f{i}" for i in range(len(mean_abs))]

            return pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs
            }).sort_values('importance', ascending=False)

        return pd.DataFrame()

    def get_feature_interactions(
        self,
        X: pd.DataFrame,
        max_interactions: int = 20
    ) -> pd.DataFrame:
        """
        Compute SHAP interaction values.

        Shows how features interact with each other.

        Args:
            X: Data
            max_interactions: Maximum interactions to return

        Returns:
            DataFrame of feature interactions
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            return pd.DataFrame()

        try:
            # Only tree explainer supports interactions
            if not isinstance(self._explainer, shap.TreeExplainer):
                logger.warning("Interaction values only supported for tree models")
                return pd.DataFrame()

            X_array = X.values if isinstance(X, pd.DataFrame) else X
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else self.feature_names

            # Compute interaction values (expensive)
            interaction_values = self._explainer.shap_interaction_values(X_array)

            # Handle multi-class
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1] if len(interaction_values) == 2 else interaction_values[-1]

            # Average across samples
            mean_interactions = np.abs(interaction_values).mean(axis=0)

            # Build interactions DataFrame
            interactions = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'interaction_strength': mean_interactions[i, j]
                    })

            df = pd.DataFrame(interactions)
            return df.nlargest(max_interactions, 'interaction_strength')

        except Exception as e:
            logger.error(f"Failed to compute interactions: {e}")
            return pd.DataFrame()

    def plot_summary(
        self,
        X: Optional[pd.DataFrame] = None,
        max_display: int = 20,
        plot_type: str = 'dot'
    ):
        """
        Create SHAP summary plot.

        Args:
            X: Data (uses cached if None)
            max_display: Maximum features to display
            plot_type: 'dot', 'bar', 'violin'
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for plotting")
            return

        try:
            if X is not None:
                result = self.explain(X)
                if result is None:
                    return
                shap_values = result.shap_values
                data = result.data
                feature_names = result.feature_names
            else:
                if self._shap_values is None:
                    logger.warning("No SHAP values computed")
                    return
                shap_values = self._shap_values
                data = None
                feature_names = self.feature_names

            shap.summary_plot(
                shap_values,
                features=data,
                feature_names=feature_names,
                max_display=max_display,
                plot_type=plot_type
            )

        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")

    def plot_waterfall(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        max_display: int = 15
    ):
        """
        Create waterfall plot for a single prediction.

        Args:
            X: Data
            idx: Sample index
            max_display: Maximum features to display
        """
        if not SHAP_AVAILABLE:
            return

        try:
            result = self.explain(X.iloc[[idx]])
            if result is None:
                return

            explanation = shap.Explanation(
                values=result.shap_values[0],
                base_values=result.base_value,
                data=result.data[0],
                feature_names=result.feature_names
            )

            shap.plots.waterfall(explanation, max_display=max_display)

        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")

    def plot_force(
        self,
        X: pd.DataFrame,
        idx: int = 0
    ):
        """
        Create force plot for a single prediction.

        Args:
            X: Data
            idx: Sample index
        """
        if not SHAP_AVAILABLE:
            return

        try:
            result = self.explain(X.iloc[[idx]])
            if result is None:
                return

            shap.force_plot(
                result.base_value,
                result.shap_values[0],
                X.iloc[idx],
                matplotlib=True
            )

        except Exception as e:
            logger.error(f"Failed to create force plot: {e}")

    def plot_dependence(
        self,
        feature: str,
        X: pd.DataFrame,
        interaction_feature: Optional[str] = None
    ):
        """
        Create dependence plot for a feature.

        Args:
            feature: Feature to plot
            X: Data
            interaction_feature: Feature for interaction coloring
        """
        if not SHAP_AVAILABLE:
            return

        try:
            result = self.explain(X)
            if result is None:
                return

            shap.dependence_plot(
                feature,
                result.shap_values,
                X,
                interaction_index=interaction_feature
            )

        except Exception as e:
            logger.error(f"Failed to create dependence plot: {e}")

    def generate_report(
        self,
        X: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        sample_indices: Optional[List[int]] = None,
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report.

        Args:
            X: Feature data
            predictions: Model predictions
            sample_indices: Specific samples to explain
            n_samples: Number of samples for individual explanations

        Returns:
            Report dictionary
        """
        report = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': list(X.columns),
        }

        # Global feature importance
        result = self.explain(X)
        if result:
            report['global_importance'] = result.feature_importance.to_dict('records')
            report['base_value'] = result.base_value
            report['mean_abs_shap'] = result.mean_abs_shap

            # Top features
            top_features = result.feature_importance.head(10)
            report['top_10_features'] = top_features['feature'].tolist()

        # Individual explanations
        if sample_indices is None:
            # Random samples
            sample_indices = np.random.choice(
                len(X), size=min(n_samples, len(X)), replace=False
            ).tolist()

        individual_explanations = []
        for idx in sample_indices:
            exp = self.explain_prediction(X, idx)
            if exp:
                individual_explanations.append({
                    'sample_idx': idx,
                    'prediction': exp.prediction,
                    'base_value': exp.base_value,
                    'top_positive': exp.top_positive,
                    'top_negative': exp.top_negative
                })

        report['individual_explanations'] = individual_explanations

        # Statistics
        if result:
            shap_stats = {
                'shap_mean': result.shap_values.mean(axis=0).tolist(),
                'shap_std': result.shap_values.std(axis=0).tolist(),
                'shap_min': result.shap_values.min(axis=0).tolist(),
                'shap_max': result.shap_values.max(axis=0).tolist()
            }
            report['shap_statistics'] = shap_stats

        return report

    def get_coherence_score(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate explanation coherence score.

        Measures how well SHAP explanations align with actual predictions.

        Args:
            X: Features
            y_true: True labels
            y_pred: Predictions

        Returns:
            Coherence score (0-1)
        """
        result = self.explain(X)
        if result is None:
            return 0.0

        try:
            # Calculate predicted values from SHAP
            shap_predicted = result.base_value + result.shap_values.sum(axis=1)

            # Correlation between SHAP predictions and actual predictions
            correlation = np.corrcoef(shap_predicted, y_pred)[0, 1]

            return float(max(0, correlation))

        except Exception as e:
            logger.error(f"Failed to calculate coherence: {e}")
            return 0.0


class TradingExplainer(ModelExplainer):
    """
    Specialized explainer for trading models.

    Adds trading-specific explanation features.
    """

    def explain_signal(
        self,
        X: pd.DataFrame,
        idx: int,
        signal_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Explain a trading signal decision.

        Args:
            X: Feature data
            idx: Sample index
            signal_threshold: Threshold for signal interpretation

        Returns:
            Signal explanation
        """
        exp = self.explain_prediction(X, idx)

        if exp is None:
            return {}

        # Determine signal
        if exp.prediction > signal_threshold:
            signal = "LONG"
        elif exp.prediction < -signal_threshold:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        # Categorize features
        technical_features = []
        fundamental_features = []
        microstructure_features = []
        other_features = []

        for feature, contrib in exp.contributions.items():
            feature_info = {'name': feature, 'contribution': contrib}

            if any(x in feature.lower() for x in ['rsi', 'macd', 'ema', 'sma', 'bb_', 'atr']):
                technical_features.append(feature_info)
            elif any(x in feature.lower() for x in ['spread', 'obi', 'depth', 'micro']):
                microstructure_features.append(feature_info)
            elif any(x in feature.lower() for x in ['pe', 'pb', 'eps', 'dividend']):
                fundamental_features.append(feature_info)
            else:
                other_features.append(feature_info)

        return {
            'signal': signal,
            'confidence': abs(exp.prediction),
            'prediction_value': exp.prediction,
            'base_value': exp.base_value,
            'technical_drivers': sorted(technical_features, key=lambda x: abs(x['contribution']), reverse=True)[:5],
            'microstructure_drivers': sorted(microstructure_features, key=lambda x: abs(x['contribution']), reverse=True)[:5],
            'fundamental_drivers': sorted(fundamental_features, key=lambda x: abs(x['contribution']), reverse=True)[:5],
            'top_positive_factors': exp.top_positive[:5],
            'top_negative_factors': exp.top_negative[:5]
        }

    def get_regime_explanations(
        self,
        X: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance per market regime.

        Args:
            X: Feature data
            regime_labels: Regime labels (e.g., bull/bear/sideways)

        Returns:
            Dict mapping regime to importance DataFrame
        """
        result = self.explain(X)
        if result is None:
            return {}

        regime_importance = {}
        unique_regimes = np.unique(regime_labels)

        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_shap = result.shap_values[mask]

            mean_abs = np.abs(regime_shap).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': result.feature_names,
                'importance': mean_abs
            }).sort_values('importance', ascending=False)

            regime_importance[str(regime)] = importance_df

        return regime_importance


def is_shap_available() -> bool:
    """Check if SHAP is available."""
    return SHAP_AVAILABLE
