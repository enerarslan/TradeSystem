"""
Model Explanation Module.

JPMorgan Institutional-Level Model Interpretability and Explanation.

Implements comprehensive model explanation tools:
1. SHAP (SHapley Additive exPlanations)
2. LIME (Local Interpretable Model-agnostic Explanations)
3. Permutation Feature Importance
4. Partial Dependence Plots
5. Feature Interaction Analysis
6. Decision Path Analysis

Reference:
    Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"

Model explanation is critical for:
- Regulatory compliance (MiFID II, SR 11-7)
- Risk management oversight
- Strategy validation
- Debugging unexpected behavior
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    method: str
    feature_importance: pd.DataFrame
    local_explanations: Optional[np.ndarray] = None
    interaction_values: Optional[np.ndarray] = None
    base_value: Optional[float] = None
    expected_value: Optional[float] = None


class SHAPExplainer:
    """
    SHAP-based model explanation.

    SHAP values provide theoretically grounded feature attributions
    based on cooperative game theory (Shapley values).

    Benefits:
    - Consistent: Same attribution regardless of model type
    - Local accuracy: Sum of SHAP values = prediction
    - Missingness: Features with no effect get 0 attribution
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        explainer_type: str = "auto",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained ML model
            background_data: Background data for SHAP (sample of training data)
            explainer_type: "auto", "tree", "kernel", "deep", "linear"
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap is required for SHAP explanations")

        self.model = model
        self.background_data = background_data
        self.explainer_type = explainer_type

        self.explainer_ = None
        self._setup_explainer()

    def _setup_explainer(self):
        """Set up the appropriate SHAP explainer."""
        model_type = type(self.model).__name__.lower()

        if self.explainer_type == "auto":
            # Auto-detect best explainer
            if any(x in model_type for x in ["lgbm", "xgb", "catboost", "forest", "tree"]):
                self.explainer_ = shap.TreeExplainer(self.model)
            elif "linear" in model_type or "logistic" in model_type or "ridge" in model_type:
                self.explainer_ = shap.LinearExplainer(
                    self.model, self.background_data
                )
            else:
                # Fall back to KernelExplainer (model-agnostic)
                if self.background_data is None:
                    raise ValueError("KernelExplainer requires background_data")
                self.explainer_ = shap.KernelExplainer(
                    self.model.predict if hasattr(self.model, 'predict') else self.model,
                    self.background_data
                )

        elif self.explainer_type == "tree":
            self.explainer_ = shap.TreeExplainer(self.model)

        elif self.explainer_type == "kernel":
            if self.background_data is None:
                raise ValueError("KernelExplainer requires background_data")
            predict_func = (
                self.model.predict_proba if hasattr(self.model, 'predict_proba')
                else self.model.predict
            )
            self.explainer_ = shap.KernelExplainer(predict_func, self.background_data)

        elif self.explainer_type == "linear":
            self.explainer_ = shap.LinearExplainer(self.model, self.background_data)

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ExplanationResult:
        """
        Generate SHAP explanations for predictions.

        Args:
            X: Feature matrix to explain
            feature_names: List of feature names

        Returns:
            ExplanationResult with SHAP values
        """
        if self.explainer_ is None:
            raise ValueError("Explainer not initialized")

        # Calculate SHAP values
        shap_values = self.explainer_.shap_values(X)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # For classification, take absolute mean across classes
            shap_values_combined = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values_combined = shap_values

        # Calculate feature importance (mean absolute SHAP)
        importance = np.abs(shap_values_combined).mean(axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        importance_df["importance_normalized"] = (
            importance_df["importance"] / importance_df["importance"].sum()
        )

        # Get expected value
        expected_value = (
            self.explainer_.expected_value
            if hasattr(self.explainer_, 'expected_value')
            else None
        )
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value.mean()

        return ExplanationResult(
            method="shap",
            feature_importance=importance_df,
            local_explanations=shap_values_combined,
            expected_value=expected_value,
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Explain a single prediction.

        Args:
            instance: Single feature vector (1D or 2D with 1 row)
            feature_names: List of feature names

        Returns:
            Dictionary of feature -> SHAP value
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        shap_values = self.explainer_.shap_values(instance)

        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        values = shap_values.flatten()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(values))]

        return dict(zip(feature_names, values))

    def get_interaction_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate SHAP interaction values.

        Shows how features interact with each other.

        Args:
            X: Feature matrix
            feature_names: Feature names

        Returns:
            DataFrame of interaction values
        """
        if not hasattr(self.explainer_, 'shap_interaction_values'):
            logger.warning("Interaction values not available for this explainer type")
            return pd.DataFrame()

        interaction_values = self.explainer_.shap_interaction_values(X)

        # Average across samples
        avg_interactions = np.abs(interaction_values).mean(axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(avg_interactions.shape[0])]

        return pd.DataFrame(
            avg_interactions,
            index=feature_names,
            columns=feature_names,
        )


class PermutationImportanceExplainer:
    """
    Permutation-based feature importance.

    Model-agnostic method that measures importance by
    shuffling feature values and measuring impact on predictions.

    More reliable than model's built-in feature importance
    as it accounts for feature interactions.
    """

    def __init__(
        self,
        model: Any,
        scoring: str = "accuracy",
        n_repeats: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize permutation importance explainer.

        Args:
            model: Trained model
            scoring: Scoring metric
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

    def explain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ExplanationResult:
        """
        Calculate permutation importance.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names

        Returns:
            ExplanationResult with importance values
        """
        result = permutation_importance(
            self.model,
            X,
            y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        importance = result.importances_mean
        importance_std = result.importances_std

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "importance_std": importance_std,
        }).sort_values("importance", ascending=False)

        # Normalize (importance can be negative)
        importance_df["importance_normalized"] = (
            importance_df["importance"] / importance_df["importance"].abs().sum()
        )

        return ExplanationResult(
            method="permutation_importance",
            feature_importance=importance_df,
        )


class PartialDependenceAnalyzer:
    """
    Partial Dependence Plot (PDP) analysis.

    Shows the marginal effect of features on predictions,
    averaging out the effects of other features.
    """

    def __init__(self, model: Any):
        """
        Initialize PDP analyzer.

        Args:
            model: Trained model
        """
        self.model = model

    def calculate_pdp(
        self,
        X: np.ndarray,
        feature_idx: int,
        grid_resolution: int = 50,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate partial dependence for a single feature.

        Args:
            X: Feature matrix
            feature_idx: Index of feature to analyze
            grid_resolution: Number of points in the grid
            percentile_range: Percentile range for grid

        Returns:
            Tuple of (grid_values, pdp_values)
        """
        feature_values = X[:, feature_idx]

        # Create grid
        lower = np.percentile(feature_values, percentile_range[0] * 100)
        upper = np.percentile(feature_values, percentile_range[1] * 100)
        grid = np.linspace(lower, upper, grid_resolution)

        # Calculate PDP
        pdp_values = []
        for val in grid:
            X_modified = X.copy()
            X_modified[:, feature_idx] = val

            if hasattr(self.model, 'predict_proba'):
                preds = self.model.predict_proba(X_modified)[:, 1]
            else:
                preds = self.model.predict(X_modified)

            pdp_values.append(preds.mean())

        return grid, np.array(pdp_values)

    def calculate_2d_pdp(
        self,
        X: np.ndarray,
        feature_idx_1: int,
        feature_idx_2: int,
        grid_resolution: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate 2D partial dependence (interaction effect).

        Args:
            X: Feature matrix
            feature_idx_1: First feature index
            feature_idx_2: Second feature index
            grid_resolution: Number of points per dimension

        Returns:
            Tuple of (grid_1, grid_2, pdp_values_2d)
        """
        feature_1 = X[:, feature_idx_1]
        feature_2 = X[:, feature_idx_2]

        grid_1 = np.linspace(
            np.percentile(feature_1, 5),
            np.percentile(feature_1, 95),
            grid_resolution
        )
        grid_2 = np.linspace(
            np.percentile(feature_2, 5),
            np.percentile(feature_2, 95),
            grid_resolution
        )

        pdp_2d = np.zeros((grid_resolution, grid_resolution))

        for i, val_1 in enumerate(grid_1):
            for j, val_2 in enumerate(grid_2):
                X_modified = X.copy()
                X_modified[:, feature_idx_1] = val_1
                X_modified[:, feature_idx_2] = val_2

                if hasattr(self.model, 'predict_proba'):
                    preds = self.model.predict_proba(X_modified)[:, 1]
                else:
                    preds = self.model.predict(X_modified)

                pdp_2d[i, j] = preds.mean()

        return grid_1, grid_2, pdp_2d


class DecisionPathAnalyzer:
    """
    Decision path analysis for tree-based models.

    Shows how each feature contributes to the final prediction
    along the decision path.
    """

    def __init__(self, model: Any):
        """
        Initialize decision path analyzer.

        Args:
            model: Tree-based model (RandomForest, XGBoost, etc.)
        """
        self.model = model

    def analyze_decision_path(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze decision path for a single instance.

        Args:
            instance: Single feature vector
            feature_names: Feature names

        Returns:
            Dictionary with decision path information
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        model_type = type(self.model).__name__.lower()

        if "forest" in model_type:
            return self._analyze_forest_path(instance, feature_names)
        elif "lgbm" in model_type or "xgb" in model_type:
            return self._analyze_boosting_path(instance, feature_names)
        else:
            logger.warning(f"Decision path not available for {model_type}")
            return {}

    def _analyze_forest_path(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze Random Forest decision path."""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Get decision path from first tree
        if hasattr(self.model, 'estimators_'):
            tree = self.model.estimators_[0]
            feature_indices = tree.tree_.feature
            thresholds = tree.tree_.threshold

            # Get the path taken
            path = tree.decision_path(instance).toarray()[0]
            nodes_in_path = np.where(path == 1)[0]

            decisions = []
            for node in nodes_in_path:
                if feature_indices[node] != -2:  # Not a leaf
                    feature_idx = feature_indices[node]
                    threshold = thresholds[node]
                    feature_val = instance[0, feature_idx]
                    feature_name = (
                        feature_names[feature_idx]
                        if feature_names
                        else f"feature_{feature_idx}"
                    )
                    direction = "left" if feature_val <= threshold else "right"

                    decisions.append({
                        "node": node,
                        "feature": feature_name,
                        "feature_value": feature_val,
                        "threshold": threshold,
                        "direction": direction,
                    })

            return {
                "n_nodes_in_path": len(nodes_in_path),
                "decisions": decisions,
            }

        return {}

    def _analyze_boosting_path(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze gradient boosting decision path."""
        # For boosting models, use SHAP for interpretability
        if SHAP_AVAILABLE:
            explainer = SHAPExplainer(self.model, explainer_type="tree")
            shap_dict = explainer.explain_instance(instance, feature_names)

            # Sort by absolute value
            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            return {
                "method": "shap_based",
                "feature_contributions": dict(sorted_features[:10]),
                "prediction_explanation": (
                    "Features are ranked by their contribution to this prediction"
                ),
            }

        return {}


class ModelExplainer:
    """
    Comprehensive model explanation toolkit.

    Combines multiple explanation methods for thorough
    model interpretability analysis.

    Example:
        explainer = ModelExplainer(model)
        explainer.fit(X_train, y_train)
        report = explainer.explain(X_test, feature_names=features)
    """

    def __init__(
        self,
        model: Any,
        background_sample_size: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize model explainer.

        Args:
            model: Trained ML model
            background_sample_size: Number of samples for background data
            random_state: Random seed
        """
        self.model = model
        self.background_sample_size = background_sample_size
        self.random_state = random_state

        self.background_data_ = None
        self.shap_explainer_ = None
        self.perm_explainer_ = None
        self.pdp_analyzer_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "ModelExplainer":
        """
        Fit the explainer with training data.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self
        """
        np.random.seed(self.random_state)

        # Sample background data
        n_samples = min(self.background_sample_size, len(X))
        indices = np.random.choice(len(X), size=n_samples, replace=False)
        self.background_data_ = X[indices]
        self.y_background_ = y[indices]

        # Initialize explainers
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer_ = SHAPExplainer(
                    self.model,
                    background_data=self.background_data_,
                )
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")

        self.perm_explainer_ = PermutationImportanceExplainer(
            self.model,
            random_state=self.random_state,
        )

        self.pdp_analyzer_ = PartialDependenceAnalyzer(self.model)

        return self

    def explain(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model explanations.

        Args:
            X: Features to explain
            y: True labels (optional, for permutation importance)
            feature_names: List of feature names

        Returns:
            Dictionary with all explanation results
        """
        results = {}

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # SHAP explanations
        if self.shap_explainer_ is not None:
            try:
                results["shap"] = self.shap_explainer_.explain(X, feature_names)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # Permutation importance
        if y is not None:
            try:
                results["permutation_importance"] = self.perm_explainer_.explain(
                    X, y, feature_names
                )
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")

        # Built-in feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            results["builtin_importance"] = pd.DataFrame({
                "feature": feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)

        # Summary
        results["summary"] = self._create_summary(results, feature_names)

        return results

    def explain_prediction(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            instance: Single feature vector
            feature_names: Feature names

        Returns:
            Dictionary with prediction explanation
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        result = {}

        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance)[0]
            result["prediction_proba"] = proba.tolist()
            result["predicted_class"] = int(proba.argmax())
        else:
            result["prediction"] = float(self.model.predict(instance)[0])

        # SHAP values
        if self.shap_explainer_ is not None:
            try:
                shap_values = self.shap_explainer_.explain_instance(
                    instance, feature_names
                )
                result["shap_values"] = shap_values

                # Top contributors
                sorted_shap = sorted(
                    shap_values.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                result["top_positive_contributors"] = [
                    (f, v) for f, v in sorted_shap if v > 0
                ][:5]
                result["top_negative_contributors"] = [
                    (f, v) for f, v in sorted_shap if v < 0
                ][:5]
            except Exception as e:
                logger.warning(f"SHAP instance explanation failed: {e}")

        return result

    def _create_summary(
        self,
        results: Dict[str, Any],
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Create summary of all explanations."""
        summary = {"top_features": {}}

        # Aggregate importance across methods
        importance_scores = {f: [] for f in feature_names}

        if "shap" in results:
            shap_importance = results["shap"].feature_importance
            for _, row in shap_importance.iterrows():
                importance_scores[row["feature"]].append(row["importance_normalized"])

        if "permutation_importance" in results:
            perm_importance = results["permutation_importance"].feature_importance
            for _, row in perm_importance.iterrows():
                if row["importance"] > 0:  # Only positive importance
                    max_imp = perm_importance["importance"].max()
                    if max_imp > 0:
                        importance_scores[row["feature"]].append(
                            row["importance"] / max_imp
                        )

        # Average importance across methods
        avg_importance = {
            f: np.mean(scores) if scores else 0
            for f, scores in importance_scores.items()
        }

        # Top 10 features
        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        summary["top_features"] = dict(sorted_features)
        summary["n_features"] = len(feature_names)

        return summary

    def generate_report(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a text report of model explanations.

        Args:
            X: Features to explain
            y: True labels (optional)
            feature_names: Feature names
            output_path: Path to save report (optional)

        Returns:
            Report text
        """
        results = self.explain(X, y, feature_names)

        report = []
        report.append("=" * 60)
        report.append("MODEL EXPLANATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary
        if "summary" in results:
            report.append("TOP IMPORTANT FEATURES")
            report.append("-" * 40)
            for feature, importance in results["summary"]["top_features"].items():
                report.append(f"  {feature}: {importance:.4f}")
            report.append("")

        # SHAP Results
        if "shap" in results:
            report.append("SHAP FEATURE IMPORTANCE")
            report.append("-" * 40)
            shap_df = results["shap"].feature_importance.head(10)
            for _, row in shap_df.iterrows():
                report.append(
                    f"  {row['feature']}: {row['importance']:.4f} "
                    f"({row['importance_normalized']*100:.1f}%)"
                )
            report.append("")

        # Permutation Importance
        if "permutation_importance" in results:
            report.append("PERMUTATION IMPORTANCE")
            report.append("-" * 40)
            perm_df = results["permutation_importance"].feature_importance.head(10)
            for _, row in perm_df.iterrows():
                report.append(
                    f"  {row['feature']}: {row['importance']:.4f} "
                    f"± {row['importance_std']:.4f}"
                )
            report.append("")

        report.append("=" * 60)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text
