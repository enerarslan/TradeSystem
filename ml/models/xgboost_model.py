"""
KURUMSAL XGBOOST MODEL
JPMorgan Quantitative Research Division Tarzı

XGBoost - Gradient Boosting için en popüler library.
Financial data için mükemmel performans.

Özellikler:
- Hyperparameter tuning
- Early stopping
- Feature importance
- Cross-validation
- Custom evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import time
import pickle
from pathlib import Path

# Base model import (yukarıda yazdığımız)
# from ml.models.base import BaseMLModel, PredictionResult, TrainingResult, ModelMetadata


class XGBoostPredictor:
    """
    XGBoost tabanlı fiyat/return tahmini modeli.
    
    Kullanım:
        model = XGBoostPredictor(
            objective='regression',
            n_estimators=100,
            max_depth=6
        )
        
        result = model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        model_name: str = "XGBoost_Predictor",
        objective: str = "reg:squarederror",  # reg:squarederror, binary:logistic
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,  # L1 regularization
        reg_lambda: float = 1.0,  # L2 regularization
        early_stopping_rounds: int = 10,
        random_state: int = 42
    ):
        """
        Args:
            objective: Loss function
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            early_stopping_rounds: Early stopping patience
            random_state: Random seed
        """
        self.model_name = model_name
        
        # Hyperparameters
        self.params = {
            'objective': objective,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'tree_method': 'hist',  # Fast histogram-based
            'eval_metric': 'rmse' if 'reg' in objective else 'logloss'
        }
        
        self.early_stopping_rounds = early_stopping_rounds
        
        # Model
        self.model: Optional[xgb.XGBRegressor] = None
        self.is_fitted = False
        
        # Feature tracking
        self.feature_names: List[str] = []
        self.feature_importance_: Dict[str, float] = {}
        
        # Training history
        self.training_history: List[Dict] = []
        self.best_iteration: int = 0
        self.best_score: float = 0.0
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Model training.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            verbose: Print training progress
        
        Returns:
            Training result dictionary
        """
        start_time = time.time()
        
        # Feature names kaydet
        self.feature_names = list(X_train.columns)
        
        # XGBoost model oluştur
        if 'reg' in self.params['objective']:
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)
        
        # Validation set varsa early stopping kullan
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=verbose
            )
            
            # Best iteration
            self.best_iteration = self.model.best_iteration
            self.best_score = self.model.best_score
            
            # Validation predictions
            y_val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            
        else:
            self.model.fit(X_train, y_train, verbose=verbose)
            val_metrics = {}
        
        # Training predictions
        y_train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Feature importance
        self.feature_importance_ = self._get_feature_importance()
        
        self.is_fitted = True
        
        training_time = time.time() - start_time
        
        result = {
            'success': True,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iteration': self.best_iteration,
            'feature_importance': self.feature_importance_
        }
        
        if verbose:
            self._print_training_summary(result)
        
        return result
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Prediction.
        
        Args:
            X: Features
            return_confidence: Confidence score döndür (tree variance)
        
        Returns:
            Prediction result dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Feature validation
        if set(self.feature_names) != set(X.columns):
            # Reorder columns
            X = X[self.feature_names]
        
        # Predictions
        predictions = self.model.predict(X)
        
        result = {
            'predictions': predictions,
            'feature_names': self.feature_names
        }
        
        # Confidence (prediction variance across trees)
        if return_confidence:
            # Tree-level predictions
            tree_preds = np.array([
                tree.predict(xgb.DMatrix(X))
                for tree in self.model.get_booster().get_dump()
            ])
            confidence = 1.0 / (1.0 + tree_preds.std(axis=0))
            result['confidence'] = confidence
        
        return result
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probability prediction (classification için).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if 'reg' in self.params['objective']:
            raise ValueError("predict_proba only available for classification")
        
        X = X[self.feature_names]
        return self.model.predict_proba(X)
    
    def feature_importance(
        self,
        importance_type: str = "gain"  # gain, weight, cover
    ) -> pd.DataFrame:
        """
        Feature importance DataFrame.
        
        Args:
            importance_type: Importance metric
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return df
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        shuffle: bool = False
    ) -> Dict[str, Any]:
        """
        Time-series cross-validation.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of CV folds
            shuffle: Shuffle data (False for time-series)
        
        Returns:
            CV results
        """
        # Time-series split (chronological)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Create temporary model
        temp_model = xgb.XGBRegressor(**self.params) if 'reg' in self.params['objective'] else xgb.XGBClassifier(**self.params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            temp_model,
            X,
            y,
            cv=tscv,
            scoring='neg_mean_squared_error' if 'reg' in self.params['objective'] else 'accuracy'
        )
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'n_splits': n_splits
        }
    
    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Simple grid search for hyperparameter tuning.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            param_grid: Parameter grid
        
        Returns:
            Best parameters and scores
        """
        best_score = float('inf') if 'reg' in self.params['objective'] else 0.0
        best_params = {}
        
        from itertools import product
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            # Update params
            temp_params = self.params.copy()
            temp_params.update(params)
            
            # Train model
            if 'reg' in temp_params['objective']:
                model = xgb.XGBRegressor(**temp_params)
            else:
                model = xgb.XGBClassifier(**temp_params)
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_val)
            
            if 'reg' in temp_params['objective']:
                score = np.sqrt(((y_val - y_pred) ** 2).mean())  # RMSE
                if score < best_score:
                    best_score = score
                    best_params = params
            else:
                score = (y_val == y_pred).mean()  # Accuracy
                if score > best_score:
                    best_score = score
                    best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }
    
    def save(self, path: str):
        """Model'i kaydet"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_obj = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_obj, f)
    
    def load(self, path: str) -> 'XGBoostPredictor':
        """Model'i yükle"""
        with open(path, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.model = save_obj['model']
        self.params = save_obj['params']
        self.feature_names = save_obj['feature_names']
        self.feature_importance_ = save_obj['feature_importance']
        self.best_iteration = save_obj['best_iteration']
        self.best_score = save_obj['best_score']
        self.is_fitted = save_obj['is_fitted']
        
        return self
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Feature importance dictionary"""
        importance = self.model.get_booster().get_score(importance_type='gain')
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Performance metrics"""
        metrics = {}
        
        if 'reg' in self.params['objective']:
            # Regression metrics
            mse = ((y_true - y_pred) ** 2).mean()
            metrics['mse'] = float(mse)
            metrics['rmse'] = float(np.sqrt(mse))
            metrics['mae'] = float(np.abs(y_true - y_pred).mean())
            
            # R2 score
            ss_res = ((y_true - y_pred) ** 2).sum()
            ss_tot = ((y_true - y_true.mean()) ** 2).sum()
            metrics['r2'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            # Direction accuracy (financial metric)
            direction_correct = np.sign(y_true) == np.sign(y_pred)
            metrics['direction_accuracy'] = float(direction_correct.mean())
            
        else:
            # Classification metrics
            accuracy = (y_true == y_pred).mean()
            metrics['accuracy'] = float(accuracy)
        
        return metrics
    
    def _print_training_summary(self, result: Dict[str, Any]):
        """Training summary yazdır"""
        print("\n" + "="*70)
        print("   🎯 XGBOOST TRAINING SUMMARY")
        print("="*70)
        print(f"   Model            : {self.model_name}")
        print(f"   Training Time    : {result['training_time']:.2f}s")
        print(f"   Best Iteration   : {self.best_iteration}")
        print(f"   Features         : {len(self.feature_names)}")
        
        print("\n   📊 TRAINING METRICS:")
        for metric, value in result['train_metrics'].items():
            print(f"      {metric:<20}: {value:.6f}")
        
        if result['val_metrics']:
            print("\n   📊 VALIDATION METRICS:")
            for metric, value in result['val_metrics'].items():
                print(f"      {metric:<20}: {value:.6f}")
        
        print("\n   🔝 TOP 10 FEATURES:")
        top_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"      {i:2}. {feature:<30}: {importance:.6f}")
        
        print("="*70 + "\n")


# Export
__all__ = ['XGBoostPredictor']