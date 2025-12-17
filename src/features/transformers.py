"""
Feature Transformers for Preprocessing.

Includes:
- Cyclical Time Encoding (Sin/Cos transformation)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TimeCyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes time features (hour, day, month) into cyclical sin/cos components.
    This preserves the cyclical nature of time (e.g. hr 23 is close to hr 0).
    """
    
    def __init__(self, time_col: str = 'timestamp'):
        self.time_col = time_col
        self.periods = {
            'hour': 24.0,
            'minute': 60.0,
            'day_of_week': 7.0,
            'month': 12.0
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Avoid modifying original dataframe
        X_encoded = X.copy()
        
        # Ensure we have datetime accessor
        if self.time_col in X_encoded.columns:
            if not pd.api.types.is_datetime64_any_dtype(X_encoded[self.time_col]):
                dt_series = pd.to_datetime(X_encoded[self.time_col])
            else:
                dt_series = X_encoded[self.time_col].dt
        elif isinstance(X_encoded.index, pd.DatetimeIndex):
            dt_series = X_encoded.index
        else:
            # If no time column found, return as is (or raise error)
            return X_encoded

        # Encode available components
        # Hour
        if hasattr(dt_series, 'hour'):
            X_encoded['hour_sin'] = np.sin(2 * np.pi * dt_series.hour / self.periods['hour'])
            X_encoded['hour_cos'] = np.cos(2 * np.pi * dt_series.hour / self.periods['hour'])
            
        # Day of Week
        if hasattr(dt_series, 'dayofweek'):
            X_encoded['dow_sin'] = np.sin(2 * np.pi * dt_series.dayofweek / self.periods['day_of_week'])
            X_encoded['dow_cos'] = np.cos(2 * np.pi * dt_series.dayofweek / self.periods['day_of_week'])
            
        return X_encoded