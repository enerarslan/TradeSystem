"""
KURUMSAL FEATURE NORMALIZATION
JPMorgan Quantitative Research Division Tarzı

Normalization Methods:
- Z-Score (Standardization)
- Min-Max Scaling
- Robust Scaling (IQR-based)
- Rolling Normalization
- Power Transform (Box-Cox, Yeo-Johnson)

Bu modül ML modellerine girmeden önce feature'ları normalize eder.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    PowerTransformer
)
import warnings

warnings.filterwarnings('ignore')


@dataclass
class NormalizationStats:
    """Normalization istatistikleri"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float


class FeatureNormalizer:
    """
    Feature normalization sınıfı.
    
    Desteklenen yöntemler:
    - zscore: (x - mean) / std
    - minmax: (x - min) / (max - min)
    - robust: (x - median) / IQR
    - rolling: Rolling window normalization
    - power: Power transform (Box-Cox, Yeo-Johnson)
    
    Kullanım:
        normalizer = FeatureNormalizer(method='zscore')
        normalizer.fit(train_df)
        normalized = normalizer.transform(test_df)
        
        # Save/Load
        normalizer.save('models/normalizer.pkl')
        normalizer.load('models/normalizer.pkl')
    """
    
    def __init__(
        self, 
        method: str = "zscore", 
        window: int = 252,
        clip_outliers: bool = True,
        clip_std: float = 3.0
    ):
        """
        Args:
            method: Normalization yöntemi
            window: Rolling window size (rolling method için)
            clip_outliers: Outlier'ları clip et
            clip_std: Clip threshold (std cinsinden)
        """
        self.method = method
        self.window = window
        self.clip_outliers = clip_outliers
        self.clip_std = clip_std
        
        # Normalization parameters
        self.params: Dict[str, NormalizationStats] = {}
        self.feature_names: List[str] = []
        
        # Sklearn scalers (fit edildikten sonra saklanır)
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler, RobustScaler]] = {}
        self.power_transformer: Optional[PowerTransformer] = None
        
        # Fitted flag
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> 'FeatureNormalizer':
        """
        Normalization parametrelerini hesapla.
        
        Args:
            df: Training DataFrame
            columns: Normalize edilecek kolonlar (None = tümü)
        
        Returns:
            self
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names = columns
        
        for col in columns:
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # İstatistikleri hesapla
            self.params[col] = NormalizationStats(
                mean=float(data.mean()),
                std=float(data.std()),
                min=float(data.min()),
                max=float(data.max()),
                median=float(data.median()),
                q25=float(data.quantile(0.25)),
                q75=float(data.quantile(0.75)),
                iqr=float(data.quantile(0.75) - data.quantile(0.25))
            )
            
            # Sklearn scaler'ları fit et (method'a göre)
            if self.method in ['zscore', 'minmax', 'robust']:
                self.scalers[col] = self._get_scaler(self.method)
                self.scalers[col].fit(data.values.reshape(-1, 1))
        
        # Power transform için
        if self.method == 'power':
            self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            self.power_transformer.fit(df[columns].fillna(0))
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalization uygula.
        
        Args:
            df: DataFrame to normalize
        
        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        result = df.copy()
        
        for col in self.feature_names:
            if col not in df.columns:
                continue
            
            if col not in self.params:
                continue
            
            if self.method == 'zscore':
                result[col] = self._zscore_transform(df[col], self.params[col])
            
            elif self.method == 'minmax':
                result[col] = self._minmax_transform(df[col], self.params[col])
            
            elif self.method == 'robust':
                result[col] = self._robust_transform(df[col], self.params[col])
            
            elif self.method == 'rolling':
                result[col] = self._rolling_transform(df[col])
            
            # Outlier clipping
            if self.clip_outliers and self.method != 'rolling':
                result[col] = self._clip_outliers(result[col])
        
        # Power transform
        if self.method == 'power' and self.power_transformer:
            result[self.feature_names] = self.power_transformer.transform(
                df[self.feature_names].fillna(0)
            )
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit ve transform birlikte"""
        return self.fit(df, columns).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizasyonu geri al.
        
        Args:
            df: Normalized DataFrame
        
        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first")
        
        result = df.copy()
        
        for col in self.feature_names:
            if col not in df.columns or col not in self.params:
                continue
            
            stats = self.params[col]
            
            if self.method == 'zscore':
                result[col] = df[col] * stats.std + stats.mean
            
            elif self.method == 'minmax':
                result[col] = df[col] * (stats.max - stats.min) + stats.min
            
            elif self.method == 'robust':
                result[col] = df[col] * stats.iqr + stats.median
        
        # Power transform inverse
        if self.method == 'power' and self.power_transformer:
            result[self.feature_names] = self.power_transformer.inverse_transform(
                df[self.feature_names]
            )
        
        return result
    
    def _zscore_transform(self, data: pd.Series, stats: NormalizationStats) -> pd.Series:
        """Z-score normalization"""
        if stats.std == 0:
            return pd.Series(0, index=data.index)
        return (data - stats.mean) / stats.std
    
    def _minmax_transform(self, data: pd.Series, stats: NormalizationStats) -> pd.Series:
        """Min-max normalization"""
        range_val = stats.max - stats.min
        if range_val == 0:
            return pd.Series(0, index=data.index)
        return (data - stats.min) / range_val
    
    def _robust_transform(self, data: pd.Series, stats: NormalizationStats) -> pd.Series:
        """Robust normalization (IQR-based)"""
        if stats.iqr == 0:
            return pd.Series(0, index=data.index)
        return (data - stats.median) / stats.iqr
    
    def _rolling_transform(self, data: pd.Series) -> pd.Series:
        """Rolling window normalization"""
        mean = data.rolling(window=self.window, min_periods=20).mean()
        std = data.rolling(window=self.window, min_periods=20).std()
        
        result = (data - mean) / std
        result = result.fillna(0).replace([np.inf, -np.inf], 0)
        
        return result
    
    def _clip_outliers(self, data: pd.Series) -> pd.Series:
        """Outlier'ları clip et"""
        lower_bound = -self.clip_std
        upper_bound = self.clip_std
        
        return data.clip(lower_bound, upper_bound)
    
    def _get_scaler(self, method: str):
        """Sklearn scaler oluştur"""
        if method == 'zscore':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler method: {method}")
    
    def save(self, path: str):
        """Normalizer'ı kaydet"""
        save_obj = {
            'method': self.method,
            'window': self.window,
            'clip_outliers': self.clip_outliers,
            'clip_std': self.clip_std,
            'params': self.params,
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'power_transformer': self.power_transformer,
            'is_fitted': self.is_fitted
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(save_obj, f)
    
    def load(self, path: str) -> 'FeatureNormalizer':
        """Normalizer'ı yükle"""
        with open(path, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.method = save_obj['method']
        self.window = save_obj['window']
        self.clip_outliers = save_obj['clip_outliers']
        self.clip_std = save_obj['clip_std']
        self.params = save_obj['params']
        self.feature_names = save_obj['feature_names']
        self.scalers = save_obj['scalers']
        self.power_transformer = save_obj['power_transformer']
        self.is_fitted = save_obj['is_fitted']
        
        return self
    
    def get_stats_summary(self) -> pd.DataFrame:
        """Normalization istatistiklerini DataFrame olarak döndür"""
        data = []
        
        for col, stats in self.params.items():
            data.append({
                'feature': col,
                'mean': stats.mean,
                'std': stats.std,
                'min': stats.min,
                'max': stats.max,
                'median': stats.median,
                'iqr': stats.iqr
            })
        
        return pd.DataFrame(data)


class MultiMethodNormalizer:
    """
    Farklı feature grupları için farklı normalization yöntemleri.
    
    Örnek:
        normalizer = MultiMethodNormalizer({
            'price_features': 'zscore',
            'volume_features': 'robust',
            'time_features': 'minmax'
        })
    """
    
    def __init__(self, method_map: Dict[str, str]):
        """
        Args:
            method_map: Feature pattern -> method mapping
        """
        self.method_map = method_map
        self.normalizers: Dict[str, FeatureNormalizer] = {}
        self.feature_groups: Dict[str, List[str]] = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'MultiMethodNormalizer':
        """Fit all normalizers"""
        # Feature gruplarını belirle
        for pattern, method in self.method_map.items():
            matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            
            if matching_cols:
                self.feature_groups[pattern] = matching_cols
                self.normalizers[pattern] = FeatureNormalizer(method=method)
                self.normalizers[pattern].fit(df, matching_cols)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform with appropriate normalizer"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        result = df.copy()
        
        for pattern, normalizer in self.normalizers.items():
            cols = self.feature_groups[pattern]
            result[cols] = normalizer.transform(df[cols])
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform"""
        return self.fit(df).transform(df)
    
    def save(self, directory: str):
        """Save all normalizers"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for pattern, normalizer in self.normalizers.items():
            safe_pattern = pattern.replace('/', '_').replace(' ', '_')
            normalizer.save(str(dir_path / f"{safe_pattern}_normalizer.pkl"))
        
        # Save metadata
        metadata = {
            'method_map': self.method_map,
            'feature_groups': self.feature_groups,
            'is_fitted': self.is_fitted
        }
        
        with open(dir_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, directory: str) -> 'MultiMethodNormalizer':
        """Load all normalizers"""
        dir_path = Path(directory)
        
        # Load metadata
        with open(dir_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.method_map = metadata['method_map']
        self.feature_groups = metadata['feature_groups']
        self.is_fitted = metadata['is_fitted']
        
        # Load normalizers
        for pattern in self.feature_groups.keys():
            safe_pattern = pattern.replace('/', '_').replace(' ', '_')
            normalizer = FeatureNormalizer()
            normalizer.load(str(dir_path / f"{safe_pattern}_normalizer.pkl"))
            self.normalizers[pattern] = normalizer
        
        return self


# Export
__all__ = [
    'FeatureNormalizer',
    'MultiMethodNormalizer',
    'NormalizationStats'
]