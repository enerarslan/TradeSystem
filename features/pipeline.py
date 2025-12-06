"""
KURUMSAL FEATURE ENGİNEERİNG PIPELINE
JPMorgan Quantitative Research Division Tarzı

Bu pipeline:
- Tüm feature modüllerini birleştirir
- Otomatik feature generation yapar
- Normalization & Scaling
- Feature selection
- Missing value handling
- Feature caching

ML modelleri için hazır feature matrix üretir.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from features.technical import TechnicalIndicators
from features.price_features import PriceFeatures
from features.time_features import TimeFeatures
from features.volume_features import VolumeFeatures
from utils.logger import log

warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """Feature engineering konfigürasyonu"""
    # Feature categories to include
    use_technical: bool = True
    use_price: bool = True
    use_time: bool = True
    use_volume: bool = True
    
    # Technical indicator periods
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    
    # Normalization
    normalize: bool = True
    normalization_method: str = "zscore"  # zscore, minmax, robust
    normalization_window: int = 252  # Rolling window for normalization
    
    # Missing values
    fillna_method: str = "ffill"  # ffill, bfill, mean, zero
    
    # Feature selection
    remove_constant: bool = True
    remove_correlated: bool = True
    correlation_threshold: float = 0.95
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "data/feature_cache"
    
    # Advanced
    include_lags: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    include_rolling_stats: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class FeatureStats:
    """Feature istatistikleri"""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    missing_pct: float
    unique_count: int
    is_constant: bool


class FeatureNormalizer:
    """
    Feature normalization sınıfı.
    
    Desteklenen yöntemler:
    - zscore: (x - mean) / std
    - minmax: (x - min) / (max - min)
    - robust: (x - median) / IQR
    """
    
    def __init__(self, method: str = "zscore", window: int = 252):
        self.method = method
        self.window = window
        self.params: Dict[str, Dict] = {}
    
    def fit(self, df: pd.DataFrame) -> 'FeatureNormalizer':
        """Normalization parametrelerini hesapla"""
        for col in df.columns:
            data = df[col].dropna()
            
            if self.method == "zscore":
                self.params[col] = {
                    'mean': data.mean(),
                    'std': data.std()
                }
            elif self.method == "minmax":
                self.params[col] = {
                    'min': data.min(),
                    'max': data.max()
                }
            elif self.method == "robust":
                self.params[col] = {
                    'median': data.median(),
                    'iqr': data.quantile(0.75) - data.quantile(0.25)
                }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalization uygula"""
        result = df.copy()
        
        for col in df.columns:
            if col not in self.params:
                continue
            
            params = self.params[col]
            
            if self.method == "zscore":
                if params['std'] > 0:
                    result[col] = (df[col] - params['mean']) / params['std']
                else:
                    result[col] = 0
                    
            elif self.method == "minmax":
                range_val = params['max'] - params['min']
                if range_val > 0:
                    result[col] = (df[col] - params['min']) / range_val
                else:
                    result[col] = 0
                    
            elif self.method == "robust":
                if params['iqr'] > 0:
                    result[col] = (df[col] - params['median']) / params['iqr']
                else:
                    result[col] = 0
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit ve transform birlikte"""
        return self.fit(df).transform(df)
    
    def rolling_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window normalization"""
        result = df.copy()
        
        for col in df.columns:
            if self.method == "zscore":
                mean = df[col].rolling(window=self.window, min_periods=20).mean()
                std = df[col].rolling(window=self.window, min_periods=20).std()
                result[col] = (df[col] - mean) / std
                
            elif self.method == "minmax":
                min_val = df[col].rolling(window=self.window, min_periods=20).min()
                max_val = df[col].rolling(window=self.window, min_periods=20).max()
                range_val = max_val - min_val
                result[col] = (df[col] - min_val) / range_val
                
            elif self.method == "robust":
                median = df[col].rolling(window=self.window, min_periods=20).median()
                q75 = df[col].rolling(window=self.window, min_periods=20).quantile(0.75)
                q25 = df[col].rolling(window=self.window, min_periods=20).quantile(0.25)
                iqr = q75 - q25
                result[col] = (df[col] - median) / iqr
        
        return result.fillna(0).replace([np.inf, -np.inf], 0)
    
    def save(self, path: str):
        """Parametreleri kaydet"""
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load(self, path: str):
        """Parametreleri yükle"""
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
        return self


class FeatureEngineeringPipeline:
    """
    Ana Feature Engineering Pipeline.
    
    Kullanım:
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.fit_transform(df)
        
    Veya:
        features = pipeline.generate_features(df)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Args:
            config: Feature engineering konfigürasyonu
        """
        self.config = config or FeatureConfig()
        
        # Feature generators
        self.technical = TechnicalIndicators(fillna=True)
        self.price_features = PriceFeatures(fillna=True)
        self.time_features = TimeFeatures(fillna=True)
        self.volume_features = VolumeFeatures(fillna=True)
        
        # Normalizer
        self.normalizer = FeatureNormalizer(
            method=self.config.normalization_method,
            window=self.config.normalization_window
        )
        
        # Feature metadata
        self.feature_names: List[str] = []
        self.feature_stats: Dict[str, FeatureStats] = {}
        self.removed_features: List[str] = []
        
        # Cache
        self.cache_dir = Path(self.config.cache_dir)
        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("🔧 Feature Engineering Pipeline başlatıldı")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Tüm özellikleri üret.
        
        Args:
            df: OHLCV DataFrame
            symbol: Sembol adı (cache key için)
            use_cache: Cache kullan
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        log.info(f"📊 Feature generation başlıyor: {symbol}")
        start_time = datetime.now()
        
        # Cache check
        if use_cache and self.config.use_cache:
            cache_key = self._get_cache_key(df, symbol)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                log.info(f"💾 Cache'den yüklendi: {len(cached.columns)} feature")
                return cached
        
        # Initialize result
        result = pd.DataFrame(index=df.index)
        
        # Add original OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                result[col] = df[col]
        
        # Generate features by category
        if self.config.use_technical:
            log.debug("📈 Technical indicators hesaplanıyor...")
            tech_features = self.technical.calculate_all(df)
            result = pd.concat([result, tech_features], axis=1)
        
        if self.config.use_price:
            log.debug("💰 Price features hesaplanıyor...")
            price_feat = self.price_features.calculate_all(df)
            result = pd.concat([result, price_feat], axis=1)
        
        if self.config.use_volume:
            log.debug("📊 Volume features hesaplanıyor...")
            vol_feat = self.volume_features.calculate_all(df)
            result = pd.concat([result, vol_feat], axis=1)
        
        if self.config.use_time:
            log.debug("⏰ Time features hesaplanıyor...")
            time_feat = self.time_features.calculate_all(df)
            result = pd.concat([result, time_feat], axis=1)
        
        # Add lag features
        if self.config.include_lags:
            log.debug("⏮️ Lag features ekleniyor...")
            lag_features = self._generate_lag_features(result)
            result = pd.concat([result, lag_features], axis=1)
        
        # Add rolling statistics
        if self.config.include_rolling_stats:
            log.debug("📉 Rolling statistics ekleniyor...")
            rolling_features = self._generate_rolling_stats(result)
            result = pd.concat([result, rolling_features], axis=1)
        
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        # Handle missing values
        result = self._handle_missing_values(result)
        
        # Remove constant features
        if self.config.remove_constant:
            result = self._remove_constant_features(result)
        
        # Remove highly correlated features
        if self.config.remove_correlated:
            result = self._remove_correlated_features(result)
        
        # Store feature names
        self.feature_names = list(result.columns)
        
        # Calculate feature statistics
        self._calculate_feature_stats(result)
        
        # Cache
        if use_cache and self.config.use_cache:
            self._save_to_cache(result, cache_key)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log.success(f"✅ Feature generation tamamlandı: {len(result.columns)} feature ({elapsed:.2f}s)")
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Feature üret ve normalize et.
        
        Args:
            df: OHLCV DataFrame
            symbol: Sembol adı
            normalize: Normalization uygula
        
        Returns:
            pd.DataFrame: Normalize edilmiş feature matrix
        """
        # Generate features
        features = self.generate_features(df, symbol)
        
        # Normalize
        if normalize and self.config.normalize:
            log.debug("🔄 Normalization uygulanıyor...")
            features = self.normalizer.rolling_normalize(features)
        
        return features
    
    def transform(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Daha önce fit edilmiş pipeline ile transform.
        """
        return self.fit_transform(df, symbol)
    
    def _generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag feature'ları üret"""
        result = pd.DataFrame(index=df.index)
        
        # Key features for lagging
        key_features = ['close', 'volume', 'RSI_14', 'MACD', 'ATR_14', 'Return_1']
        available_features = [f for f in key_features if f in df.columns]
        
        for feature in available_features:
            for lag in self.config.lag_periods:
                result[f"{feature}_Lag{lag}"] = df[feature].shift(lag)
        
        return result.fillna(0)
    
    def _generate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistics üret"""
        result = pd.DataFrame(index=df.index)
        
        # Key features for rolling stats
        key_features = ['close', 'Return_1', 'volume']
        available_features = [f for f in key_features if f in df.columns]
        
        for feature in available_features:
            for window in self.config.rolling_windows:
                # Rolling mean
                result[f"{feature}_RollMean{window}"] = df[feature].rolling(window=window).mean()
                # Rolling std
                result[f"{feature}_RollStd{window}"] = df[feature].rolling(window=window).std()
                # Rolling min
                result[f"{feature}_RollMin{window}"] = df[feature].rolling(window=window).min()
                # Rolling max
                result[f"{feature}_RollMax{window}"] = df[feature].rolling(window=window).max()
        
        return result.fillna(0)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Missing value handling"""
        result = df.copy()
        
        if self.config.fillna_method == "ffill":
            result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        elif self.config.fillna_method == "bfill":
            result = result.fillna(method='bfill').fillna(method='ffill').fillna(0)
        elif self.config.fillna_method == "mean":
            result = result.fillna(result.mean()).fillna(0)
        elif self.config.fillna_method == "zero":
            result = result.fillna(0)
        
        # Handle inf
        result = result.replace([np.inf, -np.inf], 0)
        
        return result
    
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sabit feature'ları kaldır"""
        constant_cols = []
        
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            log.debug(f"🗑️ {len(constant_cols)} constant feature kaldırıldı")
            self.removed_features.extend(constant_cols)
            df = df.drop(columns=constant_cols)
        
        return df
    
    def _remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yüksek korelasyonlu feature'ları kaldır"""
        # Numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return df
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns to drop
        to_drop = [column for column in upper.columns if any(upper[column] > self.config.correlation_threshold)]
        
        if to_drop:
            log.debug(f"🗑️ {len(to_drop)} correlated feature kaldırıldı (threshold: {self.config.correlation_threshold})")
            self.removed_features.extend(to_drop)
            df = df.drop(columns=to_drop)
        
        return df
    
    def _calculate_feature_stats(self, df: pd.DataFrame):
        """Feature istatistiklerini hesapla"""
        for col in df.columns:
            data = df[col]
            
            self.feature_stats[col] = FeatureStats(
                name=col,
                mean=float(data.mean()),
                std=float(data.std()),
                min_val=float(data.min()),
                max_val=float(data.max()),
                missing_pct=float(data.isnull().sum() / len(data) * 100),
                unique_count=int(data.nunique()),
                is_constant=data.nunique() <= 1
            )
    
    def _get_cache_key(self, df: pd.DataFrame, symbol: str) -> str:
        """Cache key oluştur"""
        # DataFrame hash
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df.head(100)).values.tobytes()
        ).hexdigest()[:8]
        
        # Config hash
        config_str = str(self.config.__dict__)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{symbol}_{df_hash}_{config_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Cache'den yükle"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_key: str):
        """Cache'e kaydet"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            log.debug(f"💾 Cache kaydedildi: {cache_key}")
        except Exception as e:
            log.warning(f"Cache kayıt hatası: {e}")
    
    def get_feature_importance_proxy(self, df: pd.DataFrame, target_col: str = "Return_1") -> pd.DataFrame:
        """
        Feature importance tahmini (basit korelasyon bazlı).
        
        Gerçek importance için ML model training gerekir.
        """
        if target_col not in df.columns:
            log.warning(f"Target column not found: {target_col}")
            return pd.DataFrame()
        
        target = df[target_col]
        correlations = {}
        
        for col in df.columns:
            if col == target_col:
                continue
            try:
                corr = df[col].corr(target)
                correlations[col] = abs(corr)
            except:
                correlations[col] = 0
        
        importance_df = pd.DataFrame({
            'feature': list(correlations.keys()),
            'importance': list(correlations.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Feature özet tablosu"""
        if not self.feature_stats:
            return pd.DataFrame()
        
        data = []
        for name, stats in self.feature_stats.items():
            data.append({
                'Feature': name,
                'Mean': f"{stats.mean:.4f}",
                'Std': f"{stats.std:.4f}",
                'Min': f"{stats.min_val:.4f}",
                'Max': f"{stats.max_val:.4f}",
                'Missing%': f"{stats.missing_pct:.2f}%",
                'Unique': stats.unique_count
            })
        
        return pd.DataFrame(data)
    
    def print_feature_report(self):
        """Feature raporu yazdır"""
        print("\n" + "="*80)
        print("   📊 FEATURE ENGINEERING RAPORU")
        print("="*80)
        print(f"   Toplam Feature    : {len(self.feature_names)}")
        print(f"   Technical         : {sum(1 for f in self.feature_names if 'SMA' in f or 'EMA' in f or 'RSI' in f or 'MACD' in f)}")
        print(f"   Price Features    : {sum(1 for f in self.feature_names if 'Return' in f or 'Momentum' in f)}")
        print(f"   Volume Features   : {sum(1 for f in self.feature_names if 'Vol' in f or 'OBV' in f)}")
        print(f"   Time Features     : {sum(1 for f in self.feature_names if 'Hour' in f or 'Day' in f or 'Month' in f)}")
        print(f"   Removed Features  : {len(self.removed_features)}")
        print("="*80)
        
        if self.removed_features:
            print(f"   Kaldırılan: {', '.join(self.removed_features[:10])}{'...' if len(self.removed_features) > 10 else ''}")
        
        print("="*80 + "\n")
    
    def save_pipeline(self, path: str):
        """Pipeline'ı kaydet"""
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'normalizer_params': self.normalizer.params,
                'feature_names': self.feature_names,
                'feature_stats': self.feature_stats
            }, f)
        log.info(f"💾 Pipeline kaydedildi: {path}")
    
    def load_pipeline(self, path: str) -> 'FeatureEngineeringPipeline':
        """Pipeline'ı yükle"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.config = data['config']
        self.normalizer.params = data['normalizer_params']
        self.feature_names = data['feature_names']
        self.feature_stats = data['feature_stats']
        
        log.info(f"📂 Pipeline yüklendi: {path}")
        return self


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_features(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    normalize: bool = True,
    config: Optional[FeatureConfig] = None
) -> pd.DataFrame:
    """
    Hızlı feature generation.
    
    Args:
        df: OHLCV DataFrame
        symbol: Sembol adı
        normalize: Normalization uygula
        config: Feature config
    
    Returns:
        pd.DataFrame: Feature matrix
    """
    pipeline = FeatureEngineeringPipeline(config)
    return pipeline.fit_transform(df, symbol, normalize)


def get_default_config() -> FeatureConfig:
    """Varsayılan config döndür"""
    return FeatureConfig()


# Export
__all__ = [
    'FeatureEngineeringPipeline',
    'FeatureConfig',
    'FeatureNormalizer',
    'FeatureStats',
    'create_features',
    'get_default_config'
]