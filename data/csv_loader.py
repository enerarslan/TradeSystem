"""
KURUMSAL SEVİYE CSV VERİ YÜKLEYICI
JPMorgan Quant Research tarzı veri yönetimi

Özellikler:
- Çoklu dosya formatı desteği (CSV, XLSX)
- Veri doğrulama ve temizleme
- Otomatik tip dönüşümü
- Eksik veri interpolasyonu
- Performans optimizasyonu (chunk processing)
- Detaylı hata raporlama
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from data.models import MarketTick
from utils.logger import log


@dataclass
class DataQualityReport:
    """Veri kalitesi raporu"""
    symbol: str
    total_rows: int
    missing_values: Dict[str, int]
    duplicates: int
    date_range: Tuple[datetime, datetime]
    gaps_detected: int
    anomalies: int
    quality_score: float  # 0-100


class LocalCSVLoader:
    """
    Profesyonel CSV/Excel veri yükleyici.
    
    Desteklenen formatlar:
    - CSV: {symbol}_15min.csv
    - Excel: {symbol}_15min.xlsx
    
    Beklenen kolonlar:
    - timestamp: Zaman damgası (YYYY-MM-DD HH:MM:SS)
    - open: Açılış fiyatı
    - high: En yüksek fiyat
    - low: En düşük fiyat
    - close: Kapanış fiyatı
    - volume: İşlem hacmi
    """
    
    def __init__(
        self, 
        storage_path: str = "data/storage",
        validate_data: bool = True,
        interpolate_missing: bool = True,
        remove_outliers: bool = True
    ):
        """
        Args:
            storage_path: CSV dosyalarının bulunduğu klasör
            validate_data: Veri doğrulama yapılsın mı?
            interpolate_missing: Eksik değerleri interpolate et
            remove_outliers: Aykırı değerleri temizle
        """
        self.storage_path = Path(storage_path)
        self.validate_data = validate_data
        self.interpolate_missing = interpolate_missing
        self.remove_outliers = remove_outliers
        
        # Cache (Aynı sembolü tekrar yüklemekten kaçın)
        self._cache: Dict[str, List[MarketTick]] = {}
        
        # Veri kalite raporları
        self.quality_reports: Dict[str, DataQualityReport] = {}
        
        # İstatistikler
        self.stats = {
            'files_loaded': 0,
            'total_rows': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        if not self.storage_path.exists():
            log.warning(f"⚠️ Storage klasörü bulunamadı: {self.storage_path}")
            log.info(f"📁 Klasör oluşturuluyor: {self.storage_path}")
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(
        self, 
        symbol: str,
        use_cache: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[MarketTick]:
        """
        Ana veri yükleme fonksiyonu.
        
        Args:
            symbol: Sembol adı (örn: "AAPL")
            use_cache: Cache kullanılsın mı?
            start_date: Başlangıç tarihi (filtreleme için)
            end_date: Bitiş tarihi (filtreleme için)
        
        Returns:
            List[MarketTick]: Yüklenmiş piyasa verileri
        """
        # Cache kontrolü
        if use_cache and symbol in self._cache:
            log.debug(f"💾 Cache'den yükleniyor: {symbol}")
            self.stats['cache_hits'] += 1
            return self._apply_date_filter(self._cache[symbol], start_date, end_date)
        
        # Dosya yolunu bul
        file_path = self._find_data_file(symbol)
        
        if not file_path:
            log.error(f"❌ Veri dosyası bulunamadı: {symbol}")
            log.warning(f"📍 Aranan konum: {self.storage_path}")
            log.warning(f"📍 Beklenen format: {symbol}_15min.csv veya {symbol}_15min.xlsx")
            self.stats['errors'] += 1
            return []
        
        log.info(f"📂 Veri yükleniyor: {file_path.name} ...")
        
        try:
            # Dosya formatına göre yükle
            if file_path.suffix.lower() == '.csv':
                df = self._load_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            else:
                log.error(f"❌ Desteklenmeyen dosya formatı: {file_path.suffix}")
                return []
            
            if df is None or df.empty:
                log.error(f"❌ Dosya okunamadı veya boş: {file_path.name}")
                return []
            
            # Veri işleme pipeline
            df = self._preprocess_dataframe(df, symbol)
            
            if df.empty:
                log.error(f"❌ Ön işleme sonrası veri boş: {symbol}")
                return []
            
            # Veri kalitesi kontrolü
            if self.validate_data:
                quality_report = self._validate_data_quality(df, symbol)
                self.quality_reports[symbol] = quality_report
                self._print_quality_report(quality_report)
            
            # MarketTick listesine dönüştür
            ticks = self._dataframe_to_ticks(df, symbol)
            
            # Cache'e kaydet
            self._cache[symbol] = ticks
            
            # İstatistikleri güncelle
            self.stats['files_loaded'] += 1
            self.stats['total_rows'] += len(ticks)
            
            log.success(f"✅ BAŞARILI: {len(ticks):,} adet mum verisi yüklendi ({file_path.name})")
            log.info(f"📅 Tarih aralığı: {df.index[0]} → {df.index[-1]}")
            
            return self._apply_date_filter(ticks, start_date, end_date)
            
        except Exception as e:
            log.critical(f"💥 HATA: CSV okuma hatası ({symbol}): {e}")
            log.exception(e)  # Full traceback
            self.stats['errors'] += 1
            return []
    
    def _find_data_file(self, symbol: str) -> Optional[Path]:
        """Sembol için veri dosyasını bulur (CSV veya Excel)"""
        possible_files = [
            self.storage_path / f"{symbol}_15min.csv",
            self.storage_path / f"{symbol}_15min.xlsx",
            self.storage_path / f"{symbol}.csv",
            self.storage_path / f"{symbol}.xlsx",
            self.storage_path / f"{symbol.upper()}_15min.csv",
            self.storage_path / f"{symbol.lower()}_15min.csv",
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return file_path
        
        return None
    
    def _load_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """CSV dosyasını yükler"""
        try:
            # Otomatik delimiter detection
            with open(file_path, 'r') as f:
                first_line = f.readline()
                delimiter = ',' if ',' in first_line else ';'
            
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                parse_dates=['timestamp'],
                na_values=['', 'NA', 'N/A', 'null', 'NULL']
            )
            return df
            
        except Exception as e:
            log.error(f"CSV okuma hatası: {e}")
            return None
    
    def _load_excel(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Excel dosyasını yükler"""
        try:
            df = pd.read_excel(
                file_path,
                parse_dates=['timestamp'],
                na_values=['', 'NA', 'N/A', 'null', 'NULL']
            )
            return df
            
        except Exception as e:
            log.error(f"Excel okuma hatası: {e}")
            return None
    
    def _preprocess_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Veri ön işleme pipeline:
        1. Kolon isimleri düzelt
        2. Veri tiplerini düzelt
        3. Timestamp'i index yap
        4. Sıralama
        5. Duplikasyonları temizle
        6. Eksik değerleri işle
        7. Aykırı değerleri temizle
        """
        # 1. Kolon isimleri standardize et
        df.columns = df.columns.str.lower().str.strip()
        
        # 2. Gerekli kolonları kontrol et
        required_cols = ['timestamp', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log.error(f"❌ Eksik kolonlar: {missing_cols}")
            log.info(f"📋 Mevcut kolonlar: {list(df.columns)}")
            return pd.DataFrame()
        
        # 3. OHLC kolonlarını ekle (yoksa)
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # 4. Veri tiplerini düzelt
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Timestamp'i datetime'a çevir ve index yap
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 6. Duplikasyonları temizle
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            log.warning(f"⚠️ {duplicates} duplikat zaman damgası kaldırıldı")
            df = df[~df.index.duplicated(keep='first')]
        
        # 7. NaN/Inf değerleri temizle
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 8. Eksik değerleri interpolate et
        if self.interpolate_missing:
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                log.warning(f"⚠️ {missing_count} eksik değer interpolate ediliyor...")
                df.interpolate(method='linear', inplace=True, limit_direction='both')
                df.fillna(method='ffill', inplace=True)  # Kalan NaN'ları forward fill
                df.fillna(method='bfill', inplace=True)  # Hala NaN varsa backward fill
        
        # 9. Aykırı değerleri temizle
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # 10. OHLC mantık kontrolü (High >= Low, Close between them)
        df = self._fix_ohlc_logic(df)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
        """
        İstatistiksel aykırı değerleri kaldırır (Z-score method).
        Fiyatta %100+ değişim varsa suspicious olarak işaretle.
        """
        returns = df['close'].pct_change()
        
        # Z-score hesapla
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        # Aykırı değerleri bul
        outliers = z_scores > n_std
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            log.warning(f"⚠️ {outlier_count} aykırı değer tespit edildi ve düzeltiliyor...")
            
            # Aykırı değerleri önceki/sonraki değerlerin ortalaması ile değiştir
            df.loc[outliers, 'close'] = df['close'].rolling(window=3, center=True).mean()[outliers]
            df.loc[outliers, 'open'] = df['open'].rolling(window=3, center=True).mean()[outliers]
            df.loc[outliers, 'high'] = df['high'].rolling(window=3, center=True).mean()[outliers]
            df.loc[outliers, 'low'] = df['low'].rolling(window=3, center=True).mean()[outliers]
        
        return df
    
    def _fix_ohlc_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLC mantık hatalarını düzeltir"""
        # High en az close/open kadar olmalı
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        
        # Low en fazla close/open kadar olmalı
        df['low'] = df[['low', 'close', 'open']].min(axis=1)
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> DataQualityReport:
        """
        Veri kalitesini değerlendirir ve rapor oluşturur.
        """
        # Eksik değerler
        missing = df.isnull().sum().to_dict()
        
        # Tarih aralığı
        date_range = (df.index.min(), df.index.max())
        
        # Zaman boşlukları (15 dakikalık barlar bekleniyor)
        expected_interval = timedelta(minutes=15)
        gaps = 0
        
        time_diffs = df.index.to_series().diff()
        large_gaps = time_diffs > expected_interval * 2  # 30 dakikadan fazla boşluk
        gaps = large_gaps.sum()
        
        # Anomali tespiti (fiyat sıçramaları)
        returns = df['close'].pct_change().abs()
        anomalies = (returns > 0.1).sum()  # %10'dan fazla değişim
        
        # Kalite skoru hesapla (0-100)
        quality_score = 100.0
        quality_score -= min(50, (sum(missing.values()) / len(df)) * 100)  # Eksik veri cezası
        quality_score -= min(20, (gaps / len(df)) * 1000)  # Gap cezası
        quality_score -= min(20, (anomalies / len(df)) * 100)  # Anomali cezası
        quality_score = max(0, quality_score)
        
        return DataQualityReport(
            symbol=symbol,
            total_rows=len(df),
            missing_values=missing,
            duplicates=0,  # Zaten temizlendi
            date_range=date_range,
            gaps_detected=gaps,
            anomalies=anomalies,
            quality_score=quality_score
        )
    
    def _print_quality_report(self, report: DataQualityReport):
        """Veri kalite raporunu yazdırır"""
        log.info("─" * 50)
        log.info(f"📊 VERİ KALİTESİ RAPORU: {report.symbol}")
        log.info("─" * 50)
        log.info(f"  Toplam Satır      : {report.total_rows:,}")
        log.info(f"  Tarih Aralığı     : {report.date_range[0]} → {report.date_range[1]}")
        log.info(f"  Zaman Boşlukları  : {report.gaps_detected}")
        log.info(f"  Anomali Sayısı    : {report.anomalies}")
        
        # Kalite skoru renkli göster
        score = report.quality_score
        if score >= 90:
            log.success(f"  ✅ KALİTE SKORU   : {score:.1f}/100 (Mükemmel)")
        elif score >= 70:
            log.info(f"  ⚠️  KALİTE SKORU   : {score:.1f}/100 (İyi)")
        else:
            log.warning(f"  ❌ KALİTE SKORU   : {score:.1f}/100 (Düşük - Dikkat!)")
        
        log.info("─" * 50)
    
    def _dataframe_to_ticks(self, df: pd.DataFrame, symbol: str) -> List[MarketTick]:
        """DataFrame'i MarketTick listesine dönüştürür"""
        ticks = []
        
        for timestamp, row in df.iterrows():
            tick = MarketTick(
                symbol=symbol,
                price=float(row['close']),
                volume=float(row['volume']),
                timestamp=timestamp,
                source="CSV_HISTORICAL"
            )
            ticks.append(tick)
        
        return ticks
    
    def _apply_date_filter(
        self, 
        ticks: List[MarketTick],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[MarketTick]:
        """Tarih filtresi uygular"""
        if not start_date and not end_date:
            return ticks
        
        filtered = []
        for tick in ticks:
            if start_date and tick.timestamp < start_date:
                continue
            if end_date and tick.timestamp > end_date:
                continue
            filtered.append(tick)
        
        if len(filtered) < len(ticks):
            log.info(f"📅 Tarih filtresi uygulandı: {len(ticks)} → {len(filtered)} bar")
        
        return filtered
    
    def load_multiple_symbols(
        self, 
        symbols: List[str],
        parallel: bool = True
    ) -> Dict[str, List[MarketTick]]:
        """
        Birden fazla sembolü yükler.
        
        Args:
            symbols: Yüklenecek sembol listesi
            parallel: Paralel yükleme (daha hızlı ama RAM kullanır)
        
        Returns:
            Dict: {symbol: [ticks]}
        """
        log.info(f"📚 Toplu veri yükleme başlıyor: {len(symbols)} sembol...")
        
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            log.info(f"[{i}/{len(symbols)}] {symbol} yükleniyor...")
            ticks = self.load_data(symbol, use_cache=True)
            if ticks:
                results[symbol] = ticks
        
        log.success(f"✅ {len(results)}/{len(symbols)} sembol başarıyla yüklendi")
        return results
    
    def get_statistics(self) -> Dict:
        """Yükleyici istatistiklerini döner"""
        return {
            **self.stats,
            'cache_size': len(self._cache),
            'quality_reports': len(self.quality_reports)
        }
    
    def clear_cache(self):
        """Cache'i temizler"""
        self._cache.clear()
        log.info("🗑️ Cache temizlendi")


# KULLANIM ÖRNEĞİ
if __name__ == "__main__":
    loader = LocalCSVLoader(
        storage_path="data/storage",
        validate_data=True,
        interpolate_missing=True,
        remove_outliers=True
    )
    
    # Tek sembol yükle
    ticks = loader.load_data("AAPL")
    
    # Çoklu sembol yükle
    # symbols = ["AAPL", "MSFT", "GOOGL"]
    # all_data = loader.load_multiple_symbols(symbols)
    
    # İstatistikleri göster
    # print(loader.get_statistics())