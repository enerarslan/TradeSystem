import pandas as pd
from data.models import MarketTick
from datetime import datetime
from utils.logger import log
import os

class LocalCSVLoader:
    """
    Yerel diskteki (data/storage) CSV dosyalarını okur.
    Senin Polygon verilerine göre özel ayarlandı.
    """
    def __init__(self, storage_path="data/storage"):
        # Kodun çalıştığı yerden data/storage yolunu bul
        self.storage_path = os.path.join(os.getcwd(), storage_path)

    def load_data(self, symbol: str) -> list[MarketTick]:
        # Dosya ismini oluştur: AAPL -> AAPL_15min.csv
        file_name = f"{symbol}_15min.csv"
        file_path = os.path.join(self.storage_path, file_name)
        
        if not os.path.exists(file_path):
            log.error(f"Dosya BULUNAMADI: {file_path}")
            log.warning("Lütfen CSV dosyasını 'data/storage' klasörüne attığından emin ol.")
            return []

        log.info(f"Geçmiş Veri Yükleniyor: {file_name} ...")
        
        try:
            # CSV'yi oku
            df = pd.read_csv(file_path)
            
            # Tarih formatını düzelt (Senin formatın: 2021-01-04 09:00:00)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Veriyi hızla listeye çevir
            ticks = []
            for _, row in df.iterrows():
                tick = MarketTick(
                    symbol=symbol,
                    price=float(row['close']),  # Kapanış fiyatını alıyoruz
                    volume=float(row['volume']),
                    timestamp=row['timestamp'],
                    source="POLYGON_CSV"
                )
                ticks.append(tick)
            
            log.success(f"BAŞARILI: {len(ticks)} adet mum verisi yüklendi.")
            return ticks

        except Exception as e:
            log.critical(f"CSV Okuma Hatası: {e}")
            return []