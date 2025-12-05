from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional
from typing import List

class MarketTick(BaseModel):
    """
    Sistemin içindeki standart veri paketi.
    Hangi borsadan gelirse gelsin, tüm veriler bu formata dönüşecek.
    """
    symbol: str          # Örn: BTC/USDT veya AAPL
    price: float         # Anlık Fiyat
    volume: float        # Hacim
    timestamp: datetime  # Zaman Damgası
    source: str          # Veri Kaynağı (Binance, Nasdaq vb.)

    class Config:
        frozen = True # Veri oluşturulduktan sonra değiştirilemez (Immutability - Güvenlik için)
# İşlem Yönü için Standart (Hata yapmayı engeller)
class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeSignal(BaseModel):
    """
    Stratejinin ürettiği emir sinyali.
    Risk motoru bu sinyali inceleyip onaylayacak veya reddedecek.
    """
    symbol: str
    side: Side           # AL, SAT veya BEKLE
    price: float         # Sinyal anındaki fiyat
    quantity: float      # Önerilen miktar (Risk motoru bunu değiştirebilir)
    strategy_name: str   # Hangi strateji bunu üretti?
    timestamp: datetime
    
    class Config:
        frozen = True
class RiskCheckResult(BaseModel):
    """
    Risk kontrolünün sonucu.
    """
    passed: bool          # İşlem onaylandı mı?
    adjusted_quantity: float # Risk yöneticisi miktarı değiştirebilir (Örn: 100 istedin, 50 al dedi)
    reason: str           # Reddedilme veya değiştirilme sebebi
    timestamp: datetime = datetime.now()

class PortfolioState(BaseModel):
    """
    Anlık cüzdan durumu.
    Risk hesaplamaları için gereklidir.
    """
    total_balance: float      # Toplam Varlık (USD)
    cash_balance: float       # Boşta Duran Nakit
    daily_pnl: float          # Günlük Kâr/Zarar
    open_positions_count: int # Açık pozisyon sayısı
    daily_trade_count: int    # Bugün kaç işlem yapıldı?