from strategies.base import BaseStrategy
from data.models import MarketTick, TradeSignal, Side
from datetime import datetime
from collections import deque
import numpy as np
from utils.logger import log

class SimpleMomentum(BaseStrategy):
    def __init__(self, symbol: str, window_size: int = 20):
        super().__init__(name="SimpleMomentum_V1")
        self.symbol = symbol
        self.history = deque(maxlen=window_size) # Son 20 fiyatı hafızada tut
        self.window_size = window_size
        self.last_side = Side.HOLD # Son durumu hatırla (Sürekli al emri göndermemek için)

    async def on_tick(self, tick: MarketTick) -> TradeSignal:
        # Sadece hedeflediğimiz sembolle ilgilen
        if tick.symbol != self.symbol:
            return None

        # Fiyatı hafızaya kaydet
        self.history.append(tick.price)

        # Yeterli veri yoksa bekle (Cold Start)
        if len(self.history) < self.window_size:
            return None

        # Ortalamayı hesapla
        avg_price = np.mean(self.history)
        
        # Karar Mekanizması
        signal_side = Side.HOLD

        # Fiyat ortalamanın %0.01 üstündeyse ve elimizde yoksa -> AL
        if tick.price > avg_price * 1.0001: 
            signal_side = Side.BUY
        
        # Fiyat ortalamanın %0.01 altındaysa ve elimizde varsa -> SAT
        elif tick.price < avg_price * 0.9999:
            signal_side = Side.SELL

        # Eğer karar değişmediyse (Zaten AL modundaysak tekrar AL deme) sinyal üretme
        if signal_side == self.last_side or signal_side == Side.HOLD:
            return None
        
        self.last_side = signal_side # Durumu güncelle

        # Sinyal Paketini Oluştur
        log.info(f"STRATEJİ TETİKLENDİ: {self.name} -> {signal_side}")
        
        return TradeSignal(
            symbol=tick.symbol,
            side=signal_side,
            price=tick.price,
            quantity=0.001, # Şimdilik sabit, sonra Risk Yönetimi belirleyecek
            strategy_name=self.name,
            timestamp=datetime.now()
        )