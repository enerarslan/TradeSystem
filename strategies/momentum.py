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
        self.history = deque(maxlen=window_size)
        self.window_size = window_size
        # self.last_side = Side.HOLD  <-- BU KALDIRILDI (Devlet tutmayı PortfolioManager yapmalı)

    async def on_tick(self, tick: MarketTick) -> TradeSignal:
        if tick.symbol != self.symbol:
            return None

        self.history.append(tick.price)

        if len(self.history) < self.window_size:
            return None

        avg_price = np.mean(self.history)
        signal_side = Side.HOLD

        # Eşik değerleri biraz daha genişletildi ki işlem yapsın
        if tick.price > avg_price * 1.0005: 
            signal_side = Side.BUY
        elif tick.price < avg_price * 0.9995:
            signal_side = Side.SELL

        if signal_side == Side.HOLD:
            return None
        
        # Log kirliliğini önlemek için buraya print koymuyoruz, main.py halledecek.
        
        return TradeSignal(
            symbol=tick.symbol,
            side=signal_side,
            price=tick.price,
            quantity=1, # Adet olarak 1 (Hisse senedi mantığı)
            strategy_name=self.name,
            timestamp=datetime.now()
        )