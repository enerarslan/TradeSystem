from sqlalchemy import Column, Integer, String, Float, DateTime
from data.db import Base
from datetime import datetime

class TradeRecord(Base):
    """
    Gerçekleşen her işlemin 'Kara Kutu' kaydı.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String)
    side = Column(String)      # BUY / SELL
    price = Column(Float)      # Gerçekleşen Fiyat
    quantity = Column(Float)   # Adet
    strategy = Column(String)  # Hangi strateji yaptı?
    pnl = Column(Float, default=0.0) # Kâr/Zarar (Pozisyon kapanınca dolar)
    timestamp = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Trade {self.symbol} {self.side} @ {self.price}>"