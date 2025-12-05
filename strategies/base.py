from abc import ABC, abstractmethod
from data.models import MarketTick, TradeSignal

class BaseStrategy(ABC):
    """
    Tüm stratejiler bu sınıftan miras almak ZORUNDADIR.
    Bu sayede sistem, hangi strateji gelirse gelsin nasıl çalıştıracağını bilir.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def on_tick(self, tick: MarketTick) -> TradeSignal:
        """
        Her yeni fiyat geldiğinde bu fonksiyon çalışacak.
        """
        pass