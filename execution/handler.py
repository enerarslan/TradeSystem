from data.models import TradeSignal, Side
from data.schema import TradeRecord
from data.db import async_session
from execution.portfolio import PortfolioManager
from utils.logger import log

class ExecutionHandler:
    """
    Emirlerin piyasaya iletilmesinden ve kayıt altına alınmasından sorumlu sınıf.
    JPMorgan'da buna OMS (Order Management System) denir.
    """
    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio

    async def execute_order(self, signal: TradeSignal, approved_quantity: float):
        """
        1. Emri Borsaya İletir (Şimdilik Simülasyon)
        2. Veritabanına Kaydeder
        3. Portföyü Günceller
        """
        if approved_quantity <= 0:
            return

        # 1. Borsa İletimi (Burada gerçek API çağrısı olacak)
        # await exchange.create_order(...) 
        # Şimdilik "Filled" varsayıyoruz.
        fill_price = signal.price # Gerçek hayatta slippage (kayma) olur
        
        log.info(f"⚡ EXECUTION: {signal.symbol} için {approved_quantity} adet {signal.side} emri iletildi.")

        # 2. Portföy Güncellemesi (RAM)
        self.portfolio.update_after_trade(
            symbol=signal.symbol,
            quantity=approved_quantity,
            price=fill_price,
            side=signal.side
        )

        # 3. Veritabanı Kaydı (Disk)
        await self._save_trade_to_db(signal, approved_quantity, fill_price)

    async def _save_trade_to_db(self, signal: TradeSignal, quantity: float, price: float):
        """Asenkron olarak veritabanına yazar."""
        try:
            async with async_session() as session:
                async with session.begin():
                    new_trade = TradeRecord(
                        symbol=signal.symbol,
                        side=signal.side,
                        price=price,
                        quantity=quantity,
                        strategy=signal.strategy_name,
                        timestamp=signal.timestamp
                    )
                    session.add(new_trade)
                # Otomatik commit olur
                log.debug(f"💾 DB KAYIT: İşlem veritabanına işlendi (ID: Otomatik)")
        except Exception as e:
            log.error(f"Veritabanı kayıt hatası: {e}")