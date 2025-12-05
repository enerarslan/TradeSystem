import ccxt.async_support as ccxt  # Asenkron versiyonu kullanıyoruz
import asyncio
from datetime import datetime
from utils.logger import log
from data.models import MarketTick

class DataStream:
    def __init__(self, exchange_id='binance'):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)()
        self.running = False
    
    async def connect(self):
        """Borsa API bağlantısını kurar."""
        try:
            log.info(f"{self.exchange_id.upper()} bağlantısı başlatılıyor...")
            await self.exchange.load_markets() # Market bilgilerini yükle
            log.success(f"{self.exchange_id.upper()} bağlantısı başarılı!")
        except Exception as e:
            log.critical(f"Borsa bağlantı hatası: {e}")
            raise e

    async def get_latest_price(self, symbol: str) -> MarketTick:
        """
        Tek bir sembol için anlık fiyat çeker ve standart modele çevirir.
        """
        if not self.exchange.has['fetchTicker']:
            raise NotImplementedError(f"{self.exchange_id} ticker çekmeyi desteklemiyor.")

        try:
            # API'den veriyi çek (await ile beklemeden)
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Ham veriyi bizim standart modelimize (MarketTick) dönüştür
            tick_data = MarketTick(
                symbol=symbol,
                price=ticker['last'],
                volume=ticker['baseVolume'] if ticker.get('baseVolume') else 0.0,
                timestamp=datetime.now(),
                source=self.exchange_id
            )
            return tick_data
            
        except Exception as e:
            log.error(f"Veri çekme hatası ({symbol}): {e}")
            return None

    async def close(self):
        """Bağlantıyı temiz bir şekilde kapatır."""
        await self.exchange.close()
        log.info("Borsa bağlantısı kapatıldı.")