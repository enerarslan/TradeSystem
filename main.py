import asyncio
from utils.logger import log
from config.settings import settings
from data.feed import DataStream
from data.db import init_db
from strategies.momentum import SimpleMomentum 
from risk.core import RiskManager
from execution.portfolio import PortfolioManager
from execution.handler import ExecutionHandler

async def main_system():
    # --- AÇILIŞ EKRANI ---
    log.info("==================================================")
    log.info(f"   {settings.PROJECT_NAME} v{settings.VERSION}")
    log.info("   JPMorgan Grade Architecture | Active")
    log.info("==================================================")

    # 1. Veritabanını Başlat (Tablolar yoksa oluşturur)
    await init_db()
    log.success("Veritabanı bağlantısı kuruldu.")
    
    # 2. Modülleri Yükle
    stream = DataStream(exchange_id='binance')
    portfolio = PortfolioManager(initial_balance=10000.0)
    risk_engine = RiskManager()
    execution_handler = ExecutionHandler(portfolio) # Handler, portföyü yönetecek
    
    # 3. Stratejiyi Seç
    target_symbol = "BTC/USDT"
    strategy = SimpleMomentum(symbol=target_symbol, window_size=15)

    # 4. Bağlantı
    await stream.connect()

    try:
        log.info("🚀 Motor Çalışıyor. Piyasalar dinleniyor...")
        
        while True:
            # --- FAZ 1: GÖZLEM (Data) ---
            tick = await stream.get_latest_price(target_symbol)
            
            if tick:
                # --- FAZ 2: ANALİZ (Strategy) ---
                signal = await strategy.on_tick(tick)
                
                if signal:
                    # --- FAZ 3: KORUMA (Risk) ---
                    # Risk motoruna "Şu anki cüzdanımla bu işlemi yapabilir miyim?" diye sor
                    portfolio_state = portfolio.get_state()
                    risk_decision = risk_engine.analyze_signal(signal, portfolio_state)
                    
                    if risk_decision.passed:
                        # --- FAZ 4: İCRA (Execution) ---
                        # Onaylanan miktarı (Risk tarafından düşürülmüş olabilir) uygula
                        log.success(f"✅ ONAY: {signal.side} Sinyali geçerli. İletiliyor...")
                        
                        await execution_handler.execute_order(
                            signal=signal,
                            approved_quantity=risk_decision.adjusted_quantity
                        )
                        
                        # Bakiyeyi ekrana bas
                        new_state = portfolio.get_state()
                        log.info(f"💰 CÜZDAN: {new_state.cash_balance:.2f} USD | Açık Pozisyon: {new_state.open_positions_count}")
                        
                    else:
                        log.warning(f"⛔ RED: Risk limiti engeli -> {risk_decision.reason}")

            # CPU'yu rahatlat (HFT değilsek 1 saniye iyidir)
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        log.warning("Kullanıcı tarafından durduruluyor...")
    except Exception as e:
        log.exception(f"KRİTİK SİSTEM HATASI: {e}")
    finally:
        await stream.close()
        log.success("Sistem güvenli kapatıldı.")

if __name__ == "__main__":
    asyncio.run(main_system())