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
    log.info("=== SİSTEM BAŞLATILIYOR ===")
    
    # 1. Veritabanı
    await init_db()
    
    # 2. Modüller
    # NOT: API Key olmadığı için 'binance' sadece public veri çeker.
    stream = DataStream(exchange_id='binance') 
    portfolio = PortfolioManager(initial_balance=10000.0)
    risk_engine = RiskManager()
    execution_handler = ExecutionHandler(portfolio)
    
    # 3. Strateji
    target_symbol = "BTC/USDT"
    strategy = SimpleMomentum(symbol=target_symbol, window_size=10) # Pencereyi 10'a düşürdük daha hızlı başlasın

    # 4. Bağlantı
    await stream.connect()

    log.info("⏳ Tampon veri toplanıyor (İlk 10 saniye işlem olmaz)...")

    try:
        while True:
            tick = await stream.get_latest_price(target_symbol)
            
            if tick:
                # Portföydeki anlık fiyatı güncelle (Mark-to-Market)
                portfolio.update_price(tick.symbol, tick.price)
                
                # Sinyal Üret
                signal = await strategy.on_tick(tick)
                
                if signal:
                    portfolio_state = portfolio.get_state()
                    
                    # --- BASİT FİLTRE: Zaten pozisyon varsa ve AL diyorsa engelle ---
                    current_qty = portfolio.positions.get(signal.symbol, 0)
                    if signal.side == "BUY" and current_qty > 0:
                        pass # Zaten elimizde var, ekleme yapma (Simple Momentum kuralı)
                    elif signal.side == "SELL" and current_qty == 0:
                        pass # Elimizde yokken satamayız
                    else:
                        # Risk Analizi
                        risk_decision = risk_engine.analyze_signal(signal, portfolio_state)
                        
                        if risk_decision.passed:
                            await execution_handler.execute_order(
                                signal=signal,
                                approved_quantity=risk_decision.adjusted_quantity
                            )
                            # Bakiye Bilgisi
                            st = portfolio.get_state()
                            log.info(f"💰 Bakiye: {st.cash_balance:.2f} USD | PnL: {st.daily_pnl:.2f}")

            await asyncio.sleep(1) # API limitlerine takılmamak için

    except KeyboardInterrupt:
        log.warning("Durduruluyor...")
    finally:
        await stream.close()

if __name__ == "__main__":
    asyncio.run(main_system())