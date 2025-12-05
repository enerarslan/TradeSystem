import asyncio
from utils.logger import log
from data.csv_loader import LocalCSVLoader
from strategies.momentum import SimpleMomentum
from risk.core import RiskManager
from execution.portfolio import PortfolioManager
import time

async def run_backtest():
    log.info("=== JPMORGAN KURUMSAL BACKTEST MOTORU ===")
    
    # 1. Veriyi Yükle (Senin dosyanı kullanıyoruz)
    target_symbol = "AAPL" 
    loader = LocalCSVLoader()
    ticks = loader.load_data(target_symbol)
    
    if not ticks:
        return

    # 2. Simülasyon Ortamını Kur
    # Başlangıç sermayesi 100.000$
    portfolio = PortfolioManager(initial_balance=100000.0) 
    
    # Stratejiyi Yükle (20 periyotluk ortalama)
    strategy = SimpleMomentum(symbol=target_symbol, window_size=20)
    
    risk_engine = RiskManager()
    
    trade_count = 0
    start_time = time.time()

    print(f"\n--- SİMÜLASYON BAŞLIYOR ({len(ticks)} Veri Noktası) ---")

    # 3. HIZLI ZAMAN DÖNGÜSÜ
    # Canlı sistemdeki gibi 'bekle' (sleep) yok, işlemci hızında çalışır.
    for tick in ticks:
        # A. Stratejiye veriyi göster
        signal = await strategy.on_tick(tick)
        
        if signal:
            # B. Risk Kontrolü
            state = portfolio.get_state()
            risk_decision = risk_engine.analyze_signal(signal, state)
            
            if risk_decision.passed:
                # C. Sanal İşlem
                portfolio.update_after_trade(
                    symbol=signal.symbol,
                    quantity=risk_decision.adjusted_quantity,
                    price=tick.price,
                    side=signal.side
                )
                trade_count += 1
                
                # İşlem olunca ekrana küçük bir nokta bas (İlerleme çubuğu gibi)
                if trade_count % 10 == 0:
                    print(".", end="", flush=True)

    # 4. SONUÇ RAPORU
    end_time = time.time()
    final_state = portfolio.get_state()
    net_profit = final_state.total_balance - 100000.0
    roi = (net_profit / 100000.0) * 100 # Yatırım Getirisi (ROI)
    
    print("\n" + "="*50)
    print(f"   BACKTEST RAPORU: {target_symbol} (15dk)")
    print("="*50)
    print(f"Analiz Süresi : {end_time - start_time:.2f} saniye")
    print(f"Veri Seti     : 2021 -> 2025")
    print("-" * 50)
    print(f"Toplam İşlem  : {trade_count}")
    print(f"Başlangıç     : $100,000.00")
    print(f"Bitiş         : ${final_state.total_balance:,.2f}")
    print(f"Net Kâr/Zarar : ${net_profit:,.2f}")
    print(f"ROI (Getiri)  : %{roi:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_backtest())