import asyncio
from utils.logger import log
from data.csv_loader import LocalCSVLoader
from strategies.momentum import SimpleMomentum
from risk.core import RiskManager
from execution.portfolio import PortfolioManager
import time

async def run_backtest():
    log.info("=== JPMORGAN KURUMSAL BACKTEST MOTORU (V2) ===")
    
    # 1. Veriyi Yükle
    target_symbol = "AAPL" 
    loader = LocalCSVLoader()
    ticks = loader.load_data(target_symbol)
    
    if not ticks:
        return

    # 2. Simülasyon Ortamını Kur
    portfolio = PortfolioManager(initial_balance=100000.0) 
    strategy = SimpleMomentum(symbol=target_symbol, window_size=20)
    risk_engine = RiskManager()
    
    trade_count = 0
    start_time = time.time()

    print(f"\n--- SİMÜLASYON BAŞLIYOR ({len(ticks)} Mum) ---")

    # 3. HIZLI ZAMAN DÖNGÜSÜ
    for tick in ticks:
        # ÖNEMLİ: Her tick'te portföye güncel fiyatı bildir (Mark-to-Market)
        portfolio.update_price(tick.symbol, tick.price)

        # A. Strateji Analizi
        signal = await strategy.on_tick(tick)
        
        if signal:
            # B. Risk Kontrolü
            state = portfolio.get_state()
            risk_decision = risk_engine.analyze_signal(signal, state)
            
            if risk_decision.passed:
                # C. İşlem İcrası
                # Miktarı int'e çeviriyoruz çünkü hisse senetleri genelde tam sayı alınır (opsiyonel)
                qty = int(risk_decision.adjusted_quantity) if risk_decision.adjusted_quantity >= 1 else 0
                
                if qty > 0:
                    portfolio.update_after_trade(
                        symbol=signal.symbol,
                        quantity=qty,
                        price=tick.price,
                        side=signal.side
                    )
                    trade_count += 1
                    
                    # İlerleme efekti
                    if trade_count % 10 == 0:
                        print(".", end="", flush=True)

    # 4. SONUÇ RAPORU
    end_time = time.time()
    final_state = portfolio.get_state() # Son fiyatlarla güncellenmiş durum
    net_profit = final_state.total_balance - 100000.0
    roi = (net_profit / 100000.0) * 100
    
    print("\n" + "="*50)
    print(f"   BACKTEST RAPORU: {target_symbol}")
    print("="*50)
    print(f"Toplam İşlem  : {trade_count}")
    print(f"Başlangıç     : $100,000.00")
    print(f"Bitiş (Varlık): ${final_state.total_balance:,.2f}")
    print(f"Nakit Durumu  : ${final_state.cash_balance:,.2f}")
    print(f"Net Kâr/Zarar : ${net_profit:,.2f}")
    print(f"ROI (Getiri)  : %{roi:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_backtest())