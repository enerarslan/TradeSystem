from data.models import PortfolioState
from utils.logger import log

class PortfolioManager:
    """
    Gelişmiş Portföy Yöneticisi.
    Hem nakdi hem de eldeki hisselerin anlık değerini (Mark-to-Market) takip eder.
    """
    def __init__(self, initial_balance=100000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance     # Kullanılabilir Nakit
        self.positions = {}             # { 'AAPL': 10, 'AMZN': 5 } (Adetler)
        self.average_costs = {}         # { 'AAPL': 150.0 } (Ortalama Maliyet)
        self.trades_today = 0
        self.realized_pnl = 0.0         # Cebe giren kâr/zarar
        
        # Değerleme için son görülen fiyatlar
        self.last_prices = {} 

    def update_price(self, symbol: str, price: float):
        """Piyasa fiyatı değiştikçe varlık değerlemesi için fiyatı günceller."""
        self.last_prices[symbol] = price

    def get_total_equity(self) -> float:
        """
        Toplam Varlık = Nakit + (Hisse Adedi * Güncel Fiyat)
        """
        equity = self.cash
        for symbol, quantity in self.positions.items():
            # Eğer güncel fiyatı biliyorsak onla çarp, bilmiyorsak maliyetle (fail-safe)
            price = self.last_prices.get(symbol, self.average_costs.get(symbol, 0))
            equity += quantity * price
        return equity

    def get_state(self) -> PortfolioState:
        """
        Risk yöneticisine anlık durumu raporlar.
        """
        total_equity = self.get_total_equity()
        
        # Günlük PnL = Şu anki Varlık - Başlangıç (Basitleştirilmiş)
        # Gerçek sistemde günlük resetlenir, şimdilik kümülatif bakıyoruz.
        current_pnl = total_equity - self.initial_balance

        return PortfolioState(
            total_balance=total_equity,
            cash_balance=self.cash,
            daily_pnl=current_pnl,
            open_positions_count=len(self.positions),
            daily_trade_count=self.trades_today
        )

    def update_after_trade(self, symbol: str, quantity: float, price: float, side: str):
        """
        İşlem gerçekleştikten sonra cüzdanı günceller.
        """
        cost = quantity * price
        
        # İşlem anındaki fiyatı kaydet (Değerleme için)
        self.update_price(symbol, price)

        if side == "BUY":
            if self.cash >= cost:
                self.cash -= cost
                
                # Ortalama Maliyet Hesabı (Weighted Average)
                old_qty = self.positions.get(symbol, 0)
                if old_qty > 0:
                    old_cost = self.average_costs.get(symbol, 0)
                    total_val = (old_qty * old_cost) + cost
                    new_avg = total_val / (old_qty + quantity)
                    self.average_costs[symbol] = new_avg
                else:
                    self.average_costs[symbol] = price
                
                self.positions[symbol] = old_qty + quantity
                self.trades_today += 1
            else:
                log.error("Portfolio Manager: Yetersiz Bakiye (Burası tetiklenmemeliydi)")

        elif side == "SELL":
            current_qty = self.positions.get(symbol, 0)
            if current_qty >= quantity:
                self.cash += cost
                self.positions[symbol] -= quantity
                
                # Realize Edilen Kârı Hesapla
                avg_cost = self.average_costs.get(symbol, 0)
                profit = (price - avg_cost) * quantity
                self.realized_pnl += profit
                
                # Eğer pozisyon bittiyse temizle
                if self.positions[symbol] <= 0.000001: # Floating point hatası için tolerans
                    del self.positions[symbol]
                    del self.average_costs[symbol]
                
                self.trades_today += 1