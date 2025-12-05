from data.models import PortfolioState

class PortfolioManager:
    """
    Şimdilik simülasyon (Paper Trading) için sanal bakiye yönetir.
    İleride burası borsa API'sine bağlanıp gerçek bakiyeyi çekecek.
    """
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.cash = initial_balance
        self.positions = {} # { 'BTC/USDT': 0.5 }
        self.trades_today = 0
        self.daily_pnl = 0.0

    def get_state(self) -> PortfolioState:
        """
        Risk yöneticisine anlık durumu raporlar.
        """
        return PortfolioState(
            total_balance=self.current_balance,
            cash_balance=self.cash,
            daily_pnl=self.daily_pnl,
            open_positions_count=len(self.positions),
            daily_trade_count=self.trades_today
        )

    def update_after_trade(self, symbol: str, quantity: float, price: float, side: str):
        """
        İşlem gerçekleştikten sonra cüzdanı günceller.
        """
        cost = quantity * price
        
        if side == "BUY":
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            self.trades_today += 1
        elif side == "SELL":
            self.cash += cost
            current_qty = self.positions.get(symbol, 0)
            if current_qty >= quantity:
                self.positions[symbol] -= quantity
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
            self.trades_today += 1
            
        # Basit PnL hesabı (Gerçek sistemde burası daha karmaşık olur)
        # Şimdilik bakiye değişimini simüle ediyoruz.