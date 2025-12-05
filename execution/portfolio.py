"""
KURUMSAL PORTFOLIO YÖNETİMİ
JPMorgan Asset Management Tarzı

Özellikler:
- Mark-to-Market Değerleme
- Multi-asset Portfolio
- Realized/Unrealized PnL Tracking
- Average Cost Basis (Weighted)
- Position History
- Performance Attribution
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
from datetime import datetime
from dataclasses import dataclass

from data.models import PortfolioState, Side
from utils.logger import log


# Hassas hesaplamalar için
getcontext().prec = 10


@dataclass
class Position:
    """Tek bir pozisyonun detayları"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    last_update: datetime


@dataclass
class TradeHistory:
    """İşlem geçmişi kaydı"""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    realized_pnl: Optional[float]


class PortfolioManager:
    """
    Gelişmiş portföy yönetim sistemi.
    
    Özellikler:
    - Real-time Mark-to-Market
    - Multi-asset support
    - Average cost tracking (weighted average)
    - Realized vs Unrealized PnL
    - Position-level analytics
    - Historical tracking
    """
    
    def __init__(self, initial_balance: float = 100_000.0):
        """
        Args:
            initial_balance: Başlangıç sermayesi
        """
        self.initial_balance = Decimal(str(initial_balance))
        self.cash = Decimal(str(initial_balance))
        
        # Pozisyonlar: {symbol: quantity}
        self.positions: Dict[str, float] = {}
        
        # Ortalama maliyetler: {symbol: avg_cost}
        self.average_costs: Dict[str, float] = {}
        
        # Son bilinen fiyatlar: {symbol: price}
        self.last_prices: Dict[str, float] = {}
        
        # Pozisyon açılış zamanları: {symbol: datetime}
        self.position_entry_times: Dict[str, datetime] = {}
        
        # Günlük tracking
        self.trades_today = 0
        self.realized_pnl_today = Decimal('0')
        self.peak_balance_today = Decimal(str(initial_balance))
        
        # Tüm zamanların toplam realize PnL'si
        self.total_realized_pnl = Decimal('0')
        
        # İşlem geçmişi
        self.trade_history: List[TradeHistory] = []
        
        # Performance tracking
        self.daily_equity_curve: List[Tuple[datetime, float]] = []
        
        log.info(f"💼 Portfolio Manager başlatıldı (Sermaye: ${initial_balance:,.2f})")
    
    def update_price(self, symbol: str, price: float):
        """
        Bir varlığın piyasa fiyatını günceller (Mark-to-Market).
        
        Args:
            symbol: Sembol adı
            price: Güncel piyasa fiyatı
        """
        self.last_prices[symbol] = price
    
    def get_total_equity(self) -> float:
        """
        Toplam portföy değerini hesaplar.
        
        Returns:
            float: Nakit + Pozisyonların piyasa değeri
        """
        equity = float(self.cash)
        
        for symbol, quantity in self.positions.items():
            if quantity > 0.000001:  # Floating point tolerance
                current_price = self.last_prices.get(symbol, self.average_costs.get(symbol, 0))
                market_value = quantity * current_price
                equity += market_value
        
        return equity
    
    def get_unrealized_pnl(self) -> float:
        """
        Tüm pozisyonların toplam realize olmamış kar/zararı.
        
        Returns:
            float: Toplam unrealized PnL
        """
        unrealized = 0.0
        
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                avg_cost = self.average_costs.get(symbol, 0)
                current_price = self.last_prices.get(symbol, avg_cost)
                pnl = (current_price - avg_cost) * quantity
                unrealized += pnl
        
        return unrealized
    
    def get_state(self) -> PortfolioState:
        """
        Anlık portföy durumunu döndürür.
        
        Returns:
            PortfolioState: Portfolio snapshot
        """
        total_equity = self.get_total_equity()
        unrealized_pnl = self.get_unrealized_pnl()
        
        # Günlük PnL = Realize edilen + Realize edilmemiş
        daily_pnl = float(self.realized_pnl_today) + unrealized_pnl
        
        return PortfolioState(
            total_balance=total_equity,
            cash_balance=float(self.cash),
            daily_pnl=daily_pnl,
            open_positions_count=len([q for q in self.positions.values() if q > 0]),
            daily_trade_count=self.trades_today
        )
    
    def update_after_trade(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        side: str
    ):
        """
        İşlem gerçekleştikten sonra portföyü günceller.
        
        Args:
            symbol: Sembol
            quantity: Miktar
            price: Gerçekleşme fiyatı
            side: "BUY" veya "SELL"
        """
        # Fiyatı kaydet
        self.update_price(symbol, price)
        
        if side == Side.BUY or side == "BUY":
            self._handle_buy(symbol, quantity, price)
        elif side == Side.SELL or side == "SELL":
            self._handle_sell(symbol, quantity, price)
        
        # Günlük işlem sayısını artır
        self.trades_today += 1
        
        # Peak balance güncelle
        current_equity = Decimal(str(self.get_total_equity()))
        if current_equity > self.peak_balance_today:
            self.peak_balance_today = current_equity
    
    def _handle_buy(self, symbol: str, quantity: float, price: float):
        """
        Alım işlemini işler.
        """
        cost = Decimal(str(price * quantity))
        
        # Nakit kontrolü
        if cost > self.cash:
            log.error(f"❌ Yetersiz nakit! Gerekli: ${cost:.2f}, Mevcut: ${self.cash:.2f}")
            return
        
        # Nakit düş
        self.cash -= cost
        
        # Pozisyon var mı?
        old_quantity = self.positions.get(symbol, 0)
        
        if old_quantity > 0:
            # Mevcut pozisyona ekleme - Weighted Average Cost
            old_cost = self.average_costs.get(symbol, 0)
            old_value = old_quantity * old_cost
            new_value = quantity * price
            
            total_quantity = old_quantity + quantity
            new_avg_cost = (old_value + new_value) / total_quantity
            
            self.average_costs[symbol] = new_avg_cost
            self.positions[symbol] = total_quantity
            
            log.debug(f"📊 Pozisyon artırıldı: {symbol}")
            log.debug(f"   Eski: {old_quantity:.2f} @ ${old_cost:.2f}")
            log.debug(f"   Yeni: {total_quantity:.2f} @ ${new_avg_cost:.2f}")
        else:
            # Yeni pozisyon
            self.positions[symbol] = quantity
            self.average_costs[symbol] = price
            self.position_entry_times[symbol] = datetime.now()
            
            log.info(f"🟢 YENİ POZİSYON: {symbol} - {quantity:.2f} @ ${price:.2f}")
        
        # İşlem geçmişine ekle
        self.trade_history.append(
            TradeHistory(
                timestamp=datetime.now(),
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                price=price,
                commission=0,  # Komisyon ayrı hesaplanabilir
                realized_pnl=None
            )
        )
    
    def _handle_sell(self, symbol: str, quantity: float, price: float):
        """
        Satım işlemini işler.
        """
        current_quantity = self.positions.get(symbol, 0)
        
        # Pozisyon kontrolü
        if current_quantity < quantity:
            log.error(f"❌ Yetersiz pozisyon! Mevcut: {current_quantity}, İstenen: {quantity}")
            return
        
        # Nakit artır
        proceeds = Decimal(str(price * quantity))
        self.cash += proceeds
        
        # Realized PnL hesapla
        avg_cost = self.average_costs.get(symbol, 0)
        realized_pnl = (price - avg_cost) * quantity
        
        self.realized_pnl_today += Decimal(str(realized_pnl))
        self.total_realized_pnl += Decimal(str(realized_pnl))
        
        # Pozisyonu güncelle
        new_quantity = current_quantity - quantity
        self.positions[symbol] = new_quantity
        
        # Pozisyon tamamen kapandıysa temizle
        if new_quantity < 0.000001:  # Floating point tolerance
            self.positions.pop(symbol, None)
            self.average_costs.pop(symbol, None)
            entry_time = self.position_entry_times.pop(symbol, None)
            
            holding_period = (datetime.now() - entry_time).total_seconds() / 3600 if entry_time else 0
            
            log.info(f"🔴 POZİSYON KAPANDI: {symbol}")
            log.info(f"   Realized PnL: ${realized_pnl:+,.2f} ({(realized_pnl/(avg_cost*quantity)*100):+.2f}%)")
            log.info(f"   Holding: {holding_period:.1f} saat")
        else:
            log.info(f"🟡 POZİSYON AZALTILDI: {symbol} - {new_quantity:.2f} kaldı")
            log.info(f"   Partial PnL: ${realized_pnl:+,.2f}")
        
        # İşlem geçmişine ekle
        self.trade_history.append(
            TradeHistory(
                timestamp=datetime.now(),
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                commission=0,
                realized_pnl=realized_pnl
            )
        )
    
    def get_positions_summary(self) -> List[Position]:
        """
        Tüm açık pozisyonların detaylı listesi.
        
        Returns:
            List[Position]: Pozisyon detayları
        """
        positions = []
        
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                avg_cost = self.average_costs.get(symbol, 0)
                current_price = self.last_prices.get(symbol, avg_cost)
                market_value = quantity * current_price
                unrealized_pnl = (current_price - avg_cost) * quantity
                unrealized_pnl_pct = (unrealized_pnl / (avg_cost * quantity)) * 100 if avg_cost > 0 else 0
                
                entry_time = self.position_entry_times.get(symbol, datetime.now())
                
                positions.append(
                    Position(
                        symbol=symbol,
                        quantity=quantity,
                        average_cost=avg_cost,
                        current_price=current_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        entry_time=entry_time,
                        last_update=datetime.now()
                    )
                )
        
        return positions
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Detaylı portföy metrikleri.
        
        Returns:
            Dict: Metrikler
        """
        total_equity = self.get_total_equity()
        unrealized_pnl = self.get_unrealized_pnl()
        
        # Invested capital
        invested = sum([
            self.positions.get(symbol, 0) * self.average_costs.get(symbol, 0)
            for symbol in self.positions.keys()
        ])
        
        # Cash weight
        cash_weight = (float(self.cash) / total_equity) * 100 if total_equity > 0 else 0
        
        # Total return
        total_return = ((total_equity / float(self.initial_balance)) - 1) * 100
        
        # Today's return
        if self.peak_balance_today > 0:
            today_return = ((Decimal(str(total_equity)) / self.peak_balance_today) - 1) * 100
        else:
            today_return = 0
        
        return {
            'total_equity': total_equity,
            'cash': float(self.cash),
            'cash_weight_pct': cash_weight,
            'invested_capital': invested,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl_today': float(self.realized_pnl_today),
            'total_realized_pnl': float(self.total_realized_pnl),
            'total_return_pct': total_return,
            'today_return_pct': float(today_return),
            'num_positions': len([q for q in self.positions.values() if q > 0]),
            'trades_today': self.trades_today
        }
    
    def print_portfolio_summary(self):
        """
        Portföy özetini yazdırır.
        """
        metrics = self.get_portfolio_metrics()
        positions = self.get_positions_summary()
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*20 + "💼 PORTFÖY ÖZETİ" + " "*32 + "║")
        print("╠" + "═"*68 + "╣")
        print(f"║  Toplam Varlık      : ${metrics['total_equity']:>12,.2f}{' '*30} ║")
        print(f"║  Nakit              : ${metrics['cash']:>12,.2f} ({metrics['cash_weight_pct']:>5.1f}%){' '*18} ║")
        print(f"║  Yatırılan Sermaye  : ${metrics['invested_capital']:>12,.2f}{' '*30} ║")
        print(f"║  Unrealized PnL     : ${metrics['unrealized_pnl']:>+12,.2f}{' '*30} ║")
        print(f"║  Realized PnL (Gün) : ${metrics['realized_pnl_today']:>+12,.2f}{' '*30} ║")
        print(f"║  Toplam Return      : {metrics['total_return_pct']:>+11.2f}%{' '*32} ║")
        print(f"║  Açık Pozisyon      : {metrics['num_positions']:>12}{' '*37} ║")
        print(f"║  Günlük İşlem       : {metrics['trades_today']:>12}{' '*37} ║")
        print("╠" + "═"*68 + "╣")
        
        if positions:
            print("║  AÇIK POZİSYONLAR" + " "*50 + "║")
            print("║  " + "─"*66 + "║")
            
            for pos in positions:
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                print(f"║  {pos.symbol:<8} | {pos.quantity:>8.2f} @ ${pos.average_cost:>8.2f} | PnL: ${pnl_sign}{pos.unrealized_pnl:>8,.2f} ({pnl_sign}{pos.unrealized_pnl_pct:>5.1f}%) ║")
        else:
            print("║  Açık pozisyon yok" + " "*49 + "║")
        
        print("╚" + "═"*68 + "╝\n")
    
    def reset_daily_stats(self):
        """
        Günlük istatistikleri sıfırla (Yeni gün başlangıcında çağrılır).
        """
        self.trades_today = 0
        self.realized_pnl_today = Decimal('0')
        current_equity = Decimal(str(self.get_total_equity()))
        self.peak_balance_today = current_equity
        
        log.info("🔄 Günlük portföy istatistikleri sıfırlandı")
    
    def export_trade_history(self, filename: str = "trade_history.csv"):
        """
        İşlem geçmişini CSV'ye export eder.
        """
        import pandas as pd
        
        if not self.trade_history:
            log.warning("Export edilecek işlem kaydı yok")
            return
        
        data = []
        for trade in self.trade_history:
            data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'realized_pnl': trade.realized_pnl if trade.realized_pnl else 0
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        log.success(f"📁 İşlem geçmişi kaydedildi: {filename}")


# KULLANIM ÖRNEĞİ
"""
portfolio = PortfolioManager(initial_balance=100_000)

# Fiyat güncellemesi (Mark-to-Market)
portfolio.update_price("AAPL", 150.25)

# Alım işlemi
portfolio.update_after_trade(
    symbol="AAPL",
    quantity=10,
    price=150.25,
    side="BUY"
)

# Satım işlemi
portfolio.update_after_trade(
    symbol="AAPL",
    quantity=5,
    price=155.50,
    side="SELL"
)

# Durumu göster
portfolio.print_portfolio_summary()

# Metrikleri al
metrics = portfolio.get_portfolio_metrics()
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
"""