from decimal import Decimal, getcontext
from utils.logger import log
from config.settings import settings
from data.models import TradeSignal, PortfolioState, RiskCheckResult, Side

# Finansal hassasiyet için Decimal ayarı
getcontext().prec = 8

class RiskManager:
    """
    Kurumsal Seviye Risk Yönetim Motoru.
    Stratejiden gelen sinyalleri süzer, onaylar veya reddeder.
    """
    def __init__(self):
        self.max_drawdown = settings.MAX_DAILY_DRAWDOWN_PERCENT
        self.max_pos_size = settings.MAX_POSITION_SIZE_PERCENT
        self.max_daily_trades = settings.MAX_TRADES_PER_DAY
        log.info("Risk Yönetim Modülü Başlatıldı [HARD LIMITS ACTIVE]")

    def analyze_signal(self, signal: TradeSignal, portfolio: PortfolioState) -> RiskCheckResult:
        """
        Gelen bir ticaret sinyalini (Signal) portföy durumuna göre analiz eder.
        """
        log.debug(f"Risk Analizi Başlıyor: {signal.symbol} - {signal.side}")

        # 1. KONTROL: Günlük İşlem Limiti (Overtrading Check)
        if portfolio.daily_trade_count >= self.max_daily_trades:
            return RiskCheckResult(
                passed=False, 
                adjusted_quantity=0.0, 
                reason=f"Günlük işlem limiti aşıldı! ({portfolio.daily_trade_count}/{self.max_daily_trades})"
            )

        # 2. KONTROL: Günlük Zarar Limiti (Circuit Breaker)
        # Eğer bugün portföy %2'den fazla eridiyse, yeni işlem açma.
        current_drawdown_pct = (portfolio.daily_pnl / portfolio.total_balance) * 100
        if current_drawdown_pct < -self.max_drawdown:
            return RiskCheckResult(
                passed=False, 
                adjusted_quantity=0.0, 
                reason=f"GÜNLÜK ZARAR LİMİTİ AŞILDI! (Drawdown: {current_drawdown_pct:.2f}%)"
            )

        # 3. KONTROL: Nakit Yeterliliği (Liquidity Check)
        required_capital = signal.price * signal.quantity
        if signal.side == Side.BUY and required_capital > portfolio.cash_balance:
            # Bakiyeden fazla almaya çalışıyor -> Reddetme, MİKTARI DÜŞÜR.
            max_buyable = portfolio.cash_balance * 0.99 # %1 komisyon payı bırak
            new_quantity = max_buyable / signal.price
            
            log.warning(f"Yetersiz Bakiye. Miktar revize ediliyor: {signal.quantity} -> {new_quantity}")
            
            # Eğer revize edilen miktar çok küçükse reddet
            if new_quantity * signal.price < 10.0: # Minimum 10$ işlem
                 return RiskCheckResult(
                    passed=False,
                    adjusted_quantity=0.0,
                    reason="Yetersiz bakiye (Minimum işlem tutarının altında)"
                )
            
            return RiskCheckResult(
                passed=True,
                adjusted_quantity=new_quantity,
                reason="Bakiye yetersizliği nedeniyle miktar düşürüldü."
            )

        # 4. KONTROL: Pozisyon Büyüklüğü (Position Sizing)
        # Tek bir varlığa portföyün %10'undan fazlasını yatırma.
        max_allowed_capital = portfolio.total_balance * (self.max_pos_size / 100.0)
        if required_capital > max_allowed_capital:
            allowed_qty = max_allowed_capital / signal.price
            return RiskCheckResult(
                passed=True,
                adjusted_quantity=allowed_qty,
                reason=f"Risk Dağılımı: İşlem büyüklüğü %{self.max_pos_size} ile sınırlandı."
            )

        # Tüm kontrollerden geçtiyse onayla
        return RiskCheckResult(
            passed=True,
            adjusted_quantity=signal.quantity,
            reason="Tüm risk kontrolleri başarılı."
        )