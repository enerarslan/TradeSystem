"""
KURUMSAL SEVİYE RİSK YÖNETİM MOTORUjp
JPMorgan Risk Management Division Tarzı

Çok Katmanlı Risk Kontrol Sistemi:
1. Position-Level Risk (Pozisyon riski)
2. Portfolio-Level Risk (Portföy riski)
3. Concentration Risk (Yoğunlaşma riski)
4. Market Risk (Piyasa riski)
5. Liquidity Risk (Likidite riski)
6. Operational Risk (Operasyonel risk)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from dataclasses import dataclass, field

from utils.logger import log
from config.settings import settings
from data.models import TradeSignal, PortfolioState, RiskCheckResult, Side


# Hassas hesaplamalar için Decimal precision
getcontext().prec = 10


@dataclass
class RiskMetrics:
    """Anlık risk metrikleri"""
    var_1d: float  # 1-günlük Value at Risk
    cvar_1d: float  # Conditional VaR (Expected Shortfall)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    concentration_score: float  # 0-100, yüksekse tehlikeli
    liquidity_score: float  # 0-100, düşükse tehlikeli


@dataclass
class RiskLimitConfig:
    """Risk limitleri konfigürasyonu"""
    # Position limits
    max_position_size_usd: float = 50_000
    max_position_size_pct: float = 10.0  # Portföyün %10'u
    
    # Portfolio limits
    max_portfolio_leverage: float = 1.0  # Spot trading için 1.0
    max_daily_loss_pct: float = 2.0  # Günlük max %2 zarar
    max_total_drawdown_pct: float = 10.0  # Toplam max %10 drawdown
    
    # Concentration limits
    max_single_sector_pct: float = 30.0  # Tek sektöre max %30
    max_correlated_positions: int = 3  # Yüksek korelasyonlu max 3 pozisyon
    
    # Trading limits
    max_daily_trades: int = 50
    max_open_positions: int = 10
    
    # Market risk limits
    max_var_1d: float = 5_000  # Günlük max $5k VaR
    max_volatility_exposure: float = 0.03  # %3 volatilite limiti
    
    # Liquidity limits
    min_cash_reserve_pct: float = 20.0  # Min %20 nakit bulundur


class EnterpriseRiskManager:
    """
    Kurumsal seviye risk yönetim motoru.
    
    Tüm işlemler bu sistemden geçer ve onay alır.
    Risk limitlerini aşan işlemler otomatik reddedilir veya ayarlanır.
    """
    
    def __init__(self, config: Optional[RiskLimitConfig] = None):
        """
        Args:
            config: Risk limitleri konfigürasyonu
        """
        self.config = config or RiskLimitConfig()
        
        # Geçmiş veri (risk metrikleri için)
        self.historical_returns: List[float] = []
        self.historical_equity: List[float] = []
        self.trade_history: List[Dict] = []
        
        # Günlük istatistikler
        self.daily_stats = {
            'trades_count': 0,
            'rejected_trades': 0,
            'start_balance': 0.0,
            'peak_balance': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown_today': 0.0
        }
        
        # Sektör exposure tracking
        self.sector_exposure: Dict[str, float] = {}
        
        # Korelasyon matrisi (basitleştirilmiş)
        self.correlation_pairs: Dict[Tuple[str, str], float] = {}
        
        log.info("🛡️ Enterprise Risk Manager başlatıldı")
        self._log_risk_limits()
    
    def _log_risk_limits(self):
        """Risk limitlerini logla"""
        log.info("=" * 60)
        log.info("   RİSK LİMİTLERİ KONFIGÜRASYONU")
        log.info("=" * 60)
        log.info(f"  Max Pozisyon Büyüklüğü  : ${self.config.max_position_size_usd:,.0f}")
        log.info(f"  Max Günlük Zarar        : %{self.config.max_daily_loss_pct}")
        log.info(f"  Max Toplam Drawdown     : %{self.config.max_total_drawdown_pct}")
        log.info(f"  Max Günlük İşlem        : {self.config.max_daily_trades}")
        log.info(f"  Max Açık Pozisyon       : {self.config.max_open_positions}")
        log.info(f"  Max VaR (1-gün)         : ${self.config.max_var_1d:,.0f}")
        log.info(f"  Min Nakit Rezervi       : %{self.config.min_cash_reserve_pct}")
        log.info("=" * 60)
    
    def analyze_signal(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict] = None
    ) -> RiskCheckResult:
        """
        Ana risk analizi fonksiyonu - Çok katmanlı kontrol.
        
        Args:
            signal: Strateji sinyali
            portfolio: Anlık portföy durumu
            market_data: Opsiyonel piyasa verileri
        
        Returns:
            RiskCheckResult: Onay/red kararı ve ayarlanmış miktar
        """
        log.debug(f"🔍 Risk Analizi: {signal.symbol} {signal.side} {signal.quantity} @ ${signal.price:.2f}")
        
        # Günlük statsları güncelle
        if self.daily_stats['start_balance'] == 0:
            self.daily_stats['start_balance'] = portfolio.total_balance
            self.daily_stats['peak_balance'] = portfolio.total_balance
        
        # Risk kontrolleri (sıralı)
        checks = [
            self._check_trading_limits,
            self._check_daily_loss_limit,
            self._check_cash_availability,
            self._check_position_sizing,
            self._check_concentration_risk,
            self._check_liquidity,
            self._check_portfolio_risk,
            self._check_market_conditions
        ]
        
        for check_func in checks:
            result = check_func(signal, portfolio, market_data)
            
            if not result.passed:
                self.daily_stats['rejected_trades'] += 1
                log.warning(f"❌ Risk Check Failed: {result.reason}")
                return result
            
            # Miktar ayarlandıysa, sonraki kontrollerde yeni miktarı kullan
            if result.adjusted_quantity != signal.quantity:
                signal.quantity = result.adjusted_quantity
        
        # Tüm kontroller geçti
        self.daily_stats['trades_count'] += 1
        log.success(f"✅ Risk Onayı: {signal.symbol} {signal.side} {signal.quantity}")
        
        return RiskCheckResult(
            passed=True,
            adjusted_quantity=signal.quantity,
            reason="Tüm risk kontrolleri başarılı ✅",
            timestamp=datetime.now()
        )
    
    def _check_trading_limits(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Temel ticaret limitleri:
        - Günlük işlem sayısı
        - Açık pozisyon sayısı
        """
        # 1. Günlük işlem limiti
        if self.daily_stats['trades_count'] >= self.config.max_daily_trades:
            return RiskCheckResult(
                passed=False,
                adjusted_quantity=0,
                reason=f"❌ Günlük işlem limiti aşıldı ({self.config.max_daily_trades})"
            )
        
        # 2. Açık pozisyon limiti (sadece BUY için)
        if signal.side == Side.BUY:
            if portfolio.open_positions_count >= self.config.max_open_positions:
                return RiskCheckResult(
                    passed=False,
                    adjusted_quantity=0,
                    reason=f"❌ Max açık pozisyon limitine ulaşıldı ({self.config.max_open_positions})"
                )
        
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_daily_loss_limit(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Günlük zarar limiti kontrolü (Circuit Breaker).
        """
        start_balance = self.daily_stats['start_balance']
        if start_balance == 0:
            return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
        
        # Günlük PnL yüzdesi
        daily_pnl_pct = (portfolio.daily_pnl / start_balance) * 100
        
        # Günlük zarar limiti aşıldı mı?
        if daily_pnl_pct < -self.config.max_daily_loss_pct:
            return RiskCheckResult(
                passed=False,
                adjusted_quantity=0,
                reason=f"❌ CIRCUIT BREAKER: Günlük zarar limiti aşıldı ({daily_pnl_pct:.2f}% < -{self.config.max_daily_loss_pct}%)"
            )
        
        # Uyarı seviyesi (%75'ine yaklaştı)
        warning_threshold = self.config.max_daily_loss_pct * 0.75
        if abs(daily_pnl_pct) > warning_threshold:
            log.warning(f"⚠️ Günlük zarar limitine yaklaşıldı: {daily_pnl_pct:.2f}%")
        
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_cash_availability(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Nakit yeterliliği ve likidite rezervi kontrolü.
        """
        if signal.side != Side.BUY:
            return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
        
        required_capital = Decimal(str(signal.price)) * Decimal(str(signal.quantity))
        required_capital = float(required_capital)
        
        # 1. Yetersiz nakit kontrolü
        if required_capital > portfolio.cash_balance:
            # Mevcut nakitle alabileceğimiz maksimum miktarı hesapla
            # %1 komisyon payı bırak
            available_capital = portfolio.cash_balance * 0.99
            max_quantity = int(available_capital / signal.price)
            
            if max_quantity < 1:
                return RiskCheckResult(
                    passed=False,
                    adjusted_quantity=0,
                    reason=f"❌ Yetersiz nakit (Gerekli: ${required_capital:,.2f}, Mevcut: ${portfolio.cash_balance:,.2f})"
                )
            
            log.warning(f"⚠️ Nakit yetersizliği - Miktar düşürüldü: {signal.quantity} → {max_quantity}")
            return RiskCheckResult(
                passed=True,
                adjusted_quantity=max_quantity,
                reason=f"Nakit sınırlaması: Miktar {signal.quantity} → {max_quantity}"
            )
        
        # 2. Minimum nakit rezervi kontrolü
        cash_after_trade = portfolio.cash_balance - required_capital
        min_cash_reserve = portfolio.total_balance * (self.config.min_cash_reserve_pct / 100)
        
        if cash_after_trade < min_cash_reserve:
            # İşlem sonrası min rezervi koruyacak şekilde miktarı düşür
            max_spendable = portfolio.cash_balance - min_cash_reserve
            max_quantity = int(max_spendable / signal.price)
            
            if max_quantity < 1:
                return RiskCheckResult(
                    passed=False,
                    adjusted_quantity=0,
                    reason=f"❌ Min nakit rezervi korunamıyor (%{self.config.min_cash_reserve_pct})"
                )
            
            log.warning(f"⚠️ Likidite rezervi korunuyor - Miktar: {signal.quantity} → {max_quantity}")
            return RiskCheckResult(
                passed=True,
                adjusted_quantity=max_quantity,
                reason="Likidite rezervi korundu"
            )
        
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_position_sizing(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Pozisyon büyüklüğü limitleri:
        - USD bazlı limit
        - Portföy yüzde bazlı limit
        """
        if signal.side != Side.BUY:
            return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
        
        required_capital = signal.price * signal.quantity
        
        # 1. Mutlak USD limiti
        if required_capital > self.config.max_position_size_usd:
            max_quantity = int(self.config.max_position_size_usd / signal.price)
            log.warning(f"⚠️ USD pozisyon limiti - Miktar: {signal.quantity} → {max_quantity}")
            return RiskCheckResult(
                passed=True,
                adjusted_quantity=max_quantity,
                reason=f"USD limiti (${self.config.max_position_size_usd:,.0f})"
            )
        
        # 2. Portföy yüzde bazlı limit
        max_allowed_capital = portfolio.total_balance * (self.config.max_position_size_pct / 100)
        
        if required_capital > max_allowed_capital:
            max_quantity = int(max_allowed_capital / signal.price)
            log.warning(f"⚠️ Portföy %{self.config.max_position_size_pct} limiti - Miktar: {signal.quantity} → {max_quantity}")
            return RiskCheckResult(
                passed=True,
                adjusted_quantity=max_quantity,
                reason=f"Portföy yüzde limiti (%{self.config.max_position_size_pct})"
            )
        
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_concentration_risk(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Yoğunlaşma riski kontrolü:
        - Tek sektöre fazla yatırım yapılmasını engelle
        - Yüksek korelasyonlu varlıklarda limit
        """
        # Basitleştirilmiş - Gerçek implementasyon sektör bilgisi gerektirir
        # Şimdilik tüm kontrolleri geçir
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_liquidity(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Likidite kontrolü:
        - Portföyde yeterli likidite var mı?
        - Acil çıkış yapılabilir mi?
        """
        liquidity_ratio = portfolio.cash_balance / portfolio.total_balance
        
        if liquidity_ratio < (self.config.min_cash_reserve_pct / 100):
            return RiskCheckResult(
                passed=False,
                adjusted_quantity=0,
                reason=f"❌ Düşük likidite: %{liquidity_ratio*100:.1f} (Min: %{self.config.min_cash_reserve_pct})"
            )
        
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_portfolio_risk(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Portföy seviyesi risk metrikleri:
        - Value at Risk (VaR)
        - Maximum Drawdown
        - Leverage
        """
        # VaR kontrolü
        if len(self.historical_returns) >= 30:
            returns = np.array(self.historical_returns[-30:])
            var_95 = np.percentile(returns, 5) * portfolio.total_balance
            
            if abs(var_95) > self.config.max_var_1d:
                log.warning(f"⚠️ VaR limiti yaklaşıldı: ${abs(var_95):,.0f} / ${self.config.max_var_1d:,.0f}")
                
                # Pozisyon büyüklüğünü %50 azalt
                reduced_quantity = max(1, int(signal.quantity * 0.5))
                return RiskCheckResult(
                    passed=True,
                    adjusted_quantity=reduced_quantity,
                    reason="VaR limiti nedeniyle pozisyon küçültüldü"
                )
        
        # Max Drawdown kontrolü
        if self.daily_stats['peak_balance'] > 0:
            current_dd = (portfolio.total_balance / self.daily_stats['peak_balance'] - 1) * 100
            self.daily_stats['current_drawdown'] = current_dd
            
            if abs(current_dd) > self.config.max_total_drawdown_pct:
                return RiskCheckResult(
                    passed=False,
                    adjusted_quantity=0,
                    reason=f"❌ Max Drawdown limiti aşıldı ({abs(current_dd):.2f}% > {self.config.max_total_drawdown_pct}%)"
                )
        
        # Peak balance güncelle
        if portfolio.total_balance > self.daily_stats['peak_balance']:
            self.daily_stats['peak_balance'] = portfolio.total_balance
        
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def _check_market_conditions(
        self, 
        signal: TradeSignal, 
        portfolio: PortfolioState,
        market_data: Optional[Dict]
    ) -> RiskCheckResult:
        """
        Piyasa koşulları kontrolü:
        - Volatilite
        - Gap riski
        - Market hours
        """
        # Basitleştirilmiş - Gelişmiş implementasyon piyasa verisi gerektirir
        return RiskCheckResult(passed=True, adjusted_quantity=signal.quantity, reason="OK")
    
    def update_historical_data(self, daily_return: float, equity: float):
        """
        Geçmiş veri güncelleme (risk metrikleri için).
        
        Args:
            daily_return: Günlük getiri (decimal, örn: 0.02 = %2)
            equity: Toplam portföy değeri
        """
        self.historical_returns.append(daily_return)
        self.historical_equity.append(equity)
        
        # Son 252 günü sakla (1 yıl)
        if len(self.historical_returns) > 252:
            self.historical_returns.pop(0)
            self.historical_equity.pop(0)
    
    def calculate_risk_metrics(self, portfolio: PortfolioState) -> RiskMetrics:
        """
        Detaylı risk metriklerini hesaplar.
        """
        if len(self.historical_returns) < 30:
            return RiskMetrics(
                var_1d=0, cvar_1d=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, current_drawdown=0, leverage=0,
                concentration_score=0, liquidity_score=100
            )
        
        returns = np.array(self.historical_returns[-30:])
        
        # VaR ve CVaR
        var_95 = np.percentile(returns, 5) * portfolio.total_balance
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * portfolio.total_balance
        
        # Sharpe Ratio (annualized)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
        sortino = (np.mean(returns) / downside_std) * np.sqrt(252)
        
        # Max Drawdown
        equity_curve = np.array(self.historical_equity[-30:])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve / running_max - 1)
        max_dd = np.min(drawdown) * 100
        
        # Current Drawdown
        current_dd = self.daily_stats['current_drawdown']
        
        # Leverage
        leverage = (portfolio.total_balance - portfolio.cash_balance) / portfolio.total_balance
        
        # Concentration Score (basitleştirilmiş)
        concentration = (1 - portfolio.cash_balance / portfolio.total_balance) * 100
        
        # Liquidity Score
        liquidity = (portfolio.cash_balance / portfolio.total_balance) * 100
        
        return RiskMetrics(
            var_1d=var_95,
            cvar_1d=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            leverage=leverage,
            concentration_score=concentration,
            liquidity_score=liquidity
        )
    
    def get_risk_report(self, portfolio: PortfolioState) -> str:
        """Detaylı risk raporu oluşturur"""
        metrics = self.calculate_risk_metrics(portfolio)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              🛡️  KURUMSAL RİSK YÖNETİM RAPORU                ║
╠══════════════════════════════════════════════════════════════╣
║  PORTFÖY RİSK METRİKLERİ                                     ║
║  ─────────────────────────────────────────────────────────   ║
║  Value at Risk (95%)    : ${metrics.var_1d:>12,.2f}          ║
║  CVaR (Expected Short.) : ${metrics.cvar_1d:>12,.2f}         ║
║  Sharpe Ratio           : {metrics.sharpe_ratio:>12.3f}      ║
║  Sortino Ratio          : {metrics.sortino_ratio:>12.3f}     ║
║  Max Drawdown (30d)     : {metrics.max_drawdown:>11.2f}%     ║
║  Current Drawdown       : {metrics.current_drawdown:>11.2f}% ║
║                                                               ║
║  PORTFÖY YAPISI                                              ║
║  ─────────────────────────────────────────────────────────   ║
║  Leverage               : {metrics.leverage:>11.2f}x         ║
║  Concentration Score    : {metrics.concentration_score:>11.1f}/100 ║
║  Liquidity Score        : {metrics.liquidity_score:>11.1f}/100    ║
║                                                               ║
║  GÜNLÜK İSTATİSTİKLER                                        ║
║  ─────────────────────────────────────────────────────────   ║
║  Toplam İşlem           : {self.daily_stats['trades_count']:>12} ║
║  Reddedilen İşlem       : {self.daily_stats['rejected_trades']:>12} ║
║  Açık Pozisyon          : {portfolio.open_positions_count:>12} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
    
    def reset_daily_stats(self):
        """Günlük istatistikleri sıfırla (Yeni gün başlangıcında)"""
        self.daily_stats = {
            'trades_count': 0,
            'rejected_trades': 0,
            'start_balance': 0.0,
            'peak_balance': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown_today': 0.0
        }
        log.info("🔄 Günlük risk istatistikleri sıfırlandı")