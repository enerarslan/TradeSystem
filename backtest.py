"""
KURUMSAL SEVİYE BACKTEST MOTORU
JPMorgan Quantitative Research Division Tarzı

Özellikler:
- Detaylı performans analizi
- Tick-by-tick simülasyon
- Komisyon ve slippage hesabı
- Sharpe, Sortino, Calmar ratio
- Drawdown analizi
- Trade-level analytics
- Görselleştirme hazır raporlama
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

from utils.logger import log
from data.csv_loader import LocalCSVLoader
from strategies.momentum import AdvancedMomentum
from risk.core import EnterpriseRiskManager, RiskLimitConfig
from execution.portfolio import PortfolioManager
from data.models import Side


@dataclass
class TradeRecord:
    """Tek bir işlemin detaylı kaydı"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    strategy_name: str


@dataclass
class BacktestMetrics:
    """Backtest sonuç metrikleri"""
    # Return metrics
    total_return: float
    annualized_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    
    # Position metrics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: float
    
    # Portfolio metrics
    final_balance: float
    peak_balance: float
    volatility: float
    
    # Performance metrics
    roi: float
    cagr: float


class ProfessionalBacktester:
    """
    Profesyonel seviye backtest motoru.
    
    Gerçekçi simülasyon için:
    - Komisyon modeli (%0.1 varsayılan)
    - Slippage modeli (volatiliteye göre)
    - Gerçek zaman ilerlemesi
    - Risk yönetimi entegrasyonu
    """
    
    def __init__(
        self,
        symbol: str,
        initial_capital: float = 100_000,
        commission_pct: float = 0.001,  # %0.1 komisyon
        slippage_pct: float = 0.0005,   # %0.05 slippage
        use_risk_management: bool = True
    ):
        """
        Args:
            symbol: Test edilecek sembol
            initial_capital: Başlangıç sermayesi
            commission_pct: İşlem komisyonu (decimal)
            slippage_pct: Ortalama slippage (decimal)
            use_risk_management: Risk yönetimi aktif mi?
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.use_risk_management = use_risk_management
        
        # Modüller
        self.loader = LocalCSVLoader(
            validate_data=True,
            interpolate_missing=True,
            remove_outliers=True
        )
        
        self.portfolio = PortfolioManager(initial_balance=initial_capital)
        
        self.risk_manager = EnterpriseRiskManager(
            config=RiskLimitConfig()
        ) if use_risk_management else None
        
        # İşlem kayıtları
        self.trade_records: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Performans tracking
        self.stats = {
            'ticks_processed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_rejected': 0
        }
    
    async def run(
        self,
        strategy_class = AdvancedMomentum,
        strategy_params: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestMetrics:
        """
        Ana backtest çalıştırma fonksiyonu.
        
        Args:
            strategy_class: Strateji sınıfı
            strategy_params: Strateji parametreleri
            start_date: Başlangıç tarihi (filtreleme için)
            end_date: Bitiş tarihi (filtreleme için)
        
        Returns:
            BacktestMetrics: Detaylı performans metrikleri
        """
        log.info("="*70)
        log.info("   🏦 KURUMSAL BACKTEST MOTORU - BAŞLATILIYOR")
        log.info("="*70)
        log.info(f"   Sembol          : {self.symbol}")
        log.info(f"   Başlangıç Sermayesi : ${self.initial_capital:,.2f}")
        log.info(f"   Komisyon        : %{self.commission_pct*100:.3f}")
        log.info(f"   Slippage        : %{self.slippage_pct*100:.3f}")
        log.info(f"   Risk Yönetimi   : {'Aktif ✅' if self.use_risk_management else 'Devre Dışı ❌'}")
        log.info("="*70 + "\n")
        
        start_time = time.time()
        
        # 1. Veriyi yükle
        log.info("📂 Geçmiş veriler yükleniyor...")
        ticks = self.loader.load_data(
            self.symbol, 
            use_cache=True,
            start_date=start_date,
            end_date=end_date
        )
        
        if not ticks:
            log.error("❌ Veri yüklenemedi!")
            return None
        
        log.info(f"✅ {len(ticks):,} adet bar yüklendi\n")
        
        # 2. Stratejiyi başlat
        params = strategy_params or {}
        strategy = strategy_class(symbol=self.symbol, **params)
        log.info(f"🎯 Strateji: {strategy.name}")
        log.info(f"   Parametreler: {params}\n")
        
        # 3. Backtest döngüsü
        log.info("⚡ Backtest simülasyonu başlıyor...\n")
        log.info(f"{'─'*70}")
        
        for i, tick in enumerate(ticks):
            self.stats['ticks_processed'] += 1
            
            # Progress indicator (her 1000 tick'te bir)
            if i % 1000 == 0 and i > 0:
                progress = (i / len(ticks)) * 100
                current_equity = self.portfolio.get_total_equity()
                pnl = current_equity - self.initial_capital
                print(f"\r⚡ İlerleme: {progress:5.1f}% | Varlık: ${current_equity:>12,.2f} | PnL: ${pnl:>+10,.2f}", end="", flush=True)
            
            # Mark-to-Market güncelleme
            self.portfolio.update_price(tick.symbol, tick.price)
            
            # Strateji sinyali
            signal = await strategy.on_tick(tick)
            
            if signal:
                self.stats['signals_generated'] += 1
                
                # Risk kontrolü
                if self.risk_manager:
                    portfolio_state = self.portfolio.get_state()
                    risk_result = self.risk_manager.analyze_signal(signal, portfolio_state)
                    
                    if not risk_result.passed:
                        self.stats['trades_rejected'] += 1
                        continue
                    
                    # Miktarı düzelt
                    approved_quantity = int(risk_result.adjusted_quantity)
                else:
                    approved_quantity = int(signal.quantity)
                
                if approved_quantity < 1:
                    continue
                
                # İşlemi gerçekleştir
                self._execute_trade(
                    symbol=signal.symbol,
                    side=signal.side,
                    price=tick.price,
                    quantity=approved_quantity,
                    timestamp=tick.timestamp,
                    strategy_name=strategy.name
                )
            
            # Equity curve kaydet (her 100 tick'te bir)
            if i % 100 == 0:
                self.equity_curve.append((tick.timestamp, self.portfolio.get_total_equity()))
        
        print()  # Progress bar'dan sonra newline
        
        # 4. Sonuç analizi
        elapsed_time = time.time() - start_time
        metrics = self._calculate_metrics(ticks[0].timestamp, ticks[-1].timestamp)
        
        log.info(f"\n{'─'*70}")
        log.success(f"✅ Backtest tamamlandı! (Süre: {elapsed_time:.2f}s)")
        log.info(f"{'─'*70}\n")
        
        # 5. Rapor
        self._print_report(metrics, ticks[0].timestamp, ticks[-1].timestamp)
        
        # 6. Risk raporu
        if self.risk_manager:
            print(self.risk_manager.get_risk_report(self.portfolio.get_state()))
        
        return metrics
    
    def _execute_trade(
        self,
        symbol: str,
        side: Side,
        price: float,
        quantity: float,
        timestamp: datetime,
        strategy_name: str
    ):
        """
        İşlemi gerçekleştirir ve kaydeder (komisyon ve slippage ile).
        """
        # Slippage hesapla (volatiliteye göre random)
        slippage = price * self.slippage_pct * np.random.uniform(0.5, 1.5)
        
        # Execution price (BUY için yüksek, SELL için düşük)
        if side == Side.BUY:
            exec_price = price + slippage
        else:
            exec_price = price - slippage
        
        # Komisyon hesapla
        commission = exec_price * quantity * self.commission_pct
        
        # Portfolio güncelle
        self.portfolio.update_after_trade(
            symbol=symbol,
            quantity=quantity,
            price=exec_price,
            side=side
        )
        
        # Trade record oluştur (çıkış için placeholder)
        if side == Side.BUY:
            # Yeni pozisyon açıldı
            trade = TradeRecord(
                entry_time=timestamp,
                exit_time=None,
                symbol=symbol,
                side="LONG",
                entry_price=exec_price,
                exit_price=None,
                quantity=quantity,
                commission=commission,
                slippage=slippage,
                pnl=0,
                pnl_pct=0,
                holding_period_hours=0,
                strategy_name=strategy_name
            )
            self.trade_records.append(trade)
        
        elif side == Side.SELL:
            # Pozisyon kapatıldı - son trade'i güncelle
            if self.trade_records:
                last_trade = self.trade_records[-1]
                if last_trade.exit_time is None:  # Henüz kapatılmamış
                    last_trade.exit_time = timestamp
                    last_trade.exit_price = exec_price
                    last_trade.commission += commission
                    last_trade.slippage += slippage
                    
                    # PnL hesapla
                    gross_pnl = (exec_price - last_trade.entry_price) * quantity
                    last_trade.pnl = gross_pnl - last_trade.commission
                    last_trade.pnl_pct = (last_trade.pnl / (last_trade.entry_price * quantity)) * 100
                    
                    # Holding period
                    holding_period = (timestamp - last_trade.entry_time).total_seconds() / 3600
                    last_trade.holding_period_hours = holding_period
        
        self.stats['trades_executed'] += 1
    
    def _calculate_metrics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> BacktestMetrics:
        """
        Detaylı performans metriklerini hesaplar.
        """
        # Equity curve'den returns hesapla
        equity_values = [eq for _, eq in self.equity_curve]
        equity_array = np.array(equity_values)
        
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns)]  # NaN'ları temizle
        
        # Final balance
        final_balance = self.portfolio.get_total_equity()
        peak_balance = np.max(equity_array)
        
        # Total return
        total_return = ((final_balance / self.initial_capital) - 1) * 100
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (((final_balance / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 26)  # 15-min bars
        else:
            sharpe = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino = (np.mean(returns) / downside_std) * np.sqrt(252 * 26) if downside_std > 0 else 0
        else:
            sortino = 0
        
        # Drawdown analizi
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array / running_max - 1)
        max_drawdown = np.min(drawdown) * 100
        avg_drawdown = np.mean(drawdown[drawdown < 0]) * 100 if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Calmar ratio
        calmar = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        completed_trades = [t for t in self.trade_records if t.exit_time is not None]
        
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        losing_trades = len([t for t in completed_trades if t.pnl < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum([t.pnl for t in completed_trades if t.pnl > 0])
        gross_loss = abs(sum([t.pnl for t in completed_trades if t.pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Win/Loss stats
        wins = [t.pnl for t in completed_trades if t.pnl > 0]
        losses = [t.pnl for t in completed_trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Avg holding period
        holding_periods = [t.holding_period_hours for t in completed_trades]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252 * 26) * 100 if len(returns) > 1 else 0
        
        # ROI
        roi = total_return
        
        # CAGR (Compound Annual Growth Rate)
        cagr = annualized_return
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_period=avg_holding_period,
            final_balance=final_balance,
            peak_balance=peak_balance,
            volatility=volatility,
            roi=roi,
            cagr=cagr
        )
    
    def _print_report(
        self, 
        metrics: BacktestMetrics, 
        start_date: datetime,
        end_date: datetime
    ):
        """Detaylı backtest raporunu yazdırır"""
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*15 + "📊 BACKTEST PERFORMANS RAPORU" + " "*24 + "║")
        print("╠" + "═"*68 + "╣")
        
        # Genel Bilgiler
        print("║  📅 GENEL BİLGİLER" + " "*49 + "║")
        print("║  " + "─"*66 + "║")
        print(f"║  Sembol               : {self.symbol:<45} ║")
        print(f"║  Tarih Aralığı        : {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d'):<22} ║")
        print(f"║  Test Süresi          : {(end_date - start_date).days} gün{' '*36} ║")
        print(f"║  Başlangıç Sermayesi  : ${self.initial_capital:>12,.2f}{' '*29} ║")
        print(f"║  Bitiş Sermayesi      : ${metrics.final_balance:>12,.2f}{' '*29} ║")
        print(f"║  Peak Balance         : ${metrics.peak_balance:>12,.2f}{' '*29} ║")
        
        # Getiri Metrikleri
        print("║" + " "*68 + "║")
        print("║  💰 GETİRİ METRİKLERİ" + " "*46 + "║")
        print("║  " + "─"*66 + "║")
        
        return_color = "+" if metrics.total_return >= 0 else "-"
        print(f"║  Toplam Getiri        : {return_color}{abs(metrics.total_return):>11.2f}%{' '*29} ║")
        print(f"║  Yıllık Getiri (CAGR) : {return_color}{abs(metrics.cagr):>11.2f}%{' '*29} ║")
        print(f"║  ROI                  : {return_color}{abs(metrics.roi):>11.2f}%{' '*29} ║")
        print(f"║  Net PnL              : ${metrics.final_balance - self.initial_capital:>+12,.2f}{' '*29} ║")
        
        # Risk Metrikleri
        print("║" + " "*68 + "║")
        print("║  ⚠️  RİSK METRİKLERİ" + " "*47 + "║")
        print("║  " + "─"*66 + "║")
        print(f"║  Sharpe Ratio         : {metrics.sharpe_ratio:>12.3f}{' '*33} ║")
        print(f"║  Sortino Ratio        : {metrics.sortino_ratio:>12.3f}{' '*33} ║")
        print(f"║  Calmar Ratio         : {metrics.calmar_ratio:>12.3f}{' '*33} ║")
        print(f"║  Max Drawdown         : {metrics.max_drawdown:>11.2f}%{' '*33} ║")
        print(f"║  Avg Drawdown         : {metrics.avg_drawdown:>11.2f}%{' '*33} ║")
        print(f"║  Volatility (Yıllık)  : {metrics.volatility:>11.2f}%{' '*33} ║")
        
        # İşlem Metrikleri
        print("║" + " "*68 + "║")
        print("║  📈 İŞLEM METRİKLERİ" + " "*47 + "║")
        print("║  " + "─"*66 + "║")
        print(f"║  Toplam İşlem         : {metrics.total_trades:>12}{' '*37} ║")
        print(f"║  Kazanan İşlem        : {metrics.winning_trades:>12} ({metrics.win_rate:>5.1f}%){' '*24} ║")
        print(f"║  Kaybeden İşlem       : {metrics.losing_trades:>12}{' '*37} ║")
        print(f"║  Win Rate             : {metrics.win_rate:>11.2f}%{' '*33} ║")
        print(f"║  Profit Factor        : {metrics.profit_factor:>12.2f}{' '*33} ║")
        
        # Pozisyon Metrikleri
        print("║" + " "*68 + "║")
        print("║  📊 POZİSYON METRİKLERİ" + " "*44 + "║")
        print("║  " + "─"*66 + "║")
        print(f"║  Ort. Kazanç          : ${metrics.avg_win:>12,.2f}{' '*30} ║")
        print(f"║  Ort. Kayıp           : ${metrics.avg_loss:>12,.2f}{' '*30} ║")
        print(f"║  En Büyük Kazanç      : ${metrics.largest_win:>12,.2f}{' '*30} ║")
        print(f"║  En Büyük Kayıp       : ${metrics.largest_loss:>12,.2f}{' '*30} ║")
        print(f"║  Ort. Holding Period  : {metrics.avg_holding_period:>11.1f} saat{' '*29} ║")
        
        # İstatistikler
        print("║" + " "*68 + "║")
        print("║  📋 SİSTEM İSTATİSTİKLERİ" + " "*42 + "║")
        print("║  " + "─"*66 + "║")
        print(f"║  İşlenen Tick         : {self.stats['ticks_processed']:>12,}{' '*33} ║")
        print(f"║  Üretilen Sinyal      : {self.stats['signals_generated']:>12,}{' '*33} ║")
        print(f"║  Gerçekleşen İşlem    : {self.stats['trades_executed']:>12,}{' '*33} ║")
        print(f"║  Reddedilen İşlem     : {self.stats['trades_rejected']:>12,}{' '*33} ║")
        
        print("╚" + "═"*68 + "╝\n")
        
        # Performans değerlendirmesi
        self._print_performance_rating(metrics)
    
    def _print_performance_rating(self, metrics: BacktestMetrics):
        """Performans değerlendirmesi yapar"""
        print("═"*70)
        print("   🎯 PERFORMANS DEĞERLENDİRMESİ")
        print("═"*70)
        
        score = 0
        max_score = 100
        
        # Getiri (30 puan)
        if metrics.total_return > 20:
            score += 30
            print("   ✅ Getiri: Mükemmel (>%20)")
        elif metrics.total_return > 10:
            score += 20
            print("   ✅ Getiri: İyi (>%10)")
        elif metrics.total_return > 0:
            score += 10
            print("   ⚠️  Getiri: Orta (Pozitif)")
        else:
            print("   ❌ Getiri: Zayıf (Negatif)")
        
        # Sharpe Ratio (25 puan)
        if metrics.sharpe_ratio > 2.0:
            score += 25
            print("   ✅ Sharpe Ratio: Mükemmel (>2.0)")
        elif metrics.sharpe_ratio > 1.0:
            score += 15
            print("   ✅ Sharpe Ratio: İyi (>1.0)")
        elif metrics.sharpe_ratio > 0.5:
            score += 10
            print("   ⚠️  Sharpe Ratio: Orta (>0.5)")
        else:
            print("   ❌ Sharpe Ratio: Zayıf (<0.5)")
        
        # Max Drawdown (25 puan)
        if abs(metrics.max_drawdown) < 10:
            score += 25
            print("   ✅ Drawdown: Mükemmel (<%10)")
        elif abs(metrics.max_drawdown) < 20:
            score += 15
            print("   ✅ Drawdown: İyi (<%20)")
        elif abs(metrics.max_drawdown) < 30:
            score += 10
            print("   ⚠️  Drawdown: Orta (<%30)")
        else:
            print("   ❌ Drawdown: Yüksek (>%30)")
        
        # Win Rate (20 puan)
        if metrics.win_rate > 60:
            score += 20
            print("   ✅ Win Rate: Mükemmel (>%60)")
        elif metrics.win_rate > 50:
            score += 15
            print("   ✅ Win Rate: İyi (>%50)")
        elif metrics.win_rate > 40:
            score += 10
            print("   ⚠️  Win Rate: Orta (>%40)")
        else:
            print("   ❌ Win Rate: Düşük (<40)")
        
        print("─"*70)
        print(f"   📊 GENEL SKOR: {score}/{max_score}")
        
        if score >= 80:
            print("   🏆 Sonuç: MÜKEMMEL - Canlı trading için uygun!")
        elif score >= 60:
            print("   ✅ Sonuç: İYİ - İyileştirmeler yapılabilir")
        elif score >= 40:
            print("   ⚠️  Sonuç: ORTA - Strateji optimize edilmeli")
        else:
            print("   ❌ Sonuç: ZAYIF - Strateji değiştirilmeli")
        
        print("═"*70 + "\n")
    
    def export_results(self, filename: str = "backtest_results.csv"):
        """Sonuçları CSV'ye export eder"""
        if not self.trade_records:
            log.warning("Export edilecek işlem kaydı yok")
            return
        
        # Trade records to DataFrame
        trades_data = []
        for trade in self.trade_records:
            if trade.exit_time:  # Sadece tamamlanmış işlemler
                trades_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'holding_hours': trade.holding_period_hours,
                    'commission': trade.commission,
                    'slippage': trade.slippage
                })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(filename, index=False)
        log.success(f"📁 Sonuçlar kaydedildi: {filename}")



async def main():
    """
    UPDATED MAIN - Optimize edilmiş risk ile test
    """
    from risk.optimized_configs import RiskProfiles
    
    log.info("="*70)
    log.info("   🎯 OPTİMİZE EDİLMİŞ BACKTEST")
    log.info("="*70 + "\n")
    
    # Test 1: AAPL - MODERATE RISK (ÖNERİLEN)
    log.info("📊 Test 1: AAPL - MODERATE Risk Profili")
    backtester = ProfessionalBacktester(
        symbol="AAPL",
        initial_capital=100_000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        use_risk_management=True
    )
    
    # Risk profilini değiştir
    backtester.risk_manager.config = RiskProfiles.MODERATE
    
    metrics = await backtester.run(
        strategy_class=AdvancedMomentum,
        strategy_params={
            'fast_period': 10,
            'slow_period': 30,
            'min_confidence': 0.5  # Daha düşük (daha fazla işlem)
        }
    )
    
    if metrics:
        log.success(f"✅ CAGR: {metrics.cagr:.2f}%, Sharpe: {metrics.sharpe_ratio:.3f}")


if __name__ == "__main__":
    asyncio.run(main())