"""
KURUMSAL TİCARET SİSTEMİ - ANA PROGRAM
JPMorgan Algorithmic Trading Division Tarzı

Production-Ready Özellikler:
- Graceful startup/shutdown
- Error handling & recovery
- Performance monitoring
- Real-time dashboard
- Logging
- Configuration management
"""

import asyncio
import signal
import sys
from typing import Optional
from datetime import datetime

from utils.logger import log
from config.settings import settings
from data.feed import DataStream
from data.db import init_db
from strategies.momentum import AdvancedMomentum
from risk.core import EnterpriseRiskManager, RiskLimitConfig
from execution.portfolio import PortfolioManager
from execution.handler import ExecutionHandler


class TradingSystem:
    """
    Ana ticaret sistemi orchestrator'ı.
    
    Tüm bileşenleri koordine eder ve yaşam döngüsünü yönetir.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Sistem konfigürasyonu
        """
        self.config = config
        self.running = False
        self.shutdown_initiated = False
        
        # Core components
        self.portfolio: Optional[PortfolioManager] = None
        self.risk_manager: Optional[EnterpriseRiskManager] = None
        self.execution_handler: Optional[ExecutionHandler] = None
        self.data_stream: Optional[DataStream] = None
        self.strategies: dict = {}
        
        # Performance tracking
        self.start_time = None
        self.stats = {
            'uptime_seconds': 0,
            'total_ticks': 0,
            'total_signals': 0,
            'total_trades': 0,
            'errors': 0
        }
        
        log.info("🏦 Trading System başlatıldı")
    
    async def initialize(self):
        """
        Sistem başlatma rutini.
        
        1. Database initialization
        2. Component initialization
        3. Health checks
        4. Strategy loading
        """
        log.info("="*70)
        log.info("   🚀 KURUMSAL TİCARET SİSTEMİ - BAŞLATILIYOR")
        log.info("="*70)
        log.info(f"   Mod: {settings.APP_MODE}")
        log.info(f"   Versiyon: {settings.VERSION}")
        log.info(f"   Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("="*70 + "\n")
        
        try:
            # 1. Database
            log.info("📊 Veritabanı başlatılıyor...")
            await init_db()
            log.success("✅ Veritabanı hazır\n")
            
            # 2. Portfolio Manager
            log.info("💼 Portfolio Manager başlatılıyor...")
            initial_capital = self.config.get('initial_capital', 10_000)
            self.portfolio = PortfolioManager(initial_balance=initial_capital)
            log.success(f"✅ Portfolio hazır (Sermaye: ${initial_capital:,.2f})\n")
            
            # 3. Risk Manager
            log.info("🛡️ Risk Manager başlatılıyor...")
            risk_config = RiskLimitConfig(
                max_position_size_usd=self.config.get('max_position_size', 5_000),
                max_position_size_pct=self.config.get('max_position_pct', 10.0),
                max_daily_loss_pct=self.config.get('max_daily_loss', 2.0),
                max_daily_trades=self.config.get('max_daily_trades', 50),
                max_var_1d=self.config.get('max_var', 1_000)
            )
            self.risk_manager = EnterpriseRiskManager(config=risk_config)
            log.success("✅ Risk Manager hazır\n")
            
            # 4. Execution Handler
            log.info("⚡ Execution Handler başlatılıyor...")
            self.execution_handler = ExecutionHandler(self.portfolio)
            log.success("✅ Execution Handler hazır\n")
            
            # 5. Data Stream
            log.info("📡 Data Stream bağlantısı kuruluyor...")
            exchange = self.config.get('exchange', 'binance')
            self.data_stream = DataStream(exchange_id=exchange)
            await self.data_stream.connect()
            log.success(f"✅ {exchange.upper()} bağlantısı başarılı\n")
            
            # 6. Strategy Loading
            await self._load_strategies()
            
            # 7. Health Check
            self._health_check()
            
            self.start_time = datetime.now()
            log.success("✅ SİSTEM HAZIR!\n")
            
        except Exception as e:
            log.critical(f"❌ Başlatma hatası: {e}")
            log.exception(e)
            raise
    
    async def _load_strategies(self):
        """Stratejileri yükler"""
        log.info("🎯 Stratejiler yükleniyor...")
        
        symbols = self.config.get('symbols', ['BTC/USDT'])
        strategy_type = self.config.get('strategy_type', 'momentum')
        
        for symbol in symbols:
            if strategy_type == 'momentum':
                strategy = AdvancedMomentum(
                    symbol=symbol,
                    fast_period=self.config.get('fast_period', 10),
                    slow_period=self.config.get('slow_period', 30),
                    min_confidence=self.config.get('min_confidence', 0.6)
                )
                self.strategies[symbol] = strategy
                log.info(f"   ✅ {symbol}: Advanced Momentum yüklendi")
        
        log.success(f"✅ {len(self.strategies)} strateji hazır\n")
    
    def _health_check(self):
        """Sistem sağlık kontrolü"""
        log.info("🏥 Sistem sağlık kontrolü...")
        
        checks = {
            'Portfolio Manager': self.portfolio is not None,
            'Risk Manager': self.risk_manager is not None,
            'Execution Handler': self.execution_handler is not None,
            'Data Stream': self.data_stream is not None,
            'Strategies': len(self.strategies) > 0
        }
        
        all_healthy = all(checks.values())
        
        for component, status in checks.items():
            icon = "✅" if status else "❌"
            log.info(f"   {icon} {component}")
        
        if not all_healthy:
            raise SystemError("❌ Sistem sağlık kontrolü başarısız!")
        
        log.success("✅ Tüm bileşenler sağlıklı\n")
    
    async def run(self):
        """
        Ana ticaret döngüsü.
        
        Sürekli çalışır ve şunları yapar:
        1. Veri akışını dinle
        2. Strateji sinyallerini işle
        3. Risk kontrolü yap
        4. İşlemleri gerçekleştir
        5. Performance tracking
        """
        self.running = True
        
        log.info("="*70)
        log.info("   🔥 TİCARET DÖNGÜSÜ BAŞLIYOR")
        log.info("="*70)
        log.info("   Durdurmak için Ctrl+C kullanın")
        log.info("="*70 + "\n")
        
        # Initial portfolio state
        state = self.portfolio.get_state()
        log.info(f"💰 Başlangıç Bakiyesi: ${state.total_balance:,.2f}")
        log.info(f"💵 Nakit: ${state.cash_balance:,.2f}\n")
        
        last_status_time = datetime.now()
        status_interval = 60  # Her 60 saniyede bir durum raporu
        
        try:
            while self.running:
                # Her sembol için işlem yap
                for symbol, strategy in self.strategies.items():
                    try:
                        # Veri al
                        tick = await self.data_stream.get_latest_price(symbol)
                        
                        if not tick:
                            continue
                        
                        self.stats['total_ticks'] += 1
                        
                        # Portfolio'yu güncelle (Mark-to-Market)
                        self.portfolio.update_price(tick.symbol, tick.price)
                        
                        # Strateji sinyali
                        signal = await strategy.on_tick(tick)
                        
                        if signal:
                            self.stats['total_signals'] += 1
                            log.info(f"📊 Sinyal: {signal.side} {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                            
                            # Risk analizi
                            portfolio_state = self.portfolio.get_state()
                            risk_result = self.risk_manager.analyze_signal(
                                signal, 
                                portfolio_state
                            )
                            
                            if risk_result.passed:
                                # İşlemi gerçekleştir
                                await self.execution_handler.execute_order(
                                    signal, 
                                    risk_result.adjusted_quantity
                                )
                                self.stats['total_trades'] += 1
                                
                                # Güncellenmiş bakiye
                                new_state = self.portfolio.get_state()
                                log.success(f"✅ İşlem başarılı!")
                                log.info(f"💰 Yeni Bakiye: ${new_state.total_balance:,.2f} (PnL: ${new_state.daily_pnl:+,.2f})\n")
                            else:
                                log.warning(f"⚠️ Risk reddetti: {risk_result.reason}\n")
                    
                    except Exception as e:
                        self.stats['errors'] += 1
                        log.error(f"❌ İşlem hatası ({symbol}): {e}")
                        # Devam et, crash olmasın
                
                # Periyodik durum raporu
                now = datetime.now()
                if (now - last_status_time).seconds >= status_interval:
                    self._print_status()
                    last_status_time = now
                
                # Rate limiting
                await asyncio.sleep(self.config.get('tick_interval', 1.0))
        
        except KeyboardInterrupt:
            log.warning("\n⚠️ Kullanıcı durdurma (Ctrl+C)")
        except Exception as e:
            log.critical(f"\n❌ Kritik hata: {e}")
            log.exception(e)
        finally:
            await self.shutdown()
    
    def _print_status(self):
        """Periyodik durum raporu"""
        state = self.portfolio.get_state()
        uptime = (datetime.now() - self.start_time).seconds if self.start_time else 0
        
        log.info("─"*70)
        log.info("   📊 DURUM RAPORU")
        log.info("─"*70)
        log.info(f"   Uptime         : {uptime}s")
        log.info(f"   Toplam Varlık  : ${state.total_balance:,.2f}")
        log.info(f"   Nakit          : ${state.cash_balance:,.2f}")
        log.info(f"   Günlük PnL     : ${state.daily_pnl:+,.2f}")
        log.info(f"   Açık Pozisyon  : {state.open_positions_count}")
        log.info(f"   Günlük İşlem   : {state.daily_trade_count}")
        log.info(f"   Toplam Sinyal  : {self.stats['total_signals']}")
        log.info(f"   Toplam İşlem   : {self.stats['total_trades']}")
        log.info(f"   Hatalar        : {self.stats['errors']}")
        log.info("─"*70 + "\n")
    
    async def shutdown(self):
        """
        Graceful shutdown - Sistemi güvenli şekilde kapat.
        """
        if self.shutdown_initiated:
            return
        
        self.shutdown_initiated = True
        self.running = False
        
        log.info("\n" + "="*70)
        log.warning("   🛑 SİSTEM KAPATILIYOR...")
        log.info("="*70 + "\n")
        
        try:
            # 1. Data stream'i kapat
            if self.data_stream:
                log.info("📡 Data stream kapatılıyor...")
                await self.data_stream.close()
                log.success("✅ Data stream kapatıldı\n")
            
            # 2. Açık pozisyonları kontrol et
            if self.portfolio:
                state = self.portfolio.get_state()
                if state.open_positions_count > 0:
                    log.warning(f"⚠️ DİKKAT: {state.open_positions_count} açık pozisyon var!")
                    log.warning("   Lütfen manuel olarak kapatın veya sistemin devam etmesine izin verin\n")
            
            # 3. Final rapor
            self._print_final_report()
            
            # 4. Risk raporunu yazdır
            if self.risk_manager and self.portfolio:
                print(self.risk_manager.get_risk_report(self.portfolio.get_state()))
            
            log.success("✅ Sistem güvenli şekilde kapatıldı")
            log.info("="*70 + "\n")
            
        except Exception as e:
            log.error(f"Shutdown hatası: {e}")
    
    def _print_final_report(self):
        """Final performans raporu"""
        if not self.portfolio or not self.start_time:
            return
        
        state = self.portfolio.get_state()
        initial_balance = self.config.get('initial_capital', 10_000)
        net_pnl = state.total_balance - initial_balance
        roi = (net_pnl / initial_balance) * 100
        uptime = (datetime.now() - self.start_time).seconds
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*20 + "📊 FINAL RAPOR" + " "*34 + "║")
        print("╠" + "═"*68 + "╣")
        print(f"║  Çalışma Süresi       : {uptime}s ({uptime//3600}h {(uptime%3600)//60}m){' '*20} ║")
        print(f"║  Başlangıç Sermayesi  : ${initial_balance:>12,.2f}{' '*29} ║")
        print(f"║  Bitiş Sermayesi      : ${state.total_balance:>12,.2f}{' '*29} ║")
        print(f"║  Net PnL              : ${net_pnl:>+12,.2f}{' '*29} ║")
        print(f"║  ROI                  : {roi:>+11.2f}%{' '*32} ║")
        print(f"║  Toplam İşlem         : {self.stats['total_trades']:>12}{' '*37} ║")
        print(f"║  Toplam Sinyal        : {self.stats['total_signals']:>12}{' '*37} ║")
        print(f"║  İşlem Başarı Oranı   : {(self.stats['total_trades']/max(1, self.stats['total_signals'])*100):>11.1f}%{' '*30} ║")
        print(f"║  Hatalar              : {self.stats['errors']:>12}{' '*37} ║")
        print("╚" + "═"*68 + "╝\n")


def setup_signal_handlers(system: TradingSystem):
    """
    Signal handler'ları ayarla (Ctrl+C, SIGTERM vb.)
    """
    def signal_handler(signum, frame):
        log.warning(f"\n⚠️ Signal alındı: {signum}")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """
    Ana giriş noktası.
    """
    # Konfigürasyon
    config = {
        # Portfolio
        'initial_capital': 10_000,
        
        # Trading
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'exchange': 'binance',
        'tick_interval': 1.0,  # saniye
        
        # Strategy
        'strategy_type': 'momentum',
        'fast_period': 10,
        'slow_period': 30,
        'min_confidence': 0.6,
        
        # Risk Management
        'max_position_size': 5_000,
        'max_position_pct': 10.0,
        'max_daily_loss': 2.0,
        'max_daily_trades': 50,
        'max_var': 1_000,
    }
    
    # Sistem oluştur
    system = TradingSystem(config)
    
    # Signal handlers
    setup_signal_handlers(system)
    
    # Başlat
    await system.initialize()
    
    # Çalıştır
    await system.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\n👋 Görüşmek üzere!")
    except Exception as e:
        log.critical(f"❌ Fatal error: {e}")
        sys.exit(1)