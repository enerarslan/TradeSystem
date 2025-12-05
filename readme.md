# 🏦 KURUMSAL TİCARET SİSTEMİ
## JPMorgan Seviyesinde Algoritmik Trading Platformu

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)]()

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Özellikler](#özellikler)
3. [Kurulum](#kurulum)
4. [Kullanım](#kullanım)
5. [Konfigürasyon](#konfigürasyon)
6. [Backtest Çalıştırma](#backtest-çalıştırma)
7. [Live Trading](#live-trading)
8. [Modül Açıklamaları](#modül-açıklamaları)
9. [Performans Metrikleri](#performans-metrikleri)
10. [Sorun Giderme](#sorun-giderme)

---

## 🎯 Genel Bakış

Bu sistem, kurumsal yatırım bankalarının (JPMorgan, Goldman Sachs, vb.) kullandığı seviyede profesyonel bir algoritmik trading platformudur.

### Temel Bileşenler

```
AlphaTrade/
├── data/               # Veri yönetimi
│   ├── csv_loader.py  # Gelişmiş CSV yükleyici
│   ├── feed.py        # Canlı veri akışı
│   ├── models.py      # Veri modelleri
│   └── storage/       # CSV dosyaları buraya
│
├── strategies/         # Trading stratejileri
│   ├── base.py        # Temel strateji sınıfı
│   └── momentum.py    # Gelişmiş momentum stratejisi
│
├── risk/              # Risk yönetimi
│   └── core.py        # Kurumsal risk motoru
│
├── execution/         # İşlem yürütme
│   ├── portfolio.py   # Portfolio yönetimi
│   └── handler.py     # Emir yönetimi
│
├── backtest.py        # Backtest motoru
└── main.py            # Ana program
```

---

## ✨ Özellikler

### 🎯 Strateji Motoru
- **Gelişmiş Momentum Stratejisi**: Çoklu teknik gösterge kombinasyonu
- **Dinamik Eşik Ayarlama**: Piyasa rejimine göre adaptif parametreler
- **Position Sizing**: ATR ve confidence tabanlı miktar hesaplama
- **Otomatik Stop Loss/Take Profit**: Risk/ödül optimizasyonu

### 🛡️ Risk Yönetimi
- **Çok Katmanlı Risk Kontrol**: Position, Portfolio, Market seviyesi
- **Value at Risk (VaR)**: Günlük risk limiti
- **Maximum Drawdown Control**: Otomatik circuit breaker
- **Concentration Limits**: Sektör bazlı çeşitlendirme
- **Liquidity Management**: Minimum nakit rezervi

### 📊 Veri Yönetimi
- **Otomatik Veri Temizleme**: Aykırı değer tespiti ve düzeltme
- **Eksik Veri Interpolasyonu**: Forward/backward fill
- **Veri Kalitesi Raporu**: Detaylı quality score (0-100)
- **Multi-format Support**: CSV, XLSX, XLS

### 📈 Backtest Motoru
- **Gerçekçi Simülasyon**: Komisyon ve slippage modeli
- **Detaylı Performans Analizi**: 20+ metrik
- **Trade-level Analytics**: Her işlemin detaylı kaydı
- **Equity Curve Tracking**: Zaman serisi analizi

### 💼 Portfolio Yönetimi
- **Mark-to-Market**: Anlık değerleme
- **Multi-asset Support**: Aynı anda birden fazla varlık
- **Average Cost Tracking**: FIFO/LIFO/Weighted Average
- **Realized/Unrealized PnL**: Ayrıştırılmış kar/zarar

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.10+
- pip (Python paket yöneticisi)

### 1. Repository'yi Clone

```bash
git clone https://github.com/yourusername/alphatrade.git
cd alphatrade
```

### 2. Virtual Environment Oluştur

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

**requirements.txt içeriği:**
```txt
pandas>=2.0.0
numpy>=1.24.0
asyncio
sqlalchemy>=2.0.0
aiosqlite
pydantic>=2.0.0
pydantic-settings
python-dotenv
loguru
ccxt>=4.0.0
scikit-learn>=1.3.0
joblib
```

### 4. Klasör Yapısını Oluştur

```bash
mkdir -p data/storage
mkdir -p logs
mkdir -p data/backtest_results
```

### 5. Veri Dosyalarını Ekle

CSV dosyalarını `data/storage/` klasörüne kopyalayın:

```bash
cp /path/to/your/AAPL_15min.csv data/storage/
cp /path/to/your/MSFT_15min.csv data/storage/
# ... diğer 44 hisse
```

**Beklenen CSV formatı:**
```csv
timestamp,open,high,low,close,volume
2021-01-04 09:00:00,130.28,130.28,130.25,130.25,12345
2021-01-04 09:15:00,130.25,130.30,130.20,130.28,15678
...
```

---

## 🎮 Kullanım

### Backtest Çalıştırma (Önerilen Başlangıç)

```bash
python backtest.py
```

**Çıktı örneği:**
```
═══════════════════════════════════════════════════════════════════
   🏦 KURUMSAL BACKTEST MOTORU - BAŞLATILIYOR
═══════════════════════════════════════════════════════════════════
   Sembol          : AAPL
   Başlangıç Sermayesi : $100,000.00
   Komisyon        : %0.100
   Slippage        : %0.050
   Risk Yönetimi   : Aktif ✅
═══════════════════════════════════════════════════════════════════

📂 Geçmiş veriler yükleniyor...
✅ 12,485 adet bar yüklendi

🎯 Strateji: AdvancedMomentum_V2
   Parametreler: {'fast_period': 10, 'slow_period': 30}

⚡ Backtest simülasyonu başlıyor...

⚡ İlerleme: 100.0% | Varlık: $   115,234.56 | PnL: $ +15,234.56

╔════════════════════════════════════════════════════════════════╗
║                 📊 BACKTEST PERFORMANS RAPORU                  ║
╠════════════════════════════════════════════════════════════════╣
║  💰 GETİRİ METRİKLERİ                                          ║
║  ──────────────────────────────────────────────────────────    ║
║  Toplam Getiri        : +15.23%                                ║
║  Yıllık Getiri (CAGR) : +12.45%                                ║
║  Sharpe Ratio         :   1.856                                ║
║  Max Drawdown         :  -8.34%                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Live Trading Simülasyonu

```bash
python main.py
```

**Not:** Varsayılan olarak Binance'e bağlanır (API key gerektirmez, sadece public data).

---

## ⚙️ Konfigürasyon

### config/settings.py

```python
# Risk Limitleri
MAX_DAILY_DRAWDOWN_PERCENT = 2.0      # Günlük max %2 zarar
MAX_POSITION_SIZE_PERCENT = 10.0      # Tek pozisyon max %10
MAX_TRADES_PER_DAY = 50               # Günlük max işlem
DEFAULT_STOP_LOSS_PERCENT = 0.02      # %2 stop loss
```

### Strateji Parametreleri (backtest.py veya main.py içinde)

```python
config = {
    'initial_capital': 100_000,
    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
    
    # Strateji
    'strategy_type': 'momentum',
    'fast_period': 10,
    'slow_period': 30,
    'min_confidence': 0.6,  # Minimum sinyal güveni
    
    # Risk
    'max_position_size': 50_000,
    'max_daily_loss': 2.0,
    'max_var': 20_000,
}
```

---

## 📊 Backtest Çalıştırma (Detaylı)

### 1. Tek Sembol Backtest

```python
# backtest.py
from backtest import ProfessionalBacktester
from strategies.momentum import AdvancedMomentum

async def main():
    backtester = ProfessionalBacktester(
        symbol="AAPL",
        initial_capital=100_000,
        commission_pct=0.001,  # %0.1
        slippage_pct=0.0005,   # %0.05
        use_risk_management=True
    )
    
    metrics = await backtester.run(
        strategy_class=AdvancedMomentum,
        strategy_params={
            'fast_period': 10,
            'slow_period': 30,
            'min_confidence': 0.6
        }
    )
    
    # Sonuçları export et
    backtester.export_results("aapl_results.csv")

asyncio.run(main())
```

### 2. Çoklu Sembol Backtest

```python
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

for symbol in symbols:
    backtester = ProfessionalBacktester(symbol=symbol)
    metrics = await backtester.run()
```

### 3. Parametre Optimizasyonu

```python
# Grid search
fast_periods = [5, 10, 15]
slow_periods = [20, 30, 40]

best_sharpe = -999
best_params = None

for fast in fast_periods:
    for slow in slow_periods:
        backtester = ProfessionalBacktester(symbol="AAPL")
        metrics = await backtester.run(
            strategy_params={
                'fast_period': fast,
                'slow_period': slow
            }
        )
        
        if metrics.sharpe_ratio > best_sharpe:
            best_sharpe = metrics.sharpe_ratio
            best_params = (fast, slow)

print(f"En iyi parametreler: Fast={best_params[0]}, Slow={best_params[1]}")
```

---

## 🔴 Live Trading (Dikkatli Kullanın!)

**⚠️ UYARI:** Gerçek para ile trading yapmadan önce mutlaka paper trading yapın!

### 1. Paper Trading (Önerilen)

```python
# main.py
config = {
    'initial_capital': 10_000,
    'symbols': ['BTC/USDT'],
    'exchange': 'binance',
    'tick_interval': 1.0,
}

system = TradingSystem(config)
await system.initialize()
await system.run()
```

### 2. Canlı İzleme

Sistem çalışırken her 60 saniyede bir durum raporu verir:

```
──────────────────────────────────────────────────────────────────
   📊 DURUM RAPORU
──────────────────────────────────────────────────────────────────
   Uptime         : 3600s
   Toplam Varlık  : $10,234.56
   Nakit          : $8,500.00
   Günlük PnL     : $+234.56
   Açık Pozisyon  : 2
   Günlük İşlem   : 15
──────────────────────────────────────────────────────────────────
```

### 3. Graceful Shutdown

Sistemi durdurmak için `Ctrl+C` kullanın. Sistem:
- Açık pozisyonları kontrol eder
- Final raporu yazdırır
- Güvenli şekilde kapanır

---

## 📚 Modül Açıklamaları

### data/csv_loader.py
**LocalCSVLoader**: Gelişmiş CSV veri yükleyici

**Özellikler:**
- Otomatik veri temizleme
- Eksik değer interpolasyonu
- Aykırı değer tespiti
- Veri kalitesi skoru (0-100)
- Cache desteği

**Kullanım:**
```python
loader = LocalCSVLoader(
    storage_path="data/storage",
    validate_data=True,
    interpolate_missing=True,
    remove_outliers=True
)

ticks = loader.load_data("AAPL")
```

### strategies/momentum.py
**AdvancedMomentum**: Çoklu teknik gösterge stratejisi

**Göstergeler:**
- Dual MA Crossover
- RSI (14-period)
- MACD (12,26,9)
- Bollinger Bands
- ATR (volatility)
- Volume confirmation

**Sinyal Üretme:**
```python
strategy = AdvancedMomentum(
    symbol="AAPL",
    fast_period=10,
    slow_period=30,
    min_confidence=0.6
)

signal = await strategy.on_tick(tick)
```

### risk/core.py
**EnterpriseRiskManager**: Kurumsal risk yönetimi

**Kontroller:**
1. Position-level risk
2. Portfolio-level risk
3. Daily loss limits
4. Value at Risk (VaR)
5. Liquidity management

**Kullanım:**
```python
risk_manager = EnterpriseRiskManager()

result = risk_manager.analyze_signal(signal, portfolio_state)

if result.passed:
    execute_trade(signal, result.adjusted_quantity)
```

### execution/portfolio.py
**PortfolioManager**: Portföy yönetimi

**Özellikler:**
- Mark-to-Market değerleme
- Average cost tracking
- Realized/Unrealized PnL
- Multi-asset support

**Kullanım:**
```python
portfolio = PortfolioManager(initial_balance=100_000)

# Fiyat güncellemesi
portfolio.update_price("AAPL", 150.25)

# İşlem
portfolio.update_after_trade(
    symbol="AAPL",
    quantity=10,
    price=150.25,
    side="BUY"
)

# Durum
state = portfolio.get_state()
print(f"Total: ${state.total_balance:.2f}")
```

---

## 📊 Performans Metrikleri

### Return Metrikleri
- **Total Return**: Toplam getiri (%)
- **Annualized Return (CAGR)**: Yıllıklandırılmış getiri
- **ROI**: Return on Investment

### Risk Metrikleri
- **Sharpe Ratio**: Risk-adjusted return (>1.0 iyi, >2.0 mükemmel)
- **Sortino Ratio**: Downside risk adjusted return
- **Calmar Ratio**: Return / Max Drawdown
- **Max Drawdown**: En büyük düşüş (%)
- **Volatility**: Yıllık volatilite (%)

### Trade Metrikleri
- **Win Rate**: Kazanan işlem oranı (%)
- **Profit Factor**: Toplam kazanç / Toplam kayıp
- **Avg Win/Loss**: Ortalama kazanç/kayıp
- **Avg Holding Period**: Ortalama pozisyon tutma süresi

---

## 🐛 Sorun Giderme

### Problem: CSV dosyası bulunamıyor
**Çözüm:**
```bash
# Dosya ismini kontrol et
ls data/storage/

# Beklenen format: AAPL_15min.csv
# Hatalı: aapl.csv, AAPL-15min.csv
```

### Problem: "Insufficient data" hatası
**Çözüm:**
- CSV'de en az 200 satır veri olmalı
- Timestamp formatını kontrol et: `YYYY-MM-DD HH:MM:SS`
- Eksik kolonları kontrol et: timestamp, close, volume

### Problem: Düşük Sharpe Ratio (<0.5)
**Çözüm:**
- `min_confidence` parametresini artır (0.6 → 0.7)
- `fast_period` / `slow_period` oranını optimize et
- Farklı semboller dene (volatilite farklı olabilir)

### Problem: Çok fazla işlem reddediliyor
**Çözüm:**
- Risk limitlerini gevşet:
  - `max_position_size` artır
  - `max_daily_trades` artır
- `min_confidence` düşür (0.6 → 0.55)

### Problem: Memory hatası (Büyük CSV'ler)
**Çözüm:**
```python
# Tarih filtresi kullan
ticks = loader.load_data(
    "AAPL", 
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 1)
)
```

---

## 📝 Önemli Notlar

### 1. Backtest vs Live Trading Farkı

**Backtest:**
- ✅ Risk yok
- ✅ Hızlı iterasyon
- ✅ Parametr optimizasyonu kolay
- ❌ Gerçek piyasa koşulları yok

**Live Trading:**
- ⚠️ Gerçek para riski
- ⚠️ Slippage ve latency
- ⚠️ Psikolojik faktörler
- ✅ Gerçek kar potansiyeli

### 2. Strateji Geliştirme İpuçları

1. **Backtest üzerinde test et**: Hiçbir stratejiyi direkt canlıya alma
2. **Paper trading yap**: En az 1 hafta simülasyon
3. **Küçük başla**: İlk canlı işlemde $1000 ile başla
4. **Risk yönetimini aktif tut**: Asla devre dışı bırakma
5. **Günlükleri takip et**: `logs/` klasöründeki logları incele

### 3. Performans Benchmark'ları

**İyi bir strateji:**
- Sharpe Ratio > 1.5
- Max Drawdown < %15
- Win Rate > %50
- CAGR > %15

**Mükemmel bir strateji:**
- Sharpe Ratio > 2.5
- Max Drawdown < %10
- Win Rate > %60
- CAGR > %25

---

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır! Büyük değişiklikler için önce issue açın.

---

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 📞 Destek

- **Issues**: GitHub Issues kullanın
- **Discussions**: GitHub Discussions
- **Email**: support@alphatrade.com

---

## 🎯 Sonraki Adımlar

1. ✅ Backtest çalıştırın (`python backtest.py`)
2. ✅ Sonuçları analiz edin
3. ✅ Parametreleri optimize edin
4. ✅ Paper trading yapın (`python main.py`)
5. ⚠️ (Opsiyonel) Canlı trading

---

**⚠️ DİKKAT:** Bu yazılım eğitim amaçlıdır. Gerçek para ile trading yapmadan önce riskleri anlayın ve profesyonel danışmanlık alın.

**📊 İyi Trading'ler!** 🚀