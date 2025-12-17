# AlphaTrade System - AI Agent Action Plan
# JPMorgan Seviyesi Backtest Sistemi İçin Komut Dökümanı

**Tarih:** 17 Aralık 2025  
**Amaç:** Dağınık kodu tek komutla çalışan JPMorgan seviyesi backtest sistemine dönüştürmek  
**Format:** AI Agent'a verilecek sıralı komutlar

---

# BÖLÜM 1: TESPİT EDİLEN KRİTİK HATALAR

## 🔴 KRİTİK HATA #1: Dosya İsimlendirme Uyumsuzluğu

**Konum:** `scripts/generate_sample_data.py` satır 85 ve `src/data/loaders/data_loader.py` satır 70

**Problem Açıklaması:**
- DataLoader `AAPL_15min.csv` formatında dosya arıyor
- Ama generate_sample_data.py `AAPL.csv` formatında oluşturuyor
- Sonuç: Sistem 0 sembol buluyor

**Mevcut Durum:**
```
data/raw/ klasöründe:
- AAPL_15min.csv ✓ (doğru format - zaten var)
- generate_sample_data.py yanlış format üretiyor
```

**AI Agent Komutu:**
```
DOSYA: scripts/generate_sample_data.py
SATIR 85 CIVARINDA BUL: filepath = output_path / f"{symbol}.csv"
DEĞİŞTİR: filepath = output_path / f"{symbol}_15min.csv"

AYRICA SATIR 56 CIVARINDA BUL: "date" column name
DEĞİŞTİR: "timestamp" olmalı (DataLoader bunu bekliyor)
```

---

## 🔴 KRİTİK HATA #2: Feature Pipeline'da Data Leakage

**Konum:** `main.py` fonksiyon `generate_features()` satır 340-400

**Problem Açıklaması:**
- `FeaturePipeline` sınıfı `fit()` ve `transform()` metodlarına sahip
- AMA main.py bunları KULLANMIYOR
- Direkt `TechnicalIndicators.generate_all_features()` çağırıyor
- Scaling parametreleri (mean, std) TÜM data üzerinden hesaplanıyor
- Bu GELECEK BİLGİSİNİ geçmişe sızdırıyor

**Neden Kritik:**
JPMorgan'da bu hata milyonlarca dolarlık yanlış kararlara yol açar. Backtest sonuçları gerçekçi değil.

**AI Agent Komutu:**
```
DOSYA: main.py
FONKSİYON: generate_features()

MEVCUT YAKLAŞIMI SİL (TechnicalIndicators direkt kullanımı)

YENİ YAKLAŞIM:
1. FeaturePipeline instance oluştur
2. Train data üzerinde pipeline.fit() çağır
3. Tüm data için pipeline.transform() çağır
4. ASLA fit_transform() tek seferde tüm data üzerinde çağırma

MANTIK:
- Scaling parametreleri SADECE train data'dan öğrenilmeli
- Test data'ya aynı parametreler UYGULANMALI
- Bu şekilde gelecek bilgisi sızmaz
```

---

## 🔴 KRİTİK HATA #3: Cross-Validation Purge Gap Uygulanmıyor

**Konum:** `main.py` fonksiyon `train_ml_model()` satır 540-620

**Problem Açıklaması:**
- `purge_gap` hesaplanıyor (doğru formül var)
- `PurgedKFoldCV` oluşturuluyor (doğru sınıf var)
- AMA sonra sklearn'ün `cross_validate()` fonksiyonu çağrılıyor
- sklearn'ün cross_validate'i PURGE GAP'I UYGULAMIYOR!
- Custom CV splitter'ın split() metodu hiç çalışmıyor

**Sonuç:** Train ve test setleri arasında veri sızıntısı var. Model olduğundan iyi görünüyor.

**AI Agent Komutu:**
```
DOSYA: main.py
FONKSİYON: train_ml_model()

sklearn cross_validate() ÇAĞRISINI KALDIR

MANUEL CV DÖNGÜSÜ YAZ:
1. cv.split(X, y) ile fold indekslerini al
2. Her fold için:
   a. train_idx ve test_idx ayır
   b. Leakage kontrolü yap (set kesişimi boş olmalı)
   c. Model oluştur ve train_idx üzerinde eğit
   d. test_idx üzerinde skorla
   e. Skoru listeye ekle
3. Tüm fold skorlarının ortalamasını al

PURGE GAP HESAPLAMASI:
purge_gap = prediction_horizon + max_feature_lookback + buffer
Örnek: 5 + 200 + 10 = 215 bar
```

---

## 🔴 KRİTİK HATA #4: TimescaleDB Entegre Değil

**Konum:** `src/data/storage/timescale_client.py` (tam implementasyon var) vs `main.py` (hiç kullanmıyor)

**Problem Açıklaması:**
- 800+ satırlık profesyonel TimescaleDB client yazılmış
- Connection pooling, retry logic, batch insert var
- Continuous aggregates, compression, retention policies var
- AMA main.py SADECE CSV dosyalarından okuma yapıyor
- TimescaleDB'nin tüm avantajları kullanılmıyor

**AI Agent Komutu:**
```
DOSYA: main.py
FONKSİYON: load_data()

KONTROL EKLE:
1. Config'de "use_timescale: true" var mı?
2. TIMESCALE_AVAILABLE flag'i True mu?
3. Evetse TimescaleClient kullan
4. Hayırsa mevcut CSV loading'e devam et

TİMESCALE KULLANIMI İÇİN:
1. ConnectionConfig oluştur (host, port, database, user, password)
2. TimescaleClient context manager ile aç
3. client.get_ohlcv(symbol, start, end, "15min") ile data çek
4. DataFrame formatına çevir ve döndür

CONFIG DOSYASINA EKLE (config/trading_config.yaml):
timescale:
  enabled: false  # true yapınca aktif olur
  host: localhost
  port: 5432
  database: alphatrade_db
  user: alphatrade
  password: ""
```

---

## 🔴 KRİTİK HATA #5: Market Impact Modeli Yanlış ADV Kullanıyor

**Konum:** `main.py` fonksiyon `run_backtest()` satır 430-490

**Problem Açıklaması:**
- `calculate_symbol_adv()` her sembol için ADV hesaplıyor (doğru)
- Sonra `np.mean()` ile ORTALAMA alınıyor (yanlış!)
- AlmgrenChrissModel bu ortalama ADV ile oluşturuluyor
- Her trade için AYNI ortalama ADV kullanılıyor

**Sonuç:**
- AAPL gibi likit hisseler için market impact FAZLA hesaplanıyor
- Küçük hisseler için market impact AZ hesaplanıyor
- Backtest sonuçları gerçekçi değil

**AI Agent Komutu:**
```
DOSYA: main.py ve src/backtesting/engine.py

DEĞİŞİKLİK 1 - main.py:
symbol_adv dictionary'sini BacktestEngine'e PARAMETRE olarak geçir

DEĞİŞİKLİK 2 - engine.py BacktestEngine.run():
Her trade için o sembolün KENDİ ADV'sini kullan:
- trade sembolünü al
- symbol_adv[symbol] ile o sembolün ADV'sini bul
- market_impact.calculate_impact(trade_value, symbol_adv) çağır

DOĞRU MANTIK:
- AAPL (ADV: $10B) için 1M$'lık trade = minimal impact
- Küçük hisse (ADV: $10M) için 1M$'lık trade = büyük impact
```

---

## 🟡 ORTA SEVİYE HATA #6: Walk-Forward Validation Bağlı Değil

**Konum:** `config/trading_config.yaml` satır 85-92 ve `main.py`

**Problem Açıklaması:**
- Config dosyasında walk_forward ayarları var:
  ```yaml
  walk_forward:
    enabled: true
    train_period_days: 126
    test_period_days: 21
  ```
- AMA main.py bu ayarları HİÇ OKUMUYOR
- WalkForwardValidator sınıfı MEVCUT ama kullanılmıyor
- Sistem sadece PurgedKFoldCV kullanıyor

**AI Agent Komutu:**
```
DOSYA: main.py
FONKSİYON: train_ml_model()

WALK-FORWARD KONTROLÜ EKLE:
1. config["backtest"]["walk_forward"]["enabled"] oku
2. True ise WalkForwardValidator kullan
3. False ise mevcut CV'ye devam et

WalkForwardValidator PARAMETRELERI:
- train_period = train_period_days * 26 (günde 26 bar)
- test_period = test_period_days * 26
- step_size = test_period (non-overlapping)
- expanding = config'deki "anchored" değeri
- purge_gap = hesaplanan purge_gap

WALK-FORWARD AVANTAJI:
- Gerçek trading'i simüle eder
- Her dönem için model yeniden eğitilir
- Out-of-sample performance daha gerçekçi
```

---

## 🟡 ORTA SEVİYE HATA #7: Survivorship Bias Düzeltmesi Pasif

**Konum:** `main.py` fonksiyon `load_data()` satır 230-250

**Problem Açıklaması:**
- Kod `symbol_metadata.json` dosyasını arıyor
- Bu dosya YOK (sadece .gitkeep var)
- UniverseManager hiçbir zaman aktif olmuyor
- Sonuç: Sadece hayatta kalan hisseler test ediliyor
- Backtest sonuçları aşırı iyimser

**Survivorship Bias Nedir:**
- 2010'da 100 hisse vardı
- 20 tanesi battı/delisted oldu
- Bugün sadece 80 hisse var
- Sadece bu 80 üzerinde test yapmak = sadece başarılıları test etmek

**AI Agent Komutu:**
```
YENİ DOSYA OLUŞTUR: scripts/generate_universe_metadata.py

İÇERİK:
1. data/raw/ klasöründeki tüm sembolleri listele
2. Her sembol için metadata oluştur:
   - listing_date: İlk veri tarihi
   - delisting_date: null (veya son veri tarihi)
   - sector: Sektör bilgisi
   - is_active: true/false
3. JSON dosyasına kaydet: data/raw/symbol_metadata.json

SONRA main.py'de:
UniverseManager aktif olacak
Backtest başlangıç tarihindeki universe kullanılacak
O tarihte mevcut olmayan hisseler dahil edilmeyecek
```

---

## 🟡 ORTA SEVİYE HATA #8: Deflated Sharpe Ratio Raporlarda Yok

**Konum:** `src/backtesting/metrics.py` (hesaplama VAR) vs `src/backtesting/reports/` (kullanılmıyor)

**Problem Açıklaması:**
- `calculate_deflated_sharpe_ratio()` fonksiyonu mevcut ve doğru
- `calculate_sharpe_statistics()` tam SharpeStatistics döndürüyor
- AMA raporlar sadece NORMAL Sharpe gösteriyor
- DSR, PSR, MinTRL hiçbir raporda yok

**Neden Kritik:**
JPMorgan'da normal Sharpe'a bakılmaz. Multiple testing için düzeltilmiş DSR bakılır.
10 strateji test edip en iyi Sharpe'ı seçmek = şans eseri iyi sonuç bulmak.

**AI Agent Komutu:**
```
DOSYA: src/backtesting/reports/report_generator.py

calculate_sharpe_statistics() IMPORT ET

RAPOR OLUŞTURURKEN:
1. n_trials parametresi al (kaç strateji test edildi)
2. Her strateji için SharpeStatistics hesapla
3. Rapora EKLE:
   - Deflated Sharpe Ratio (DSR)
   - Probabilistic Sharpe Ratio (PSR) 
   - Minimum Track Record Length (ay)
   - Is Statistically Significant (Evet/Hayır)

TABLO FORMATINDA GÖSTER:
| Strateji | Return | Sharpe | DSR | PSR | Significant |
|----------|--------|--------|-----|-----|-------------|
| Momentum | 15%    | 1.2    | 0.8 | 89% | Evet        |
| MeanRev  | 8%     | 0.7    | 0.3 | 62% | Hayır       |
```

---

# BÖLÜM 2: EKSİK FONKSİYONELLİKLER

## 📌 EKSİK #1: Tek Komutla Çalıştırma

**Mevcut Durum:**
- `python main.py` çalışıyor AMA çok fazla parametre var
- Kullanıcı hangi mode, engine, model kullanacağını bilmeli
- Hata durumunda ne yapacağı belirsiz

**AI Agent Komutu:**
```
YENİ DOSYA OLUŞTUR: orchestrate.py

BU DOSYA TEK ENTRY POINT OLMALI

KULLANIM:
python orchestrate.py                    # Her şeyi çalıştır
python orchestrate.py --quick            # ML training atla
python orchestrate.py --validate-only    # Sadece data kontrol
python orchestrate.py --holdout 0.2      # %20 holdout ayır

İÇ YAPI:
1. Banner göster (versiyon, tarih, mode)
2. Config yükle
3. Data yükle ve validate et
4. Eğer data yoksa veya hatalıysa -> hata mesajı ve çık
5. Feature generate et (leakage-safe)
6. ML model eğit (purged CV ile)
7. Tüm stratejileri backtest et
8. Ensemble oluştur ve test et
9. Institutional-grade rapor üret
10. Holdout validation (varsa)
11. Sonuç özeti göster

HER ADIMDA:
- Progress göster
- Hata olursa anlaşılır mesaj ver
- Log'a yaz
```

---

## 📌 EKSİK #2: Pre-Flight Data Validation

**Mevcut Durum:**
- DataValidator sınıfı var ve iyi çalışıyor
- AMA backtest başlamadan önce TÜM data kontrol edilmiyor
- Tek bir bozuk sembol tüm backtest'i bozabiliyor

**AI Agent Komutu:**
```
YENİ DOSYA OLUŞTUR: scripts/validate_all_data.py

FONKSİYON:
1. data/raw/ klasöründeki TÜM dosyaları tara
2. Her dosya için:
   - Format kontrolü (timestamp, OHLCV kolonları)
   - Missing value kontrolü (max %5)
   - Fiyat anomalisi kontrolü (tek bar'da max %50 değişim)
   - Volume kontrolü (negatif olmamalı)
   - Tarih sıralaması kontrolü (monotonic increasing)
3. Sonuçları tablo olarak göster
4. Hatalı dosya varsa listele
5. Exit code: 0 (başarılı) veya 1 (hatalı)

BACKTEST'TEN ÖNCE ÇALIŞTIR:
python scripts/validate_all_data.py
Eğer exit code 1 ise backtest başlamasın
```

---

## 📌 EKSİK #3: Transaction Cost Sensitivity Analysis

**Mevcut Durum:**
- Tek bir commission ve slippage değeri kullanılıyor
- Backtest sonucu bu değerlere çok hassas olabilir
- JPMorgan'da farklı cost senaryoları test edilmeli

**AI Agent Komutu:**
```
YENİ FONKSİYON EKLE: main.py veya orchestrate.py

FONKSİYON ADI: run_cost_sensitivity()

MANTIK:
1. Commission değerleri: [0.0005, 0.001, 0.002, 0.005]
2. Slippage değerleri: [0.0002, 0.0005, 0.001, 0.002]
3. Her kombinasyon için backtest çalıştır
4. Sonuçları matris olarak göster:

         | Slip 0.02% | Slip 0.05% | Slip 0.1% | Slip 0.2% |
---------|------------|------------|-----------|-----------|
Comm 0.05%|   12.5%   |   11.2%    |   9.8%    |   7.1%    |
Comm 0.1% |   11.8%   |   10.5%    |   9.1%    |   6.4%    |
Comm 0.2% |   10.4%   |    9.1%    |   7.7%    |   5.0%    |

YORUM:
- Strateji cost'a ne kadar hassas?
- Hangi cost seviyesinde karlılık kayboluyor?
- Break-even cost nedir?
```

---

## 📌 EKSİK #4: Out-of-Sample Holdout Validation

**Mevcut Durum:**
- TÜM data train ve backtest için kullanılıyor
- Gerçek out-of-sample test yok
- Model overfit olmuş olabilir ve bilemeyiz

**AI Agent Komutu:**
```
DEĞİŞİKLİK: load_data() fonksiyonu

YENİ PARAMETRE: holdout_pct (default: 0.0)

MANTIK:
1. Data yükle
2. holdout_pct > 0 ise:
   - Son %X'i ayır (holdout_data)
   - Geri kalanı train_data
3. Train/backtest SADECE train_data üzerinde
4. En iyi strateji belirlendikten SONRA
5. Holdout_data üzerinde FINAL test
6. Bu sonuç "gerçek" out-of-sample performance

NEDEN ÖNEMLİ:
- Backtest'te 10 strateji test ettin
- En iyi Sharpe 1.5 olan seçildi
- AMA bu "data snooping" olabilir
- Holdout'ta 0.8 gelirse gerçek performance o
```

---

## 📌 EKSİK #5: Regime-Aware Backtest

**Mevcut Durum:**
- Tek bir backtest tüm dönem için yapılıyor
- Bull market, bear market, sideways ayrı ayrı analiz yok
- Strateji belirli rejimlerde kötü olabilir

**AI Agent Komutu:**
```
MEVCUT SINIF KULLAN: src/features/regime/volatility_regime.py

BACKTEST SONRASI ANALİZ:
1. Tüm dönem için volatilite rejimi belirle
2. Günleri kategorize et: low_vol, normal_vol, high_vol, crisis
3. Her rejim için ayrı metrikler hesapla:

   | Rejim    | Gün | Return | Sharpe | MaxDD |
   |----------|-----|--------|--------|-------|
   | Low Vol  | 150 | +8%    | 1.8    | -3%   |
   | Normal   | 200 | +12%   | 1.2    | -8%   |
   | High Vol | 80  | -5%    | -0.3   | -15%  |
   | Crisis   | 20  | -10%   | -1.5   | -25%  |

YORUM:
- Strateji hangi rejimlerde iyi/kötü?
- Crisis'te hedge var mı?
- Volatilite spike'ta ne oluyor?
```

---

# BÖLÜM 3: KALDIRILMASI GEREKEN FAZLALIKLAR

## 🗑️ FAZLALIK #1: Duplicate Validation Logic

**Konum:** 
- `src/data/validators/data_validator.py` - DataValidator sınıfı
- `main.py` satır 160-195 - validate_data_for_backtest() fonksiyonu

**Problem:** Aynı validation logic iki yerde yazılmış

**AI Agent Komutu:**
```
DOSYA: main.py
SİL: validate_data_for_backtest() fonksiyonunu tamamen kaldır
KULLAN: Sadece DataValidator sınıfını her yerde
```

---

## 🗑️ FAZLALIK #2: VectorizedBacktest Sınıfı

**Konum:** `src/backtesting/engine.py` satır 250-320

**Problem:**
- BacktestEngine var (tam özellikli)
- VectorizedBacktest var (basitleştirilmiş)
- İkisi de aynı işi yapıyor
- VectorizedBacktest trade kaydı tutmuyor

**AI Agent Komutu:**
```
DOSYA: src/backtesting/engine.py
SİL: VectorizedBacktest sınıfını tamamen kaldır
KALSIN: BacktestEngine (primary) ve EventDrivenEngine (advanced)
GÜNCELLE: __init__.py'den VectorizedBacktest export'unu kaldır
```

---

## 🗑️ FAZLALIK #3: Kullanılmayan Config Dosyaları

**Konum:** `config/` klasörü

**Mevcut Dosyalar:**
- base.yaml
- development.yaml
- staging.yaml
- production.yaml
- trading_config.yaml
- feature_params.yaml
- ml_config.yaml
- risk_limits.yaml
- institutional_defaults.yaml

**Problem:** Çok fazla config dosyası, hangisinin kullanıldığı belirsiz

**AI Agent Komutu:**
```
SADELEŞTIR:
1. trading_config.yaml - ANA CONFIG (her şey burada)
2. production.yaml - Sadece prod'a özel override'lar
3. Diğerlerini BİRLEŞTİR trading_config.yaml içine

VEYA:
Tek config.yaml dosyası oluştur, environment bazlı section'larla
```

---

# BÖLÜM 4: JPMORGAN SEVİYESİ GELİŞTİRMELER

## 🚀 GELİŞTİRME #1: Execution Quality Metrics

**Mevcut:** Sadece slippage yüzdesi var

**Gerekli:**
```
EKLE: Execution quality metrikleri
- Implementation Shortfall
- Arrival Price vs Execution Price
- VWAP vs Execution Price
- Market Impact (realized vs estimated)

RAPORDA GÖSTER:
"Execution Quality Report"
- Ortalama slippage: 0.05%
- VWAP'a göre performance: -0.02%
- Toplam execution cost: $45,230
```

---

## 🚀 GELİŞTİRME #2: Risk Attribution

**Mevcut:** Toplam risk metrikleri var

**Gerekli:**
```
EKLE: Risk decomposition
- Systematic risk (market beta)
- Idiosyncratic risk (stock-specific)
- Sector risk
- Style risk (momentum, value, size)

RAPORDA GÖSTER:
"Risk Attribution"
- Toplam Volatilite: 15%
  - Market: 8%
  - Sector: 4%
  - Stock-specific: 3%
- Active Risk: 7%
```

---

## 🚀 GELİŞTİRME #3: Stress Testing

**Mevcut:** Monte Carlo var ama stress test yok

**Gerekli:**
```
EKLE: Historical stress scenarios
- 2008 Financial Crisis
- 2020 COVID Crash
- 2022 Rate Hike

HER SENARYO İÇİN:
- O dönemdeki market koşullarını simüle et
- Stratejinin performansını hesapla
- Maximum loss'u göster

RAPORDA GÖSTER:
"Stress Test Results"
| Scenario      | Duration | Market | Strategy | Max Loss |
|---------------|----------|--------|----------|----------|
| 2008 Crisis   | 6 months | -50%   | -25%     | -35%     |
| COVID Crash   | 1 month  | -35%   | -15%     | -20%     |
| 2022 Rates    | 9 months | -25%   | -10%     | -18%     |
```

---

## 🚀 GELİŞTİRME #4: Liquidity Risk Monitoring

**Mevcut:** ADV kontrolü var ama gerçek zamanlı değil

**Gerekli:**
```
EKLE: Liquidity metrics
- Days to liquidate (DTL) - pozisyonu tasfiye etme süresi
- Liquidity score per position
- Portfolio liquidity score
- Liquidation cost estimate

SINIRLAR:
- Max position ADV %: 5%
- Max portfolio DTL: 3 days
- Alert at DTL > 2 days

RAPORDA GÖSTER:
"Liquidity Risk Report"
- Portfolio DTL: 1.5 days
- Least liquid position: XYZ (DTL: 4 days) ⚠️
- Estimated liquidation cost: $125,000
```

---

## 🚀 GELİŞTİRME #5: Model Decay Monitoring

**Mevcut:** Model bir kere eğitiliyor, sonra kullanılıyor

**Gerekli:**
```
EKLE: Model performance tracking
- Rolling out-of-sample performance
- Feature importance stability
- Prediction accuracy over time
- Model refresh triggers

MANTIK:
1. Her hafta model performansını ölç
2. Son 4 hafta Sharpe < 0 ise ALERT
3. Feature importance değişimi > %30 ise ALERT
4. Otomatik retrain trigger'ı

RAPORDA GÖSTER:
"Model Health Dashboard"
- Current model age: 45 days
- Rolling Sharpe (4w): 0.8 (↓ from 1.2)
- Feature stability: 85%
- Recommendation: RETRAIN SOON
```

---

# BÖLÜM 5: UYGULAMA ÖNCELİK SIRASI

## Faz 1: Kritik Düzeltmeler (İLK YAPILACAK)

**Öncelik 1 - Data Loading:**
```
1. generate_sample_data.py dosya adı düzeltmesi
2. timestamp kolon adı düzeltmesi
3. validate_all_data.py script'i oluştur
```

**Öncelik 2 - Data Leakage:**
```
1. main.py generate_features() fonksiyonunu düzelt
2. FeaturePipeline.fit() -> transform() akışı uygula
3. Scaler parametreleri sadece train data'dan
```

**Öncelik 3 - CV Düzeltmesi:**
```
1. sklearn cross_validate() kaldır
2. Manuel purged CV loop yaz
3. Leakage kontrolü ekle
```

---

## Faz 2: Entegrasyon (İKİNCİ YAPILACAK)

**Öncelik 4 - Market Impact:**
```
1. Per-symbol ADV kullanımına geç
2. BacktestEngine'e symbol_adv parametresi ekle
3. Her trade için doğru ADV kullan
```

**Öncelik 5 - Walk-Forward:**
```
1. Config'den walk_forward ayarlarını oku
2. WalkForwardValidator'ı entegre et
3. Expanding vs sliding window seçeneği
```

**Öncelik 6 - TimescaleDB:**
```
1. Config'de timescale ayarları ekle
2. load_data() içinde TimescaleClient kullanımı
3. Fallback: CSV loading
```

---

## Faz 3: Raporlama (ÜÇÜNCÜ YAPILACAK)

**Öncelik 7 - Institutional Metrics:**
```
1. DSR, PSR, MinTRL raporlara ekle
2. n_trials parametresi (test edilen strateji sayısı)
3. Statistical significance göstergesi
```

**Öncelik 8 - Sensitivity Analysis:**
```
1. Cost sensitivity matrix
2. Regime-aware breakdown
3. Stress test scenarios
```

---

## Faz 4: Tek Komut Sistemi (SON YAPILACAK)

**Öncelik 9 - Orchestrator:**
```
1. orchestrate.py oluştur
2. Tüm adımları sırala
3. Hata yönetimi ekle
4. Progress gösterimi
5. Holdout validation
```

**Öncelik 10 - Final Test:**
```
1. Tüm sistemi baştan sona test et
2. Sample data ile full pipeline çalıştır
3. Raporları kontrol et
4. README güncelle
```

---

# BÖLÜM 6: KALİTE KONTROL CHECKLIST

## Backtest Başlamadan Önce

- [ ] Tüm data dosyaları validate edildi mi?
- [ ] Feature pipeline FIT train data üzerinde yapıldı mı?
- [ ] Purge gap doğru hesaplandı mı? (horizon + lookback + buffer)
- [ ] Survivorship bias kontrolü yapıldı mı?
- [ ] Holdout data ayrıldı mı?

## Backtest Sırasında

- [ ] Cash balance hiç negatif olmuyor mu?
- [ ] Market impact per-symbol ADV ile mi hesaplanıyor?
- [ ] Slippage realistic mi?
- [ ] Trade execution t+1'de mi yapılıyor (look-ahead yok)?

## Backtest Sonrasında

- [ ] DSR pozitif mi?
- [ ] PSR > 95% mi (statistical significance)?
- [ ] Holdout performance in-sample'a yakın mı?
- [ ] Cost sensitivity makul mü?
- [ ] Regime analysis yapıldı mı?

---

# SONUÇ

Bu döküman AI Agent'ın AlphaTrade sistemini JPMorgan seviyesine getirmesi için gereken TÜM adımları içermektedir.

**Tahmini Süre:** 
- Faz 1: 2 saat
- Faz 2: 2 saat  
- Faz 3: 1.5 saat
- Faz 4: 1 saat
- **TOPLAM: 6.5 saat**

**Kritik Başarı Metrikleri:**
1. Data leakage: SIFIR
2. Purge gap: Doğru hesaplanmış
3. DSR: Tüm raporlarda mevcut
4. Tek komut: `python orchestrate.py` her şeyi çalıştırıyor
5. Holdout validation: Out-of-sample sonuç mevcut

---

*Bu döküman AI Agent'a verilecek komutlar formatında hazırlanmıştır.*
*Kod içermez, sadece NE yapılması gerektiğini açıklar.*
