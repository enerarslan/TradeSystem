# AlphaTrade System - Kurulum ve Çalıştırma Rehberi

## 📋 Gereksinimler

### Sistem
- Python 3.10+ (3.11 önerilir)
- 16GB+ RAM (Deep Learning için 32GB önerilir)
- NVIDIA GPU (opsiyonel, Deep Learning için)
- PostgreSQL 14+ + TimescaleDB (opsiyonel, production için önerilir)

### Veri
- 46 adet hisse senedi (15 dakikalık OHLCV)
- 4.5 yıllık veri (2021-01 ~ 2025)
- Konum: `data/raw/*.csv`

---

## 🚀 ADIM 1: Environment Kurulumu

```bash
# Proje dizinine git
cd C:\Users\enera\Desktop\AlphaTrade_System

# Virtual environment oluştur (eğer yoksa)
python -m venv venv

# Aktive et (Windows)
venv\Scripts\activate

# Tüm paketleri yükle
pip install -r requirements.txt
```

### GPU Desteği (Opsiyonel - Deep Learning için)
```bash
# CUDA destekli PyTorch (NVIDIA GPU varsa)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### TimescaleDB Kurulumu (Opsiyonel - Production için önerilir)
```bash
# Windows için:
# 1. PostgreSQL 14+ indir: https://www.postgresql.org/download/windows/
# 2. TimescaleDB indir: https://docs.timescale.com/install/latest/self-hosted/installation-windows/

# Kurulumdan sonra:
python scripts/setup_timescaledb.py --host localhost --port 5432 --user postgres --password postgres

# Doğrulama
python scripts/setup_timescaledb.py --verify-only
```

**TimescaleDB Avantajları:**
- Zaman serisi verileri için optimize edilmiş PostgreSQL
- Otomatik chunk'lama ve sıkıştırma
- Hızlı time-bucket sorguları
- Model, prediction ve metrik kayıtları

---

## 🧪 ADIM 2: Kurulumu Test Et

```bash
# Temel importları test et
python -c "from src.training import ModelFactory, Trainer; print('Training OK')"
python -c "from src.features import FeaturePipeline; print('Features OK')"
python -c "from src.backtesting import BacktestEngine; print('Backtest OK')"

# PyTorch test (opsiyonel)
python -c "import torch; print(f'PyTorch OK, CUDA: {torch.cuda.is_available()}')"

# Unit testleri çalıştır
pytest tests/unit/ -v --tb=short
```

---

## 📊 ADIM 3: Veriyi Doğrula

```bash
# Veri kalitesini kontrol et
python -c "
import pandas as pd
from pathlib import Path

data_dir = Path('data/raw')
files = list(data_dir.glob('*.csv'))
print(f'Toplam {len(files)} hisse bulundu')

# İlk dosyayı kontrol et
df = pd.read_csv(files[0])
print(f'Örnek: {files[0].name}')
print(f'  Satır sayısı: {len(df):,}')
print(f'  Tarih aralığı: {df.timestamp.min()} ~ {df.timestamp.max()}')
print(f'  Kolonlar: {list(df.columns)}')
"
```

---

## 🤖 ADIM 4: ML Model Eğitimi

### 4.1 Tek Hisse - Hızlı Test
```bash
# LightGBM ile hızlı test (AAPL)
python main.py --mode train --model lightgbm --symbols AAPL --n-trials 10
```

### 4.2 Tüm Modeller - Tek Hisse
```bash
# Tüm ML modellerini eğit (LightGBM, XGBoost, CatBoost)
python main.py --mode train --symbols AAPL --n-trials 50
```

### 4.3 Tüm Hisseler - Production Eğitimi
```bash
# Tüm hisseler için model eğitimi (UZUN SÜRER ~2-4 saat)
python main.py --mode train --n-trials 100

# Walk-forward validation ile
python main.py --mode train --n-trials 100 --cv-splits 5
```

### 4.4 Feature Selection ile Eğitim
```bash
# En önemli 30 feature seç ve eğit
python main.py --mode train --feature-selection --n-features 30 --symbols AAPL MSFT GOOGL
```

### 4.5 Drift Detection ile Eğitim
```bash
# Veri drift kontrolü yap
python main.py --mode train --check-drift --symbols AAPL
```

### 4.6 Training Pipeline ile Eğitim (ÖNERİLEN)
```bash
# TrainingPipeline orchestrator kullan (JPMorgan-level workflow)
# 8 aşamalı tam eğitim: validation → feature gen → training → evaluation → registration
python main.py --mode train --use-pipeline --symbols AAPL --n-trials 50

# Tüm hisseler için pipeline ile eğitim
python main.py --mode train --use-pipeline --n-trials 100

# Pipeline + Feature Selection
python main.py --mode train --use-pipeline --feature-selection --n-features 30 --symbols AAPL
```

**TrainingPipeline Avantajları:**
- Otomatik data validation
- Feature leakage prevention
- Purged cross-validation
- Statistical significance testing
- Otomatik model registration (metric threshold'u geçerse)
- Stage-by-stage timing ve error tracking

---

## 🧠 ADIM 5: Deep Learning Eğitimi

### 5.1 LSTM Model
```bash
# LSTM eğitimi (GPU önerilir)
python main.py --mode train --deep-learning --dl-model lstm --symbols AAPL

# Özel parametrelerle
python main.py --mode train --deep-learning --dl-model lstm --epochs 100 --batch-size 64
```

### 5.2 Attention LSTM
```bash
python main.py --mode train --deep-learning --dl-model attention_lstm --symbols AAPL MSFT
```

### 5.3 Transformer
```bash
python main.py --mode train --deep-learning --dl-model transformer --symbols AAPL
```

---

## 📈 ADIM 6: Backtest

### 6.1 Eğitilmiş Model ile Backtest
```bash
# Momentum stratejisi ile backtest
python main.py --mode backtest --strategy momentum

# Mean reversion ile
python main.py --mode backtest --strategy mean_reversion

# ML tahminleri ile
python main.py --mode backtest --strategy ml_predictions --model-path models/best_model.joblib
```

### 6.2 Walk-Forward Backtest
```bash
python main.py --mode backtest --walk-forward --train-period 5040 --test-period 1260
```

---

## 📊 ADIM 7: Sonuçları İncele

### MLflow Dashboard
```bash
# MLflow UI başlat
mlflow ui --port 5000

# Tarayıcıda aç: http://localhost:5000
```

### Model Karşılaştırma
```bash
python -c "
import mlflow
mlflow.set_tracking_uri('mlruns')

# Son deneyleri listele
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f'{exp.name}: {exp.experiment_id}')
"
```

---

## 🔧 Önemli CLI Parametreleri

| Parametre | Açıklama | Örnek |
|-----------|----------|-------|
| `--mode` | Çalışma modu | `train`, `backtest`, `full` |
| `--model` | ML model tipi | `lightgbm`, `xgboost`, `catboost` |
| `--symbols` | Hisse sembolleri | `AAPL MSFT GOOGL` |
| `--n-trials` | Optuna deneme sayısı | `50`, `100` |
| `--cv-splits` | CV fold sayısı | `5` |
| `--use-pipeline` | TrainingPipeline orchestrator | flag (önerilen) |
| `--deep-learning` | DL modu aktif | flag |
| `--dl-model` | DL model tipi | `lstm`, `attention_lstm`, `transformer` |
| `--epochs` | DL epoch sayısı | `100` |
| `--batch-size` | DL batch size | `64` |
| `--device` | DL device | `auto`, `cpu`, `cuda`, `mps` |
| `--feature-selection` | Feature seçimi aktif | flag |
| `--n-features` | Seçilecek feature sayısı | `30`, `50` |
| `--check-drift` | Drift kontrolü | flag |
| `--dry-run` | Sadece validasyon | flag |
| `--resume` | Checkpoint'tan devam | path |
| `--validate-features` | Feature leakage kontrolü | flag |

---

## 📁 Çıktı Dosyaları

```
AlphaTrade_System/
├── models/                    # Eğitilmiş modeller
│   ├── lightgbm_AAPL_*.joblib
│   ├── lstm_AAPL_*.pt
│   └── model_metadata.json
├── mlruns/                    # MLflow deneyleri
├── checkpoints/               # Eğitim checkpoint'ları
├── reports/                   # Backtest raporları
│   └── tearsheet_*.html
└── logs/                      # Log dosyaları
```

---

## ⚡ Hızlı Başlangıç (Tek Komut)

```bash
# En hızlı test - tek hisse, az deneme
python main.py --mode train --model lightgbm --symbols AAPL --n-trials 10

# Tam pipeline - tek hisse
python main.py --mode full --symbols AAPL --n-trials 50

# Production - tüm hisseler (UZUN)
python main.py --mode full --n-trials 100
```

---

## ❓ Sorun Giderme

### Import Hatası
```bash
pip install --upgrade -r requirements.txt
```

### CUDA Hatası
```bash
# CPU kullan
python main.py --mode train --device cpu
```

### Bellek Hatası
```bash
# Batch size küçült
python main.py --mode train --deep-learning --batch-size 32

# Daha az hisse
python main.py --mode train --symbols AAPL MSFT GOOGL
```

### MLflow Hatası
```bash
# MLflow dizinini temizle
rm -rf mlruns/
```
