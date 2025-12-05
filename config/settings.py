import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings  # <-- BURASI DEĞİŞTİ

# .env dosyasını yükle
load_dotenv()

class Settings(BaseSettings):
    """
    Tüm sistem ayarlarının merkezi olarak yönetildiği sınıf.
    """
    PROJECT_NAME: str = "AlphaTrade Institutional Engine"
    VERSION: str = "1.0.0"
    APP_MODE: str = os.getenv("APP_MODE", "PRODUCTION")
    
    # Veritabanı Ayarları
    DB_URL: str = os.getenv("DATABASE_URL", "sqlite:///./trade_system.db")

    # --- GELİŞMİŞ RİSK YÖNETİMİ AYARLARI ---
    # Maksimum günlük zarar limiti (Örn: Portföyün %2'si erirse robot durur)
    MAX_DAILY_DRAWDOWN_PERCENT: float = 2.0 
    
    # Tek bir işleme girilebilecek maksimum sermaye oranı (Örn: %10)
    MAX_POSITION_SIZE_PERCENT: float = 10.0
    
    # Bir günde yapılabilecek maksimum işlem sayısı (Overtrading engelleme)
    MAX_TRADES_PER_DAY: int = 50
    
    # Kaldıraç Limiti (Spot piyasa için 1.0)
    MAX_LEVERAGE: float = 1.0
    
    # Stop-Loss (Zarar Kes) Oranı (Otomatik eklenir)
    DEFAULT_STOP_LOSS_PERCENT: float = 0.02  # %2

settings = Settings()