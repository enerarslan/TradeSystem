import sys
from loguru import logger
from config.settings import settings

# Mevcut loglayıcıyı temizle
logger.remove()

# 1. Konsola (Ekrana) Yazdır
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO" if settings.APP_MODE == "PRODUCTION" else "DEBUG"
)

# 2. Dosyaya Kaydet (Hata olursa geçmişe bakabilmek için)
logger.add(
    "logs/system_error.log",
    rotation="500 MB", # Dosya 500MB olunca yenisini aç
    retention="10 days", # 10 günden eski logları sil
    level="ERROR",
    compression="zip"
)

logger.add(
    "logs/trade_history.log",
    rotation="1 day",
    level="INFO",
    filter=lambda record: "TRADE" in record["message"] # Sadece ticaret mesajlarını buraya yaz
)

# Dışarıya bu özelleştirilmiş logger'ı veriyoruz
log = logger