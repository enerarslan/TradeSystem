from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import settings

# Veritabanı dosyasının yeri (SQLite)
DATABASE_URL = "sqlite+aiosqlite:///./trade_history.db"

# Asenkron Motoru Oluştur
engine = create_async_engine(DATABASE_URL, echo=False)

# Session (Oturum) Fabrikası
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Tablolar için temel sınıf
Base = declarative_base()

async def init_db():
    """Veritabanı tablolarını oluşturur (İlk açılışta)"""
    async with engine.begin() as conn:
        # Tüm tabloları oluştur
        await conn.run_sync(Base.metadata.create_all)