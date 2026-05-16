# Read environment variables from .env (e.g. DB_HOST, DB_USER)
import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: str = "no") -> bool:
    return os.getenv(name, default).strip().lower() in ("yes", "true", "1")


DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
DB_DRIVER = os.getenv("DB_DRIVER")
DB_TRUSTED_CONNECTION = _env_flag("DB_TRUSTED_CONNECTION")
DB_TRUST_CERT = _env_flag("DB_TRUST_CERT")

# MSSQL via pyodbc (ODBC Driver for SQL Server)
_driver = quote_plus(DB_DRIVER)
_query = f"driver={_driver}"
if DB_TRUST_CERT:
    _query += "&TrustServerCertificate=yes"
# Windows auth only when no SQL login is configured
if DB_TRUSTED_CONNECTION and not DB_USER:
    _query += "&Trusted_Connection=yes"

_use_sql_auth = bool(DB_USER)
if _use_sql_auth:
    _user = quote_plus(DB_USER)
    _pass = quote_plus(DB_PASS or "")
    DATABASE_URL = (
        f"mssql+pyodbc://{_user}:{_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}?{_query}"
    )
else:
    DATABASE_URL = f"mssql+pyodbc://@{DB_HOST}:{DB_PORT}/{DB_NAME}?{_query}"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
