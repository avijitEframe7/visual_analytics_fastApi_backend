# Read environment variables from .env (e.g. DB_HOST, DB_USER)
import os
from urllib.parse import quote_plus
# Create DB engine and manage connections
from sqlalchemy import create_engine
# SessionLocal = per-request sessions; declarative_base = base class for ORM models
from sqlalchemy.orm import sessionmaker, declarative_base
# Load .env so DB credentials are not hardcoded
from dotenv import load_dotenv

# Load variables from .env into os.environ before reading them
load_dotenv()

# DB connection settings from env (keeps secrets out of code)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "1433")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
DB_TRUSTED_CONNECTION = os.getenv("DB_TRUSTED_CONNECTION", "yes")
DB_TRUST_CERT = os.getenv("DB_TRUST_CERT", "yes")

# SQLAlchemy connection string for SQL Server (MSSQL) via ODBC.
# Supports both trusted connection and SQL login.
if (DB_TRUSTED_CONNECTION or "").strip().lower() in {"yes", "true", "1"}:
    DATABASE_URL = (
        f"mssql+pyodbc://@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?driver={quote_plus(DB_DRIVER)}"
        f"&trusted_connection=yes"
        f"&TrustServerCertificate={DB_TRUST_CERT}"
    )
else:
    DATABASE_URL = (
        f"mssql+pyodbc://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?driver={quote_plus(DB_DRIVER)}"
        f"&TrustServerCertificate={DB_TRUST_CERT}"
    )

# Single shared engine; pool_pre_ping checks connections before use (auto reconnect)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # auto reconnect
    pool_size=10,         # keep up to 10 connections in the pool
    max_overflow=20       # allow 20 extra connections under load
)

# Factory for per-request sessions: no auto-commit/flush so we control transactions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# All ORM models inherit from Base so SQLAlchemy can create tables and map rows
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
