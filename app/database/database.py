# Read environment variables from .env (e.g. DB_HOST, DB_USER)
import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "12345")
DB_NAME = os.getenv("DB_NAME", "employeeinfo")
DB_PORT = os.getenv("DB_PORT", "3306")

# MySQL via mysql-connector-python (see requirements: mysql-connector-python)
# Password is URL-encoded for special characters.
_user = quote_plus(DB_USER or "")
_pass = quote_plus(DB_PASS or "")
DATABASE_URL = (
    f"mysql+mysqlconnector://{_user}:{_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)

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
