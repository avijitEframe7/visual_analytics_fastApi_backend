from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.database import get_db

router = APIRouter()

@router.get("/health/db")
def db_health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"db_status": "connected ✅"}
    except Exception as e:
        return {"db_status": "failed ❌", "error": str(e)}
