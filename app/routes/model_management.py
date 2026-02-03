import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database.database import get_db

router = APIRouter(prefix="/api/model_management", tags=["model_management"])


@router.get("/models")
def get_models(db: Session = Depends(get_db)):
    """Get all registered models."""
    try:
        result = db.execute(text("SELECT * FROM employeeinfo.models"))
        models = result.mappings().all()
        return models
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
