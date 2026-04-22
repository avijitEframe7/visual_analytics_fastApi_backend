import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database.database import get_db
from app.security.rbac import require_permission

router = APIRouter(
    prefix="/api/model_management",
    tags=["model_management"],
    dependencies=[Depends(require_permission("model-management.view"))],
)


@router.get("/models")
def get_models(db: Session = Depends(get_db)):
    """Get all registered models."""
    try:
        result = db.execute(text("SELECT * FROM models"))
        models = result.mappings().all()
        return models
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
