import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database.database import get_db

router = APIRouter(prefix="/api/camera_management", tags=["camera_management"])


@router.get("/cameras")
def get_cameras(db: Session = Depends(get_db)):
    """Get all cameras."""
    try:
        result = db.execute(text("SELECT * FROM employeeinfo.camera"))
        cameras = result.mappings().all()
        return cameras
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
