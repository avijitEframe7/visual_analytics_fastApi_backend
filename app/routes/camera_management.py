import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel

from app.database.database import get_db
from app.security.rbac import require_permission

router = APIRouter(
    prefix="/api/camera_management",
    tags=["camera_management"],
    dependencies=[Depends(require_permission("camera-management.view"))],
)


class CameraCreateRequest(BaseModel):
    camera_id: str
    camera_name: str
    zone_name: str
    ip_address: str
    streaming_url: str


class CameraDeleteRequest(BaseModel):
    camera_id: str


@router.get("/cameras")
def get_cameras(db: Session = Depends(get_db)):
    """Get all cameras."""
    try:
        result = db.execute(text("SELECT * FROM camera"))
        cameras = result.mappings().all()
        return cameras
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/set_camera",
    dependencies=[Depends(require_permission("camera-dashboard.control"))],
)
def set_camera(data: CameraCreateRequest, db: Session = Depends(get_db)):
    """Insert a camera into camera."""
    try:
        db.execute(
            text(
                """
                INSERT INTO camera
                (camera_id, camera_name, zone_name, ip_address, streaming_url)
                VALUES (:camera_id, :camera_name, :zone_name, :ip_address, :streaming_url)
                """
            ),
            {
                "camera_id": data.camera_id,
                "camera_name": data.camera_name,
                "zone_name": data.zone_name,
                "ip_address": data.ip_address,
                "streaming_url": data.streaming_url,
            },
        )
        db.commit()
        return {"status": "success", "message": "Camera inserted successfully"}
    except Exception as e:
        db.rollback()
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/del_camera",
    dependencies=[Depends(require_permission("camera-dashboard.control"))],
)
def del_camera(data: CameraDeleteRequest, db: Session = Depends(get_db)):
    """Delete a camera from camera by camera_id."""
    try:
        result = db.execute(
            text("DELETE FROM camera WHERE camera_id = :camera_id"),
            {"camera_id": data.camera_id},
        )
        db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Camera not found")
        return {"status": "success", "message": "Camera deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
