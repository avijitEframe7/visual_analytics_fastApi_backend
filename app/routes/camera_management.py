import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.database.database import get_db
from app.security.rbac import require_permission

router = APIRouter(
    prefix="/api/camera_management",
    tags=["camera_management"],
    dependencies=[Depends(require_permission("camera-management.view"))],
)


class CameraCreateRequest(BaseModel):
    """
    camera_id is optional and ignored on insert — SQL Server assigns it via IDENTITY on dbo.camera.
    Kept for backwards compatibility with clients that still send camera_id.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    camera_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("camera_id", "cameraId"),
    )
    camera_name: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("camera_name", "cameraName"),
    )
    zone_name: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("zone_name", "zoneName"),
    )
    ip_address: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("ip_address", "ipAddress"),
    )
    streaming_url: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("streaming_url", "streamingURL"),
    )


class CameraDeleteRequest(BaseModel):
    camera_id: str


class CameraUpdateRequest(BaseModel):
    """Update an existing row in dbo.camera by camera_id."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    camera_id: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("camera_id", "cameraId"),
    )
    camera_name: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("camera_name", "cameraName"),
    )
    zone_name: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("zone_name", "zoneName"),
    )
    ip_address: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("ip_address", "ipAddress"),
    )
    streaming_url: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("streaming_url", "streamingURL"),
    )


@router.get("/cameras")
def get_cameras(db: Session = Depends(get_db)):
    """Get all cameras."""
    try:
        result = db.execute(text("SELECT * FROM dbo.camera"))
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
    """Insert a camera into dbo.camera; camera_id comes from SQL Server IDENTITY."""
    try:
        result = db.execute(
            text(
                """
                INSERT INTO dbo.camera (camera_name, zone_name, ip_address, streaming_url)
                OUTPUT INSERTED.camera_id
                VALUES (:camera_name, :zone_name, :ip_address, :streaming_url)
                """
            ),
            {
                "camera_name": data.camera_name,
                "zone_name": data.zone_name,
                "ip_address": data.ip_address,
                "streaming_url": data.streaming_url,
            },
        )
        row = result.fetchone()
        if not row:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Insert did not return new camera_id",
            )
        new_id = row[0]
        db.commit()
        return {
            "status": "success",
            "message": "Camera inserted successfully",
            "camera_id": int(new_id) if new_id is not None else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/update_camera",
    dependencies=[Depends(require_permission("camera-dashboard.control"))],
)
def update_camera(data: CameraUpdateRequest, db: Session = Depends(get_db)):
    """Update a camera row by camera_id."""
    try:
        result = db.execute(
            text(
                """
                UPDATE dbo.camera
                SET camera_name = :camera_name,
                    zone_name = :zone_name,
                    ip_address = :ip_address,
                    streaming_url = :streaming_url
                WHERE camera_id = :camera_id
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
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Camera not found")
        return {
            "status": "success",
            "message": "Camera updated successfully",
            "camera_id": data.camera_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/del_camera",
    dependencies=[Depends(require_permission("camera-dashboard.control"))],
)
def del_camera(data: CameraDeleteRequest, db: Session = Depends(get_db)):
    """Delete a camera from dbo.camera by camera_id."""
    try:
        result = db.execute(
            text("DELETE FROM dbo.camera WHERE camera_id = :camera_id"),
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
