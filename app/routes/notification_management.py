import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Any

from app.database.database import get_db
from app.exception_log_paths import exception_logs_has_incident_image_path
from app.routes.camera_config import get_camera_config
from app.security.rbac import require_permission

router = APIRouter(
    prefix="/api/notification_management",
    tags=["notification_management"],
    dependencies=[Depends(require_permission("notifications.view"))],
)

# Sync with camera_dashboard: media/exception_logs/detections/
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_VIOLATION_SNAPSHOT_DIR = _BACKEND_ROOT / "media" / "exception_logs" / "detections"


def _decode_bytes(val: Any) -> str:
    """Decode DB values (e.g. bytes) to string for JSON; violation/exception type may vary."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return str(val)
    return str(val)


def _incident_image_for_json(val: Any) -> str:
    """
    Incident_image is stored as varbinary (JPEG). Expose as base64 for JSON clients;
    display as <img src=\"data:image/jpeg;base64,...\"> if needed.
    """
    if val is None:
        return ""
    if isinstance(val, (bytes, bytearray)):
        return base64.b64encode(bytes(val)).decode("ascii")
    return str(val)


def _format_timestamp(time_occurred) -> str:
    """Stable display string for exception time (local server formatting)."""
    if time_occurred is None:
        return ""
    if isinstance(time_occurred, str):
        return time_occurred.strip()
    if hasattr(time_occurred, "strftime"):
        return time_occurred.strftime("%Y-%m-%d %H:%M:%S")
    return str(time_occurred)


def _time_ago(time_occurred) -> str:
    """Format time_occurred as relative string (Just now, X mins ago, etc.)."""
    if time_occurred is None:
        return ""
    if isinstance(time_occurred, str):
        try:
            time_occurred = datetime.strptime(time_occurred, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(time_occurred)
    elif not isinstance(time_occurred, datetime):
        return ""
    time_diff = datetime.now() - time_occurred
    if time_diff < timedelta(minutes=1):
        return "Just now"
    if time_diff < timedelta(hours=1):
        return f"{int(time_diff.seconds / 60)} mins ago"
    if time_diff < timedelta(days=1):
        return f"{int(time_diff.seconds / 3600)} hours ago"
    return f"{time_diff.days} days ago"


@router.get("/notifications")
def get_notifications(db: Session = Depends(get_db)):
    """Get latest 12 exception/violation log entries with camera context and timestamps."""
    try:
        path_column_sql = (
            "el.incident_image_path,\n                    "
            if exception_logs_has_incident_image_path(db)
            else ""
        )
        result = db.execute(
            text(f"""
                SELECT
                    TOP 12
                    el.log_id,
                    et.exception_name AS Exception_Type,
                    el.Incident_image,
                    {path_column_sql}el.time_occurred,
                    el.updated_at,
                    el.camera_id,
                    c.camera_name,
                    c.zone_name,
                    c.ip_address,
                    c.streaming_url
                FROM dbo.exception_logs el
                JOIN dbo.exception_type et
                  ON et.exception_type_id = el.exception_type_id
                LEFT JOIN dbo.camera c
                  ON c.camera_id = el.camera_id
                ORDER BY el.time_occurred DESC
            """)
        )
        rows = result.mappings().all()
        notifications = []
        for row in rows:
            notification = dict(row)
            if "Exception_Type" in notification:
                notification["Exception_Type"] = _decode_bytes(notification["Exception_Type"])
            path_raw = None
            for key in ("incident_image_path", "Incident_image_path"):
                v = notification.get(key)
                if v is not None and str(v).strip():
                    path_raw = v
                    break
            if path_raw is not None:
                rel = str(path_raw).strip().replace("\\", "/").lstrip("/")
                # Root-relative so the SPA can prefix VITE_API_BASE_URL (avoids 127.0.0.1 vs LAN IP mismatches).
                notification["image_url"] = f"/static/{rel}"
                notification["Incident_image"] = ""
            elif "Incident_image" in notification:
                b64 = _incident_image_for_json(notification["Incident_image"])
                notification["Incident_image"] = b64
                if b64:
                    notification["image_url"] = f"data:image/jpeg;base64,{b64}"
            for k in ("camera_name", "zone_name", "ip_address", "streaming_url"):
                if k in notification and notification[k] is not None:
                    notification[k] = _decode_bytes(notification[k])
            # If JOIN missed (deleted camera, legacy bad FK) but camera_id exists, resolve from dbo.camera config
            cid_raw = notification.get("camera_id")
            if cid_raw is not None and not (notification.get("camera_name") or "").strip():
                try:
                    cfg_map = get_camera_config()
                    key = str(int(cid_raw)) if isinstance(cid_raw, (int, float)) else str(cid_raw).strip()
                    if key in cfg_map:
                        c = cfg_map[key]
                        notification["camera_name"] = c.get("name") or f"Camera {key}"
                        if not notification.get("zone_name"):
                            notification["zone_name"] = (c.get("description") or "").strip() or None
                        if not notification.get("ip_address"):
                            notification["ip_address"] = (c.get("ip_address") or "").strip() or None
                except Exception:
                    pass
            if "time_occurred" in notification:
                notification["time_occurred_formatted"] = _format_timestamp(
                    notification["time_occurred"]
                )
                notification["time_ago"] = _time_ago(notification["time_occurred"])
            notifications.append(notification)
        return notifications
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violation-images")
def list_violation_snapshot_images(
    limit: int = Query(20, ge=1, le=50, description="Max files to return (newest first)."),
):
    """
    List current violation snapshot JPEGs saved under media/exception_logs/detections/
    (live detection pipeline). URLs are root-relative /static/... for the SPA to resolve.
    """
    images = []
    try:
        if not _VIOLATION_SNAPSHOT_DIR.is_dir():
            return {"images": [], "directory": str(_VIOLATION_SNAPSHOT_DIR)}
        paths = sorted(
            _VIOLATION_SNAPSHOT_DIR.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        for p in paths:
            try:
                st = p.stat()
                rel = f"exception_logs/detections/{p.name}".replace("\\", "/")
                images.append(
                    {
                        "filename": p.name,
                        "url": f"/static/{rel}",
                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    }
                )
            except OSError:
                continue
        return {
            "images": images,
            "directory": str(_VIOLATION_SNAPSHOT_DIR),
        }
    except Exception as e:
        logging.error(f"violation-images listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
