from typing import Dict, List, Optional

from sqlalchemy import text

from app.database.database import SessionLocal


def get_camera_config(ids: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
    """
    Build camera config from camera.
    Keys are camera_id as strings to preserve current caller expectations.
    """
    db = SessionLocal()
    try:
        query = text(
            """
            SELECT camera_id, camera_name, zone_name, ip_address, streaming_url
            FROM camera
            ORDER BY camera_id
            """
        )
        rows = db.execute(query).mappings().all()
        id_filter = {str(i) for i in ids} if ids is not None else None
        config: Dict[str, Dict[str, str]] = {}
        for row in rows:
            cid = str(row.get("camera_id"))
            if id_filter is not None and cid not in id_filter:
                continue
            config[cid] = {
                "name": row.get("camera_name") or f"Camera {cid}",
                "type": "rtsp",
                "url": row.get("streaming_url"),
                "description": row.get("zone_name") or "",
                "ip_address": row.get("ip_address") or "",
            }
        return config
    finally:
        db.close()


def get_rtsp_urls(ids: Optional[List[str]] = None) -> List[str]:
    """ids=None -> all RTSP URLs from DB (sorted); else URLs for selected ids."""
    camera_config = get_camera_config(ids=ids)
    return [
        cfg["url"]
        for _, cfg in sorted(camera_config.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
        if cfg.get("type") == "rtsp" and cfg.get("url")
    ]


CAMERA_TYPES = {
    "laptop": "Built-in laptop camera",
    "rtsp": "Network RTSP camera",
    "usb": "USB camera",
    "ip": "IP camera",
}
