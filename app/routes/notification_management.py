import glob
import json
import logging
import os
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/notification_management", tags=["notification_management"])

NOTIFICATION_LOGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs")
)
NOTIFICATION_LOG_SINGLE = os.path.join(NOTIFICATION_LOGS_DIR, "notification_logs.json")
NOTIFICATION_LOG_GLOB = os.path.join(NOTIFICATION_LOGS_DIR, "notification_logs_camera_*.json")
NOTIFICATION_MAX_RETURN = 50


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


def _parse_time(s: str) -> datetime:
    """Parse time_occurred string for sorting."""
    if not s:
        return datetime.min
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.min


def _load_log_file(path: str) -> list:
    """Load a notification log file; return list of entries or empty list."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError, OSError) as e:
        logging.warning(f"Could not read {path}: {e}")
        return []
    if not isinstance(data, list):
        return []
    return data


@router.get("/notifications")
def get_notifications():
    """Get latest notification log entries from notification_logs.json and per-camera files (merged, newest first)."""
    try:
        combined = []
        # Single file: notification_logs.json
        if os.path.isfile(NOTIFICATION_LOG_SINGLE):
            combined.extend(_load_log_file(NOTIFICATION_LOG_SINGLE))
        # Per-camera files: notification_logs_camera_*.json
        for path in glob.glob(NOTIFICATION_LOG_GLOB):
            combined.extend(_load_log_file(path))
        # Sort by time_occurred descending (newest first), take up to NOTIFICATION_MAX_RETURN
        combined.sort(key=lambda e: _parse_time(e.get("time_occurred") or ""), reverse=True)
        latest = combined[:NOTIFICATION_MAX_RETURN]
        notifications = []
        for entry in latest:
            notification = dict(entry)
            if "time_occurred" in notification:
                notification["time_ago"] = _time_ago(notification["time_occurred"])
            notifications.append(notification)
        return notifications
    except Exception as e:
        logging.error(f"Notification log error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
