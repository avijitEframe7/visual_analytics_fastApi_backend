import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database.database import get_db

router = APIRouter(prefix="/api/notification_management", tags=["notification_management"])


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
    """Get latest 12 exception log entries with relative time."""
    try:
        result = db.execute(
            text("""
                SELECT Exception_Type, time_occurred
                FROM employeeinfo.exception_logs
                ORDER BY time_occurred DESC
                LIMIT 12
            """)
        )
        rows = result.mappings().all()
        notifications = []
        for row in rows:
            notification = dict(row)
            if "time_occurred" in notification:
                notification["time_ago"] = _time_ago(notification["time_occurred"])
            notifications.append(notification)
        return notifications
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
