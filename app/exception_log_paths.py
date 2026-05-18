"""
Feature-detect optional dbo.exception_logs.incident_image_path column (disk path vs blob).

Used by notification_management and camera_dashboard INSERT/SELECT paths.
"""

from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

_cached_has_path_column: Optional[bool] = None


def exception_logs_has_incident_image_path(db: Session) -> bool:
    """
    Return True if dbo.exception_logs has incident_image_path (SQL Server).

    Result is cached per process after the first successful check so the worker
    does not query sys tables on every violation insert.
    """
    global _cached_has_path_column
    if _cached_has_path_column is not None:
        return _cached_has_path_column

    try:
        row = db.execute(
            text(
                """
                SELECT 1 AS ok
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'dbo'
                  AND TABLE_NAME = 'exception_logs'
                  AND COLUMN_NAME = 'incident_image_path'
                """
            )
        ).fetchone()
        _cached_has_path_column = row is not None
    except Exception:
        _cached_has_path_column = False

    return _cached_has_path_column
