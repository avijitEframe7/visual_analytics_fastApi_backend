from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import date as date_type
from datetime import datetime, timedelta
import calendar
from collections import Counter
from typing import Any, List, Dict
import io

from fastapi.responses import StreamingResponse
from app.database.database import get_db
from app.security.rbac import require_permission

# Helper functions
def decode_bytes(val: Any) -> str:
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return str(val)
    return str(val)


def safe_datetime_str(val: Any) -> str:
    if val is None:
        return str(val)
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)


def _row_to_date(val: Any):
    """Normalize SQL date/datetime for comparison with Python date (weekly trend)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date_type):
        return val
    return val


def _row_period_int(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(float(val))

router = APIRouter(
    prefix="/api/main_dashboard",
    tags=["Main Dashboard"],
    dependencies=[Depends(require_permission("dashboard.view"))],
)

@router.get("/trend_analysis")
def fetch_logs_by_trend_analysis(
    time_range: str = Query(..., alias="range", description="weekly | monthly | quarterly | yearly"),
    db: Session = Depends(get_db)
):
    # ----------------------------
    # 1️⃣ Select query by range
    # ----------------------------
    # MySQL: group by selected expressions / subquery columns as needed for each range.
    if time_range == "weekly":
        query = text("""
            SELECT
                DAYNAME(el.time_occurred) AS label,
                DATE(el.time_occurred) AS date_val,
                COUNT(*) AS count
            FROM exception_logs el
            WHERE el.time_occurred >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY DATE(el.time_occurred), DAYNAME(el.time_occurred)
            ORDER BY MIN(DATE(el.time_occurred))
        """)

    elif time_range == "monthly":
        query = text("""
            SELECT
                wk AS period,
                CONCAT('Week ', wk) AS label,
                COUNT(*) AS count
            FROM (
                SELECT
                    FLOOR((DAY(el.time_occurred) - 1) / 7) + 1 AS wk
                FROM exception_logs el
                WHERE MONTH(el.time_occurred) = MONTH(CURDATE())
                  AND YEAR(el.time_occurred) = YEAR(CURDATE())
            ) t
            GROUP BY wk
            ORDER BY wk
        """)

    elif time_range == "quarterly":
        query = text("""
            SELECT
                QUARTER(el.time_occurred) AS period,
                CASE QUARTER(el.time_occurred)
                    WHEN 1 THEN 'Q1 (Jan-Mar)'
                    WHEN 2 THEN 'Q2 (Apr-Jun)'
                    WHEN 3 THEN 'Q3 (Jul-Sep)'
                    WHEN 4 THEN 'Q4 (Oct-Dec)'
                END AS label,
                COUNT(*) AS count
            FROM exception_logs el
            WHERE YEAR(el.time_occurred) = YEAR(CURDATE())
            GROUP BY QUARTER(el.time_occurred)
            ORDER BY QUARTER(el.time_occurred)
        """)

    elif time_range == "yearly":
        query = text("""
            SELECT
                MONTHNAME(el.time_occurred) AS label,
                MONTH(el.time_occurred) AS period,
                COUNT(*) AS count
            FROM exception_logs el
            WHERE YEAR(el.time_occurred) = YEAR(CURDATE())
            GROUP BY MONTH(el.time_occurred), MONTHNAME(el.time_occurred)
            ORDER BY MONTH(el.time_occurred)
        """)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid range. Use weekly, monthly, quarterly, or yearly."
        )

    # ----------------------------
    # 2️⃣ Execute query (SQLAlchemy)
    # ----------------------------
    try:
        results = db.execute(query).mappings().all()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}",
        )

    # ----------------------------
    # 3️⃣ Build full data ranges
    # ----------------------------
    data = []

    if time_range == "weekly":
        today = datetime.now().date()

        for i in range(7):
            current_date = today - timedelta(days=6 - i)
            label = current_date.strftime("%A")

            row = next(
                (
                    r
                    for r in results
                    if _row_to_date(r.get("date_val")) == current_date
                ),
                None,
            )

            data.append({
                "label": f"{label} ({current_date})",
                "day": label,
                "date": str(current_date),
                "count": int(row["count"]) if row else 0,
            })

    elif time_range == "monthly":
        current_year = datetime.now().year
        current_month = datetime.now().month
        _, last_day = calendar.monthrange(current_year, current_month)

        weeks = sorted(
            set(((day - 1) // 7) + 1 for day in range(1, last_day + 1))
        )

        for week in weeks:
            row = next(
                (r for r in results if _row_period_int(r.get("period")) == week),
                None,
            )
            data.append({
                "label": f"Week {week}",
                "week_number": week,
                "count": int(row["count"]) if row else 0,
            })

    elif time_range == "quarterly":
        for q in range(1, 5):
            row = next(
                (r for r in results if _row_period_int(r.get("period")) == q),
                None,
            )
            data.append({
                "label": row["label"] if row else f"Q{q}",
                "quarter": q,
                "count": int(row["count"]) if row else 0,
            })

    elif time_range == "yearly":
        months = [
            ("January", 1),
            ("February", 2),
            ("March", 3),
            ("April", 4),
            ("May", 5),
            ("June", 6),
            ("July", 7),
            ("August", 8),
            ("September", 9),
            ("October", 10),
            ("November", 11),
            ("December", 12),
        ]

        for name, num in months:
            row = next(
                (
                    r
                    for r in results
                    if _row_period_int(r.get("period")) == num
                ),
                None,
            )
            lbl = decode_bytes(row["label"]) if row and row.get("label") is not None else name
            data.append({
                "label": lbl,
                "month": num,
                "count": int(row["count"]) if row else 0,
            })

    # ----------------------------
    # 4️⃣ Response
    # ----------------------------
    return {
        "range": time_range,
        "total_records": len(data),
        "data": data
    }


@router.get("/exception_piechart")
def get_exception_piechart(
    time_range: str = Query("all"),
    db: Session = Depends(get_db)
):
    query = """
        SELECT et.exception_name, COUNT(*)
        FROM exception_logs el
        JOIN exception_type et
          ON et.exception_type_id = el.exception_type_id
    """

    if time_range == "day":
        query += " WHERE el.time_occurred >= DATE_SUB(NOW(), INTERVAL 1 DAY)"
    elif time_range == "week":
        query += " WHERE el.time_occurred >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
    elif time_range == "month":
        query += " WHERE el.time_occurred >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
    elif time_range == "quarter":
        query += " WHERE el.time_occurred >= DATE_SUB(NOW(), INTERVAL 3 MONTH)"
    elif time_range == "year":
        query += " WHERE el.time_occurred >= DATE_SUB(NOW(), INTERVAL 1 YEAR)"

    query += " GROUP BY et.exception_name"

    result = db.execute(text(query)).fetchall()

    return [
        {"label": decode_bytes(r[0]), "value": int(r[1])}
        for r in result
    ]


def _exception_logs_time_sql_fragment(time_range: str, table_alias: str = "el") -> str:
    """Append to WHERE ... for exception_logs time filtering (matches exception_piechart)."""
    col = f"{table_alias}.time_occurred"
    if time_range == "all" or not time_range:
        return ""
    if time_range == "day":
        return f" AND {col} >= DATE_SUB(NOW(), INTERVAL 1 DAY)"
    if time_range == "week":
        return f" AND {col} >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
    if time_range == "month":
        return f" AND {col} >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
    if time_range == "quarter":
        return f" AND {col} >= DATE_SUB(NOW(), INTERVAL 3 MONTH)"
    if time_range == "year":
        return f" AND {col} >= DATE_SUB(NOW(), INTERVAL 1 YEAR)"
    raise HTTPException(
        status_code=400,
        detail="Invalid time_range. Use all, day, week, month, quarter, or year.",
    )


@router.get("/camera_zone_violations")
def get_camera_zone_violations(
    time_range: str = Query("week", description="all | day | week | month | quarter | year"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Violation counts by zone and by camera for dashboard charts.
    Uses exception_logs + camera; includes all cameras with zero counts in by_camera.
    """
    tr = (time_range or "week").strip().lower()
    time_and = _exception_logs_time_sql_fragment(tr, "el")

    # --- By zone (includes unknown camera / unassigned zone buckets)
    zone_sql = f"""
        SELECT
            CASE
                WHEN el.camera_id IS NULL THEN 'Unknown camera'
                ELSE IFNULL(NULLIF(TRIM(c.zone_name), ''), 'Unassigned')
            END AS zone_name,
            COUNT(*) AS violation_count
        FROM exception_logs el
        LEFT JOIN camera c ON c.camera_id = el.camera_id
        WHERE 1 = 1
        {time_and}
        GROUP BY
            CASE
                WHEN el.camera_id IS NULL THEN 'Unknown camera'
                ELSE IFNULL(NULLIF(TRIM(c.zone_name), ''), 'Unassigned')
            END
        ORDER BY violation_count DESC
    """

    # --- By camera (every row in camera; zero if no matching logs in range)
    join_time = _exception_logs_time_sql_fragment(tr, "el")

    camera_sql = f"""
        SELECT
            c.camera_id,
            c.camera_name,
            c.zone_name,
            COUNT(el.log_id) AS violation_count
        FROM camera c
        LEFT JOIN exception_logs el ON el.camera_id = c.camera_id{join_time}
        GROUP BY c.camera_id, c.camera_name, c.zone_name
        ORDER BY violation_count DESC, c.camera_id
    """

    try:
        zone_rows = db.execute(text(zone_sql)).fetchall()
        cam_rows = db.execute(text(camera_sql)).fetchall()
        total_cams = db.execute(text("SELECT COUNT(*) AS n FROM camera")).scalar()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    by_zone: List[Dict[str, Any]] = []
    for r in zone_rows:
        by_zone.append(
            {
                "zone": decode_bytes(r[0]),
                "count": int(r[1]),
            }
        )

    by_camera: List[Dict[str, Any]] = []
    for r in cam_rows:
        by_camera.append(
            {
                "camera_id": int(r[0]) if r[0] is not None else None,
                "camera_name": decode_bytes(r[1]) if r[1] is not None else "",
                "zone_name": decode_bytes(r[2]) if r[2] is not None else "",
                "count": int(r[3]),
            }
        )

    return {
        "time_range": tr,
        "total_cameras": int(total_cams or 0),
        "by_zone": by_zone,
        "by_camera": by_camera,
    }


@router.get("/bargraph-user-exception-counts")
def get_user_exception_counts(db: Session = Depends(get_db)):
    """
    Kept route path for frontend compatibility.
    New exception_logs schema uses exception_type_id; resolve labels from exception_type table.
    """
    result = db.execute(text("""
        SELECT et.exception_name, COUNT(*)
        FROM exception_logs el
        JOIN exception_type et
          ON et.exception_type_id = el.exception_type_id
        GROUP BY et.exception_name
    """)).fetchall()

    return {
        "usernames": [decode_bytes(r[0]) for r in result],
        "exception_counts": [int(r[1]) for r in result]
    }

@router.get("/exception-heatmap")
def exception_heatmap(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT time_occurred
        FROM exception_logs
    """)).fetchall()

    timestamps = [safe_datetime_str(r[0]) for r in rows]
    counter = Counter(timestamps)

    x = list(counter.keys())
    y = list(counter.values())

    return {
        "x": x,
        "y": y,
        "max_count": max(y) if y else 0,
        "max_time": x[y.index(max(y))] if y else None
    }
