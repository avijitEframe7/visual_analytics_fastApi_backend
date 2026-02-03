from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
import calendar
from collections import Counter
from typing import Any
import io

from fastapi.responses import StreamingResponse
from app.database.database import get_db

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

router = APIRouter(
    prefix="/api/main_dashboard",
    tags=["Main Dashboard"]
)

@router.get("/trend_analysis")
def fetch_logs_by_trend_analysis(
    time_range: str = Query(..., alias="range", description="weekly | monthly | quarterly | yearly"),
    db: Session = Depends(get_db)
):
    # ----------------------------
    # 1️⃣ Select query by range
    # ----------------------------
    if time_range == "weekly":
        query = text("""
            SELECT 
                DAYNAME(time_occurred) AS label,
                DATE(time_occurred) AS date_val,
                COUNT(*) AS count
            FROM exception_logs
            WHERE time_occurred >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY DATE(time_occurred), DAYNAME(time_occurred)
            ORDER BY DATE(time_occurred)
        """)

    elif time_range == "monthly":
        query = text("""
            SELECT 
                CONCAT('Week ', FLOOR((DAY(time_occurred) - 1) / 7) + 1) AS label,
                FLOOR((DAY(time_occurred) - 1) / 7) + 1 AS period,
                COUNT(*) AS count
            FROM exception_logs
            WHERE MONTH(time_occurred) = MONTH(NOW())
              AND YEAR(time_occurred) = YEAR(NOW())
            GROUP BY period, label
            ORDER BY period
        """)

    elif time_range == "quarterly":
        query = text("""
            SELECT 
                CASE 
                    WHEN QUARTER(time_occurred) = 1 THEN 'Q1 (Jan-Mar)'
                    WHEN QUARTER(time_occurred) = 2 THEN 'Q2 (Apr-Jun)'
                    WHEN QUARTER(time_occurred) = 3 THEN 'Q3 (Jul-Sep)'
                    WHEN QUARTER(time_occurred) = 4 THEN 'Q4 (Oct-Dec)'
                END AS label,
                QUARTER(time_occurred) AS period,
                COUNT(*) AS count
            FROM exception_logs
            WHERE YEAR(time_occurred) = YEAR(NOW())
            GROUP BY period, label
            ORDER BY period
        """)

    elif time_range == "yearly":
        query = text("""
            SELECT 
                MONTHNAME(time_occurred) AS label,
                MONTH(time_occurred) AS period,
                COUNT(*) AS count
            FROM exception_logs
            WHERE YEAR(time_occurred) = YEAR(NOW())
            GROUP BY period, label
            ORDER BY period
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
                (r for r in results if str(r["date_val"]) == str(current_date)),
                None
            )

            data.append({
                "label": f"{label} ({current_date})",
                "day": label,
                "date": str(current_date),
                "count": int(row["count"]) if row else 0
            })

    elif time_range == "monthly":
        current_year = datetime.now().year
        current_month = datetime.now().month
        _, last_day = calendar.monthrange(current_year, current_month)

        weeks = sorted(set(
            ((day - 1) // 7) + 1 for day in range(1, last_day + 1)
        ))

        for week in weeks:
            row = next((r for r in results if r["period"] == week), None)
            data.append({
                "label": f"Week {week}",
                "week_number": week,
                "count": int(row["count"]) if row else 0
            })

    elif time_range == "quarterly":
        for q in range(1, 5):
            row = next((r for r in results if r["period"] == q), None)
            data.append({
                "label": row["label"] if row else f"Q{q}",
                "quarter": q,
                "count": int(row["count"]) if row else 0
            })

    elif time_range == "yearly":
        months = [
            ("January", 1), ("February", 2), ("March", 3),
            ("April", 4), ("May", 5), ("June", 6),
            ("July", 7), ("August", 8), ("September", 9),
            ("October", 10), ("November", 11), ("December", 12)
        ]

        for name, num in months:
            row = next((r for r in results if r["period"] == num), None)
            data.append({
                "label": name,
                "month": num,
                "count": int(row["count"]) if row else 0
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
    query = "SELECT Exception_Type, COUNT(*) FROM exception_logs"

    if time_range == "day":
        query += " WHERE time_occurred >= NOW() - INTERVAL 1 DAY"
    elif time_range == "week":
        query += " WHERE time_occurred >= NOW() - INTERVAL 7 DAY"
    elif time_range == "month":
        query += " WHERE time_occurred >= NOW() - INTERVAL 1 MONTH"
    elif time_range == "quarter":
        query += " WHERE time_occurred >= NOW() - INTERVAL 3 MONTH"
    elif time_range == "year":
        query += " WHERE time_occurred >= NOW() - INTERVAL 1 YEAR"

    query += " GROUP BY Exception_Type"

    result = db.execute(text(query)).fetchall()

    return [
        {"label": decode_bytes(r[0]), "value": int(r[1])}
        for r in result
    ]

@router.get("/bargraph-user-exception-counts")
def get_user_exception_counts(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT Username, COUNT(*) 
        FROM exception_logs
        GROUP BY Username
    """)).fetchall()

    return {
        "usernames": [decode_bytes(r[0]) for r in result],
        "exception_counts": [int(r[1]) for r in result]
    }

@router.get("/exception-heatmap")
def exception_heatmap(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT time_occurred, Exception_Type
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

@router.get("/exception-heatmap")
def exception_heatmap(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT time_occurred, Exception_Type
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
