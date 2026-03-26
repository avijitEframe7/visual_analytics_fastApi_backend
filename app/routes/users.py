from fastapi import Depends, APIRouter, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.database import get_db   # IMPORTANT
from app.security.rbac import require_role, resolve_role_for_role_id

router = APIRouter(dependencies=[Depends(require_role("admin"))])


@router.get("/get_users")
def get_users(
    exclude_admins: bool = Query(
        False,
        description="If true, omit users whose role resolves to admin (for Access Control user picker).",
    ),
    db: Session = Depends(get_db),
):
    result = db.execute(text("SELECT * FROM dbo.users"))
    rows = result.mappings().all()
    if not exclude_admins:
        return rows
    out = []
    for row in rows:
        rid = row.get("role_id")
        if resolve_role_for_role_id(db, rid) != "admin":
            out.append(dict(row))
    return out

# @router.get()
