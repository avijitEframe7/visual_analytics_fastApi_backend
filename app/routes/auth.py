from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.models.admins import Admin
from app.models.role_page_permissions import RolePagePermission
from app.schemas.auth import LoginRequest
from app.security.rbac import (
    KNOWN_PAGE_KEYS,
    create_access_token,
    get_role_allowed_page_keys,
    require_authenticated,
    require_role,
)
from pydantic import BaseModel
import hashlib

def sha256_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

router = APIRouter(prefix="/api", tags=["Auth"])

@router.post("/auth/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    admin = (
        db.query(Admin)
        .filter(
            Admin.AdminId == data.adminId,
            Admin.Username == data.username
        )
        .first()
    )

    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid id or username"
        )
    hashed_password = sha256_hash(data.password)
    # Plain-text comparison shown for learning
    if admin.Password != hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )

    role = (admin.Role or "").strip().lower()
    token = create_access_token(
        {
            "sub": admin.AdminId,
            "username": admin.Username,
            "role": role,
        }
    )

    return {
        "success": True,
        "message": "Login successful!!!",
        "user": {
            "adminId": admin.AdminId,
            "username": admin.Username,
            "role": role,
        },
        "token": token,
    }


@router.get("/auth/me")
def me(admin: Dict = Depends(require_authenticated), db: Session = Depends(get_db)):
    permissions = get_role_allowed_page_keys(db=db, role=admin["role"])
    return {
        "adminId": admin["adminId"],
        "username": admin.get("username"),
        "role": admin.get("role"),
        "permissions": permissions,
    }


class RolePagePermissionsUpdateRequest(BaseModel):
    role: str
    # Map of page_key -> allowed (true/false)
    pages: Dict[str, bool]


@router.get("/admin/role-page-permissions")
def get_role_page_permissions(
    role: str = Query(..., description="Role name like 'user' or 'admin'"),
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    role_n = role.strip().lower()
    pages: Dict[str, bool] = {}

    if role_n == "admin":
        for pk in KNOWN_PAGE_KEYS:
            pages[pk] = True
    else:
        allowed_keys = set(get_role_allowed_page_keys(db=db, role=role_n))
        for pk in KNOWN_PAGE_KEYS:
            pages[pk] = pk in allowed_keys

    return {"role": role_n, "pages": pages}


@router.post("/admin/role-page-permissions")
def update_role_page_permissions(
    data: RolePagePermissionsUpdateRequest,
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    role_n = data.role.strip().lower()

    # Only allow updating known page keys to avoid accidental privilege grants.
    for page_key, allowed in data.pages.items():
        if page_key not in KNOWN_PAGE_KEYS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown page_key: {page_key}",
            )

        existing: Optional[RolePagePermission] = (
            db.query(RolePagePermission)
            .filter(RolePagePermission.role == role_n, RolePagePermission.page_key == page_key)
            .first()
        )
        if existing is None:
            db.add(RolePagePermission(role=role_n, page_key=page_key, allowed=bool(allowed)))
        else:
            existing.allowed = bool(allowed)

    db.commit()
    return {"success": True, "role": role_n, "pages": data.pages}
