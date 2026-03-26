from typing import Dict, Optional, List, Tuple

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.models.users import User
from app.models.roles import Role
from app.models.user_page_permissions import UserPagePermission
from app.schemas.auth import LoginRequest
from app.security.rbac import (
    KNOWN_PAGE_KEYS,
    create_access_token,
    get_user_allowed_page_keys,
    require_authenticated,
    require_role,
    resolve_role_for_role_id,
    seed_default_user_page_permissions,
)
from pydantic import BaseModel
import hashlib

logger = logging.getLogger(__name__)

def sha256_hash_with_encoding(password: str, encoding: str) -> str:
    return hashlib.sha256(password.encode(encoding)).hexdigest()


def _normalize_password_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_sha256_hex(value: Optional[str]) -> str:
    """
    Normalize stored/derived SHA256 values for comparison:
    - trim whitespace
    - strip optional 0x prefix
    - case-insensitive
    """
    v = _normalize_password_value(value)
    if v.lower().startswith("0x"):
        v = v[2:]
    return v.lower()


def _sha256_candidates(plain_text: str) -> List[str]:
    """
    Return normalized (lowercase, hex-only) SHA256 values for multiple encodings.
    """
    pt = plain_text or ""
    candidates: List[str] = []
    for encoding in ("utf-8", "utf-16le", "cp1252", "latin1", "ascii"):
        try:
            h = sha256_hash_with_encoding(pt, encoding)
        except Exception:
            continue
        nh = _normalize_sha256_hex(h)
        if nh and nh not in candidates:
            candidates.append(nh)
    return candidates


def _sha256_candidate_encodings(plain_text: str) -> List[Tuple[str, str]]:
    """
    Return list of (encoding_name, normalized_sha256_hex) candidates.
    """
    pt = plain_text or ""
    out: List[Tuple[str, str]] = []
    for encoding in ("utf-8", "utf-16le", "cp1252", "latin1", "ascii"):
        try:
            h = sha256_hash_with_encoding(pt, encoding)
        except Exception:
            continue
        nh = _normalize_sha256_hex(h)
        if nh:
            out.append((encoding, nh))
    return out


def _parse_stored_password(stored_raw) -> Tuple[Optional[str], str]:
    """
    Read password as stored by SQL Server / ODBC.

    - VARBINARY(32) raw SHA256 digest -> (64-char hex lower, "")
    - VARCHAR(64) hex from CONVERT(..., HASHBYTES(...), 2) -> (hex lower, "")
    - Legacy plain text -> (None, plaintext)

    Important: if `stored_raw` is bytes of length 32, never use str(bytes) — that breaks compares.
    """
    if stored_raw is None:
        return (None, "")

    if isinstance(stored_raw, (bytes, bytearray, memoryview)):
        b = bytes(stored_raw)
        if len(b) == 32:
            return (b.hex().lower(), "")
        if len(b) == 64:
            try:
                s = b.decode("ascii").strip()
                if len(s) == 64 and all(
                    c in "0123456789abcdefABCDEF" for c in s
                ):
                    return (_normalize_sha256_hex(s), "")
            except Exception:
                pass
        return (None, "")

    s = str(stored_raw).strip()
    h = _normalize_sha256_hex(s)
    if len(h) == 64 and all(c in "0123456789abcdef" for c in h):
        return (h, "")
    return (None, s)


router = APIRouter(prefix="/api", tags=["Auth"])

@router.post("/auth/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    username_in = (data.username or "").strip()
    user = (
        db.query(User)
        .filter(
            User.user_id == data.adminId,
            User.username == username_in,
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid id or username"
        )
    # Verify password (one-way): compare SHA256(entered) to stored hash.
    # SQL Server may store HASHBYTES as VARBINARY(32) — ODBC returns bytes; handle that in _parse_stored_password.
    entered = data.password or ""
    stored_sha, legacy_plain = _parse_stored_password(user.password)
    matched_encoding: Optional[str] = None

    def _match_sha256_for_plain(plain: str) -> None:
        nonlocal matched_encoding
        if not stored_sha or len(stored_sha) != 64:
            return
        for enc, enc_hex in _sha256_candidate_encodings(plain):
            if stored_sha == enc_hex:
                matched_encoding = enc
                return

    _match_sha256_for_plain(entered)
    trimmed = entered.strip()
    if matched_encoding is None and trimmed and trimmed != entered:
        _match_sha256_for_plain(trimmed)

    is_password_match = matched_encoding is not None

    if not is_password_match and legacy_plain:
        if legacy_plain == entered or legacy_plain == entered.strip():
            is_password_match = True

    if not is_password_match:
        logger.warning(
            "Login failed: password mismatch for user_id=%s username=%s stored_sha_len=%s",
            user.user_id,
            user.username,
            len(stored_sha) if stored_sha else 0,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )
    logger.info(
        "Login password matched encoding=%s for user_id=%s",
        matched_encoding or "legacy-plaintext",
        user.user_id,
    )

    role = resolve_role_for_role_id(db, user.role_id)
    token = create_access_token(
        {
            "sub": user.user_id,
            "username": user.username,
            "role": role,
        }
    )

    return {
        "success": True,
        "message": "Login successful!!!",
        "user": {
            "adminId": user.user_id,
            "username": user.username,
            "role": role,
        },
        "token": token,
    }


@router.get("/auth/me")
def me(admin: Dict = Depends(require_authenticated), db: Session = Depends(get_db)):
    """
    Return current user from DB for username/role so profile edits show up without re-login.
    JWT may still carry an old username claim until the next login.
    """
    user = db.query(User).filter(User.user_id == admin["adminId"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    role_n = resolve_role_for_role_id(db, user.role_id)
    permissions = get_user_allowed_page_keys(
        db=db, user_id=user.user_id, role=role_n
    )
    return {
        "adminId": user.user_id,
        "username": user.username,
        "role": role_n,
        "permissions": permissions,
    }


class UserPagePermissionsUpdateRequest(BaseModel):
    userId: int
    # Map of page_key -> allowed (true/false)
    pages: Dict[str, bool]


class UpdateProfileRequest(BaseModel):
    username: str


class ChangePasswordRequest(BaseModel):
    # Kept to match the frontend modal payload. Server uses JWT to decide the user.
    adminId: int
    username: str
    currentPassword: str
    newPassword: str


class AdminUserCreateRequest(BaseModel):
    username: str
    password: str
    role_id: int


class AdminUserUpdateRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role_id: Optional[int] = None


@router.get("/admin/roles")
def admin_list_roles(
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    rows = db.query(Role).order_by(Role.role_id).all()
    return [{"role_id": r.role_id, "role_name": r.role_name} for r in rows]


@router.post("/admin/users")
def admin_create_user(
    data: AdminUserCreateRequest,
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    username = (data.username or "").strip()
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="username is required",
        )
    role = db.query(Role).filter(Role.role_id == data.role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role_id",
        )
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    plain = data.password or ""
    if not plain.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="password is required",
        )
    password_hashed = sha256_hash_with_encoding(plain.strip(), "utf-8").upper()
    user = User(username=username, password=password_hashed, role_id=data.role_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    seed_default_user_page_permissions(db)
    return {
        "success": True,
        "user_id": user.user_id,
        "username": user.username,
        "role_id": user.role_id,
    }


@router.put("/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    data: AdminUserUpdateRequest,
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if data.username is not None:
        new_name = (data.username or "").strip()
        if not new_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="username cannot be empty",
            )
        taken = (
            db.query(User)
            .filter(User.username == new_name, User.user_id != user_id)
            .first()
        )
        if taken:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )
        user.username = new_name

    if data.role_id is not None:
        role = db.query(Role).filter(Role.role_id == data.role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role_id",
            )
        user.role_id = data.role_id

    if data.password is not None and str(data.password).strip():
        plain = str(data.password).strip()
        user.password = sha256_hash_with_encoding(plain, "utf-8").upper()

    db.commit()
    db.refresh(user)
    return {
        "success": True,
        "user_id": user.user_id,
        "username": user.username,
        "role_id": user.role_id,
    }


@router.get("/admin/user-page-permissions")
def get_user_page_permissions(
    userId: int = Query(..., description="User id"),
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.user_id == userId).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    role_n = resolve_role_for_role_id(db, user.role_id)
    allowed_keys = set(
        get_user_allowed_page_keys(db=db, user_id=user.user_id, role=role_n)
    )

    pages: Dict[str, bool] = {}
    for pk in KNOWN_PAGE_KEYS:
        pages[pk] = pk in allowed_keys

    return {
        "userId": user.user_id,
        "role": role_n,
        "is_admin": role_n == "admin",
        "pages": pages,
    }


@router.post("/admin/user-page-permissions")
def update_user_page_permissions(
    data: UserPagePermissionsUpdateRequest,
    admin_auth: Dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.user_id == data.userId).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Only allow updating known page keys to avoid accidental privilege grants.
    for page_key, allowed in data.pages.items():
        if page_key not in KNOWN_PAGE_KEYS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown page_key: {page_key}",
            )

        existing: Optional[UserPagePermission] = (
            db.query(UserPagePermission)
            .filter(
                UserPagePermission.user_id == data.userId,
                UserPagePermission.page_key == page_key,
            )
            .first()
        )

        if existing is None:
            db.add(
                UserPagePermission(
                    user_id=data.userId,
                    page_key=page_key,
                    allowed=bool(allowed),
                )
            )
        else:
            existing.allowed = bool(allowed)

    db.commit()
    return {"success": True, "userId": data.userId, "pages": data.pages}


@router.post("/auth/update-profile")
def update_profile(
    data: UpdateProfileRequest,
    admin: Dict = Depends(require_authenticated),
    db: Session = Depends(get_db),
):
    """
    Update basic user profile details stored in `dbo.users`.
    Currently, only `username` is supported by your frontend profile UI.
    """
    new_username = (data.username or "").strip()
    if not new_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="username is required",
        )

    user = db.query(User).filter(User.user_id == admin["adminId"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.username = new_username
    db.commit()

    # Keep response compatible with `/auth/me` consumers.
    return {"success": True, "username": user.username}


@router.post("/auth/change-password")
def change_password(
    data: ChangePasswordRequest,
    admin: Dict = Depends(require_authenticated),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.user_id == admin["adminId"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Do NOT strip current password: must match whatever was hashed in SQL.
    entered_current = data.currentPassword or ""
    stored_sha, legacy_plain = _parse_stored_password(user.password)

    candidates = _sha256_candidates(entered_current)
    is_current_match = False
    # If current matches one of the SHA candidates, keep track of which encoding matched.
    matched_encoding: Optional[str] = None

    if stored_sha and len(stored_sha) == 64 and stored_sha in candidates:
        is_current_match = True
        for enc, enc_hex in _sha256_candidate_encodings(entered_current):
            if stored_sha == enc_hex:
                matched_encoding = enc
                break
    elif legacy_plain and (
        legacy_plain == entered_current
        or legacy_plain == entered_current.strip()
    ):
        is_current_match = True
        matched_encoding = "utf-8"

    if not is_current_match:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid current password",
        )

    # Always store the new password as SHA256(UTF-8) hex so it matches login verification
    # (login compares against utf-8 / latin1 / utf-16le candidates; utf-8 is the canonical
    # choice after an in-app password change). Using matched_encoding here could store
    # utf-16le hashes that confuse users expecting the same password to work at login.
    new_plain = (data.newPassword or "").strip()
    if not new_plain:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password is required",
        )
    new_password_hashed = sha256_hash_with_encoding(new_plain, "utf-8").upper()

    user.password = new_password_hashed
    db.commit()

    return {"success": True, "message": "Password changed successfully"}
