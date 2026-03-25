import os
import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.users import User
from app.models.roles import Role
from app.models.user_page_permissions import UserPagePermission


# -----------------------------
# JWT config
# -----------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev_change_me__replace_in_env")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)


def _b64url_encode(data: bytes) -> str:
    """
    Base64 URL-safe encoding without padding, per JWT spec.
    """
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    """
    Base64 URL-safe decoding with automatic padding.
    """
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _jwt_hs256_sign(signing_input: str, secret: str) -> str:
    secret_bytes = secret.encode("utf-8")
    signature = hmac.new(secret_bytes, signing_input.encode(
        "ascii"), hashlib.sha256).digest()
    return _b64url_encode(signature)


def _jwt_encode_hs256(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = datetime.now(timezone.utc)

    # Keep exp/iat numeric for easy comparisons and JSON serialization.
    payload_with_claims = dict(payload)
    if "iat" not in payload_with_claims:
        payload_with_claims["iat"] = int(now.timestamp())
    if "exp" not in payload_with_claims:
        payload_with_claims["exp"] = int(
            (now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
        )

    header_b64 = _b64url_encode(json.dumps(
        header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(
        payload_with_claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}"
    signature_b64 = _jwt_hs256_sign(signing_input, secret)
    return f"{signing_input}.{signature_b64}"


def _jwt_decode_hs256(token: str, secret: str) -> Dict[str, Any]:
    segments = token.split(".")
    if len(segments) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    header_b64, payload_b64, signature_b64 = segments

    # Verify signature before trusting payload.
    signing_input = f"{header_b64}.{payload_b64}"
    expected_signature_b64 = _jwt_hs256_sign(signing_input, secret)

    # constant-time compare (decode both as bytes)
    try:
        sig_bytes = _b64url_decode(signature_b64)
        exp_sig_bytes = _b64url_decode(expected_signature_b64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")

    if not hmac.compare_digest(sig_bytes, exp_sig_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")

    try:
        payload_json = _b64url_decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_ts = int(exp)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid exp in token")
        if datetime.now(timezone.utc).timestamp() >= exp_ts:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")

    return payload


# -----------------------------
# Frontend page keys (must match frontend route guards/menu)
# -----------------------------
KNOWN_PAGE_KEYS: List[str] = [
    "dashboard.view",
    "camera-dashboard.view",
    "camera-dashboard.control",
    "model-management.view",
    "camera-management.view",
    "notifications.view",
    "profile.view",
    "admin.permissions.view",
]


# -----------------------------
# Default per-user permission seeding (in-code only; DB table is user_page_permissions)
# -----------------------------
DEFAULT_ROLE_PAGE_ALLOWED: Dict[str, List[str]] = {
    # Admin can see everything by default.
    "admin": KNOWN_PAGE_KEYS,
    # "user" is limited by default; admin can override per user via UI/API.
    "user": [
        "dashboard.view",
        "camera-dashboard.view",
        "camera-management.view",
        # no camera-dashboard.control by default (users can view feeds but not start/stop)
        "notifications.view",
        "profile.view",
        # no admin permissions page, no model management by default
    ],
}


def _normalize_role(role: Optional[str]) -> str:
    """
    Normalize role names coming from DB/token.
    Maps common admin/user variants to canonical keys used in RBAC seeding.
    """
    r = (role or "").strip().lower()
    # Normalize separators
    r = r.replace("_", " ")
    r = " ".join(r.split())

    admin_aliases = {"admin", "administrator", "super admin", "superadmin"}
    user_aliases = {"user", "viewer", "employee", "operator"}

    if r in admin_aliases:
        return "admin"
    if r in user_aliases:
        return "user"
    return r


def resolve_role_for_role_id(db: Session, role_id: Optional[int]) -> str:
    """
    Map users.role_id to canonical RBAC role string via the `roles` table.
    If the row is missing, treat role_id 1 as admin (common convention); otherwise user.
    """
    if role_id is None:
        return "user"
    role = db.query(Role).filter(Role.role_id == role_id).first()
    if role and (role.role_name or "").strip():
        return _normalize_role(role.role_name)
    if role_id == 1:
        return "admin"
    return "user"


def create_access_token(payload: Dict[str, Any]) -> str:
    if JWT_ALGORITHM != "HS256":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported JWT_ALGORITHM: {JWT_ALGORITHM}",
        )
    to_encode = dict(payload)
    to_encode.update(
        {
            "type": "access",
            # exp/iat are filled by the encoder using JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        }
    )
    return _jwt_encode_hs256(to_encode, JWT_SECRET_KEY)


def decode_access_token(token: str) -> Dict[str, Any]:
    if JWT_ALGORITHM != "HS256":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported JWT_ALGORITHM: {JWT_ALGORITHM}",
        )

    decoded = _jwt_decode_hs256(token, JWT_SECRET_KEY)
    if decoded.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    return decoded


def extract_token(request: Request) -> Optional[str]:
    # Standard for API calls from fetch:
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    # Fallback for endpoints loaded via <img src="..."> / query-only clients:
    return request.query_params.get("token")


def get_current_admin_payload(request: Request, db: Session) -> Dict[str, Any]:
    token = extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token",
        )

    decoded = decode_access_token(token)

    admin_id = decoded.get("sub")
    if admin_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    # Prefer role from token. If missing, resolve from users.role_id via roles table.
    role = decoded.get("role")
    if not role:
        user = db.query(User).filter(User.user_id == admin_id).first()
        role = resolve_role_for_role_id(db, user.role_id) if user else None

    return {
        "adminId": int(admin_id),
        "username": decoded.get("username"),
        "role": _normalize_role(role),
    }


def require_authenticated(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Dependency for endpoints that only require a valid JWT (no page permission check).
    """
    return get_current_admin_payload(request=request, db=db)


def get_user_allowed_page_keys(db: Session, user_id: int, role: str) -> List[str]:
    """
    Final source of truth for sidebar/page visibility.

    - Admin bypass: always allowed for every known page key.
    - Non-admin: allowed keys are stored per-user in `user_page_permissions`.
    """
    role_n = _normalize_role(role)
    if role_n == "admin":
        return list(KNOWN_PAGE_KEYS)

    rows = (
        db.query(UserPagePermission)
        .filter(
            UserPagePermission.user_id == user_id,
            UserPagePermission.allowed == True,
        )
        .all()
    )
    return [r.page_key for r in rows]


def require_permission(page_key: str):
    """
    Dependency factory for FastAPI endpoints.
    Checks the user's per-page permissions from `user_page_permissions`.
    """

    if page_key not in KNOWN_PAGE_KEYS:
        # Keep strict so missing/typos do not silently create broad access.
        raise ValueError(f"Unknown page_key: {page_key}")

    def dependency(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
        admin = get_current_admin_payload(request=request, db=db)
        allowed_keys = get_user_allowed_page_keys(
            db=db, user_id=admin["adminId"], role=admin["role"]
        )
        if page_key not in allowed_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )
        return admin

    return dependency


def require_role(required_role: str):
    required_role_n = _normalize_role(required_role)

    def dependency(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
        admin = get_current_admin_payload(request=request, db=db)
        if _normalize_role(admin.get("role")) != required_role_n:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )
        return admin

    return dependency


def seed_default_user_page_permissions(db: Session) -> None:
    """
    Seed per-user permissions in `user_page_permissions` based on the user's role.
    We only insert missing rows so admin overrides persist.
    """
    users = db.query(User).all()
    for user in users:
        role_n = resolve_role_for_role_id(db, user.role_id)

        if role_n == "admin":
            allowed_pages = set(KNOWN_PAGE_KEYS)
        else:
            allowed_pages = set(DEFAULT_ROLE_PAGE_ALLOWED.get(role_n, []))

        for page_key in KNOWN_PAGE_KEYS:
            existing = (
                db.query(UserPagePermission)
                .filter(
                    UserPagePermission.user_id == user.user_id,
                    UserPagePermission.page_key == page_key,
                )
                .first()
            )
            if existing is not None:
                continue

            db.add(
                UserPagePermission(
                    user_id=user.user_id,
                    page_key=page_key,
                    allowed=(page_key in allowed_pages),
                )
            )

    db.commit()
