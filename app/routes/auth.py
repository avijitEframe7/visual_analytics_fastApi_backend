from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.models.admins import Admin
from app.schemas.auth import LoginRequest
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

    return {
        "success": True,
        "message": "Login successful!!!",
        "user": {
            "adminId": admin.AdminId,
            "username": admin.Username,
        },
    }
