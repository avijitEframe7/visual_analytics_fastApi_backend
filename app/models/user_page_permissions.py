from sqlalchemy import Column, Integer, String, Boolean, UniqueConstraint

from app.database.database import Base


class UserPagePermission(Base):
    """
    Per-user page permissions (final source of truth for sidebar visibility).

    Admin bypass is handled in RBAC logic, but we still store values here so the
    UI can be consistent and permissions are auditable.
    """

    __tablename__ = "user_page_permissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    page_key = Column(String(120), nullable=False, index=True)
    allowed = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("user_id", "page_key", name="uq_user_page_key"),
    )

