from sqlalchemy import Column, Integer, String, Boolean, UniqueConstraint

from app.database.database import Base


class RolePagePermission(Base):
    """
    RBAC page permissions configured by admins.

    We store permissions per role (not per individual user) because your roles are already
    defined via the `admins.Role` column.
    """

    __tablename__ = "role_page_permissions"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50), nullable=False, index=True)  # normalized lower-case
    page_key = Column(String(120), nullable=False, index=True)
    allowed = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("role", "page_key", name="uq_role_page_key"),
    )

