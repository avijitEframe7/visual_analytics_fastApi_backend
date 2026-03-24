from sqlalchemy import Column, Integer, String

from app.database.database import Base


class Role(Base):
    """
    Lookup table for user.role_id. Adjust `role_name` if your DB uses a different column name.
    """

    __tablename__ = "roles"

    role_id = Column(Integer, primary_key=True)
    role_name = Column(String(50), nullable=False)
