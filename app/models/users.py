from sqlalchemy import Column, Integer, String

from app.database.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100))
    password = Column(String(255))
    role_id = Column(Integer, nullable=False, index=True)
