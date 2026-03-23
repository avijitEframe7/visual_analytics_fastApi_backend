from sqlalchemy import Column, Integer, String
from app.database.database import Base

class Admin(Base):
    __tablename__ = "admins"

    AdminId = Column(Integer, primary_key=True, index=True)
    Username = Column(String(100))
    Password = Column(String(255))  # hashed password (recommended)
    # Role is expected to exist in the `admins` table (e.g. 'admin' or 'user')
    Role = Column(String(50), nullable=False, default="user", index=True)
