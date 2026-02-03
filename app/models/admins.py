from sqlalchemy import Column, Integer, String
from app.database.database import Base

class Admin(Base):
    __tablename__ = "admins"

    AdminId = Column(Integer, primary_key=True, index=True)
    Username = Column(String(100))
    Password = Column(String(255))  # hashed password (recommended)
