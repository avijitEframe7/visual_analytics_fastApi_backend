from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.database import get_db   # IMPORTANT

router = APIRouter()

@router.get("/get_users")
# def get_users(db = Depends(get_db)):
#     cursor = db.cursor(dictionary=True)
#     cursor.execute("SELECT * FROM admins")
#     result = cursor.fetchall()
#     cursor.close()
#     return result
def get_users(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM admins"))
    rows = result.mappings().all()  # list of dicts
    return rows

# @router.get()
