from pydantic import BaseModel

class LoginRequest(BaseModel):
    adminId: int
    username: str
    password: str
