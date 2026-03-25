from pydantic import BaseModel, field_validator


class LoginRequest(BaseModel):
    adminId: int
    username: str
    password: str

    @field_validator("adminId", mode="before")
    @classmethod
    def coerce_admin_id(cls, v):
        """Accept JSON numbers or string digits (frontend often sends \"1\")."""
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
        return v
