# Run from project root (visual_analytics_fastapi_backend), not from app/:
#   uvicorn main:app --host 127.0.0.1 --port 8000
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import users, health, auth, main_dashboard, camera_dashboard, model_management, camera_management, notification_management

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],         # Authorization, Content-Type, etc.
)

app.include_router(users.router)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(main_dashboard.router)
app.include_router(camera_dashboard.router)
app.include_router(model_management.router)
app.include_router(camera_management.router)
app.include_router(notification_management.router)
