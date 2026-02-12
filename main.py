# Run from project root (visual_analytics_fastapi_backend), not from app/:
#   uvicorn main:app --host 127.0.0.1 --port 8000

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import (
    users, 
    health, 
    auth, 
    main_dashboard, 
    camera_dashboard, 
    model_management, 
    camera_management, 
    notification_management
)

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Visual Analytics API",
    description="FastAPI backend for visual analytics with camera management",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],         # Authorization, Content-Type, etc.
)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Visual Analytics API...")
    logger.info("Application startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Visual Analytics API...")

# Include routers
app.include_router(users.router)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(main_dashboard.router)
app.include_router(camera_dashboard.router)
app.include_router(model_management.router)
app.include_router(camera_management.router)
app.include_router(notification_management.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Visual Analytics API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        log_level="info"
    )