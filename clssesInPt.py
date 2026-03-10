from ultralytics import YOLO

model = YOLO(r"visual_analytics_fastapi_backend\app\ML_models\latest_16_1_2026.pt")

print(model.names)