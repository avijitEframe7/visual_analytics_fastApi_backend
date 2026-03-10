from ultralytics import YOLO

model = YOLO(r"app\ML_models\latest_16_1_2026.pt")

print(model.names)