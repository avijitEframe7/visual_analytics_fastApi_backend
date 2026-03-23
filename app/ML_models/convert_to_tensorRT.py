from ultralytics import YOLO

# Load YOLO model
model = YOLO("New_Model_24.2.26..pt")


model.export(
    format="engine",   # TensorRT
    half=True,         # FP16 (faster)
    device=0           # GPU id
)