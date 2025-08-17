from ultralytics import YOLO

# Load pretrained YOLOv8s
model = YOLO("yolov8s.pt")

# Train on CPU
model.train(
    data="./dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,  # Reduced batch size for CPU training
    device="cpu"  # Use CPU instead of GPU
)