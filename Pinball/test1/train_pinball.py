from ultralytics import YOLO
from pathlib import Path

yaml_path = Path(__file__).parent / 'pinball.yaml'  # resolves to full path

model = YOLO('yolov8s.pt')
model.train(
    data=str(yaml_path),  # use full path
    epochs=15,
    imgsz=960,
    batch=16,
    patience=8,
    project='pinball_training',
    name='yolov8_pinball',
    exist_ok=True
)
