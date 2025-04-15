from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('pinball_training/yolov8_pinball/weights/best.pt')

# Run inference on an image
results = model('some_image.jpg')  # can also be a frame or array

# Show the results
results[0].show()  # or .save() to write to disk
