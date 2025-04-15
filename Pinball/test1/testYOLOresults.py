

import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('pinball_training/yolov8_pinball/weights/best.pt')  # Make sure to use the best.pt from your training

# Initialize OpenCV video capture (use 0 for webcam or provide a video file path)
cap = cv2.VideoCapture('data/pinball-video.mp4')  # Replace with video path or webcam index (e.g., 0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break  # Exit if the video has ended

    # Run detection on the frame
    results = model(frame)

    # Plot the predictions on the frame
    annotated_frame = results[0].plot()  # Use plot() to annotate the frame

    # Display the frame with the detections
    cv2.imshow('Pinball Detection', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()




