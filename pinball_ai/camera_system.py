import cv2

def get_camera_stream(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception("Camera not accessible")
    return cap

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture frame")
    return frame