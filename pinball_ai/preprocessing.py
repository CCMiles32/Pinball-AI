import cv2
import numpy as np
from config import IMAGE_SIZE

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)