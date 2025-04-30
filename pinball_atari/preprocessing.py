# pinball_ai/preprocessing.py

import cv2
import numpy as np
from config import IMAGE_SIZE 

def preprocess_frame(frame):
    """
    Preprocesses a single frame: Converts to grayscale, resizes, normalizes.
    Returns a single channel image ready for stacking.
    """
    # Ensure frame is in BGR format if it's not already (Gym normally provides RGB)
    if frame.shape[-1] == 3:
         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Use COLOR_RGB2GRAY if input is RGB
    else: # Assuming it's already grayscale
         gray = frame

    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking
    normalized = resized / 255.0
    # Return shape (height, width) - channel dim added during stacking
    return normalized.astype(np.float32)