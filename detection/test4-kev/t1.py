import cv2
import numpy as np

# Load the video
input_video = "../data/kevins-pinball.mp4"
cap = cv2.VideoCapture(input_video)

# Read the first frame to initialize
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Preprocess the first frame
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # --- Motion Detection ---
    frame_delta = cv2.absdiff(prev_gray, blurred)
    _, motion_mask = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    # --- Bright Spot Detection ---
    bright_blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, bright_mask = cv2.threshold(bright_blur, 200, 255, cv2.THRESH_BINARY)

    # --- Combine both masks ---
    combined_mask = cv2.bitwise_and(motion_mask, bright_mask)

    # Clean up mask
    combined_mask = cv2.erode(combined_mask, None, iterations=1)
    combined_mask = cv2.dilate(combined_mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            # Draw circle and center
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show frame
    cv2.imshow("Ball Tracking with Motion", frame)

    # Update previous frame
    prev_gray = blurred.copy()

    # Exit on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()