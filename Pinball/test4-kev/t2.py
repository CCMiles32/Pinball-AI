import cv2
import numpy as np

# Load the video
input_video = "../data/kevins-pinball.mp4"
cap = cv2.VideoCapture(input_video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Threshold for bright spots (adjust threshold value as needed)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up small noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest bright spot
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            # Draw circle and center
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow("Ball Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()