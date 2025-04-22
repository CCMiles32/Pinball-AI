import cv2
import numpy as np

# Load the video
#input_video = "../data/kevins-pinball.mp4"
input_video = "../data/pinball-video-shortened-30fps.mp4"

cap = cv2.VideoCapture(input_video)

# Define the color range for the ball in HSV
# Example for red ball — adjust according to your ball's color
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the ball's color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours were found
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show the frame
    cv2.imshow('Ball Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()