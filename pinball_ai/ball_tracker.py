import cv2
import numpy as np

def detect_ball(frame):
    """Detect the pinball in the frame and return (x, y) coordinates."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold the bright areas (assuming the ball is bright)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_position = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 300:  # Filter small contours
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < 20:  # Ball should be small
                ball_position = (int(x), int(y))
                break

    return ball_position  # (x, y) or None if not found

def draw_ball(frame, ball_position):
    """Draw a circle where the ball is detected."""
    if ball_position:
        cv2.circle(frame, ball_position, 10, (0, 255, 0), 2)
    return frame