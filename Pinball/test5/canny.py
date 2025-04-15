import cv2
import numpy as np

input_video = "../data/kevins-pinball.mp4"

def canny_side_by_side(vid_path, wait_ms=100):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, 100, 200)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        side_by_side = np.hstack((frame, edges_color))

        cv2.imshow("Original | Canny Edge Detection", side_by_side)

        # Slow down playback
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run with 3x slower playback (100 ms delay)
canny_side_by_side(input_video, wait_ms=100)
