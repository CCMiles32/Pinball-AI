import cv2

input_path = "Pinball/data/pinball-video-shortened.mp4"
output_path = "Pinball/data/pinball-video-shortened-30fps.mp4"

cap = cv2.VideoCapture(input_path)

# Get original video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Target FPS (30)
target_fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only write every other frame (60fps → 30fps)
    if frame_count % 2 == 0:
        out.write(frame)

    frame_count += 1

cap.release()
out.release()
print("Done! Saved:", output_path)
