import cv2
import random
import os

# Global variables for bounding box selection
ref_point = []
cropping = False
image_copy = None

def extract_random_frame(video_path):
    """
    Opens the video file and extracts a random frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_index = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to read frame from video.")
    
    return frame, random_index

def click_and_crop(event, x, y, flags, param):
    """
    Mouse callback function to record the bounding box.
    - Left button press starts the bounding box.
    - Mouse movement draws the current rectangle.
    - Left button release finishes the bounding box.
    """
    global ref_point, cropping, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        image_copy = param.copy()
        cv2.rectangle(image_copy, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Frame", image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(param, ref_point[0], ref_point[1], (0, 255, 0), 2)
        image_copy = param.copy()
        cv2.imshow("Frame", image_copy)

def annotate_frame(frame, frame_index):
    """
    Displays the frame and lets the user annotate a bounding box.
    You can:
      - Draw a bounding box and press 'c' to confirm,
      - Press 'n' if no bounding box is needed,
      - Press 'r' to reset the current selection.
    Returns the YOLO formatted annotation string (or an empty string if no box is drawn).
    """
    global image_copy, ref_point
    image_copy = frame.copy()
    ref_point = []

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_and_crop, frame)

    print("\nAnnotation Instructions:")
    print(" - Click and drag to draw a bounding box around the pinball.")
    print(" - Press 'r' to reset the selection.")
    print(" - Press 'c' to confirm the annotation (if a box was drawn).")
    print(" - Press 'n' if you wish to save the image without any bounding box (empty label).\n")
    
    # Loop until a valid key is pressed.
    while True:
        cv2.imshow("Frame", image_copy)
        key = cv2.waitKey(1) & 0xFF

        # Reset selection on 'r'
        if key == ord("r"):
            image_copy = frame.copy()
            ref_point = []
            print("Selection reset. Draw again.")

        # If the user decides to save without annotation
        elif key == ord("n"):
            # User skipped annotating by pressing 'n'
            print("No bounding box selected; saving an empty label file.")
            cv2.destroyWindow("Frame")
            return ""
            
        # Confirm selection only if a box was drawn.
        elif key == ord("c"):
            # Only confirm if a bounding box has been drawn
            if len(ref_point) == 2:
                cv2.destroyWindow("Frame")
                break
            else:
                print("No bounding box drawn. Press 'n' to save with no annotation or draw a box and press 'c'.")

    # If a box was drawn, calculate YOLO normalized coordinates.
    if len(ref_point) == 2:
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        # Sort coordinates to get top-left and bottom-right
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        box_width = x_max - x_min
        box_height = y_max - y_min

        image_h, image_w = frame.shape[:2]

        # Calculate YOLO coordinates: normalized center_x, center_y, width, height.
        center_x = ((x_min + x_max) / 2.0) / image_w
        center_y = ((y_min + y_max) / 2.0) / image_h
        norm_width = box_width / image_w
        norm_height = box_height / image_h

        # Format: class_index center_x center_y width height
        # Using "0" as the class index for the pinball.
        yolo_annotation = f"0 {center_x} {center_y} {norm_width} {norm_height}\n"
        return yolo_annotation
    else:
        # No valid box was drawn; should not reach here due to earlier check.
        return ""

def main():
    video_path = 'data/pinball-video.mp4'
    
    # Directories to save images and labels
    images_dir = "dataset/images"
    labels_dir = "dataset/labels"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print("Starting the pinball annotation tool.")
    print("After each annotation, press 'p' to stop labeling or any other key to continue.\n")

    while True:
        # Extract a random frame from the video.
        try:
            frame, frame_index = extract_random_frame(video_path)
        except Exception as e:
            print("Error extracting frame:", e)
            break

        # Annotate the frame (or decide not to annotate a bounding box)
        annotation = annotate_frame(frame, frame_index)
        
        # Define file paths for image and label saving.
        image_filename = os.path.join(images_dir, f'frame_{frame_index:04d}.jpg')
        label_filename = os.path.join(labels_dir, f'frame_{frame_index:04d}.txt')
        
        # Save image
        cv2.imwrite(image_filename, frame)
        print(f"Saved image to '{image_filename}'.")
        
        # Save annotation file even if it's empty.
        with open(label_filename, 'w') as f:
            f.write(annotation)
        print(f"Saved annotation to '{label_filename}'.")
        
        # Ask user if they want to continue
        print("Press 'p' to stop labeling, or any other key to label another frame.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord("p"):
            print("Annotation process terminated by user.")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
