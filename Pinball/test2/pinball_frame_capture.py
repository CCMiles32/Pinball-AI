import cv2
import numpy as np
import os
import time

def capture_pinball_frames(video_path=None, camera_id=0, output_dir="pinball_frames"):
    """
    Capture frames where the pinball is clearly visible
    
    Parameters:
    -----------
    video_path : str, optional
        Path to video file, or None to use camera
    camera_id : int
        Camera ID to use if video_path is None
    output_dir : str
        Directory to save captured frames
    """
    # Open video source
    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"Opening video file: {video_path}")
    else:
        cap = cv2.VideoCapture(camera_id)
        print(f"Opening camera: {camera_id}")
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Variables
    frame_count = 0
    saved_count = 0
    paused = False
    
    print("\nInstructions:")
    print("- Press SPACE to pause/unpause the video")
    print("- When paused, press 'c' to capture the current frame")
    print("- Press 'q' to quit")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break
            frame_count += 1
        
        # Display status on frame
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Saved: {saved_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Frame Capture Tool', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar
            paused = not paused
            print(f"Video {'paused' if paused else 'resumed'}")
        elif key == ord('c') and paused:
            # Save the current frame
            timestamp = int(time.time())
            filename = f"{output_dir}/pinball_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved frame to {filename}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {saved_count} frames")
    
    return output_dir

def manual_color_calibration(frame_path):
    """
    Manually calibrate HSV values for a pinball using a captured frame
    
    Parameters:
    -----------
    frame_path : str
        Path to a captured frame containing the pinball
    
    Returns:
    --------
    dict
        Dictionary containing the lower and upper HSV bounds
    """
    # Load image
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Cannot load image from {frame_path}")
        return None
    
    # Initial HSV values (default to a broad range for metallic/silver pinballs)
    hsv_values = {
        'hue_min': 0, 'hue_max': 180,
        'sat_min': 0, 'sat_max': 30,
        'val_min': 200, 'val_max': 255
    }
    
    # Create window and trackbars
    cv2.namedWindow('Manual Calibration')
    cv2.resizeWindow('Manual Calibration', 800, 600)
    
    # Create trackbars for HSV adjustment
    cv2.createTrackbar('H Min', 'Manual Calibration', hsv_values['hue_min'], 180, 
                      lambda x: update_hsv_value(hsv_values, 'hue_min', x))
    cv2.createTrackbar('H Max', 'Manual Calibration', hsv_values['hue_max'], 180, 
                      lambda x: update_hsv_value(hsv_values, 'hue_max', x))
    cv2.createTrackbar('S Min', 'Manual Calibration', hsv_values['sat_min'], 255, 
                      lambda x: update_hsv_value(hsv_values, 'sat_min', x))
    cv2.createTrackbar('S Max', 'Manual Calibration', hsv_values['sat_max'], 255, 
                      lambda x: update_hsv_value(hsv_values, 'sat_max', x))
    cv2.createTrackbar('V Min', 'Manual Calibration', hsv_values['val_min'], 255, 
                      lambda x: update_hsv_value(hsv_values, 'val_min', x))
    cv2.createTrackbar('V Max', 'Manual Calibration', hsv_values['val_max'], 255, 
                      lambda x: update_hsv_value(hsv_values, 'val_max', x))
    
    print("\nInstructions:")
    print("- Adjust the sliders until only the pinball is visible in the 'Mask' window")
    print("- Press 's' to save the values")
    print("- Press 'q' to quit without saving")
    
    while True:
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask with current HSV values
        lower_hsv = np.array([
            hsv_values['hue_min'], 
            hsv_values['sat_min'], 
            hsv_values['val_min']
        ])
        upper_hsv = np.array([
            hsv_values['hue_max'], 
            hsv_values['sat_max'], 
            hsv_values['val_max']
        ])
        
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy for drawing contours
        contour_img = result.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Filter contours by area (for visual feedback)
        filtered_contours = []
        min_area = 100
        max_area = 5000
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                filtered_contours.append(contour)
                
                # Draw a bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Draw the center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(contour_img, (cx, cy), 5, (255, 0, 0), -1)
        
        # Display the number of detected contours in the appropriate size range
        cv2.putText(contour_img, f"Potential pinballs: {len(filtered_contours)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display HSV values
        hsv_text = f"HSV Range: H({hsv_values['hue_min']}-{hsv_values['hue_max']}), " \
                   f"S({hsv_values['sat_min']}-{hsv_values['sat_max']}), " \
                   f"V({hsv_values['val_min']}-{hsv_values['val_max']})"
        cv2.putText(contour_img, hsv_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display images
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Manual Calibration', contour_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save values
            result = {
                'lower_hsv': lower_hsv.tolist(),
                'upper_hsv': upper_hsv.tolist()
            }
            
            # Print values for manual configuration
            print("\nHSV Calibration Results:")
            print(f"Lower HSV: {lower_hsv}")
            print(f"Upper HSV: {upper_hsv}")
            print("\nPython code for configuration:")
            print(f"PINBALL_LOWER_COLOR = {lower_hsv.tolist()}")
            print(f"PINBALL_UPPER_COLOR = {upper_hsv.tolist()}")
            
            # Save to file
            with open('hsv_calibration.txt', 'w') as f:
                f.write(f"Lower HSV: {lower_hsv.tolist()}\n")
                f.write(f"Upper HSV: {upper_hsv.tolist()}\n")
            
            print("Values saved to hsv_calibration.txt")
            cv2.destroyAllWindows()
            return result
    
    cv2.destroyAllWindows()
    return None

def update_hsv_value(hsv_values, key, value):
    """Update HSV value when trackbar changes"""
    hsv_values[key] = value


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pinball Frame Capture and Calibration Tool')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use if no video file is provided')
    parser.add_argument('--mode', type=str, choices=['capture', 'calibrate', 'both'], default='both',
                        help='Tool mode: capture frames, calibrate from existing frame, or both')
    parser.add_argument('--frame_path', type=str, help='Path to frame for calibration (required if mode=calibrate)')
    
    args = parser.parse_args()
    
    if args.mode == 'capture' or args.mode == 'both':
        output_dir = capture_pinball_frames(args.video, args.camera)
        
        if args.mode == 'both' and output_dir:
            # Find the first captured frame
            frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
            if frames:
                frame_path = os.path.join(output_dir, frames[0])
                print(f"Calibrating using frame: {frame_path}")
                manual_color_calibration(frame_path)
    
    elif args.mode == 'calibrate':
        if not args.frame_path:
            print("Error: --frame_path is required when mode=calibrate")
        else:
            manual_color_calibration(args.frame_path)