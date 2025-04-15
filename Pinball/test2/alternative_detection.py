import cv2
import numpy as np
import time
import os

class PinballMultiMethod:
    """
    Class providing multiple methods for pinball detection when the ball is not always visible
    """
    
    def __init__(self):
        # Default HSV ranges for common pinball colors
        self.color_presets = {
            'SILVER': [np.array([0, 0, 150]), np.array([180, 30, 255])],  # Metallic/silver ball
            'GREEN': [np.array([40, 40, 40]), np.array([80, 255, 255])],  # Green ball
            'ORANGE': [np.array([10, 100, 100]), np.array([25, 255, 255])],  # Orange ball
            'BLUE': [np.array([100, 50, 50]), np.array([130, 255, 255])],  # Blue ball
            'CUSTOM': [np.array([7, 0, 72]), np.array([7, 219, 164])]
        }
        
        # Default parameters for motion detection
        self.motion_detection_params = {
            'history': 500,
            'var_threshold': 16,
            'detect_shadows': True
        }
        
        # Default parameters for physical detection
        self.physical_min_area = 100
        self.physical_max_area = 5000
        self.physical_min_circularity = 0.7
    
    def detect_pinball_by_color(self, frame, color_preset='SILVER', custom_range=None):
        """
        Detect pinball using color thresholding
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The frame to process
        color_preset : str
            Predefined color preset to use
        custom_range : list
            Custom HSV range as [lower_bound, upper_bound] if provided
            
        Returns:
        --------
        tuple
            (x, y) coordinates of the pinball, or None if not found
        """
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use custom range if provided, otherwise use preset
        if custom_range is not None:
            lower_bound, upper_bound = custom_range
        else:
            if color_preset not in self.color_presets:
                print(f"Warning: '{color_preset}' not found in presets. Using SILVER instead.")
                color_preset = 'SILVER'
            
            lower_bound, upper_bound = self.color_presets[color_preset]
        
        # Create mask and filter
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and circularity
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.physical_min_area < area < self.physical_max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > self.physical_min_circularity:
                    # Get center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy)
        
        return None
    
    def setup_motion_detector(self):
        """
        Set up the background subtraction motion detector
        """
        return cv2.createBackgroundSubtractorMOG2(
            history=self.motion_detection_params['history'],
            varThreshold=self.motion_detection_params['var_threshold'],
            detectShadows=self.motion_detection_params['detect_shadows']
        )
    
    def detect_pinball_by_motion(self, frame, bg_subtractor, prev_frame=None):
        """
        Detect pinball using motion detection
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The current frame
        bg_subtractor : cv2.BackgroundSubtractor
            Background subtraction object
        prev_frame : numpy.ndarray, optional
            Previous frame for frame differencing
            
        Returns:
        --------
        tuple
            (x, y) coordinates of the pinball, or None if not found
        """
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Apply thresholding
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = list(contours)  # Convert tuple to list so we can extend it
        
        # If we also have a previous frame, do frame differencing
        if prev_frame is not None:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute the absolute difference
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Apply thresholding
            _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find contours in the difference image
            diff_contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Add contours from frame differencing to our list
            all_contours.extend(diff_contours)
        
        # Filter contours by area and shape
        for contour in all_contours:
            area = cv2.contourArea(contour)
            
            if self.physical_min_area < area < self.physical_max_area:
                # Check if it's reasonably circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > self.physical_min_circularity:
                    # Get center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy)
        
        return None
    
    def detect_pinball_by_template(self, frame, template):
        """
        Detect pinball using template matching
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The frame to process
        template : numpy.ndarray
            Template image of the pinball
            
        Returns:
        --------
        tuple
            (x, y) coordinates of the pinball, or None if not found
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim > 2 else template
        
        # Apply template matching
        result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # Get the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Use threshold to determine if the match is good enough
        if max_val > 0.8:  # Adjust threshold as needed
            # Calculate center of template
            h, w = gray_template.shape
            x = max_loc[0] + w // 2
            y = max_loc[1] + h // 2
            return (x, y)
        
        return None
    
    def create_pinball_template(self, frame, roi):
        """
        Create a template image from a region of interest containing the pinball
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The frame containing the pinball
        roi : tuple
            Region of interest (x, y, width, height)
            
        Returns:
        --------
        numpy.ndarray
            Template image of the pinball
        """
        x, y, w, h = roi
        return frame[y:y+h, x:x+w]
    
    def detect_pinball_multi_method(self, frame, prev_frame=None, bg_subtractor=None, 
                                  color_preset='SILVER', custom_range=None, template=None):
        """
        Detect pinball using multiple methods
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The frame to process
        prev_frame : numpy.ndarray, optional
            Previous frame for motion detection
        bg_subtractor : cv2.BackgroundSubtractor, optional
            Background subtraction object
        color_preset : str
            Predefined color preset to use
        custom_range : list
            Custom HSV range as [lower_bound, upper_bound] if provided
        template : numpy.ndarray, optional
            Template image of the pinball
            
        Returns:
        --------
        tuple
            (x, y) coordinates of the pinball, or None if not found
        dict
            Method used for detection and confidence
        """
        results = {}
        
        # Try color detection
        color_result = self.detect_pinball_by_color(frame, color_preset, custom_range)
        if color_result is not None:
            results['color'] = color_result
        
        # Try motion detection if applicable
        if bg_subtractor is not None:
            motion_result = self.detect_pinball_by_motion(frame, bg_subtractor, prev_frame)
            if motion_result is not None:
                results['motion'] = motion_result
        
        # Try template matching if applicable
        if template is not None:
            template_result = self.detect_pinball_by_template(frame, template)
            if template_result is not None:
                results['template'] = template_result
        
        # If no results, return None
        if not results:
            return None, {'method': None, 'confidence': 0}
        
        # If only one method succeeded, return that result
        if len(results) == 1:
            method = next(iter(results))
            return results[method], {'method': method, 'confidence': 1.0}
        
        # If multiple methods succeeded, return the average position
        x_sum = sum(pos[0] for pos in results.values())
        y_sum = sum(pos[1] for pos in results.values())
        avg_pos = (int(x_sum / len(results)), int(y_sum / len(results)))
        
        return avg_pos, {'method': 'combined', 'confidence': len(results) / 3}


def demonstrate_multi_method(video_path=None, camera_id=0):
    """
    Demonstrate the multi-method detection
    
    Parameters:
    -----------
    video_path : str, optional
        Path to video file, or None to use camera
    camera_id : int
        Camera ID to use if video_path is None
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
    
    # Create detector
    detector = PinballMultiMethod()
    
    # Set up background subtractor
    bg_subtractor = detector.setup_motion_detector()
    
    # Variables
    template = None
    template_roi = None
    prev_frame = None
    color_preset = 'SILVER'
    custom_range = None
    
    print("\nInstructions:")
    print("- Press 'p' to pause/unpause the video")
    print("- When paused, press 't' to select a template for template matching")
    print("- Press 'c' to cycle through color presets (SILVER, GREEN, ORANGE, BLUE)")
    print("- Press 'q' to quit")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break
            frame_count += 1
        
        # Detect pinball using multiple methods
        position, info = detector.detect_pinball_multi_method(
            frame, prev_frame, bg_subtractor, 
            color_preset, custom_range, template
        )
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Draw detection results
        if position is not None:
            # Draw a circle at the position
            cv2.circle(display_frame, position, 15, (0, 255, 0), 2)
            
            # Add text for method and confidence
            cv2.putText(display_frame, f"Method: {info['method']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {info['confidence']:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No pinball detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw status information
        cv2.putText(display_frame, f"Status: {'PAUSED' if paused else 'PLAYING'}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Color Preset: {color_preset}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Template: {'Yes' if template is not None else 'No'}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # If we have a template ROI, draw it
        if template_roi is not None:
            x, y, w, h = template_roi
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Show the frame
        cv2.imshow('Multi-Method Pinball Detection', display_frame)
        
        # Store previous frame for motion detection
        if not paused:
            prev_frame = frame.copy()
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Pause/unpause
            paused = not paused
            print(f"Video {'paused' if paused else 'resumed'}")
        elif key == ord('t') and paused:  # Select template
            print("Select ROI for template matching")
            template_roi = cv2.selectROI("Select Pinball Template", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Pinball Template")
            
            if template_roi[2] > 0 and template_roi[3] > 0:  # If ROI has non-zero width and height
                template = detector.create_pinball_template(frame, template_roi)
                print(f"Template created: {template.shape}")
            else:
                template = None
                template_roi = None
                print("Template selection cancelled")
        elif key == ord('c'):  # Cycle through color presets
            presets = list(detector.color_presets.keys())
            current_index = presets.index(color_preset)
            color_preset = presets[(current_index + 1) % len(presets)]
            print(f"Switched to color preset: {color_preset}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Method Pinball Detection Demo')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use if no video file is provided')
    
    args = parser.parse_args()
    
    demonstrate_multi_method(args.video, args.camera)