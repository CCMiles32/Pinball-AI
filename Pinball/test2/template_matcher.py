import cv2
import numpy as np
import time
import os
import pickle

class PinballTemplateDetector:
    """
    Class for pinball detection using only template matching
    """
    
    def __init__(self):
        # Default parameters for template matching
        self.template_match_threshold = 0.7
    
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
        float
            Confidence score of the match
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim > 2 else template
        
        # Apply template matching
        result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # Get the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Use threshold to determine if the match is good enough
        if max_val > self.template_match_threshold:
            # Calculate center of template
            h, w = gray_template.shape
            x = max_loc[0] + w // 2
            y = max_loc[1] + h // 2
            return (x, y), max_val
        
        return None, 0.0
    
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
    
    def adjust_template_threshold(self, new_threshold):
        """
        Adjust the template matching threshold
        
        Parameters:
        -----------
        new_threshold : float
            New threshold value between 0 and 1
        """
        if 0 <= new_threshold <= 1:
            self.template_match_threshold = new_threshold
            return True
        return False
    
    def save_template(self, template, filename="pinball_template.pkl"):
        """
        Save the template to a file
        
        Parameters:
        -----------
        template : numpy.ndarray
            The template image to save
        filename : str
            Path to save the template to
        
        Returns:
        --------
        bool
            True if saved successfully, False otherwise
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(template, f)
            print(f"Template saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving template: {e}")
            return False
    
    def load_template(self, filename="pinball_template.pkl"):
        """
        Load a template from a file
        
        Parameters:
        -----------
        filename : str
            Path to load the template from
        
        Returns:
        --------
        numpy.ndarray or None
            The loaded template, or None if loading failed
        """
        if not os.path.exists(filename):
            print(f"Template file {filename} not found")
            return None
        
        try:
            with open(filename, 'rb') as f:
                template = pickle.load(f)
            print(f"Template loaded from {filename}")
            return template
        except Exception as e:
            print(f"Error loading template: {e}")
            return None


def demonstrate_template_matching(video_path=None, camera_id=0, template_path="pinball_template.pkl"):
    """
    Demonstrate template matching for pinball detection
    
    Parameters:
    -----------
    video_path : str, optional
        Path to video file, or None to use camera
    camera_id : int
        Camera ID to use if video_path is None
    template_path : str
        Path to load/save template from/to
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
    detector = PinballTemplateDetector()
    
    # Variables
    template = None
    template_roi = None
    
    # Try to load saved template
    template = detector.load_template(template_path)
    
    print("\nInstructions:")
    print("- Press 'p' to pause/unpause the video")
    print("- When paused, press 't' to select a new template for template matching")
    print("- Press 's' to save the current template")
    print("- Press 'l' to load a saved template")
    print("- Press '+' to increase template matching threshold")
    print("- Press '-' to decrease template matching threshold")
    print("- Press 'q' to quit")
    
    paused = False
    frame_count = 0
    
    # Get initial frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from video source")
        cap.release()
        return
    
    # If no template was loaded, prompt for initial selection
    if template is None:
        print("No saved template found. Select initial ROI for template matching")
        template_roi = cv2.selectROI("Select Pinball Template", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Pinball Template")
        
        if template_roi[2] > 0 and template_roi[3] > 0:  # If ROI has non-zero width and height
            template = detector.create_pinball_template(frame, template_roi)
            print(f"Template created: {template.shape}")
            
            # Offer to save the template
            print("Would you like to save this template for future use? Press 's' during execution.")
        else:
            print("Warning: No template selected. Press 't' when paused to select a template.")
    else:
        # If template was loaded, compute its size for displaying the ROI
        h, w = template.shape[:2] if len(template.shape) > 2 else template.shape
        template_roi = (10, 10, w, h)  # Default position in top-left corner
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break
            frame_count += 1
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect pinball using template matching if we have a template
        if template is not None:
            position, confidence = detector.detect_pinball_by_template(frame, template)
            
            # Draw detection results
            if position is not None:
                # Draw a circle at the position
                cv2.circle(display_frame, position, 15, (0, 255, 0), 2)
                
                # Add text for confidence
                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No pinball detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No template selected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw status information
        cv2.putText(display_frame, f"Status: {'PAUSED' if paused else 'PLAYING'}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Threshold: {detector.template_match_threshold:.2f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Template: {'Loaded' if template is not None else 'None'}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show template in corner
        if template is not None:
            # Calculate scale to show a reasonably sized preview
            max_preview_size = 100
            scale = min(max_preview_size / template.shape[0], max_preview_size / template.shape[1])
            preview_h = int(template.shape[0] * scale)
            preview_w = int(template.shape[1] * scale)
            preview = cv2.resize(template, (preview_w, preview_h))
            
            # Place preview in top-right corner
            h, w = preview.shape[:2]
            display_frame[10:10+h, display_frame.shape[1]-10-w:display_frame.shape[1]-10] = preview
            
            # Add border
            cv2.rectangle(display_frame, 
                          (display_frame.shape[1]-10-w-1, 9), 
                          (display_frame.shape[1]-9, 11+h), 
                          (255, 0, 0), 1)
        
        # Show the frame
        cv2.imshow('Template Matching Pinball Detection', display_frame)
        
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
                print("Press 's' to save this template for future use.")
            else:
                template = None
                template_roi = None
                print("Template selection cancelled")
        elif key == ord('s'):  # Save template
            if template is not None:
                detector.save_template(template, template_path)
            else:
                print("No template to save")
        elif key == ord('l'):  # Load template
            loaded_template = detector.load_template(template_path)
            if loaded_template is not None:
                template = loaded_template
                # Update template_roi for display purposes
                h, w = template.shape[:2] if len(template.shape) > 2 else template.shape
                template_roi = (10, 10, w, h)
        elif key == ord('+') or key == ord('='):  # Increase threshold
            new_threshold = min(detector.template_match_threshold + 0.05, 1.0)
            detector.adjust_template_threshold(new_threshold)
            print(f"Threshold increased to: {detector.template_match_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):  # Decrease threshold
            new_threshold = max(detector.template_match_threshold - 0.05, 0.0)
            detector.adjust_template_threshold(new_threshold)
            print(f"Threshold decreased to: {detector.template_match_threshold:.2f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Template Matching Pinball Detection Demo')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use if no video file is provided')
    parser.add_argument('--template', type=str, default="pinball_template.pkl", help='Path to template file to load/save')
    
    args = parser.parse_args()
    
    demonstrate_template_matching(args.video, args.camera, args.template)