import cv2
import numpy as np
import argparse

class PinballCalibrator:
    def __init__(self):
        self.frame = None
        self.roi_selector = None
        self.hsv_values = {
            'hue_min': 0, 'hue_max': 180,
            'sat_min': 0, 'sat_max': 255,
            'val_min': 0, 'val_max': 255
        }

    def create_trackbars(self):
        """Create trackbars for HSV adjustment"""
        cv2.namedWindow('HSV Trackbars')
        
        # Create trackbars for HSV adjustment
        cv2.createTrackbar('H Min', 'HSV Trackbars', self.hsv_values['hue_min'], 180, 
                          lambda x: self.update_hsv_value('hue_min', x))
        cv2.createTrackbar('H Max', 'HSV Trackbars', self.hsv_values['hue_max'], 180, 
                          lambda x: self.update_hsv_value('hue_max', x))
        cv2.createTrackbar('S Min', 'HSV Trackbars', self.hsv_values['sat_min'], 255, 
                          lambda x: self.update_hsv_value('sat_min', x))
        cv2.createTrackbar('S Max', 'HSV Trackbars', self.hsv_values['sat_max'], 255, 
                          lambda x: self.update_hsv_value('sat_max', x))
        cv2.createTrackbar('V Min', 'HSV Trackbars', self.hsv_values['val_min'], 255, 
                          lambda x: self.update_hsv_value('val_min', x))
        cv2.createTrackbar('V Max', 'HSV Trackbars', self.hsv_values['val_max'], 255, 
                          lambda x: self.update_hsv_value('val_max', x))
    
    def update_hsv_value(self, key, value):
        """Update HSV value when trackbar changes"""
        self.hsv_values[key] = value
    
    def select_roi(self, frame):
        """Let user select a region containing the pinball"""
        print("Select the region containing the pinball, then press ENTER")
        roi = cv2.selectROI("Select Pinball", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Pinball")
        
        if roi[2] > 0 and roi[3] > 0:  # If ROI has non-zero width and height
            x, y, w, h = roi
            ball_roi = frame[y:y+h, x:x+w]
            
            # Convert to HSV and calculate average color
            hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
            avg_color = cv2.mean(hsv_roi)[:3]
            
            # Set initial HSV range based on the average color
            h_margin = 15
            s_margin = 50
            v_margin = 50
            
            self.hsv_values['hue_min'] = max(0, int(avg_color[0] - h_margin))
            self.hsv_values['hue_max'] = min(180, int(avg_color[0] + h_margin))
            self.hsv_values['sat_min'] = max(0, int(avg_color[1] - s_margin))
            self.hsv_values['sat_max'] = min(255, int(avg_color[1] + s_margin))
            self.hsv_values['val_min'] = max(0, int(avg_color[2] - v_margin))
            self.hsv_values['val_max'] = min(255, int(avg_color[2] + v_margin))
            
            return roi
        return None
    
    def calibrate_from_video(self, video_path=None, camera_id=0):
        """
        Calibrate HSV values using video input
        
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
        else:
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open video source")
            return None
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from video source")
            cap.release()
            return None
        
        # Let user select ROI with the pinball
        roi = self.select_roi(frame)
        if roi is None:
            print("No valid ROI selected")
            cap.release()
            return None
        
        # Create trackbars with initial values
        self.create_trackbars()
        
        paused = False
        roi_x, roi_y, roi_w, roi_h = roi
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
            
            # Draw the ROI on the frame
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            # Convert to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask with current HSV values
            lower_hsv = np.array([
                self.hsv_values['hue_min'], 
                self.hsv_values['sat_min'], 
                self.hsv_values['val_min']
            ])
            upper_hsv = np.array([
                self.hsv_values['hue_max'], 
                self.hsv_values['sat_max'], 
                self.hsv_values['val_max']
            ])
            
            mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            # Display results
            cv2.imshow('Original', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Toggle pause
                paused = not paused
            elif key == ord('s'):  # Save values
                self.save_hsv_values()
                
        cap.release()
        cv2.destroyAllWindows()
        
        return {
            'lower_hsv': lower_hsv.tolist(), 
            'upper_hsv': upper_hsv.tolist()
        }
    
    def save_hsv_values(self):
        """Save the current HSV values to a file"""
        lower_hsv = [
            self.hsv_values['hue_min'], 
            self.hsv_values['sat_min'], 
            self.hsv_values['val_min']
        ]
        upper_hsv = [
            self.hsv_values['hue_max'], 
            self.hsv_values['sat_max'], 
            self.hsv_values['val_max']
        ]
        
        # Print values for manual configuration
        print("\nHSV Calibration Results:")
        print(f"Lower HSV: {lower_hsv}")
        print(f"Upper HSV: {upper_hsv}")
        print("\nPython code for configuration:")
        print(f"PINBALL_LOWER_COLOR = {lower_hsv}")
        print(f"PINBALL_UPPER_COLOR = {upper_hsv}")
        
        # Save to file
        with open('hsv_calibration.txt', 'w') as f:
            f.write(f"Lower HSV: {lower_hsv}\n")
            f.write(f"Upper HSV: {upper_hsv}\n")
        
        print("Values saved to hsv_calibration.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate HSV values for pinball detection')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use if no video file is provided')
    
    args = parser.parse_args()
    
    calibrator = PinballCalibrator()
    hsv_values = calibrator.calibrate_from_video(args.video, args.camera)
    
    if hsv_values:
        print("Calibration completed successfully")
    else:
        print("Calibration failed or was cancelled")