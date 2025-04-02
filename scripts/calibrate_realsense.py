#!/usr/bin/env python3

import numpy as np
import cv2
import pyrealsense2 as rs
import time
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional
import sys
import os
import urx

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraCalibrator:
    def __init__(self, camera_id: str, robot_ip: str = "192.168.1.2"):
        """Initialize the camera calibrator.
        
        Args:
            camera_id: The serial number of the RealSense camera
            robot_ip: IP address of the UR5 robot
        """
        print("Initializing camera calibrator...")

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(camera_id)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        # Start the pipeline
        self.pipeline.start(self.config)
        
        # Get the depth sensor and set units to meters
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.depth_units, 0.001)  # Set to meters
        
        # Initialize robot connection
        self.robot = urx.Robot(robot_ip)
        self.calibration_points: List[np.ndarray] = []
        self.robot_positions: List[List[float]] = []
        
        # Create calibration directory if it doesn't exist
        self.calibration_dir = Path(project_root) / "calibration"
        self.calibration_dir.mkdir(exist_ok=True)
        
        # Define checkerboard parameters
        self.pattern_size = (8, 13)  # Inner corners (one less than squares: 9-1, 14-1)
        self.square_size = 0.020    # Size of each square in meters (20mm)

    def generate_calibration_positions(self) -> List[List[float]]:
        """Generate a set of robot positions for calibration.
        
        Returns:
            List of robot joint positions [base, shoulder, elbow, wrist1, wrist2, wrist3]
        """
        # Define a set of positions with joint 5 (wrist2) fixed at -90 degrees
        # and ensuring z-axis constraints
        positions = [
            [0.0, -1.2, 1.0, -1.57, -1.57, 0.0],   # Front center (higher)
            [0.5, -1.2, 1.0, -1.57, -1.57, 0.0],   # Front right (higher)
            [-0.5, -1.2, 1.0, -1.57, -1.57, 0.0],  # Front left (higher)
            [0.0, -1.0, 0.8, -1.57, -1.57, 0.0],   # Front upper
            [0.0, -1.4, 1.2, -1.57, -1.57, 0.0],   # Front middle
            [0.3, -1.2, 1.0, -1.57, -1.57, 0.0],   # Side right (higher)
            [-0.3, -1.2, 1.0, -1.57, -1.57, 0.0],  # Side left (higher)
        ]
        return positions
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get color and depth frames from the RealSense camera.
        
        Returns:
            Tuple of (color_image, depth_image)
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    
    def capture_calibration_image(self) -> Tuple[bool, np.ndarray]:
        """Capture and process a calibration image."""
        # Get color image
        color_image, _ = self.get_frames()
        
        # Convert to grayscale for pattern detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Add text to show what we're looking for
        cv2.putText(color_image, f"Looking for {self.pattern_size[0]}x{self.pattern_size[1]} corners", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Try with different flags for better detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
        
        if ret:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw the corners
            cv2.drawChessboardCorners(color_image, self.pattern_size, corners2, ret)
            
            # Save the calibration point
            self.calibration_points.append(corners2)
            
            return True, color_image
        
        return False, color_image

    def calibrate_camera(self):
        """Perform camera calibration."""
        logger.info("Starting camera calibration...")
        
        # Create window for visualization
        cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)
        
        # Generate calibration positions
        positions = self.generate_calibration_positions()
        
        try:
            # Move to each position and capture images
            for i, pos in enumerate(positions):
                logger.info(f"Moving to position {i+1}/{len(positions)}")
                
                # Move robot to position - use slower movements for safety
                self.robot.movej(pos, acc=0.3, vel=0.2)
                
                time.sleep(2)  # Wait for robot to settle
                
                # Try to capture calibration image
                success = False
                attempts = 0
                manual_override = False
                
                while not success and attempts < 10:  # More attempts
                    # Show live feed until successful capture
                    color_image, _ = self.get_frames()
                    success, annotated_image = self.capture_calibration_image()
                    
                    # Add instructions for manual override
                    cv2.putText(annotated_image, "Press 'm' for manual capture if checkerboard visible", 
                              (20, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display the annotated image
                    cv2.imshow('Camera Feed', annotated_image)
                    
                    # Check for keyboard input
                    key = cv2.waitKey(1)
                    if key == ord('q'):  # Press 'q' to quit
                        raise KeyboardInterrupt
                    elif key == ord('s'):  # Press 's' to skip this position
                        break
                    elif key == ord('m'):  # Press 'm' for manual override
                        manual_override = True
                        success = True
                        self.calibration_points.append(np.zeros((self.pattern_size[0] * self.pattern_size[1], 1, 2), dtype=np.float32))
                        logger.info(f"Manual capture at position {i+1}")
                        cv2.imwrite(str(self.calibration_dir / f"manual_capture_{i}.jpg"), color_image)
                    
                    if not success:
                        attempts += 1
                        time.sleep(0.1)
                
                if success:
                    # Save the image
                    image_path = self.calibration_dir / f"calibration_{i}.jpg"
                    cv2.imwrite(str(image_path), annotated_image)
                    logger.info(f"Saved calibration image {i+1}")
                    # Store robot position
                    self.robot_positions.append(pos)
                else:
                    logger.warning(f"Skipping position {i+1} - could not detect checkerboard")

            # Prepare object points
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size

            # Get image size from the last captured image
            color_image, _ = self.get_frames()
            image_size = color_image.shape[:2][::-1]  # width, height

            # Perform calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                [objp] * len(self.calibration_points),  # object points
                self.calibration_points,                # image points
                image_size,                            # image size
                None, None                             # initial guess for camera matrix and distortion
            )
            
            # Save calibration results
            calibration_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
                "rotation_vectors": [r.tolist() for r in rvecs],
                "translation_vectors": [t.tolist() for t in tvecs],
                "robot_positions": self.robot_positions,
                "pattern_size": self.pattern_size,
                "square_size": self.square_size
            }
            
            calibration_file = self.calibration_dir / "calibration_data.json"
            with open(calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            
            logger.info(f"Calibration completed. Results saved to {calibration_file}")
            
            # Calculate and display calibration error
            mean_error = 0
            for i in range(len(self.calibration_points)):
                imgpoints2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.calibration_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            logger.info(f"Total calibration error: {mean_error/len(self.calibration_points)}")
            
            return mtx, dist, rvecs, tvecs
            
        finally:
            # Clean up OpenCV windows
            cv2.destroyAllWindows()

    def close(self):
        """Clean up resources."""
        self.pipeline.stop()
        self.robot.close()
        cv2.destroyAllWindows()  # Ensure windows are closed

def main():
    # Camera ID for RealSense D455
    camera_id = "207322251049"
    calibrator = None
    
    try:
        calibrator = CameraCalibrator(camera_id)
        mtx, dist, rvecs, tvecs = calibrator.calibrate_camera()
        
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        sys.exit(1)
    
    finally:
        # Clean up
        calibrator.close()

if __name__ == "__main__":
    main() 