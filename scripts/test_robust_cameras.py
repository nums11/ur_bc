#!/usr/bin/env python3
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from interfaces.RSCameraInterface import RSCameraInterface

def display_camera_status(status):
    """Pretty print camera status information"""
    print("\n===== Camera Status =====")
    
    # Camera 1 status
    print(f"Camera 1 (Top):")
    print(f"  Connected: {status['camera1']['connected']}")
    if status['camera1']['error']:
        print(f"  Error: {status['camera1']['error']}")
    if status['camera1']['last_heartbeat'] > 0:
        time_since = time.time() - status['camera1']['last_heartbeat']
        print(f"  Last heartbeat: {time_since:.2f} seconds ago")
    
    # Camera 2 status
    print(f"Camera 2 (Wrist):")
    print(f"  Connected: {status['camera2']['connected']}")
    if status['camera2']['error']:
        print(f"  Error: {status['camera2']['error']}")
    if status['camera2']['last_heartbeat'] > 0:
        time_since = time.time() - status['camera2']['last_heartbeat'] 
        print(f"  Last heartbeat: {time_since:.2f} seconds ago")
    
    print("========================\n")

def main():
    print("Testing robust RealSense camera interface with separate threads")
    print("Press 'q' to exit, 's' to get current status")
    
    # Initialize camera interface with any serial number (not actually used anymore)
    camera = RSCameraInterface(serial_number='any')
    
    try:
        # Start camera capture threads
        camera.startCapture()
        
        # Main loop
        while True:
            # Give some time for camera threads to update
            time.sleep(0.1)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
                
            elif key == ord('s'):
                # Display camera status
                status = camera.getCameraStatus()
                display_camera_status(status)
                
            # You can also get camera images here for processing
            images = camera.getCurrentImage()
            if images:
                img1, img2 = images
                # Do something with the images if needed
                # For now we just display them via the camera threads
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    finally:
        # Clean up
        print("Stopping camera capture...")
        camera.stopCapture()
        print("Done!")

if __name__ == "__main__":
    main() 