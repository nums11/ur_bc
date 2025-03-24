#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import time

def test_camera(camera_id=0, width=640, height=480, fps=30, exposure=-1, gain=-1):
    """
    Test a single camera with optional settings adjustments
    
    Args:
        camera_id: Camera device ID (default 0)
        width: Resolution width (default 640)
        height: Resolution height (default 480)
        fps: Target frame rate (default 30)
        exposure: Camera exposure value, -1 means auto (default -1)
        gain: Camera gain value, -1 means auto (default -1)
    """
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Set exposure if provided (not -1)
    if exposure != -1:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 means manual exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(f"Set manual exposure: {exposure}")
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
        print("Using auto exposure")
    
    # Set gain if provided (not -1)
    if gain != -1:
        cap.set(cv2.CAP_PROP_GAIN, gain)
        print(f"Set gain: {gain}")
    
    # Display camera properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_gain = cap.get(cv2.CAP_PROP_GAIN)
    
    print(f"Camera initialized with:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps}")
    print(f"  Exposure: {actual_exposure}")
    print(f"  Gain: {actual_gain}")
    
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 1.0  # Update FPS display every second
    last_fps_update = start_time
    
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  + - Increase exposure (if manual)")
    print("  - - Decrease exposure (if manual)")
    print("  a - Toggle auto/manual exposure")
    
    is_auto_exposure = (exposure == -1)
    current_exposure = actual_exposure
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate and display FPS every second
        if current_time - last_fps_update >= fps_update_interval:
            fps = frame_count / (current_time - last_fps_update)
            last_fps_update = current_time
            frame_count = 0
            
            # Create a copy of the frame to add text
            display_frame = frame.copy()
            
            # Add FPS text and camera info
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Exposure: {'Auto' if is_auto_exposure else current_exposure}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Camera Test', display_frame)
        else:
            cv2.imshow('Camera Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"camera_{camera_id}_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
        elif key == ord('+') and not is_auto_exposure:
            # Increase exposure
            current_exposure += 1
            cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            print(f"Increased exposure to {current_exposure}")
        elif key == ord('-') and not is_auto_exposure:
            # Decrease exposure
            current_exposure -= 1
            cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            print(f"Decreased exposure to {current_exposure}")
        elif key == ord('a'):
            # Toggle auto/manual exposure
            is_auto_exposure = not is_auto_exposure
            if is_auto_exposure:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto
                print("Switched to auto exposure")
            else:
                current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual
                cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                print(f"Switched to manual exposure: {current_exposure}")
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    parser = argparse.ArgumentParser(description='Test a single camera')
    parser.add_argument('--camera_id', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera resolution width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera resolution height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--exposure', type=int, default=-1,
                       help='Camera exposure (-1 for auto, default: -1)')
    parser.add_argument('--gain', type=int, default=-1,
                       help='Camera gain (-1 for auto, default: -1)')
    
    args = parser.parse_args()
    
    test_camera(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        exposure=args.exposure,
        gain=args.gain
    )

if __name__ == "__main__":
    main() 