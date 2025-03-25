import cv2
import sys
import numpy as np

def setup_camera(cap):
    """Configure camera settings with default values"""
    # Disable automatic settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_GAIN, 0)  # Set gain to minimum
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # Start with minimum focus

    # Set manual values
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 128)

def main():
    # Open both cameras
    cap1 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(3)
    
    # Check if both cameras opened successfully
    if not cap1.isOpened() or not cap3.isOpened():
        print("Error: Could not open one or both cameras")
        sys.exit(1)
    
    # Initialize camera settings
    setup_camera(cap1)
    setup_camera(cap3)
    
    # Initialize parameters
    exposure = -6
    brightness = 128
    contrast = 128
    gain = 0
    focus = 0
    wb_auto = False
    
    print("Camera Controls:")
    print("E/D: Increase/Decrease Exposure")
    print("B/N: Increase/Decrease Brightness")
    print("C/V: Increase/Decrease Contrast")
    print("G/H: Increase/Decrease Gain")
    print("F/J: Increase/Decrease Focus")
    print("W: Toggle White Balance")
    print("R: Reset to defaults")
    print("Q: Quit")
    
    while True:
        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret3, frame3 = cap3.read()
        
        if not ret1 or not ret3:
            print("Error: Can't receive frames from one or both cameras. Exiting...")
            break
            
        # Resize frames to have the same height
        height = min(frame1.shape[0], frame3.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height/frame1.shape[0]), height))
        frame3 = cv2.resize(frame3, (int(frame3.shape[1] * height/frame3.shape[0]), height))
        
        # Add parameter values to the frames
        info_text = f"Exp: {exposure} | Bright: {brightness} | Cont: {contrast} | Gain: {gain} | Focus: {focus} | WB: {'Auto' if wb_auto else 'Off'}"
        cv2.putText(frame1, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine frames horizontally
        combined_frame = np.hstack((frame1, frame3))
        
        # Display the combined frame
        cv2.imshow('Dual Camera Stream', combined_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):  # Increase exposure
            exposure = min(exposure + 1, -1)
            cap1.set(cv2.CAP_PROP_EXPOSURE, exposure)
            cap3.set(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key == ord('d'):  # Decrease exposure
            exposure = max(exposure - 1, -8)
            cap1.set(cv2.CAP_PROP_EXPOSURE, exposure)
            cap3.set(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key == ord('b'):  # Increase brightness
            brightness = min(brightness + 5, 255)
            cap1.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            cap3.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        elif key == ord('n'):  # Decrease brightness
            brightness = max(brightness - 5, 0)
            cap1.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            cap3.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        elif key == ord('c'):  # Increase contrast
            contrast = min(contrast + 5, 255)
            cap1.set(cv2.CAP_PROP_CONTRAST, contrast)
            cap3.set(cv2.CAP_PROP_CONTRAST, contrast)
        elif key == ord('v'):  # Decrease contrast
            contrast = max(contrast - 5, 0)
            cap1.set(cv2.CAP_PROP_CONTRAST, contrast)
            cap3.set(cv2.CAP_PROP_CONTRAST, contrast)
        elif key == ord('g'):  # Increase gain
            gain = min(gain + 1, 100)
            cap1.set(cv2.CAP_PROP_GAIN, gain)
            cap3.set(cv2.CAP_PROP_GAIN, gain)
        elif key == ord('h'):  # Decrease gain
            gain = max(gain - 1, 0)
            cap1.set(cv2.CAP_PROP_GAIN, gain)
            cap3.set(cv2.CAP_PROP_GAIN, gain)
        elif key == ord('f'):  # Increase focus
            focus = min(focus + 5, 255)
            cap1.set(cv2.CAP_PROP_FOCUS, focus)
            cap3.set(cv2.CAP_PROP_FOCUS, focus)
        elif key == ord('j'):  # Decrease focus
            focus = max(focus - 5, 0)
            cap1.set(cv2.CAP_PROP_FOCUS, focus)
            cap3.set(cv2.CAP_PROP_FOCUS, focus)
        elif key == ord('w'):  # Toggle white balance
            wb_auto = not wb_auto
            cap1.set(cv2.CAP_PROP_AUTO_WB, 1 if wb_auto else 0)
            cap3.set(cv2.CAP_PROP_AUTO_WB, 1 if wb_auto else 0)
        elif key == ord('r'):  # Reset to defaults
            exposure = -6
            brightness = 128
            contrast = 128
            gain = 0
            focus = 0
            wb_auto = False
            setup_camera(cap1)
            setup_camera(cap3)
    
    # Release both cameras and close all windows
    cap1.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 