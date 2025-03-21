import cv2
import sys
import numpy as np

def main():
    # Open both cameras
    cap1 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(3)
    
    # Check if both cameras opened successfully
    if not cap1.isOpened() or not cap3.isOpened():
        print("Error: Could not open one or both cameras")
        sys.exit(1)
    
    print("Cameras opened successfully. Press 'q' to quit.")
    
    while True:
        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret3, frame3 = cap3.read()
        
        # If either frame is not read correctly, break
        if not ret1 or not ret3:
            print("Error: Can't receive frames from one or both cameras. Exiting...")
            break
            
        # Resize frames to have the same height (optional)
        height = min(frame1.shape[0], frame3.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height/frame1.shape[0]), height))
        frame3 = cv2.resize(frame3, (int(frame3.shape[1] * height/frame3.shape[0]), height))
        
        # Combine frames horizontally
        combined_frame = np.hstack((frame1, frame3))
        
        # Display the combined frame
        cv2.imshow('Dual Camera Stream', combined_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release both cameras and close all windows
    cap1.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 