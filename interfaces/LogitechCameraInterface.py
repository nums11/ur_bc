import cv2
import numpy as np
import threading
from time import sleep, time
import traceback

class LogitechCameraInterface:
    def __init__(self):
        # Camera indices
        self.top_camera_index = 1  # Top camera
        self.wrist_camera_index = 3  # Wrist camera
        
        # Thread safety - locks to protect shared resources
        self.img1_lock = threading.Lock()
        self.img2_lock = threading.Lock()
        
        # Image storage
        self.img1 = None
        self.img2 = None
        
        # Thread control
        self.running = False
        self.capture_thread1 = None
        self.capture_thread2 = None
        
        # Camera state
        self.camera1_connected = False
        self.camera2_connected = False
        self.camera1_error = None
        self.camera2_error = None
        
        # Heartbeat tracking to detect frozen threads
        self.camera1_last_heartbeat = 0
        self.camera2_last_heartbeat = 0
        self.heartbeat_thread = None
        
        print("LogitechCameraInterface: Initialized Logitech Camera Interface")
    
    def startCapture(self):
        """Start both camera capture threads"""
        self.running = True
        
        # Start individual camera threads
        self.capture_thread1 = threading.Thread(
            target=self._captureLoopCamera1, 
            name="Camera1_Thread"
        )
        self.capture_thread2 = threading.Thread(
            target=self._captureLoopCamera2,
            name="Camera2_Thread"
        )
        
        # Set as daemon threads so they automatically terminate when main program exits
        self.capture_thread1.daemon = True
        self.capture_thread2.daemon = True
        
        # Start camera threads
        self.capture_thread1.start()
        self.capture_thread2.start()
        
        # Start heartbeat monitoring
        self.heartbeat_thread = threading.Thread(
            target=self._monitor_heartbeats,
            name="Heartbeat_Monitor"
        )
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        print("LogitechCameraInterface: Started camera capture threads")

    def _captureLoopCamera1(self):
        """Thread function for Camera 1 (Top)"""
        print(f"Starting Camera 1 (Top) thread with index: {self.top_camera_index}")
        try:
            # Initialize camera
            cap = cv2.VideoCapture(self.top_camera_index)
            if not cap.isOpened():
                raise Exception(f"Could not open camera with index {self.top_camera_index}")
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera1_connected = True
            print(f"Camera 1 (Top) successfully connected")
            
            # Main capture loop
            while self.running:
                try:
                    # Update heartbeat
                    self.camera1_last_heartbeat = time()
                    
                    # Capture frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Camera 1 (Top): Failed to grab frame")
                        sleep(0.01)
                        continue
                    
                    # Process the frame
                    with self.img1_lock:
                        self.img1 = frame.copy()
                    
                    # Display image (optional - can be disabled)
                    cv2.imshow('Top camera', self.img1)
                    cv2.waitKey(1)
                    
                    # Small sleep to prevent thread from hogging CPU
                    sleep(0.01)
                    
                except Exception as e:
                    self.camera1_error = f"Camera 1 (Top) error: {str(e)}"
                    print(self.camera1_error)
                    traceback.print_exc()
                    sleep(0.5)  # Pause before retry
                
        except Exception as e:
            self.camera1_connected = False
            self.camera1_error = f"Camera 1 (Top) failed to start: {str(e)}"
            print(self.camera1_error)
            traceback.print_exc()
            
        finally:
            # Cleanup camera resources
            try:
                if cap is not None:
                    cap.release()
                self.camera1_connected = False
                print("Camera 1 (Top) released")
            except Exception as e:
                print(f"Error stopping Camera 1 (Top): {str(e)}")

    def _captureLoopCamera2(self):
        """Thread function for Camera 2 (Wrist)"""
        print(f"Starting Camera 2 (Wrist) thread with index: {self.wrist_camera_index}")
        try:
            # Initialize camera
            cap = cv2.VideoCapture(self.wrist_camera_index)
            if not cap.isOpened():
                raise Exception(f"Could not open camera with index {self.wrist_camera_index}")
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera2_connected = True
            print(f"Camera 2 (Wrist) successfully connected")
            
            # Main capture loop
            while self.running:
                try:
                    # Update heartbeat
                    self.camera2_last_heartbeat = time()
                    
                    # Capture frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Camera 2 (Wrist): Failed to grab frame")
                        sleep(0.01)
                        continue
                    
                    # Process the frame
                    with self.img2_lock:
                        self.img2 = frame.copy()
                    
                    # Display image (optional - can be disabled)
                    cv2.imshow('Wrist camera', self.img2)
                    cv2.waitKey(1)
                    
                    # Small sleep to prevent thread from hogging CPU
                    sleep(0.01)
                    
                except Exception as e:
                    self.camera2_error = f"Camera 2 (Wrist) error: {str(e)}"
                    print(self.camera2_error)
                    traceback.print_exc()
                    sleep(0.5)  # Pause before retry
                
        except Exception as e:
            self.camera2_connected = False
            self.camera2_error = f"Camera 2 (Wrist) failed to start: {str(e)}"
            print(self.camera2_error)
            traceback.print_exc()
            
        finally:
            # Cleanup camera resources
            try:
                if cap is not None:
                    cap.release()
                self.camera2_connected = False
                print("Camera 2 (Wrist) released")
            except Exception as e:
                print(f"Error stopping Camera 2 (Wrist): {str(e)}")

    def _monitor_heartbeats(self):
        """Thread to monitor camera threads and detect if they freeze"""
        HEARTBEAT_TIMEOUT = 10  # seconds
        
        while self.running:
            current_time = time()
            
            # Check Camera 1 heartbeat
            if self.camera1_connected and current_time - self.camera1_last_heartbeat > HEARTBEAT_TIMEOUT:
                print(f"WARNING: Camera 1 (Top) thread may be frozen! Last heartbeat: {self.camera1_last_heartbeat}")
            
            # Check Camera 2 heartbeat
            if self.camera2_connected and current_time - self.camera2_last_heartbeat > HEARTBEAT_TIMEOUT:
                print(f"WARNING: Camera 2 (Wrist) thread may be frozen! Last heartbeat: {self.camera2_last_heartbeat}")
            
            sleep(2)  # Check every 2 seconds

    def stopCapture(self):
        """Properly shut down camera threads and resources"""
        print("LogitechCameraInterface: Stopping camera capture")
        self.running = False
        
        # Stop camera threads with timeout
        if self.capture_thread1 and self.capture_thread1.is_alive():
            print("Waiting for Camera 1 (Top) thread to stop...")
            self.capture_thread1.join(timeout=3)
            if self.capture_thread1.is_alive():
                print("WARNING: Camera 1 (Top) thread did not stop gracefully")
        
        if self.capture_thread2 and self.capture_thread2.is_alive():
            print("Waiting for Camera 2 (Wrist) thread to stop...")
            self.capture_thread2.join(timeout=3)
            if self.capture_thread2.is_alive():
                print("WARNING: Camera 2 (Wrist) thread did not stop gracefully")

    def getCurrentImage(self):
        """Get the current images from both cameras"""
        img1 = None
        img2 = None
        
        with self.img1_lock:
            if self.img1 is not None:
                img1 = self.img1.copy()
        
        with self.img2_lock:
            if self.img2 is not None:
                img2 = self.img2.copy()
        
        return img1, img2

    def getCameraStatus(self):
        """Get the current status of both cameras"""
        return {
            'camera1_connected': self.camera1_connected,
            'camera2_connected': self.camera2_connected,
            'camera1_error': self.camera1_error,
            'camera2_error': self.camera2_error
        }

    def __del__(self):
        """Cleanup when the object is deleted"""
        self.stopCapture() 