import cv2
import numpy as np
import multiprocessing
from time import sleep, time
import traceback

class LogitechCameraInterface:
    def __init__(self):
        # Camera indices
        self.top_camera_index = 1  # Top camera
        self.wrist_camera_index = 3  # Wrist camera
        
        # Queues for inter-process communication - limited size for real-time performance
        self.img1_queue = multiprocessing.Queue(maxsize=1)  # Only keep latest frame
        self.img2_queue = multiprocessing.Queue(maxsize=1)  # Only keep latest frame
        
        # Process control
        self.running = multiprocessing.Value('b', False)
        self.capture_process1 = None
        self.capture_process2 = None
        
        # Camera state
        self.camera1_connected = multiprocessing.Value('b', False)
        self.camera2_connected = multiprocessing.Value('b', False)
        self.camera1_error = multiprocessing.Value('b', False)
        self.camera2_error = multiprocessing.Value('b', False)
        
        # Frame rate control
        self.capture_interval = 0.02  # 50Hz capture rate
        self.display_interval = 0.033  # 30Hz display rate - increased for smoother display
        
        # Queue operation timeout (seconds)
        self.queue_timeout = 0.1  # 100ms timeout for queue operations
        
        # Display control
        self.last_display_time = time()
        
        # Last valid frames storage
        self.last_valid_img1 = np.zeros((480, 640, 3), dtype=np.uint8)  # Last valid top camera frame
        self.last_valid_img2 = np.zeros((480, 640, 3), dtype=np.uint8)  # Last valid wrist camera frame
        
        print("LogitechCameraInterface: Initialized Logitech Camera Interface")
    
    def startCapture(self):
        """Start both camera capture processes"""
        self.running.value = True
        
        # Start individual camera processes
        self.capture_process1 = multiprocessing.Process(
            target=self._captureLoopCamera1, 
            name="Camera1_Process"
        )
        self.capture_process2 = multiprocessing.Process(
            target=self._captureLoopCamera2,
            name="Camera2_Process"
        )
        
        # Start camera processes
        self.capture_process1.start()
        self.capture_process2.start()
        
        print("LogitechCameraInterface: Started camera capture processes")

    def _should_update_display(self):
        """Check if we should update the display based on display interval"""
        current_time = time()
        if current_time - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            return True
        return False
    
    def _is_valid_frame(self, frame):
        """Check if a frame is valid and usable"""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            print(f"Invalid frame: Not a numpy array, got {type(frame)}")
            return False
        if frame.size == 0:
            print(f"Invalid frame: Empty array with size 0")
            return False
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            print(f"Invalid frame: Bad dimensions {frame.shape}")
            return False
        if len(frame.shape) != 3:
            print(f"Invalid frame: Not a color image, shape is {frame.shape}")
            return False
        if frame.shape[0] != 480 or frame.shape[1] != 640 or frame.shape[2] != 3:
            print(f"Warning: Unexpected frame dimensions {frame.shape}, expected (480, 640, 3)")
            # Don't return False here, as we might be able to work with different dimensions
        
        return True

    def _captureLoopCamera1(self):
        """Process function for Camera 1 (Top)"""
        print(f"Starting Camera 1 (Top) process with index: {self.top_camera_index}")
        cap = None
        try:
            # Initialize camera
            cap = cv2.VideoCapture(self.top_camera_index)
            if not cap.isOpened():
                raise Exception(f"Could not open camera with index {self.top_camera_index}")
            
            # Set resolution - keeping the same resolution as before
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera1_connected.value = True
            print(f"Camera 1 (Top) successfully connected")
            
            # Main capture loop
            while self.running.value:
                try:
                    # Capture frame
                    ret, frame = cap.read()
                    
                    if not ret or not self._is_valid_frame(frame):
                        print("Camera 1 (Top): Failed to grab valid frame")
                        sleep(self.capture_interval)
                        continue
                    
                    # Store this valid frame as the last known good frame
                    # Need to use a manager for sharing this between processes
                    # For now, update the queue which will be used by getCurrentImage
                    
                    # Clear queue before putting new frame to ensure real-time updates
                    while not self.img1_queue.empty():
                        try:
                            self.img1_queue.get(block=False)  # Clear old frames
                        except:
                            break
                    
                    # Put frame in queue with timeout
                    try:
                        self.img1_queue.put(frame, block=True, timeout=self.queue_timeout)
                    except:
                        print("Camera 1 (Top): Queue full, frame dropped")
                    
                    # Update display if interval has passed
                    if self._should_update_display():
                        cv2.imshow('Top camera', frame)
                        cv2.waitKey(1)
                    
                    # Sleep to control frame rate
                    sleep(self.capture_interval)
                    
                except Exception as e:
                    self.camera1_error.value = True
                    print(f"Camera 1 (Top) error: {str(e)}")
                    traceback.print_exc()
                    sleep(0.5)  # Pause before retry
                
        except Exception as e:
            self.camera1_connected.value = False
            self.camera1_error.value = True
            print(f"Camera 1 (Top) failed to start: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Cleanup camera resources
            try:
                if cap is not None:
                    cap.release()
                self.camera1_connected.value = False
                print("Camera 1 (Top) released")
            except Exception as e:
                print(f"Error stopping Camera 1 (Top): {str(e)}")

    def _captureLoopCamera2(self):
        """Process function for Camera 2 (Wrist)"""
        print(f"Starting Camera 2 (Wrist) process with index: {self.wrist_camera_index}")
        cap = None
        try:
            # Initialize camera
            cap = cv2.VideoCapture(self.wrist_camera_index)
            if not cap.isOpened():
                raise Exception(f"Could not open camera with index {self.wrist_camera_index}")
            
            # Set resolution - keeping the same resolution as before
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera2_connected.value = True
            print(f"Camera 2 (Wrist) successfully connected")
            
            # Main capture loop
            while self.running.value:
                try:
                    # Capture frame
                    ret, frame = cap.read()
                    
                    if not ret or not self._is_valid_frame(frame):
                        print("Camera 2 (Wrist): Failed to grab valid frame")
                        sleep(self.capture_interval)
                        continue
                    
                    # Store this valid frame as the last known good frame
                    # Need to use a manager for sharing this between processes
                    # For now, update the queue which will be used by getCurrentImage
                    
                    # Clear queue before putting new frame to ensure real-time updates
                    while not self.img2_queue.empty():
                        try:
                            self.img2_queue.get(block=False)  # Clear old frames
                        except:
                            break
                    
                    # Put frame in queue with timeout
                    try:
                        self.img2_queue.put(frame, block=True, timeout=self.queue_timeout)
                    except:
                        print("Camera 2 (Wrist): Queue full, frame dropped")
                    
                    # Update display if interval has passed
                    if self._should_update_display():
                        cv2.imshow('Wrist camera', frame)
                        cv2.waitKey(1)
                    
                    # Sleep to control frame rate
                    sleep(self.capture_interval)
                    
                except Exception as e:
                    self.camera2_error.value = True
                    print(f"Camera 2 (Wrist) error: {str(e)}")
                    traceback.print_exc()
                    sleep(0.5)  # Pause before retry
                
        except Exception as e:
            self.camera2_connected.value = False
            self.camera2_error.value = True
            print(f"Camera 2 (Wrist) failed to start: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Cleanup camera resources
            try:
                if cap is not None:
                    cap.release()
                self.camera2_connected.value = False
                print("Camera 2 (Wrist) released")
            except Exception as e:
                print(f"Error stopping Camera 2 (Wrist): {str(e)}")

    def stopCapture(self):
        """Properly shut down camera processes and resources"""
        print("LogitechCameraInterface: Stopping camera capture")
        self.running.value = False
        
        # Stop camera processes
        if self.capture_process1 is not None:
            self.capture_process1.join(timeout=3)  # Add timeout to avoid hanging
            if self.capture_process1.is_alive():
                self.capture_process1.terminate()
            print("Camera 1 (Top) process stopped")
        
        if self.capture_process2 is not None:
            self.capture_process2.join(timeout=3)  # Add timeout to avoid hanging
            if self.capture_process2.is_alive():
                self.capture_process2.terminate()
            print("Camera 2 (Wrist) process stopped")
        
        # Close all windows
        cv2.destroyAllWindows()

    def getCurrentImage(self):
        """Get the current images from both cameras, waiting for next valid frames"""
        img1 = None
        img2 = None
        
        # Max time to wait for a valid frame (seconds)
        max_wait_time = 1.0
        start_time = time()
        
        # Get top camera image - keep trying until we get a valid frame or timeout
        while img1 is None and (time() - start_time) < max_wait_time:
            try:
                # Wait for a new frame to arrive with timeout
                frame = self.img1_queue.get(block=True, timeout=self.queue_timeout)
                if self._is_valid_frame(frame):
                    img1 = frame.copy()
                    # Store as last valid frame for future fallback
                    self.last_valid_img1 = img1.copy()
                else:
                    print(f"Camera 1 (Top): Retrieved invalid frame, waiting for next frame...")
                    sleep(0.01)  # Short sleep to prevent CPU spinning
            except Exception as e:
                if not str(e).startswith("Empty"):  # Don't log expected empty queue
                    print(f"Camera 1 (Top) get error: {str(e)}, waiting for next frame...")
                sleep(0.01)  # Short sleep to prevent CPU spinning
        
        # If we still don't have a valid frame after waiting, use last valid frame
        if img1 is None:
            print("Camera 1 (Top): Timeout waiting for valid frame, using last good frame")
            img1 = self.last_valid_img1.copy()
        
        # Reset timer for second camera
        start_time = time()
        
        # Get wrist camera image - keep trying until we get a valid frame or timeout
        while img2 is None and (time() - start_time) < max_wait_time:
            try:
                # Wait for a new frame to arrive with timeout
                frame = self.img2_queue.get(block=True, timeout=self.queue_timeout)
                if self._is_valid_frame(frame):
                    img2 = frame.copy()
                    # Store as last valid frame for future fallback
                    self.last_valid_img2 = img2.copy()
                else:
                    print(f"Camera 2 (Wrist): Retrieved invalid frame, waiting for next frame...")
                    sleep(0.01)  # Short sleep to prevent CPU spinning
            except Exception as e:
                if not str(e).startswith("Empty"):  # Don't log expected empty queue
                    print(f"Camera 2 (Wrist) get error: {str(e)}, waiting for next frame...")
                sleep(0.01)  # Short sleep to prevent CPU spinning
        
        # If we still don't have a valid frame after waiting, use last valid frame
        if img2 is None:
            print("Camera 2 (Wrist): Timeout waiting for valid frame, using last good frame")
            img2 = self.last_valid_img2.copy()
        
        # Final verification that we never return None
        if not self._is_valid_frame(img1):
            print("WARNING: Top camera frame still invalid, returning zeros")
            img1 = np.zeros((480, 640, 3), dtype=np.uint8)
            
        if not self._is_valid_frame(img2):
            print("WARNING: Wrist camera frame still invalid, returning zeros")
            img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return img1, img2

    def getCameraStatus(self):
        """Get the current status of both cameras"""
        return {
            'camera1_connected': self.camera1_connected.value,
            'camera2_connected': self.camera2_connected.value,
            'camera1_error': self.camera1_error.value,
            'camera2_error': self.camera2_error.value
        }

    def __del__(self):
        """Cleanup when the object is deleted"""
        self.stopCapture() 