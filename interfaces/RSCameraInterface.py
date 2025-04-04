import pyrealsense2 as rs
import numpy as np
import cv2
import multiprocessing
from time import sleep, time
import sys
import signal
import traceback

class RSCameraInterface:
    def __init__(self, serial_number=None):
        assert serial_number is not None, "Serial number of the RealSense camera must be provided"
        
        # Camera serial numbers and identifiers
        self.cam1_serial = '207322251049'  # Top camera (D455)
        self.cam2_serial = '746112060198'  # Wrist camera (D415)
        
        # Camera exposure settings
        self.cam1_exposure = 200  # Fixed exposure for cam1 (Top D455)
        self.cam2_exposure = 500  # Fixed exposure for cam2 (Wrist D415)
        
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
        self.camera1_error = multiprocessing.Value('i', 0)  # 0 = no error
        self.camera2_error = multiprocessing.Value('i', 0)  # 0 = no error
        
        # Queue operation timeout (seconds)
        self.queue_timeout = 0.1  # 100ms timeout for queue operations
        
        # Frame rate control
        self.capture_interval = 0.02  # 50Hz capture rate
        self.display_interval = 0.033  # 30Hz display rate
        
        # Last valid frames storage - for fallback
        self.last_valid_img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.last_valid_img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("RSCameraInterface: Initialized RealSense Camera Interface")
    
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
        
        print("RSCameraInterface: Started camera capture processes")
    
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
        return True
    
    def _should_update_display(self, last_display_time):
        """Check if we should update the display based on display interval"""
        current_time = time()
        if current_time - last_display_time >= self.display_interval:
            return True, current_time
        return False, last_display_time

    def _captureLoopCamera1(self):
        """Process function for Camera 1 (Top)"""
        print(f"Starting Camera 1 (Top) process with serial: {self.cam1_serial}")
        pipeline = None
        last_display_time = 0
        
        try:
            # Create a pipeline for this process
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable specific device
            config.enable_device(self.cam1_serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start the pipeline
            profile = pipeline.start(config)
            
            # Configure fixed exposure for Camera 1
            sensor = profile.get_device().first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto exposure
            sensor.set_option(rs.option.exposure, self.cam1_exposure)  # Set fixed exposure value
            print(f"Camera 1 (Top) exposure set to {self.cam1_exposure}")
            
            self.camera1_connected.value = True
            print(f"Camera 1 (Top) successfully connected")
            
            # Main capture loop
            while self.running.value:
                try:
                    # Wait for frames with timeout
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("Camera 1 (Top): No color frame received")
                        sleep(self.capture_interval)
                        continue
                    
                    # Process the frame
                    frame = np.asanyarray(color_frame.get_data()).copy()
                    
                    if not self._is_valid_frame(frame):
                        print("Camera 1 (Top): Invalid frame")
                        sleep(self.capture_interval)
                        continue
                    
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
                    should_update, last_display_time = self._should_update_display(last_display_time)
                    if should_update:
                        cv2.imshow('Top camera', frame)
                        cv2.waitKey(1)
                    
                    # Sleep to control frame rate
                    sleep(self.capture_interval)
                    
                except rs.error as e:
                    self.camera1_error.value = 1
                    print(f"Camera 1 (Top) RealSense error: {str(e)}")
                    sleep(0.5)  # Pause before retry
                    
                except Exception as e:
                    self.camera1_error.value = 1
                    print(f"Camera 1 (Top) error: {str(e)}")
                    traceback.print_exc()
                    sleep(0.5)  # Pause before retry
                
        except Exception as e:
            self.camera1_connected.value = False
            self.camera1_error.value = 1
            print(f"Camera 1 (Top) failed to start: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Cleanup camera resources
            try:
                if pipeline is not None:
                    pipeline.stop()
                self.camera1_connected.value = False
                print("Camera 1 (Top) pipeline stopped")
            except Exception as e:
                print(f"Error stopping Camera 1 (Top): {str(e)}")

    def _captureLoopCamera2(self):
        """Process function for Camera 2 (Wrist)"""
        print(f"Starting Camera 2 (Wrist) process with serial: {self.cam2_serial}")
        pipeline = None
        last_display_time = 0
        
        try:
            # Create a pipeline for this process
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable specific device
            config.enable_device(self.cam2_serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start the pipeline
            profile = pipeline.start(config)
            
            # Configure fixed exposure for Camera 2
            sensor = profile.get_device().first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto exposure
            sensor.set_option(rs.option.exposure, self.cam2_exposure)  # Set fixed exposure value
            print(f"Camera 2 (Wrist) exposure set to {self.cam2_exposure}")
            
            self.camera2_connected.value = True
            print(f"Camera 2 (Wrist) successfully connected")
            
            # Main capture loop
            while self.running.value:
                try:
                    # Wait for frames with timeout
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("Camera 2 (Wrist): No color frame received")
                        sleep(self.capture_interval)
                        continue
                    
                    # Process the frame
                    frame = np.asanyarray(color_frame.get_data()).copy()
                    
                    if not self._is_valid_frame(frame):
                        print("Camera 2 (Wrist): Invalid frame")
                        sleep(self.capture_interval)
                        continue
                    
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
                    should_update, last_display_time = self._should_update_display(last_display_time)
                    if should_update:
                        cv2.imshow('Wrist camera', frame)
                        cv2.waitKey(1)
                    
                    # Sleep to control frame rate
                    sleep(self.capture_interval)
                    
                except rs.error as e:
                    self.camera2_error.value = 1
                    print(f"Camera 2 (Wrist) RealSense error: {str(e)}")
                    sleep(0.5)  # Pause before retry
                    
                except Exception as e:
                    self.camera2_error.value = 1
                    print(f"Camera 2 (Wrist) error: {str(e)}")
                    traceback.print_exc()
                    sleep(0.5)  # Pause before retry
                
        except Exception as e:
            self.camera2_connected.value = False
            self.camera2_error.value = 1
            print(f"Camera 2 (Wrist) failed to start: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Cleanup camera resources
            try:
                if pipeline is not None:
                    pipeline.stop()
                self.camera2_connected.value = False
                print("Camera 2 (Wrist) pipeline stopped")
            except Exception as e:
                print(f"Error stopping Camera 2 (Wrist): {str(e)}")

    def stopCapture(self):
        """Properly shut down camera processes and resources"""
        print("RSCameraInterface: Stopping camera capture")
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
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("RSCameraInterface: Camera capture stopped")

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
            'camera1_error': self.camera1_error.value > 0,
            'camera2_error': self.camera2_error.value > 0
        }
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        try:
            self.stopCapture()
        except:
            pass