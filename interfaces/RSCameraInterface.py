import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from time import sleep, time
import sys
import signal
import traceback
import concurrent.futures

class RSCameraInterface:
    def __init__(self, serial_number=None):
        assert serial_number is not None, "Serial number of the RealSense camera must be provided"
        
        # Camera serial numbers and identifiers
        self.cam1_serial = '746112060198'  # Top camera
        self.cam2_serial = '123622270810'  # Wrist camera
        
        # Create pipelines and configurations
        self.pipeline1 = rs.pipeline()
        self.pipeline2 = rs.pipeline()
        self.config1 = rs.config()
        self.config2 = rs.config()
        
        # Configure cameras
        self.config1.enable_device(self.cam1_serial)
        self.config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.config2.enable_device(self.cam2_serial)
        self.config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
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
        
        print("RSCameraInterface: Initialized RealSense Camera Interface")
    
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
        
        print("RSCameraInterface: Started camera capture threads")

    def _captureLoopCamera1(self):
        """Thread function for Camera 1 (Top)"""
        print(f"Starting Camera 1 (Top) thread with serial: {self.cam1_serial}")
        try:
            # Start the pipeline
            self.pipeline1.start(self.config1)
            self.camera1_connected = True
            print(f"Camera 1 (Top) successfully connected")
            
            # Main capture loop
            while self.running:
                try:
                    # Update heartbeat
                    self.camera1_last_heartbeat = time()
                    
                    # Wait for frames with timeout
                    frames = self.pipeline1.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("Camera 1 (Top): No color frame received")
                        sleep(0.01)
                        continue
                    
                    # Process the frame
                    with self.img1_lock:
                        self.img1 = np.asanyarray(color_frame.get_data()).copy()
                    
                    # Display image (optional - can be disabled)
                    cv2.imshow('Top camera', self.img1)
                    cv2.waitKey(1)
                    
                    # Small sleep to prevent thread from hogging CPU
                    sleep(0.01)
                    
                except rs.error as e:
                    self.camera1_error = f"Camera 1 (Top) RealSense error: {str(e)}"
                    print(self.camera1_error)
                    sleep(0.5)  # Pause before retry
                    
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
                if self.camera1_connected:
                    self.pipeline1.stop()
                    self.camera1_connected = False
                    print("Camera 1 (Top) pipeline stopped")
            except Exception as e:
                print(f"Error stopping Camera 1 (Top): {str(e)}")

    def _captureLoopCamera2(self):
        """Thread function for Camera 2 (Wrist)"""
        print(f"Starting Camera 2 (Wrist) thread with serial: {self.cam2_serial}")
        try:
            # Start the pipeline
            self.pipeline2.start(self.config2)
            self.camera2_connected = True
            print(f"Camera 2 (Wrist) successfully connected")
            
            # Main capture loop
            while self.running:
                try:
                    # Update heartbeat
                    self.camera2_last_heartbeat = time()
                    
                    # Wait for frames with timeout
                    frames = self.pipeline2.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("Camera 2 (Wrist): No color frame received")
                        sleep(0.01)
                        continue
                    
                    # Process the frame
                    with self.img2_lock:
                        self.img2 = np.asanyarray(color_frame.get_data()).copy()
                    
                    # Display image (optional - can be disabled)
                    cv2.imshow('Wrist camera', self.img2)
                    cv2.waitKey(1)
                    
                    # Small sleep to prevent thread from hogging CPU
                    sleep(0.01)
                    
                except rs.error as e:
                    self.camera2_error = f"Camera 2 (Wrist) RealSense error: {str(e)}"
                    print(self.camera2_error)
                    sleep(0.5)  # Pause before retry
                    
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
                if self.camera2_connected:
                    self.pipeline2.stop()
                    self.camera2_connected = False
                    print("Camera 2 (Wrist) pipeline stopped")
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
                # You can optionally try to restart the thread here
            
            # Check Camera 2 heartbeat
            if self.camera2_connected and current_time - self.camera2_last_heartbeat > HEARTBEAT_TIMEOUT:
                print(f"WARNING: Camera 2 (Wrist) thread may be frozen! Last heartbeat: {self.camera2_last_heartbeat}")
                # You can optionally try to restart the thread here
            
            sleep(2)  # Check every 2 seconds

    def stopCapture(self):
        """Properly shut down camera threads and resources"""
        print("RSCameraInterface: Stopping camera capture")
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
        
        # Stop heartbeat monitor
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2)
        
        # Ensure pipelines are stopped
        try:
            if self.camera1_connected:
                self.pipeline1.stop()
                print("Camera 1 (Top) pipeline stopped")
        except Exception as e:
            print(f"Error stopping Camera 1 (Top) pipeline: {e}")
            
        try:
            if self.camera2_connected:
                self.pipeline2.stop()
                print("Camera 2 (Wrist) pipeline stopped")
        except Exception as e:
            print(f"Error stopping Camera 2 (Wrist) pipeline: {e}")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("RSCameraInterface: Camera capture stopped")

    def getCurrentImage(self):
        """Thread-safe method to get the current images from both cameras"""
        img1_copy = None
        img2_copy = None
        
        # Get Camera 1 image with lock protection
        if self.img1 is not None:
            with self.img1_lock:
                img1_copy = self.img1.copy() if self.img1 is not None else None
        
        # Get Camera 2 image with lock protection
        if self.img2 is not None:
            with self.img2_lock:
                img2_copy = self.img2.copy() if self.img2 is not None else None
        
        # Return only if both images are available
        if img1_copy is not None and img2_copy is not None:
            return img1_copy, img2_copy
        
        # Report which camera is missing if either is None
        if img1_copy is None:
            print("Camera 1 (Top) image not available")
        if img2_copy is None:
            print("Camera 2 (Wrist) image not available") 
        
        return None

    def getCameraStatus(self):
        """Get the current status of both cameras"""
        return {
            'camera1': {
                'connected': self.camera1_connected,
                'error': self.camera1_error,
                'last_heartbeat': self.camera1_last_heartbeat
            },
            'camera2': {
                'connected': self.camera2_connected,
                'error': self.camera2_error,
                'last_heartbeat': self.camera2_last_heartbeat
            }
        }
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        try:
            self.stopCapture()
        except:
            pass