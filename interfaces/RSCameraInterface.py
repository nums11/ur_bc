import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from time import sleep
import sys
import signal

class RSCameraInterface:
    def __init__(self):
        # Create a pipeline to configure, start, and manage the RealSense camera
        self.pipeline = rs.pipeline()
        # Create a configuration object for the pipeline
        self.config = rs.config()
        # Enable the color stream (RGB)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start the pipeline with the configuration
        self.current_image = None
        self.capture_thread = None
        self.running = False
        print("RSCameraInterface: Initialized RealSense Camera")
    
    def startCapture(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._captureLoop)
        self.capture_thread.start()

    def _captureLoop(self):
        self.pipeline.start(self.config)
        try:
            while self.running:
                # Wait for a coherent set of frames (color frame)
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # Convert the color frame to a NumPy array
                self.current_image = np.asanyarray(color_frame.get_data())
                # Display the image
                cv2.imshow('RealSense Camera', self.current_image)
                cv2.waitKey(1) # Waits 1ms to allow the image to be displayed
        except KeyboardInterrupt:
            self.stopCapture()
            sys.exit(0)

    def stopCapture(self):
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def getCurrentImage(self):
        if self.current_image is not None:
            # Resize the image to 256x256
            resized_image = cv2.resize(self.current_image, (256, 256))
            return resized_image
        return None
    
    def __del__(self):
        self.stopCapture()