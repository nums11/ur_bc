import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from time import sleep
import sys
import signal

class RSCameraInterface:
    def __init__(self, serial_number=None):
        assert serial_number is not None, "Serial number of the RealSense camera must be provided"
        # Create a pipeline to configure, start, and manage the RealSense camera
        self.pipeline1 = rs.pipeline()
        self.pipeline2 = rs.pipeline()
        # Create a configuration object for the pipeline
        self.config1 = rs.config()
        self.config2 = rs.config()

        self.config1.enable_device('746112060198')
        self.config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.config2.enable_device('123622270802')
        self.config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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
        self.pipeline1.start(self.config1)
        self.pipeline2.start(self.config2)
        try:
            while self.running:
                # Wait for a coherent set of frames (color frame)
                frame1 = self.pipeline1.wait_for_frames()
                color_frame1 = frame1.get_color_frame()
                frame2 = self.pipeline2.wait_for_frames()
                color_frame2 = frame2.get_color_frame()
                if not color_frame1 or not color_frame2:
                    print("No frames")
                    continue
                # Convert the color frame to a NumPy array
                self.img1 = np.asanyarray(color_frame1.get_data()).copy()
                self.img2 = np.asanyarray(color_frame2.get_data()).copy()
                # Display the image
                cv2.imshow('Top camera', self.img1)
                cv2.imshow('Wrist camera', self.img2)
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
        if self.img1 is not None and self.img2 is not None:
            # Resize the image to 256x256
            # resized_image = cv2.resize(self.current_image, (256, 256))
            return self.img1, self.img2
        return None
    
    def __del__(self):
        self.stopCapture()