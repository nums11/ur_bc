import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams for multiple cameras
pipeline_1 = rs.pipeline()
pipeline_2 = rs.pipeline()

config_1 = rs.config()
config_2 = rs.config()

# Enable the first camera
# D405 123622270802
config_1.enable_device('123622270802')
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Enable the second camera
# D415 746112060198
config_2.enable_device('746112060198')
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()

        frames_2 = pipeline_2.wait_for_frames()
        color_frame_2 = frames_2.get_color_frame()

        if not color_frame_1 or not color_frame_2:
            continue

        # Convert images to numpy arrays
        color_image_1 = np.asanyarray(color_frame_1.get_data()).copy()

        color_image_2 = np.asanyarray(color_frame_2.get_data()).copy()

        # Show images from both cameras
        cv2.imshow('RealSense Camera 1', color_image_1)
        cv2.imshow('RealSense Camera 2', color_image_2)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
    cv2.destroyAllWindows()