import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

def main():
    # Create a pipeline to configure, start, and manage the RealSense camera
    pipeline = rs.pipeline()

    # Create a configuration object for the pipeline
    config = rs.config()

    # Enable the color stream (RGB)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline with the configuration
    pipeline.start(config)

    print("Starting RealSense camera. Press 's' to save an image, 'q' to exit.")

    try:
        while True:
            # Wait for a coherent set of frames (color frame)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the color frame to a NumPy array
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.flip(color_image, 0)

            # Display the image
            cv2.imshow('RealSense Camera', color_image)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Exit the loop if 'q' is pressed
                break
            elif key == ord('s'):
                # Save the image if 's' is pressed
                save_image(color_image)

    finally:
        # Stop the pipeline and release resources
        pipeline.stop()
        cv2.destroyAllWindows()

def save_image(image):
    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("/home/weirdlab/ur_bc/", f"image_{timestamp}.png")

    # Save the resized image using OpenCV
    cv2.imwrite(filename, resized_image)
    print(f"Image saved: {filename}")

if __name__ == "__main__":
    main()
