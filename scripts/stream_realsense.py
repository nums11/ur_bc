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

    # D455 207322251049
    # D415 746112060198
    config.enable_device('746112060198')

    # Enable the color stream (RGB)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline with the configuration
    profile = pipeline.start(config)

    # Get the sensor to control exposure
    sensor = profile.get_device().first_color_sensor()
    
    # Initial exposure settings
    auto_exposure = True
    exposure_value = 100  # Initial manual exposure value in microseconds
    
    # Set initial auto exposure
    sensor.set_option(rs.option.enable_auto_exposure, 1)

    print("Starting RealSense camera. Press 's' to save an image, 'q' to exit.")
    print("Press 'e' to toggle auto/manual exposure, '+'/'-' to adjust manual exposure.")

    try:
        while True:
            # Wait for a coherent set of frames (color frame)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the color frame to a NumPy array
            color_image = np.asanyarray(color_frame.get_data())

            # Add exposure info to the displayed image
            exposure_text = f"Exposure: {'Auto' if auto_exposure else f'Manual ({exposure_value})'}"
            cv2.putText(color_image, exposure_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
            elif key == ord('e'):
                # Toggle between auto and manual exposure
                auto_exposure = not auto_exposure
                if auto_exposure:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    print("Auto exposure enabled")
                else:
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    sensor.set_option(rs.option.exposure, exposure_value)
                    print(f"Manual exposure enabled: {exposure_value}")
            elif key == ord('+') or key == ord('='):
                # Increase manual exposure
                if not auto_exposure:
                    exposure_value = min(exposure_value + 50, 10000)  # Max value
                    sensor.set_option(rs.option.exposure, exposure_value)
                    print(f"Exposure increased to: {exposure_value}")
            elif key == ord('-') or key == ord('_'):
                # Decrease manual exposure
                if not auto_exposure:
                    exposure_value = max(exposure_value - 50, 1)  # Min value
                    sensor.set_option(rs.option.exposure, exposure_value)
                    print(f"Exposure decreased to: {exposure_value}")

    finally:
        # Stop the pipeline and release resources
        print("Stopping RealSense camera.")
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
