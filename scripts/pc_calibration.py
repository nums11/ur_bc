import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
import urx
import os
import json
from datetime import datetime

# ========================
# ---- CONFIGURATION ----
# ========================
NUM_POSES = 15  # How many poses to capture
DEPTH_SCALE = 0.001  # RealSense D435 is in mm, usually 0.001
OUTPUT_DIR = "calibration_data"  # Directory to save calibration data

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect to UR5
robot = urx.Robot("192.168.1.2")  # Change to your robot's IP

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('207322251049')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth to color
align = rs.align(rs.stream.color)

points_camera = []
points_robot = []

def get_click_coordinates(image):
    coords = []
    img_display = image.copy()  # Create a copy for display

    def click_event(event, x, y, flags, param):
        nonlocal img_display, coords
        if event == cv2.EVENT_LBUTTONDOWN:
            # Reset previous clicks if needed
            if coords:
                img_display = image.copy()  # Reset display image
                coords = []  # Clear previous coordinates
            
            # Add new click coordinates
            coords.append((x, y))
            
            # Draw visual marker on click location
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)  # Green circle
            cv2.circle(img_display, (x, y), 8, (255, 0, 0), 2)   # Red circle outline
            
            # Draw crosshairs
            cv2.line(img_display, (x-15, y), (x+15, y), (0, 255, 255), 2)  # Horizontal
            cv2.line(img_display, (x, y-15), (x, y+15), (0, 255, 255), 2)  # Vertical
            
            # Add text showing coordinates
            text = f"({x}, {y})"
            cv2.putText(img_display, text, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Update display
            cv2.imshow("Click End Effector", img_display)
            print(f"Clicked at coordinates: ({x}, {y})")

    # Initial window setup with instructions
    instruction_img = img_display.copy()
    cv2.putText(instruction_img, "Click on the end effector", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(instruction_img, "Press any key when done", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Click End Effector", instruction_img)
    cv2.setMouseCallback("Click End Effector", click_event)
    print("Click the end effector in the image. Press any key after clicking.")
    
    # Wait for a key press
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    # Force a small delay to ensure windows are closed
    cv2.waitKey(1)
    
    if not coords:
        print("No point was selected!")
        return None
    
    print(f"Selected point at coordinates: {coords[0]}")
    return coords[0]

# =====================
# ---- DATA LOOP ----
# =====================
i = 0
while i < NUM_POSES:
    user_input = input(f"\nMove robot to pose #{i+1}/{NUM_POSES} and press ENTER (or type 'x' to skip)... ")
    
    if user_input.lower() == 'x':
        print("Skipping this pose. Move robot to a new position.")
        continue
    
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        print("Frame capture failed.")
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Just use the original image without text overlays
    display_image = color_image.copy()
    
    # Use a simpler approach with mutable objects to avoid scope issues
    click_result: list[tuple[int, int]] = []
    
    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Using a list as a mutable container to store results
            result_list = param
            
            # Clear any previous clicks
            if result_list:
                result_list.clear()
                # Refresh the display image
                cv2.imshow("Select End Effector", display_image)
            
            # Store coordinates
            result_list.append((x, y))
            
            # Create a temporary image for drawing
            temp_img = display_image.copy()
            
            # Draw markers
            cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)  # Green circle
            cv2.circle(temp_img, (x, y), 8, (255, 0, 0), 2)   # Red circle outline
            cv2.line(temp_img, (x-15, y), (x+15, y), (0, 255, 255), 2)  # Horizontal
            cv2.line(temp_img, (x, y-15), (x, y+15), (0, 255, 255), 2)  # Vertical
            
            # Add coordinates text
            text = f"({x}, {y})"
            cv2.putText(temp_img, text, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Update display
            cv2.imshow("Select End Effector", temp_img)
            print(f"Clicked at coordinates: ({x}, {y})")
    
    # Show image and setup click callback
    cv2.imshow("Select End Effector", display_image)
    cv2.setMouseCallback("Select End Effector", click_handler, click_result)
    
    # Print instructions in the console instead of on the image
    print("\nINSTRUCTIONS:")
    print("- Click on the robot end effector in the image")
    print("- Press 'x' to retry with a different pose")
    print("- Press any other key to confirm your selection")
    
    # Wait for key press - 'x' to retry, any other key to continue
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensure windows close properly
    
    if key == ord('x'):
        print("Retrying with a different pose...")
        continue
    
    if not click_result:
        print("No point selected. Retrying...")
        continue
    
    u, v = click_result[0]
    depth = depth_image[v, u] * depth_scale
    
    # Check if depth is valid
    if depth <= 0 or depth > 10:  # 10 meters is a reasonable maximum distance
        print(f"Invalid depth value: {depth} meters. Retrying...")
        continue

    # Get intrinsics
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    x = (u - intrinsics.ppx) / intrinsics.fx
    y = (v - intrinsics.ppy) / intrinsics.fy
    z = depth
    cam_point = np.array([x * z, y * z, z])

    print(f"Camera frame 3D point: {cam_point}")

    # Get robot pose
    ee_pose = robot.get_pose()
    base_point = ee_pose.pos  # returns URX's urx.transformation.Position (x, y, z)
    base_point = np.array([base_point[0], base_point[1], base_point[2]])

    points_camera.append(cam_point)
    points_robot.append(base_point)
    
    # Only increment counter if we successfully captured this pose
    i += 1
    print(f"Successfully captured pose {i}/{NUM_POSES}")

print(f"\nCompleted data collection with {len(points_camera)} poses")

# =============================
# ---- RIGID TRANSFORM FIT ----
# =============================
def estimate_transform(A, B):
    assert len(A) == len(B)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

points_camera_np = np.array(points_camera)
points_robot_np = np.array(points_robot)

T_cam_to_base = estimate_transform(points_camera_np, points_robot_np)

print("\nEstimated Transformation (Camera to Robot Base):")
print(T_cam_to_base)

# ============================
# ---- SAVE CALIBRATION DATA ----
# ============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"calibration_{timestamp}")

# Save camera points
np.save(f"{output_file}_camera_points.npy", points_camera_np)

# Save robot points
np.save(f"{output_file}_robot_points.npy", points_robot_np)

# Save transformation matrix
np.save(f"{output_file}_transform.npy", T_cam_to_base)

# Save a readable text file with the transformation
with open(f"{output_file}_transform.txt", 'w') as f:
    f.write("Camera to Robot Base Transformation Matrix:\n")
    f.write(str(T_cam_to_base))

print(f"\nCalibration data saved to {output_file}_*.npy")
print(f"Transformation matrix also saved as text to {output_file}_transform.txt")
