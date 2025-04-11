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

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))

    cv2.imshow("Click End Effector", image)
    cv2.setMouseCallback("Click End Effector", click_event)
    print("Click the end effector in the image. Press any key after clicking.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coords[0] if coords else None

# =====================
# ---- DATA LOOP ----
# =====================
for i in range(NUM_POSES):
    input(f"\nMove robot to pose #{i+1} and press ENTER...")

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        print("Frame capture failed.")
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    u, v = get_click_coordinates(color_image)
    depth = depth_image[v, u] * depth_scale

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
