#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob
import argparse
from pathlib import Path
import math

def load_latest_calibration(data_dir="calibration_data"):
    """Load the most recent calibration data."""
    # Find all transform files
    transform_files = glob.glob(os.path.join(data_dir, "*_transform.npy"))
    
    if not transform_files:
        print(f"No calibration data found in {data_dir}")
        return None
    
    # Sort by modification time (newest first)
    transform_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = transform_files[0]
    
    # Load data
    transform = np.load(latest_file)
    print(f"Loaded calibration data from {latest_file}")
    return transform

def get_camera_pose(transform):
    """
    Calculate the camera pose (position and orientation) in robot coordinates.
    
    The transform matrix maps from camera to robot frame:
    p_robot = T * p_camera
    
    Returns:
        position: 3D position of camera in robot frame
        rotation_matrix: 3x3 rotation matrix describing camera orientation 
        euler_angles: Roll, Pitch, Yaw angles in degrees
    """
    # Position is the translation part of the transform
    position = transform[:3, 3]
    
    # Orientation is the rotation part of the transform 
    rotation_matrix = transform[:3, :3]
    
    # Calculate Euler angles (in degrees)
    # We use the ZYX convention (same as RPY - Roll, Pitch, Yaw)
    # This assumes the rotation matrix is orthogonal
    euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
    
    return position, rotation_matrix, euler_angles

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Uses the ZYX (roll-pitch-yaw) convention.
    """
    # Check if we're in the gimbal lock case (pitch = ±90°)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])  # Roll
        y = math.atan2(-R[2, 0], sy)      # Pitch
        z = math.atan2(R[1, 0], R[0, 0])  # Yaw
    else:
        # Gimbal lock case
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    # Convert from radians to degrees
    return np.array([
        math.degrees(x),  # Roll
        math.degrees(y),  # Pitch
        math.degrees(z)   # Yaw
    ])

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Get camera position in robot frame")
    parser.add_argument("--data-dir", default="calibration_data", help="Directory containing calibration data")
    parser.add_argument("--specific-file", help="Specific calibration timestamp to use (format: YYYYMMDD_HHMMSS)")
    args = parser.parse_args()
    
    # Load calibration data
    if args.specific_file:
        timestamp = args.specific_file
        base_name = f"calibration_{timestamp}"
        transform_file = os.path.join(args.data_dir, f"{base_name}_transform.npy")
        transform = np.load(transform_file)
    else:
        transform = load_latest_calibration(args.data_dir)
    
    if transform is None:
        sys.exit(1)
    
    # Print transformation matrix for reference
    print("\nTransformation Matrix (Camera to Robot):")
    print(transform)
    
    # Get camera pose
    position, rotation_matrix, euler_angles = get_camera_pose(transform)
    
    # Output results
    print("\n--- Camera Position in Robot Frame ---")
    print(f"X: {position[0]:.6f} meters")
    print(f"Y: {position[1]:.6f} meters")
    print(f"Z: {position[2]:.6f} meters")
    print("\nPosition vector:")
    print(position)
    
    print("\n--- Camera Orientation in Robot Frame ---")
    print("Rotation Matrix:")
    print(rotation_matrix)
    
    print("\nEuler Angles (Roll-Pitch-Yaw, in degrees):")
    print(f"Roll:  {euler_angles[0]:.3f}°")
    print(f"Pitch: {euler_angles[1]:.3f}°")
    print(f"Yaw:   {euler_angles[2]:.3f}°")
    
    # Print a quick explanation of what the angles mean
    print("\nOrientation explanation:")
    print("- Roll:  Rotation around X-axis")
    print("- Pitch: Rotation around Y-axis")
    print("- Yaw:   Rotation around Z-axis")
    print("(Using robot's frame of reference)")

if __name__ == "__main__":
    main() 