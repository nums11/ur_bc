#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import sys
import glob
from pathlib import Path
import argparse

def load_latest_calibration(data_dir="calibration_data"):
    """Load the most recent calibration data."""
    # Find all transform files
    transform_files = glob.glob(os.path.join(data_dir, "*_transform.npy"))
    
    if not transform_files:
        print(f"No calibration data found in {data_dir}")
        return None, None, None
    
    # Sort by modification time (newest first)
    transform_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = transform_files[0]
    
    # Get the base filename without extension
    base_name = os.path.basename(latest_file).replace("_transform.npy", "")
    
    print(f"Found latest transform file: {latest_file}")
    print(f"Base filename: {base_name}")
    
    # Load data
    transform = np.load(latest_file)
    camera_points_file = os.path.join(data_dir, f"{base_name}_camera_points.npy")
    robot_points_file = os.path.join(data_dir, f"{base_name}_robot_points.npy")
    
    print(f"Looking for camera points at: {camera_points_file}")
    print(f"Looking for robot points at: {robot_points_file}")
    
    # Check if files exist
    if not os.path.exists(camera_points_file) or not os.path.exists(robot_points_file):
        print("Error: Could not find matching camera or robot point files")
        print("Available files in directory:")
        for file in os.listdir(data_dir):
            print(f"  {file}")
        return None, None, None
    
    camera_points = np.load(camera_points_file)
    robot_points = np.load(robot_points_file)
    
    print(f"Successfully loaded calibration data")
    return transform, camera_points, robot_points

def create_coordinate_frame(size=0.1):
    """Create a coordinate frame visualization."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def create_sphere(position, radius=0.01, color=None):
    """Create a sphere at the given position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    
    if color is not None:
        sphere.paint_uniform_color(color)
    
    return sphere

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize camera-robot calibration")
    parser.add_argument("--data-dir", default="calibration_data", help="Directory containing calibration data")
    parser.add_argument("--specific-file", help="Specific calibration timestamp to use (format: YYYYMMDD_HHMMSS)")
    args = parser.parse_args()
    
    # Load calibration data
    if args.specific_file:
        timestamp = args.specific_file
        base_name = f"calibration_{timestamp}"
        transform = np.load(os.path.join(args.data_dir, f"{base_name}_transform.npy"))
        camera_points = np.load(os.path.join(args.data_dir, f"{base_name}_camera_points.npy"))
        robot_points = np.load(os.path.join(args.data_dir, f"{base_name}_robot_points.npy"))
    else:
        transform, camera_points, robot_points = load_latest_calibration(args.data_dir)
    
    if transform is None:
        sys.exit(1)
    
    # Print transformation matrix for reference
    print("\nTransformation Matrix (Camera to Robot):")
    print(transform)
    
    # Create visualization
    geometries = []
    
    # Add coordinate frames for camera and robot
    camera_frame = create_coordinate_frame(size=0.05)
    robot_frame = create_coordinate_frame(size=0.05)
    
    # Add the frames
    geometries.append(camera_frame)
    geometries.append(robot_frame)
    
    # Add camera points (red) 
    for point in camera_points:
        sphere = create_sphere(point, radius=0.01, color=[1, 0, 0])
        geometries.append(sphere)
    
    # Add robot points (blue)
    for point in robot_points:
        sphere = create_sphere(point, radius=0.01, color=[0, 0, 1])
        geometries.append(sphere)
    
    # Transform camera points to robot frame using the transformation
    transformed_points = []
    for point in camera_points:
        # Convert to homogeneous coordinates
        point_h = np.append(point, 1.0)
        # Apply transformation
        transformed_point = transform @ point_h
        transformed_points.append(transformed_point[:3])
    
    # Add transformed points (green)
    for point in transformed_points:
        sphere = create_sphere(point, radius=0.01, color=[0, 1, 0])
        geometries.append(sphere)
    
    # Calculate and display error
    errors = []
    for i in range(len(robot_points)):
        error = np.linalg.norm(transformed_points[i] - robot_points[i])
        errors.append(error)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\nCalibration Errors:")
    print(f"Mean Error: {mean_error:.6f} meters ({mean_error*1000:.2f} mm)")
    print(f"Max Error: {max_error:.6f} meters ({max_error*1000:.2f} mm)")
    print(f"Individual Errors (meters):")
    for i, error in enumerate(errors):
        print(f"  Point {i+1}: {error:.6f} ({error*1000:.2f} mm)")
    
    # Add lines connecting original and transformed points
    for i in range(len(robot_points)):
        # Line from robot point to transformed camera point
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([robot_points[i], transformed_points[i]])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow
        geometries.append(line)
    
    # Visualize
    print("\nVisualization Legend:")
    print("  Red points: Camera coordinate frame points")
    print("  Blue points: Robot coordinate frame points")
    print("  Green points: Camera points transformed to robot frame")
    print("  Yellow lines: Error between transformed camera points and robot points")
    print("\nClose visualization window to exit.")
    
    o3d.visualization.draw_geometries(geometries,
                                     window_name="Camera-Robot Calibration Visualization",
                                     width=1024, height=768)

if __name__ == "__main__":
    main() 