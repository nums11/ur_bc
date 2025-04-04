#!/usr/bin/env python3

import urx
import numpy as np
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def get_robot_pose():
    """Connect to the robot, get its current pose, and print it in different formats."""
    robot_ip = "192.168.1.2"
    
    print(f"Connecting to robot at {robot_ip}...")
    
    try:
        # Connect to the robot
        robot = urx.Robot(robot_ip)
        
        # Wait for connection to stabilize
        time.sleep(0.5)
        
        # Get the current pose
        current_pose = robot.get_pose()
        
        # Print the pose object directly
        print("\nRobot Pose:")
        print(current_pose)
        
        # Extract position (in meters)
        position = current_pose.pos
        print("\nPosition (meters):")
        print(f"X: {position.x:.6f}, Y: {position.y:.6f}, Z: {position.z:.6f}")
        
        # Extract orientation as a rotation vector (axis-angle representation, in radians)
        orientation = current_pose.orient
        rot_vec = orientation.log.array_ref
        print("\nOrientation (axis-angle, radians):")
        print(f"RX: {rot_vec[0]:.6f}, RY: {rot_vec[1]:.6f}, RZ: {rot_vec[2]:.6f}")
        
        # Convert to a 4x4 transformation matrix
        pose_matrix = np.eye(4)
        
        # Set the translation part (position)
        pose_matrix[0:3, 3] = position.array_ref[0:3]
        
        # Convert rotation vector to rotation matrix
        import cv2
        rot_mat, _ = cv2.Rodrigues(np.array(rot_vec))
        pose_matrix[0:3, 0:3] = rot_mat
        
        # Print the transformation matrix
        print("\nTransformation Matrix:")
        np.set_printoptions(precision=6, suppress=True)
        print(pose_matrix)
        
        # Close the connection
        robot.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'robot' in locals():
            robot.close()
        sys.exit(1)

if __name__ == "__main__":
    get_robot_pose() 