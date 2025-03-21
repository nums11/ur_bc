import requests
import time
import math
import json
import numpy as np
import threading
from environments.UREnv import UREnv
from environments.BimanualUREnv import BimanualUREnv

class MelloTeleopInterface:
    def __init__(self, env, mello_url="http://10.19.2.209/", offsets_file="joint_offsets.json"):
        # Initialize the interface
        self.env = env
        assert self.env.action_type == "joint_modbus", "MelloTeleopInterface only supports action_type 'joint_modbus'"
        
        self.mello_url = mello_url
        self.offsets_file = offsets_file
        self.resetting = False
        self.obs = {}
        
        # Load joint offsets
        self.load_offsets()
        
        print(f"Initialized MelloTeleopInterface with Mello endpoint: {self.mello_url}")
    
    def load_offsets(self):
        """Load joint offsets from the JSON file."""
        try:
            with open(self.offsets_file, 'r') as f:
                self.offset_data = json.load(f)
            print(f"Loaded joint offsets from {self.offsets_file}")
            
            # Reference position from the offset file
            self.reference_position_rad = self.offset_data["reference_position_rad"]
            print(f"Reference position (rad): {[f'{angle:.6f}' for angle in self.reference_position_rad]}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading offsets file: {e}")
            print("Please run joint_offset_calculator.py first to generate the offset file.")
            raise

    def fetch_mello_data(self):
        """Fetch joint position data from the Mello endpoint."""
        try:
            response = requests.get(self.mello_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Extract joint positions
            if isinstance(data, dict) and "joint_positions" in data:
                joint_positions = data["joint_positions"]
            elif isinstance(data, dict) and all(f"q{i}" in data for i in range(6)):
                joint_positions = [data[f"q{i}"] for i in range(6)]
            elif isinstance(data, list) and len(data) >= 6:
                joint_positions = data[:6]
            else:
                print(f"Warning: Could not extract joint positions from data: {data}")
                return None
            
            return joint_positions
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Mello: {e}")
            return None
    
    def convert_to_ur_joints(self, mello_joints):
        """
        Convert Mello joint positions to UR robot joint positions.
        
        Args:
            mello_joints: List of 6 joint values from Mello (in degrees)
            
        Returns:
            List of 6 joint values for the UR robot (in radians)
        """
        if not mello_joints or len(mello_joints) != 6:
            return None
        
        # 1. Convert values from degrees to radians (always assuming degrees)
        normalized = [math.radians(pos) for pos in mello_joints]
        
        # 2. Calculate UR joint positions based on normalized positions and offsets
        ur_joints = []
        
        # Get the normalized reference positions from the configuration we measured
        reference_normalized = self.offset_data["normalized_position_rad"]
        
        # Calculate the delta between current and reference normalized positions
        deltas = [current - ref for current, ref in zip(normalized, reference_normalized)]
        
        # Apply these deltas to the UR reference positions
        for i in range(6):
            ur_joint = self.reference_position_rad[i] + deltas[i]
            ur_joints.append(ur_joint)
        
        # Debug output
        print("\nMello joints (deg):", [f"{j:.2f}" for j in mello_joints])
        print("Mello joints (rad):", [f"{j:.4f}" for j in normalized])
        print("Reference normalized (rad):", [f"{j:.4f}" for j in reference_normalized])
        print("Delta (rad):", [f"{d:.4f}" for d in deltas])
        print("UR joints (rad):", [f"{j:.4f}" for j in ur_joints])
        print("UR joints (deg):", [f"{math.degrees(j):.2f}" for j in ur_joints])
        
        return ur_joints
    
    def start(self):
        """Start the teleoperation thread."""
        print("MelloTeleopInterface: Start")
        print("MelloTeleopInterface: Starting teleoperation from Mello to UR robot")
        self.teleop_thread = threading.Thread(target=self._teleopThread)
        self.teleop_thread.daemon = True
        self.teleop_thread.start()

    def _teleopThread(self):
        """Thread function for continuous teleoperation."""
        self.reset()
        while True:
            if not self.resetting:
                # Get joint positions from Mello
                mello_joints = self.fetch_mello_data()
                
                if mello_joints:
                    # Convert to UR joint positions
                    ur_joints = self.convert_to_ur_joints(mello_joints)
                    
                    if ur_joints:
                        # Construct action
                        action = self._constructActionBasedOnEnv(ur_joints)
                        
                        # Step the environment
                        self.obs = self.env.step(action)
                
                # Sleep to control the update rate (e.g., 250Hz)
                time.sleep(0.004)

    def reset(self):
        """Reset the environment."""
        self.resetting = True
        self.obs = self.env.reset()
        self.resetting = False
        print("MelloTeleopInterface: Environment reset")

    def _constructActionBasedOnEnv(self, ur_joints):
        """Construct the action dictionary based on environment type."""
        action = None
        if isinstance(self.env, BimanualUREnv):
            raise ValueError("BimanualUREnv not supported yet")
        elif isinstance(self.env, UREnv):
            # Default gripper state (closed)
            gripper = False
            
            # Construct Action
            action = {
                'arm_j': ur_joints,
                'gripper': gripper
            }
        return action
    
    def getObservation(self):
        """Get the current observation from the environment."""
        return self.obs 