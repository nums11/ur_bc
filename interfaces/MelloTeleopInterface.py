from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
import serial
from time import sleep
import threading
import json
import math
import ast
import time  # Add at the top if not already imported
import numpy as np

class MelloTeleopInterface:
    def __init__(self, env, port='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', baudrate=115200):
        # Initialize the interface
        self.env = env
        assert self.env.action_type == "joint_modbus", "MelloTeleopInterface only supports action_type 'joint_modbus'"
        self.resetting = False
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.obs = {}
        self._setup_serial()
        # Initialize previous values
        self.prev_joints = [0] * 6
        self.prev_gripper = 0
        # Admittance parameters
        self.admittance_gain_linear = 0.0003 # Meters per Newton (adjust as needed)
        self.admittance_gain_angular = 0.003 # Radians per Newton-meter (adjust as needed)
        self.current_force_torque = np.zeros(6) # Initialize FT reading storage
        
        # Force filtering parameters
        self.filtered_ft = np.zeros(6)  # Filtered force-torque state
        self.filter_alpha = 0.8  # Filter strength (0-1, lower = more filtering)
        self.force_deadband = 60.0  # Newtons (ignore forces below this threshold)
        self.torque_deadband = 6.0  # Newton-meters (ignore torques below this threshold)
        
        print("Initialized MelloTeleopInterface")

    def _setup_serial(self):
        """Set up serial connection to Mello device."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1  # 1 second timeout
            )
            print(f"Successfully connected to {self.port}")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            raise

    def start(self):
        """Start the teleop interface."""
        print("MelloTeleopInterface: Start")
        print("MelloTeleopInterface: MOVE MELLO TO STARTING POSITION BEFORE STARTING UR PROGRAM")
        self.teleop_thread = threading.Thread(target=self._teleopThread)
        self.teleop_thread.daemon = True  # Make thread daemon so it exits when main program exits
        self.teleop_thread.start()

    def _degrees_to_radians(self, degrees):
        """Convert a list of angles from degrees to radians."""
        return [math.radians(deg) for deg in degrees]

    def _read_serial_values(self):
        """Read and parse serial values from Mello device."""
        try:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()
                try:
                    # Parse the Python literal data
                    data = ast.literal_eval(line)
                    
                    # Get joint positions array
                    joint_positions = data.get('joint_positions:', [0]*7)
                    
                    # Take first 6 values for joints (in degrees) and convert to radians
                    joints_deg = joint_positions[:6]
                    # Negate joint 2 (index 2)
                    joints_deg[2] = -1 * joints_deg[2]
                    joints_rad = self._degrees_to_radians(joints_deg)
                    
                    # Get gripper value (last value, between -4096 and 4096)
                    gripper_value = joint_positions[-1] if len(joint_positions) > 6 else self.prev_gripper
                    
                    # Update previous joint values if current values are not zeros
                    if not all(j == 0 for j in joints_rad):
                        self.prev_joints = joints_rad
                    # Update previous gripper value
                    self.prev_gripper = gripper_value
                    
                    return self.prev_joints, self.prev_gripper
                    
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing serial data: {line}")
                    print(f"Parse error: {str(e)}")
                    return self.prev_joints, self.prev_gripper
            return self.prev_joints, self.prev_gripper
        except Exception as e:
            print(f"Error reading serial data: {e}")
            return self.prev_joints, self.prev_gripper

    def _teleopThread(self):
        """Main teleop thread that reads serial data and controls robot."""
        self.reset()
        print("MelloTeleopInterface: Listening to Mello")
        while True:
            if not self.resetting:
                # Get the servo readings from Mello
                joints, gripper = self._read_serial_values()
                
                # Bug where sometimes Mello doesn't send all 6 joint values
                if len(joints) < 6:
                    continue

                # Compute kinematic pose from Mello joints
                ee_pose_kinematic = self.forward_kinematics(joints)
                # print(f"Kinematic EE Pose: {ee_pose_kinematic[:3]}") # Optional: Print only position for clarity

                # Get latest force-torque reading (assuming key 'force')
                # Ensure self.obs is populated and has the key
                if isinstance(self.obs, dict):
                    # Get raw force-torque data
                    raw_ft = self.obs.get('force', np.zeros(6))
                    
                    # Apply low-pass filter
                    self.filtered_ft = self.filter_alpha * raw_ft + (1 - self.filter_alpha) * self.filtered_ft
                    
                    # Apply deadband
                    filtered_deadbanded_ft = np.copy(self.filtered_ft)
                    # For forces (first 3 elements)
                    for i in range(3):
                        if abs(filtered_deadbanded_ft[i]) < self.force_deadband:
                            filtered_deadbanded_ft[i] = 0.0
                    # For torques (last 3 elements)
                    for i in range(3, 6):
                        if abs(filtered_deadbanded_ft[i]) < self.torque_deadband:
                            filtered_deadbanded_ft[i] = 0.0
                    
                    # Update current_force_torque with filtered and deadbanded values
                    self.current_force_torque = filtered_deadbanded_ft
                else:
                    # Handle cases where obs might not be a dict initially or after reset
                    self.current_force_torque = np.zeros(6)
                    self.filtered_ft = np.zeros(6)  # Reset filter state
                
                # Calculate admittance adjustment
                delta_pos = self.admittance_gain_linear * self.current_force_torque[:3]
                delta_rot = self.admittance_gain_angular * self.current_force_torque[3:]
                delta_pose = np.concatenate((delta_pos, delta_rot))

                # Calculate commanded pose
                ee_pose_commanded = ee_pose_kinematic + delta_pose


                if ee_pose_commanded[1] > 0.2:
                    ee_pose_commanded[1] = 0.2
                elif ee_pose_commanded[1] < -0.3:
                    ee_pose_commanded[1] = -0.3

                if ee_pose_commanded[2] > 0.4:
                    ee_pose_commanded[2] = 0.4
                elif ee_pose_commanded[2] < 0.17:
                    ee_pose_commanded[2] = 0.17

                print(f"Commanded EE Pose: {ee_pose_commanded}")
                print(f"Force: {self.current_force_torque}") # Optional: Print force
                print(f"Delta Pos: {delta_pos}") # Optional: Print delta position
                
                # Construct action with commanded Cartesian pose
                action = self._constructActionBasedOnEnv(ee_pose_commanded, gripper)
                
                # Step the environment
                self.obs = self.env.step(action)
                
                # Optional: Update FT reading immediately after step if needed, 
                # otherwise it's updated at the start of the next loop.
                # if isinstance(self.obs, dict):
                #     self.current_force_torque = self.obs.get('ft_reading', np.zeros(6))

                sleep(1/200)

    def reset(self):
        """Reset the environment."""
        self.resetting = True
        self.obs = self.env.reset()
        self.resetting = False
        self.current_force_torque = np.zeros(6) # Reset FT reading
        self.filtered_ft = np.zeros(6) # Reset filter state

    def _constructActionBasedOnEnv(self, joints, gripper):
        """Construct action based on environment type."""
        action = None
        if type(self.env) == BimanualUREnv:
            raise ValueError("BimanualUREnv not supported")
        elif type(self.env) == UREnv:
            gripper_bool = self._clipGripper(gripper)
            action = {
                'arm_j': joints,
                'gripper': gripper_bool
            }
        return action
    
    def _clipGripper(self, gripper_value):
        """
        Convert gripper value to boolean.
        Args:
            gripper_value: Value between -4096 and 4096
        Returns:
            True if gripper should be closed (value < 0), False otherwise
        """
        return gripper_value < 0
    
    def getObservation(self):
        """Get the latest observation."""
        return self.obs
    
    def cleanup(self):
        """Clean up resources."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial port closed")

    def forward_kinematics(self, joint_angles):
        """
        Compute the forward kinematics for a UR5 CB2 robot.
        Args:
            joint_angles: List of 6 joint angles in radians.
        Returns:
            A numpy array representing the Cartesian pose [x, y, z, rx, ry, rz],
            where (x, y, z) is the position and (rx, ry, rz) is the rotation vector (axis-angle).
        """
        # DH Parameters for UR5
        dh_params = [
            (0, 0.089159, math.pi/2),
            (-0.425, 0, 0),
            (-0.39225, 0, 0),
            (0, 0.10915, math.pi/2),
            (0, 0.09465, -math.pi/2),
            (0, 0.0823, 0)
        ]

        # Initialize transformation matrix
        T = np.eye(4)

        for i, (a, d, alpha) in enumerate(dh_params):
            theta = joint_angles[i]
            # Compute transformation matrix for each joint
            T_i = np.array([
                [math.cos(theta), -math.sin(theta) * math.cos(alpha), math.sin(theta) * math.sin(alpha), a * math.cos(theta)],
                [math.sin(theta), math.cos(theta) * math.cos(alpha), -math.cos(theta) * math.sin(alpha), a * math.sin(theta)],
                [0, math.sin(alpha), math.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            # Multiply the current transformation matrix
            T = np.dot(T, T_i)

        # Extract the position of the end effector
        x, y, z = T[0, 3], T[1, 3], T[2, 3]

        # Extract the rotation matrix
        R = T[:3, :3]

        # Convert rotation matrix to rotation vector (axis-angle) rx, ry, rz
        trace = np.trace(R)
        # Clamp trace to avoid numerical errors with arccos
        trace = max(-1.0, min(3.0, trace))
        theta = math.acos((trace - 1.0) / 2.0)

        if abs(theta) < 1e-9:  # Threshold for ~0 rotation
            # Identity rotation, rotation vector is zero
            rx, ry, rz = 0.0, 0.0, 0.0
        elif abs(theta - math.pi) < 1e-9: # Threshold for ~180 degree rotation
            # Rotation by pi radians, singularity in standard formula
            # We need to find the axis u from R = 2u*u^T - I
            # ux^2 = (R[0,0]+1)/2, uy^2 = (R[1,1]+1)/2, uz^2 = (R[2,2]+1)/2
            # Need to determine signs based on off-diagonal elements
            
            # Find axis u for theta = pi
            if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
                ux = math.sqrt((R[0, 0] + 1) / 2)
                uy = R[0, 1] / (2 * ux) if abs(ux) > 1e-6 else math.sqrt((R[1, 1] + 1) / 2) * np.sign(R[0,1]) if abs(R[0,1]) > 1e-6 else 0.0
                uz = R[0, 2] / (2 * ux) if abs(ux) > 1e-6 else math.sqrt((R[2, 2] + 1) / 2) * np.sign(R[0,2]) if abs(R[0,2]) > 1e-6 else 0.0

            elif R[1, 1] >= R[0, 0] and R[1, 1] >= R[2, 2]:
                uy = math.sqrt((R[1, 1] + 1) / 2)
                ux = R[0, 1] / (2 * uy) if abs(uy) > 1e-6 else math.sqrt((R[0, 0] + 1) / 2) * np.sign(R[0,1]) if abs(R[0,1]) > 1e-6 else 0.0
                uz = R[1, 2] / (2 * uy) if abs(uy) > 1e-6 else math.sqrt((R[2, 2] + 1) / 2) * np.sign(R[1,2]) if abs(R[1,2]) > 1e-6 else 0.0

            else: # R[2,2] is largest diagonal
                uz = math.sqrt((R[2, 2] + 1) / 2)
                ux = R[0, 2] / (2 * uz) if abs(uz) > 1e-6 else math.sqrt((R[0, 0] + 1) / 2) * np.sign(R[0,2]) if abs(R[0,2]) > 1e-6 else 0.0
                uy = R[1, 2] / (2 * uz) if abs(uz) > 1e-6 else math.sqrt((R[1, 1] + 1) / 2) * np.sign(R[1,2]) if abs(R[1,2]) > 1e-6 else 0.0

            # Normalize the axis vector in case of slight numerical inaccuracies
            u_norm = math.sqrt(ux*ux + uy*uy + uz*uz)
            if u_norm > 1e-6:
                ux /= u_norm
                uy /= u_norm
                uz /= u_norm

            rx, ry, rz = theta * ux, theta * uy, theta * uz
        else:
            # General case: 0 < theta < pi
            # Use the formula r = theta * u = theta / (2 * sin(theta)) * [R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1)]
            # The magnitude of the skew-symmetric vector part is 2 * sin(theta)
            vx = R[2, 1] - R[1, 2]
            vy = R[0, 2] - R[2, 0]
            vz = R[1, 0] - R[0, 1]
            v_norm = math.sqrt(vx*vx + vy*vy + vz*vz) # This is 2 * sin(theta)
            
            # Avoid division by zero if sin(theta) is very small (should be covered by theta checks, but added for safety)
            if v_norm < 1e-9:
                 rx, ry, rz = 0.0, 0.0, 0.0 # Should not happen given prior checks
            else:
                scale = theta / v_norm
                rx = vx * scale
                ry = vy * scale
                rz = vz * scale

        # Return pose as [x, y, z, rx, ry, rz]
        return np.array([x, y, z, rx, ry, rz]) 