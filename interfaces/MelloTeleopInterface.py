from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
import serial
from time import sleep
import threading
import json
import math

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
                    # Parse the JSON data
                    data = json.loads(line)
                    
                    # Get joint positions array
                    joint_positions = data.get('joint_positions:', [0]*7)
                    
                    # Take first 6 values for joints (in degrees) and convert to radians
                    joints_deg = joint_positions[:6]
                    joints_rad = self._degrees_to_radians(joints_deg)
                    
                    # Get gripper value (last value, between -4096 and 4096)
                    gripper_value = joint_positions[-1] if len(joint_positions) > 6 else 0
                    
                    return joints_rad, gripper_value
                    
                except json.JSONDecodeError:
                    print(f"Error parsing serial data: {line}")
                    return [0]*6, 0
            return [0]*6, 0
        except Exception as e:
            print(f"Error reading serial data: {e}")
            return [0]*6, 0

    def _teleopThread(self):
        """Main teleop thread that reads serial data and controls robot."""
        self.reset()
        while True:
            if not self.resetting:
                # Get the servo readings from Mello
                joints, gripper = self._read_serial_values()
                print(f"Joints: {joints}, Gripper: {gripper}")
                # Construct action
                action = self._constructActionBasedOnEnv(joints, gripper)
                
                # Step the environment
                self.obs = self.env.step(action)
                sleep(0.004)  # 250hz

    def reset(self):
        """Reset the environment."""
        self.resetting = True
        self.obs = self.env.reset()
        self.resetting = False

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