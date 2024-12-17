from oculus_reader.oculus_reader.reader import OculusReader
from time import sleep
import urx
import numpy as np
import pymodbus.client as ModbusClient
from pymodbus.framer import Framer
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver
import m3d
import threading

class OculusTeleopInterface:
    def __init__(self, right_arm_ip='192.168.2.2', left_arm_ip='192.168.1.2', reset_arms=False,
                    right_arm_start_joint__positions=None, left_arm_start_joint__positions=None,
                    robotiq_gripper_port='/dev/ttyUSB0'):
        # Initialize member variables
        self.reset_arms = reset_arms
        self.right_arm_ip = right_arm_ip
        self.left_arm_ip = left_arm_ip
        self.robotiq_gripper_port = robotiq_gripper_port
        if right_arm_start_joint__positions == None:
            self.right_arm_start_joint__positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                        -1.095669650507004, -4.386985456001609, 3.2958897411425156])
        else:
            self.right_arm_start_joint__positions = right_arm_start_joint__positions
        if left_arm_start_joint__positions == None:
            self.left_arm_start_joint__positions = tuple([0.10010272221997439, -1.313795512335239, 2.1921907366841067,
                                            3.7562696849438524, 1.2427944188620925, 0.8873570727182682])
        else:
            self.left_arm_start_joint__positions = left_arm_start_joint__positions

        # Initialize Oculus
        self.oculus = OculusReader()
        print("OculusTeleopInterface: Initialized Oculus")

        # Initialize UR Arms
        self.right_arm = urx.Robot(self.right_arm_ip)
        self.left_arm = urx.Robot(self.left_arm_ip)
        print("OculusTeleopInterface: Initialized URX Connections")

        # Initialize Modbus Clients
        self.right_arm_modbus_client = ModbusClient.ModbusTcpClient(
            right_arm_ip,
            port=502,
            framer=Framer.SOCKET
        )
        self.left_arm_modbus_client = ModbusClient.ModbusTcpClient(
            left_arm_ip,
            port=502,
            framer=Framer.SOCKET
        )
        self.right_arm_modbus_client.connect()
        self.left_arm_modbus_client.connect()
        print("OculusTeleopInterface: Initialized Modbus Clients")

        # Initialize Robotiq 3f Gripper
        self.robotiq_gripper = RobotiqModbusRtuDriver(self.robotiq_gripper_port)
        self.robotiq_gripper.connect()
        self.robotiq_gripper.activate()
        print("OculusTeleopInterface: Initialized Robotiq 3f Gripper")

        # Optionally reset arms to a predefined start position
        if self.reset_arms:
            print("OculusTeleopInterface: Setting right arm to start position")
            self.right_arm.movej(self.right_arm_start_joint__positions)
            print("OculusTeleopInterface: Finished setting right arm to start position")
            print("OculusTeleopInterface: Setting left arm to start position")
            self.left_arm.movej(self.left_arm_start_joint__positions)
            print("OculusTeleopInterface: Finished setting left arm to start position")
                    
        # Start arm control threads
        right_arm_thread = threading.Thread(target=self.arm_control_thread,
                                            args=(self.right_arm, self.robotiq_gripper,
                                                  self.right_arm_modbus_client, True))
        left_arm_thread = threading.Thread(target=self.arm_control_thread,
                                        args=(self.left_arm, None, self.left_arm_modbus_client, False))
        right_arm_thread.start()
        print("OculusTeleopInterface: Begin Right arm Teleop")
        left_arm_thread.start()
        print("OculusTeleopInterface: Begin Left arm Teleop")

        """ Arm Control """
    def arm_control_thread(self, arm, gripper, modbus_client, is_right_arm):
        global oculus
        # Get Initial Controller Position and Robot Pose
        controller_pose, _, _ = self.getControllerPoseAndTrigger(is_right_arm)
        robot_pose = arm.get_pose_array()
        joint_positions = arm.getj()

        # Until the gripper is pressed, constanly send the current robot pose so that
        # the UR program will start with this pose and not jump to pose values previously stored in its registers
        gripper_pressed_before = False
        while not gripper_pressed_before:
            new_controller_pose, gripper_pressed, _ = self.getControllerPoseAndTrigger(is_right_arm)
            self.updateRobotPose(modbus_client, robot_pose, joint_positions[4])
            if gripper_pressed:
                gripper_pressed_before = True
                break

        # Update Robot arm and gripper based on controller
        prev_gripper = False
        while True:
            new_controller_pose, gripper_pressed, trigger_pressed = self.getControllerPoseAndTrigger(is_right_arm)
            # Only handle movements when right controller gripper is pressed
            if gripper_pressed:
                if not prev_gripper:
                    # Update current controller position to be new controller position on new gripper press
                    controller_pose = new_controller_pose
                else:
                    # Get the delta in controller position and rotation
                    ee_delta = self.getEEDelta(controller_pose, new_controller_pose, is_right_arm)
                    # Only update robot position if change in controller position meets a 
                    # certain threshold. This prevents robot from changing position when
                    # the control is almost still
                    if self.deltaMeetsThreshold(ee_delta):
                        # Restrict deltas to a maximum value to avoid large jumps in robot position
                        ee_delta = self.restrictDelta(ee_delta)
                        new_robot_pose = robot_pose + ee_delta
                        self.updateRobotPose(modbus_client, new_robot_pose, 0)
                        controller_pose = new_controller_pose
                        robot_pose = new_robot_pose
                    else:
                        pass

                # Robot Gripper open and close (currently only for the right arm)
                if is_right_arm:
                    if trigger_pressed:
                        print("Moving gripper")
                        self.moveGripper(gripper, close=True)
                    else:
                        self.moveGripper(gripper, close=False)
        
                prev_gripper = True
            else:
                prev_gripper = False

            sleep(0.005)

    def get_roll_pitch_yaw_from_matrix(self, matrix):
        # Extract the rotation part from the matrix
        rotation_matrix = matrix[:3, :3]
        # Check for gimbal lock (singularities)
        if np.isclose(rotation_matrix[2, 0], 1.0):
            # Gimbal lock, positive singularity
            pitch = np.pi / 2
            roll = 0
            yaw = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
        elif np.isclose(rotation_matrix[2, 0], -1.0):
            # Gimbal lock, negative singularity
            pitch = -np.pi / 2
            roll = 0
            yaw = np.arctan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
        else:
            # General case
            pitch = np.arcsin(-rotation_matrix[2, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return roll, pitch, yaw

    def get6dPoseFromMatrix(self, matrix):
        if matrix.shape != (4, 4):
            raise ValueError("Input must be a 4x4 transformation matrix.")
        # Extract translation components
        x, y, z = matrix[:3, 3]
        # Extract rotation components and compute roll, pitch, yaw
        roll, pitch, yaw = self.get_roll_pitch_yaw_from_matrix(matrix)
        return np.array([x, y, z, roll, pitch, yaw])

    """ Get Oculus pose with RTr and RG gripper values """
    def getControllerPoseAndTrigger(self, is_right_arm):
        transformations, buttons = self.oculus.get_transformations_and_buttons()
        transformation_key = 'r'
        gripper_key = 'RG'
        trigger_key = 'RTr'
        if not is_right_arm:
            transformation_key = 'l'
            gripper_key = 'LG'
            trigger_key = 'LTr'
        pose = self.get6dPoseFromMatrix(transformations[transformation_key])
        return pose, buttons[gripper_key], buttons[trigger_key]

    """ Get the EE delta of the UR from the oculus delta """
    def getEEDelta(self, controller_pose, new_controller_pose, is_right_arm):
        ee_delta = new_controller_pose - controller_pose
        # Flip some axes for left arm
        if not is_right_arm:
            ee_delta[0] = -1 * ee_delta[0]
            ee_delta[2] = -1 * ee_delta[2]
        # Move controller axes to correspond with the UR
        ee_delta = np.array([ee_delta[2], ee_delta[0], ee_delta[1],
                            ee_delta[5], ee_delta[4], -1 * ee_delta[3]])
        # ee_delta = np.array([0,0,0, 0, 0, 0])
        return ee_delta

    """ Updates the robot position via modbus """
    def updateRobotPose(self, client, target_pose, wrist_position):
        target_pose = np.array(target_pose) * 100
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        for i in range(6):
            builder.reset()
            builder.add_16bit_int(int(target_pose[i]))
            payload = builder.to_registers()
            client.write_register(128 + i, payload[0])
        # print("Writing wrist position", wrist_position)
        wrist_position *= 100
        builder.add_16bit_int(int(wrist_position))
        payload = builder.to_registers()
        client.write_register(134, payload[0])

    """ Returns True if a position delta is greater than some threshold across any axis """
    def deltaMeetsThreshold(self, delta):
        threshold = 1e-2
        for delta_axis in delta:
            if abs(delta_axis) > threshold:
                return True
        return False

    """ Ensures the delta across any axis is not larger than a fixed value"""
    def restrictDelta(self, delta):
        delta_restricted = False
        for axis, delta_axis in enumerate(delta):
            if delta_axis > 0 and delta_axis > 0.5:
                print("----------- RESTRICTED DELTA ---------------------")
                print("Delta  on axis", axis, "was", delta_axis, "setting to 0.05")
                delta[axis] = 0.05
                delta_restricted = True
            elif delta_axis < 0 and delta_axis < -0.5:
                print("----------- RESTRICTED DELTA ---------------------")
                print("Delta  on axis", axis, "was", delta_axis, "setting to -0.05")
                delta[axis] = -0.05
                delta_restricted = True
        if delta_restricted:
            print("delta after", delta)
        return delta

    def decreaseRotationSensitivies(self, delta):
        for axis in range(3,6):
            rotation_value = delta[axis]
            less_sensitive_rotation = rotation_value / 2
            if less_sensitive_rotation < 1e-2:
                delta[axis] = 0
            else:
                delta[axis] = less_sensitive_rotation
        return delta

    """ Moves gripper to position 0 (open) or 200 (closed) """
    def moveGripper(self, gripper, close=True):
        gripper_speed = 4
        gripper_force = 1
        gripper_pos = 0
        if close:
            gripper_pos = 200
            

