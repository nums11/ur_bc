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

# def get_roll_pitch_yaw_from_matrix(matrix):
#     """
#     Extract roll, pitch, and yaw from a 3x3 or 4x4 transformation matrix using the ZYX convention.

#     Parameters:
#         matrix (numpy.ndarray): A 3x3 or 4x4 transformation matrix.

#     Returns:
#         tuple: (roll, pitch, yaw) in radians.
#     """
#     if matrix.shape == (4, 4):
#         # Extract the rotation part from the 4x4 matrix
#         matrix = matrix[:3, :3]

#     if matrix.shape != (3, 3):
#         raise ValueError("Input must be a 3x3 or 4x4 transformation matrix.")

#     # Check for gimbal lock (singularities)
#     if np.isclose(matrix[2, 0], 1.0):
#         # Gimbal lock, positive singularity
#         pitch = np.pi / 2
#         roll = 0
#         yaw = np.arctan2(matrix[0, 1], matrix[0, 2])
#     elif np.isclose(matrix[2, 0], -1.0):
#         # Gimbal lock, negative singularity
#         pitch = -np.pi / 2
#         roll = 0
#         yaw = np.arctan2(-matrix[0, 1], -matrix[0, 2])
#     else:
#         # General case
#         pitch = np.arcsin(-matrix[2, 0])
#         roll = np.arctan2(matrix[2, 1], matrix[2, 2])
#         yaw = np.arctan2(matrix[1, 0], matrix[0, 0])

#     return roll, pitch, yaw

def get_roll_pitch_yaw_from_matrix(matrix):
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

def get6dPoseFromMatrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")
    # Extract translation components
    x, y, z = matrix[:3, 3]
    # Extract rotation components and compute roll, pitch, yaw
    roll, pitch, yaw = get_roll_pitch_yaw_from_matrix(matrix)
    return np.array([x, y, z, roll, pitch, yaw])

""" Get Oculus pose with RTr and RG gripper values """
def getControllerPoseAndTrigger(oculus):
    transformations, buttons = oculus.get_transformations_and_buttons()
    pose = get6dPoseFromMatrix(transformations['r'])
    return pose, buttons['RG'], buttons['RTr']

""" Get the EE delta of the UR from the oculus delta """
def getEEDelta(controller_pose, new_controller_pose):
    ee_delta = new_controller_pose - controller_pose
    # Move controller axes to correspond with the UR
    ee_delta = np.array([ee_delta[2], ee_delta[0], ee_delta[1],
                        ee_delta[5], ee_delta[4], -1 * ee_delta[3]])
    return ee_delta

""" Updates the robot position via modbus """
def updateRobotPose(client, target_pose, wrist_position):
    target_pose = np.array(target_pose) * 100
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    for i in range(6):
        builder.reset()
        builder.add_16bit_int(int(target_pose[i]))
        payload = builder.to_registers()
        client.write_register(128 + i, payload[0])
    wrist_position *= 100
    builder.add_16bit_int(int(wrist_position))
    payload = builder.to_registers()
    client.write_register(134, payload[0])

def updateRobotJoints(client, joint_positions):
    joint_positions = np.array(joint_positions) * 100
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    for i in range(6):
        builder.reset()
        builder.add_16bit_int(int(joint_positions[i]))
        payload = builder.to_registers()
        client.write_register(128 + i, payload[0])

""" Returns True if a position delta is greater than some threshold across any axis """
def deltaMeetsThreshold(delta):
    threshold = 1e-2
    for delta_axis in delta:
        if abs(delta_axis) > threshold:
            return True
    return False

""" Ensures the delta across any axis is not larger than a fixed value"""
def restrictDelta(delta):
    delta_restricted = False
    for axis, delta_axis in enumerate(delta):
        if delta_axis > 0.5:
            print("----------- RESTRICTED DELTA ---------------------")
            print("Delta  on axis", axis, "was", delta_axis, "setting to 0.05")
            delta[axis] = 0.05
            delta_restricted = True
    if delta_restricted:
        print("delta after", delta)
    return delta

def decreaseRotationSensitivies(delta):
    for axis in range(3,6):
        rotation_value = delta[axis]
        less_sensitive_rotation = rotation_value / 2
        if less_sensitive_rotation < 1e-2:
            delta[axis] = 0
        else:
            delta[axis] = less_sensitive_rotation
    return delta

""" Moves gripper to position 0 (open) or 200 (closed) """
def moveGripper(gripper, close=True):
    gripper_speed = 4
    gripper_force = 1
    gripper_pos = 0
    if close:
        gripper_pos = 200
    gripper.move(pos=gripper_pos, speed=gripper_speed, force=gripper_force)


""" Initialize Oculus """
oculus = OculusReader()
print("Initialized Oculus -------")

""" Initialize UR """
robot_ip = "192.168.2.2"
robot = urx.Robot(robot_ip)
print("Initialized URX Robot -----")
ur_modbus_client = ModbusClient.ModbusTcpClient(
    robot_ip,
    port=502,
    framer=Framer.SOCKET
)
ur_modbus_client.connect()
print("Initialized Modbus Client")
print("Setting to start position ----")
start_joint__positions = tuple([-0.10555254050793472, -2.176040565310071, -2.020456103497305,
                                0.4152529015018597, -3.4580182915458724, 4.000654384219917])
robot.movej(start_joint__positions)
# updateRobotPose(ur_modbus_client, start_joint__positions)
print("Robot reset to start position ------")
sleep(1)

"""Initialize Gripper"""
gripper_port = "/dev/ttyUSB0"
gripper = RobotiqModbusRtuDriver(gripper_port)
gripper.connect()
gripper.activate()
status = gripper.status()
print("Gripper status", status)

""" Get Initial Controller Position and Robot Pose """
controller_pose, _, _ = getControllerPoseAndTrigger(oculus)
robot_pose = robot.get_pose_array()
print("Initial controller pose", controller_pose)
print("Initial robot pose", robot_pose, len(robot_pose))
joint_positions = robot.getj()
print("Joint positions", joint_positions)

right_gripper_pressed_before = False
while not right_gripper_pressed_before:
    new_controller_pose, right_gripper, right_trigger = getControllerPoseAndTrigger(oculus)
    print("Never pressed right gripper ---")
    updateRobotPose(ur_modbus_client, robot_pose, joint_positions[5])
    if right_gripper:
        right_gripper_pressed_before = True
        print("Breaking")
        break

prev_right_gripper = False
while True:
    new_controller_pose, right_gripper, right_trigger = getControllerPoseAndTrigger(oculus)
    # Only handle movements when right gripper is pressed
    if right_gripper:
        if not prev_right_gripper:
            # Update current controller position to be new controller position on new gripper press
            controller_pose = new_controller_pose
        else:
            ee_delta = getEEDelta(controller_pose, new_controller_pose)
            if deltaMeetsThreshold(ee_delta):
                ee_delta = restrictDelta(ee_delta)
                # ee_delta = decreaseRotationSensitivies(ee_delta)
                # Keep the pitch the same and handle wrist rotation separately later
                wrist_delta = ee_delta[4]
                ee_delta[4] = 0
                new_robot_pose = robot_pose + ee_delta
                print("About to update robot pose. Wrist delta", wrist_delta, "ee_delta", ee_delta[4])
                # updateRobotPose(ur_modbus_client, new_robot_pose, wrist_delta)
                # joint_positions[5] += wrist_delta
                updateRobotPose(ur_modbus_client, new_robot_pose, joint_positions[5])
                # updateRobotJoints(ur_modbus_client, joint_positions)
                # Decrease sensitivity of wrist delta

                print("Moved robot to pose", new_robot_pose, "Wrist delta", wrist_delta)

                controller_pose = new_controller_pose
                robot_pose = new_robot_pose
            else:
                print("delta did not meet threshold", ee_delta)

        if right_trigger:
            moveGripper(gripper, close=True)
        else:
            moveGripper(gripper, close=False)

        prev_right_gripper = True
    else:
        prev_right_gripper = False

    sleep(0.005)