from oculus_reader.oculus_reader.reader import OculusReader
from time import sleep
import urx
import numpy as np
import pymodbus.client as ModbusClient
from pymodbus.framer import Framer
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver

""" Get Cartesian Position of the Oculus """
def getControllerPositionAndTrigger(oculus):
    transformations, buttons = oculus.get_transformations_and_buttons()
    pos = transformations['r'][:, 3]
    return (pos[:3], buttons['RG'], buttons['RTr'])

""" Get the EE delta of the UR from the oculus delta """
def getEEDelta(controller_pos, new_controller_pos):
    ee_delta = controller_pos - new_controller_pos
    ee_delta = np.array([ee_delta[2], ee_delta[0], ee_delta[1]])
    ee_delta *= -1
    return ee_delta

""" Updates the robot position via modbus """
def updateRobotPose(client, position):
    position = np.array(position) * 100
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    for i in range(6): # don't update the ypr for now
        builder.reset()
        builder.add_16bit_int(int(position[i]))
        payload = builder.to_registers()
        client.write_register(128 + i, payload[0])

def deltaMeetsThreshold(delta):
    threshold = 1e-2
    return abs(delta[0]) > threshold or abs(delta[1]) > threshold or abs(delta[2]) > threshold

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
controller_pos, _, _ = getControllerPositionAndTrigger(oculus)
robot_pose = robot.get_pose_array()
print("Initial controller position", controller_pos)
print("Initial robot pose", robot_pose)

prev_right_gripper = False
while True:
    # print(transformations, buttons)
    new_controller_pos, right_gripper, right_trigger = getControllerPositionAndTrigger(oculus)
    if right_gripper:
        if not prev_right_gripper:
            # Update current controller position to be new controller position on new gripper press
            controller_pos = new_controller_pos
        else:
            ee_delta = getEEDelta(controller_pos, new_controller_pos)
            if deltaMeetsThreshold(ee_delta):
                robot_pos = robot_pose[:3]
                new_robot_pos = robot_pos + ee_delta
                new_robot_pose = [new_robot_pos[0], new_robot_pos[1], new_robot_pos[2], robot_pose[3], robot_pose[4], robot_pose[5]]
                updateRobotPose(ur_modbus_client, new_robot_pose)
                print("Moved robot to pose", new_robot_pose)

                controller_pos = new_controller_pos
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