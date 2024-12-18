import urx
import pymodbus.client as ModbusClient
from pymodbus.framer import Framer
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from RobotiqGripperInterface import RobotiqGripperInterface
import numpy as np

class URInterface:
    def __init__(self, ip, has_robotiq_gripper=False, robotiq_gripper_port='/dev/ttyUSB0'):
        # Initialize member variables
        self.ip = ip
        self.has_robotiq_gripper = has_robotiq_gripper
        self.robotiq_gripper_port = robotiq_gripper_port

        # Initialize URX connection
        self.arm = urx.Robot(self.ip)
        print("URInterface: Initialized URX Connection To IP", self.ip)

        # Initialize Modbus Client
        self.modbus_client = ModbusClient.ModbusTcpClient(
            self.ip,
            port=502,
            framer=Framer.SOCKET
        )
        self.modbus_client.connect()
        print("URInterface: Initialized Modbus Connection To IP", self.ip)

        # Initialize Robotiq 3f Gripper
        if self.has_robotiq_gripper:
            self.robotiq_gripper = RobotiqGripperInterface(port=self.robotiq_gripper_port)

    """ Send a movej command using urx """
    def movej(self, joint_positions):
        self.arm.movej(joint_positions)

    """ Get arm pose using urx """
    def getPose(self):
        return self.arm.get_pose_array()

    """ Updates the robot position via modbus """
    def updateArmPose(self, target_pose):
        # Pose values will be divided by 100 in URScript
        target_pose = np.array(target_pose) * 100
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        # Loop through each pose value and write it to a register
        for i in range(6):
            builder.reset()
            builder.add_16bit_int(int(target_pose[i]))
            payload = builder.to_registers()
            self.modbus_client.write_register(128 + i, payload[0])

    """ Moves gripper to position 0 (open) or 200 (closed) """
    def moveRobotiqGripper(self, close=True):
        self.robotiq_gripper.moveRobotiqGripper(close)

    """ Get observation (joint pos, gripper) """
    def getObservation(self):
        if self.has_robotiq_gripper:
            return (self.arm.getj(), self.robotiq_gripper.getGripperStatus())
        else:
            return self.arm.getj()