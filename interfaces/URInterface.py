import urx
import pymodbus.client as ModbusClient
from pymodbus.framer import Framer
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from .Robotiq3fGripperInterface import Robotiq3fGripperInterface
from .Robotiq2f85Interface import Robotiq2f85Interface
from time import sleep
import numpy as np

class URInterface:
    def __init__(self, ip, start_joint_positions, has_3f_gripper=False, robotiq_gripper_port='/dev/ttyUSB0'):
        # Initialize member variables
        self.ip = ip
        self.start_joint_positions = start_joint_positions
        self.has_3f_gripper = has_3f_gripper
        self.robotiq_gripper_port = robotiq_gripper_port

        # Initialize URX connection
        self.arm = urx.Robot(self.ip, use_rt=True)
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
        if self.has_3f_gripper:
            self.robotiq_gripper = Robotiq3fGripperInterface(port=self.robotiq_gripper_port)
        else:
            self.robotiq_gripper = Robotiq2f85Interface()

    """ Send a movej command using urx """
    def movej(self, joint_positions, blocking=False):
        self.arm.movej(joint_positions, vel=0.5, wait=blocking)

    """ Send a movej(get_inverse_kin) command using urx """
    def movejInvKin(self, joint_positions):
        self.arm.movejInvKin(joint_positions)

    """ Send a servoj command using urx """
    def servoj(self, joint_positions):
        self.arm.servoj(joint_positions)

    """ Get arm pose using urx """
    def getPose(self):
        return np.array(self.arm.get_pose_array())

    def getForce(self):
        return np.array(self.arm.get_force())
    
    def getj(self):
        return np.array(self.arm.getj())

    def getGripper(self):
        return self.robotiq_gripper.getGripperStatus()

    """ Send values to the arm via modbus. Values are either ee pose or joint positions """
    def sendModbusValues(self, values):
        # Values will be divided by 100 in URScript
        values = np.array(values) * 100
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        # Loop through each pose value and write it to a register
        for i in range(6):
            builder.reset()
            builder.add_16bit_int(int(values[i]))
            payload = builder.to_registers()
            self.modbus_client.write_register(128 + i, payload[0])

    """ Moves gripper to position 0 (open) or 200 (closed) """
    def moveRobotiqGripper(self, close=True):
        self.robotiq_gripper.moveRobotiqGripper(close)

    """ Get observation (joint pos, gripper) """
    def getObservation(self):
        if self.has_3f_gripper:
            return (self.arm.getj(), self.robotiq_gripper.getGripperStatus())
        else:
            return self.arm.getj()
        
    """ Reset arm to start joint positions and reset gripper """
    def resetPositionURX(self):
        print("URInterface: Resetting Arm at IP", self.ip, "to start position")
        self.resetGripper()
        self.arm.movej(self.start_joint_positions)
        print("URInterface: Finished Resetting Arm at IP", self.ip, "to start position")
    
    """ Reset arm to either start joint positions or start ee pose """
    def resetPositionModbus(self, current_values, start_values):
        print("URInterface: Resetting Arm at IP", self.ip, "to start position")
        self.resetGripper()
        path = np.linspace(current_values, start_values, num=10)
        # Execute the path
        for joint_positions in path:
            self.sendModbusValues(joint_positions)
            sleep(0.001)
        print("URInterface: Finished Resetting Arm at IP", self.ip, "to start position")

    def resetGripper(self):
        self.robotiq_gripper.resetPosition()
        # Wait until the gripper is open
        while self.robotiq_gripper.getGripperPosition() > 10:
            continue
        print("URInterface: Finished resetting Robotiq Gripper to start position")