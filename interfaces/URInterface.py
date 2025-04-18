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
    def __init__(self, ip, use_current_joint_positions, start_joint_positions, has_3f_gripper=False, robotiq_gripper_port='/dev/ttyUSB0'):
        # Initialize member variables
        self.ip = ip
        self.has_3f_gripper = has_3f_gripper
        self.robotiq_gripper_port = robotiq_gripper_port

        # Initialize URX connection
        self.arm = urx.Robot(self.ip, use_rt=True)
        if use_current_joint_positions:
            self.start_joint_positions = tuple(self.arm.getj())
        else:
            self.start_joint_positions = start_joint_positions
            if start_joint_positions == None:
                self.start_joint_positions = tuple([0.04474830963143529, -1.6422924423175793, 1.9634950313912025,
                                                4.267360912521422, -1.4365121397580038, 2.3399834772053114])

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

        self.previous_values = None  # Initialize previous_values to None

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
        return np.array(self.convertURXPoseToArray(self.arm.get_pose()))

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

    # Interpolation is not working, so we are not using it
    # def sendModbusValues(self, values):
    #     # Values will be divided by 100 in URScript
    #     values = np.array(values) * 100

    #     # Initialize previous_values if it's the first call
    #     if self.previous_values is None:
    #         self.previous_values = values

    #     # Interpolation setup
    #     original_freq = 50  # Original frequency in Hz
    #     target_freq = 125  # Target frequency in Hz
    #     # interpolation_steps = int(target_freq / original_freq)
    #     interpolation_steps = 100

    #     # Perform linear interpolation for each value
    #     # Create an array of steps from 0 to interpolation_steps-1
    #     steps = np.arange(interpolation_steps)[:, None]
        
    #     # Use broadcasting to create the interpolation for all values at once
    #     interpolated_values = self.previous_values + (values - self.previous_values) * steps / (interpolation_steps - 1)

    #     # print("previous_values", self.previous_values, "values", values, "interpolation_steps", interpolation_steps, "interpolated_values", interpolated_values)
    #     builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    #     print("About to send interpolated values")
    #     for interpolated_value in interpolated_values:
    #         print("Sending interpolated value", interpolated_value)
    #         for i in range(6):
    #             builder.reset()
    #             builder.add_16bit_int(int(interpolated_value[i]))
    #             payload = builder.to_registers()
    #             self.modbus_client.write_register(128 + i, payload[0])
    #         # sleep(1 / target_freq)  # Sleep to maintain the target frequency

    #     # # Update previous_values
    #     self.previous_values = values
    #     print()


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

    def convertURXPoseToArray(self, urx_pose):
        output = np.zeros(6)
        output[:3] = urx_pose.pos.array_ref
        output[3:] = urx_pose.orient.log.array_ref
        return output

    def close(self):
        """Safely close all connections"""
        print("URInterface: Closing connections...")
        try:
            if hasattr(self, 'arm') and self.arm is not None:
                self.arm.close()
                print("URInterface: URX connection closed")
        except Exception as e:
            print(f"URInterface: Error closing URX connection: {e}")
        
        try:
            if hasattr(self, 'modbus_client') and self.modbus_client is not None:
                self.modbus_client.close()
                print("URInterface: Modbus connection closed")
        except Exception as e:
            print(f"URInterface: Error closing Modbus connection: {e}")


    def __del__(self):
        """Cleanup when the object is deleted"""
        self.close() 
        