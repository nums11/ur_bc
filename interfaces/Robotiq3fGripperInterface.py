from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver

class Robotiq3fGripperInterface:
    def __init__(self, port='/dev/ttyUSB0'):
        # Initialize member variables
        self.port = port
        self.gripper_closed = False

        # Initialize Robotiq 3f gripper
        self.robotiq_gripper = RobotiqModbusRtuDriver(self.port)
        self.robotiq_gripper.connect()
        self.robotiq_gripper.activate()
        print("Robotiq3fGripperInterface: Initialized Robotiq 3f Gripper at port", self.port)

    """ Moves gripper to position 0 (open) or 200 (closed) """
    def moveRobotiqGripper(self, close=True):
        gripper_speed = 4
        gripper_force = 1
        gripper_pos = 0
        if close:
            gripper_pos = 250
        self.robotiq_gripper.move(pos=gripper_pos, speed=gripper_speed, force=gripper_force)
        self.gripper_closed = close

    """ Returns true if gripper is closed, false otherwise """
    def getGripperStatus(self):
        return self.gripper_closed
    
    def getGripperPosition(self):
        return self.robotiq_gripper.status().position.po
    
    """ Reset to start position (open)"""
    def resetPosition(self):
        print("Robotiq3fGripperInterface: Resetting Robotiq 3f Gripper to start position")
        self.moveRobotiqGripper(close=False)
        self.gripper_closed = False