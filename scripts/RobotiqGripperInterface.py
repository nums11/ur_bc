from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver

class RobotiqGripperInterface:
    def __init__(self, port='/dev/ttyUSB0'):
        # Initialize member variables
        self.port = port

        # Initialize Robotiq 3f gripper
        self.robotiq_gripper = RobotiqModbusRtuDriver(self.port)
        self.robotiq_gripper.connect()
        self.robotiq_gripper.activate()
        print("RobotiqGripperInterface: Initialized Robotiq 3f Gripper at port", self.port)

        # Open the gripper
        self.moveRobotiqGripper(close=False)
        self.gripper_closed = False
        print("RobotiqGripperInterface: Opened Robotiq 3f Gripper")

    """ Moves gripper to position 0 (open) or 200 (closed) """
    def moveRobotiqGripper(self, close=True):
        gripper_speed = 4
        gripper_force = 1
        gripper_pos = 0
        if close:
            gripper_pos = 200
        self.robotiq_gripper.move(pos=gripper_pos, speed=gripper_speed, force=gripper_force)
        self.gripper_closed = close

    """ Returns true if gripper is closed, false otherwise """
    def getGripperStatus(self):
        return self.gripper_closed