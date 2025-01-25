from pyRobotiqGripper import RobotiqGripper

class Robotiq2f85Interface:
    def __init__(self):
        self.gripper = RobotiqGripper()
        self.gripper.activate()
        self.gripper.calibrate(0, 40)
        self.gripper_closed = False
        print("Robotiq2f85Interface: Initialized Gripper")

    def moveRobotiqGripper(self, close=True):
        if close:
            self.gripper.close()
        else:
            self.gripper.open()
        self.gripper_closed = close

    def getGripperPosition(self):
        return self.gripper.getPosition()
    
    def getGripperStatus(self):
        return self.gripper_closed
    
    """ Reset to start position (open)"""
    def resetPosition(self):
        print("Robotiq2f85Interface: Resetting Robotiq 2F85 Gripper to start position")
        self.moveRobotiqGripper(close=False)