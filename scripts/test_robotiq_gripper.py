from RobotiqGripperInterface import RobotiqGripperInterface
from time import sleep

robotiq_gripper = RobotiqGripperInterface(port="/dev/ttyUSB2")
sleep(1)

print("Closing")
robotiq_gripper.moveRobotiqGripper(close=False)
# sleep(5)
# print("opening")
# robotiq_gripper.moveRobotiqGripper(close=False)

