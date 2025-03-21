import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.Robotiq3fGripperInterface import Robotiq3fGripperInterface
from time import sleep

robotiq_gripper = Robotiq3fGripperInterface(port="/dev/ttyUSB0")
sleep(1)

print("Closing")
robotiq_gripper.moveRobotiqGripper(close=False)
sleep(5)
print("opening")
robotiq_gripper.moveRobotiqGripper(close=False)c

