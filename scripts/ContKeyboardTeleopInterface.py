from URnterface import URInterface
from time import sleep
from ContinuousControlKeyboard import ContinuousControlKeyboard
from RSCameraInterface import RSCameraInterface
import threading
import numpy as np

class ContKeyboardTeleopInterface:
    def __init__(self, right_arm_ip='192.168.2.2', left_arm_ip='192.168.1.2', reset_arms=True,
                 right_arm_start_joint__positions=None, left_arm_start_joint__positions=None,
                 right_arm_has_gripper=True, left_arm_has_gripper=True):
        
        self.right_arm_start_joint__positions = right_arm_start_joint__positions
        if right_arm_start_joint__positions == None:
            self.right_arm_start_joint__positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                                -1.095669650507004, -4.386985456001609, 3.2958897411425156])
        self.left_arm_start_joint__positions = left_arm_start_joint__positions
        if left_arm_start_joint__positions == None:
            self.left_arm_start_joint__positions = tuple([0.04474830963143529, -1.6422924423175793, 1.9634950313912025,
                                                4.267360912521422, -1.4365121397580038, 2.3399834772053114])
            # self.left_arm_start_joint__positions = tuple([0.1001404325810099, -1.9640431421070108, 2.192831819213297,
            #                                             4.154566166681737, -1.5883319440702799, 2.385492181115367])
        self.right_arm = URInterface(right_arm_ip, self.right_arm_start_joint__positions, has_robotiq_gripper=right_arm_has_gripper)
        self.left_arm = URInterface(left_arm_ip, self.left_arm_start_joint__positions, has_robotiq_gripper=left_arm_has_gripper,
                            robotiq_gripper_port='/dev/ttyUSB1')
        self.right_arm_pose = self.right_arm.getPose()
        self.left_arm_pose = self.left_arm.getPose()
        self.left_gripper = self.left_arm.getGripper()
        self.right_gripper = self.right_arm.getGripper()
        
        self.keyboard = ContinuousControlKeyboard()
        self.is_running = False
        self.resetting = False
        self.last_observation = {}
        self.last_action = {}

        self.rs_camera = RSCameraInterface()
        self.rs_camera.startCapture()

        if reset_arms:
            self.resetArms()

        print("Initialized ContKeyboardTeleopInterface")

    def resetArms(self):
        print("ContKeyboardTeleopInterface: Resetting Arms")
        self.resetting = True
        self.right_arm.resetPosition()
        self.left_arm.resetPosition()
        # Send current pose to arms so that it won't jump when the programs are started
        self.right_arm_pose = self.right_arm.getPose()
        self.left_arm_pose = self.left_arm.getPose()
        self.left_gripper = self.left_arm.getGripper()
        self.right_gripper = self.right_arm.getGripper()
        for _ in range(10):
            self.right_arm.updateArmPose(self.right_arm_pose)
            self.left_arm.updateArmPose(self.left_arm_pose)
        # Makes sure the keyboard knows that the grippers are open since the
        # arm was reset
        self.keyboard.resetGripperValues()
        self.resetting = False
        print("ContKeyboardTeleopInterface: Finished Resetting Arms")

    def startTeleop(self):
        self.teleop_thread = threading.Thread(target=self.teleopThread)
        self.gripper_thread = threading.Thread(target=self.gripperThread)
        self.teleop_thread.start()
        self.gripper_thread.start()

    def teleopThread(self):
        print("ContKeyboardTeleopInterface: Start UR Programs and Begin Teleoperation")
        self.is_running = True
        while self.is_running:
            if not self.resetting:
                # Get the arm deltas and gripper actions that should be applied
                # based on the key board press
                left_arm_info, right_arm_info = self.keyboard.getArmDeltasAndGrippersFromKeyPress()
                left_arm_delta, left_gripper = left_arm_info
                right_arm_delta, right_gripper = right_arm_info

                # Store the observation and action
                self.last_observation = {
                    'left_arm_j': self.left_arm.getj(),
                    'right_arm_j': self.right_arm.getj(),
                    'left_gripper': self.left_arm.getGripper(),
                    'right_gripper': self.right_arm.getGripper(),
                    'image': self.rs_camera.getCurrentImage()
                }
                self.last_action = {
                    'left_arm_delta': left_arm_delta,
                    'left_gripper': left_gripper,
                    'right_arm_delta': right_arm_delta,
                    'right_gripper': right_gripper,
                }

                # Apply deltas and grippers
                self.left_arm_pose += left_arm_delta
                self.right_arm_pose += right_arm_delta
                self.left_gripper = left_gripper
                self.right_gripper = right_gripper
                self.left_arm.updateArmPose(self.left_arm_pose)
                self.right_arm.updateArmPose(self.right_arm_pose)
                sleep(0.004) # 250hz

    def gripperThread(self):
        while self.is_running:
            if not self.resetting:
                # self.left_arm.moveRobotiqGripper(self.left_gripper)
                # self.right_arm.moveRobotiqGripper(self.right_gripper)
                # sleep(0.004)
                pass

    def stopTeleop(self):
        self.is_running = False

    def getLastObsAndAction(self):
        return self.last_observation, self.last_action