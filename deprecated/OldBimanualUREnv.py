from URnterface import URInterface
from time import sleep
import threading
import numpy as np

class BimanualUREnv():
    def __init__(self, right_arm_ip='192.168.2.2', left_arm_ip='192.168.1.2', reset_arms=False,
                    right_arm_start_joint_positions=None, left_arm_start_joint_positions=None,
                    robotiq_gripper_port='/dev/ttyUSB0'):
        print()
        # Initialize member variables
        self.reset_arms = reset_arms
        self.right_arm_ip = right_arm_ip
        self.left_arm_ip = left_arm_ip
        self.robotiq_gripper_port = robotiq_gripper_port
        self.right_arm_action = None
        self.left_arm_action = None
        self.lock = threading.Lock()
        if right_arm_start_joint_positions == None:
            self.right_arm_start_joint_positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                        -1.095669650507004, -4.386985456001609, 3.2958897411425156])
        else:
            self.right_arm_start_joint_positions = right_arm_start_joint_positions
        if left_arm_start_joint_positions == None:
            self.left_arm_start_joint_positions = tuple([0.1001404325810099, -1.9640431421070108, 2.192831819213297,
                                                          4.154566166681737, -1.5883319440702799, 2.385492181115367])
        else:
            self.left_arm_start_joint_positions = left_arm_start_joint_positions

        # Initialize UR Arms
        self.right_arm = URInterface(
            self.right_arm_ip, self.right_arm_start_joint_positions, has_3f_gripper=True)
        self.left_arm = URInterface(
            self.left_arm_ip, self.left_arm_start_joint_positions, has_3f_gripper=True,
            robotiq_gripper_port="/dev/ttyUSB2")
        print("BimanualUREnv: Initialized UR Interfaces")
        print()

    def step(self, action, convert_oculus_deltas=False, return_obs=False):
        if convert_oculus_deltas:
            action = self.convertOculusDeltas(action)
        with self.lock:
            # self.applied_left_arm_action = False
            # self.applied_right_arm_action = False
            self.left_arm_action, self.right_arm_action = action
        print("Set variables", self.left_arm_action, self.right_arm_action)
        # Wait until both of the actions have been applied in the arm control thread
        if return_obs:
            while not (self.applied_left_arm_action and self.applied_right_arm_action):
                continue
            return self.getObservation()

    """ Get the observation for each arm (joint_pos, gripper) """
    def getObservation(self):
        return [self.left_arm.getObservation(), self.right_arm.getObservation()]

    """ Arm Control """
    def arm_control_thread(self, arm, is_right_arm):
        arm_pose = arm.getPose()
        while not self.resetting:
            # Take an action when the action variable is updated in the step function
            # if self.right_arm_action is not None and self.left_arm_action is not None:
            if (is_right_arm and self.right_arm_action is not None) or \
                (not is_right_arm and self.left_arm_action is not None):
                new_pose = None
                close_gripper = None
                if is_right_arm:
                    arm_action = self.right_arm_action
                    print("Setting right arm action", arm_action)
                else:
                    arm_action = self.left_arm_action
                    print("Setting left arm action", arm_action)

                delta = arm_action[:6]
                close_gripper = arm_action[6]
                new_pose = arm_pose + delta
                
                # print("Updating pose")
                # Take the action
                arm.updateArmPose(new_pose)
                # print("Moving gripper")
                arm.moveRobotiqGripper(close_gripper)
                # Update the arm pose
                arm_pose = new_pose
                # Notify the step function that the action has been applied
                if is_right_arm:
                    with self.lock:
                        # self.applied_right_arm_action = True
                        self.right_arm_action = None
                else:
                    with self.lock:
                        # self.applied_left_arm_action = True
                        self.left_arm_action = None
                # print("End of logic")
        print("Exiting arm control thread")
            
    """ Resets arm movement threads, and optionally resets arm to start position """
    def reset(self):
        print()
        # Kills arm control threads if they are active
        self.resetting = True
        sleep(0.5)
        # Optionally reset arms to start positions
        if self.reset_arms:
            self.resetArms()
        self.resetting = False
        # Initialize Threads and start them
        self.right_arm_thread = threading.Thread(target=self.arm_control_thread, args=(self.right_arm, True))
        self.left_arm_thread = threading.Thread(target=self.arm_control_thread, args=(self.left_arm, False))
        self.right_arm_thread.start()
        print("BimanualUREnv: Right arm Teleop Ready")
        self.left_arm_thread.start()
        print("BimanualUREnv: Left arm Teleop Ready")
        print("BimanualUREnv: Start UR Programs and Begin Teleoperation")
        print()
        return self.getObservation()

    """ Reset arms to start arms and gripper to start positions """
    def resetArms(self):
        print("BimanualUREnv: Resetting arms to start positions")
        self.right_arm.resetPosition()
        self.left_arm.resetPosition()
        print("BimanualUREnv: Finished resetting arms to start positions")

    """ Ensures the delta across any axis is not larger than a fixed value"""
    def restrictDelta(self, delta):
        for axis, delta_axis in enumerate(delta):
            if delta_axis > 0 and delta_axis > 0.5:
                delta[axis] = 0.05
            elif delta_axis < 0 and delta_axis < -0.5:
                delta[axis] = -0.05
        return delta
    
    def translateOculusAxesToURAxes(self, oculus_delta):
        return np.array([oculus_delta[2], oculus_delta[0], oculus_delta[1],
                            oculus_delta[5], oculus_delta[4], -1 * oculus_delta[3],
                            oculus_delta[6]])
    
    def convertOculusDeltas(self, action):
        left_arm_action, right_arm_action = action
        # Flip axes for left arm
        left_arm_action[0] = -1 * left_arm_action[0]
        left_arm_action[2] = -1 * left_arm_action[2]
        return self.translateOculusAxesToURAxes(left_arm_action), \
            self.translateOculusAxesToURAxes(right_arm_action)