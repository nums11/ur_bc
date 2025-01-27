
from environments.BimanualUREnv import BimanualUREnv
from controllers.KeyboardController import KeyboardController
from time import sleep
import threading

class KeyboardTeleopInterface:
    def __init__(self, left_arm_start_joint_positions=None, right_arm_start_joint_positions=None,
                 left_arm_has_3f_gripper=True, right_arm_has__3f_gripper=True,
                 use_camera=False):
        # Initialize the interface
        self.bimanual_ur_env = BimanualUREnv(left_arm_start_joint_positions=left_arm_start_joint_positions,
                                             right_arm_start_joint_positions=right_arm_start_joint_positions,
                                             left_arm_has_3f_gripper=left_arm_has_3f_gripper,
                                             right_arm_has__3f_gripper=right_arm_has__3f_gripper,
                                             use_camera=use_camera)
        self.keyboard_controller = KeyboardController()
        self.resetting = False
        self.obs = {}
        print("Initialized KeyboardTeleopInterface")

    def start(self):
        print("KeyboardTeleopInterface: Start")
        print("KeyboardTeleopInterface: Left arm controls - 'w', 's', 'a', 'd', 'q', 'e', 'r', 'f', 'z', 'x'")
        print("KeyboardTeleopInterface: Right arm controls - 'i', 'k', 'j', 'l', 'u', 'o', 'y', 'h', 'n', 'm'")
        self.teleop_thread = threading.Thread(target=self._teleopThread)
        self.teleop_thread.start()

    def _teleopThread(self):
        self.reset()
        while True:
            if not self.resetting:
                # Get deltas and grippers from keyboard
                left_arm_info, right_arm_info = self.keyboard_controller.getArmDeltasAndGrippersFromKeyPress()
                left_arm_delta, left_gripper = left_arm_info
                right_arm_delta, right_gripper = right_arm_info

                # Get current arm poses from env
                left_arm_pose = self.obs['left_arm_pose']
                right_arm_pose = self.obs['right_arm_pose']

                # Apply deltas
                left_arm_pose += left_arm_delta
                right_arm_pose += right_arm_delta

                # Construct action
                action = {
                    'left_arm_pose': left_arm_pose,
                    'right_arm_pose': right_arm_pose,
                    'left_gripper': left_gripper,
                    'right_gripper': right_gripper
                }
                
                # Step the environment
                self.obs = self.bimanual_ur_env.step(action)
                sleep(0.004) # 250hz

    def reset(self):
        self.resetting = True
        self.obs = self.bimanual_ur_env.reset()
        self.keyboard_controller.resetGripperValues()
        self.resetting = False

    def getObservation(self):
        return self.obs