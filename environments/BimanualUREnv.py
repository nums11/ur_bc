from interfaces.URInterface import URInterface
from interfaces.RSCameraInterface import RSCameraInterface
import threading
from time import sleep

class BimanualUREnv:
    def __init__(self, ee_actions=True, right_arm_ip='192.168.2.2', left_arm_ip='192.168.1.2',
                 right_arm_start_joint_positions=None, left_arm_start_joint_positions=None,
                 right_arm_has_gripper=True, left_arm_has_gripper=True, use_camera=False):
        
        self.right_arm_start_joint_positions = right_arm_start_joint_positions
        if right_arm_start_joint_positions == None:
            self.right_arm_start_joint_positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                                -1.095669650507004, -4.386985456001609, 3.2958897411425156])
        self.left_arm_start_joint_positions = left_arm_start_joint_positions
        if left_arm_start_joint_positions == None:
            self.left_arm_start_joint_positions = tuple([0.04474830963143529, -1.6422924423175793, 1.9634950313912025,
                                                4.267360912521422, -1.4365121397580038, 2.3399834772053114])
            
        self.right_arm = URInterface(right_arm_ip, self.right_arm_start_joint_positions, has_robotiq_gripper=right_arm_has_gripper,
                                     robotiq_gripper_port='/dev/ttyUSB2')
        self.left_arm = URInterface(left_arm_ip, self.left_arm_start_joint_positions, has_robotiq_gripper=left_arm_has_gripper,
                                    robotiq_gripper_port='/dev/ttyUSB1')
        self.right_arm_pose = self.right_arm.getPose()
        self.left_arm_pose = self.left_arm.getPose()
        self.left_gripper = self.left_arm.getGripper()
        self.right_gripper = self.right_arm.getGripper()

        self.resetting = False
        self.last_observation = {}

        self.ee_actions = ee_actions
        if self.ee_actions:
            self.arm_thread = threading.Thread(target=self._armEEThread)
            self.gripper_thread = threading.Thread(target=self._gripperThread)
            self.arm_thread.start()
            self.gripper_thread.start()

        self.use_camera = use_camera
        if self.use_camera:
            self.rs_camera = RSCameraInterface()
            self.rs_camera.startCapture()

        print("Initialized BimanualUREnv")

    def reset(self):
        print("BimanualUREnv: Resetting")
        self.resetting = True
        self.right_arm.resetPosition()
        self.left_arm.resetPosition()
        # Send current pose to arms so that it won't jump when the programs are started
        if self.ee_actions:
            self.left_arm_pose = self.left_arm.getPose()
            self.right_arm_pose = self.right_arm.getPose()
            self.left_gripper = self.left_arm.getGripper()
            self.right_gripper = self.right_arm.getGripper()
            for _ in range(10):
                self.left_arm.updateArmPose(self.left_arm_pose)
                self.right_arm.updateArmPose(self.right_arm_pose)
        # Makes sure the keyboard knows that the grippers are open since the
        # arm was reset
        # self.keyboard.resetGripperValues()
        self.resetting = False
        print("BimanualUREnv: Finished Resetting. Start UR Programs")
        return self._getObservation()

    def step(self, action):
        if self.ee_actions:
            self._stepEE(action)
        else:
            self._stepJoints(action)
        return self._getObservation()
    
    def _stepEE(self, action):
        self.left_arm_pose = action['left_arm_pose']
        self.right_arm_pose = action['right_arm_pose']
        self.left_gripper = action['left_gripper']
        self.right_gripper = action['right_gripper']
    
    def _stepJoints(self, action):
        left_arm_j = action['left_arm_j']
        left_gripper = action['left_gripper']
        right_arm_j = action['right_arm_j']
        right_gripper = action['right_gripper']
        left_arm_thread = threading.Thread(target=self._armJThread,
                                            args=(self.left_arm, left_arm_j, left_gripper))
        right_arm_thread = threading.Thread(target=self._armJThread,
                                            args=(self.right_arm, right_arm_j, right_gripper))
        left_arm_thread.start()
        right_arm_thread.start()
        left_arm_thread.join()
        right_arm_thread.join()

    def _armJThread(self, arm, joint_postiions, gripper):
        arm.movej(joint_postiions)
        arm.moveRobotiqGripper(gripper)

    def _armEEThread(self):
        while True:
            if not self.resetting:
                self.left_arm.updateArmPose(self.left_arm_pose)
                self.right_arm.updateArmPose(self.right_arm_pose)
                sleep(0.004)
        
    def _gripperThread(self):
        while True:
            if not self.resetting:
                self.left_arm.moveRobotiqGripper(self.left_gripper)
                self.right_arm.moveRobotiqGripper(self.right_gripper)
                sleep(0.004)

    def _getObservation(self):
        # Don't query the arms for ee poses, instead just maintain them
        # to avoid controller error which causes arm drift
        obs = {
                'left_arm_pose': self.left_arm_pose,
                'right_arm_pose': self.right_arm_pose,
                'left_arm_j': self.left_arm.getj(),
                'right_arm_j': self.right_arm.getj(),
                'left_gripper': self.left_arm.getGripper(),
                'right_gripper': self.right_arm.getGripper()
                }
        if self.use_camera:
            obs['image'] = self.rs_camera.getCurrentImage()
        return obs

    def render(self):
        pass

    def close(self):
        pass