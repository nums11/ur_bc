from interfaces.URInterface import URInterface
from interfaces.RSCameraInterface import RSCameraInterface
import threading
from time import sleep

class UREnv:
    def __init__(self, ee_actions=True, arm_ip='192.168.1.2', start_joint_positions=None,
                 has_3f_gripper=True, robotiq_gripper_port='/dev/ttyUSB0', use_camera=False):
                
        self.start_joint_positions = start_joint_positions
        if start_joint_positions == None:
            self.start_joint_positions = tuple([0.04474830963143529, -1.6422924423175793, 1.9634950313912025,
                                                4.267360912521422, -1.4365121397580038, 2.3399834772053114])
            
        self.arm = URInterface(arm_ip, self.start_joint_positions, has_3f_gripper=has_3f_gripper,
                                    robotiq_gripper_port=robotiq_gripper_port)
        self.arm_pose = self.arm.getPose()
        self.gripper = self.arm.getGripper()

        self.resetting = False

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

        print("Initialized UREnv")

    def reset(self):
        print("UREnv: Resetting")
        self.resetting = True
        sleep(1)
        self.arm.resetPosition()
        # Send current pose to arms so that it won't jump when the programs are started
        if self.ee_actions:
            self.arm_pose = self.arm.getPose()
            self.gripper = self.arm.getGripper()
            for _ in range(10):
                self.arm.updateArmPose(self.arm_pose)
        self.resetting = False
        print("UREnv: Finished Resetting. Start UR Program")
        return self._getObservation()
    
    def step(self, action, blocking=True):
        if self.ee_actions:
            self._stepEE(action)
        else:
            self._stepJoints(action, blocking)
        return self._getObservation()
    
    def _stepEE(self, action):
        self.arm_pose = self._limitWorkspace(action['arm_pose'])
        self.gripper = action['gripper']

    def _stepJoints(self, action, blocking=True):
        arm_j = action['arm_j']
        gripper = action['gripper']
        arm_thread = threading.Thread(target=self._armJThread,
                                            args=(arm_j, gripper, blocking))
        arm_thread.start()
        arm_thread.join()
    
    def _armJThread(self, joint_postiions, gripper, blocking=True):
        self.arm.movej(joint_postiions, blocking=blocking)
        self.arm.moveRobotiqGripper(gripper)

    def _armEEThread(self):
        while True:
            if not self.resetting:
                self.arm.updateArmPose(self.arm_pose)
                sleep(0.004)
    
    def _gripperThread(self):
        while True:
            if not self.resetting:
                self.arm.moveRobotiqGripper(self.gripper)
                sleep(0.004)

    def _getObservation(self):
        # Don't query the arms for ee poses, instead just maintain them
        # to avoid controller error which causes arm drift
        obs = {
                'arm_pose': self.arm_pose,
                'arm_j': self.arm.getj(),
                'gripper': self.arm.getGripper(),
                }
        if self.use_camera:
            obs['image'] = self.rs_camera.getCurrentImage()
        return obs
    
    def _limitWorkspace(self, pose, is_right_arm=False):
        if is_right_arm and pose[2] < 0.21:
            pose[2] = 0.21
        elif not is_right_arm and pose[2] < 0.08:
            pose[2] = 0.08
        elif pose[2] > 0.55:
            pose[2] = 0.55
        return pose
