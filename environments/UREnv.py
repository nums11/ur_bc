from interfaces.URInterface import URInterface
from interfaces.RSCameraInterface import RSCameraInterface
import threading
from time import sleep

class UREnv:
    def __init__(self, action_type='ee', arm_ip='192.168.1.2', limit_workspace=False, start_joint_positions=None,
                 has_3f_gripper=True, robotiq_gripper_port='/dev/ttyUSB0', use_camera=False):
        
        self.limit_workspace = limit_workspace
                
        self.start_joint_positions = start_joint_positions
        if start_joint_positions == None:
            self.start_joint_positions = tuple([0.04474830963143529, -1.6422924423175793, 1.9634950313912025,
                                                4.267360912521422, -1.4365121397580038, 2.3399834772053114])
            
        self.arm = URInterface(arm_ip, self.start_joint_positions, has_3f_gripper=has_3f_gripper,
                                    robotiq_gripper_port=robotiq_gripper_port)
        self.arm_pose = self.arm.getPose()
        self.arm_j = self.arm.getj()
        self.gripper = self.arm.getGripper()

        self.resetting = False

        self.valid_action_types = ['ee', 'joint_urx', 'joint_modbus']
        if not action_type in self.valid_action_types:
            raise ValueError(f"Invalid action type: {action_type}. Valid action types are {self.valid_action_types}")
        self.action_type = action_type
        if self.usesEEActions() or self.usesJointModbusActions():
            self.arm_thread = threading.Thread(target=self._armModbusThread)
            self.gripper_thread = threading.Thread(target=self._gripperThread)
            self.arm_thread.start()
            self.gripper_thread.start()

        self.use_camera = use_camera
        if self.use_camera:
            self.rs_camera = RSCameraInterface(serial_number='746112060198')
            self.rs_camera.startCapture()

        self.reset_counter = 0

        print("Initialized UREnv")

    def reset(self):
        print("UREnv: Resetting")
        self.resetting = True
        sleep(1)
        if self.reset_counter == 0:
            self.arm.resetPositionURX()
            self.start_arm_pose = self.arm.getPose()
        else:
            if self.usesEEActions():
                self.arm.resetPositionModbus(self.arm_pose, self.start_arm_pose)
            elif self.usesJointModbusActions():
                self.arm.resetPositionModbus(self.arm_j, self.start_joint_positions)
            elif self.usesJointURXActions():
                self.arm.resetPositionURX()
        # Send current pose or joint values to arms so that it won't jump when the programs are started
        if self.usesEEActions() or self.usesJointModbusActions():
            self.arm_pose = self.arm.getPose()
            self.arm_j = self.arm.getj()
            self.gripper = self.arm.getGripper()
            for _ in range(10):
                if self.usesEEActions():
                    self.arm.sendModbusValues(self.arm_pose)
                elif self.usesJointModbusActions():
                    self.arm.sendModbusValues(self.arm_j)
        self.resetting = False
        print("UREnv: Finished Resetting. Start UR Program")
        self.reset_counter += 1
        return self._getObservation()
    
    def step(self, action, blocking=True):
        if self.usesEEActions():
            self._stepEE(action)
        elif self.usesJointURXActions():
            self._stepJointsURX(action, blocking)
        elif self.usesJointModbusActions():
            self._stepJointModbus(action)
        return self._getObservation()
    
    def _stepEE(self, action):
        if self.limit_workspace:
            self.arm_pose = self._limitWorkspace(action['arm_pose'])
        else:
            self.arm_pose = action['arm_pose']
        self.gripper = action['gripper']

    def _stepJointsURX(self, action, blocking=True):
        arm_j = action['arm_j']
        gripper = action['gripper']
        arm_thread = threading.Thread(target=self._armJThread,
                                            args=(arm_j, gripper, blocking))
        arm_thread.start()
        arm_thread.join()

    def _stepJointModbus(self, action):
        self.arm_j = action['arm_j']
        self.gripper = action['gripper']
    
    def _armJThread(self, joint_postiions, gripper, blocking=True):
        self.arm.movej(joint_postiions, blocking=blocking)
        self.arm.moveRobotiqGripper(gripper)

    def _armModbusThread(self):
        while True:
            if not self.resetting:
                if self.usesEEActions():
                    self.arm.sendModbusValues(self.arm_pose)
                elif self.usesJointModbusActions():
                    self.arm.sendModbusValues(self.arm_j)
                sleep(0.004)
    
    def _gripperThread(self):
        while True:
            if not self.resetting:
                self.arm.moveRobotiqGripper(self.gripper)
                sleep(0.004)

    def _getObservation(self):
        obs = {
                'gripper': self.arm.getGripper(),
                }

        obs['arm_pose'] = self.arm.getPose()
        obs['arm_j'] = self.arm.getj()
        obs['force'] = self.arm.getForce()
        if self.use_camera:
            obs['image'], obs['wrist_image'] = self.rs_camera.getCurrentImage()

        return obs
    
    def _limitWorkspace(self, pose, is_right_arm=False):
        if is_right_arm and pose[2] < 0.21:
            pose[2] = 0.21
        elif not is_right_arm and pose[2] < 0.08:
            pose[2] = 0.08
        elif pose[2] > 0.55:
            pose[2] = 0.55
        return pose
    
    def usesEEActions(self):
        return self.action_type == 'ee'
    
    def usesJointURXActions(self):
        return self.action_type == 'joint_urx'
    
    def usesJointModbusActions(self):
        return self.action_type == 'joint_modbus'
