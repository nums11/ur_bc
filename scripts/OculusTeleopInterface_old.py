from oculus_reader.oculus_reader.reader import OculusReader
from time import sleep
from URnterface import URInterface
import numpy as np
import threading

class OculusTeleopInterface:
    def __init__(self, right_arm_ip='192.168.2.2', left_arm_ip='192.168.1.2', reset_arms=False,
                    right_arm_start_joint__positions=None, left_arm_start_joint__positions=None,
                    robotiq_gripper_port='/dev/ttyUSB0'):
        print()
        # Initialize member variables
        self.reset_arms = reset_arms
        self.resetting = False
        self.right_arm_ip = right_arm_ip
        self.left_arm_ip = left_arm_ip
        self.robotiq_gripper_port = robotiq_gripper_port
        self.current_obs = None
        self.current_right_arm_action = None
        self.current_left_arm_action = None
        self.right_arm_action_updated = False
        self.left_arm_action_updated = False
        self.current_obs_action_pair = ()
        self.lock = threading.Lock()
        if right_arm_start_joint__positions == None:
            self.right_arm_start_joint__positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                        -1.095669650507004, -4.386985456001609, 3.2958897411425156])
        else:
            self.right_arm_start_joint__positions = right_arm_start_joint__positions
        if left_arm_start_joint__positions == None:
            self.left_arm_start_joint__positions = tuple([0.1001404325810099, -1.9640431421070108, 2.192831819213297,
                                                          4.154566166681737, -1.5883319440702799, 2.385492181115367])
        else:
            self.left_arm_start_joint__positions = left_arm_start_joint__positions

        # Initialize Oculus
        self.oculus = OculusReader()
        print("OculusTeleopInterface: Initialized Oculus")

        # Initialize UR Arms
        self.right_arm = URInterface(
            self.right_arm_ip, self.right_arm_start_joint__positions, has_robotiq_gripper=True)
        self.left_arm = URInterface(
            self.left_arm_ip, self.left_arm_start_joint__positions, has_robotiq_gripper=True,
            robotiq_gripper_port='/dev/ttyUSB2')
        print("OculusTeleopInterface: Initialized UR Interfaces")

        # Optionally reset arms to start positions
        if self.reset_arms:
            self.resetArms()
        print("OculusTeleopInterface: Finished Initializing OculusTeleopInterface")
        print()

    """ Observation """
    def obsActionCaptureThread(self):
        # Loop constantly getting the observation, then waiting until the next action
        # updates in the left and right arm, then store the observation and action pair
        while not self.resetting:
            obs = self.getObservation()
            while not (self.right_arm_action_updated and self.left_arm_action_updated):
                if self.resetting:
                    break
                else:
                    continue
            self.current_obs_action_pair = (obs, {'left_arm': self.current_left_arm_action, 'right_arm': self.current_right_arm_action})
            with self.lock:
                self.right_arm_action_updated = False
                self.left_arm_action_updated = False

        print("OculusTeleopInterface: Stopped observation, action capture thread due to reset")

    """ Arm Control """
    def arm_control_thread(self, arm, is_right_arm):
        global oculus
        # Get Initial Controller Position and Robot Pose
        controller_pose, _, _ = self.getControllerPoseAndTrigger(is_right_arm)
        robot_pose = arm.getPose()

        # Until the gripper is pressed, constanly send the current robot pose so that
        # the UR program will start with this pose and not jump to pose values previously stored in its registers
        gripper_pressed_before = False
        while not gripper_pressed_before and not self.resetting:
            # No movement yet so set action to 0
            self.storeAction(is_right_arm, self.zeroAction())
            new_controller_pose, gripper_pressed, _ = self.getControllerPoseAndTrigger(is_right_arm)
            arm.updateArmPose(robot_pose)
            if gripper_pressed:
                gripper_pressed_before = True
                break

        # Update Robot arm and gripper based on controller
        prev_gripper = False
        while not self.resetting:
            new_controller_pose, gripper_pressed, trigger_pressed = self.getControllerPoseAndTrigger(is_right_arm)
            # Only handle movements when right controller gripper is pressed
            if gripper_pressed:
                if not prev_gripper:
                    # Update current controller position to be new controller position on new gripper press
                    controller_pose = new_controller_pose
                else:
                    # Get the delta in controller position and rotation
                    ee_delta = self.getEEDelta(controller_pose, new_controller_pose, is_right_arm)
                    # Only update robot position if change in controller position meets a 
                    # certain threshold. This prevents robot from changing position when
                    # the control is almost still
                    if self.deltaMeetsThreshold(ee_delta):
                        # Restrict deltas to a maximum value to avoid large jumps in robot position
                        ee_delta = self.restrictDelta(ee_delta)
                        # Store the current action
                        self.storeAction(is_right_arm, ee_delta)
                        # Apply the action
                        new_robot_pose = robot_pose + ee_delta
                        arm.updateArmPose(new_robot_pose)
                        # Update the current controller and robot pose for the next iteration
                        controller_pose = new_controller_pose
                        robot_pose = new_robot_pose
                    else:
                        self.storeAction(is_right_arm, self.zeroAction())

                if trigger_pressed:
                    arm.moveRobotiqGripper(close=True)
                else:
                    arm.moveRobotiqGripper(close=False)
        
                prev_gripper = True
            else:
                self.storeAction(is_right_arm, self.zeroAction())
                prev_gripper = False

            sleep(0.005)
        
        print("OculusTeleopInterface: Stopped arm control thread due to reset")
        

    """ Get the rotational roll, pitch, and yaw from a transformation matrix """
    def get_roll_pitch_yaw_from_matrix(self, matrix):
        # Extract the rotation part from the matrix
        rotation_matrix = matrix[:3, :3]
        # Check for gimbal lock (singularities)
        if np.isclose(rotation_matrix[2, 0], 1.0):
            # Gimbal lock, positive singularity
            pitch = np.pi / 2
            roll = 0
            yaw = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
        elif np.isclose(rotation_matrix[2, 0], -1.0):
            # Gimbal lock, negative singularity
            pitch = -np.pi / 2
            roll = 0
            yaw = np.arctan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
        else:
            # General case
            pitch = np.arcsin(-rotation_matrix[2, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return roll, pitch, yaw

    """ Get translational and rotational pose from a transformation matrix  """
    def get6dPoseFromMatrix(self, matrix):
        if matrix.shape != (4, 4):
            raise ValueError("Input must be a 4x4 transformation matrix.")
        # Extract translation components
        x, y, z = matrix[:3, 3]
        # Extract rotation components and compute roll, pitch, yaw
        roll, pitch, yaw = self.get_roll_pitch_yaw_from_matrix(matrix)
        return np.array([x, y, z, roll, pitch, yaw])

    """ Get Oculus pose with RTr and RG gripper values """
    def getControllerPoseAndTrigger(self, is_right_arm):
        transformations, buttons = self.oculus.get_transformations_and_buttons()
        transformation_key = 'r'
        gripper_key = 'RG'
        trigger_key = 'RTr'
        if not is_right_arm:
            transformation_key = 'l'
            gripper_key = 'LG'
            trigger_key = 'LTr'
        pose = self.get6dPoseFromMatrix(transformations[transformation_key])
        return pose, buttons[gripper_key], buttons[trigger_key]

    """ Get the EE delta of the UR from the oculus delta """
    def getEEDelta(self, controller_pose, new_controller_pose, is_right_arm):
        ee_delta = new_controller_pose - controller_pose
        # Flip some axes for left arm
        if not is_right_arm:
            ee_delta[0] = -1 * ee_delta[0]
            ee_delta[2] = -1 * ee_delta[2]
            ee_delta[5] = -1 * ee_delta[5]
            # ee_delta[3] = 0

        # ee_delta[3] = 0
        # ee_delta[4] = 0
        # ee_delta[5] = 0

        # Move controller axes to correspond with the UR
        # ee_delta = np.array([ee_delta[2], ee_delta[0], ee_delta[1],
        #                     ee_delta[5], ee_delta[4], -1 * ee_delta[3]])
        ee_delta = np.array([ee_delta[2], ee_delta[0], ee_delta[1],
                    ee_delta[5], 0, -1 * ee_delta[3]])
        # ee_delta = np.array([0,0,0, 0, 0, 0])
        return ee_delta

    """ Returns True if a position delta is greater than some threshold across any axis """
    def deltaMeetsThreshold(self, delta):
        threshold = 1e-2
        for delta_axis in delta:
            if abs(delta_axis) > threshold:
                return True
        return False

    """ Ensures the delta across any axis is not larger than a fixed value"""
    def restrictDelta(self, delta):
        for axis, delta_axis in enumerate(delta):
            if delta_axis > 0 and delta_axis > 0.5:
                delta[axis] = 0.05
            elif delta_axis < 0 and delta_axis < -0.5:
                delta[axis] = -0.05
        return delta

    """ Get the observation for each arm (joint_pos, gripper) """
    def getObservation(self):
        return [self.left_arm.getObservation(), self.right_arm.getObservation()]

    """ Return the buttons from the oculus controller """ 
    def getButtons(self):
        _, buttons = self.oculus.get_transformations_and_buttons()
        return buttons
    
    """ Store the most recent action for the right or left arm """
    def storeAction(self, is_right_arm, action):
        if is_right_arm:
            self.current_right_arm_action = action
            with self.lock:
                self.right_arm_action_updated = True
        else:
            self.current_left_arm_action = action
            with self.lock:
                self.left_arm_action_updated = True
    
    """ Return the current observation and action """
    def getObsAndAction(self):
        return self.current_obs_action_pair
    
    """ Return a zero action """
    def zeroAction(self):
        return [0,0,0,0,0,0]
    
    """ Reset arms to start arms and gripper to start positions """
    def resetArms(self):
        print("OculusTeleopInterface: Resetting arms to start positions")
        # Kills observation threads and arm control threads if they are active
        self.resetting = True
        self.right_arm.resetPosition()
        self.left_arm.resetPosition()
        self.resetting = False
        print("Set resetting to false", self.resetting)
        print("OculusTeleopInterface: Finished resetting arms to start positions")

    """ Start the threads for observation capture and teleoperation """
    def startTeleop(self):
        print()
        # Initialize Threads and start them
        self.obs_action_capture_thread = threading.Thread(target=self.obsActionCaptureThread)
        self.right_arm_thread = threading.Thread(target=self.arm_control_thread, args=(self.right_arm, True))
        self.left_arm_thread = threading.Thread(target=self.arm_control_thread, args=(self.left_arm, False))
        print("OculusTeleopInterface: Initialized Observation and Arm Control Threads")
        self.obs_action_capture_thread.start()
        print("OculusTeleopInterface: Started thread for observation and action recording")
        self.right_arm_thread.start()
        print("OculusTeleopInterface: Right arm Teleop Ready")
        self.left_arm_thread.start()
        print("OculusTeleopInterface: Left arm Teleop Ready")
        print("OculusTeleopInterface: Start UR Programs and Begin Teleoperation")
        print()