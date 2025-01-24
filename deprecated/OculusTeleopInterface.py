from oculus_reader.oculus_reader.reader import OculusReader
import threading
import numpy as np
from time import sleep

class OculusTeleopInterface:
    def __init__(self, movement_threshold=1e-2, max_delta=0.5):
        # Initialize member variables
        self.movement_threshold = movement_threshold
        self.max_delta = max_delta
        self.setZeroDeltas()
        self.left_trigger_pressed = False
        self.right_trigger_pressed = False

        # Initialize Oculus
        self.oculus = OculusReader()
        # OculusReader needs time to initialize otherwise controller
        # reads will be empty
        sleep(1)
        print("OculusTeleopInterface: Initialized Oculus")

    """ Sets controller deltas when the gripper for a controller is pressed.
        Deltas that are below a minimum movement threshold are returned as 0 and deltas above 
        a maxium movement threshold are decreased to a fixed value. """
    def teleopThread(self, is_right_controller=False):
        controller_pose, _, _ = self.getControllerPoseAndTrigger(is_right_controller)
        prev_gripper = False
        while True:
            new_controller_pose, gripper_pressed, trigger_pressed = self.getControllerPoseAndTrigger(is_right_controller)
            # Only calculate deltas when the gripper for the controller is pressed
            if gripper_pressed:
                # Update current controller position to be new controller position on new gripper press
                if not prev_gripper:
                    controller_pose = new_controller_pose
                else:
                    # Get the delta in controller position and rotation
                    ee_delta = new_controller_pose - controller_pose
                    # Only register deltas if they meet a certain threshold. This prevents deltas from being
                    # returned when the controllers are almost still
                    if self.deltaMeetsThreshold(ee_delta):
                        # Restrict deltas to a maximum value to avoid large jumps in robot position
                        ee_delta = self.restrictDelta(ee_delta)
                        if is_right_controller:
                            self.right_controller_delta = ee_delta
                        else:
                            self.left_controller_delta = ee_delta
                    else:
                        self.setZeroDeltas()

                    # Update the value of the trigger
                    self.updateTriggerPressed(trigger_pressed, is_right_controller)

                prev_gripper = True
            else:
                self.setZeroDeltas()
                prev_gripper = False

    """ Returns controller deltas and trigger values """
    def getDeltasAndTriggers(self):
        return self.left_controller_delta, self.right_controller_delta, \
            self.left_trigger_pressed, self.right_trigger_pressed

    " Starts teleop threads for both controllers "
    def startTeleop(self):
        self.right_controller_teleop_thread = threading.Thread(
            target=self.teleopThread, args=(True,))
        self.left_controller_teleop_thread = threading.Thread(
            target=self.teleopThread)
        self.right_controller_teleop_thread.start()
        self.left_controller_teleop_thread.start()
        print("OculusTeleopInterface: Started Teleop Threads")

    """ Sets the right and left controller deltas to 0 """
    def setZeroDeltas(self):
        self.right_controller_delta = self.zeroDelta()
        self.left_controller_delta = self.zeroDelta()

    """ Returns a delta of 0 across all axes """
    def zeroDelta(self):
        return [0,0,0,0,0,0]
    
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
    
    """ Get Oculus pose with trigger and gripper values """
    def getControllerPoseAndTrigger(self, is_right_controller):
        transformations, buttons = self.oculus.get_transformations_and_buttons()
        transformation_key = 'r'
        gripper_key = 'RG'
        trigger_key = 'RTr'
        if not is_right_controller:
            transformation_key = 'l'
            gripper_key = 'LG'
            trigger_key = 'LTr'
        pose = self.get6dPoseFromMatrix(transformations[transformation_key])
        return pose, buttons[gripper_key], buttons[trigger_key]
    
    """ Returns True if a position delta is greater than some threshold across any axis """
    def deltaMeetsThreshold(self, delta):
        for delta_axis in delta:
            if abs(delta_axis) > self.movement_threshold:
                return True
        return False
    
    """ Ensures the delta across any axis is not larger than a fixed value"""
    def restrictDelta(self, delta):
        for axis, delta_axis in enumerate(delta):
            if delta_axis > 0 and delta_axis > self.max_delta:
                delta[axis] = 0.05
            elif delta_axis < 0 and delta_axis < -1 * self.max_delta:
                delta[axis] = -0.05
        return delta
    
    def updateTriggerPressed(self, trigger_pressed, is_right_controller):
        if is_right_controller:
            self.right_trigger_pressed = trigger_pressed
        else:
            self.left_trigger_pressed = trigger_pressed
