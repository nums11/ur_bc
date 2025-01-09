from URnterface import URInterface
from pynput.keyboard import Listener
from time import sleep
import numpy as np

# Define keyboard control class
class ContinuousControlKeyboard:
    def __init__(self, translational_sensitivity=0.006, rotational_sensitivity=0.01):
        # Define the mappings between keys and their corresponding translational or
        # rotational deltas
        self.left_arm_key_to_delta_mappings = {
            'w': np.asarray([-1.0, 0.0, 0.0]) * translational_sensitivity,
            's': np.asarray([1.0, 0.0, 0.0]) * translational_sensitivity,
            'a': np.asarray([0.0, -1.0, 0.0]) * translational_sensitivity,
            'd': np.asarray([0.0, 1.0, 0.0]) * translational_sensitivity,
            'q': np.asarray([0.0, 0.0, 1.0]) * translational_sensitivity,
            'e': np.asarray([0.0, 0.0, -1.0]) * translational_sensitivity,
            'r': np.asarray([0.0, 0.0, 1.0]) * rotational_sensitivity,
            'f': np.asarray([0.0, 0.0, -1.0]) * rotational_sensitivity,
            'z': np.asarray([-1.0, 0.0, 0.0]) * rotational_sensitivity,
            'x': np.asarray([1.0, 0.0, 0.0]) * rotational_sensitivity,
        }
        self.right_arm_key_to_delta_mappings = {
            'i': np.asarray([1.0, 0.0, 0.0]) * translational_sensitivity,
            'k': np.asarray([-1.0, 0.0, 0.0]) * translational_sensitivity,
            'j': np.asarray([0.0, 1.0, 0.0]) * translational_sensitivity,
            'l': np.asarray([0.0, -1.0, 0.0]) * translational_sensitivity,
            'u': np.asarray([0.0, 0.0, 1.0]) * translational_sensitivity,
            'o': np.asarray([0.0, 0.0, -1.0]) * translational_sensitivity,
            'y': np.asarray([0.0, 0.0, 1.0]) * rotational_sensitivity,
            'h': np.asarray([0.0, 0.0, -1.0]) * rotational_sensitivity,
            'n': np.asarray([1.0, 0.0, 0.0]) * rotational_sensitivity,
            'm': np.asarray([-1.0, 0.0, 0.0]) * rotational_sensitivity,
        }

        # Define the valid keys for translational and rotaional movement
        # as well as gripper toggle
        self.left_arm_translational_chars = ['w', 's', 'a', 'd','q', 'e']
        self.right_arm_translational_chars = ['i', 'k', 'j', 'l','u', 'o']
        self.left_arm_rotational_chars = ['r', 'f', 'z', 'x']
        self.right_arm_rotational_chars = ['y', 'h', 'n', 'm']
        self.gripper_chars = ['v','b']

        self.pressed_keys = []
        self.left_arm_pos_delta = np.zeros(3)
        self.right_arm_pos_delta = np.zeros(3)
        self.left_arm_rot_delta = np.zeros(3)
        self.right_arm_rot_delta = np.zeros(3)
        self.close_left_gripper = False
        self.close_right_gripper = False

        # Start the pynput keyboard listener
        self.keyboard_listener = Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

    def on_press(self, key):
        if not hasattr(key, 'char'):
            return
        
        char = key.char
        # Toggle the gripper when a gripper character is pressed
        if char in self.gripper_chars:
            if char == 'v':
                self.close_left_gripper = not self.close_left_gripper
            elif char == 'b':
                self.close_right_gripper = not self.close_right_gripper
        # Handle newly pressed characters
        elif char not in self.pressed_keys:
            # Store the character in the list of pressed keys so that when it is
            # released it's delta will be subtracted
            self.pressed_keys.append(char)
            # Modify the positional or rotational delta for the left or right arm
            if char in self.left_arm_translational_chars:
                self.left_arm_pos_delta += self.left_arm_key_to_delta_mappings[char]
            elif char in self.left_arm_rotational_chars:
                self.left_arm_rot_delta += self.left_arm_key_to_delta_mappings[char]
            elif char in self.right_arm_translational_chars:
                self.right_arm_pos_delta += self.right_arm_key_to_delta_mappings[char]
            elif char in self.right_arm_rotational_chars:
                self.right_arm_rot_delta += self.right_arm_key_to_delta_mappings[char]

    def on_release(self, key):        
        if not hasattr(key, 'char'):
            return
        
        char = key.char
        # Remove non gripper characters from the list of pressed characters
        if char not in self.gripper_chars:
            self.pressed_keys.remove(char)
        # Subtract it's left or right arm, positional or rotational delta
        if char in self.left_arm_translational_chars:
            self.left_arm_pos_delta -= self.left_arm_key_to_delta_mappings[char]
        elif char in self.left_arm_rotational_chars:
            self.left_arm_rot_delta -= self.left_arm_key_to_delta_mappings[char]
        elif char in self.right_arm_translational_chars:
            self.right_arm_pos_delta -= self.right_arm_key_to_delta_mappings[char]
        elif char in self.right_arm_rotational_chars:
            self.right_arm_rot_delta -= self.right_arm_key_to_delta_mappings[char]

    def advance(self):
        return ((np.concatenate((self.left_arm_pos_delta, self.left_arm_rot_delta)), self.close_left_gripper), 
                (np.concatenate((self.right_arm_pos_delta, self.right_arm_rot_delta)), self.close_right_gripper))
    
# Initialize Arms
right_arm_start_joint__positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                        -1.095669650507004, -4.386985456001609, 3.2958897411425156])
left_arm_start_joint__positions = tuple([0.1001404325810099, -1.9640431421070108, 2.192831819213297,
                                                4.154566166681737, -1.5883319440702799, 2.385492181115367])
right_arm = URInterface('192.168.2.2', right_arm_start_joint__positions, has_robotiq_gripper=True)
left_arm = URInterface('192.168.1.2', left_arm_start_joint__positions, has_robotiq_gripper=True,
                       robotiq_gripper_port='/dev/ttyUSB2')
right_arm.resetPosition()
left_arm.resetPosition()
right_arm_pose = right_arm.getPose()
left_arm_pose = left_arm.getPose()
for _ in range(10):
    right_arm.updateArmPose(right_arm_pose)
    left_arm.updateArmPose(left_arm_pose)

# Control loop
keyboard = ContinuousControlKeyboard()
while True:
    left_arm_info, right_arm_info = keyboard.advance()
    left_arm_delta, left_gripper = left_arm_info
    right_arm_delta, right_gripper = right_arm_info
    # print("left_arm_delta", left_arm_delta, "left_gripper", left_gripper, "right_arm_delta", right_arm_delta, "right_gripper", right_gripper)
    # Update the poses basded on the delta
    left_arm_pose += left_arm_delta
    right_arm_pose += right_arm_delta
    # Move the arms and grippers
    left_arm.updateArmPose(left_arm_pose)
    right_arm.updateArmPose(right_arm_pose)
    left_arm.moveRobotiqGripper(left_gripper)
    right_arm.moveRobotiqGripper(right_gripper)
    sleep(0.05)