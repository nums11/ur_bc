import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robomimic.utils.file_utils import policy_from_checkpoint
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
import numpy as np
from time import sleep
from pynput.keyboard import Listener

class ModelEvalInterface:
    def __init__(self, model_path, env):
        self.env = env
        self.normalize_types = ['min_max', 'mean_std']
        self.model, _ = policy_from_checkpoint(ckpt_path=model_path)
        self.model.start_episode()
        # Transformer stuff
        # Initialize buffers with zeros
        # self.frame_stack_size = 10
        # self.joint_and_gripper_size = 7  # Assuming 6 joint positions + 1 gripper position
        # self.batch_size = 1  # Assuming batch size of 1 for simplicity
        # self.left_joint_and_gripper_buffer = np.zeros((self.frame_stack_size, self.batch_size, self.joint_and_gripper_size))
        # self.right_joint_and_gripper_buffer = np.zeros((self.frame_stack_size, self.batch_size, self.joint_and_gripper_size))


        # Start the pynput keyboard listener
        self.start_evaluation = False
        self.keyboard_listener = Listener(
            on_release=self._on_release
        )
        self.keyboard_listener.start()
        print("ModelEvalInterface: Initialized")

            
    def _on_release(self, key):
        if not hasattr(key, 'char'):
            return
        if key.char == '1':
            self.start_evaluation = True

    def evaluate(self, blocking=False, freq=5, normalize=False, normalize_type='min_max'):
        print("ModelEvalInterface Evaluating blocking:", blocking, "freq:", freq, "normalize:", normalize, "normalize_type:", normalize_type)

        assert normalize_type in self.normalize_types, "Invalid normalize type valid types are: " + str(self.normalize_types)

        if blocking:
            assert self.env.action_type == 'joint_urx', "Blocking mode only supported for joint_urx action type"
        else:
            assert self.env.action_type == 'joint_modbus', "Non-blocking mode only supported for joint_modbus action type"

        if normalize and normalize_type == 'min_max':
            self.min_joint_positions, self.max_joint_positions = self._calculate_min_max(data_dir='/home/weirdlab/ur_bc/data')
            print("min_joint_positions:", self.min_joint_positions, "max_joint_positions:", self.max_joint_positions)
        elif normalize and normalize_type == 'mean_std':
            self.joint_mean, self.joint_std, self.gripper_mean, self.gripper_std = self._calculate_mean_std(data_dir='/home/weirdlab/ur_bc/data')
            print("joint_mean:", self.joint_mean, "joint_std:", self.joint_std)

        env_obs = self.env.reset()
        sleep(2)

        freq_sleep = 0
        if not blocking:
            freq_sleep = 1.0 / freq
            print("ModelEvalInterface: Start the UR Prgram then press '1' to start evaluation")
            while not self.start_evaluation:
                continue
        
        while True:
            model_obs = self._convertEnvObsToModelObs(env_obs, normalize, normalize_type)
            print("Observed")
            print(model_obs)

            predictions = self.model(model_obs)
            action = self._constructActionBasedOnEnv(env_obs, predictions, normalize)
            env_obs = self.env.step(action, blocking)
            if not blocking:
                sleep(freq_sleep)

    def _calculate_min_max(self, data_dir):
        joint_positions = []
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            for t in range(len(traj)):
                obs = traj[str(t)][0]
                joint_positions.append(obs['arm_j'])
        joint_positions = np.array(joint_positions)
        return np.min(joint_positions, axis=0), np.max(joint_positions, axis=0)
    
    def _calculate_mean_std(self, data_dir):
        joint_positions = []
        gripper_values = []
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            for t in range(len(traj)):
                obs = traj[str(t)][0]
                joint_positions.append(obs['arm_j'])
                gripper_values.append(obs['gripper'])
        joint_positions = np.array(joint_positions)
        # Min and values for each joint
        mean_joint_positions = np.mean(joint_positions, axis=0)
        std_joint_positions = np.std(joint_positions, axis=0)
        mean_gripper = np.mean(gripper_values)
        std_gripper = np.std(gripper_values)
        return mean_joint_positions, std_joint_positions, mean_gripper, std_gripper
    
    def _convertEnvObsToModelObs(self, obs, normalize, normalize_type):
        model_obs = None
        if type(self.env) == BimanualUREnv:
            left_arm_j = obs['left_arm_j']
            right_arm_j = obs['right_arm_j']
            # left_obs_gripper = np.expand_dims(self._normalizeGripper(obs['left_gripper']), axis=0)
            left_obs_gripper = np.expand_dims(obs['left_gripper'], axis=0)
            # right_obs_gripper = np.expand_dims(self._normalizeGripper(obs['right_gripper']), axis=0)
            right_obs_gripper = np.expand_dims(obs['right_gripper'], axis=0)
            model_obs = {
                'left_joint_and_gripper': np.concatenate((left_arm_j, left_obs_gripper)),
                'right_joint_and_gripper': np.concatenate((right_arm_j, right_obs_gripper)),
            }
        elif type(self.env) == UREnv:
            arm_j = obs['arm_j']
            if normalize and normalize_type == 'min_max':
                arm_j = (arm_j - self.min_joint_positions) / (self.max_joint_positions - self.min_joint_positions)
            elif normalize and normalize_type == 'mean_std':
                arm_j = (arm_j - self.joint_mean) / self.joint_std
            obs_gripper = np.expand_dims(obs['gripper'], axis=0)
            model_obs = {
                'joint_and_gripper': np.concatenate((arm_j, obs_gripper))
            }

        if self.env.use_camera:
            image = obs['image']
            # Change image shape to have channels first
            image = np.transpose(image, (2, 0, 1))
            model_obs['images'] = image
            
        return model_obs
    
    def _constructActionBasedOnEnv(self, env_obs, predictions, unnormalize):
        action = None
        if type(self.env) == BimanualUREnv:
            left_arm_delta = predictions[:6]
            # left_gripper = self._unnormalizeGripper(predictions[6])
            left_gripper = predictions[6]
            right_arm_delta = predictions[7:13]
            # right_gripper = self._unnormalizeGripper(predictions[13])
            right_gripper = predictions[13]

            print("Predicted")
            print("left_arm_delta", left_arm_delta)
            print("right_arm_delta", right_arm_delta)
            print("left_gripper", left_gripper)
            print("right_gripper", right_gripper)

            left_arm_j = env_obs['left_arm_j'] + left_arm_delta
            right_arm_j = env_obs['right_arm_j'] + right_arm_delta
            action = {
                'left_arm_j': left_arm_j,
                'right_arm_j': right_arm_j,
                'left_gripper': self._convertGripperToBinary(left_gripper),
                'right_gripper': self._convertGripperToBinary(right_gripper)
            }
        elif type(self.env) == UREnv:
            arm_delta = predictions[:6]
            # if unnormalize:
            #     arm_delta = arm_delta * (self.max_joint_positions - self.min_joint_positions) + self.min_joint_positions
            gripper = predictions[6]
            if unnormalize:
                print("UNNORMALIZING")
                gripper = gripper * self.gripper_std + self.gripper_mean

            print("Predicted")
            print("arm_delta", arm_delta)
            print("gripper", gripper)

            arm_j = env_obs['arm_j'] + arm_delta
            action = {
                'arm_j': arm_j,
                'gripper': self._convertGripperToBinary(gripper),
            }
        return action
    
    def _convertGripperToBinary(self, gripper_value):
        return gripper_value > 0.5
    
    # def _convertEnvObsToTransformerObs(self, obs):
    #     left_arm_j = obs['left_arm_j']
    #     right_arm_j = obs['right_arm_j']
    #     left_obs_gripper = np.expand_dims(obs['left_gripper'], axis=0)
    #     right_obs_gripper = np.expand_dims(obs['right_gripper'], axis=0)

    #     left_joint_and_gripper = np.concatenate((left_arm_j, left_obs_gripper))
    #     right_joint_and_gripper = np.concatenate((right_arm_j, right_obs_gripper))

    #     # Update buffers
    #     self.left_joint_and_gripper_buffer = np.roll(self.left_joint_and_gripper_buffer, -1, axis=0)
    #     self.right_joint_and_gripper_buffer = np.roll(self.right_joint_and_gripper_buffer, -1, axis=0)
    #     self.left_joint_and_gripper_buffer[-1, 0, :] = left_joint_and_gripper
    #     self.right_joint_and_gripper_buffer[-1, 0, :] = right_joint_and_gripper
        
    #     model_obs = {
    #         'left_joint_and_gripper': self.left_joint_and_gripper_buffer,
    #         'right_joint_and_gripper': self.right_joint_and_gripper_buffer,
    #     }

    #     return model_obs