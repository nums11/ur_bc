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

        # Frame stack buffers
        self.joint_and_gripper_buffer = None
        self.image_buffer = None
        self.wrist_image_buffer = None

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

    def evaluate(self, blocking=False, is_delta_action=False, freq=5, normalize=False, normalize_type='min_max', transformer_model=False,
        diffusion_model=False, frame_stack_size=10):
        print("ModelEvalInterface Evaluating blocking:", blocking, "freq:", freq, "normalize:", normalize, "normalize_type:", normalize_type,
              "transformer_model:", transformer_model, "diffusion_model:", diffusion_model, "frame_stack_size:", frame_stack_size)
        self.frame_stack_size = frame_stack_size
        self.is_delta_action = is_delta_action
        assert normalize_type in self.normalize_types, "Invalid normalize type valid types are: " + str(self.normalize_types)
        assert not (transformer_model and diffusion_model), "Cannot have both transformer and diffusion models"

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
            if transformer_model:
                model_obs = self._convertEnvObsToTransformerObs(env_obs, normalize, normalize_type)
            elif diffusion_model:
                model_obs = self._convertEnvObsToDiffusionObs(env_obs)
            else:
                model_obs = self._convertEnvObsToModelObs(env_obs, normalize, normalize_type)
            print("Observed")
            print(model_obs)

            predictions = self.model(model_obs)
            action = self._constructActionBasedOnEnv(env_obs, predictions, normalize, is_delta_action)
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
            obs_gripper = np.expand_dims(obs['gripper'], axis=0)
            if normalize and normalize_type == 'min_max':
                arm_j = (arm_j - self.min_joint_positions) / (self.max_joint_positions - self.min_joint_positions)
            elif normalize and normalize_type == 'mean_std':
                arm_j = (arm_j - self.joint_mean) / self.joint_std
                obs_gripper = (obs_gripper - self.gripper_mean) / self.gripper_std
            model_obs = {
                'joint_and_gripper': np.concatenate((arm_j, obs_gripper))
            }

        if self.env.use_camera:
            image = obs['image']
            wrist_image = obs['wrist_image']
            # Change image shape to have channels first
            image = np.transpose(image, (2, 0, 1))
            wrist_image = np.transpose(wrist_image, (2, 0, 1))
            model_obs['images'] = image
            model_obs['wrist_images'] = wrist_image
            
        return model_obs
    
    def _constructActionBasedOnEnv(self, env_obs, predictions, unnormalize, is_delta_action):
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

            if is_delta_action:
                arm_j = env_obs['arm_j'] + arm_delta
            else:
                arm_j = arm_delta

            action = {
                'arm_j': arm_j,
                'gripper': self._convertGripperToBinary(gripper),
            }
        return action
    
    def _convertGripperToBinary(self, gripper_value):
        return gripper_value > 0.5
    
    def _convertEnvObsToTransformerObs(self, obs, normalize, normalize_type):
        arm_j = obs['arm_j']
        obs_gripper = np.expand_dims(obs['gripper'], axis=0)

        if normalize and normalize_type == 'min_max':
            arm_j = (arm_j - self.min_joint_positions) / (self.max_joint_positions - self.min_joint_positions)
        elif normalize and normalize_type == 'mean_std':
            arm_j = (arm_j - self.joint_mean) / self.joint_std
            obs_gripper = (obs_gripper - self.gripper_mean) / self.gripper_std

        joint_and_gripper = np.concatenate((arm_j, obs_gripper))

        if self.joint_and_gripper_buffer is None:
            self.joint_and_gripper_buffer = self._initialize_buffer(joint_and_gripper)

        # Update buffers
        self.joint_and_gripper_buffer = np.roll(self.joint_and_gripper_buffer, -1, axis=0)
        self.joint_and_gripper_buffer[-1, :] = joint_and_gripper

        model_obs = {
            'joint_and_gripper': self.joint_and_gripper_buffer,
        }

        return model_obs

    def _convertEnvObsToDiffusionObs(self, obs):
        # Concatenate arm_j and gripper
        arm_j = obs['arm_j']
        obs_gripper = np.expand_dims(obs['gripper'], axis=0)
        joint_and_gripper = np.concatenate((arm_j, obs_gripper))

        # Transpose images to have channels first
        image = np.transpose(obs['image'], (2, 0, 1))
        wrist_image = np.transpose(obs['wrist_image'], (2, 0, 1))

        # Initialize buffers if they are not initialized
        if self.joint_and_gripper_buffer is None:
            self.joint_and_gripper_buffer = self._initialize_buffer(joint_and_gripper)
        if self.image_buffer is None:
            self.image_buffer = self._initialize_buffer(image)
        if self.wrist_image_buffer is None:
            self.wrist_image_buffer = self._initialize_buffer(wrist_image)

        # Update buffers
        self.joint_and_gripper_buffer = np.roll(self.joint_and_gripper_buffer, -1, axis=0)
        self.joint_and_gripper_buffer[-1, :] = joint_and_gripper
        self.image_buffer = np.roll(self.image_buffer, -1, axis=0)
        self.image_buffer[-1, :, :, :] = image
        self.wrist_image_buffer = np.roll(self.wrist_image_buffer, -1, axis=0)
        self.wrist_image_buffer[-1, :, :, :] = wrist_image

        # Create model observation
        model_obs = {}
        model_obs['joint_and_gripper'] = self.joint_and_gripper_buffer
        model_obs['images'] = self.image_buffer
        model_obs['wrist_images'] = self.wrist_image_buffer
        return model_obs
    
    def _initialize_buffer(self, initial_frame):
        # Initialize the buffer with the initial frame duplicated
        print("Initializing buffer")
        return np.array([initial_frame] * self.frame_stack_size)
