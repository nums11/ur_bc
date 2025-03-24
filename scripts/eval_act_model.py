import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
import threading
import cv2
from time import sleep

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../act')))
from environments.UREnv import UREnv

from act.utils import set_seed # helper functions


def main(args):
    try:
        # Load the UR Env
        env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=True, use_logitech_camera=True,
            start_joint_positions=tuple([-0.10184682002801448, -1.8316009921757344, 2.2237440184163777,
                    -1.9278720721999862, -1.5840280733482741, 0.04111786366790808]))
        
        set_seed(1000)

        from act.policy import ACTPolicy
        def make_policy(policy_config):
            return ACTPolicy(policy_config)

        # Construct policy_cfg
        camera_names = ['camera', 'wrist_camera']
        state_dim = 7
        lr_backbone = 1e-5
        backbone = 'resnet18'
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }
        ckpt_dir = args['ckpt_dir']
        ckpt_name = args['ckpt_name']
        policy_class = 'ACT'

        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy = make_policy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        policy.cuda()
        policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        query_frequency = policy_config['num_queries']

        # Start the control loop in a separate thread
        control_thread = threading.Thread(
            target=run_control_loop,
            args=(env, policy, pre_process, post_process, query_frequency),
            daemon=True  # Make it a daemon so it exits when the main thread exits
        )
        control_thread.start()
        
        # Keep the main thread alive to process window events
        while control_thread.is_alive():
            cv2.waitKey(1)  # Process window events
            sleep(0.01)  # Sleep a bit to avoid hogging CPU
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up all resources
        print("Performing cleanup...")
        # Clean up environment
        if 'env' in locals() and env:
            # Stop any running threads
            env.resetting = True  # This should stop any ongoing threads
            # Close camera if it's running
            if env.use_camera and hasattr(env, 'camera'):
                try:
                    env.camera.stopCapture()
                    print("Stopped camera capture")
                except Exception as e:
                    print(f"Error stopping camera: {e}")
            # Close robot connection
            if hasattr(env, 'arm') and hasattr(env.arm, 'close'):
                try:
                    env.arm.close()
                    print("Closed robot arm connection")
                except Exception as e:
                    print(f"Error closing arm connection: {e}")
        print("Cleanup complete")

# Create a function for the control loop
def run_control_loop(env, policy, pre_process, post_process, query_frequency):
    obs = env.reset()

    # Add a 10-second countdown before starting
    print("Starting in 10 seconds. Get ready...")
    for i in range(10, 0, -1):
        print(f"{i}...")
        cv2.waitKey(1)  # Process window events during countdown
        sleep(1)
        
    t = 0
    with torch.inference_mode():
        while True:
            # Get obs from env
            arm_j = obs['arm_j']
            obs_gripper = np.expand_dims(obs['gripper'], axis=0)
            qpos_numpy = np.concatenate((arm_j, obs_gripper))
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            # Convert images to correct format
            image = rearrange(obs['image'], 'h w c -> c h w')
            wrist_image = rearrange(obs['wrist_image'], 'h w c -> c h w')
            all_images = [image, wrist_image]
            all_images = np.stack(all_images, axis=0)
            all_images = torch.from_numpy(all_images / 255.0).float().cuda().unsqueeze(0)

            # query policy
            if t % query_frequency == 0:
                all_actions = policy(qpos, all_images)
            raw_action = all_actions[:, t % query_frequency]
            
            # post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action
            target_j = target_qpos[:6]
            target_gripper = target_qpos[6:]
            print("Predicted target_j: ", target_j, "Predicted target_gripper: ", target_gripper)
            if target_gripper > 0.5:
                target_gripper = True
            else:
                target_gripper = False
            env_action = {'arm_j': target_j, 'gripper': target_gripper}
            
            # step the environment
            obs = env.step(env_action)

            t += 1
            sleep(1/50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000)

        # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200)
    parser.add_argument('--temporal_agg', action='store_true')


    main(vars(parser.parse_args()))