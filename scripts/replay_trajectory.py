import sys
import os
import argparse

# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Replay a trajectory from a dataset.')
    parser.add_argument('--file', type=str, default=None, 
                        help='Path to the HDF5 dataset file')
    parser.add_argument('--ip', type=str, default='192.168.1.2',
                        help='Robot arm IP address')
    parser.add_argument('--use_camera', action='store_true',
                        help='Enable camera interface')
    parser.add_argument('--use_logitech_camera', action='store_true',
                        help='Use Logitech camera instead of RealSense')
    parser.add_argument('--freq', type=int, default=30,
                        help='Replay frequency in Hz')
    
    args = parser.parse_args()
    
    # Default file path if none provided
    if args.file is None:
        args.file = os.path.join(os.environ.get('HOME', '/home/nums'), 
                               'projects/ur_bc/data/episode_0.hdf5')
    
    # Configure robot environment
    print(f"Initializing robot environment with IP: {args.ip}")
    print(f"Camera enabled: {args.use_camera}, Using Logitech: {args.use_logitech_camera}")
    
    env = UREnv(
        arm_ip=args.ip, 
        action_type='joint_modbus', 
        has_3f_gripper=False, 
        use_camera=args.use_camera,
        use_logitech_camera=args.use_logitech_camera,
        start_joint_positions=tuple([
            -0.10184682002801448, 
            -1.8316009921757344, 
            2.2237440184163777,
            -1.9278720721999862, 
            -1.5840280733482741, 
            0.04111786366790808
        ])
    )

    # Initialize data replay interface
    data_interface = DataReplayInterface(env=env)
    
    # Start replay
    print(f"Replaying trajectory from file: {args.file}")
    
    data_interface.replayTrajectory(
        hdf5_path=args.file, 
        replay_frequency_hz=args.freq
    )

if __name__ == "__main__":
    main()