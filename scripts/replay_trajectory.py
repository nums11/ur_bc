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
    parser.add_argument('--use_camera', default=True,
                        help='Enable camera interface')
    parser.add_argument('--use_logitech_camera', default=True,
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
    
    try:

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

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up all resources
        print("Performing cleanup...")
        
        # Stop data collection if running
        if data_interface and hasattr(data_interface, 'keyboard_listener'):
            data_interface.keyboard_listener.stop()
            print("Stopped keyboard listener")
        
        # Clean up environment
        if env:
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

if __name__ == "__main__":
    main()