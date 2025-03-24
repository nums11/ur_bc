import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataCollectionInterface import DataCollectionInterface
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface
from interfaces.GelloTeleopInterface import GelloTeleopInterface
from environments.UREnv import UREnv
import signal

def main():
    # Initialize variables to None so they can be checked in finally block
    env = None
    teleop_interface = None
    data_interface = None
    
    try:
        # Kitchen setup
        env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, 
                    use_camera=True, use_logitech_camera=True,
                    start_joint_positions=tuple([-0.10184682002801448, -1.8316009921757344, 2.2237440184163777,
                        -1.9278720721999862, -1.5840280733482741, 0.04111786366790808]))
        
        teleop_interface = GelloTeleopInterface(env=env)
        data_interface = DataCollectionInterface(teleop_interface=teleop_interface)
        
        # Start collection (this is blocking)
        data_interface.startDataCollection(remove_zero_actions=False, collection_freq_hz=50)
        
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