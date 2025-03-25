import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.GelloTeleopInterface import GelloTeleopInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

def main():
    try:
        env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus',
                    has_3f_gripper=False, use_camera=False, use_current_joint_positions=True)

        teleop = GelloTeleopInterface(env=env)
        teleop.start()
        
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
