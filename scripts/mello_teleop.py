#!/usr/bin/env python3
import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces.MelloTeleopInterface import MelloTeleopInterface
from environments.UREnv import UREnv

def main():
    env = None
    teleop = None
    
    try:
        # Initialize the UR environment
        env = UREnv(
            arm_ip='192.168.1.2',
            action_type='joint_modbus',
            has_3f_gripper=False,
            use_camera=False,
            start_joint_positions=tuple([-0.10184682002801448, -1.8316009921757344, 2.2237440184163777,
                        -1.9278720721999862, -1.5840280733482741, 0.04111786366790808])
        )

        # Initialize and start the teleop interface
        teleop = MelloTeleopInterface(env=env)
        teleop.start()
        
        # Keep the main thread alive
        print("Teleop running. Press Ctrl+C to stop...")
        while True:
            # Sleep to prevent CPU overuse
            # The teleop thread will handle the control loop
            sleep(1)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up all resources
        print("Performing cleanup...")
        
        # Clean up teleop interface
        if teleop:
            try:
                teleop.cleanup()
                print("Cleaned up teleop interface")
            except Exception as e:
                print(f"Error cleaning up teleop: {e}")
        
        # Clean up environment
        if env:
            # Stop any running threads
            env.resetting = True
            
            # Close robot connection
            if hasattr(env, 'arm') and hasattr(env.arm, 'close'):
                try:
                    env.arm.close()
                    print("Closed robot arm connection")
                except Exception as e:
                    print(f"Error closing arm connection: {e}")
                    
        print("Cleanup complete")

if __name__ == "__main__":
    from time import sleep
    main() 