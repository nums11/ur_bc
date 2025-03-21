#!/usr/bin/env python3
import argparse
import time
import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.UREnv import UREnv
from interfaces.MelloTeleopInterface import MelloTeleopInterface

def parse_args():
    parser = argparse.ArgumentParser(description="Test MelloTeleopInterface")
    parser.add_argument("--arm_ip", default="192.168.1.2", help="IP address of the UR robot")
    parser.add_argument("--mello_url", default="http://10.19.2.209/", help="URL of the Mello endpoint")
    parser.add_argument("--offsets_file", default="joint_offsets.json", help="Path to the joint offsets file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Initializing UR environment with IP: {args.arm_ip}")
    env = UREnv(
        action_type='joint_modbus',
        arm_ip=args.arm_ip,
        limit_workspace=False,
        start_joint_positions=tuple([0.04476716481195305, -1.642307238840596,
            1.9634761762106852, 4.267337051929634, -1.4365360003497916, 2.3400644905656756]),
        has_3f_gripper=False,
        use_camera=False
    )
    
    print(f"Initializing MelloTeleopInterface with Mello URL: {args.mello_url}")
    teleop = MelloTeleopInterface(
        env=env,
        mello_url=args.mello_url,
        offsets_file=args.offsets_file
    )
    
    print("Starting teleoperation...")
    teleop.start()
        
if __name__ == "__main__":
    main() 