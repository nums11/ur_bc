import numpy as np
from URnterface import URInterface
import threading

data_filepath = "/home/weirdlab/ur_bc/data/traj_1.npz"
data = dict(np.load(data_filepath, allow_pickle=True).items())

right_arm = URInterface('192.168.2.2',
                        tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                -1.095669650507004, -4.386985456001609, 3.2958897411425156]),
                                has_robotiq_gripper=True)
left_arm = URInterface('192.168.1.2',
                       tuple([0.10010272221997439, -1.313795512335239, 2.1921907366841067,
                                            3.7562696849438524, 1.2427944188620925, 0.8873570727182682]))
right_arm.resetPosition()
left_arm.resetPosition()


def armMovementThread(arm, joint_positions, gripper=None):
    print("Moving arm")
    arm.movej(joint_positions)
    if gripper is not None:
        arm.moveRobotiqGripper(gripper)

traj_len = len(data)
for t in range(traj_len):
    print("Timestep", t)
    obs, action = data[str(t)]
    left_arm_obs, right_arm_obs = obs
    left_arm_j = left_arm_obs
    right_arm_j, right_arm_gripper = right_arm_obs

    right_arm_thread = threading.Thread(
        target=armMovementThread, args=(right_arm, right_arm_j, right_arm_gripper))
    left_arm_thread = threading.Thread(
        target=armMovementThread, args=(left_arm, left_arm_j))
    right_arm_thread.start()
    left_arm_thread.start()
    right_arm_thread.join()
    left_arm_thread.join()    