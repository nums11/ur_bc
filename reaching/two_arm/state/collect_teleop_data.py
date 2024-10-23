import sys
import os
from pynput import keyboard
from time import sleep
import numpy as np
import urx

left_arm = urx.Robot("192.168.1.2")
print("Connected to left arm")
right_arm = urx.Robot("192.168.2.2")
print("Connected to right arm")
print("Setting left-arm to start position")
left_start_joint__positions = tuple([-0.028764690017641925, -1.6407542814320468,
    2.2395414191558425, -0.28259952827121754, 0.6279560199271199, 0.142008067713095])
left_arm.movej(left_start_joint__positions, acc=0.5, vel=0.5)
print("Setting right-arm to start position")
right_start_joint__positions = tuple([-0.1309072295818936, -1.5460030301533012,
    -2.182326690539821, -0.6676762594523065, -4.574149040532557, 2.8684232724884806])
right_arm.movej(right_start_joint__positions, acc=0.5, vel=0.5)
trajectory = {}
timestep = 0

def moveRobot(action):
    move_increment = 0.02
    robot = None
    if action in range(6):
        robot = left_arm
    else:
        robot = right_arm
    pose = robot.get_pose_array()

    if action == 0:
        pose[0] -= move_increment
    elif action == 1:
        pose[0] += move_increment
    elif action == 2:
        pose[1] -= move_increment
    elif action == 3:
        pose[1] += move_increment
    elif action == 4:
        pose[2] += move_increment
    elif action == 5:
        pose[2] -= move_increment
    elif action == 6:
        pose[0] += move_increment
    elif action == 7:
        pose[0] -= move_increment
    elif action == 8:
        pose[1] += move_increment
    elif action == 9:
        pose[1] -= move_increment
    elif action == 10:
        pose[2] += move_increment
    elif action == 11:
        pose[2] -= move_increment
    robot.movejInvKin(pose)

def getValidActions():
    return ['w','s','a','d','r','f','i','k','j','l','y','h',]

def getDatasetDirectory():
    data_base_dir = '/home/weirdlab/ur_bc/two_arm_state/data/'
    num_trajectories = len(os.listdir(data_base_dir))
    dir_name = data_base_dir + 'traj_' + str(num_trajectories) + '/'
    os.mkdir(dir_name)
    return dir_name

def saveSample(data_dir, timestep, obs, action):
    sample = np.array([obs, action], dtype=object)
    filename = data_dir + 'sample_' + str(timestep)
    np.save(filename, sample)

def saveTrajectory():
    data_dir = getDatasetDirectory()
    print("Saving trajectory to path", data_dir)
    for timestep, (observation, action) in trajectory.items():
        print(timestep, observation, action)
        saveSample(data_dir, timestep, observation, action)
    print("Finished Saving Trajectory -------")

def getJointPositions():
    leftj = left_arm.getj()
    rightj = right_arm.getj()
    return leftj + rightj

def on_press(key):
    global trajectory, timestep
    valid_actions = getValidActions()
    if hasattr(key, 'char') :
        if key.char == 'x':
            saveTrajectory()
        elif key.char in valid_actions:
            action = valid_actions.index(key.char)
            moveRobot(action)
            observation = getJointPositions()
            trajectory[timestep] = [observation, action]
            print("t:", timestep, "observation", observation, "action", action)
            timestep += 1
        else:
            print("Invalid action")

print("Keyboard Teleop")
print("Left arm: 'w', 'a', 's', 'd', 'r', 'f'")
print("Right arm: 'i', 'j', 'k', 'l', 'y', 'hs'")
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()
