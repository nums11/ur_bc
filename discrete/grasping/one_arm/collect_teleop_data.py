import os
from pynput import keyboard
from time import sleep
import numpy as np
import urx
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver
from time import sleep
import numpy as np

# Initialize gripper
port = "/dev/ttyUSB0"
gripper = RobotiqModbusRtuDriver(port)
gripper.connect()
gripper.activate()
status = gripper.status()
print("Initialized Gripper. Status:", status)

# Initialize robot
robot = urx.Robot("192.168.2.2")
print("Connected to arm")


trajectory = {}
timestep = 0

def resetRobot():
    print("Resetting gripper")
    gripper.move(pos=0, speed=4, force=1)
    print("Setting robot to start position")
    start_joint__positions = tuple([-0.0012431955565164188, -2.1738289740754846, -2.020386051707992,
                                    0.4159095531371285, -3.4600751680349213, 4.000865030941088])
    robot.movej(start_joint__positions)
    print("Robot reset to start position ------")

def moveRobot(pose, action):
    translation_delta = 0.02
    rotation_delta = 0.1
    gripper_speed = 4
    if action == 0:
        pose[0] += translation_delta
    elif action == 1:
        pose[0] -= translation_delta
    elif action == 2:
        pose[1] -= translation_delta
    elif action == 3:
        pose[1] += translation_delta
    elif action == 4:
        pose[2] += translation_delta
    elif action == 5:
        pose[2] -= translation_delta
    elif action == 6:
        pose[3] += rotation_delta
    elif action == 7:
        pose[3] -= rotation_delta
    elif action == 8:
        pose[4] += rotation_delta
    elif action == 9:
        pose[4] -= rotation_delta
    elif action == 10:
        pose[5] += rotation_delta
    elif action == 11:
        pose[5] -= rotation_delta
    elif action == 12:
        gripper.move(pos=0, speed=gripper_speed, force=1)
        sleep(3)
    elif action == 13:
        gripper.move(pos=140, speed=gripper_speed, force=1)
        sleep(3)
    
    if action in list(range(12)):
        robot.movejInvKin(pose)
 

def getValidActions():
    return ['w','s','d','a','q','e','r','f','z','x','c','v','t','g']

def getDatasetDirectory():
    data_base_dir = '/home/weirdlab/ur_bc/grasping/one_arm/data/'
    num_trajectories = len(os.listdir(data_base_dir))
    dir_name = data_base_dir + 'traj_' + str(num_trajectories) + '/'
    os.mkdir(dir_name)
    return dir_name

def saveSample(data_dir, timestep, obs, action):
    sample = np.array([obs, action], dtype=object)
    filename = data_dir + 'sample_' + str(timestep)
    np.save(filename, sample)

def saveTrajectory():
    global trajectory, timestep
    data_dir = getDatasetDirectory()
    print("Saving trajectory to path", data_dir)
    for timestep, (observation, action) in trajectory.items():
        print(timestep, observation, action)
        saveSample(data_dir, timestep, observation, action)
    print("Finished Saving Trajectory -------")
    resetRobot()
    trajectory = {}
    timestep = 0

def removeTrajectory():
    print("Removing Trajectory")
    global trajectory, timestep
    resetRobot()
    trajectory = {}
    timestep = 0

def gripperClosed():
    return gripper.status().position.po > 10

def on_press(key):
    global trajectory, timestep
    valid_actions = getValidActions()
    if hasattr(key, 'char') :
        if key.char == 'm':
            saveTrajectory()
        elif key.char == 'b':
            removeTrajectory()
        elif key.char in valid_actions:
            pose = robot.get_pose_array()
            action = valid_actions.index(key.char)
            observation = np.append(pose, gripperClosed())
            moveRobot(pose, action)
            # print("joint positions", robot.getj())
            trajectory[timestep] = [observation, action]
            print("t:", timestep, "observation", observation, "action", action)
            timestep += 1
        else:
            print("Invalid action")

print("Keyboard Teleop")
print("x plane: 'w', 's'  y plane: 'a', 'd'   z plane: 'q', 'e'")
resetRobot()
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()
