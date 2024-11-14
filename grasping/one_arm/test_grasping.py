import numpy as np
import torch
from torch import nn
import sys
from time import sleep
import urx
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver

# Initialize gripper
port = "/dev/ttyUSB0"
gripper = RobotiqModbusRtuDriver(port)
gripper.connect()
gripper.activate()
status = gripper.status()
gripper.move(pos=0, speed=4, force=1)
print("Initialized Gripper. Status:", status)

# Initialize robot
robot = urx.Robot("192.168.2.2")
print("Connected to arm. Setting to start position ----")
start_joint__positions = tuple([-0.0012431955565164188, -2.1738289740754846, -2.020386051707992,
                                    0.4159095531371285, -3.4600751680349213, 4.000865030941088])
robot.movej(start_joint__positions)
print("Robot reset to start position ------")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device", device)

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
        gripper.move(pos=100, speed=gripper_speed, force=1)
        sleep(3)
    
    if action in list(range(12)):
        robot.movejInvKin(pose)

def gripperClosed():
    return gripper.status().position.po > 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 14)
        )

    def forward(self, x):
        predictions = self.linear_relu_stack(x)
        return predictions
    
model = NeuralNetwork().to(device)
model_path = './bc_model_200_epochs.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

while True:
    pose = robot.get_pose_array()
    observation = np.append(pose, gripperClosed())
    observation = torch.Tensor(observation).to(device)
    print("Observation", observation)
    predictions = model(observation)
    predicted_action = predictions.argmax()
    print("Prediction Action",predicted_action)
    moveRobot(pose, predicted_action)
    sleep(0.2)
