import numpy as np
import torch
from torch import nn
import sys
from time import sleep
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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device", device)

def moveRobot(pose, action):
    robot = None
    if action in range(6):
        robot = left_arm
    else:
        robot = right_arm
    pose = robot.get_pose_array()
    if action == 0:
        pose[0] -= 0.02
    elif action == 1:
        pose[0] += 0.02
    elif action == 2:
        pose[1] += 0.02
    elif action == 3:
        pose[1] -= 0.02
    elif action == 4:
        pose[2] += 0.02
    elif action == 5:
        pose[2] -= 0.02
    elif action == 6:
        pose[0] += 0.02
    elif action == 7:
        pose[0] -= 0.02
    elif action == 8:
        pose[1] -= 0.02
    elif action == 9:
        pose[1] += 0.02
    elif action == 10:
        pose[2] += 0.02
    elif action == 11:
        pose[2] -= 0.02
    robot.movejInvKin(pose)

def getJointPositions():
    leftj = left_arm.getj()
    rightj = right_arm.getj()
    return leftj + rightj

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 12)
        )

    def forward(self, x):
        predictions = self.linear_relu_stack(x)
        return predictions
    
model = NeuralNetwork().to(device)
model_path = './bc_model_2_arms_state.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

while True:
    pose = getJointPositions()
    observation = torch.Tensor(pose).to(device)
    print("Observation", observation)
    predictions = model(observation)
    predicted_action = predictions.argmax()
    print("Prediction Action",predicted_action)
    moveRobot(pose, predicted_action)
    # sleep(0.2)
