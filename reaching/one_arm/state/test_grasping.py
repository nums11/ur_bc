import numpy as np
import torch
from torch import nn
import sys
from time import sleep
import urx

robot = urx.Robot("192.168.1.2")
print("Connected to arm. Setting to start position ----")
start_joint__positions = tuple([-0.4877165042104066, -1.593502988132193,
    2.1684843291414904, 0.33130235142425873, 0.23414970368740268, -0.4921767767939169])
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
    move_increment = 0.02
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
    robot.movejInvKin(pose)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        predictions = self.linear_relu_stack(x)
        return predictions
    
model = NeuralNetwork().to(device)
model_path = './bc_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

while True:
    pose = robot.get_pose_array()
    observation = torch.Tensor(pose).to(device)
    print("Observation", observation)
    predictions = model(observation)
    predicted_action = predictions.argmax()
    print("Prediction Action",predicted_action)
    moveRobot(pose, predicted_action)
    sleep(0.2)
