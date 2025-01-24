from deprecated.OldContKeyboardTeleopInterface import ContKeyboardTeleopInterface
import torch
from torch import nn
import numpy as np
from time import sleep
import threading

def convertTeleopObsToModelObs(obs):
    left_arm_j = obs['left_arm_j']
    right_arm_j = obs['right_arm_j']
    left_obs_gripper = np.expand_dims(obs['left_gripper'], axis=0)
    right_obs_gripper = np.expand_dims(obs['right_gripper'], axis=0)
    # return np.concatenate((left_arm_j, left_obs_gripper, right_arm_j, right_obs_gripper))
    return np.concatenate((left_arm_j, left_obs_gripper))

# For ee delta
# def armMovementThread(arm, delta, gripper=None):
#     # delta = np.zeros(6)
#     # delta[5] = 0.01
#     pose = arm.getPose()
#     pose += delta
#     arm.movejInvKin(pose)
#     if gripper is not None:
#         if gripper < 0.5:
#             gripper = False
#         else:
#             gripper = True
#         arm.moveRobotiqGripper(gripper)

def armMovementThread(arm, joint_positions, gripper=None):
    arm.movej(joint_positions)
    print("Received gripper", gripper)
    if gripper is not None:
        if gripper < 0.4:
            gripper = False
        else:
            gripper = True
            print("Should close gripper! --------")
        arm.moveRobotiqGripper(gripper)
python examples/train_bc_rnn.py            print("Waiting for gripper to close")
            sleep(1)



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device", device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        predictions = self.linear_relu_stack(x)
        return predictions
    

model = NeuralNetwork().to(device)
model_path = '/home/weirdlab/ur_bc/models/cont_simple_pick_model_jpdelta_35_demos_3_epochs_no_zero_actions_noisy_d7.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

teleop = ContKeyboardTeleopInterface()
teleop.startTeleop()
sleep(2)

while True:
    obs, _ = teleop.getLastObsAndAction()
    # print(obs)
    model_obs = convertTeleopObsToModelObs(obs)
    model_obs = torch.Tensor(model_obs).to(device)
    print("model obs", model_obs)
    # sleep(100)
    predictions = model(model_obs).detach().cpu().numpy()
    print("predictions", predictions)
    # left_arm_delta = predictions[:6]
    # left_gripper = predictions[6]

    # new_left_arm_j = predictions[:6]
    current_left_arm_j = model_obs[:6].detach().cpu().numpy()
    joint_delta = predictions[:6]
    new_left_arm_j = np.add(current_left_arm_j, joint_delta)
    new_left_gripper = predictions[6]
    print("new_left_gripper before", new_left_gripper)
    new_left_gripper /= 0.02
    print("new_left_gripper after", new_left_gripper)

    # For joint delta
    # right_arm_delta = predictions[7:13]
    # right_arm_delta = np.zeros(6)
    # right_gripper = predictions[13]
    # # print("predcitions", predictions)
    # print("left_arm_delta", left_arm_delta)
    # print("left_gripper", left_gripper)
    # print("right_arm_delta", right_arm_delta)
    # print("right_gripper", right_gripper)
    left_arm_thread = threading.Thread(target=armMovementThread,
                                    args=(teleop.left_arm, new_left_arm_j, new_left_gripper))

    # right_arm_thread = threading.Thread(target=armMovementThread,
    #                                 args=(teleop.right_arm, right_arm_delta, right_gripper))
    left_arm_thread.start()
    # right_arm_thread.start()
    left_arm_thread.join()
    # right_arm_thread.join()
    # print(predictions)
    # sleep(20)