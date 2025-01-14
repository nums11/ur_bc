import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randrange

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device", device)

"""Load Data"""

features = []
labels = []
data_dir = '/home/weirdlab/ur_bc/data/'
traj_filenames = os.listdir(data_dir)
for traj_filename in traj_filenames:
    traj_path = data_dir + traj_filename
    traj = dict(np.load(traj_path, allow_pickle=True).items())
    sorted_timesteps = sorted(traj.keys(), key=lambda x: int(x))
    # print("Len before", len(sorted_timesteps))
    # Bring traj back to 10hz
    sorted_timesteps = sorted_timesteps[2::3]  # Start at index 2 (3rd item) and step by 3
    # print("Len after", len(sorted_timesteps))
    for i, t in enumerate(sorted_timesteps):
        if i == len(sorted_timesteps) - 1:
            continue
        obs, _ = traj[t]
        next_obs, _ = traj[sorted_timesteps[i+1]]
        left_arm_j = obs['left_arm_j']
        left_obs_gripper = np.expand_dims(obs['left_gripper'], axis=0)
        next_left_arm_j = next_obs['left_arm_j']
        next_left_obs_gripper = np.expand_dims(next_obs['left_gripper'], axis=0)
        concat_obs = np.concatenate((left_arm_j, left_obs_gripper))
        concat_action = np.concatenate((next_left_arm_j, next_left_obs_gripper))
        features.append(concat_obs)
        labels.append(concat_action)

    # for obs, action in traj.values():
    #     left_arm_j = obs['left_arm_j']
    #     # right_arm_j = obs['right_arm_j']
    #     left_arm_delta = action['left_arm_delta']
    #     # right_arm_delta = action['right_arm_delta']
    #     # Convert gripper values into arrays so they can later be concatenated
    #     left_obs_gripper = np.expand_dims(obs['left_gripper'], axis=0)
    #     # right_obs_gripper = np.expand_dims(obs['right_gripper'], axis=0)
    #     left_action_gripper = np.expand_dims(action['left_gripper'], axis=0)
    #     # right_action_gripper = np.expand_dims(action['right_gripper'], axis=0)
    #     # concatenate into a 14d observation (12 joints + 2 gripper values)
    #     # concat_obs = np.concatenate((left_arm_j, left_obs_gripper, right_arm_j, right_obs_gripper))
    #     concat_obs = np.concatenate((left_arm_j, left_obs_gripper))
    #     #concatenate into a 14d action (12 pose values + 2 gripper values)
    #     # concat_action = np.concatenate((left_arm_delta, left_action_gripper, right_arm_delta, right_action_gripper))
    #     concat_action = np.concatenate((left_arm_delta, left_action_gripper))
    #     # Add concatenated observations and actions to the dataset
    #     features.append(concat_obs)
    #     labels.append(concat_action)

# Create a test set
test_set_size = 30
test_features = []
test_labels = []
for _ in range(test_set_size):
    index = randrange(len(features))
    test_features.append(features.pop(index))
    test_labels.append(labels.pop(index))

# Create torch tensors from the features and labels
x = torch.Tensor(features)
test_x = torch.Tensor(test_features)
y = torch.Tensor(labels)
test_y = torch.Tensor(test_labels)

batch_size = 8
train_dataset = TensorDataset(x,y) # create your datset
test_dataset = TensorDataset(test_x,test_y) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size) # create your dataloader
test_dataloader = DataLoader(test_dataset, batch_size) # create your dataloader

""" Define Model """

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
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model --------------\n", model, "\n# Trainable Params:", num_trainable_params)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

""" Train """

def datasetPass(dataloader, is_train, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    num_batches = len(dataloader)
    return total_loss / num_batches

epochs = 15
metrics = {
    'training_losses': [],
    'test_losses': [],
}
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train_loss = datasetPass(train_dataloader, True, model, loss_fn, optimizer)
    test_loss = datasetPass(test_dataloader, False, model, loss_fn, optimizer)
    print("Training Loss:", train_loss)
    print("Test Loss:", test_loss)
    metrics['training_losses'].append(train_loss)
    metrics['test_losses'].append(test_loss)


print("Finished Training")

""" Save Model """

model_dir = '/home/weirdlab/ur_bc/models/'
torch.save(model.state_dict(), model_dir + 'cont_bc_model_1_arm_35_demos_10hz_15_epochs_joint.pth')
print("Saved PyTorch Model State")

""" Plot Results """

epochs_range = range(epochs)
plt.plot(epochs_range, metrics['training_losses'], label='Training Loss')
plt.plot(epochs_range, metrics['test_losses'], label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()

