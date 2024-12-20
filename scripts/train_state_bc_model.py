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

print(traj_filenames)
for filename in traj_filenames:
    traj_path = data_dir + filename
    traj = dict(np.load(traj_path, allow_pickle=True).items())
    for t, [obs, action] in traj.items():
        left_arm_j, (right_arm_j, gripper) = obs
        combined_obs = np.append(np.concatenate((left_arm_j, right_arm_j)), gripper)
        left_arm_action = action['left_arm']
        right_arm_action = action['right_arm']
        combined_action = np.concatenate((left_arm_action, right_arm_action))
        features.append(combined_obs)
        labels.append(combined_action)

test_set_size = 30
test_features = []
test_labels = []
for _ in range(test_set_size):
    index = randrange(len(features))
    test_features.append(features.pop(index))
    test_labels.append(labels.pop(index))

x = torch.Tensor(features)
test_x = torch.Tensor(test_features)
y = torch.Tensor(labels)
test_y = torch.Tensor(test_labels)
print("x:", x.shape, "test_x", test_x.shape, "y", y.shape, "test_y", test_y.shape)

batch_size = 8
train_dataset = TensorDataset(x,y) # create your datset
test_dataset = TensorDataset(test_x,test_y) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size) # create your dataloader
test_dataloader = DataLoader(test_dataset, batch_size) # create your dataloader

# """ Define Model """

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12)
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

def evaluate(dataloader, model, loss_fn, metrics, is_test_set=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    set_str = None
    if is_test_set:
        metrics['test_losses'].append(test_loss)
        metrics['test_accuracies'].append(round(100*correct, 2))
        set_str = "Test Set ---- "
    else:
        metrics['training_losses'].append(test_loss)
        metrics['training_accuracies'].append(round(100*correct, 2))
        set_str = "Training Set ---- "
    print(set_str + f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

epochs = 10
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
torch.save(model.state_dict(), model_dir + 'state_bc_model_10_demos.pth')
print("Saved PyTorch Model State")

""" Plot Results """

epochs_range = range(epochs)
plt.plot(epochs_range, metrics['training_losses'], label='Training Loss')
plt.plot(epochs_range, metrics['test_losses'], label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()

