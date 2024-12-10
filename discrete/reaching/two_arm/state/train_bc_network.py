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

def oneHotEncode(labels):
    num_actions = 12
    one_hot_labels = []
    for label in labels:
        one_hot = np.zeros(num_actions)
        one_hot[label] = 1
        one_hot_labels.append(one_hot)
    return one_hot_labels

features = []
labels = []
trajectories = os.listdir('./data')
for trajectory in trajectories:
    traj_path = './data/' + trajectory + '/'
    sample_names = os.listdir(traj_path)
    for sample_name in sample_names:
        sample_path = traj_path + sample_name
        sample = np.load(sample_path, allow_pickle=True)
        observation, action = sample
        features.append(observation)
        labels.append(action)

labels = oneHotEncode(labels)

test_set_size = 5
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

batch_size = 4
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
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model --------------\n", model, "\n# Trainable Params:", num_trainable_params)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

""" Train """

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

epochs = 50
metrics = {
    'training_losses': [],
    'test_losses': [],
    'training_accuracies': [],
    'test_accuracies': [],
}
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    evaluate(train_dataloader, model, loss_fn, metrics)
    evaluate(test_dataloader, model, loss_fn, metrics, is_test_set=True)
print("Finished Training")

""" Plot Results """

epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, metrics['training_accuracies'], label='Training Accuracy')
plt.plot(epochs_range, metrics['test_accuracies'], label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, metrics['training_losses'], label='Training Loss')
plt.plot(epochs_range, metrics['test_losses'], label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

""" Save Model """

torch.save(model.state_dict(), "bc_model_2_arms_state.pth")
print("Saved PyTorch Model State")