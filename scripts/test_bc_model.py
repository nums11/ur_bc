import torch
from torch import nn
from URnterface import URInterface

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device", device)

right_arm = URInterface('192.168.2.2',
                        tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                -1.095669650507004, -4.386985456001609, 3.2958897411425156]),
                                has_3f_gripper=True)
left_arm = URInterface('192.168.1.2',
                       tuple([0.10010272221997439, -1.313795512335239, 2.1921907366841067,
                                            3.7562696849438524, 1.2427944188620925, 0.8873570727182682]))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        predictions = self.linear_relu_stack(x)
        return predictions
    

model = NeuralNetwork().to(device)
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model --------------\n", model, "\n# Trainable Params:", num_trainable_params)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with torch.inference_mode():
    observation = getBCObservation(obs_dict)
    print("observation", observation)
    predictions = model(observation)
    action = predictions.argmax()
    print("Predictions", predictions)
    print("Using Predicted Action", action)
    action_tensor = convertActionToTensor(action, env)
    obs_dict, rewards, terminated, truncated, info = env.step(action_tensor)
