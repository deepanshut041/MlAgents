import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQNLinear(nn.Module):
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(DDQNLinear, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.advantage = nn.Linear(64, action_size)
        self.value = nn.Linear(64, 1)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage  - advantage.mean()