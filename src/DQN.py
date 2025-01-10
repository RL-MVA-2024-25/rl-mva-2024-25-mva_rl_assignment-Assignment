import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, input_size=6, output_size=4,hidden_dim = 1024): # 6 states and 4 actions
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 256)
        self.fc6 = nn.Linear(256, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x