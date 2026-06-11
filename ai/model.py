import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    """Deep Q-Network mit 2 Hidden Layers"""
    
    def __init__(self, state_size=13, action_size=4, learning_rate=0.001):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
        self.relu = nn.ReLU()
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_step(self, state, target):
        output = self(state)
        loss = self.loss_fn(output, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
