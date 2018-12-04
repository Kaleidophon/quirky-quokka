"""
Define DQN and Double DQN for this project.
"""

# STD
import random

# EXT
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, n_in, n_out, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n_in, num_hidden)
        self.l2 = nn.Linear(num_hidden, n_out)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        return out


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if self.capacity == len(self.memory):
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
