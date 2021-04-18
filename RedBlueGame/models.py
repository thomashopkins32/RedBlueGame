'''
network.py

Module for different neural networks useful for
Deep Q-Learning Agents
'''
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    '''
    Memory for (s, a, s', r) transitions.
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        ''' Saves a transition '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ''' Uniformly sample a batch of transitions '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n):
        '''
        Create CNN initially random with nxn input tensor
        and n dim output tensor.
        '''
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=5, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convn = conv2d_size_out(conv2d_size_out(conv2d_size_out(n)))
        linear_input_size = convn*convn*1
        self.l1 = nn.Linear(linear_input_size, n)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.l1(x.view(x.size(0), -1))