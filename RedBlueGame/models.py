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


class DQFFN(nn.Module):
    def __init__(self, n):
        '''
        Create Feed-forward Network with n dim input and n dim output
        '''
        super(DQFFN, self).__init__()
        self.n = n
        # input is flattened upper triangle (diagonal included) of
        # state adjacency matrix
        self.l1 = nn.Linear(n*(n+1)//2, 2048)
        #self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(2048, 1024)
        #self.bn2 = nn.BatchNorm1d(64)
        self.l3 = nn.Linear(1024, n)

    def forward(self, x):
        '''
        input is of shape (batch_size, n, n)
        '''
        # convert upper triangle + diagonal to flat tensor
        upper_indices = torch.triu_indices(self.n, self.n)
        x = x[:, upper_indices[0], upper_indices[1]]
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
