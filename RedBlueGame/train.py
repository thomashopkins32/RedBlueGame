'''
train.py

Module for different agent learning algorithms.
'''
import torch
import torch.optim as optim

from game import Game

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


