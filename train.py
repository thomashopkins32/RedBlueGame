'''
train.py

Script for training Deep Q-Network based Agents

'''
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import copy
import sys

from agents import GreedyAgent, DifferenceAgent, RandomAgent, DQNAgent
from game import Game
from models import DQFFN, ReplayMemory, Transition


DEBUG_MODE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256    # sampling size from replay memory
GAMMA = 0.5         # discount factor for estimated future rewards
EPS_START = 0.9     # starting exploration probability
EPS_END = 0.05      # ending exploration probability
EPS_DECAY = 500     # rate of decay of exploration probability
TARGET_UPDATE = 100 # number of episodes between target network update
I_EPISODE = 0       # current episode number
NUM_EPISODES = 1000 # number of games to learn from
N = 15              # number of nodes in the game graph
OPPONENT_TYPE = 'RandomAgent' # opponent to play against TODO: implement 'self'
MEMORY_SIZE = 10000 # size of replay memory


def select_opponent(op_type):
    if op_type == 'random':
        select = np.random.randint(0, 3)
        if select == 0:
            return GreedyAgent()
        if select == 1:
            return DifferenceAgent()
        if select == 2:
            return RandomAgent()
    elif op_type == 'GreedyAgent':
        return GreedyAgent()
    elif op_type == 'DifferenceAgent':
        return DifferenceAgent()
    elif op_type == 'RandomAgent':
        return RandomAgent()
    return None


def optimize_model():
    # not enough transitions available to train yet
    if len(memory) < BATCH_SIZE:
        return
    # sample a batch of transitions
    transitions = memory.sample(BATCH_SIZE)
    if DEBUG_MODE:
        assert len(transitions) == BATCH_SIZE
    # form batch using Transition tuple
    batch = Transition(*zip(*transitions))

    # generate binary mask for non game ending states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    if DEBUG_MODE:
        print(f'DEBUG: non_final_mask shape: {non_final_mask.shape}')
        print(f'DEBUG: non_final_mask[0]: {non_final_mask[0]}')
        assert non_final_mask.shape == (BATCH_SIZE, 1)

    # generate all non game ending states (for reward prediction in target network)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    if DEBUG_MODE:
        print(f'DEBUG: non_final_next_states shape: {non_final_next_states.shape}')
        print(f'DEBUG: non_final_next_states[0]: {non_final_next_states[0]}')
        assert non_final_next_states.shape == (BATCH_SIZE, N, N)

    # setup separate batches from transitions
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    if DEBUG_MODE:
        print(f'DEBUG: state_batch shape: {state_batch.shape}')
        print(f'DEBUG: state_batch[0]: {state_batch[0]}')
        print(f'DEBUG: action_batch shape: {action_batch.shape}')
        print(f'DEBUG: action_batch[0]: {action_batch[0]}')
        print(f'DEBUG: reward_batch shape: {reward_batch.shape}')
        print(f'DEBUG: reward_batch[0]: {reward_batch[0]}')
        assert state_batch.shape == (BATCH_SIZE, N, N)
        assert action_batch.shape == (BATCH_SIZE, 1)
        assert reward_batch.shape == (BATCH_SIZE, 1)

    # get predicted Q-values from policy network
    preds = policy_net(state_batch).to(device)
    if DEBUG_MODE:
        print(f'DEBUG: preds shape: {preds.shape}')
        print(f'DEBUG: preds[0]: {preds[0]}')
        assert preds.shape == (BATCH_SIZE, N)

    # get only the Q-value of the action selected
    state_action_values = preds.gather(1, action_batch)
    if DEBUG_MODE:
        print(f'DEBUG: state_action_values shape: {state_action_values.shape}')
        print(f'DEBUG: state_action_values[0]: {state_action_values[0]}')
        assert state_action_values.shape == (BATCH_SIZE, 1)

    # get target Q-values of next state
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    targets = target_net(non_final_next_states).to(device)
    if DEBUG_MODE:
        print(f'DEBUG: targets shape: {targets.shape}')
        print(f'DEBUG: targets[0]: {targets[0]}')
        assert targets.shape == (BATCH_SIZE, N)

    # determine which next actions are invalid
    invalid_actions = torch.diagonal(non_final_next_states, axis1=1, axis2=2).copy().view(-1,N)
    invalid_indices = invalid_actions.nonzero()
    if DEBUG_MODE:
        print(f'DEBUG: invalid_actions shape: {invalid_actions.shape}')
        print(f'DEBUG: invalid_actions[0]: {invalid_actions[0]}')
        assert invalid_actions.shape == (BATCH_SIZE, N)
        print(f'DEBUG: invalid_indices shape: {invalid_indices.shape}')
        print(f'DEBUG: invalid_indices[0]: {invalid_indices[0]}')

    # make it so the network never chooses the invalid action (max operation is performed next)
    targets[invaid_indices] = float('-inf')
    if DEBUG_MODE:
        print(f'DEBUG: targets after invalid shape: {targets.shape}')
        print(f'DEBUG: targets after invalid [0]: {targets[0]}')
        assert targets.shape == (BATCH_SIZE, N)

    next_state_values[non_final_mask] = target_outs.max(1)[0].detach()
    if DEBUG_MODE:
        print(f'DEBUG: next_state_action_values shape: {next_state_action_values.shape}')
        print(f'DEBUG: next_state_action_values[0]: {next_state_action_values[0]}')
        assert next_state_action_values.shape == (BATCH_SIZE, 1)

    expected_state_action_values = (next_state_values*GAMMA) + reward_batch
    if DEBUG_MODE:
        print(f'DEBUG: expected_state_action_values shape: {expected_state_action_values.shape}')
        print(f'DEBUG: expected_state_action_values[0]: {expected_state_action_values[0]}')
        assert expected_state_action_values.shape == (BATCH_SIZE, 1)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    if DEBUG_MODE:
        print(f'DEBUG: loss shape: {loss.shape}')
        print(f'DEBUG: loss item: {loss.item()}')
        assert loss.shape == (1)







agent = DQNAgent(N, training=True, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)
policy_net = agent.model.to(device)
target_net = DQFFN(N).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if DEBUG_MODE:
    policy_params = [param for param in policy_net.parameters()]
    target_params = [param for param in target_net.parameters()]
    for i in range(len(policy_params)):
        assert torch.eq(policy_params[i], target_params[i])


