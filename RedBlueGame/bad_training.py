'''
train.py

Module for different agent learning algorithms.
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
from models import DQN, DQFFN, ReplayMemory, Transition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 100
I_EPISODE = 0
LOSSES = []
EPISODES = []
REWARDS = []
# number of game nodes
N = 51
NUM_EPISODES = 5000
OPPONENT_TYPE = 'RandomAgent'

NETWORK = 'ffn'
agent = DQNAgent(N, training=True, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, 
                 device=device, model=NETWORK)
policy_net = agent.model.to(device)
target_net = None
if NETWORK == 'ffn':
    target_net = DQFFN(N).to(device)
else:
    target_net = DQN(N).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)


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
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # print(f'batch: {batch}')
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    outs = policy_net(state_batch).to(device)
    # print(f'outs: {outs}')
    # colored_nodes = np.diagonal(state_batch, axis1=2, axis2=3).copy().reshape(-1, N)
    # print(f'colored: {colored_nodes}')
    #colored_nodes[colored_nodes == -1] = 1
    #invalid_nodes = colored_nodes.nonzero()
    # print(f'invalid_nodes: {invalid_nodes}')
    #outs[invalid_nodes] = 0.0
    # print(f'outs after: {outs}')
    state_action_values = outs.gather(1, action_batch)
    # print(f'Q-values: {state_action_values}')

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    target_outs = target_net(non_final_next_states).to(device)
    # print(f'targets: {target_outs}')
    colored_next = np.diagonal(non_final_next_states, axis1=2, axis2=3).copy().reshape(-1,N)
    invalid_next = colored_next.nonzero()
    # print(f'invalid_next: {invalid_next}')
    target_outs[invalid_next] = -float('inf')
    # print(f'target outs: {target_outs}')
    next_state_values[non_final_mask] = target_outs.max(1)[0].detach()
    # print(f'next state Q-values: {next_state_values}')

    expected_state_action_values = (next_state_values*GAMMA) + reward_batch
    REWARDS.append(torch.mean(expected_state_action_values).item())
    bn = nn.BatchNorm1d(1, track_running_stats=False)
    expected_state_action_values = bn(expected_state_action_values.unsqueeze(1))
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    LOSSES.append(loss.item())
    EPISODES.append(I_EPISODE)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_won = 0
for I_EPISODE in tqdm(range(NUM_EPISODES), total=NUM_EPISODES):
    game = Game(N, 10, 10, verbose=False)
    opponent = select_opponent(OPPONENT_TYPE)
    player = 'blue'
    if np.random.rand() < 0.5:
        game.set_player(agent) # agent is red player
        game.set_player(opponent) # opponent is blue player
        player = 'red'
    else:
        game.set_player(opponent)
        game.set_player(agent)
    result = 'continue'
    s = torch.tensor(game.state.to_numpy(color_pref=player), device=device).reshape(1,1,N,N)
    sp = s
    while result == 'continue':
        # take a step and get reward (actions are handled within DQNAgent)
        blue_action, red_action, result = game.step()
        if player == 'blue':
            action = blue_action
        else:
            # game continued but you did not get to take an action
            # since blue player goes first (just keep prev action)
            if red_action != -1:
                action = red_action
        red_nodes = len(game.state.get_nodes(color='red'))
        blue_nodes = len(game.state.get_nodes(color='blue'))
        if result == 'red' or result == 'blue':
            if player == 'blue':
                reward = blue_nodes - red_nodes
            elif player == 'red':
                reward = red_nodes - blue_nodes
        else:
            reward = 0.0
        if reward > 0.0:
            num_won += 1 
        reward = torch.tensor([reward], device=device)
        action = torch.tensor([[action]], device=device)
        # observe new state
        if result == 'continue':
            sp = torch.tensor(game.state.to_numpy(color_pref=player), device=device).reshape(1,1,N,N)
        else:
            sp = None
        # add transition to replay memory
        memory.push(s, action, sp, reward)
        # move to the next state
        s = sp
        # optimize for one step
        optimize_model()
    # update target network
    if I_EPISODE % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
print(num_won)

torch.save(policy_net.state_dict(), sys.argv[1])
plt.plot(EPISODES, LOSSES)
plt.savefig('losses.png')
plt.close()
plt.plot(EPISODES, REWARDS)
plt.savefig('rewards.png')
plt.close()
