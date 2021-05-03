'''
train.py

Script for training Deep Q-Network based Agents

Deep Q-Learning can be difficult so there is a lot of debugging code
to make sure that everything works as intended.

TODO: Move debugging code to unit testing
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


DEBUG_MODE = False
if DEBUG_MODE:
    DEBUG_OFFSET = 1500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPERPARAMETERS
BATCH_SIZE = 512    # sampling size from replay memory
GAMMA = 0.75         # discount factor for estimated future rewards
EPS_START = 0.99     # ftarting exploration probability
EPS_END = 0.1      # ending exploration probability
EPS_DECAY = 3000000     # rate of linear decay of exploration probability
TARGET_UPDATE = 3000 # number of episodes between target network update
I_EPISODE = 0       # current episode number
NUM_EPISODES = 500000 # number of games to learn from
N = 31               # number of nodes in the game graph
OPPONENT_TYPE = 'RandomAgent' # opponent to play against TODO: implement 'self'
MEMORY_SIZE = 500000 # size of replay memory
LEARNING_RATE = 0.0001 # learning rate for optimizer
DOUBLE_DQN = True     # use Double DQN target (reduces overestimation bias)

# Statistics
EPISODES = []
LOSSES = []
AVG_EXPECTED_REWARDS = []
REWARDS = []

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
    elif op_type == 'order':
        if I_EPISODE < NUM_EPISODES // 3:
            return RandomAgent()
        elif I_EPISODE < 2*NUM_EPISODES // 3:
            return DifferenceAgent()
        else:
            return GreedyAgent()
    return None


# define agents and networks
agent = DQNAgent(N, training=True, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, device=device)
policy_net = agent.model.to(device)
target_net = DQFFN(N).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)
if DEBUG_MODE:
    # are the parameters the same?
    policy_params = [param for param in policy_net.parameters()]
    target_params = [param for param in target_net.parameters()]
    for i in range(len(policy_params)):
        assert torch.equal(policy_params[i], target_params[i])


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
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: non_final_mask shape: {non_final_mask.shape}')
        print(f'DEBUG: non_final_mask[0]: {non_final_mask[0]}')
        assert non_final_mask.shape == (BATCH_SIZE,)

    # generate all non game ending states (for reward prediction in target network)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: non_final_next_states shape: {non_final_next_states.shape}')
        print(f'DEBUG: non_final_next_states[0]: {non_final_next_states[0]}')
        assert non_final_next_states.shape == (torch.count_nonzero(non_final_mask), N, N)

    # setup separate batches from transitions
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: state_batch shape: {state_batch.shape}')
        print(f'DEBUG: state_batch[0]: {state_batch[0]}')
        print(f'DEBUG: action_batch shape: {action_batch.shape}')
        print(f'DEBUG: action_batch[0]: {action_batch[0]}')
        print(f'DEBUG: reward_batch shape: {reward_batch.shape}')
        print(f'DEBUG: reward_batch[0]: {reward_batch[0]}')
        assert state_batch.shape == (BATCH_SIZE, N, N)
        assert action_batch.shape == (BATCH_SIZE, 1)
        assert reward_batch.shape == (BATCH_SIZE,)

    # get predicted Q-values from policy network
    preds = policy_net(state_batch).to(device)
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: preds shape: {preds.shape}')
        print(f'DEBUG: preds[0]: {preds[0]}')
        assert preds.shape == (BATCH_SIZE, N)

    # get only the Q-value of the action selected
    state_action_values = preds.gather(1, action_batch)
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: state_action_values shape: {state_action_values.shape}')
        print(f'DEBUG: state_action_values[0]: {state_action_values[0]}')
        assert state_action_values.shape == (BATCH_SIZE, 1)

    # get target Q-values of next state
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # select action according to policy network
    if DOUBLE_DQN:
        next_preds = policy_net(non_final_next_states).to(device)
    targets = target_net(non_final_next_states).to(device)
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: targets shape: {targets.shape}')
        print(f'DEBUG: targets[0]: {targets[0]}')
        assert targets.shape == (torch.count_nonzero(non_final_mask), N)
        if DOUBLE_DQN:
            print(f'DEBUG: next_preds shape: {next_preds.shape}')
            print(f'DEBUG: next_preds[0]: {next_preds[0]}')
            assert next_preds.shape == (torch.count_nonzero(non_final_mask), N)


    # determine which next actions are invalid
    invalid_actions = torch.diagonal(non_final_next_states, dim1=1, dim2=2).clone().view(-1,N)
    invalid_indices = invalid_actions.nonzero()
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        valid_indices = (invalid_actions == 0).nonzero()
        print(f'DEBUG: invalid_actions shape: {invalid_actions.shape}')
        print(f'DEBUG: invalid_actions[0]: {invalid_actions[0]}')
        assert invalid_actions.shape == (torch.count_nonzero(non_final_mask), N)
        print(f'DEBUG: invalid_indices shape: {invalid_indices.shape}')
        print(f'DEBUG: valid_indices shape : {valid_indices.shape}')
        print(f'DEBUG: invalid_indices[0]: {invalid_indices[0]}')
        print(f'DEBUG: valid_indices[0]: {valid_indices[0]}')

    # make it so the network never chooses the invalid action (max operation is performed next)
    targets[invalid_indices[:,0], invalid_indices[:,1]] = float('-inf')
    if DOUBLE_DQN:
        next_preds[invalid_indices[:,0], invalid_indices[:,1]] = float('-inf')
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: targets after invalid shape: {targets.shape}')
        print(f'DEBUG: targets after invalid [0]: {targets[0]}')
        assert targets.shape == (torch.count_nonzero(non_final_mask), N)
        if DOUBLE_DQN:
            print(f'DEBUG: next_preds after invalid shape: {next_preds.shape}')
            print(f'DEBUG: next_preds after invalid [0]: {next_preds[0]}')
            assert targets.shape == (torch.count_nonzero(non_final_mask), N)


    # get the optimisitc Q-value for the next best action in non-final states
    if DOUBLE_DQN:
        next_actions = next_preds.argmax(dim=1).unsqueeze(1)
        next_state_values[non_final_mask] = targets.gather(1, next_actions).squeeze()
    else:
        next_state_values[non_final_mask] = targets.max(1)[0].detach()
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: next_state_action_values shape: {next_state_values.shape}')
        print(f'DEBUG: next_state_action_values[0]: {next_state_values[0]}')
        assert next_state_values.shape == (BATCH_SIZE,)
        if DOUBLE_DQN:
            print(f'DEBUG: next_actions shape: {next_actions.shape}')
            print(f'DEBUG: next_actions[0]: {next_actions[0]}')

    # calculate the discounted expected future reward
    expected_state_action_values = (next_state_values*GAMMA) + reward_batch
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: expected_state_action_values shape: {expected_state_action_values.shape}')
        print(f'DEBUG: expected_state_action_values[0]: {expected_state_action_values[0]}')
        assert expected_state_action_values.shape == (BATCH_SIZE,)
    AVG_EXPECTED_REWARDS.append(torch.mean(expected_state_action_values, 0).item())
    # calculate the temporal difference (TD) error
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
        print(f'DEBUG: loss shape: {loss.shape}')
        print(f'DEBUG: loss item: {loss.item()}')
        assert loss.shape == ()
    LOSSES.append(loss.item())
    EPISODES.append(I_EPISODE)

    # optimize model
    optimizer.zero_grad()
    loss.backward()
    if DEBUG_MODE:
        old_params = [param.clone() for param in policy_net.parameters()]
    # clip gradients (regularization)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# training loop
accumulated_reward = 0
for I_EPISODE in tqdm(range(NUM_EPISODES), total=NUM_EPISODES):

    # create new game and opponent
    game = Game(N, 10, 10, verbose=False)
    opponent = select_opponent(OPPONENT_TYPE)

    # assign red/blue randomly
    '''
    player = ''
    opp_color = ''
    if np.random.rand() < 0.5:
        game.set_player(agent) # red player
        game.set_player(opponent) # blue player
        player = 'red'
        opp_color = 'blue'
    else:
    '''
    game.set_player(opponent)
    game.set_player(agent)
    player = 'blue'
    #opp_color = 'red'
    result = 'continue'

    # set up states and next states
    state = torch.tensor(game.state.to_numpy(color_pref=player), device=device).unsqueeze(0)
    #mirror_state = torch.tensor(game.state.to_numpy(color_pref=opp_color), device=device).unsqueeze(0)
    #prev_state = state
    #mirror_prev_state = mirror_state
    next_state = None
    #mirror_next_state = None
    while result == 'continue':
        # take a step of the game
        blue_action, red_action, result = game.step()
        #if player == 'blue':
        action = blue_action
            #if red_action == -1:
                #mirror_state = mirror_prev_state
            #else:
        
            # mirror_action = red_action
        '''
        else:
            # red_action could be -1 if all nodes are colored
            # before red gets to select (just keep previous action and rollback state)
            if red_action == -1:
                state = prev_state
            else:
                action = red_action
            #mirror_action = blue_action
        '''
        # get immediate reward
        reward = 0.0
        if result == 'red' or result == 'blue':
            red_nodes = len(game.state.get_nodes(color='red'))
            blue_nodes = len(game.state.get_nodes(color='blue'))
            reward = blue_nodes - red_nodes
            '''
            elif player == 'red':
                reward = red_nodes - blue_nodes
            '''
        accumulated_reward += reward
        #mirror_reward = -reward
        # convert to tensors
        reward = torch.tensor([reward], device=device)
        #mirror_reward = torch.tensor([mirror_reward], device=device)
        if DEBUG_MODE:
            invalid_actions = torch.diagonal(state, dim1=1, dim2=2).nonzero()[:,1]
            if I_EPISODE % DEBUG_OFFSET == 0:
                print(f'DEBUG: player: {player}')
                print(f'DEBUG: state: {state}')
                print(f'DEBUG: action taken: {action}')
                print(f'DEBUG: invalid_actions: {invalid_actions}')
            assert not action in invalid_actions
        if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
            print(f'DEBUG: player: {player}')
            print(f'DEBUG: reward shape: {reward.shape}')
            print(f'DEBUG: reward[0]: {reward[0]}')
            assert reward.shape == (1,)

        action = torch.tensor([[action]], device=device, dtype=torch.int64)
        #mirror_action = torch.tensor([[mirror_action]], device=device, dtype=torch.int64)
        if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
            print(f'DEBUG: action shape: {action.shape}')
            print(f'DEBUG: action[0]: {action[0]}')
            assert action.shape == (1,1)

        # TODO: generate mirrored transition
        # if player is red generate (s,a,s',r) as if he was playing blue and took blue action
        # this might be an issue since he didn't choose blue action
        # good for exploration bad for exploitation (could lead to divergence if used in later
        # steps of training) i.e. only use for I_EPISODE < NUM_EPISODES//2
        # this can also be tuned


        # observe new state
        if result == 'continue':
            next_state = torch.tensor(game.state.to_numpy(color_pref=player),
                                      device=device).unsqueeze(0)
            #mirror_next_state = torch.tensor(game.state.to_numpy(color_pref=opp_color),
            #                                 device=device).unsqueeze(0)
        else:
            next_state = None
            #mirror_next_state = None
        if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
            print(f'DEBUG: state shape: {state.shape}')
            print(f'DEBUG: state: {state}')
            if result == 'continue':
                print(f'DEBUG: next_state shape: {next_state.shape}')
                print(f'DEBUG: next_state: {next_state}')
                assert next_state.shape == (1, N, N)
                # if this fails then an invalid action was taken
                # or states aren't being generated correctly
                assert not torch.equal(state, next_state)
            else:
                assert next_state is None
            assert state.shape == (1, N, N)

        # add transition to replay memory
        memory.push(state, action, next_state, reward)
        #memory.push(mirror_state, mirror_action, mirror_next_state, mirror_reward)
        # move to next state
        #prev_state = state
        state = next_state
        #mirror_prev_state = mirror_state
        #mirror_state = mirror_next_state

        # optimize for one step
        optimize_model()
        if DEBUG_MODE and I_EPISODE % DEBUG_OFFSET == 0:
            print('=========================================================')

    # update target network
    if I_EPISODE % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # add accumulated reward for 100 episodes
    if I_EPISODE % 100 == 0:
        REWARDS.append(accumulated_reward)
        accumulated_reward = 0

# save model + statistics
if DOUBLE_DQN:
    header = f'n{N}-ep{NUM_EPISODES}-op{OPPONENT_TYPE}-ddqn'
else:
    header = f'n{N}-ep{NUM_EPISODES}-op{OPPONENT_TYPE}'
torch.save(policy_net.state_dict(), f'./saved_models/model-{header}.pt')
plt.plot(EPISODES, LOSSES)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Training Loss over Episodes')
plt.savefig(f'./analysis/loss-{header}.png')
plt.close()
plt.plot(EPISODES, AVG_EXPECTED_REWARDS)
plt.xlabel('Episodes')
plt.ylabel('Avg Expected Reward')
plt.title('Average Expected Reward over Episodes')
plt.savefig(f'./analysis/e-reward-{header}.png')
plt.close()
every_100 = np.linspace(0, NUM_EPISODES, num=len(REWARDS))
plt.plot(every_100, REWARDS)
plt.xlabel('Episodes')
plt.ylabel('Accumulated Reward')
plt.title('Accumulated Reward per 100 Episodes')
plt.savefig(f'./analysis/a-reward-{header}.png')
plt.close()
