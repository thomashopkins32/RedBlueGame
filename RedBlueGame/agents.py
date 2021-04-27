'''
agents.py

This module contains all of the agents that will participate in the game.
Feel free to contribute code for your agent here.

Note:
    All agent classes should inherit the Agent class.
'''
import numpy as np
import torch

import pdb
import time
import copy
import math

from models import DQN, DQFFN


class Agent:
    ''' Abstract class for various Agents '''
    def __init__(self):
        self.name = 'Agent'

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def get_action(self, state, player):
        '''
        Returns the action given the state of the game and the player type

        Parameters
        ----------
        state : Graph
            current state of the game graph (see graph.py)
        player : str
            'blue' or 'red'

        Returns
        -------
        int
            node number of graph to color
        '''
        abstract


class RandomAgent(Agent):
    ''' Agent that chooses actions randomly '''
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = 'RandomAgent'

    def get_action(self, state, player):
        possible_actions = state.get_nodes(color='grey')
        return np.random.choice(possible_actions)


class TimeoutAgent(Agent):
    ''' Agent that can never seem to finish thinking '''
    def __init__(self):
        super(TimeoutAgent, self).__init__()
        self.name = 'TimeoutAgent'

    def get_action(self, state, player):
        time.sleep(1000000)


class GreedyAgent(Agent):
    ''' Agent that always picks the node with the most edges '''
    def __init__(self):
        super(GreedyAgent, self).__init__()
        self.name = 'GreedyAgent'

    def get_action(self, state, player):
        possible_actions = state.get_nodes(color='grey')
        sorted_actions = sorted(possible_actions, key=lambda x : len(state.get_node_neighbors(x)))
        return sorted_actions[-1]


class DifferenceAgent(Agent):
    ''' Agent that picks the node that causes the largest difference in the game score '''
    def __init__(self):
        super(DifferenceAgent, self).__init__()
        self.name = 'DifferenceAgent'

    def _difference(self, state, action, player):
        new_state = copy.deepcopy(state)
        new_state.set_node_attrs(action, {'color': player})
        neighbors = new_state.get_node_neighbors(action)
        for n in neighbors:
            new_state.set_node_attrs(n, {'color': player})
        blue_count = len(new_state.get_nodes(color='blue'))
        red_count = len(new_state.get_nodes(color='red'))
        if player == 'blue':
            return blue_count - red_count
        return red_count - blue_count

    def get_action(self, state, player):
        possible_actions = state.get_nodes(color='grey')
        sorted_actions = sorted(possible_actions, key=lambda x: self._difference(state, x, player))
        return sorted_actions[-1]

class MiniMaxAgent(Agent):
    ''' Agent that utilizes minimax tree search to select actions '''
    def __init__(self, depth=5):
        super(MiniMaxAgent, self).__init__()
        self.name = 'MiniMaxAgent'
        self.depth = depth

    def get_action(self, state, player):
        self.player = player
        possible_actions = state.get_nodes(color='grey')
        key_func = lambda x: self._value(self._transition(state, x, 0), 1, 0, float('-inf'),
                                         float('inf'))
        sorted_actions = sorted(possible_actions, key=key_func)
        return sorted_actions[0]

    def _value(self, state, player, depth, alpha, beta):
        if player > 1:
            depth += 1
            player = 0
        if depth >= self.depth or len(state.get_nodes(color='grey')) == 0:
            return self._eval(state, player)

        if player == 0:
            return self._max(state, player, depth, alpha, beta)
        else:
            return self._min(state, player, depth, alpha, beta)

    def _max(self, state, player, depth, alpha, beta):
        actions = state.get_nodes(color='grey')
        if depth >= self.depth or len(actions) == 0:
            return self._eval(state, player)
        v = float('-inf')
        for action in actions:
            val = self._value(self._transition(state, action, player), player + 1, depth, alpha,
                              beta)
            v = max((v, val))
            alpha = max((v, alpha))
            if v > beta:
                return v
        return v

    def _min(self, state, player, depth, alpha, beta):
        actions = state.get_nodes(color='grey')
        if depth >= self.depth or len(actions) == 0:
            return self._eval(state, player)
        v = float('-inf')
        for action in actions:
            val = self._value(self._transition(state, action, player), player + 1, depth, alpha,
                              beta)
            v = min((v, val))
            beta = min((v, beta))
            if v < alpha:
                return v
        return v

    def _transition(self, state, action, player):
        new_state = copy.deepcopy(state)
        if player % 2 == 0:
            curr_player = self.player
        else:
            if self.player == 'blue':
                curr_player = 'red'
            else:
                curr_player = 'blue'
        new_state.set_node_attrs(action, {'color': curr_player})
        neighbors = new_state.get_node_neighbors(action)
        for n in neighbors:
            new_state.set_node_attrs(n, {'color': curr_player})
        return new_state

    def _eval(self, state, player):
        red_count = len(state.get_nodes(color='red'))
        blue_count = len(state.get_nodes(color='blue'))
        if self.player == 'red':
            return red_count - blue_count
        return blue_count - red_count


class DQNAgent(Agent):
    '''
    Agent that utilizes a Deep Q-Network to estimate the value
    of state-action pairs

    See train.py for training methods
    See model.py for DQN neural network models

    This Agent should load its learned state from a file
    otherwise it will have random parameters.
    '''
    def __init__(self, n, network_param_file='', training=False, eps_end=0.05, eps_start=0.9,
                 eps_decay=200, device='cpu', model='ffn'):
        super(DQNAgent, self).__init__()
        self.name = 'DQNAgent'
        self.n = n
        self.model = None
        if model == 'ffn':
            self.model = DQFFN(n).to(device)
        else:
            self.model = DQN(n).to(device)
        self.training = training
        self.device = device
        if network_param_file != '':
            self.load(network_param_file)
        if not self.training:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.eps_end = eps_end
            self.eps_start = eps_start
            self.eps_decay = eps_decay
            self.steps_done = 0

    def get_action(self, state, player):
        n = self.n
        s = torch.tensor(state.to_numpy(color_pref=player), device=self.device).reshape(1,1,n,n)
        possible_actions = state.get_nodes(color='grey')
        # get valid actions
        if self.training:
            sample = np.random.rand()
            eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1.0 * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_thresh:
                with torch.no_grad():
                    action_dist = self.model(s).detach()[0]
                    mask = np.ones(len(action_dist), np.bool)
                    mask[possible_actions] = 0
                    action_dist[mask] = float('-inf')
                    action = torch.argmax(action_dist).item()
            else:
                action = np.random.choice(possible_actions)
        else:
            action_dist = self.model(s).detach()[0]
            mask = np.ones(len(action_dist), np.bool)
            mask[possible_actions] = 0
            action_dist[mask] = float('-inf')
            action = torch.argmax(action_dist).item()
        return action

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint)
