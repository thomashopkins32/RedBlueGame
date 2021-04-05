'''
agents.py

This module contains all of the agents that will participate in the game.
Feel free to contribute code for your agent here.

Note:
    All agent classes should inherit the Agent class.
'''
import numpy as np

import time


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
