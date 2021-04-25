'''
game.py

Contains the specifications and dynamics for the Red and Blue Game

Game Specifications
-------------------
    n: the number of nodes
    R: the number of rounds
    T: time limit
Game Dynamics
-------------
    The platform generates a random graph of n nodes
    For each round r =1 to R
        The blue player chooses a node S that has not been colored before. S and all nodes
        adjacent to S becomes blue (including the nodes that are previously red). This
        should be done in T seconds.
        The red player choose a node S' that has not been colored before S' and all nodes
        adjacent to S' becomes blue.
    The player with more nodes wins
Notes
-----
   Any player who does not answer in time automatically loses.
   If all nodes are colored, then the game immediately ends and whomever with more nodes wins.
'''
import sys
import signal
from contextlib import contextmanager

from graph import Graph

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException('Timed out!')
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class Game:
    '''
    Contains the game information and dynamics for the Red and Blue Game

    Members
    -------
    self.n : int
        the total number of nodes in the game graph
    self.r : int
        the total number of rounds in the game
    self.t : int
        the time limit for each move
    self.state : Graph
        data structure to store game state
    '''
    def __init__(self, n, r, t, verbose=True):
        '''
        Generates a random graph to serve as the game board

        Parameters
        ----------
        n : int
            number of nodes in graph
        r : int
            number of rounds in game
        t : int
            time to make decision in game
        verbose : bool, optional
            give output of rounds and show graph
        '''
        self.n = n
        self.r = r
        self.t = t
        self.state = Graph(n)
        self.round = 0
        self.turn = 'blue'
        self.players = {'red': None, 'blue': None}
        self.verbose = verbose

    def set_player(self, agent):
        if self.players['red'] is None:
            self.players['red'] = agent
        elif self.players['blue'] is None:
            self.players['blue'] = agent
        else:
            print('Already two players...')

    def _print(self, s):
        if self.verbose:
            print(s)

    def perform_action(self, player, action):
        '''
        Generates a change in game state due to an
        action by a player

        Parameters
        ----------
        player : str
            'blue' or 'red' player making the move
        action : int
            node to color in graph
        '''
        self.state.set_node_attrs(action, {'color': player})
        neighbors = self.state.get_node_neighbors(action)
        for n in neighbors:
            self.state.set_node_attrs(n, {'color': player})

    def get_possible_actions(self):
        ''' Returns a list of possible actions (node numbers) '''
        return self.state.get_nodes(color='grey')

    def determine_winner(self):
        self.points = {}
        self.points['blue'] = len(self.state.get_nodes(color='blue'))
        self.points['red'] = len(self.state.get_nodes(color='red'))
        if self.points['blue'] > self.points['red']:
            self._print(f'Blue wins with scores {self.points}')
            return 'blue'
        elif self.points['red'] > self.points['blue']:
            self._print(f'Red wins with scores {self.points}')
            return 'red'
        self._print(f'Game tied with scores {self.points}')
        return 'tied'

    def step(self):
        '''
        Process one round of the game

        Returns
        -------
        blue_action : int
            node that the blue player chose to color
        red_action : int
            node that the red player chose to color, -1 if game ended
        result : str
            one of ('continue', 'tied', 'red', 'blue')
        '''
        self._print(f'Blue player {self.players["blue"]} is choosing a node to color')
        try:
            with time_limit(self.t):
                blue_action = self.players['blue'].get_action(self.state, 'blue')
            self.perform_action('blue', blue_action)
        except TimeoutException as e:
            self._print('Blue player timed out! Blue loses the game!')
            return blue_action, -1, 'red'
        self._print(f'Blue player chose {blue_action}')
        if len(self.state.get_nodes(color='grey')) == 0:
            self._print('All nodes have been colored.')
            return blue_action, -1, self.determine_winner()
        self._print(f'Red player {self.players["red"]} is choosing a node to color')
        try:
            with time_limit(self.t):
                red_action = self.players['red'].get_action(self.state, 'red')
            self.perform_action('red', red_action)
        except TimeoutException as e:
            self._print('Red player timed out! Red loses the game!')
            return blue_action, red_action, 'blue'
        self._print(f'Red player chose {red_action}')
        if len(self.state.get_nodes(color='grey')) == 0:
            self._print('All nodes have been colored.')
            return blue_action, red_action, self.determine_winner()
        self.round += 1
        if self.round == self.r:
            self._print('Max number of rounds has been reached.')
            return blue_action, red_action, self.determine_winner()
        return blue_action, red_action, 'continue'

    def run(self):
        ''' Starts the main game loop (returns the winning player) '''
        if self.players['red'] is None or self.players['blue'] is None:
            print(f'Not enough players to start the game: {self.players}')
            return ''
        if self.verbose:
            self.state.show()
        result = 'continue'
        winner = ''
        while result == 'continue':
            self._print('=====================================================================')
            self._print(f'Starting round {self.round+1}')
            _, _, result = self.step()
            if result != 'continue':
                winner = result
                break
            self._print(f'Game after round {self.round}:')
            if self.verbose:
                self.state.show()
        self._print('=====================================================================')
        self._print(f'Game has ended after {self.round} rounds')
        if self.verbose:
            self.state.show()
        return winner


if __name__=='__main__':
    from agents import (RandomAgent, TimeoutAgent, GreedyAgent,
                        DifferenceAgent, MiniMaxAgent, DQNAgent)
    game = Game(51, 10, 10)
    game.set_player(GreedyAgent())
    game.set_player(DQNAgent(51, network_param_file='./saved_models/m_1000_random.pt'))
    game.run()
