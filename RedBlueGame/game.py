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
    def __init__(self, n, r, t):
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
        '''
        self.n = n
        self.r = r
        self.t = t
        self.state = Graph(n)
        self.round = 1
        self.points = {'red': 0, 'blue': 0}
        self.turn = 'blue'
        self.players = {'red': None, 'blue': None}

    def set_player(self, agent):
        if self.players['red'] is None:
            self.players['red'] = agent
        elif self.players['blue'] is None:
            self.players['blue'] = agent
        else:
            print('Already two players...')

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
        self.points[player] += len(neighbors) + 1

    def get_possible_actions(self):
        ''' Returns a list of possible actions (node numbers) '''
        return self.state.get_nodes(color='grey')

    def run(self):
        ''' Starts the main game loop (returns the winning player) '''
        if self.players['red'] is None or self.players['blue'] is None:
            print(f'Not enough players to start the game: {self.players}')
            return ''
        self.state.show()
        round_count = 0
        for i in range(self.r):
            print('=====================================================================')
            print(f'Starting round {i}')
            print(f'Blue player {self.players["blue"]} is choosing a node to color')
            try:
                with time_limit(self.t):
                    blue_action = self.players['blue'].get_action(self.state, 'blue')
                self.perform_action('blue', blue_action)
            except TimeoutException as e:
                print('Blue player timed out! Blue loses the game!')
                return 'red'
            print(f'Blue player chose {blue_action}')
            if len(self.state.get_nodes(color='grey')) == 0:
                print('All nodes have been colored.')
                break
            print(f'Red player {self.players["red"]} is choosing a node to color')
            try:
                with time_limit(self.t):
                    red_action = self.players['red'].get_action(self.state, 'red')
                self.perform_action('red', red_action)
            except TimeoutException as e:
                print('Red player timed out! Red loses the game!')
                return 'blue'
            print(f'Red player chose {red_action}')
            print(f'Game after round {i}:')
            self.state.show()
            if len(self.state.get_nodes(color='grey')) == 0:
                print('All nodes have been colored.')
                break
            round_count += 1
        print('=====================================================================')
        print(f'Game has ended after {round_count} rounds')
        winner = ''
        if self.points['red'] == self.points['blue']:
            print(f'Game tied with scores {self.points}')
        elif self.points['red'] > self.points['blue']:
            print(f'Red wins with scores {self.points}')
            winner = 'red'
        else:
            print(f'Blue wins with scores {self.points}')
            winner = 'blue'
        return winner


if __name__=='__main__':
    from agents import RandomAgent, TimeoutAgent
    game = Game(50, 10, 10)
    game.set_player(RandomAgent())
    game.set_player(RandomAgent())
    game.run()
