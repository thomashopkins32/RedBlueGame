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
        self.players = {'red': None, 'blue': None}
        self.verbose = verbose

    def set_player(self, agent):
        ''' Set players in the game '''
        if self.players['blue'] is None:
            self.players['blue'] = agent
        elif self.players['red'] is None:
            self.players['red'] = agent
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

    def get_score(self):
        ''' Calculates a returns the score of the game '''
        blue_nodes = len(self.state.get_nodes(color='blue'))
        red_nodes = len(self.state.get_nodes(color='red'))
        return blue_nodes, red_nodes

    def get_reward(self, player):
        ''' Gets the reward for a player after taking an action '''
        blue, red = self.get_score()
        if player == 'blue':
            return blue - red
        return red - blue

    def step(self, player):
        '''
        Process one player's turn in the game

        Parameters
        ----------
        player : str
            color of the player ['blue', 'red']

        Returns
        -------
        action : int
            node that the player chose to color
        reward : int
            difference in score if game ended otherwise 0.0
            blue_nodes - red_nodes if player is blue
            red_nodes - blue_nodes if player is red
        '''
        self._print(f'{player}  player {self.players[player]} is choosing a node to color')
        try:
            with time_limit(self.t):
                action = self.players[player].get_action(self.state, player)
            self.perform_action(player, action)
        except TimeoutException as e:
            self._print(f'{player} player timed out! {player} loses the game!')
            return action, self.get_reward(player), True
        self._print(f'{player} player chose {action}')
        if len(self.state.get_nodes(color='grey')) == 0:
            self._print('All nodes have been colored.')
            return action, self.get_reward(player), True
        # red always ends the round
        if player == 'red':
            self.round += 1
        # check if round limit has been reached
        if self.round == self.r:
            self._print('Max number of rounds has been reached.')
            return action, self.get_reward(player), True
        return action, 0.0, False

    def run(self):
        ''' Starts the main game loop (returns the winning player) '''
        if self.players['red'] is None or self.players['blue'] is None:
            print(f'Not enough players to start the game: {self.players}')
            return ''
        if self.verbose:
            self.state.show()
        end = False
        winner = ''
        while not end:
            self._print('=====================================================================')
            self._print(f'Starting round {self.round+1}')
            _, _, end = self.step('blue')
            if end:
                break
            _, _, end = self.step('red')
            self._print(f'Game after round {self.round}:')
            if self.verbose:
                self.state.show()
        self._print('=====================================================================')
        self._print(f'Game has ended after {self.round} rounds')
        if self.verbose:
            self.state.show()
        blue, red = self.get_score()
        if blue > red:
            winner = 'blue'
        elif red > blue:
            winner = 'red'
        else:
            winner = 'tied'
        self._print(f'Winner is {winner} with scores\nblue: {blue}\nred: {red}')
        return winner


if __name__=='__main__':
    from agents import (RandomAgent, TimeoutAgent, GreedyAgent,
                        DifferenceAgent, MiniMaxAgent, DQNAgent)
    n = int(sys.argv[1])
    game = Game(n, 10, 10)
    game.set_player(DifferenceAgent())
    game.set_player(GreedyAgent())
    game.run()
