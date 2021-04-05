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

from graph import Graph


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
        self.points = [0, 0]
        self.turn = 'red'

    def perform_action(self, player, action):
        '''
        Generates a change in game state due to an
        action by a player

        Parameters
        ----------
        player : str
            'blue' or 'red' player making the move
        action :
        '''
        return


