'''
performance.py

Evaluates the performance of various agents over many games
'''
import sys

import numpy as np
from tqdm import tqdm

from agents import RandomAgent, GreedyAgent, DifferenceAgent, MiniMaxAgent, DQNAgent
from game import Game

# note: MiniMaxAgent is currently too slow to test

def assign_player(player):
    if player == 'RandomAgent':
        return RandomAgent()
    elif player == 'GreedyAgent':
        return GreedyAgent()
    elif player == 'DifferenceAgent':
        return DifferenceAgent()
    elif player == 'MiniMaxAgent':
        return MiniMaxAgent(depth=2)
    elif player == 'DQNAgent':
        #return DQNAgent(51)
        return DQNAgent(51, network_param_file=sys.argv[4])
    return None

num_games = int(sys.argv[1])
player1 = sys.argv[2]
player2 = sys.argv[3]
p1 = assign_player(player1)
p2 = assign_player(player2)
print(f'Simulating {num_games} games for {p1} vs {p2}')
p1_wins = 0
p2_wins = 0
ties = 0
for i in tqdm(range(num_games), total=num_games):
    game = Game(51, 25, 10, verbose=False)
    if np.random.rand() < 0.5:
        game.set_player(p2)
        game.set_player(p1)
        players = {'blue': 'p1', 'red': 'p2'}
    else:
        game.set_player(p1)
        game.set_player(p2)
        players = {'blue': 'p2', 'red': 'p1'}
    winner = game.run()
    if winner == '':
        ties += 1
    elif players[winner] == 'p1':
        p1_wins += 1
    elif players[winner] == 'p2':
        p2_wins += 1
print(f'Results:\nPlayer 1 {p1} wins {p1_wins}/{num_games} = {(p1_wins/num_games)*100}%')
print(f'Player 2 {p2} wins {p2_wins}/{num_games} = {(p2_wins/num_games)*100}%')
print(f'Total ties: {ties}')


