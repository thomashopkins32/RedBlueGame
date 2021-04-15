# RedBlueGame
Game for CSCI-4150 at RPI to experiment with artificial intelligence

## Game Specifications
    n: the number of nodes
    R: the number of rounds
    T: time limit
## Game Dynamics
    The platform generates a random graph of n nodes
    For each round r =1 to R
        The blue player chooses a node S that has not been colored before. S and all nodes
        adjacent to S becomes blue (including the nodes that are previously red). This
        should be done in T seconds.
        The red player choose a node S' that has not been colored before S' and all nodes
        adjacent to S' becomes blue.
    The player with more nodes wins
## Notes
   Any player who does not answer in time automatically loses.
   If all nodes are colored, then the game immediately ends and whomever with more nodes wins.

## Contributing
To create your agent open a pull request and edit the `RedBlueGame/agents.py` file. Next, set up your own agent class like this:
```
class MyAgent(Agent):
    def __init__(self):
        super(MyAgent, self).__init__()
        self.name = 'MyAgent'
        
    def get_action(self, state, player):
        pass
```
Please use a descriptive name for your agent class and make sure it is unique.
`get_action()` is used by the game engine to query the agent class for its next action when its that player's turn. It should return an integer denoting the grey node to color in the graph. See other agents for examples.

You can test the performance of your agent by adding your agent as an option in the `RedBlueGame/performance.py` file or creating your own file for starting a game.

Please do not add any additional files to the repository if you are only adding or testing an agent.

Other forms of contributing include working on open issues.
