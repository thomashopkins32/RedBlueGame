'''
graph.py

Includes Graph object for manipulating the networkx graph
'''
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph
import matplotlib.pyplot as plt

class Graph:
    '''
    Creates a random graph using networkx library.
    The graph itself should be private.
    It should only be changed by setting the node attributes
    '''
    def __init__(self, n, p=0.1):
        self._n = n
        self._g = fast_gnp_random_graph(n, p)
        for n in self._g.nodes:
            self._g.nodes[n]['color'] = 'grey'

    def get_nodes(self, color=''):
        '''
        Gets the nodes from the graph

        Parameters
        ----------
        color : str
            specify the desired color of nodes ('grey', 'blue', 'red')

        Returns
        -------
        list
            list of nodes with specified color or all nodes
        '''
        if color == '':
            return self._g.nodes
        return [n for n in self._g.nodes if self._g.nodes[n]['color'] == color]

    def get_node_attrs(self, node):
        '''
        Gets the data associated with a node

        Parameters
        ----------
        node : int
            number of node in graph

        Returns
        -------
        dict
            attributes of node
        '''
        return self._g.nodes[node]

    def set_node_attrs(self, node, attrs):
        '''
        Sets the attributes of a node

        Parameters
        ----------
        node : int
            number of node in graph
        attrs : dict
            dictionary of data attributes for node
        '''
        self._g.nodes[node].update(attrs)

    def get_node_neighbors(self, node):
        ''' Returns a list of neighboring nodes of the given node '''
        return list(self._g.neighbors(node))

    def show(self):
        colors = [self.get_node_attrs(n)['color'] for n in self.get_nodes()]
        nx.draw_networkx(self._g, node_color=colors)
        plt.show()

if __name__=='__main__':
    g = Graph(50)
    print(f'Nodes: {g.get_nodes()}')
    print(f"Blue Nodes: {g.get_nodes(color='blue')}")
    print(f"Red Nodes: {g.get_nodes(color='red')}")
    print(f'Node 1 attrs: {g.get_node_attrs(1)}')
    g.set_node_attrs(1, {'color': 'red'})
    print(f'Node 1 attrs after set: {g.get_node_attrs(1)}')
    print(f'Node 1 neighbors: {g.get_node_neighbors(1)}')
    g.show()
