'''
graph.py

Includes Graph object for manipulating the networkx graph
'''
import networkx as nx
import numpy as np
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
        self._pos = nx.drawing.spring_layout(self._g)
        for n in self._g.nodes:
            self._g.nodes[n]['color'] = 'grey'

    def to_numpy(self, color_pref='blue'):
        '''
        Converts the graph to an nxn adjacency matrix with color
        info on the diagonal. Default conversion is (blue, red, grey) -> (1, -1, 0).
        When color_pref is 'red' conversion is (blue, red, grey) -> (-1, 1, 0).
        This could be useful for neural network input.

        Parameters
        ----------
        color_pref : str, optional
            color preference for diagonal values

        Returns
        -------
        np.array
            adjacency matrix with color info along the diagonal
        '''
        adj_matrix = nx.convert_matrix.to_numpy_array(self._g)
        color_data = []
        for n in self.get_nodes():
            if self._g.nodes[n]['color'] == color_pref:
                color_data.append(1.0)
            elif self._g.nodes[n]['color'] == 'grey':
                color_data.append(0.0)
            else:
                color_data.append(-1.0)
        np.fill_diagonal(adj_matrix, color_data)
        return adj_matrix.astype(np.float32)

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
        ''' Displays a plot of the graph '''
        colors = [self.get_node_attrs(n)['color'] for n in self.get_nodes()]
        nx.draw_networkx(self._g, node_color=colors, pos=self._pos)
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
