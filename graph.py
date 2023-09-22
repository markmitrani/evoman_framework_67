from typing import List, Tuple
import numpy as np



# Definition for a Graph representing a toroid grid
class GridGraph:
    class Node:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y
            self.edges: List[Tuple] = []
        def get_coords(self):
            return (self.x, self.y)

        def add_edge(self, node):
            self.edges.append(node.get_coords())

        def add_edge(self, x: int, y: int):
            # Edge to itself not allowed
            assert x != self. x or y != self.y
            self.edges.append((x, y))

        def get_neighbours(self):
            return self.edges

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        assert self.n > 0 and self.m > 0
        self.nodes: List = []
        self.__init_structure()

    def __init_structure(self):
        for n in range(self.n):
            for m in range(self.m):
                node = GridGraph.Node(n, m)

                # Add edges connecting 'immediate' neighboars
                node.add_edge((n+1) % self.n, m)
                node.add_edge(n, (m+1) % self.m)
                node.add_edge((n-1) % self.n, m)
                node.add_edge(n, (m-1) % self.m)

                self.nodes.append(node)

    def __getitem__(self, key):
        assert key < self.n
        return self.nodes[key*self.n:key*self.n+self.n]

    def __setitem__(self, key, value):
        assert key < self.n
        if value is list:
            assert len(value) == self.m
        if value is int or value is float:
            value = [value] * self.m
        self.nodes[key*self.n:key*self.n+self.n] = value

    def get_knn(self, x: int, y: int, dist_func, max_dist=1):
        assert x < self.n and y < self.m
        node: GridGraph.Node = self[x][y]
        nn = []
        stack: List = [node.get_neighbours()]
        while stack:
            # Distance and coordinate tuple
            n_x, n_y = stack.pop() 
            n_node: GridGraph.Node = self[n_x][n_y]
            # If close enough, add to list
            if dist_func(node, n_node) <= max_dist:
                nn.append(n_node)
                # Search neighbours and add them
                for new_neighbour in n_node.get_neighbours():
                    stack.append(new_neighbour)
 
def euclidean_distance(n1: GridGraph.Node, n2: GridGraph.Node):
    pass