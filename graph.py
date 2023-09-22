from typing import List, Tuple
import copy
import numpy as np
import itertools


# Definition for a Graph representing a toroid grid
class GridGraph:
    class Node:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y
            self.edges: List[Tuple] = []
            self.pop = []

        def get_pop(self):
            return self.pop

        def set_pop(self, pop):
            self.pop = pop

        def __getitem__(self, key):
            assert key < len(self.pop)
            return self.pop[key]

        def __setitem__(self, key, value):
            assert key < len(self.pop)
            self.pop[key] = value

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

        def get_best_ind(self):
            best_ind = None
            best_score = -1000
            for best_cand in self.pop:
                if best_cand.fitness.values[0] > best_score:
                    best_score = best_cand.fitness.values[0]
                    best_ind = best_cand
            return best_ind

        def __str__(self) -> str:
            return f'({self.x}, {self.y})'
        def __repr__(self) -> str:
            return f'({self.x}, {self.y})'

    def __init__(self, n: int, m: int, pop_size, toolbox = None):
        self.n = n
        self.m = m
        assert self.n > 0 and self.m > 0
        self.nodes: List = []
        self.pop_size = pop_size
        self.__init_structure(toolbox)

    def __init_structure(self, toolbox):
        for n in range(self.n):
            for m in range(self.m):
                node = GridGraph.Node(n, m)

                # Add edges connecting 'immediate' neighboars
                node.add_edge((n+1) % self.n, m)
                node.add_edge(n, (m+1) % self.m)
                node.add_edge((n-1) % self.n, m)
                node.add_edge(n, (m-1) % self.m)
                if toolbox:
                    node.set_pop(toolbox.population(n=self.pop_size))
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
    def __str__(self) -> str:
        str_rep = ''
        for n in range(self.n):
            if n == 0:
                str_rep+='     '
                for m in range(self.m):
                    str_rep += '|        '
                str_rep+='\n'
            for m in range(self.m):
                if m == 0:
                    str_rep+='- '
                str_rep += f'({n}, {m}) - '
            str_rep += '\n'
        str_rep += '     '
        for m in range(self.m):
            str_rep += '|        '
        return str_rep

    def __iter__(self):
        return self.nodes.__iter__()

    def update_fitnesses(self, toolbox):
        for n in range(self.n):
            for m in range(self.m):
                pop = [ind for ind in self[n][m].get_pop() if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, pop)
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit

    def get_knn(self, x: int, y: int, dist_func, max_dist=1) -> List[Node]:
        assert x < self.n and y < self.m
        node: GridGraph.Node = self[x][y]
        nn = []
        visited = [] 

        for i in range(self.n):
            visited.append([False] * self.m)

        visited[x][y] = True

        stack: List = node.get_neighbours()
        while stack:
            # Distance and coordinate tuple
            n_x, n_y = stack.pop() 
            n_node: GridGraph.Node = self[n_x][n_y]
            # If close enough, add to list
            if dist_func(node, n_node, self.n, self.m) <= max_dist:
                nn.append(n_node)
                # Search neighbours and add them
                for new_neighbour in n_node.get_neighbours():
                    if visited[new_neighbour[0]][new_neighbour[1]]:
                        continue
                    stack.append(new_neighbour)
            visited[n_x][n_y] = True
        return nn

    def deepcopy(self):
        new_gg = GridGraph(self.n, self.m, self.pop_size)

        for node in self.nodes:
            new_gg[node.get_coords()[0]][node.get_coords()[1]].set_pop(copy.deepcopy(node.get_pop()))
        return new_gg

    def allpop(self):
        all_population = [n.get_pop() for n in self.nodes]
        return list(itertools.chain(*all_population))

def frac(x) -> float:
    return np.abs(np.floor(x) - x)

def dist(p1, p2) -> float:
    dif_1 = frac(p1[0] - p1[1])
    dif_2 = frac(p2[0] - p2[1])

    return np.power(np.min(dif_1, 1-dif_1), 2) + \
        np.power(np.min(dif_2, 1-dif_2), 2)

def euclidean_distance(n1: GridGraph.Node, n2: GridGraph.Node) -> float:
    return dist(n1.get_coords(), n2.get_coords())

def manhattan_distance(n1: GridGraph.Node, n2: GridGraph.Node, n: int, m: int) -> int:
    x1, y1 = n1.get_coords()
    x2, y2 = n2.get_coords()

    dif_x_m = np.abs((x2-x1) % n)
    dif_y_m = np.abs((y2-y1) % m)

    dif_x_n = np.abs((x1-x2) % n)
    dif_y_n = np.abs((y1-y2) % m)


    return np.min([dif_x_m + dif_y_m, dif_x_n + dif_y_n])

if __name__ == '__main__':
    gg = GridGraph(5, 5)
    n1: GridGraph.Node = gg[0][0]
    n2: GridGraph.Node = gg[4][0]

    print(gg.get_knn(n1.get_coords()[0], n1.get_coords()[1], manhattan_distance, 1))
    print(gg)