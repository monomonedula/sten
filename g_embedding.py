import numpy as np
from scipy.sparse import csr_matrix
from pypardiso import spsolve
import networkx as net


class Systems:
    def __init__(self, graph):
        self._graph = graph
        self._system = None

    def results(self, dfactor):
        def coefficient(node, row_node):
            if node == row_node:
                return dfactor / len(self._graph[node])
            return -1.0

        if not self._system:
            self._system = System(SystemLeft(self._graph, coefficient))
        for n in range(len(self._graph)):
            self._system.result(SystemRight(self._graph, n))


class System:
    def __init__(self, left):
        self._left = left
    
    def result(self, right):
        return spsolve(self._left.matrix(), right.matrix())


class SystemLeft:
    def __init__(self, graph, coefficient):
        self._graph = graph
        self._cached = None
        self._coeff = coefficient

    def matrix(self):
        if self._cached:
            return self._cached
        row_length = (self._graph.vertices_num * 2 + self._graph.nodes_num)
        print(row_length)
        left_data = np.empty([row_length], dtype=float)
        left_indices = np.empty([row_length], dtype=int)
        left_indptr = np.empty(len(self._graph) + 1, dtype=int)
        left_indptr[0] = 0
        counter = 0    
        for row_node in range(len(self._graph)):
            neighbors = self._graph[row_node]
            for node in neighbors:
                left_data[counter] = self._coeff(node, row_node)
                left_indices[counter] = node
                counter += 1
            left_indptr[row_node + 1] = counter
        return csr_matrix((left_data, left_indices, left_indptr), dtype=float)


class SystemRight:
    def __init__(self, graph, central_node):
        self._graph = graph
        self._central = central_node

    def matrix(self):
        sys_right = np.empty([len(self._graph), 1], dtype=float)
        for node in self._graph:        
            sys_right[node] = -1.0 if node == self._central else 0.0
        return sys_right


class Graph:
    def __init__(self, neighbors_map, nodes_num, vertices_num):
        self.neighbors_map = neighbors_map
        self.nodes_num = nodes_num
        self.vertices_num = vertices_num

    def __getitem__(self, key):
        return self.neighbors_map[key]

    def __iter__(self):
        return iter(range(self.nodes_num))
    
    def __len__(self):
        return self.nodes_num

    @classmethod
    def from_networkx(cls, g: net.Graph):
        return cls(
            {n: np.array(sorted((*g.neighbors(n), n)), dtype=np.int32) for n in g},
            g.number_of_nodes(),
            g.number_of_edges(),
        )
