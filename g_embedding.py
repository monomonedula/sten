import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from pypardiso import spsolve
import networkx as net


class Systems:
    def __init__(self, graph):
        self._graph = graph
        self._system = None

    def results(self, dfactor):
        if not self._system:
            self._system = System(SysLeft(self._graph, dfactor))
        for n in range(len(self._graph)):
            self._system.result(SystemRight(self._graph, n))


class System:
    def __init__(self, left):
        self._left = left
    
    def result(self, right):
        return spsolve(self._left.matrix(), right.matrix())


class SysLeft:
    def __init__(self, graph, dfactor):
        self._graph = graph
        self._dfactor = dfactor
        self._cached = None

    def matrix(self):
        if self._cached is None:
            nodes_number = self._graph.nodes_num
            ones = csr_matrix(np.ones([1, nodes_number]))
            weights = ones.dot(self._graph.adjacency)
            weights = weights.power(-1, np.float) * self._dfactor
            weights = weights.toarray().reshape([nodes_number, ])
            weights = sparse.diags(weights).tocsr()
            matrix = self._graph.adjacency.dot(weights)
            matrix.setdiag(-1.0)
            self._cached = matrix
        return self._cached


class SystemRight:
    def __init__(self, graph, central_node):
        self._graph = graph
        self._central_node = central_node

    def matrix(self):
        sys_right = np.empty([len(self._graph), 1], dtype=float)
        for node in self._graph:        
            sys_right[node] = -1.0 if node == self._central_node else 0.0
        return sys_right


class SparseGraph:
    def __init__(self, matrix):
        self.adjacency = matrix
        self.nodes_num = matrix.shape[0]


def adj(g: net.Graph):
    m = np.empty([g.number_of_nodes(), g.number_of_nodes()])
    for i in range(g.number_of_nodes()):
        for j in range(g.number_of_nodes()):
            m[i, j] = int(g.has_edge(i, j))
    return csr_matrix(m)
