import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from pypardiso import spsolve
import networkx as net
from typing import Callable


class Systems:
    def __init__(
        self,
        system: "SysLeft",
        make_system_right: Callable[[int], "SystemRight"],
        n_nodes: int,
    ):
        self._system = system
        self._right = make_system_right
        self._n_nodes = n_nodes

    def results(self,):
        return np.stack(
            map(lambda n: self._system.result(self._right(n)), range(self._n_nodes))
        ).T


class SysLeft:
    def __init__(self, graph, dfactor):
        self._graph = graph
        self._dfactor = dfactor
        self._cached = None

    def matrix(self):
        """
        Creates left matrix of the system of equations
        for "influence" embedding generation.

        Every element Aij of the matrix is one of the following:
            0 , if there's no edge from the node i to the node j
            -1 , if i = j
            dfactor / (number of edges from* node j), otherwise

        * number of edges of a node here is computed as sum of element of the corresponding column
            of adjacency matrix
        :return: csr matrix
        """
        if self._cached is None:
            matrix = self._graph.adjacency().dot(
                DiagonalMatrix(
                    Multiplied(Reciprocals(SumConnections(self._graph)), self._dfactor)
                ).matrix()
            )
            matrix.setdiag(-1.0)
            self._cached = matrix
        return self._cached

    def result(self, right):
        return spsolve(self.matrix(), right.matrix())


class SumConnections:
    def __init__(self, graph):
        self._graph = graph

    def vector(self):
        """
        Makes a N-element vector of the adjacency matrix NxN
        where i-th element is a sum of elements in the i-th column of the adjacency matrix.

        For a simple graph adjacency
            i-th element of the vector is the number of incident edges if i-th node.

        :return: csr_matrix 1xN where N is the number of nodes in the graph
        """
        ones = csr_matrix(np.ones([1, self._graph.nodes_num()]))
        return ones.dot(self._graph.adjacency())


class Reciprocals:
    def __init__(self, vec):
        self._vec = vec

    def vector(self):
        return self._vec.vector().power(-1, np.float)


class Multiplied:
    def __init__(self, vec, factor):
        self._vec = vec
        self._factor = factor

    def vector(self):
        return self._vec.vector() * self._factor


class DiagonalMatrix:
    def __init__(self, vec):
        self._vec = vec

    def matrix(self):
        vec = self._vec.vector()
        n = vec.shape[1]
        return sparse.diags(vec.toarray().reshape([n]))


class SystemRight:
    def __init__(self, graph, central_node):
        self._graph = graph
        self._central_node = central_node

    def matrix(self):
        sys_right = np.zeros([self._graph.nodes_num(), 1], dtype=float)
        sys_right[self._central_node] = -1.0
        return sys_right


class GraphCSR:
    def __init__(self, graph: net.Graph):
        """
        :param graph: networkx.Graph with nodes as integers from 0 to N where N in the number of nodes
        """
        self._graph = graph
        self._adj = None

    def adjacency(self):
        """
        :return: csr_matrix - adjacency matrix in csr format
        """
        if self._adj is None:
            indptr = [0]
            indices = []
            data = []
            for i in sorted(self._graph.nodes):
                for j in self._graph.neighbors(i):
                    data.append(1)
                    indices.append(j)
                indptr.append(len(indices))
            self._adj = csr_matrix((data, indices, indptr), dtype=int)
        return self._adj

    def nodes_num(self):
        return self._graph.number_of_nodes()


class DiGraphCSR:
    def __init__(self, graph):
        """
        :param graph: networkx.DiGraph with nodes as integers from 0 to N where N in the number of nodes
        """
        self._graph = graph
        self._adj = None

    def adjacency(self):
        """
        :return: csr_matrix - adjacency matrix in csr format
        """
        if self._adj is None:
            adj = GraphCSR(self._graph).adjacency()
            if adj.shape[1] == adj.shape[0] - 1:
                adj = sparse.hstack((adj, np.zeros([len(self._graph), 1], dtype=int)))
            self._adj = adj
        return self._adj

    def nodes_num(self):
        return self._graph.number_of_nodes()
