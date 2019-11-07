import pytest
import networkx as net
import numpy as np

from g_embedding import SysLeft, GraphCSR, DiGraphCSR


def g_m():
    return [
        [
            net.Graph(
                [(0, 1), (0, 2), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7)]
            ),
            0.89,
            np.array(
                [
                    [-1.0, 0.89, 0.29666667, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.445, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.445, 0.0, -1.0, 0.445, 0.2225, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.29666667, -1.0, 0.2225, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.29666667, 0.445, -1.0, 0.445, 0.29666667, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.2225, -1.0, 0.29666667, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.2225, 0.445, -1.0, 0.89],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29666667, -1.0],
                ]
            ),
        ]
    ]


@pytest.mark.parametrize("graph_factor_matrix", g_m())
def test_system_left(graph_factor_matrix):
    graph, dfactor, desired = graph_factor_matrix
    actual = SysLeft(GraphCSR(graph), dfactor).matrix().toarray()
    np.testing.assert_almost_equal(actual, desired)


def dg_m():
    items = []
    dg = net.DiGraph()
    dg.add_nodes_from(range(0, 8))
    dg.add_edge(0, 1)
    dg.add_edge(0, 3)
    dg.add_edge(2, 3)
    dg.add_edge(2, 4)
    dg.add_edge(3, 4)
    dg.add_edge(4, 5)
    dg.add_edge(4, 6)
    dg.add_edge(5, 6)
    dg.add_edge(7, 6)
    items.append(dg)
    return [
        [
            dg,
            0.9,
            np.array(
                [
                    [-1.0, 0.9, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.45, 0.45, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0, 0.45, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.9, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, -1.0],
                ]
            ),
        ]
    ]


@pytest.mark.parametrize("digraph_factor_matrix", dg_m())
def test_system_left_directed(digraph_factor_matrix):
    graph, dfactor, desired = digraph_factor_matrix
    actual = SysLeft(DiGraphCSR(graph), dfactor).matrix().toarray()
    np.testing.assert_almost_equal(actual, desired)
