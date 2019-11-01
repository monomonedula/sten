import pytest
import networkx as net
import numpy as np

from g_embedding import SysLeft, SparseGraph, adj


def g_m():
    return [
        [
            net.Graph(
                [
                    (0, 1),
                    (0, 2),
                    (2, 3),
                    (2, 4),
                    (3, 4),
                    (4, 5),
                    (4, 6),
                    (5, 6),
                    (6, 7),
                ]
            ),
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


@pytest.mark.parametrize("graph_matrix", g_m())
def test_system_left(graph_matrix):
    graph, desired = graph_matrix
    actual = SysLeft(SparseGraph(adj(graph)), 0.89).matrix().toarray()
    np.testing.assert_almost_equal(actual, desired)
