import pytest
import networkx as net
import numpy as np

from sten.embedding import (
    SysLeft,
    GraphCSR,
    Systems,
    SystemRight,
    DiGraphCSR,
)


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


def g_r():
    return [
        [
            net.Graph(
                [(0, 1), (0, 2), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7)]
            ),
            0.89,
            np.array(
                [
                    [
                        2.54029649,
                        2.26086384,
                        1.20047664,
                        0.83356057,
                        0.67269321,
                        0.51182585,
                        0.47747725,
                        0.42495475,
                    ],
                    [
                        1.13043192,
                        2.00608439,
                        0.5342121,
                        0.37093445,
                        0.29934847,
                        0.2277625,
                        0.21247737,
                        0.18910486,
                    ],
                    [
                        1.80071484,
                        1.60263618,
                        2.44391408,
                        1.69695131,
                        1.36945972,
                        1.04196812,
                        0.97204172,
                        0.86511712,
                    ],
                    [
                        0.83356057,
                        0.74186889,
                        1.13130095,
                        1.95809398,
                        1.02171926,
                        0.77738606,
                        0.72521574,
                        0.645442,
                    ],
                    [
                        1.34538641,
                        1.19739389,
                        1.82594641,
                        2.04343851,
                        2.76605032,
                        2.10457907,
                        1.96334092,
                        1.74737339,
                    ],
                    [
                        0.51182585,
                        0.455525,
                        0.69464546,
                        0.77738606,
                        1.05228953,
                        2.01923453,
                        1.23812518,
                        1.1019314,
                    ],
                    [
                        0.71621583,
                        0.63743208,
                        0.97204172,
                        1.08782354,
                        1.47250559,
                        1.85718765,
                        2.70094987,
                        2.40384534,
                    ],
                    [
                        0.21247737,
                        0.18910486,
                        0.28837239,
                        0.322721,
                        0.43684335,
                        0.5509657,
                        0.80128184,
                        1.71314082,
                    ],
                ]
            ),
        ]
    ]


@pytest.mark.parametrize("graph_factor_matrix", g_m())
def test_system_left(graph_factor_matrix):
    graph, dfactor, desired = graph_factor_matrix
    actual = SysLeft(GraphCSR(graph), dfactor).matrix().toarray()
    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.parametrize("graph_factor_result", g_r())
def test_systems(graph_factor_result):
    graph, dfactor, desired = graph_factor_result
    actual = Systems(
        system=SysLeft(GraphCSR(graph), dfactor),
        make_system_right=lambda n: SystemRight(GraphCSR(graph), n),
        n_nodes=graph.number_of_nodes(),
    ).results()
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
