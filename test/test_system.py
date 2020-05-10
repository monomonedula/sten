import pytest
import networkx as net
import numpy as np

from simple_nodes_embedding.embedding import (
    SysLeft,
    GraphCSR,
    DiGraphCSR,
    Systems,
    SystemRight,
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
                        2.26086388,
                        1.20047656,
                        0.83356053,
                        0.67269317,
                        0.51182581,
                        0.47747719,
                        0.4249547,
                    ],
                    [
                        1.13043194,
                        2.00608443,
                        0.53421207,
                        0.37093444,
                        0.29934846,
                        0.22776249,
                        0.21247735,
                        0.18910484,
                    ],
                    [
                        1.80071484,
                        1.6026362,
                        2.44391399,
                        1.69695129,
                        1.36945969,
                        1.04196809,
                        0.97204163,
                        0.86511705,
                    ],
                    [
                        0.83356053,
                        0.74186887,
                        1.13130086,
                        1.95809393,
                        1.02171921,
                        0.77738602,
                        0.72521566,
                        0.64544194,
                    ],
                    [
                        1.34538634,
                        1.19739384,
                        1.82594625,
                        2.04343843,
                        2.76605022,
                        2.10457897,
                        1.96334072,
                        1.74737324,
                    ],
                    [
                        0.51182581,
                        0.45552497,
                        0.69464539,
                        0.77738602,
                        1.05228949,
                        2.01923447,
                        1.23812506,
                        1.1019313,
                    ],
                    [
                        0.71621579,
                        0.63743205,
                        0.97204163,
                        1.08782349,
                        1.47250554,
                        1.85718759,
                        2.70094972,
                        2.40384525,
                    ],
                    [
                        0.21247735,
                        0.18910484,
                        0.28837235,
                        0.32272097,
                        0.43684331,
                        0.55096565,
                        0.80128175,
                        1.71314076,
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
