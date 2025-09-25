#!/usr/bin/env python3
# 
# test_graph_paths.py
# 
# Created by Nicolas Fricker on 08/20/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import numpy as np
import networkx as nx

from gradgraph.graph.paths import find_apical_paths

def _add_branch(G, path, start_weight=1, step=1):
    for i, j in zip(path[:-1], path[1:]):
        G.add_node(i, weight=start_weight)
        start_weight += step
        G.add_node(j, weight=start_weight)
        G.add_edge(i, j, weight=start_weight)
        start_weight += step
    return G


### MARK: find_apical_paths

def test_find_apical_paths_single_branch():
    G = nx.Graph()
    G = _add_branch(G, [0, 1, 4, 5], start_weight=1)
    paths = list(find_apical_paths(G, weight="weight"))
    assert len(paths) == 0

def test_find_apical_paths_subcubic_example():
    G = nx.Graph()
    G = _add_branch(G, [0, 1, 4, 5], start_weight=1)
    G = _add_branch(G, [0, 2, 6], start_weight=1)
    G = _add_branch(G, [0, 3, 7, 8, 9], start_weight=1)
    paths = list(find_apical_paths(G, weight="weight"))
    expected = [
        np.array([0, 1, 4, 5]),
        np.array([0, 2, 6]),
        np.array([0, 3, 7, 8, 9]),
    ]
    assert len(paths) == 3
    for x, y in zip(paths, expected):
        np.testing.assert_array_equal(x, y)

def test_find_apical_paths_subcubic_example_reversed():
    G = nx.Graph()
    G = _add_branch(G, [0, 1, 4, 5], start_weight=1)
    G = _add_branch(G, [0, 2, 6], start_weight=3, step=-1)
    G = _add_branch(G, [0, 3, 7, 8, 9], start_weight=1)
    paths = list(find_apical_paths(G, weight="weight"))
    expected = [
        np.array([0, 1, 4, 5]),
        np.array([0, 3, 7, 8, 9]),
    ]
    assert len(paths) == 2
    for x, y in zip(paths, expected):
        np.testing.assert_array_equal(x, y)
