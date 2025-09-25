#!/usr/bin/env python3
# 
# test_graph_features.py
# 
# Created by Nicolas Fricker on 08/20/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import pytest
import numpy as np
import networkx as nx

from gradgraph.graph.features import (
    apical_features,
    temporal_apical_features,
)

### MARK: apical_features

def test_apical_features_simple_chain():
    G = nx.Graph()
    G.add_nodes_from([
        (1, {"t": 4., "pos": (0., 0.)}),
        (2, {"t": 3., "pos": (1., 1.)}),
        (3, {"t": 2., "pos": (2., 2.)}),
    ])
    G.add_edges_from([(1, 2), (2, 3)])
    pos = nx.get_node_attributes(G, "pos")
    out = dict(apical_features(G, pos=pos, time_attr="t"))
    assert len(out) == 0

def test_apical_features_multiple_branches_from_junction():
    G = nx.Graph()
    G.add_nodes_from([
        (1, {"t": 4., "pos": (0., 0.)}),
        (2, {"t": 3., "pos": (1., 1.)}),
        (3, {"t": 2., "pos": (2., 2.)}),
        (4, {"t": 1., "pos": (1., 3.)}),
        (5, {"t": 1., "pos": (3., 3.)}),
    ])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5)])
    pos = nx.get_node_attributes(G, "pos")
    out = dict(apical_features(G, pos=pos, time_attr="t"))
    assert len(out) == 1
    v = next(iter(out.values()))
    np.testing.assert_array_equal(v["times"], np.array([3, 4]))
    np.testing.assert_allclose(v["dists"], np.array([1.414213562373, 2.828427124746]), rtol=1e-12)

### MARK: temporal_apical_features

def test_temporal_apical_features_simple_chain():
    G = nx.Graph()
    G.add_nodes_from([
        (1, {"t": 4., "pos": (0., 0.)}),
        (2, {"t": 3., "pos": (1., 1.)}),
        (3, {"t": 2., "pos": (2., 2.)}),
    ])
    G.add_edges_from([(1, 2), (2, 3)])
    pos = nx.get_node_attributes(G, "pos")
    out = dict(temporal_apical_features(G, pos=pos, time_attr="t"))
    assert len(out) == 0

def test_temporal_apical_features_multiple_branches_from_junction():
    G = nx.Graph()
    G.add_nodes_from([
        (8, {"t": 7., "pos": (0., 3.)}),
        (7, {"t": 6., "pos": (0., 2.)}),
        (6, {"t": 5., "pos": (0., 1.)}),
        (1, {"t": 4., "pos": (0., 0.)}),
        (2, {"t": 3., "pos": (1., 1.)}),
        (3, {"t": 2., "pos": (2., 2.)}),
        (4, {"t": 1., "pos": (1., 3.)}),
        (5, {"t": 1., "pos": (3., 3.)}),
    ])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5), (6, 1), (7, 6), (8, 7)])
    pos = nx.get_node_attributes(G, "pos")
    out = dict(temporal_apical_features(G, pos, time_attr="t"))
    assert len(out) == 5
    v = next(iter(out.values()))
    np.testing.assert_array_equal(v["times"], np.array([3, 4, 5, 6, 7]))
    np.testing.assert_allclose(v["dists"], np.array([
        1.414213562373,
        2.828427124746,
        3.828427124746,
        4.828427124746,
        5.828427124746,
    ]), rtol=1e-12)
