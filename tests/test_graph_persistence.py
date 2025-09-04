#!/usr/bin/env python3
# 
# test_graph_persistence.py
# 
# Created by Nicolas Fricker on 09/04/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import pytest
import math
import networkx as nx

from gradgraph.graph.persistence import (
    compute_gudhi_persistence,
    find_local_curvatures,
    find_spectral_gaps
)

def test_compute_gudhi_persistence_empty_graph_yields_nothing():
    G = nx.Graph()
    out = list(compute_gudhi_persistence(G, weight="w"))
    assert out == []

def test_compute_gudhi_persistence_single_isolated_vertex_yields_nothing():
    G = nx.Graph()
    G.add_node(0, w=0.42)
    out = list(compute_gudhi_persistence(G, weight="w"))
    assert out == []

def test_compute_gudhi_persistence_raises_on_negative_node_ids():
    G = nx.Graph()
    G.add_node(-1, w=0.1)
    G.add_node(0, w=0.2)
    G.add_edge(-1, 0)
    with pytest.raises(ValueError):
        _ = list(compute_gudhi_persistence(G, weight="w"))

def test_compute_gudhi_persistence_raises_on_missing_weight_attribute():
    G = nx.Graph()
    G.add_node(0, w=0.1)
    G.add_node(1)  # missing 'w'
    G.add_edge(0, 1)
    with pytest.raises(KeyError):
        _ = list(compute_gudhi_persistence(G, weight="w"))

def test_compute_gudhi_persistence_single_edge_returns_expected_h0_interval():
    G = nx.Graph()
    G.add_node(0, w=0.1)
    G.add_node(1, w=0.3)
    G.add_edge(0, 1)

    out = list(compute_gudhi_persistence(G, weight="w"))
    assert any(
        dim == 0 and math.isclose(birth, 0.3, rel_tol=1e-9, abs_tol=1e-12) and math.isinf(death)
        for dim, (birth, death) in out
    ), f"Unexpected persistence output: {out}"

def test_compute_gudhi_persistence_filtration_uses_max_endpoint_weight():
    G = nx.Graph()
    G.add_node(0, w=0.2)
    G.add_node(1, w=0.5)
    G.add_edge(0, 1)

    out = list(compute_gudhi_persistence(G, weight="w"))

    assert any(
        dim == 0 and math.isclose(birth, 0.5, rel_tol=1e-9, abs_tol=1e-12)
        for dim, (birth, death) in out
    ), f"Expected an H0 interval born at 0.5, got: {out}"

def test_compute_gudhi_persistence_weight_field_name_is_respected():
    G = nx.Graph()
    G.add_node(0, z=0.9)
    G.add_node(1, z=1.1)
    G.add_edge(0, 1)

    out = list(compute_gudhi_persistence(G, weight="z"))

    assert any(
        dim == 0 and math.isclose(birth, 1.1, rel_tol=1e-9, abs_tol=1e-12)
        for dim, (birth, death) in out
    ), f"Expected an H0 interval born at 1.1 with weight='z', got: {out}"

def test_find_local_curvatures_empty_graph_returns_empty():
    G = nx.Graph()
    pos = {}
    out = list(find_local_curvatures(G, pos, radius=1.0))
    assert out == []

def test_find_local_curvatures_returns_empty_when_no_high_degree_nodes():
    G = nx.path_graph(5)
    pos = {i: (float(i), 0.0) for i in G.nodes()}
    out = list(find_local_curvatures(G, pos, radius=2.0))
    assert out == []

def test_find_local_curvatures_counts_connected_high_degree_within_radius():
    G = nx.Graph()
    A, B = 0, 1
    G.add_nodes_from([A, B])
    for u in [10, 11, 12]:
        G.add_edge(A, u)
    for v in [20, 21, 22]:
        G.add_edge(B, v)
    G.add_edge(A, B)
    pos = {
        A: (0.0, 0.0),
        B: (0.5, 0.0),
        10: (-0.2, 0.2), 11: (-0.2, -0.2), 12: (0.0, 0.3),
        20: (0.7, 0.2), 21: (0.7, -0.2), 22: (0.5, 0.3),
    }
    result = dict(find_local_curvatures(G, pos, radius=0.6))
    assert result == {A: 1, B: 1}

def test_find_local_curvatures_respects_radius_and_yields_nothing_when_isolated():
    G = nx.Graph()
    A, B = 0, 1
    G.add_nodes_from([A, B])
    for u in [10, 11, 12]:
        G.add_edge(A, u)
    for v in [20, 21, 22]:
        G.add_edge(B, v)
    G.add_edge(A, B)
    pos = {
        A: (0.0, 0.0),
        B: (0.5, 0.0),
        10: (-0.2, 0.2), 11: (-0.2, -0.2), 12: (0.0, 0.3),
        20: (0.7, 0.2), 21: (0.7, -0.2), 22: (0.5, 0.3),
    }
    out = list(find_local_curvatures(G, pos, radius=0.4))
    assert out == []

def test_find_local_curvatures_mixed_connectivity_counts_only_existing_edges():
    G = nx.Graph()
    A, B, C = 0, 1, 2
    G.add_nodes_from([A, B, C])
    for hub, leaves in [(A, [10, 11, 12]), (B, [20, 21, 22]), (C, [30, 31, 32])]:
        for leaf in leaves:
            G.add_edge(hub, leaf)
    G.add_edge(A, B)
    G.add_edge(B, C)
    pos = {
        A: (0.0, 0.0),
        B: (0.4, 0.0),
        C: (0.3, 0.3),
        10: (-0.2, 0.2), 11: (-0.2, -0.2), 12: (0.0, 0.3),
        20: (0.6, 0.2), 21: (0.6, -0.2), 22: (0.4, 0.3),
        30: (0.1, 0.5), 31: (0.5, 0.5), 32: (0.3, 0.6),
    }
    result = dict(find_local_curvatures(G, pos, radius=0.6))
    assert result[A] == 1 # sees B and C but connected only to B
    assert result[B] == 2 # connected to both A and C
    assert result[C] == 1 # sees A and B but connected only to B

def test_find_local_curvatures_excludes_self_and_yields_empty_if_no_other_high_degree():
    G = nx.Graph()
    H = 0
    for u in [10, 11, 12]:
        G.add_edge(H, u)
    pos = {H: (0.0, 0.0), 10: (1.0, 0.0), 11: (0.0, 1.0), 12: (-1.0, 0.0)}
    out = list(find_local_curvatures(G, pos, radius=10.0))
    assert out == []

def test_find_local_curvatures_positions_needed_only_for_high_degree_nodes():
    G = nx.Graph()
    A, B = 0, 1
    for u in [10, 11, 12]:
        G.add_edge(A, u)
    for v in [20, 21, 22]:
        G.add_edge(B, v)
    G.add_edge(A, B)
    pos = {A: (0.0, 0.0), B: (0.5, 0.0)}
    result = dict(find_local_curvatures(G, pos, radius=1.0))
    assert result == {A: 1, B: 1}

def test_find_spectral_gaps_empty_graph_returns_empty():
    G = nx.Graph()
    pos = {}
    out = list(find_spectral_gaps(G, pos, radius=1.0))
    assert out == []

def test_find_spectral_gaps_raises_on_missing_position_mapping():
    G = nx.path_graph(2)
    pos = {0: (0.0, 0.0)}
    with pytest.raises(KeyError):
        _ = list(find_spectral_gaps(G, pos, radius=1.0))

def test_find_spectral_gaps_single_isolated_node_returns_zero():
    G = nx.Graph()
    G.add_node(0)
    pos = {0: (0.0, 0.0)}
    result = dict(find_spectral_gaps(G, pos, radius=0.0))
    assert 0 in result
    assert result[0] == 0.0

