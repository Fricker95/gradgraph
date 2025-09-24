#!/usr/bin/env python3
# 
# tests_graph_utils.py
# 
# Created by Nicolas Fricker on 08/28/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import pytest
import numpy as np
import networkx as nx

from gradgraph.graph.utils import (
    remove_degree_k_nodes,
    relabel_negative_nodes,
    spectral_gap,
    add_missing_times,
    split_into_span,
    euclidean_distance
)

### MARK: remove_degree_k_nodes

def test_remove_degree_k_nodes_degree_2_path_collapses_with_aggregated_weights():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=2)
    G.add_edge(3, 4, weight=3)

    H = remove_degree_k_nodes(G, degree=2, weight="weight")
    assert set(H.nodes) == {1, 4}
    assert set(H.edges) == {(1, 4)}
    assert H[1][4]["weight"] == 6


def test_remove_degree_k_nodes_existing_edge_weight_is_aggregated_not_overwritten():
    u, v, x, y = "u", "v", "x", "y"
    G = nx.Graph()
    G.add_edge(u, x, weight=10)
    G.add_edge(x, y, weight=2)
    G.add_edge(y, v, weight=5)

    H = remove_degree_k_nodes(G, degree=2, weight="weight")
    assert set(H.nodes) == {u, v}
    assert H[u][v]["weight"] == 17


def test_remove_degree_k_nodes_missing_weight_defaults_to_1():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])

    H = remove_degree_k_nodes(G, degree=2, weight="weight")
    assert set(H.nodes) == {1, 3}
    assert H[1][3]["weight"] == 2


def test_remove_degree_k_nodes_custom_aggregation_function_max():
    def agg_max(vals):
        return max(vals)

    G = nx.Graph()
    G.add_edge(1, 2, weight=2)
    G.add_edge(2, 3, weight=5)

    H = remove_degree_k_nodes(G, degree=2, weight="weight", agg_func=agg_max)
    assert set(H.nodes) == {1, 3}
    assert H[1][3]["weight"] == 5


def test_remove_degree_k_nodes_iterative_removal_for_degree_1_leaves():
    G = nx.path_graph(3)  # nodes 0-1-2, but rename to 1-2-3 for clarity
    G = nx.relabel_nodes(G, {0: 1, 1: 2, 2: 3})

    H = remove_degree_k_nodes(G, degree=1, weight="weight")
    assert set(H.nodes) == {2}
    assert H.number_of_edges() == 0


def test_remove_degree_k_nodes_no_change_when_no_nodes_match_degree():
    G = nx.cycle_graph(4)  # all degree 2
    H = remove_degree_k_nodes(G, degree=3, weight="weight")
    assert set(H.nodes) == set(G.nodes)
    assert set(H.edges) == set(G.edges)
    assert H is not G

### MARK: relabel_negative_nodes

def test_relabel_negatives_according_to_docstring_example():
    G = nx.Graph()
    G.add_edges_from([(-3, 1), (-2, 5)])
    G2 = relabel_negative_nodes(G)

    assert set(G2.nodes) == {1, 5, 7, 8}
    assert set(map(frozenset, G2.edges)) == {frozenset({8, 1}), frozenset({7, 5})}


def test_relabel_preserves_attributes_and_is_copy():
    G = nx.Graph()
    G.add_node(-4, color="red")
    G.add_node(10, color="blue")
    G.add_edge(-4, 10, weight=7)

    G2 = relabel_negative_nodes(G)

    assert G2 is not G

    expected_new = 14
    assert expected_new in G2.nodes
    assert G2.nodes[expected_new]["color"] == "red"
    assert G2.nodes[10]["color"] == "blue"
    assert G2[expected_new][10]["weight"] == 7


def tes_relabel_all_negative_nodes_become_non_negative_and_unique():
    G = nx.Graph()
    G.add_edges_from([(-5, -2), (-2, -1)])
    G2 = relabel_negative_nodes(G)

    for n in G2.nodes:
        assert isinstance(n, int)
        assert n >= 0

    assert len(G2.nodes) == len(set(G2.nodes))


def test_relabel_no_change_when_no_negative_nodes():
    G = nx.path_graph([1, 2, 3])
    G2 = relabel_negative_nodes(G)
    assert set(G2.nodes) == {1, 2, 3}
    assert set(G2.edges) == { (1,2), (2,3) }
    assert G2 is not G

### MARK: spectral_gap

def test_spectral_gap_single_node_zero():
    G = nx.Graph()
    G.add_node(0)
    assert spectral_gap(G) == 0.0

def test_spectral_gap_two_node_edge_equals_2():
    G = nx.Graph()
    G.add_edge(0, 1)
    assert np.isclose(spectral_gap(G), 2.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [3, 4, 7, 10])
def test_spectral_gap_complete_graph_matches_n_over_n_minus_1(n):
    G = nx.complete_graph(n)
    expected = n / (n - 1)
    assert np.isclose(spectral_gap(G), expected, rtol=0, atol=1e-12)


def test_spectral_gap_disconnected_returns_0():
    G = nx.empty_graph(5)
    assert spectral_gap(G) == 0.0


def test_spectral_gap_epsilon_thresholding():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_node(2)
    assert np.isclose(spectral_gap(G, epsilon=1.5), 2.0, rtol=0, atol=1e-12)
    assert spectral_gap(G, epsilon=2.1) == 0.0


@pytest.mark.parametrize("n", [5, 10, 15])
def test_spectral_gap_cycle_matches_1_minus_cos_formula(n):
    G = nx.cycle_graph(n)
    expected = 1.0 - np.cos(2.0 * np.pi / n)
    assert np.isclose(spectral_gap(G), expected, rtol=0, atol=1e-12)

### MARK: add_missing_times

def test_add_missing_times_short_trim():
    t = np.array([0, 36])
    d = np.array([0, 1])
    out = add_missing_times(t, d, dt=18)
    expected = (
        np.array([0, 18, 36]),
        np.array([0, 0, 1]),
    )
    np.testing.assert_array_equal(out[0], expected[0])
    np.testing.assert_array_equal(out[1], expected[1])

def test_add_missing_times_short():
    t = np.array([36, 72])
    d = np.array([1, 2])
    out = add_missing_times(t, d, dt=18)
    expected = (
        np.array([36, 54, 72]),
        np.array([1, 1, 2]),
    )
    np.testing.assert_array_equal(out[0], expected[0])
    np.testing.assert_array_equal(out[1], expected[1])

def test_add_missing_times_long():
    t = np.array([0, 36, 90])
    d = np.array([0, 1, 3])
    out = add_missing_times(t, d, dt=18)
    expected = (
        np.array([0, 18, 36, 54, 72, 90]),
        np.array([0, 0, 1, 1, 1, 3]),
    )
    np.testing.assert_array_equal(out[0], expected[0])
    np.testing.assert_array_equal(out[1], expected[1])

### MARK: split_into_span

def test_split_into_span_basic_overlapping_windows():
    t = np.array([0, 1, 2, 3, 4])
    d = np.array([10, 11, 12, 13, 14])
    windows = list(split_into_span(t, d, span=3))

    assert len(windows) == 3
    np.testing.assert_array_equal(windows[0][0], [0, 1, 2])
    np.testing.assert_array_equal(windows[0][1], [10, 11, 12])
    np.testing.assert_array_equal(windows[1][0], [1, 2, 3])
    np.testing.assert_array_equal(windows[1][1], [11, 12, 13])
    np.testing.assert_array_equal(windows[2][0], [2, 3, 4])
    np.testing.assert_array_equal(windows[2][1], [12, 13, 14])

def test_split_into_span_span_equals_length():
    t = np.array([5, 6, 7])
    d = np.array([1.0, 1.5, 2.0])
    windows = list(split_into_span(t, d, span=3))

    assert len(windows) == 1
    tw, dw = windows[0]
    np.testing.assert_array_equal(tw, t)
    np.testing.assert_array_equal(dw, d)

def test_split_into_span_span_one_yields_all_singletons():
    t = np.array([10, 20, 30])
    d = np.array([3, 4, 5])
    windows = list(split_into_span(t, d, span=1))

    assert len(windows) == 3
    for i, (tw, dw) in enumerate(windows):
        np.testing.assert_array_equal(tw, [t[i]])
        np.testing.assert_array_equal(dw, [d[i]])

def test_split_into_span_mismatched_shapes_raises():
    t = np.array([0, 1, 2])
    d = np.array([10, 11])
    with pytest.raises(AssertionError):
        _ = list(split_into_span(t, d, span=2))

def test_split_into_span_span_greater_than_length_raises():
    t = np.array([0, 1])
    d = np.array([10, 11])
    with pytest.raises(ValueError):
        _ = list(split_into_span(t, d, span=3))

def test_split_into_span_dtype_is_preserved():
    t = np.array([0, 1, 2], dtype=np.int64)
    d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    (tw, dw), *_ = list(split_into_span(t, d, span=2))

    assert tw.dtype == t.dtype
    assert dw.dtype == d.dtype

@pytest.mark.parametrize("span", [0, 1, 2])
def test_split_into_span_empty_and_zero_span_behavior(span):
    """
    Current semantics:
    - span=0 is allowed by the implementation and yields len(t)+1 empty windows.
    - For empty arrays and span=0, yields one empty window.
    """
    t = np.array([0, 1])
    d = np.array([10, 11])

    collect_windows = lambda t, d, span: list(split_into_span(t, d, span))

    if span == 0:
        wins = collect_windows(t, d, span=0)
        assert len(wins) == len(t) + 1
        for tw, dw in wins:
            assert tw.size == 0 and dw.size == 0
    else:
        wins = collect_windows(t, d, span=span)
        assert all(len(tw) == span and len(dw) == span for tw, dw in wins)

    t0 = np.array([])
    d0 = np.array([])
    if span == 0:
        wins0 = collect_windows(t0, d0, span=0)
        assert len(wins0) == 1
        tw, dw = wins0[0]
        assert tw.size == 0 and dw.size == 0
    else:
        with pytest.raises(ValueError):
            _ = list(split_into_span(t0, d0, span=span))

### MARK: euclidean_distance

def test_euclidean_distance_returns_float_and_basic_2d():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    d = euclidean_distance(a, b)
    assert isinstance(d, float)
    assert d == 5.0

def test_euclidean_distance_symmetry():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([-4.0, 0.5, 7.0])
    dab = euclidean_distance(a, b)
    dba = euclidean_distance(b, a)
    assert np.isclose(dab, dba)

def test_euclidean_distance_triangle_inequality_simple():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([1.0, 1.0])
    ab = euclidean_distance(a, b)
    bc = euclidean_distance(b, c)
    ac = euclidean_distance(a, c)
    assert ac <= ab + bc + 1e-12

def test_euclidean_distance_shape_mismatch_raises_assertion():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        euclidean_distance(a, b)

def test_euclidean_distance_non_1d_input_raises_valueerror():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    with pytest.raises(ValueError, match="1D"):
        euclidean_distance(a, b)

    a = np.array([0.0, 0.0])
    b = np.array([[3.0, 4.0]])
    with pytest.raises(ValueError, match="1D"):
        euclidean_distance(a, b)

def test_euclidean_distance_inputs_not_modified():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([4.0, 6.0], dtype=np.int32)
    a_copy = a.copy()
    b_copy = b.copy()
    _ = euclidean_distance(a, b)
    assert np.array_equal(a, a_copy)
    assert np.array_equal(b, b_copy)

def test_euclidean_distance_large_dimension():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000)
    expected = float(np.linalg.norm(a - b))
    got = euclidean_distance(a, b)
    assert np.isclose(got, expected, rtol=1e-12, atol=0.0)
