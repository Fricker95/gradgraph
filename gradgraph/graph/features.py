#!/usr/bin/env python3
# 
# features.py
# 
# Created by Nicolas Fricker on 08/20/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import numpy as np
import networkx as nx

from typing import Iterator

from gradgraph.graph.paths import find_apical_paths
from gradgraph.graph.hash import hash
from gradgraph.graph.utils import (
    add_missing_times,
    split_into_span,
    euclidean_distance
)

def apical_features(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    time_attr: str,
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """
    Iterate over temporal and geometric features from apical paths in a graph.

    This function enumerates all apical paths in ``G`` using
    :func:`find_apical_paths`, then aggregates features along each path.
    For each unique path, it computes:

      - the sequence of nodes and edges,
      - the sorted unique arrival times (from head nodes),
      - the cumulative Euclidean distances between successive nodes.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph. Nodes must contain a time attribute ``time_attr``.
    pos : dict[int, tuple[float, float]]
        Mapping from node ID to spatial coordinates (e.g., 2D or 3D). Each
        node in an apical path must exist in this dictionary.
    time_attr : str
        Name of the node attribute representing time (e.g., time of appearance
        or activation). For each edge (u, v), the time of ``v`` is taken as
        the edge's arrival time.

    Yields
    ------
    key : str
        Unique hash string for the apical path (direction-invariant).
    features : dict of np.ndarray
        Dictionary containing:

        - ``"nodes"`` : ndarray of shape (n,)
            Ordered node IDs along the path.
        - ``"edges"`` : ndarray of shape (n-1, 2)
            Consecutive node pairs defining the path edges.
        - ``"times"`` : ndarray of shape (m,)
            Sorted unique arrival times corresponding to head nodes of edges.
        - ``"dists"`` : ndarray of shape (m,)
            Cumulative Euclidean distances aggregated per time step.

    Notes
    -----
    * Paths are deduplicated: each path and its reverse direction are stored once.
    * Edge distances are computed from coordinates in ``pos`` using the
      Euclidean metric. Missing positions will raise a ``KeyError``.
    * If multiple edges share the same time, their distances are summed before
      accumulation.

    Examples
    --------
    >>> import networkx as nx, numpy as np

    >>> G = nx.Graph()
    >>> G.add_nodes_from([
    ...    (1, {"t": 4., "pos": (0., 0.)}),
    ...    (2, {"t": 3., "pos": (1., 1.)}),
    ...    (3, {"t": 2., "pos": (2., 2.)}),
    ...    (4, {"t": 1., "pos": (1., 3.)}),
    ...    (5, {"t": 1., "pos": (3., 3.)}),
    ... ])
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5)])
    >>> pos = nx.get_node_attributes(G, "pos")
    >>> feats = dict(apical_features(G, pos=pos, time_attr="t"))
    >>> v = next(iter(feats.values()))
    >>> v["nodes"]
    array([3, 2, 1])
    >>> v["edges"]
    array([[3, 2],
           [2, 1]])
    >>> v["times"]
    array([3., 4.])
    >>> v["dists"]
    array([1.41421356, 2.82842712])
    """
    for path in find_apical_paths(G, weight=time_attr):
        nodes = np.asarray(path)
        if nodes.size < 2:
            continue

        key = hash(nodes)
        edges = np.column_stack((nodes[:-1], nodes[1:]))

        node_times = np.array([G.nodes[int(n)].get(time_attr) for n in nodes])
        edge_times = node_times[1:]

        edge_lengths = np.array([
            euclidean_distance(pos[u], pos[v]) for u, v in edges
        ])

        uniq_t, inv = np.unique(edge_times, return_inverse=True)
        per_time_len = np.bincount(inv, weights=edge_lengths, minlength=len(uniq_t)).astype(float)

        times = uniq_t
        dists = np.cumsum(per_time_len)

        yield key, {
            "nodes": nodes,
            "edges": edges,
            "times": times,
            "dists": dists,
        }

def temporal_apical_features(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    time_attr: str,
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """
    Iterate over apical features of a temporal graph at successive time steps.

    At each unique time value present in the node attribute ``time_attr``,
    this function constructs a subgraph of all nodes that have appeared
    at or before that time. It then extracts apical features from this
    subgraph using :func:`apical_features` and yields them as key–value pairs.

    Parameters
    ----------
    G : networkx.Graph
        Undirected input graph where nodes have a temporal attribute
        specified by ``time_attr``.
    pos : dict[int, tuple[float, float]]
        Mapping from node IDs to spatial coordinates (e.g., 2D or 3D
        positions). Used for computing Euclidean edge lengths.
    time_attr : str
        Name of the node attribute representing appearance time.

    Yields
    ------
    (str, dict of np.ndarray)
        For each apical path discovered at a snapshot time step, yields a
        tuple consisting of:

        - key : str
            Unique hash identifying the path (direction-invariant).
        - value : dict of np.ndarray
            Dictionary containing:

            - ``"nodes"`` : ndarray of shape (n,)
                Ordered node IDs along the path.
            - ``"edges"`` : ndarray of shape (n-1, 2)
                Consecutive node pairs forming the path.
            - ``"times"`` : ndarray of shape (m,)
                Sorted unique arrival times associated with head nodes.
            - ``"dists"`` : ndarray of shape (m,)
                Cumulative Euclidean path lengths aggregated by time.

    Notes
    -----
    - Subgraphs are built incrementally at each unique time value,
      including all nodes with appearance time less than or equal to
      the current time.
    - The same apical path may appear in multiple snapshots if it
      persists over time. Deduplication is not performed here.
    - For large graphs, recomputing paths at every snapshot can be
      computationally expensive.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_nodes_from([
    ...    (8, {"t": 7., "pos": (0., 3.)}),
    ...    (7, {"t": 6., "pos": (0., 2.)}),
    ...    (6, {"t": 5., "pos": (0., 1.)}),
    ...    (1, {"t": 4., "pos": (0., 0.)}),
    ...    (2, {"t": 3., "pos": (1., 1.)}),
    ...    (3, {"t": 2., "pos": (2., 2.)}),
    ...    (4, {"t": 1., "pos": (1., 3.)}),
    ...    (5, {"t": 1., "pos": (3., 3.)}),
    ... ])
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5), (6, 1), (7, 6), (8, 7)])
    >>> pos = nx.get_node_attributes(G, "pos")
    >>> feats_iter = temporal_apical_features(G, pos, time_attr="t")
    >>> for k, v in feats_iter:
    ...     print(k, v["times"], v["dists"])
    """
    time_attributes = nx.get_node_attributes(G, time_attr)
    times = list(reversed(sorted(set(time_attributes.values()))))

    for t in times:
        g = G.subgraph((n for n in G.nodes if time_attributes[n] <= t))
        yield from apical_features(g, pos, time_attr)

def windowed_temporal_apical_features(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    span: int,
    dt: float,
    weight: str = "t",
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Generate fixed-size temporal windows of apical features from a temporal graph.

    This function extracts apical paths from ``G`` at each time step using
    :func:`temporal_apical_features`, regularizes their time series of cumulative
    distances with :func:`add_missing_times`, and splits them into overlapping
    windows of length ``span`` using :func:`split_into_span`.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph with temporal node attributes. Each node must provide
        a ``time_attr`` (handled internally by :func:`temporal_apical_features`).
    pos : dict[int, tuple[float, float]]
        Mapping from node IDs to spatial coordinates (2D or 3D). Used for
        computing Euclidean distances along apical paths.
    span : int
        Length of each output window (number of consecutive time steps).
        Paths with fewer than ``span`` time steps after filling are skipped.
    dt : float
        Temporal resolution. Missing times are inserted at multiples of ``dt``
        and their distance values forward-filled.
    weight : str
        Name of the node attribute representing time (e.g., time of appearance,
        activation, or observation). This attribute is used to determine the
        temporal ordering of nodes and edges.

    Yields
    ------
    tuple of (str, (np.ndarray, np.ndarray))
        - key : int
            Unique identifier for the path-window combination, consisting of
            the path hash plus a zero-padded window index.
        - times : np.ndarray of shape (span,)
            Time values for the current window.
        - dists : np.ndarray of shape (span,)
            Cumulative distances for the current window.

    Notes
    -----
    * Windows are overlapping and slide by one time step.
    * Paths are deduplicated under reversal: a path and its reverse share the
      same key.
    * The function yields potentially many windows per path, depending on the
      length of its filled time series.

    Examples
    --------
    >>> import networkx as nx, numpy as np
    >>> G = nx.Graph()
    >>> G.add_nodes_from([
    ...     (1, {"t": 5, "pos": (0., 0.)}),
    ...     (2, {"t": 4, "pos": (1., 1.)}),
    ...     (3, {"t": 3, "pos": (2., 2.)}),
    ...     (4, {"t": 2, "pos": (3., 3.)}),
    ...     (5, {"t": 1, "pos": (4., 4.)}),
    ...     (6, {"t": 1, "pos": (4., 5.)}),
    ...     (7, {"t": 1, "pos": (3., 4.)}),
    ... ])
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)])
    >>> pos = nx.get_node_attributes(G, "pos")
    >>> windows = list(windowed_apical_features(G, pos=pos, span=2, dt=1))
    >>> for key, times, dists in windows:
    ...     print(key, times, dists)
    ea0d7246...00000 ([2 3], [1.41421356 2.82842712])
    ea0d7246...00001 ([3 4], [2.82842712 4.24264069])
    ea0d7246...00002 ([4 5], [4.24264069 5.65685425])
    b0e6f25f...00000 ([2 3], [1.41421356 2.82842712])
    b0e6f25f...00001 ([3 4], [2.82842712 4.24264069])
    59cd57a1...00000 ([2 3], [1.41421356 2.82842712])
    """
    features = temporal_apical_features(G, pos=pos, time_attr=weight)
    for key, values in features:
        times = values['times']
        dists = values['dists']
        times, dists = add_missing_times(times, dists, dt=dt)
        if len(dists) < span:
            continue
        split = split_into_span(times, dists, span=span)
        for i, (split_times, split_dists) in enumerate(split):
            yield f"{key}{i:05d}", (split_times, split_dists)

