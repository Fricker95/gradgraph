#!/usr/bin/env python3
# 
# utils.py
# 
# Created by Nicolas Fricker on 08/28/2025.
# 

import numpy as np
import networkx as nx

from typing import Iterator

def remove_degree_k_nodes(G: nx.Graph, degree: int, weight: str, agg_func=sum) -> nx.Graph:
    """
    Remove nodes of a given degree from a graph while preserving connectivity by linking their neighbors and aggregating edge weights.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. A copy of this graph is modified and returned.
    degree : int
        Target degree of nodes to be removed. Only nodes with exactly this
        degree will be considered for removal.
    weight : str
        Name of the edge attribute representing the weight. If missing, edges
        are assumed to have weight 1.
    agg_func : callable, optional
        Aggregation function used to combine weights when creating new edges
        between neighbors. Must accept an iterable of numbers and return a
        single number. Default is `sum`.

    Returns
    -------
    networkx.Graph
        A new graph with nodes of the specified degree removed and replaced
        by edges connecting their neighbors. Edge weights are aggregated
        according to `agg_func`.

    Notes
    -----
    - Nodes are removed iteratively: after each removal, the degrees are
      recomputed, and new nodes matching the target degree may be removed.
    - If an edge already exists between two neighbors, its weight is updated
      by aggregating the existing weight with the new weight.
    - Only nodes with *exactly* `degree` neighbors are removed.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=1)
    >>> G.add_edge(2, 3, weight=2)
    >>> G.add_edge(3, 4, weight=3)
    >>> G.add_edge(4, 5, weight=3)
    >>> G.add_edge(4, 6, weight=3)
    >>> H = remove_degree_k_nodes(G, degree=2, weight="weight")
    >>> list(H.edges(data=True))
    [(1, 4, {'weight': 6}), (4, 5, {'weight': 3}), (4, 6, {'weight': 3})]
    """
    G = G.copy()

    degrees = G.degree()

    while True:
        target_nodes = [n for n in G.nodes if degrees[n] == degree]
        if not target_nodes:
            break
        for node in target_nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) != degree:
                continue
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    u, v = neighbors[i], neighbors[j]
                    uw = G[node][u].get(weight, 1)
                    vw = G[node][v].get(weight, 1)
                    new_weight = agg_func([uw, vw])
                    if G.has_edge(u, v):
                        G[u][v][weight] = agg_func([G[u][v].get(weight, 1), new_weight])
                    else:
                        G.add_edge(u, v, **{weight: new_weight})
            G.remove_node(node)
    return G

def relabel_negative_nodes(G: nx.Graph) -> nx.Graph:
    """
    Relabels nodes with negative IDs to positive ones.

    Each negative node `n` is relabeled as `max_node_id - abs(n)`,
    where `max_node_id` is the largest existing node ID.

    Parameters
    ----------
    G : nx.Graph
        The input graph containing negative node IDs.

    Returns
    -------
    nx.Graph
        A new graph with relabeled node IDs. The original graph is unchanged.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(-3, 1), (-2, 5)])
    >>> G2 = relabel_negative_nodes(G)
    >>> list(G2.nodes)
    [1, 5, 7, 8]

    Notes
    -----
    This function assumes that node IDs are integers. The relabeling
    is performed such that the resulting node IDs are positive and
    distinct from existing positive node IDs.
    """
    max_id = max(G.nodes)
    mapping = {n: max_id - n for n in G.nodes if n < 0}
    return nx.relabel_nodes(G, mapping)

def spectral_gap(
    G: nx.Graph,
    epsilon: float = 1e-8
) -> float:
    """
    Compute the spectral gap (algebraic connectivity) of a graph.

    The spectral gap is defined as the smallest nonzero eigenvalue of the
    normalized Laplacian matrix of the graph. This value, also known as the
    Fiedler value, measures how well connected the graph is. If the graph is
    disconnected, the spectral gap is zero.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    epsilon : float, optional
        Threshold for considering an eigenvalue as nonzero. Defaults to 1e-8.

    Returns
    -------
    float
        The spectral gap of the graph, i.e., the smallest eigenvalue greater
        than ``epsilon``. Returns 0 if no such eigenvalue exists.

    Notes
    -----
    - The normalized Laplacian is defined as
      :math:`L = I - D^{-1/2} A D^{-1/2}`,
      where :math:`A` is the adjacency matrix and :math:`D` the degree matrix.
    - The multiplicity of the zero eigenvalue corresponds to the number of
      connected components in the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(4)
    >>> spectral_gap(G)
    0.49999999999999994
    >>> nx.is_connected(G)
    True
    """
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    nonzero_eigenvalues = eigenvalues[eigenvalues > epsilon]
    return nonzero_eigenvalues[0] if len(nonzero_eigenvalues) > 0 else 0

def add_missing_times(t: np.ndarray, d: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert missing timestamps in a 1-D time series using a fixed step ``dt``.

    This function constructs a uniform time grid starting at ``t[0]`` and ending
    at the last timestamp ``t[-1]`` (inclusive). Any timestamps missing from
    the original array ``t`` are inserted, and their corresponding data values
    in ``d`` are forward-filled from the most recent available value. If a
    timestamp is inserted before the first sample, its value is set to zero.
    Leading zero values in ``d`` are removed after filling, together with their
    corresponding timestamps.

    Parameters
    ----------
    t : np.ndarray
        One-dimensional array of timestamps, assumed to be sorted in increasing
        order. Must have the same length as ``d``.
    d : np.ndarray
        One-dimensional array of data values corresponding to ``t``.
    dt : float
        Positive time step used to build the expected timestamp grid.

    Returns
    -------
    t : np.ndarray
        Updated array of timestamps with missing values inserted.
    d : np.ndarray
        Updated array of data values with forward-filled entries.

    Raises
    ------
    ValueError
        If ``t`` or ``d`` is not one-dimensional, or if ``dt`` is not positive.

    Notes
    -----
    - If all values in ``d`` are zero, the input arrays are returned unchanged.
    - Forward filling ensures that each new timestamp inherits the value of the
      most recent earlier time. If the missing timestamp occurs before the
      first sample, the filled value is ``0``.
    - After insertion, any leading zeros in ``d`` (and their timestamps) are
      removed so that the series begins with the first nonzero entry.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([5])
    >>> d = np.array([2])
    >>> add_missing_times(t, d, dt=1)
    (array([1, 2, 3, 4, 5]), array([0, 0, 0, 0, 2]))

    >>> t = np.array([2, 4])
    >>> d = np.array([0, 3])
    >>> add_missing_times(t, d, dt=1)
    (array([3, 4]), array([0, 3]))
    """
    if t.ndim != 1:
        raise ValueError("t must be a 1-dimensional array")
    if d.ndim != 1:
        raise ValueError("d must be a 1-dimensional array")
    if dt <= 0:
        raise ValueError("dt must be a positive number")
    
    nonzero = np.flatnonzero(d)
    if nonzero.size == 0:
        return t, d

    t_expected = np.arange(t[0], t[-1] + dt, dt)
    missing = np.setdiff1d(t_expected, t, assume_unique=False)
    if missing.size == 0:
        return t, d

    insert_idx = np.maximum(0, np.searchsorted(t, missing))
    filled_values = np.where(insert_idx == 0, 0, d[insert_idx - 1])
    t = np.insert(t, insert_idx, missing)
    d = np.insert(d, insert_idx, filled_values)
    return t, d

def split_into_span(
    t: np.ndarray,
    d: np.ndarray,
    span: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate overlapping subarrays (windows) of a given `span` from two aligned input arrays.

    Parameters
    ----------
    t : np.ndarray
        1D array of time or index values.
    d : np.ndarray
        1D array of data values corresponding to `t`.
    span : int
        Number of elements in each window (must be <= len(t)).

    Yields
    ------
    tuple of np.ndarray
        A tuple (t_window, d_window), each of length `span`.

    Raises
    ------
    AssertionError
        If `t` and `d` have different shapes.
    ValueError
        If `span` is greater than the length of the input arrays.

    Examples
    --------
    >>> t = np.array([0, 1, 2, 3, 4])
    >>> d = np.array([10, 11, 12, 13, 14])
    >>> list(split_into_span(t, d, span=3))
    [(array([0, 1, 2]), array([10, 11, 12])),
     (array([1, 2, 3]), array([11, 12, 13])),
     (array([2, 3, 4]), array([12, 13, 14]))]
    """
    if t.shape != d.shape:
        raise AssertionError("Input arrays `t` and `d` must have the same shape.")
    if span > len(t):
        raise ValueError(f"Span ({span}) cannot be larger than input length ({len(t)}).")

    for i in range(len(t) - span + 1):
        idx = np.r_[i:i+span]
        yield t[idx], d[idx]

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two 1-D points.

    This function takes two 1-dimensional arrays of equal length and returns
    the Euclidean (L2) distance between them, defined as the square root of
    the sum of squared coordinate differences.

    Parameters
    ----------
    a : np.ndarray
        One-dimensional array of coordinates for the first point.
    b : np.ndarray
        One-dimensional array of coordinates for the second point.

    Returns
    -------
    float
        The Euclidean distance between `a` and `b`.

    Raises
    ------
    ValueError
        If either `a` or `b` is not one-dimensional.
    AssertionError
        If `a` and `b` do not have the same shape.

    Notes
    -----
    - Both inputs are converted to NumPy arrays of dtype ``float`` internally.
    - The function enforces that the inputs are 1-D; higher-dimensional arrays
      will raise an error.

    Examples
    --------
    >>> import numpy as np
    >>> euclidean_distance(np.array([0, 0]), np.array([3, 4]))
    5.0

    >>> euclidean_distance([1, 2, 3], [4, 5, 6])
    5.196152422706632

    >>> # Mismatched shapes will raise
    >>> euclidean_distance([1, 2], [1, 2, 3])
    Traceback (most recent call last):
        ...
    AssertionError: Input arrays `t` and `d` must have the same shape.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"Expected 1D input, got {a.ndim}D")
    if b.ndim != 1:
        raise ValueError(f"Expected 1D input, got {b.ndim}D")
    if a.shape != b.shape:
        raise AssertionError("Input arrays `a` and `b` must have the same shape.")
    return float(np.linalg.norm(a - b))

