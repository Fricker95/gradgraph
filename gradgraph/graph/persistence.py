#!/usr/bin/env python3
# 
# persistence.py
# 
# Created by Nicolas Fricker on 08/28/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import gudhi as gd
import numpy as np
import networkx as nx

from scipy.spatial import KDTree
from typing import Iterator

from gradgraph.graph.utils import (
    remove_degree_k_nodes,
    spectral_gap
)

def compute_gudhi_persistence(
    G: nx.Graph,
    weight: str,
) -> Iterator[tuple[int, tuple[float, float]]]:
    """
    Compute persistent homology from a graph using GUDHI.

    This function builds a GUDHI simplex tree from the given graph and computes 
    persistence pairs based on the provided node attribute weights. Each edge 
    is inserted into the simplex tree with a filtration value equal to the 
    maximum of its endpoint weights. Node IDs must be non-negative, as they 
    are shifted by +1 before insertion to match GUDHI’s indexing requirements.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. All nodes must have the attribute specified by `weight`.
        Node IDs must be non-negative integers.
    weight : str
        Name of the node attribute in `G` to be used as the filtration weight.

    Yields
    ------
    tuple of (int, tuple of (float, float))
        Persistence intervals as returned by 
        :meth:`gudhi.simplex_tree.SimplexTree.persistence`.
        Each element is a tuple ``(dimension, (birth, death))``.

    Raises
    ------
    ValueError
        If the graph contains any negative node IDs.
    KeyError
        If any node does not have the specified `weight` attribute.

    Notes
    -----
    - This function only inserts edges into the simplex tree. 
      Vertices are implicitly added during edge insertion.
    - The filtration value of an edge is ``max(weight[u], weight[v])``.
    - Persistence is computed across all dimensions 
      (controlled by ``persistence_dim_max=True``).

    Examples
    --------
    >>> import networkx as nx
    >>> import gudhi as gd
    >>> G = nx.Graph()
    >>> G.add_node(0, w=0.1)
    >>> G.add_node(1, w=0.3)
    >>> G.add_edge(0, 1)
    >>> list(compute_gudhi_persistence(G, weight="w"))
    [(0, (0.3, float('inf')))]
    """
    if any(np.array(G.nodes) < 0):
        raise ValueError('G contains negative NodeIDs. GUDHI does not support negative NodeIDs.')
    st = gd.simplex_tree.SimplexTree()
    weights = nx.get_node_attributes(G, weight)
    edges = np.array(G.edges)
    for edge in edges:
        u, v = edge
        st.insert(
            edge,
            filtration=max(weights[u], weights[v])
        )
    yield from st.persistence(persistence_dim_max=True)

def remove_degree_k_nodes_over_time(
    G: nx.Graph,
    degree: int,
    node_attr: str,
    edge_attr: str,
) -> Iterator[nx.Graph]:
    """
    Generate temporal subgraphs with degree-``k`` nodes removed at each time step.

    The function iterates over increasing values of a specified node attribute,
    forming subgraphs induced by nodes with attribute values less than or equal
    to the current threshold. For each such subgraph, nodes of degree exactly
    ``degree`` are removed (via :func:`remove_degree_k_nodes`), and the resulting
    graph is yielded.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Must have the node attribute `node_attr` defined for all
        nodes.
    degree : int
        The degree of nodes to remove from each temporal subgraph.
    node_attr : str
        Node attribute used to define temporal thresholds. Each unique value of
        this attribute induces a snapshot in time.
    edge_attr : str
        Edge attribute name to be passed through to
        :func:`remove_degree_k_nodes`.

    Yields
    ------
    nx.Graph
        A subgraph of ``G`` at a temporal threshold, with all degree-``degree``
        nodes removed.

    Raises
    ------
    KeyError
        If any node is missing the specified `node_attr`.

    Notes
    -----
    - The temporal dimension is simulated by progressively including nodes with
      larger values of `node_attr`.
    - The helper function :func:`remove_degree_k_nodes` is expected to handle
      node removal based on degree, using `edge_attr` as needed.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_node(0, time=1)
    >>> G.add_node(1, time=2)
    >>> G.add_node(2, time=2)
    >>> G.add_edges_from([(0, 1), (1, 2)], weight=1.0)
    >>> for H in remove_degree_k_nodes_over_time(G, degree=2,
    ...                                          node_attr="time",
    ...                                          edge_attr="weight"):
    ...     print(H.nodes(), H.edges())
    [0] []
    [0, 2] [(0, 2)]
    """
    weights = nx.get_node_attributes(G, node_attr)
    for w in sorted(set(weights.values())):
        g = G.subgraph((k for k, v in weights.items() if v <= w))
        yield remove_degree_k_nodes(g, degree=degree, weight=edge_attr)

def find_local_curvatures(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    radius: float
) -> Iterator[tuple[int, float]]:
    """
    Estimate local curvatures of nodes in a graph based on neighborhood connectivity.

    This function identifies nodes with degree at least 3 and, for each such node,
    counts how many of the other high-degree nodes within a given spatial radius
    are directly connected to it in the graph.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph representing the network.
    pos : dict of int to tuple of float
        A mapping from node identifiers to their 2D positions, e.g. 
        ``{node: (x, y)}``.
    radius : float
        Euclidean radius within which to search for neighboring nodes.

    Yields
    ------
    node_id : int
        The node identifier of a high-degree node (degree ≥ 3).
    curvature : float
        The number of edges from the node to other high-degree nodes 
        within the given radius.

    Notes
    -----
    - Only nodes with degree ≥ 3 are considered.
    - If no nodes with degree ≥ 3 exist, the function yields nothing.
    - The spatial queries are performed using a KD-tree for efficiency.
    - If no other qualifying nodes are found within the radius, the node 
      is skipped.
    """
    nodes = [n for n, d in G.degree() if d >= 3]
    positions = np.array([pos[n] for n in nodes])
    if positions.size == 0:
        return iter([])
    kdtree = KDTree(positions)
    for n, p in zip(nodes, positions):
        idxs = kdtree.query_ball_point(p, r=radius)
        nodes_in_radius = np.array([nodes[i] for i in idxs if (nodes[i] != n)])
        if nodes_in_radius.size == 0:
            continue
        yield n, np.sum([G.has_edge(n, nn) for nn in nodes_in_radius])

def find_spectral_gaps(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    radius: float
) -> Iterator[tuple[int, float]]:
    """
    Compute local spectral gaps for nodes based on neighborhood subgraphs.

    For each node, this function finds all other nodes within a given
    Euclidean radius (using node positions) and extracts the connected
    component containing the node. The spectral gap of that induced
    subgraph is then computed.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph representing the network.
    pos : dict of int to tuple of float
        A mapping from node identifiers to their 2D positions, e.g.
        ``{node: (x, y)}``.
    radius : float
        Euclidean radius within which to search for neighboring nodes.

    Yields
    ------
    node_id : int
        The identifier of the reference node.
    spectral_gap : float
        The spectral gap of the connected subgraph containing the node,
        defined as the difference between the two largest eigenvalues
        of the adjacency or Laplacian matrix (depending on the
        implementation of ``spectral_gap``).

    Notes
    -----
    - The function uses a KD-tree for efficient spatial queries.
    - The neighborhood is defined as all nodes within the given radius
      in Euclidean space, according to the `pos` mapping.
    - Only the connected component containing the node is considered
      when computing the spectral gap.
    - The definition of the spectral gap depends on the implementation
      of the external ``spectral_gap`` function.
    """
    nodes = np.array(G.nodes())
    positions = np.array([pos[n] for n in nodes])
    if positions.size == 0:
        return iter([])
    kdtree = KDTree(positions)
    for n, p in zip(nodes, positions):
        idxs = kdtree.query_ball_point(p, r=radius)
        nodes_in_radius = nodes[idxs]
        if nodes_in_radius.size == 0:
            continue
        g = G.subgraph(nodes_in_radius)
        connected_component = []
        for component in nx.connected_components(g):
            if n not in component:
                continue
            connected_component = component
            break
        if not connected_component:
            continue
        g = G.subgraph(connected_component)
        yield n, spectral_gap(g)



