#!/usr/bin/env python3
# 
# paths.py
# 
# Created by Nicolas Fricker on 08/20/2025.
# 

import numpy as np
import networkx as nx

from typing import Callable, Optional, Union, Iterable

def _weight_function(
    G: nx.Graph,
    weight: str | Callable[[int, int], float | int | None] | None = None
) -> Callable[[int, int], float | int | None]:
    """
    Return a weight function usable with any NetworkX graph type.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Can be a ``Graph``, ``DiGraph``, ``MultiGraph`` or ``MultiDiGraph``.
    weight : str or callable or None, optional
        Edge attribute to use as weight. If a string, it is interpreted as the name of the edge
        attribute that holds a numeric weight value.  If a callable, it must have signature
        ``f(u, v)`` and return a weight (``float`` or ``int``).  If ``None`` (default), edge
        existence is treated as weight ``1``.

    Returns
    -------
    Callable[[int, int], float | int | None]
        A function ``f(u, v)`` that returns the weight for the edge ``(u, v)``.  For graphs with
        no such edge, the returned function returns ``None``.  For multigraphs, the weight of the
        first edge with the given attribute is returned.

    Notes
    -----
    This helper makes it easy to accept either a custom callable or the name of an edge attribute
    as a weight function parameter.  The returned function is compatible with all NetworkX graph
    types, including directed and multidigraphs.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, weight=3.0)
    >>> w = _weight_function(G, "weight")
    >>> w(0, 1)
    3.0

    >>> # custom weight
    >>> def my_weight(u, v):
    ...     return u + v
    >>> w = _weight_function(G, my_weight)
    >>> w(0, 1)
    1
    """
    if callable(weight):
        return weight

    def _weight(u: int, v: int) -> float | int | None:
        if not G.has_edge(u, v):
            return None

        data = G.get_edge_data(u, v)

        if weight is None:
            return 1

        if G.is_multigraph():
            for edge_data in data.values():
                if weight in edge_data:
                    return edge_data[weight]
            return None

        return data.get(weight, None)

    return _weight

def find_apical_paths(
    G: nx.Graph,
    weight: str | None = None,
    sort_neighbors: Callable[[Iterable[int]], Iterable[int]] | None = None
):
    """
    Find all apical paths in an undirected subcubic graph.

    An *apical path* is defined as a simple path that starts from a leaf node
    (degree == 1), follows edges toward neighbors with non-increasing node
    weights, and terminates when reaching a node of degree > 2.  
    Each path is returned as a sequence of nodes ordered from the branchpoint
    (high-degree node) down to the apex (leaf).

    Parameters
    ----------
    G : networkx.Graph
        Undirected input graph. Assumed to be subcubic (maximum degree ≤ 3).
    weight : str or None, optional
        Node attribute key to use for weight comparisons.
        If None, no weight constraints are applied (paths follow only structure).
    sort_neighbors : callable or None, optional
        Function to control the traversal order of neighbors.  
        Should accept an iterable of nodes and return an iterable (e.g., 
        ``sorted``). If None, neighbors are visited in the order from
        ``G.neighbors``.

    Yields
    ------
    path : numpy.ndarray of shape (k,)
        Array of node identifiers representing one apical path, ordered from
        the branchpoint (degree > 2 node) to the apex (degree == 1 node).

    Notes
    -----
    - A node is considered an *apex* if its degree is 1 (a leaf).
    - Paths are simple (no repeated nodes).
    - If `weight` is given, traversal only proceeds to neighbors with weight
      less than or equal to the current node's weight.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> G = nx.Graph()
    >>> G.add_nodes_from([
    ...     (1, {"w": 3}),
    ...     (2, {"w": 2}),
    ...     (3, {"w": 1}),
    ...     (4, {"w": 1}),
    ...     (5, {"w": 1}),
    ...     (6, {"w": 1}),
    ... ])
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)])
    >>> for path in find_apical_paths(G, weight="w"):
    ...     print(path)
    [4 3 2 1]
    [4 5]
    [4 6]
    """
    if not isinstance(G, nx.Graph):
        raise TypeError(f"G must be a networkx.Graph not {type(G)}")
    if G.is_directed():
        G = G.to_undirected()
    neighbors_func = (
        G.neighbors
        if sort_neighbors is None
        else lambda n: iter(sort_neighbors(G.neighbors(n)))
    )
    degrees = dict(G.degree())
    apexes = (n for n, d in degrees.items() if d == 1)
    weights: dict = nx.get_node_attributes(G, weight, default=1)

    for apex in apexes:
        stack = [(apex, [apex])]
        while stack:
            u, path = stack.pop()
            uw = weights.get(u, None)
            for v in neighbors_func(u):
                if v in path:
                    continue
                vw = weights.get(v, None)
                if not (uw is not None and vw is not None and vw <= uw):
                    continue
                new_path = path + [v]
                if degrees[v] > 2:
                    yield np.array(new_path[::-1])
                    continue
                stack.append((v, new_path))
