"""
Hierarchical KD-tree for McDonald RG galaxy pairing.

Builds a binary KD-tree over the N galaxy positions.  Each internal node
represents a "sum mode" of its two children.  At each level, adjacent pairs
of leaves/nodes are grouped, halving the number of active nodes.

The tree determines the order in which difference modes are integrated out.
Nearby galaxies are paired first, which maximises sparsity in the inter-node
covariance matrix at each coarser level.

Inputs
------
positions : (N, 3) ndarray, Mpc/h
    Comoving positions of the N galaxies.

Outputs
-------
RGTree
    Hierarchical pairing structure consumed by coarsen.rg_coarsen_all.

Units
-----
Positions in Mpc/h.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree


@dataclass
class RGNode:
    """
    One node in the RG coarsening tree.

    Attributes
    ----------
    indices : list[int]
        Original galaxy indices contained in this node's "sum mode."
    level : int
        Depth from leaf (0 = leaf galaxy, k = merged from k-th coarsening).
    children : list[RGNode]
        The two child nodes (empty for leaf nodes).
    centroid : (3,) ndarray
        Mean position of member galaxies [Mpc/h].
    """
    indices: list[int]
    level: int
    children: list["RGNode"] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class RGTree:
    """
    Full RG coarsening tree.

    Attributes
    ----------
    root : RGNode
        Root of the binary tree (level = depth).
    levels : list[list[RGNode]]
        levels[k] = list of nodes at coarsening level k.
        levels[0] = leaf nodes (one per galaxy).
    depth : int
        Number of coarsening levels = ceil(log2(N)).
    N : int
        Number of leaf nodes (galaxies).
    positions : (N, 3) ndarray
        Original galaxy positions.
    """
    root: RGNode
    levels: list[list[RGNode]]
    depth: int
    N: int
    positions: np.ndarray


def build_tree(positions: np.ndarray) -> RGTree:
    """
    Build an RG pairing tree from galaxy positions using a KD-tree.

    At each level, nodes are paired greedily by nearest-neighbour
    matching in position space.  Each pair (i, j) gives one merged
    node at the next level; unpaired nodes (when count is odd) are
    passed through unchanged.

    Parameters
    ----------
    positions : (N, 3) ndarray, Mpc/h

    Returns
    -------
    RGTree
    """
    N = len(positions)
    if N < 2:
        raise ValueError("Need at least 2 galaxies to build an RG tree.")

    # Level 0: one leaf node per galaxy
    leaves = [
        RGNode(indices=[i], level=0, centroid=positions[i].copy())
        for i in range(N)
    ]

    levels = [leaves]
    current_nodes = leaves[:]
    depth = 0

    while len(current_nodes) > 1:
        depth += 1
        next_nodes, current_nodes = _pair_level(current_nodes, depth)
        levels.append(next_nodes)

    root = levels[-1][0]
    return RGTree(root=root, levels=levels, depth=depth, N=N, positions=positions)


def _pair_level(
    nodes: list[RGNode],
    level: int,
) -> tuple[list[RGNode], list[RGNode]]:
    """
    Pair nodes at one RG level using greedy nearest-neighbour matching.

    Parameters
    ----------
    nodes :
        Active nodes at the current level.
    level :
        Coarsening level for the new merged nodes.

    Returns
    -------
    next_nodes :
        Merged nodes for the next level.
    remaining :
        Same list as next_nodes (kept for caller convenience).
    """
    centroids = np.array([n.centroid for n in nodes])
    M = len(nodes)

    # Greedy NN pairing: find closest unmatched partner for each node
    kd = KDTree(centroids)
    paired = [False] * M
    pairs: list[tuple[int, int]] = []
    singletons: list[int] = []

    # Sort by index to ensure deterministic order
    for i in range(M):
        if paired[i]:
            continue
        # Find nearest unmatched neighbour
        _, inds = kd.query(centroids[i], k=min(M, M))
        for j in inds:
            if j != i and not paired[j]:
                pairs.append((i, j))
                paired[i] = True
                paired[j] = True
                break
        else:
            singletons.append(i)

    # Unpaired node if M is odd
    unpaired = [i for i in range(M) if not paired[i]]
    singletons = unpaired

    next_nodes: list[RGNode] = []

    for i, j in pairs:
        ni, nj = nodes[i], nodes[j]
        merged_indices = ni.indices + nj.indices
        merged_centroid = (
            ni.centroid * len(ni.indices) + nj.centroid * len(nj.indices)
        ) / len(merged_indices)
        merged = RGNode(
            indices=merged_indices,
            level=level,
            children=[ni, nj],
            centroid=merged_centroid,
        )
        next_nodes.append(merged)

    for i in singletons:
        # Pass through unpaired node to the next level unchanged
        node = nodes[i]
        passthrough = RGNode(
            indices=node.indices,
            level=level,
            children=[node],
            centroid=node.centroid.copy(),
        )
        next_nodes.append(passthrough)

    return next_nodes, next_nodes
