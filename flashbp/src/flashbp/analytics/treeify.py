from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from flashbp.animation.layout import edges_from_H
from .cycles import build_tanner_graph
from .style import ACTIVE_CHECK, INACTIVE_CHECK, NON_TREE_EDGE, ROOT_CHECK, TREE_EDGE


BACK_EDGE = NON_TREE_EDGE
ROOT_COLOR = ROOT_CHECK
DATA_COLOR = "white"


def treeify_layout(
    H: np.ndarray,
    root_node: int,
    max_depth: int | None = None,
) -> tuple[dict[int, tuple[float, float]], set[frozenset[int]], dict[int, int]]:
    """
    Lay out the Tanner graph as a BFS tree hanging from `root_node`.

    Node ids are tagged:
        0 .. num_vars - 1 are data nodes
        num_vars .. num_vars + num_checks - 1 are parity-check nodes

    Returns positions, tree edges, and node depths.  Non-tree edges can be
    rendered separately to show where cycles fold back into the tree.
    """
    H = np.asarray(H, dtype=np.uint8)
    graph = build_tanner_graph(H)
    if root_node not in graph:
        raise ValueError(f"root_node {root_node} is not in the Tanner graph")

    parent: dict[int, int | None] = {root_node: None}
    depth: dict[int, int] = {root_node: 0}
    children: dict[int, list[int]] = defaultdict(list)
    queue: deque[int] = deque([root_node])

    while queue:
        node = queue.popleft()
        next_depth = depth[node] + 1
        if max_depth is not None and next_depth > max_depth:
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor in depth:
                continue
            parent[neighbor] = node
            depth[neighbor] = next_depth
            children[node].append(neighbor)
            queue.append(neighbor)

    levels: dict[int, list[int]] = defaultdict(list)
    for node, d in depth.items():
        levels[d].append(node)
    for nodes in levels.values():
        nodes.sort()

    max_seen_depth = max(levels, default=0)
    positions: dict[int, tuple[float, float]] = {}
    for d in range(max_seen_depth + 1):
        nodes = levels[d]
        if not nodes:
            continue
        y = 1.0 - (d + 0.5) / (max_seen_depth + 1)
        for i, node in enumerate(nodes):
            x = (i + 1) / (len(nodes) + 1)
            positions[node] = (x, y)

    tree_edges: set[frozenset[int]] = set()
    for node, par in parent.items():
        if par is not None:
            tree_edges.add(frozenset((node, par)))

    return positions, tree_edges, depth


def plot_treeified_tanner_graph(
    decoder,
    root_check: int,
    output_path: str | Path = "tree.png",
    syndrome=None,
    max_depth: int | None = None,
    figsize: tuple[float, float] | None = None,
    show_labels: bool = True,
) -> dict[int, int]:
    """
    Render the Tanner graph as a tree rooted at parity-check `root_check`.

    The root check is placed at the top.  BFS tree edges are blue; non-tree
    edges among visible nodes are faint dashed lines, marking cycle closures.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if root_check < 0 or root_check >= num_checks:
        raise ValueError(f"root_check {root_check} outside 0..{num_checks - 1}")

    root_node = num_vars + root_check
    positions, tree_edges, depth = treeify_layout(
        H,
        root_node=root_node,
        max_depth=max_depth,
    )

    if figsize is None:
        levels = max(depth.values(), default=0) + 1
        width = max(8.0, min(26.0, 0.35 * max(1, len(positions))))
        height = max(6.0, min(30.0, 0.75 * levels + 2.0))
        figsize = (width, height)

    syndrome_arr = (
        np.asarray(syndrome, dtype=np.uint8)
        if syndrome is not None
        else np.zeros(num_checks, dtype=np.uint8)
    )

    visible = set(positions)
    graph = build_tanner_graph(H)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for u, v in graph.edges():
        if u not in visible or v not in visible:
            continue
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        edge = frozenset((u, v))
        if edge in tree_edges:
            ax.plot([x1, x2], [y1, y2],
                    color=TREE_EDGE, linewidth=1.6, alpha=0.9, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=BACK_EDGE, linewidth=0.8, alpha=0.7,
                    linestyle="--", zorder=0)

    data_nodes = sorted(n for n in visible if n < num_vars)
    check_nodes = sorted(n for n in visible if n >= num_vars)
    base_size = max(45.0, min(180.0, 4800.0 / max(1, len(visible))))

    if data_nodes:
        ax.scatter(
            [positions[n][0] for n in data_nodes],
            [positions[n][1] for n in data_nodes],
            s=base_size,
            c=DATA_COLOR,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    if check_nodes:
        faces = []
        edges = []
        widths = []
        for node in check_nodes:
            d = node - num_vars
            if node == root_node:
                faces.append(ROOT_COLOR)
                edges.append("black")
                widths.append(2.0)
            elif syndrome_arr[d]:
                faces.append(ACTIVE_CHECK)
                edges.append("black")
                widths.append(1.4)
            else:
                faces.append(INACTIVE_CHECK)
                edges.append("black")
                widths.append(0.8)
        ax.scatter(
            [positions[n][0] for n in check_nodes],
            [positions[n][1] for n in check_nodes],
            s=base_size,
            c=faces,
            edgecolors=edges,
            linewidths=widths,
            marker="s",
            zorder=3,
        )

    if show_labels:
        label_fontsize = max(4.0, min(8.0, 80.0 / np.sqrt(max(1, len(visible)))))
        for node in sorted(visible):
            if node < num_vars:
                label = str(node)
            else:
                label = f"d{node - num_vars}"
            ax.annotate(
                label,
                xy=positions[node],
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=4,
            )

    non_tree_edges = 0
    for u, v in graph.edges():
        if u in visible and v in visible and frozenset((u, v)) not in tree_edges:
            non_tree_edges += 1
    ax.set_title(
        f"Tanner graph tree from detection {root_check}    "
        f"visible={len(visible)}  cycle-closure edges={non_tree_edges}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return depth


def visible_edges_from_H(H: np.ndarray, visible: set[int]) -> list[tuple[int, int]]:
    """Return visible Tanner edges as tagged node-id pairs."""
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    out = []
    for d, v in edges_from_H(H):
        u = num_vars + d
        if u in visible and v in visible:
            out.append((v, u))
    return out
