from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from flashbp.animation.layout import bipartite_layout, edges_from_H
from .style import ACTIVE_CHECK, CYCLE, FAINT_EDGE, FAINT_NODE, INACTIVE_CHECK


CYCLE_COLOR = CYCLE


def build_tanner_graph(H: np.ndarray) -> nx.Graph:
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vars + num_checks))
    rows, cols = np.nonzero(H)
    for d, v in zip(rows, cols):
        graph.add_edge(int(v), num_vars + int(d))
    return graph


def find_cycles(
    H: np.ndarray,
    max_length: int,
    syndrome=None,
    require_active_check: bool = False,
) -> list[list[int]]:
    """
    Enumerate simple Tanner-graph cycles up to `max_length`.

    Node ids are tagged:
        0 .. num_vars - 1 are data nodes
        num_vars .. num_vars + num_checks - 1 are parity-check nodes

    If `require_active_check` is true, only cycles touching at least one
    syndrome-active parity-check node are returned.
    """
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if max_length < 4:
        return []

    syndrome_arr = None
    if syndrome is not None:
        syndrome_arr = np.asarray(syndrome, dtype=np.uint8)

    graph = build_tanner_graph(H)
    cycles: list[list[int]] = []
    for cycle in nx.simple_cycles(graph, length_bound=max_length):
        if len(cycle) < 4:
            continue
        if require_active_check:
            if syndrome_arr is None:
                raise ValueError("syndrome is required when require_active_check=True")
            if not cycle_has_active_check(cycle, num_vars, syndrome_arr):
                continue
        cycles.append(canonical_cycle(cycle))
    cycles = sorted(set(tuple(c) for c in cycles), key=lambda c: (len(c), c))
    return [list(c) for c in cycles]


def canonical_cycle(cycle: list[int]) -> list[int]:
    """Return a rotation/orientation-stable representation of a cycle."""
    if not cycle:
        return []
    seq = list(cycle)
    variants = []
    for oriented in (seq, list(reversed(seq))):
        for i in range(len(oriented)):
            variants.append(tuple(oriented[i:] + oriented[:i]))
    return list(min(variants))


def cycle_has_active_check(
    cycle: list[int],
    num_vars: int,
    syndrome: np.ndarray,
) -> bool:
    for node in cycle:
        if node >= num_vars and syndrome[node - num_vars]:
            return True
    return False


def cycle_edge_counter(cycles: list[list[int]]) -> Counter[frozenset[int]]:
    counts: Counter[frozenset[int]] = Counter()
    for cycle in cycles:
        for i in range(len(cycle)):
            counts[frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))] += 1
    return counts


def cycle_node_sets(cycles: list[list[int]], num_vars: int) -> tuple[set[int], set[int]]:
    data_nodes: set[int] = set()
    check_nodes: set[int] = set()
    for cycle in cycles:
        for node in cycle:
            if node < num_vars:
                data_nodes.add(node)
            else:
                check_nodes.add(node - num_vars)
    return data_nodes, check_nodes


def render_cycle_frame(
    cycle: list[int],
    cycle_idx: int,
    total_cycles: int,
    H: np.ndarray,
    layout: dict,
    output_path: str | Path,
    figsize: tuple[float, float] | None = None,
    syndrome=None,
) -> None:
    """Render one cycle highlighted over a faded Tanner graph."""
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape

    if figsize is None:
        figsize = layout.get("figsize") or (
            8.0, min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0)
        )
    base_size = layout.get(
        "node_size",
        max(60.0, 4000.0 / max(num_vars, num_checks)),
    )

    syndrome_arr = (
        np.asarray(syndrome, dtype=np.uint8)
        if syndrome is not None
        else np.zeros(num_checks, dtype=np.uint8)
    )
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)

    cycle_edges = {
        frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
        for i in range(len(cycle))
    }
    cycle_vars = {n for n in cycle if n < num_vars}
    cycle_checks = {n - num_vars for n in cycle if n >= num_vars}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        if frozenset((v, num_vars + d)) in cycle_edges:
            ax.plot([x1, x2], [y1, y2],
                    color=CYCLE_COLOR, linewidth=3.0, alpha=1.0, zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=FAINT_EDGE, linewidth=0.4, alpha=0.5, zorder=1)

    var_faces = [CYCLE_COLOR if v in cycle_vars else "white"
                 for v in range(num_vars)]
    var_edges = ["black" if v in cycle_vars else FAINT_NODE
                 for v in range(num_vars)]
    var_lws = [2.0 if v in cycle_vars else 0.4 for v in range(num_vars)]
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=3,
    )

    check_faces = []
    for d in range(num_checks):
        if syndrome_arr[d]:
            check_faces.append(ACTIVE_CHECK)
        elif d in cycle_checks:
            check_faces.append(CYCLE_COLOR)
        else:
            check_faces.append(INACTIVE_CHECK)
    check_edges = ["black" if d in cycle_checks or syndrome_arr[d] else FAINT_NODE
                   for d in range(num_checks)]
    check_lws = [2.0 if d in cycle_checks or syndrome_arr[d] else 0.4
                 for d in range(num_checks)]
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=check_lws,
        marker="s",
        zorder=3,
    )

    ax.set_title(
        f"active-check cycle {cycle_idx + 1}/{total_cycles}    length={len(cycle)}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_active_check_cycles(
    decoder,
    syndrome,
    output_path: str | Path = "active_cycles.png",
    max_length: int = 8,
    layout: dict | None = None,
    figsize: tuple[float, float] | None = None,
    show_labels: bool = True,
) -> list[list[int]]:
    """
    Render all cycles up to `max_length` that touch at least one active parity
    check.  Edges become darker/thicker when they appear in more such cycles.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    syndrome_arr = np.asarray(syndrome, dtype=np.uint8)
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    if figsize is None:
        figsize = layout.get("figsize") or (
            8.0, min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0)
        )
    base_size = layout.get(
        "node_size",
        max(60.0, 4000.0 / max(num_vars, num_checks)),
    )

    cycles = find_cycles(
        H,
        max_length=max_length,
        syndrome=syndrome_arr,
        require_active_check=True,
    )
    edge_counts = cycle_edge_counter(cycles)
    cycle_vars, cycle_checks = cycle_node_sets(cycles, num_vars)
    max_count = max(edge_counts.values(), default=1)

    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        count = edge_counts.get(frozenset((v, num_vars + d)), 0)
        if count:
            strength = count / max_count
            ax.plot(
                [x1, x2], [y1, y2],
                color=CYCLE_COLOR,
                linewidth=1.0 + 3.0 * strength,
                alpha=0.30 + 0.65 * strength,
                zorder=2,
            )
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=FAINT_EDGE, linewidth=0.4, alpha=0.45, zorder=1)

    var_faces = [CYCLE_COLOR if v in cycle_vars else "white"
                 for v in range(num_vars)]
    var_edges = ["black" if v in cycle_vars else FAINT_NODE
                 for v in range(num_vars)]
    var_lws = [1.6 if v in cycle_vars else 0.4 for v in range(num_vars)]
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=3,
    )

    check_faces = []
    for d in range(num_checks):
        if syndrome_arr[d]:
            check_faces.append(ACTIVE_CHECK)
        elif d in cycle_checks:
            check_faces.append(CYCLE_COLOR)
        else:
            check_faces.append(INACTIVE_CHECK)
    check_edges = ["black" if syndrome_arr[d] or d in cycle_checks else FAINT_NODE
                   for d in range(num_checks)]
    check_lws = [1.8 if syndrome_arr[d] or d in cycle_checks else 0.4
                 for d in range(num_checks)]
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=check_lws,
        marker="s",
        zorder=3,
    )

    if show_labels:
        label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
        for v in range(num_vars):
            ax.annotate(
                str(v),
                xy=var_pos[v],
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=4,
            )
        for d in range(num_checks):
            ax.annotate(
                str(d),
                xy=check_pos[d],
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=4,
            )

    active = int(syndrome_arr.sum())
    lengths = [len(c) for c in cycles]
    length_text = "none" if not lengths else f"{min(lengths)}..{max(lengths)}"
    ax.set_title(
        f"cycles touching active parity checks    active={active}  "
        f"cycles={len(cycles)}  lengths={length_text}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return cycles
