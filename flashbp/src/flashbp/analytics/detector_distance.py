from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flashbp.animation.layout import bipartite_layout, edges_from_H
from .style import ACTIVE_CHECK, FAINT_EDGE, ML_CORRECTION


def data_detector_distances(
    H,
    detectors: list[int] | np.ndarray | None = None,
    syndrome: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Tanner-graph distance from selected detector nodes to each data/error node.

    Distances are graph-edge hops.  A data node adjacent to a selected detector
    has distance 1; reaching a data node through another detector has distance 3,
    and so on.  Unreachable data nodes are returned as -1.
    """
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape

    if detectors is None:
        if syndrome is not None:
            detectors = np.flatnonzero(np.asarray(syndrome, dtype=np.uint8))
        else:
            detectors = [0]
    sources = sorted({int(d) for d in detectors if 0 <= int(d) < num_checks})
    if not sources:
        raise ValueError("No valid source detectors were selected.")

    total = num_vars + num_checks
    adj: list[list[int]] = [[] for _ in range(total)]
    rows, cols = np.nonzero(H)
    for d, v in zip(rows, cols):
        d_node = num_vars + int(d)
        v_node = int(v)
        adj[d_node].append(v_node)
        adj[v_node].append(d_node)

    dist = np.full(total, -1, dtype=np.int32)
    q: deque[int] = deque()
    for d in sources:
        node = num_vars + d
        dist[node] = 0
        q.append(node)

    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[node] + 1
            q.append(nxt)

    return dist[:num_vars]


def plot_detector_distance_graph(
    decoder,
    output_path: str | Path = "detector_distance.png",
    detectors: list[int] | np.ndarray | None = None,
    syndrome: list[int] | np.ndarray | None = None,
    layout: dict | None = None,
    figsize: tuple[float, float] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Render data-node distance from selected detector nodes on the Tanner graph.

    Data nodes are light when close to a selected detector and darker when
    farther away.  Selected detector nodes are blue squares.  Other detector
    nodes with syndrome 1 are black; inactive detectors are white.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if syndrome is not None:
        syndrome_arr = np.asarray(syndrome, dtype=np.uint8)
    else:
        syndrome_arr = np.zeros(num_checks, dtype=np.uint8)

    if detectors is None and syndrome is not None:
        detectors = np.flatnonzero(syndrome_arr)
    if detectors is None:
        detectors = [0]
    source_detectors = sorted(
        {int(d) for d in detectors if 0 <= int(d) < num_checks}
    )
    distances = data_detector_distances(
        H, detectors=source_detectors, syndrome=syndrome_arr
    )

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

    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)

    finite = distances[distances >= 0]
    if finite.size:
        min_d = int(finite.min())
        max_d = int(finite.max())
    else:
        min_d = max_d = 0

    def var_color(distance: int) -> str:
        if distance < 0:
            return "#111111"
        if max_d == min_d:
            level = 0.30
        else:
            level = 0.18 + 0.72 * ((distance - min_d) / (max_d - min_d))
        grey = int(round(255 * (1.0 - level)))
        return f"#{grey:02x}{grey:02x}{grey:02x}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=FAINT_EDGE,
            linewidth=0.5,
            alpha=0.7,
            zorder=1,
        )

    var_faces = [var_color(int(distances[v])) for v in range(num_vars)]
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    source_set = set(source_detectors)
    check_faces = []
    check_edges = []
    for d in range(num_checks):
        if d in source_set:
            check_faces.append(ML_CORRECTION)
            check_edges.append(ML_CORRECTION)
        elif syndrome_arr[d]:
            check_faces.append(ACTIVE_CHECK)
            check_edges.append("black")
        else:
            check_faces.append("white")
            check_edges.append("black")
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=1.0,
        marker="s",
        zorder=3,
    )

    if show_labels:
        label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
        for v in range(num_vars):
            x, y = var_pos[v]
            label = f"{v}\n{int(distances[v]) if distances[v] >= 0 else 'inf'}"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=4,
            )

    src_text = ",".join(str(d) for d in source_detectors)
    ax.set_title(
        f"data-node Tanner distance from detector(s) {src_text}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return distances
