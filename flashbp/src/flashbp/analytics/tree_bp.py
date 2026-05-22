from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cycles import build_tanner_graph
from .style import ACTIVE_CHECK, BP_CORRECTION, INACTIVE_CHECK, ML_CORRECTION, ROOT_CHECK, TRUE_ERROR
from .treeify import BACK_EDGE, TREE_EDGE, treeify_layout


def _normalize(message: np.ndarray) -> np.ndarray:
    total = float(message.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.asarray([0.5, 0.5], dtype=np.float64)
    return message / total


def _check_message(
    incoming: list[np.ndarray],
    target_syndrome: int,
) -> np.ndarray:
    """Parity-check factor message to one variable."""
    out = np.zeros(2, dtype=np.float64)
    for x in (0, 1):
        parity_dist = np.asarray([1.0, 0.0], dtype=np.float64)
        for msg in incoming:
            next_dist = np.zeros(2, dtype=np.float64)
            next_dist[0] = parity_dist[0] * msg[0] + parity_dist[1] * msg[1]
            next_dist[1] = parity_dist[0] * msg[1] + parity_dist[1] * msg[0]
            parity_dist = next_dist
        out[x] = parity_dist[target_syndrome ^ x]
    return _normalize(out)


def tree_bp_marginals(
    decoder,
    syndrome,
    root_check: int,
    max_depth: int | None = None,
) -> dict:
    """
    Sever non-tree edges, run exact sum-product BP on the BFS tree rooted at
    `root_check`, and return per-data-node error probabilities.

    The output `marginals` array has length num_errors.  Nodes outside the
    visible tree are NaN.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    priors = np.asarray(decoder.error_probs, dtype=np.float64)
    syndrome_arr = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    root_node = num_vars + int(root_check)

    positions, tree_edges, depth = treeify_layout(
        H,
        root_node=root_node,
        max_depth=max_depth,
    )
    visible = set(positions)
    graph = build_tanner_graph(H)
    adj: dict[int, list[int]] = {node: [] for node in visible}
    for edge in tree_edges:
        u, v = tuple(edge)
        adj[u].append(v)
        adj[v].append(u)
    for nodes in adj.values():
        nodes.sort()

    messages: dict[tuple[int, int], np.ndarray] = {}
    for u, nodes in adj.items():
        for v in nodes:
            messages[(u, v)] = np.asarray([0.5, 0.5], dtype=np.float64)

    n_iter = max(1, 2 * max(depth.values(), default=0) + 2)
    for _ in range(n_iter):
        next_messages: dict[tuple[int, int], np.ndarray] = {}
        for u, nodes in adj.items():
            if u < num_vars:
                prior = np.asarray([1.0 - priors[u], priors[u]], dtype=np.float64)
                for v in nodes:
                    msg = prior.copy()
                    for w in nodes:
                        if w == v:
                            continue
                        msg *= messages[(w, u)]
                    next_messages[(u, v)] = _normalize(msg)
            else:
                d = u - num_vars
                for v in nodes:
                    incoming = [messages[(w, u)] for w in nodes if w != v]
                    next_messages[(u, v)] = _check_message(
                        incoming,
                        int(syndrome_arr[d]),
                    )
        messages = next_messages

    marginals = np.full(num_vars, np.nan, dtype=np.float64)
    for v in range(num_vars):
        if v not in visible:
            continue
        prior = np.asarray([1.0 - priors[v], priors[v]], dtype=np.float64)
        belief = prior.copy()
        for u in adj.get(v, []):
            belief *= messages[(u, v)]
        belief = _normalize(belief)
        marginals[v] = belief[1]

    decision = np.zeros(num_vars, dtype=np.uint8)
    finite = np.isfinite(marginals)
    decision[finite] = (marginals[finite] > 0.5).astype(np.uint8)

    return {
        "root_check": int(root_check),
        "marginals": marginals,
        "decision": decision,
        "visible": np.asarray(sorted(visible), dtype=np.int64),
        "depth": depth,
        "positions": positions,
        "tree_edges": tree_edges,
    }


def plot_tree_bp_marginals(
    decoder,
    syndrome,
    root_check: int,
    output_path: str | Path = "tree_bp.png",
    max_depth: int | None = None,
    bp_correction=None,
    ml_correction=None,
    true_errors=None,
    show_labels: bool = True,
    figsize: tuple[float, float] | None = None,
) -> dict:
    """
    Render the severed-tree BP marginal probabilities from one root detection.

    Data nodes are colored by Pr(error=1).  Optional correction/error overlays:
        orange ring = simple BP correction bit
        blue ring   = ML correction bit
        red dot     = true sampled error bit
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    syndrome_arr = np.asarray(syndrome, dtype=np.uint8)
    result = tree_bp_marginals(
        decoder,
        syndrome_arr,
        root_check=root_check,
        max_depth=max_depth,
    )
    positions = result["positions"]
    tree_edges = result["tree_edges"]
    marginals = result["marginals"]
    visible = set(positions)
    graph = build_tanner_graph(H)

    if figsize is None:
        levels = max(result["depth"].values(), default=0) + 1
        width = max(8.0, min(28.0, 0.35 * max(1, len(visible))))
        height = max(6.0, min(32.0, 0.75 * levels + 2.0))
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for u, v in graph.edges():
        if u not in visible or v not in visible:
            continue
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        if frozenset((u, v)) in tree_edges:
            ax.plot([x1, x2], [y1, y2],
                    color=TREE_EDGE, linewidth=1.6, alpha=0.9, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=BACK_EDGE, linewidth=0.8, alpha=0.55,
                    linestyle="--", zorder=0)

    data_nodes = sorted(n for n in visible if n < num_vars)
    check_nodes = sorted(n for n in visible if n >= num_vars)
    base_size = max(45.0, min(180.0, 4800.0 / max(1, len(visible))))

    cmap = plt.get_cmap("viridis")
    if data_nodes:
        colors = [cmap(float(np.nan_to_num(marginals[v], nan=0.0)))
                  for v in data_nodes]
        ax.scatter(
            [positions[n][0] for n in data_nodes],
            [positions[n][1] for n in data_nodes],
            s=base_size,
            c=colors,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    def overlay_bits(bits, color: str, size_scale: float, linewidth: float):
        if bits is None:
            return
        arr = np.asarray(bits, dtype=np.uint8)
        nodes = [v for v in data_nodes if v < arr.size and arr[v]]
        if not nodes:
            return
        ax.scatter(
            [positions[n][0] for n in nodes],
            [positions[n][1] for n in nodes],
            s=base_size * size_scale,
            facecolors="none",
            edgecolors=color,
            linewidths=linewidth,
            zorder=5,
        )

    overlay_bits(bp_correction, BP_CORRECTION, 1.55, 1.7)
    overlay_bits(ml_correction, ML_CORRECTION, 2.05, 1.7)
    if true_errors is not None:
        arr = np.asarray(true_errors, dtype=np.uint8)
        nodes = [v for v in data_nodes if v < arr.size and arr[v]]
        if nodes:
            ax.scatter(
                [positions[n][0] for n in nodes],
                [positions[n][1] for n in nodes],
                s=base_size * 0.18,
                c=TRUE_ERROR,
                marker="o",
                zorder=6,
            )

    if check_nodes:
        faces = []
        widths = []
        for node in check_nodes:
            d = node - num_vars
            if d == root_check:
                faces.append(ROOT_CHECK)
                widths.append(2.0)
            elif syndrome_arr[d]:
                faces.append(ACTIVE_CHECK)
                widths.append(1.4)
            else:
                faces.append(INACTIVE_CHECK)
                widths.append(0.8)
        ax.scatter(
            [positions[n][0] for n in check_nodes],
            [positions[n][1] for n in check_nodes],
            s=base_size,
            c=faces,
            edgecolors="black",
            linewidths=widths,
            marker="s",
            zorder=3,
        )

    if show_labels:
        label_fontsize = max(4.0, min(8.0, 80.0 / np.sqrt(max(1, len(visible)))))
        for node in sorted(visible):
            if node < num_vars:
                prob = marginals[node]
                label = f"{node}\n{prob:.2f}" if np.isfinite(prob) else str(node)
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

    finite = marginals[np.isfinite(marginals)]
    range_text = "none" if finite.size == 0 else f"{finite.min():.3g}..{finite.max():.3g}"
    ax.set_title(
        f"Tree BP from detection {root_check}    "
        f"Pr(error=1)={range_text}",
        fontsize=12,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return result
