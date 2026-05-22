from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flashbp.animation.layout import bipartite_layout, edges_from_H
from .style import ACTIVE_CHECK, FAINT_EDGE, TRUE_ERROR, TRUE_ERROR_LIGHT


def plot_syndrome_graph(
    decoder,
    syndrome,
    output_path: str | Path = "syndrome.png",
    error_vector=None,
    layout: dict | None = None,
    show_labels: bool = True,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Render a Tanner graph with active detections filled and true-error edges red.

    Active parity checks are filled black.  If `error_vector` is provided, every
    Tanner edge incident to an active error mechanism is drawn red.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome_arr = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if syndrome_arr.shape[0] != num_checks:
        raise ValueError(
            f"syndrome has length {syndrome_arr.shape[0]}, expected {num_checks}"
        )

    errors = (
        np.asarray(error_vector, dtype=np.uint8)
        if error_vector is not None
        else np.zeros(num_vars, dtype=np.uint8)
    )
    if errors.shape[0] != num_vars:
        raise ValueError(f"error_vector has length {errors.shape[0]}, expected {num_vars}")

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

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        is_error_edge = bool(errors[v])
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=TRUE_ERROR if is_error_edge else FAINT_EDGE,
            linewidth=2.0 if is_error_edge else 0.5,
            alpha=0.9 if is_error_edge else 0.6,
            zorder=2 if is_error_edge else 1,
        )

    var_faces = [TRUE_ERROR_LIGHT if errors[v] else "white" for v in range(num_vars)]
    var_edges = [TRUE_ERROR if errors[v] else "black" for v in range(num_vars)]
    var_lws = [1.6 if errors[v] else 0.8 for v in range(num_vars)]
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=3,
    )

    check_faces = [ACTIVE_CHECK if syndrome_arr[d] else "white" for d in range(num_checks)]
    check_text = ["white" if syndrome_arr[d] else "black" for d in range(num_checks)]
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors="black",
        linewidths=1.0,
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
                color=check_text[d],
                zorder=4,
            )

    ax.set_title(
        f"syndrome graph    detections={int(syndrome_arr.sum())}  "
        f"errors={int(errors.sum())}",
        fontsize=12,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
