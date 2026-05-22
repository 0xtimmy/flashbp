from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flashbp.animation.layout import bipartite_layout, edges_from_H


def ml_branch_surprises(
    recording,
    shot_index: int = 0,
    num_errors: int | None = None,
    metric: str = "js_divergence",
    reducer: str = "max",
) -> np.ndarray:
    """
    Extract per-data-node contraction surprise from an MLLogger recording.

    For SurpriseML recordings, `js_divergence` is the distribution change from
    the current tensor to the candidate post-contraction tensor.  `kl_0_to_1`
    and `kl_1_to_0` are kept for logger compatibility and correspond to
    current-to-next and next-to-current KL on those recordings.

    If an error axis appears more than once, `reducer` controls aggregation and
    may be `max`, `sum`, or `mean`.
    """
    if not recording:
        raise ValueError("recording is empty")
    shot = recording[shot_index]
    steps = shot.get("steps", [])
    if num_errors is None:
        observed = [int(s.get("error_idx", -1)) for s in steps]
        observed = [e for e in observed if e >= 0]
        num_errors = max(observed) + 1 if observed else 0

    values: list[list[float]] = [[] for _ in range(num_errors)]
    for step in steps:
        e = int(step.get("error_idx", -1))
        if e < 0 or e >= num_errors:
            continue
        if metric == "sym_kl":
            a = float(step.get("kl_0_to_1", np.nan))
            b = float(step.get("kl_1_to_0", np.nan))
            value = 0.5 * (a + b)
        else:
            value = float(step.get(metric, np.nan))
        values[e].append(value)

    out = np.full(num_errors, np.nan, dtype=np.float64)
    for e, vals in enumerate(values):
        finite_vals = np.asarray(vals, dtype=np.float64)
        if finite_vals.size == 0:
            continue
        if reducer == "sum":
            out[e] = np.nansum(finite_vals)
        elif reducer == "mean":
            out[e] = np.nanmean(finite_vals)
        elif reducer == "max":
            out[e] = np.nanmax(finite_vals)
        else:
            raise ValueError("reducer must be 'max', 'sum', or 'mean'")
    return out


def plot_ml_surprise_graph(
    decoder,
    recording,
    output_path: str | Path = "ml_surprise.png",
    shot_index: int = 0,
    metric: str = "js_divergence",
    reducer: str = "max",
    layout: dict | None = None,
    syndrome=None,
    figsize: tuple[float, float] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Render data nodes colored by contraction surprise during ML contraction.

    High-surprise data nodes are darker.  The default metric is JS divergence,
    which is finite even when directional KL diverges.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    surprises = ml_branch_surprises(
        recording,
        shot_index=shot_index,
        num_errors=num_vars,
        metric=metric,
        reducer=reducer,
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

    finite = surprises[np.isfinite(surprises)]
    finite = finite[finite >= 0.0]
    if finite.size:
        vmax = float(np.percentile(finite, 95))
        if vmax <= 0.0:
            vmax = float(finite.max()) if finite.size else 1.0
        vmax = max(vmax, 1e-12)
    else:
        vmax = 1.0

    clipped = np.nan_to_num(surprises, nan=0.0, posinf=vmax, neginf=0.0)
    clipped = np.clip(clipped, 0.0, vmax)
    normed = clipped / vmax if vmax > 0.0 else clipped
    cmap = plt.get_cmap("magma_r")
    var_faces = [cmap(0.08 + 0.88 * normed[v]) for v in range(num_vars)]

    syndrome_arr = (
        np.asarray(syndrome, dtype=np.uint8)
        if syndrome is not None
        else np.zeros(num_checks, dtype=np.uint8)
    )
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color="#dddddd",
            linewidth=0.5,
            alpha=0.65,
            zorder=1,
        )

    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    check_faces = ["black" if syndrome_arr[d] else "white" for d in range(num_checks)]
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
            value = surprises[v]
            label = f"{v}\n{value:.3g}" if np.isfinite(value) else f"{v}\ninf"
            ax.annotate(
                label,
                xy=var_pos[v],
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=4,
            )

    ax.set_title(f"ML contraction surprise by data node ({metric}, {reducer})", fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return surprises
