"""
Render a single Tanner-graph frame for one BP iteration.

Visual encoding
---------------
Variable nodes (circles, left column):
  face color : white if decision=0, red if decision=1
  outline    : black

Check nodes (squares, right column):
  face color : white if syndrome=0, black if syndrome=1
  outline    : orange + thick if currently unsatisfied (parity of decision
               restricted to neighbours != syndrome bit), else black

Edges:
  color    : coolwarm map on msg_v2c + msg_c2v (sign = belief direction)
  width    : scaled by max(|msg_v2c|, |msg_c2v|)
  alpha    : scaled by the same magnitude

Annotations (when show_weights=True):
  each variable node is labelled with its current total LLR
  (= channel prior + sum of incoming check-to-variable messages,
   recovered as msg_v2c[i] + msg_c2v[i] for any edge i of that variable).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .layout import edges_from_H


def _unsatisfied_checks(H: np.ndarray, syndrome: np.ndarray,
                        decision: np.ndarray) -> np.ndarray:
    pred_syn = (H @ decision.astype(np.int32)) % 2
    return pred_syn.astype(np.uint8) != syndrome.astype(np.uint8)


def render_frame(
    iteration_record: dict,
    H: np.ndarray,
    layout: dict,
    output_path: str | Path,
    figsize: tuple[float, float] | None = None,
    llr_scale: float = 3.0,
    true_errors: np.ndarray | None = None,
    show_weights: bool = True,
) -> None:
    """
    Render one IterationRecord (dict from RecordLogger.get_recording()) to PNG.

    Parameters
    ----------
    iteration_record : dict
        Keys: iteration, syndrome, decision, msg_v2c, msg_c2v.
    H : (num_checks, num_vars) parity-check matrix.
    layout : dict from bipartite_layout().
    output_path : where to write the PNG.
    figsize : matplotlib figsize; auto-scaled with node count when None.
    llr_scale : LLR magnitude that maps to full color saturation. Larger
        values flatten the dynamic range.
    """
    num_checks, num_vars = H.shape

    if figsize is None:
        figsize = layout.get("figsize")
    if figsize is None:
        height  = max(8.0, 0.25 * max(num_vars, num_checks))
        figsize = (8.0, min(height, 30.0))

    syndrome = np.asarray(iteration_record["syndrome"])
    decision = np.asarray(iteration_record["decision"])
    msg_v2c  = np.asarray(iteration_record["msg_v2c"])
    msg_c2v  = np.asarray(iteration_record["msg_c2v"])

    edges = edges_from_H(H)

    var_pos   = layout["var_pos"]
    check_pos = layout["check_pos"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── edges first (behind nodes) ──────────────────────────────────────────
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=-llr_scale, vmax=llr_scale)
    edge_sum = msg_v2c + msg_c2v
    edge_mag = np.maximum(np.abs(msg_v2c), np.abs(msg_c2v))
    max_mag  = max(float(edge_mag.max()), 1e-6) if len(edge_mag) else 1.0

    # draw weak edges first so strong ones land on top
    order = np.argsort(edge_mag)
    for i in order:
        d, v = edges[i]
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        intensity = float(edge_mag[i] / max_mag)
        color     = cmap(norm(float(edge_sum[i])))
        ax.plot(
            [x1, x2], [y1, y2],
            color=color,
            alpha=0.15 + 0.75 * intensity,
            linewidth=0.4 + 2.5 * intensity,
            zorder=1,
        )

    # ── nodes ───────────────────────────────────────────────────────────────
    base_size = layout.get(
        "node_size",
        max(60.0, 4000.0 / max(num_vars, num_checks)),
    )

    # variable nodes — face = BP decision, outline = ground truth when known
    var_xs = [var_pos[v][0] for v in range(num_vars)]
    var_ys = [var_pos[v][1] for v in range(num_vars)]
    var_faces = ["#d62728" if decision[v] else "white" for v in range(num_vars)]
    if true_errors is not None:
        var_edges = ["#1f77b4" if true_errors[v] else "black"
                     for v in range(num_vars)]
        var_lws   = [2.2       if true_errors[v] else 1.0
                     for v in range(num_vars)]
    else:
        var_edges = ["black"] * num_vars
        var_lws   = [1.0]     * num_vars
    ax.scatter(var_xs, var_ys, s=base_size, c=var_faces,
               edgecolors=var_edges, linewidths=var_lws, zorder=3)

    # ── per-variable weight labels (total LLR) ──────────────────────────────
    # After the variable update, msg_v2c[i] + msg_c2v[i] == total_llr[var(i)]
    # for every edge of that variable, so any single edge recovers it.
    if show_weights:
        total_llr = np.zeros(num_vars, dtype=np.float64)
        seen      = np.zeros(num_vars, dtype=bool)
        for i, (_, v) in enumerate(edges):
            if not seen[v]:
                total_llr[v] = float(msg_v2c[i] + msg_c2v[i])
                seen[v] = True

        def _short(v: float) -> str:
            if not np.isfinite(v):
                return "+inf" if v > 0 else "-inf"
            a = abs(v)
            if a >= 9999.5:    return ("+" if v > 0 else "-") + "inf"
            if a >= 99.5:      return f"{v:.0f}"   # up to 4 digits, e.g. "1234"
            if a >= 9.95:      return f"{v:.1f}"   # 3 digits, e.g. "12.3"
            return f"{v:.2f}"                      # 3 digits, e.g. "1.23"

        label_fontsize = max(4.0, min(9.0, 60.0 / np.sqrt(num_vars)))
        offset_pts     = label_fontsize + 2.0
        for v in range(num_vars):
            x, y = var_pos[v]
            ax.annotate(
                _short(total_llr[v]),
                xy=(x, y),
                xytext=(0, -offset_pts),
                textcoords="offset points",
                ha="center", va="top",
                fontsize=label_fontsize,
                color="black",
                zorder=4,
            )

    # check nodes (squares)
    unsat   = _unsatisfied_checks(H, syndrome, decision)
    chk_xs  = [check_pos[d][0] for d in range(num_checks)]
    chk_ys  = [check_pos[d][1] for d in range(num_checks)]
    chk_faces = ["black" if syndrome[d] else "white" for d in range(num_checks)]
    chk_edges = ["#ff7f0e" if unsat[d] else "black" for d in range(num_checks)]
    chk_lw    = [2.5      if unsat[d] else 1.0      for d in range(num_checks)]
    ax.scatter(chk_xs, chk_ys, s=base_size, c=chk_faces,
               edgecolors=chk_edges, linewidths=chk_lw,
               marker="s", zorder=3)

    # ── title ───────────────────────────────────────────────────────────────
    n          = int(iteration_record["iteration"])
    dec_weight = int(decision.sum())
    n_unsat    = int(unsat.sum())
    ax.set_title(
        f"iter={n}    decision_weight={dec_weight}    unsatisfied_checks={n_unsat}",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
