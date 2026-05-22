from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from flashbp.analytics.gbp import (
    GBPRegionInfo,
    active_gbp_regions,
    build_gbp_regions,
    region_is_active,
)
from flashbp.analytics.style import (
    ACTIVE_CHECK,
    BP_CORRECTION,
    CYCLE,
    FAINT_EDGE,
    FAINT_NODE,
    TRUE_ERROR,
    UNSATISFIED_CHECK,
)

from .layout import bipartite_layout, edges_from_H
from .video import make_video


def _unsatisfied_checks(
    H: np.ndarray,
    syndrome: np.ndarray,
    decision: np.ndarray,
) -> np.ndarray:
    pred_syn = (H @ decision.astype(np.int32)) % 2
    return pred_syn.astype(np.uint8) != syndrome.astype(np.uint8)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _draw_region_patch(
    ax,
    region: GBPRegionInfo,
    layout: dict,
    num_vars: int,
    color: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    points = []
    for v in region.data:
        if v in layout["var_pos"]:
            points.append(layout["var_pos"][v])
    for c in set(region.cycle_checks) | set(region.internal_checks):
        if c in layout["check_pos"]:
            points.append(layout["check_pos"][c])
    if len(points) < 3:
        return
    hull = _convex_hull(points)
    if len(hull) < 3:
        return
    centroid = np.asarray(hull, dtype=np.float64).mean(axis=0)
    expanded = []
    for x, y in hull:
        vec = np.asarray([x, y]) - centroid
        expanded.append(tuple(centroid + 1.08 * vec))
    ax.add_patch(
        Polygon(
            expanded,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def _selected_region(
    active_regions: list[GBPRegionInfo],
    frame_region_index: int | None,
) -> GBPRegionInfo | None:
    if not active_regions:
        return None
    if frame_region_index is None:
        return active_regions[0]
    return active_regions[frame_region_index % len(active_regions)]


def _total_llr_by_var(
    edges: list[tuple[int, int]],
    msg_v2c: np.ndarray,
    msg_c2v: np.ndarray,
    num_vars: int,
) -> np.ndarray:
    total_llr = np.zeros(num_vars, dtype=np.float64)
    seen = np.zeros(num_vars, dtype=bool)
    for i, (_, v) in enumerate(edges):
        if not seen[v]:
            total_llr[v] = float(msg_v2c[i] + msg_c2v[i])
            seen[v] = True
    return total_llr


def render_gbp_frame(
    iteration_record: dict,
    H: np.ndarray,
    regions: list[GBPRegionInfo],
    layout: dict,
    output_path: str | Path,
    frame_region_index: int | None = None,
    true_errors: np.ndarray | None = None,
    figsize: tuple[float, float] | None = None,
    llr_scale: float = 3.0,
    show_labels: bool = True,
) -> None:
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    syndrome = np.asarray(iteration_record["syndrome"], dtype=np.uint8)
    decision = np.asarray(iteration_record["decision"], dtype=np.uint8)
    msg_v2c = np.asarray(iteration_record["msg_v2c"], dtype=np.float64)
    msg_c2v = np.asarray(iteration_record["msg_c2v"], dtype=np.float64)

    active_regions = active_gbp_regions(regions, syndrome)
    selected = _selected_region(active_regions, frame_region_index)
    selected_vars = set(selected.data) if selected else set()
    selected_checks = (
        set(selected.cycle_checks) | set(selected.internal_checks)
        if selected
        else set()
    )

    if figsize is None:
        base_figsize = layout.get("figsize")
        if base_figsize is None:
            height = max(8.0, 0.25 * max(num_vars, num_checks))
            base_figsize = (8.0, min(height, 30.0))
        figsize = (base_figsize[0] * 1.55, base_figsize[1])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.75], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    edges = edges_from_H(H)
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    base_size = layout.get(
        "node_size",
        max(60.0, 4000.0 / max(num_vars, num_checks)),
    )

    for region in active_regions:
        if selected is not None and region.index == selected.index:
            continue
        _draw_region_patch(
            ax, region, layout, num_vars,
            color=CYCLE, alpha=0.045, linewidth=0.6, zorder=0,
        )
    if selected is not None:
        _draw_region_patch(
            ax, selected, layout, num_vars,
            color=CYCLE, alpha=0.16, linewidth=1.6, zorder=1,
        )

    edge_scores = msg_v2c + msg_c2v
    edge_mag = np.maximum(np.abs(msg_v2c), np.abs(msg_c2v))
    max_mag = max(float(edge_mag.max()), 1e-6) if len(edge_mag) else 1.0
    cmap = plt.get_cmap("coolwarm")
    for i in np.argsort(edge_mag):
        d, v = edges[i]
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        in_region = selected is not None and v in selected_vars and d in selected_checks
        intensity = float(edge_mag[i] / max_mag)
        if in_region:
            color = BP_CORRECTION
            alpha = 0.70
            linewidth = 1.0 + 2.5 * intensity
            zorder = 3
        else:
            color = cmap(np.clip(0.5 + float(edge_scores[i]) / (2.0 * llr_scale), 0, 1))
            alpha = 0.08 + 0.55 * intensity
            linewidth = 0.35 + 1.5 * intensity
            zorder = 2
        ax.plot([x1, x2], [y1, y2],
                color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

    var_faces = [TRUE_ERROR if decision[v] else "white" for v in range(num_vars)]
    var_edges = []
    var_lws = []
    for v in range(num_vars):
        if v in selected_vars:
            var_edges.append(CYCLE)
            var_lws.append(2.5)
        elif true_errors is not None and true_errors[v]:
            var_edges.append(TRUE_ERROR)
            var_lws.append(2.2)
        else:
            var_edges.append("black")
            var_lws.append(1.0)
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=5,
    )

    unsat = _unsatisfied_checks(H, syndrome, decision)
    check_faces = [ACTIVE_CHECK if syndrome[d] else "white" for d in range(num_checks)]
    check_edges = []
    check_lws = []
    for d in range(num_checks):
        if d in selected_checks:
            check_edges.append(CYCLE)
            check_lws.append(2.6)
        elif unsat[d]:
            check_edges.append(UNSATISFIED_CHECK)
            check_lws.append(2.5)
        else:
            check_edges.append("black")
            check_lws.append(1.0)
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=check_lws,
        marker="s",
        zorder=5,
    )

    if show_labels:
        total_llr = _total_llr_by_var(edges, msg_v2c, msg_c2v, num_vars)
        label_fontsize = max(4.0, min(8.5, 58.0 / np.sqrt(max(1, num_vars))))
        for v in range(num_vars):
            x, y = var_pos[v]
            ax.annotate(
                str(v),
                xy=(x, y),
                xytext=(0, label_fontsize + 1),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=6,
            )
            if v in selected_vars:
                ax.annotate(
                    f"{total_llr[v]:.2g}",
                    xy=(x, y),
                    xytext=(0, -(label_fontsize + 3)),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=label_fontsize,
                    color="black",
                    zorder=6,
                )
        for d in range(num_checks):
            x, y = check_pos[d]
            ax.annotate(
                str(d),
                xy=(x, y),
                xytext=(0, label_fontsize + 1),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
                zorder=6,
            )

    iter_idx = int(iteration_record["iteration"])
    dec_weight = int(decision.sum())
    ax.set_title(
        f"GBP iter={iter_idx}    decision_weight={dec_weight}    "
        f"unsatisfied_checks={int(unsat.sum())}",
        fontsize=12,
    )

    active_flags = np.array([region_is_active(r, syndrome) for r in regions], dtype=bool)
    sizes = np.array([len(r.data) for r in regions], dtype=np.float64)
    active_sizes = np.where(active_flags, sizes, 0.0)
    colors = [CYCLE if flag else FAINT_NODE for flag in active_flags]
    if selected is not None:
        colors[selected.index] = BP_CORRECTION

    ax_stats.bar(np.arange(len(regions)), active_sizes, color=colors, edgecolor="black",
                 linewidth=0.35)
    ax_stats.set_xlabel("region")
    ax_stats.set_ylabel("data axes in active region")
    max_size = float(sizes.max()) if sizes.size else 1.0
    ax_stats.set_ylim(0, max(1.0, max_size * 1.1))
    ax_stats.grid(True, axis="y", alpha=0.25)
    ax_stats.set_title(
        f"active={int(active_flags.sum())}/{len(regions)}",
        fontsize=11,
    )
    if selected is not None:
        text = (
            f"selected region: {selected.index}\n"
            f"policy: {selected.policy}\n"
            f"activation: {selected.activation}\n"
            f"fallback: {selected.is_fallback}\n"
            f"checks: {list(selected.cycle_checks)}\n"
            f"data axes: {len(selected.data)}\n"
            f"internal checks: {len(selected.internal_checks)}"
        )
    else:
        text = "no active regions"
    ax_stats.text(
        0.02,
        -0.22,
        text,
        transform=ax_stats.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def animate_gbp_recording(
    bp,
    recording: list,
    output_dir: str | Path,
    policy: str = "check_neighborhood",
    degree: int = 2,
    shot_index: int = 0,
    framerate: float = 2.0,
    video_name: str = "gbp.mp4",
    layout: dict | None = None,
    true_errors: np.ndarray | None = None,
    max_region_frames: int = 24,
) -> Path:
    if not recording:
        raise ValueError("recording is empty; run with log=True, log_type='record'.")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H = np.asarray(bp.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    regions = build_gbp_regions(H, policy=policy, degree=degree)

    shot = recording[shot_index]
    iterations = shot["iterations"]
    if not iterations:
        raise ValueError(f"shot {shot_index} has no recorded iterations.")

    frame = 0
    for it in iterations:
        syndrome = np.asarray(it["syndrome"], dtype=np.uint8)
        active_regions = active_gbp_regions(regions, syndrome)
        n_region_frames = max(1, min(max_region_frames, len(active_regions)))
        for region_frame in range(n_region_frames):
            render_gbp_frame(
                it,
                H,
                regions,
                layout,
                frames_dir / f"frame_{frame:04d}.png",
                frame_region_index=region_frame if active_regions else None,
                true_errors=true_errors,
            )
            frame += 1

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)
