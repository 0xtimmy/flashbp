from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .layout import bipartite_layout, edges_from_H
from .video import make_video
from flashbp.analytics.style import ACTIVE_CHECK, BP_CORRECTION, FAINT_EDGE, ML_CORRECTION


def render_surprise_ml_frame(
    H,
    syndrome,
    layout,
    step,
    contracted,
    output_path: str | Path,
    figsize: tuple[float, float] | None = None,
) -> None:
    H = np.asarray(H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if figsize is None:
        figsize = layout.get("figsize") or (
            8.0, min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0)
        )
    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)

    scores = np.asarray(step.get("surprise_scores", []), dtype=np.float64)
    if scores.size != num_vars:
        scores = np.full(num_vars, np.nan, dtype=np.float64)
    chosen = int(step.get("error_idx", -1))

    finite = scores[np.isfinite(scores) & (scores >= 0.0)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    scaled = np.nan_to_num(scores, nan=0.0, posinf=vmax, neginf=0.0)
    scaled = np.clip(scaled, 0.0, vmax) / vmax
    cmap = plt.get_cmap("magma_r")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=FAINT_EDGE,
            linewidth=0.45,
            alpha=0.65,
            zorder=1,
        )

    contracted = set(contracted)
    var_faces = [cmap(0.08 + 0.88 * scaled[v]) for v in range(num_vars)]
    var_edges = []
    var_lws = []
    for v in range(num_vars):
        if v == chosen:
            var_edges.append(ML_CORRECTION)
            var_lws.append(3.2)
        elif v in contracted:
            var_edges.append(BP_CORRECTION)
            var_lws.append(2.4)
        else:
            var_edges.append("black")
            var_lws.append(0.8)
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=3,
    )

    check_faces = [ACTIVE_CHECK if syndrome[d] else "white" for d in range(num_checks)]
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

    label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
    for v in range(num_vars):
        x, y = var_pos[v]
        if np.isfinite(scores[v]) and scores[v] >= 0.0:
            label = f"{v}\n{scores[v]:.2g}"
        else:
            label = f"{v}"
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

    ax.set_title(
        f"SurpriseML scores    chosen={chosen}    JS={float(step.get('js_divergence', 0.0)):.4g}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def animate_surprise_ml_recording(
    bp,
    recording,
    output_dir: str | Path,
    shot_index: int = 0,
    framerate: float = 2.0,
    video_name: str = "surprise_ml.mp4",
    layout: dict | None = None,
) -> Path:
    if not recording:
        raise ValueError("recording is empty; run with log_type='surprise_ml'.")
    shot = recording[shot_index]
    steps = shot.get("steps", [])
    if len(steps) <= 1:
        raise ValueError("SurpriseML recording has no contraction score steps.")

    H = np.asarray(bp.H, dtype=np.uint8)
    syndrome = np.asarray(shot["syndrome"], dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)

    output_dir = Path(output_dir)
    frames_dir = output_dir / "surprise_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    contracted: list[int] = []
    score_steps = steps[1:]
    for i, step in enumerate(score_steps):
        render_surprise_ml_frame(
            H,
            syndrome,
            layout,
            step,
            contracted,
            frames_dir / f"frame_{i:04d}.png",
        )
        chosen = int(step.get("error_idx", -1))
        if chosen >= 0:
            contracted.append(chosen)

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)
