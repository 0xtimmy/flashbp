from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flashbp.animation.layout import bipartite_layout, edges_from_H
from .style import (
    ACTIVE_CHECK,
    BP_CORRECTION,
    FAINT_EDGE,
    FAINT_NODE,
    ML_CORRECTION,
    UNSATISFIED_CHECK,
)


@dataclass
class BPOscillation:
    found: bool
    start: int
    end: int
    period: int
    key: str
    residual_weights: list[int]
    decision_weights: list[int]
    flipping_data: list[int]
    active_data: list[int]
    unsatisfied_checks: list[int]


def _state_key(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=np.uint8).tobytes()


def _residual(H: np.ndarray, syndrome: np.ndarray, decision: np.ndarray) -> np.ndarray:
    predicted = (H @ decision.astype(np.int32)) % 2
    return predicted.astype(np.uint8) ^ syndrome.astype(np.uint8)


def detect_bp_oscillation(
    decoder,
    recording: list,
    shot_index: int = -1,
    key: str = "decision",
) -> BPOscillation:
    """
    Detect a repeated BP trajectory state in a RecordLogger recording.

    `key="decision"` detects cycles in the hard-decision vector.  `key="residual"`
    detects cycles in the residual syndrome.  Returned node sets describe the
    repeated orbit when one is found, otherwise the full recorded trajectory.
    """
    if key not in ("decision", "residual"):
        raise ValueError("key must be 'decision' or 'residual'")
    if not recording:
        raise ValueError("recording is empty")
    shot = recording[shot_index]
    iterations = shot.get("iterations", [])
    if not iterations:
        raise ValueError("recording shot has no iterations")

    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(iterations[0]["syndrome"], dtype=np.uint8)
    decisions = [
        np.asarray(it["decision"], dtype=np.uint8)
        for it in iterations
    ]
    residuals = [_residual(H, syndrome, decision) for decision in decisions]

    seen: dict[bytes, int] = {}
    found = False
    start = 0
    end = len(iterations) - 1
    for i, (decision, residual) in enumerate(zip(decisions, residuals)):
        item = decision if key == "decision" else residual
        item_key = _state_key(item)
        if item_key in seen:
            start = seen[item_key]
            end = i
            found = True
            break
        seen[item_key] = i

    period = end - start if found else 0
    orbit_decisions = decisions[start : end + 1] if found else decisions
    orbit_residuals = residuals[start : end + 1] if found else residuals

    decision_stack = np.stack(orbit_decisions, axis=0)
    residual_stack = np.stack(orbit_residuals, axis=0)
    flipping = np.flatnonzero(decision_stack.max(axis=0) != decision_stack.min(axis=0))
    active_data = np.flatnonzero(decision_stack.max(axis=0) > 0)
    unsatisfied = np.flatnonzero(residual_stack.max(axis=0) > 0)

    return BPOscillation(
        found=found,
        start=start,
        end=end,
        period=period,
        key=key,
        residual_weights=[int(r.sum()) for r in residuals],
        decision_weights=[int(d.sum()) for d in decisions],
        flipping_data=flipping.astype(int).tolist(),
        active_data=active_data.astype(int).tolist(),
        unsatisfied_checks=unsatisfied.astype(int).tolist(),
    )


def plot_bp_oscillation_graph(
    decoder,
    recording: list,
    output_path: str | Path = "bp_oscillation_graph.png",
    shot_index: int = -1,
    key: str = "decision",
    layout: dict | None = None,
    show_labels: bool = True,
    figsize: tuple[float, float] | None = None,
) -> BPOscillation:
    result = detect_bp_oscillation(decoder, recording, shot_index=shot_index, key=key)

    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
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

    flipping = set(result.flipping_data)
    active_data = set(result.active_data)
    unsatisfied = set(result.unsatisfied_checks)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges:
        important = v in flipping or d in unsatisfied
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=BP_CORRECTION if important else FAINT_EDGE,
            linewidth=1.8 if important else 0.5,
            alpha=0.85 if important else 0.45,
            zorder=2 if important else 1,
        )

    var_faces = []
    var_edges = []
    var_lws = []
    for v in range(num_vars):
        if v in flipping:
            var_faces.append(BP_CORRECTION)
            var_edges.append(BP_CORRECTION)
            var_lws.append(2.4)
        elif v in active_data:
            var_faces.append("#ffe1c2")
            var_edges.append(BP_CORRECTION)
            var_lws.append(1.5)
        else:
            var_faces.append("white")
            var_edges.append(FAINT_NODE)
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

    check_faces = [
        UNSATISFIED_CHECK if d in unsatisfied else "white"
        for d in range(num_checks)
    ]
    check_edges = [
        UNSATISFIED_CHECK if d in unsatisfied else ACTIVE_CHECK
        for d in range(num_checks)
    ]
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=[2.0 if d in unsatisfied else 1.0 for d in range(num_checks)],
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

    status = (
        f"oscillation period={result.period}  start={result.start}  end={result.end}"
        if result.found
        else "no repeated state detected"
    )
    ax.set_title(
        f"simple BP trapping-set candidate    {status}",
        fontsize=12,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return result


def plot_bp_oscillation_trace(
    oscillation: BPOscillation,
    output_path: str | Path = "bp_oscillation_trace.png",
    figsize: tuple[float, float] = (12, 7),
) -> None:
    xs = np.arange(len(oscillation.residual_weights))
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(xs, oscillation.residual_weights, marker="o", color=UNSATISFIED_CHECK)
    axes[0].set_ylabel("residual syndrome weight")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, oscillation.decision_weights, marker="o", color=ML_CORRECTION)
    axes[1].set_ylabel("decision weight")
    axes[1].set_xlabel("BP iteration")
    axes[1].grid(True, alpha=0.3)

    if oscillation.found:
        for ax in axes:
            ax.axvspan(
                oscillation.start,
                oscillation.end,
                color=BP_CORRECTION,
                alpha=0.15,
                label="repeated orbit",
            )
        axes[0].legend()

    fig.suptitle(
        "simple BP oscillation trace"
        + (
            f"    period={oscillation.period}"
            if oscillation.found
            else "    no repeat detected"
        )
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
