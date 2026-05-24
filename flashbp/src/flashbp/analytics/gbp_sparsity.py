from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from flashbp.animation.layout import bipartite_layout, edges_from_H

from .style import ACTIVE_CHECK, BP_CORRECTION, CYCLE, FAINT_EDGE, FAINT_NODE


_BYTE_PARITY = np.asarray([int(i).bit_count() & 1 for i in range(256)], dtype=np.uint8)


def _shot_and_iteration(recording: list, shot_index: int, iteration_index: int):
    if not recording:
        raise ValueError("GBP recording is empty")
    shot = recording[shot_index]
    if "gbp" not in shot:
        raise ValueError("recording has no GBP metadata; use log_type='gbp'")
    iterations = shot.get("iterations", [])
    if not iterations:
        raise ValueError("GBP recording shot has no iterations")
    return shot, iterations[iteration_index]


def _region_list(recording: list, shot_index: int = -1) -> list[dict]:
    shot = recording[shot_index]
    if "gbp" not in shot:
        raise ValueError("recording has no GBP metadata; use log_type='gbp'")
    return list(shot["gbp"]["regions"])


def _as_int_list(value) -> list[int]:
    return [int(x) for x in np.asarray(value, dtype=np.int64).ravel()]


def _gf2_rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for row in rows:
        x = int(row)
        while x:
            pivot = x.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = x
                rank += 1
                break
            x ^= basis[pivot]
    return rank


def _consistent_rank(masks: list[int], rhs: list[int], num_axes: int) -> tuple[bool, int]:
    augmented = [mask | (int(bit) << num_axes) for mask, bit in zip(masks, rhs)]
    rank_a = _gf2_rank(masks)
    rank_aug = _gf2_rank(augmented)
    return rank_a == rank_aug, rank_a


def region_sparsity_stats(region: dict, syndrome: np.ndarray) -> dict:
    data = _as_int_list(region["data"])
    checks = _as_int_list(region["internal_check_indices"])
    masks = _as_int_list(region["internal_check_masks"])
    num_axes = len(data)
    dense = 1 << num_axes
    rhs = [int(syndrome[c]) & 1 for c in checks]
    consistent, rank = _consistent_rank(masks, rhs, num_axes)
    valid = (1 << (num_axes - rank)) if consistent else 0
    valid_fraction = valid / dense if dense else 0.0
    compression = dense / valid if valid else float("inf")
    return {
        "index": int(region["index"]),
        "num_axes": num_axes,
        "num_internal_checks": len(checks),
        "rank": rank,
        "dense_state_count": dense,
        "valid_state_count": valid,
        "valid_fraction": valid_fraction,
        "sparsity": 1.0 - valid_fraction,
        "compression": compression,
    }


def gbp_sparsity_stats(
    recording: list,
    shot_index: int = -1,
    iteration_index: int = -1,
) -> list[dict]:
    shot, iteration = _shot_and_iteration(recording, shot_index, iteration_index)
    syndrome = np.asarray(iteration["syndrome"], dtype=np.uint8)
    active = set(_as_int_list(iteration.get("active_regions", [])))
    stats = []
    for region in shot["gbp"]["regions"]:
        row = region_sparsity_stats(region, syndrome)
        row["active"] = row["index"] in active
        stats.append(row)
    return stats


def choose_region(
    recording: list,
    shot_index: int = -1,
    iteration_index: int = -1,
    region_index: int | None = None,
) -> int:
    if region_index is not None:
        return int(region_index)
    stats = gbp_sparsity_stats(recording, shot_index, iteration_index)
    active = [row for row in stats if row["active"]]
    candidates = active or stats
    if not candidates:
        raise ValueError("GBP recording has no regions")
    return max(
        candidates,
        key=lambda row: (
            row["sparsity"],
            row["compression"] if np.isfinite(row["compression"]) else 1e300,
            row["num_axes"],
        ),
    )["index"]


def plot_gbp_sparsity_summary(
    recording: list,
    output_path: str | Path,
    shot_index: int = -1,
    iteration_index: int = -1,
    sort_by: str = "region",
) -> list[dict]:
    stats = gbp_sparsity_stats(recording, shot_index, iteration_index)
    if sort_by == "sparsity":
        stats = sorted(stats, key=lambda row: (row["sparsity"], row["index"]))
    elif sort_by == "compression":
        stats = sorted(stats, key=lambda row: (row["compression"], row["index"]))
    else:
        stats = sorted(stats, key=lambda row: row["index"])

    xs = np.arange(len(stats))
    sparsity = np.asarray([row["sparsity"] for row in stats], dtype=np.float64)
    axes = np.asarray([row["num_axes"] for row in stats], dtype=np.float64)
    active = np.asarray([row["active"] for row in stats], dtype=bool)
    labels = [str(row["index"]) for row in stats]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.26 * len(stats)), 5.2))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=max(0.0, float(axes.min(initial=0))), vmax=max(1.0, float(axes.max(initial=1))))
    colors = [cmap(norm(k)) if act else "#d8d8d8" for k, act in zip(axes, active)]
    edges = [BP_CORRECTION if act else FAINT_NODE for act in active]
    ax.bar(xs, sparsity, color=colors, edgecolor=edges, linewidth=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("GBP region")
    ax.set_ylabel("sparsity = 1 - valid / dense")
    ax.set_title("GBP region sparsity summary")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 24 else 0)
    ax.grid(True, axis="y", alpha=0.25)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("data axes K")
    active_count = int(active.sum())
    ax.text(
        0.01,
        0.98,
        f"active regions: {active_count}/{len(stats)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return stats


def _parity_for_mask(states: np.ndarray, mask: int) -> np.ndarray:
    values = np.ascontiguousarray(states & np.uint64(mask), dtype=np.uint64)
    bytes_view = values.view(np.uint8).reshape(values.size, 8)
    return np.bitwise_xor.reduce(_BYTE_PARITY[bytes_view], axis=1)


def valid_state_mask(region: dict, syndrome: np.ndarray) -> np.ndarray:
    num_axes = len(_as_int_list(region["data"]))
    dense = 1 << num_axes
    states = np.arange(dense, dtype=np.uint64)
    valid = np.ones(dense, dtype=bool)
    masks = _as_int_list(region["internal_check_masks"])
    checks = _as_int_list(region["internal_check_indices"])
    for mask, check in zip(masks, checks):
        parity = _parity_for_mask(states, mask)
        valid &= parity == (int(syndrome[check]) & 1)
    return valid


def plot_gbp_region_heatmap(
    recording: list,
    output_path: str | Path,
    region_index: int | None = None,
    shot_index: int = -1,
    iteration_index: int = -1,
) -> tuple[int, np.ndarray]:
    shot, iteration = _shot_and_iteration(recording, shot_index, iteration_index)
    syndrome = np.asarray(iteration["syndrome"], dtype=np.uint8)
    region_index = choose_region(recording, shot_index, iteration_index, region_index)
    regions = shot["gbp"]["regions"]
    region = next((r for r in regions if int(r["index"]) == region_index), None)
    if region is None:
        raise ValueError(f"region {region_index} not found")

    valid = valid_state_mask(region, syndrome)
    num_axes = len(_as_int_list(region["data"]))
    dense = valid.size
    low_bits = max(1, num_axes // 2)
    cols = 1 << low_bits
    rows = int(np.ceil(dense / cols))
    matrix = np.full(rows * cols, 1e-5, dtype=np.float64)
    matrix[:dense] = np.where(valid, 1.0, 1e-5)
    matrix = matrix.reshape(rows, cols)

    stats = region_sparsity_stats(region, syndrome)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(matrix, origin="upper", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel(f"low {low_bits} state bits")
    ax.set_ylabel(f"high {num_axes - low_bits} state bits")
    ax.set_title(
        f"region {region_index} valid-state table    "
        f"K={num_axes}  valid={stats['valid_state_count']}/{stats['dense_state_count']}  "
        f"compression={stats['compression']:.3g}x"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("valid state")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return region_index, matrix


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


def _draw_region_hull(ax, region: dict, layout: dict, color, alpha: float, linewidth: float, zorder: int):
    points = []
    for v in _as_int_list(region["data"]):
        if v in layout["var_pos"]:
            points.append(layout["var_pos"][v])
    checks = set(_as_int_list(region["cycle_checks"]))
    checks.update(_as_int_list(region["internal_check_indices"]))
    for c in checks:
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
        expanded.append(tuple(centroid + 1.08 * (np.asarray([x, y]) - centroid)))
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


def plot_gbp_sparsity_graph(
    decoder,
    recording: list,
    output_path: str | Path,
    shot_index: int = -1,
    iteration_index: int = -1,
    layout: dict | None = None,
    show_labels: bool = True,
) -> list[dict]:
    shot, iteration = _shot_and_iteration(recording, shot_index, iteration_index)
    syndrome = np.asarray(iteration["syndrome"], dtype=np.uint8)
    H = np.asarray(decoder.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    figsize = layout.get("figsize") or (
        8.0,
        min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0),
    )
    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    stats = gbp_sparsity_stats(recording, shot_index, iteration_index)
    stats_by_index = {row["index"]: row for row in stats}
    active = set(_as_int_list(iteration.get("active_regions", [])))
    regions = list(shot["gbp"]["regions"])

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for region in regions:
        idx = int(region["index"])
        if idx not in active:
            continue
        stat = stats_by_index[idx]
        _draw_region_hull(
            ax,
            region,
            layout,
            cmap(norm(stat["sparsity"])),
            alpha=0.08 + 0.22 * stat["sparsity"],
            linewidth=0.8 + 2.0 * stat["sparsity"],
            zorder=1,
        )

    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    for d, v in edges_from_H(H):
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=FAINT_EDGE,
            linewidth=0.45,
            alpha=0.55,
            zorder=2,
        )

    active_vars = set()
    active_checks = set()
    for region in regions:
        if int(region["index"]) not in active:
            continue
        active_vars.update(_as_int_list(region["data"]))
        active_checks.update(_as_int_list(region["cycle_checks"]))
        active_checks.update(_as_int_list(region["internal_check_indices"]))

    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=[CYCLE if v in active_vars else "white" for v in range(num_vars)],
        edgecolors=["black" if v in active_vars else FAINT_NODE for v in range(num_vars)],
        linewidths=[1.4 if v in active_vars else 0.6 for v in range(num_vars)],
        zorder=4,
    )
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=[ACTIVE_CHECK if syndrome[d] else ("white" if d not in active_checks else CYCLE)
           for d in range(num_checks)],
        edgecolors=["black" if d in active_checks or syndrome[d] else FAINT_NODE
                    for d in range(num_checks)],
        linewidths=[1.5 if d in active_checks or syndrome[d] else 0.6
                    for d in range(num_checks)],
        marker="s",
        zorder=4,
    )

    if show_labels:
        label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
        for v in range(num_vars):
            ax.annotate(str(v), xy=var_pos[v], xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=label_fontsize, color="black", zorder=5)
        for d in range(num_checks):
            ax.annotate(str(d), xy=check_pos[d], xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=label_fontsize,
                        color="white" if syndrome[d] else "black", zorder=5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("region sparsity")
    ax.set_title(
        f"GBP active-region sparsity overlay    active={len(active)}/{len(regions)}",
        fontsize=12,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return stats
