from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import BP_CORRECTION, FAINT_NODE, ML_CORRECTION, UNSATISFIED_CHECK


_BYTE_PARITY = np.asarray([int(i).bit_count() & 1 for i in range(256)], dtype=np.uint8)


def _as_int_list(value) -> list[int]:
    return [int(x) for x in np.asarray(value, dtype=np.int64).ravel()]


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


def _parity_for_mask(states: np.ndarray, mask: int) -> np.ndarray:
    values = np.ascontiguousarray(states & np.uint64(mask), dtype=np.uint64)
    bytes_view = values.view(np.uint8).reshape(values.size, 8)
    return np.bitwise_xor.reduce(_BYTE_PARITY[bytes_view], axis=1)


def _minsum_cost(x: np.ndarray, llr: float) -> np.ndarray:
    sign = np.where(x != 0, 1.0, -1.0)
    return np.maximum(0.0, sign * float(llr))


def _bits_from_state(state: int, width: int) -> list[int]:
    return [(int(state) >> k) & 1 for k in range(width)]


def _state_from_bits(bits: np.ndarray) -> int:
    state = 0
    for k, bit in enumerate(np.asarray(bits, dtype=np.uint8).ravel()):
        if int(bit) & 1:
            state |= 1 << k
    return state


def _region_valid_mask(region: dict, syndrome: np.ndarray, states: np.ndarray) -> np.ndarray:
    valid = np.ones(states.size, dtype=bool)
    masks = _as_int_list(region["internal_check_masks"])
    checks = _as_int_list(region["internal_check_indices"])
    for mask, check in zip(masks, checks):
        parity = _parity_for_mask(states, mask)
        valid &= parity == (int(syndrome[check]) & 1)
    return valid


def _region_weights(
    decoder,
    region: dict,
    iteration: dict,
    states: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    data = _as_int_list(region["data"])
    axis_edge = _as_int_list(region["axis_edge"])
    msg_v2c = np.asarray(iteration["msg_v2c"], dtype=np.float64)
    error_probs = np.asarray(decoder.error_probs, dtype=np.float64)
    ch_llr = np.log((1.0 - np.clip(error_probs, 1e-12, 1.0 - 1e-12)) /
                    np.clip(error_probs, 1e-12, 1.0 - 1e-12))

    incoming: list[float] = []
    weights = np.zeros(states.size, dtype=np.float64)
    for k, data_idx in enumerate(data):
        edge_idx = axis_edge[k] if k < len(axis_edge) else -1
        llr = float(msg_v2c[edge_idx]) if edge_idx >= 0 else float(ch_llr[data_idx])
        incoming.append(llr)
        x = ((states >> np.uint64(k)) & np.uint64(1)).astype(np.uint8)
        weights += _minsum_cost(x, llr)
    return weights, incoming


def _output_llrs(
    region: dict,
    states: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    incoming: list[float],
) -> list[dict]:
    outputs = []
    output_edges = _as_int_list(region["output_edges"])
    output_axes = _as_int_list(region["output_axes"])
    valid_states = states[valid]
    valid_weights = weights[valid]
    if valid_states.size == 0:
        return [
            {
                "edge": edge,
                "axis": axis,
                "llr": 0.0,
                "sign": 0,
                "favored_bit": None,
            }
            for edge, axis in zip(output_edges, output_axes)
        ]

    for edge, axis in zip(output_edges, output_axes):
        bits = ((valid_states >> np.uint64(axis)) & np.uint64(1)).astype(np.uint8)
        branch = valid_weights - _minsum_cost(bits, incoming[axis])
        W0 = float(branch[bits == 0].min()) if np.any(bits == 0) else float("inf")
        W1 = float(branch[bits == 1].min()) if np.any(bits == 1) else float("inf")
        if np.isinf(W0) and np.isinf(W1):
            llr = 0.0
        elif np.isinf(W0):
            llr = -1e30
        elif np.isinf(W1):
            llr = 1e30
        else:
            llr = W1 - W0
        outputs.append(
            {
                "edge": int(edge),
                "axis": int(axis),
                "llr": float(llr),
                "sign": int(np.sign(llr)),
                "favored_bit": int(llr < 0.0),
            }
        )
    return outputs


def gbp_region_diagnostics(
    decoder,
    recording: list,
    shot_index: int = -1,
    iteration_index: int = -1,
    only_active: bool = True,
    max_dense_states: int = 1 << 22,
) -> list[dict]:
    """
    Recompute local GBP region consistency diagnostics for one iteration.

    Each row compares the global hard decision projected onto a region with the
    region's exact locally valid state set.  Exact local tables are enumerated
    up to `max_dense_states`; wider regions are reported as skipped.
    """
    shot, iteration = _shot_and_iteration(recording, shot_index, iteration_index)
    syndrome = np.asarray(iteration["syndrome"], dtype=np.uint8)
    decision = np.asarray(iteration["decision"], dtype=np.uint8)
    active = set(_as_int_list(iteration.get("active_regions", [])))
    regions = list(shot["gbp"]["regions"])
    rows: list[dict] = []

    for region in regions:
        region_index = int(region["index"])
        is_active = region_index in active
        if only_active and not is_active:
            continue

        data = _as_int_list(region["data"])
        K = len(data)
        dense = 1 << K
        current_bits = decision[data] if data else np.zeros(0, dtype=np.uint8)
        current_state = _state_from_bits(current_bits)

        base = {
            "region": region_index,
            "active": bool(is_active),
            "num_axes": K,
            "num_internal_checks": len(_as_int_list(region["internal_check_indices"])),
            "dense_state_count": dense,
            "current_state": current_state,
            "current_bits": current_bits.astype(int).tolist(),
            "skipped": dense > max_dense_states,
        }
        if dense > max_dense_states:
            base.update(
                {
                    "current_valid": None,
                    "best_state": None,
                    "best_bits": None,
                    "best_cost": None,
                    "hamming_to_best": None,
                    "valid_state_count": None,
                    "output_llrs": [],
                }
            )
            rows.append(base)
            continue

        states = np.arange(dense, dtype=np.uint64)
        valid = _region_valid_mask(region, syndrome, states)
        weights, incoming = _region_weights(decoder, region, iteration, states)
        valid_count = int(valid.sum())

        current_valid = bool(valid[current_state]) if current_state < valid.size else False
        if valid_count:
            valid_indices = np.flatnonzero(valid)
            local_best_pos = int(np.argmin(weights[valid]))
            best_state = int(states[valid_indices[local_best_pos]])
            best_cost = float(weights[best_state])
            best_bits = np.asarray(_bits_from_state(best_state, K), dtype=np.uint8)
            hamming = int(np.count_nonzero(best_bits != current_bits))
        else:
            best_state = None
            best_cost = None
            best_bits = None
            hamming = None

        base.update(
            {
                "current_valid": current_valid,
                "best_state": best_state,
                "best_bits": None if best_bits is None else best_bits.astype(int).tolist(),
                "best_cost": best_cost,
                "hamming_to_best": hamming,
                "valid_state_count": valid_count,
                "incoming_llrs": incoming,
                "output_llrs": _output_llrs(region, states, valid, weights, incoming),
            }
        )
        rows.append(base)
    return rows


def write_gbp_region_diagnostics_csv(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "active",
        "num_axes",
        "num_internal_checks",
        "dense_state_count",
        "valid_state_count",
        "skipped",
        "current_valid",
        "hamming_to_best",
        "current_state",
        "best_state",
        "best_cost",
        "current_bits",
        "best_bits",
        "output_llrs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            for key in ("current_bits", "best_bits", "output_llrs"):
                out[key] = json.dumps(out[key], separators=(",", ":"))
            writer.writerow(out)


def plot_gbp_region_diagnostics(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    exact_rows = [row for row in rows if not row.get("skipped")]
    if not exact_rows:
        raise ValueError("no exact GBP region diagnostics to plot")

    region_ids = [int(row["region"]) for row in exact_rows]
    hamming = np.asarray([
        np.nan if row.get("hamming_to_best") is None else float(row["hamming_to_best"])
        for row in exact_rows
    ])
    invalid = np.asarray([row.get("current_valid") is False for row in exact_rows])
    valid_counts = np.asarray([
        max(1, int(row.get("valid_state_count") or 1))
        for row in exact_rows
    ], dtype=np.float64)
    dense_counts = np.asarray([
        max(1, int(row.get("dense_state_count") or 1))
        for row in exact_rows
    ], dtype=np.float64)
    sparsity = 1.0 - valid_counts / dense_counts

    xs = np.arange(len(exact_rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 0.35 * len(rows)), 9.0), sharex=True)
    colors = [UNSATISFIED_CHECK if bad else ML_CORRECTION for bad in invalid]
    axes[0].bar(xs, hamming, color=colors, alpha=0.85)
    axes[0].set_ylabel("Hamming distance to best valid state")
    axes[0].set_title("GBP region local-consistency diagnostics")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(xs, sparsity, color=[BP_CORRECTION if bad else FAINT_NODE for bad in invalid], alpha=0.8)
    axes[1].set_ylabel("region sparsity")
    axes[1].set_xlabel("GBP region")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([str(r) for r in region_ids], rotation=90 if len(xs) > 24 else 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
