from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from flashbp.animation.layout import bipartite_layout, edges_from_H
from .style import ACTIVE_CHECK, BP_CORRECTION, FAINT_EDGE, FAINT_NODE, ML_CORRECTION


def _as_int_list(value) -> list[int]:
    return [int(x) for x in np.asarray(value, dtype=np.int64).ravel()]


def _regions_by_signature(recording: list, shot_index: int = -1) -> dict[tuple, dict]:
    shot = recording[shot_index]
    if "gbp" not in shot:
        raise ValueError("recording has no GBP metadata; use log_type='gbp'")
    out = {}
    for region in shot["gbp"]["regions"]:
        sig = (
            tuple(_as_int_list(region["data"])),
            tuple(_as_int_list(region["cycle_checks"])),
            tuple(_as_int_list(region["internal_check_indices"])),
        )
        out[sig] = region
    return out


def _active_signatures(recording: list, regions_by_index: dict[int, dict], iteration_index: int) -> set[tuple]:
    shot = recording[-1]
    iterations = shot.get("iterations", [])
    if not iterations:
        raise ValueError("GBP recording shot has no iterations")
    iteration = iterations[iteration_index]
    active = set(_as_int_list(iteration.get("active_regions", [])))
    out = set()
    for idx in active:
        region = regions_by_index.get(int(idx))
        if region is None:
            continue
        out.add(
            (
                tuple(_as_int_list(region["data"])),
                tuple(_as_int_list(region["cycle_checks"])),
                tuple(_as_int_list(region["internal_check_indices"])),
            )
        )
    return out


def _region_edges(H: np.ndarray, region: dict) -> set[tuple[int, int]]:
    data = set(_as_int_list(region["data"]))
    checks = set(_as_int_list(region["cycle_checks"]))
    checks.update(_as_int_list(region["internal_check_indices"]))
    out = set()
    for c in checks:
        if c < 0 or c >= H.shape[0]:
            continue
        for v in data:
            if 0 <= v < H.shape[1] and H[c, v]:
                out.add((int(c), int(v)))
    return out


def _active_edges(
    recording: list,
    regions_by_index: dict[int, dict],
    H: np.ndarray,
    iteration_index: int,
) -> set[tuple[int, int]]:
    shot = recording[-1]
    iterations = shot.get("iterations", [])
    if not iterations:
        raise ValueError("GBP recording shot has no iterations")
    iteration = iterations[iteration_index]
    active = set(_as_int_list(iteration.get("active_regions", [])))
    out = set()
    for idx in active:
        region = regions_by_index.get(int(idx))
        if region is not None:
            out.update(_region_edges(H, region))
    return out


def gbp_policy_delta(
    fail_recording: list,
    success_recording: list,
    num_vars: int,
    num_checks: int,
) -> tuple[list[dict], list[dict]]:
    """
    Compare two GBP recordings and score regions/nodes active only in success.

    Regions are matched by structural signature `(data, cycle_checks,
    internal_checks)`, not by numeric index, so policies may produce different
    region ordering.
    """
    fail_regions = _regions_by_signature(fail_recording)
    success_regions = _regions_by_signature(success_recording)
    fail_by_index = {
        int(region["index"]): region
        for region in fail_recording[-1]["gbp"]["regions"]
    }
    success_by_index = {
        int(region["index"]): region
        for region in success_recording[-1]["gbp"]["regions"]
    }

    n_iter = max(
        len(fail_recording[-1].get("iterations", [])),
        len(success_recording[-1].get("iterations", [])),
    )
    region_counts: dict[tuple, dict] = {}
    data_scores = np.zeros(num_vars, dtype=np.int64)
    check_scores = np.zeros(num_checks, dtype=np.int64)

    for it in range(n_iter):
        fail_i = min(it, len(fail_recording[-1].get("iterations", [])) - 1)
        success_i = min(it, len(success_recording[-1].get("iterations", [])) - 1)
        fail_active = _active_signatures(fail_recording, fail_by_index, fail_i)
        success_active = _active_signatures(success_recording, success_by_index, success_i)
        for sig in sorted(success_active - fail_active):
            region = success_regions[sig]
            row = region_counts.setdefault(
                sig,
                {
                    "success_region": int(region["index"]),
                    "fail_has_region": sig in fail_regions,
                    "count": 0,
                    "first_iteration": it,
                    "data": list(sig[0]),
                    "cycle_checks": list(sig[1]),
                    "internal_checks": list(sig[2]),
                },
            )
            row["count"] += 1
            for v in sig[0]:
                data_scores[v] += 1
            for c in set(sig[1]).union(sig[2]):
                check_scores[c] += 1

    node_rows = []
    for v, score in enumerate(data_scores):
        if score:
            node_rows.append({"kind": "data", "index": v, "score": int(score)})
    for c, score in enumerate(check_scores):
        if score:
            node_rows.append({"kind": "check", "index": c, "score": int(score)})
    node_rows.sort(key=lambda row: (-row["score"], row["kind"], row["index"]))
    region_rows = list(region_counts.values())
    region_rows.sort(key=lambda row: (-row["count"], row["first_iteration"], row["success_region"]))
    return region_rows, node_rows


def write_policy_delta_csvs(
    region_rows: list[dict],
    node_rows: list[dict],
    region_path: str | Path,
    node_path: str | Path,
) -> None:
    region_path = Path(region_path)
    node_path = Path(node_path)
    region_path.parent.mkdir(parents=True, exist_ok=True)
    with region_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "success_region",
            "fail_has_region",
            "count",
            "first_iteration",
            "data",
            "cycle_checks",
            "internal_checks",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in region_rows:
            writer.writerow(row)
    with node_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "index", "score"])
        writer.writeheader()
        for row in node_rows:
            writer.writerow(row)


def gbp_delta_edge_cover(
    decoder,
    syndrome,
    baseline_correction,
    comparison_correction,
    region_rows: list[dict],
    baseline_recording: list | None = None,
    comparison_recording: list | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Search the success-only region delta for a minimum Tanner-edge cover.

    This is an analytic proxy for "what extra edges/nodes did the successful
    policy get to use?"  The universe is:
      * checks unsatisfied by the baseline correction but satisfied by comparison
      * data bits where baseline and comparison corrections differ

    Candidate edges are Tanner edges active in the comparison recording and not
    active in the baseline recording.  If recordings are not supplied, it falls
    back to the older, looser proxy: edges contained in success-only active
    regions.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    baseline_correction = np.asarray(baseline_correction, dtype=np.uint8)
    comparison_correction = np.asarray(comparison_correction, dtype=np.uint8)
    baseline_residual = ((H @ baseline_correction.astype(np.int32)) % 2).astype(np.uint8) ^ syndrome
    comparison_residual = ((H @ comparison_correction.astype(np.int32)) % 2).astype(np.uint8) ^ syndrome

    target_checks = sorted(
        int(c)
        for c in np.flatnonzero((baseline_residual == 1) & (comparison_residual == 0))
    )
    target_data = sorted(
        int(v)
        for v in np.flatnonzero(baseline_correction != comparison_correction)
    )
    universe = [f"c{c}" for c in target_checks] + [f"v{v}" for v in target_data]
    universe_index = {item: i for i, item in enumerate(universe)}
    full_mask = (1 << len(universe)) - 1

    edge_to_regions: dict[tuple[int, int], set[int]] = {}
    edge_to_count: dict[tuple[int, int], int] = {}

    if baseline_recording is not None and comparison_recording is not None:
        baseline_by_index = {
            int(region["index"]): region
            for region in baseline_recording[-1]["gbp"]["regions"]
        }
        comparison_by_index = {
            int(region["index"]): region
            for region in comparison_recording[-1]["gbp"]["regions"]
        }
        n_iter = max(
            len(baseline_recording[-1].get("iterations", [])),
            len(comparison_recording[-1].get("iterations", [])),
        )
        for it in range(n_iter):
            baseline_i = min(it, len(baseline_recording[-1].get("iterations", [])) - 1)
            comparison_i = min(it, len(comparison_recording[-1].get("iterations", [])) - 1)
            baseline_edges = _active_edges(
                baseline_recording, baseline_by_index, H, baseline_i)
            comparison_edges = _active_edges(
                comparison_recording, comparison_by_index, H, comparison_i)
            for edge in comparison_edges - baseline_edges:
                edge_to_regions.setdefault(edge, set())
                edge_to_count[edge] = edge_to_count.get(edge, 0) + 1
    else:
        for row in region_rows:
            data = set(int(v) for v in row.get("data", []))
            checks = set(int(c) for c in row.get("cycle_checks", []))
            checks.update(int(c) for c in row.get("internal_checks", []))
            count = int(row.get("count", 1))
            region_idx = int(row.get("success_region", -1))
            for c in checks:
                if c < 0 or c >= H.shape[0]:
                    continue
                for v in data:
                    if v < 0 or v >= H.shape[1] or not H[c, v]:
                        continue
                    key = (c, v)
                    edge_to_regions.setdefault(key, set()).add(region_idx)
                    edge_to_count[key] = edge_to_count.get(key, 0) + count

    if comparison_recording is not None:
        comparison_by_index = {
            int(region["index"]): region
            for region in comparison_recording[-1]["gbp"]["regions"]
        }
        for region_idx, region in comparison_by_index.items():
            for edge in _region_edges(H, region):
                if edge in edge_to_regions:
                    edge_to_regions[edge].add(region_idx)

    candidates = []
    for (c, v), regions in edge_to_regions.items():
        cover_items = []
        if f"c{c}" in universe_index:
            cover_items.append(f"c{c}")
        if f"v{v}" in universe_index:
            cover_items.append(f"v{v}")
        if not cover_items:
            continue
        mask = 0
        for item in cover_items:
            mask |= 1 << universe_index[item]
        candidates.append(
            {
                "check": c,
                "data": v,
                "cover": cover_items,
                "mask": mask,
                "regions": sorted(regions),
                "delta_count": int(edge_to_count[(c, v)]),
            }
        )
    candidates.sort(key=lambda row: (-int(row["mask"]).bit_count(), -row["delta_count"], row["check"], row["data"]))

    selected = _minimum_set_cover(candidates, full_mask)
    selected_keys = {(row["check"], row["data"]) for row in selected}
    candidate_rows = []
    for row in candidates:
        candidate_rows.append(
            {
                "selected": (row["check"], row["data"]) in selected_keys,
                "check": row["check"],
                "data": row["data"],
                "cover": row["cover"],
                "regions": row["regions"],
                "delta_count": row["delta_count"],
            }
        )
    summary = {
        "target_checks": target_checks,
        "target_data": target_data,
        "universe": universe,
        "covered": sorted({item for row in selected for item in row["cover"]}),
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "complete": _cover_mask(selected) == full_mask,
        "candidate_mode": (
            "comparison_active_minus_baseline_active"
            if baseline_recording is not None and comparison_recording is not None
            else "success_only_region_edges"
        ),
    }
    selected_rows = [
        {
            "check": row["check"],
            "data": row["data"],
            "cover": row["cover"],
            "regions": row["regions"],
            "delta_count": row["delta_count"],
        }
        for row in selected
    ]
    return selected_rows, candidate_rows, summary


def gbp_delta_region_context_cover(
    decoder,
    syndrome,
    baseline_correction,
    comparison_correction,
    region_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """
    Minimum cover over success-only region contexts.

    This is useful when the failing policy already contains the same individual
    Tanner edges, but the succeeding policy activates different *groupings* of
    those edges.  The universe is the same as `gbp_delta_edge_cover`.
    """
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    baseline_correction = np.asarray(baseline_correction, dtype=np.uint8)
    comparison_correction = np.asarray(comparison_correction, dtype=np.uint8)
    baseline_residual = ((H @ baseline_correction.astype(np.int32)) % 2).astype(np.uint8) ^ syndrome
    comparison_residual = ((H @ comparison_correction.astype(np.int32)) % 2).astype(np.uint8) ^ syndrome

    target_checks = sorted(
        int(c)
        for c in np.flatnonzero((baseline_residual == 1) & (comparison_residual == 0))
    )
    target_data = sorted(
        int(v)
        for v in np.flatnonzero(baseline_correction != comparison_correction)
    )
    universe = [f"c{c}" for c in target_checks] + [f"v{v}" for v in target_data]
    universe_index = {item: i for i, item in enumerate(universe)}
    full_mask = (1 << len(universe)) - 1

    candidates = []
    for row in region_rows:
        data = set(int(v) for v in row.get("data", []))
        checks = set(int(c) for c in row.get("cycle_checks", []))
        checks.update(int(c) for c in row.get("internal_checks", []))
        cover_items = []
        for c in sorted(checks):
            item = f"c{c}"
            if item in universe_index:
                cover_items.append(item)
        for v in sorted(data):
            item = f"v{v}"
            if item in universe_index:
                cover_items.append(item)
        if not cover_items:
            continue
        mask = 0
        for item in cover_items:
            mask |= 1 << universe_index[item]
        candidates.append(
            {
                "success_region": int(row["success_region"]),
                "cover": cover_items,
                "mask": mask,
                "count": int(row.get("count", 1)),
                "first_iteration": int(row.get("first_iteration", 0)),
                "data": list(row.get("data", [])),
                "cycle_checks": list(row.get("cycle_checks", [])),
                "internal_checks": list(row.get("internal_checks", [])),
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["mask"]).bit_count(),
            -row["count"],
            row["first_iteration"],
            row["success_region"],
        )
    )

    selected = _minimum_set_cover(candidates, full_mask)
    selected_ids = {int(row["success_region"]) for row in selected}
    candidate_rows = []
    for row in candidates:
        out = {
            "selected": int(row["success_region"]) in selected_ids,
            "success_region": row["success_region"],
            "cover": row["cover"],
            "count": row["count"],
            "first_iteration": row["first_iteration"],
            "data": row["data"],
            "cycle_checks": row["cycle_checks"],
            "internal_checks": row["internal_checks"],
        }
        candidate_rows.append(out)
    selected_rows = [
        {
            "success_region": row["success_region"],
            "cover": row["cover"],
            "count": row["count"],
            "first_iteration": row["first_iteration"],
            "data": row["data"],
            "cycle_checks": row["cycle_checks"],
            "internal_checks": row["internal_checks"],
        }
        for row in selected
    ]
    summary = {
        "target_checks": target_checks,
        "target_data": target_data,
        "universe": universe,
        "covered": sorted({item for row in selected for item in row["cover"]}),
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "complete": _cover_mask(selected) == full_mask,
        "candidate_mode": "success_only_region_context",
    }
    return selected_rows, candidate_rows, summary


def gbp_nearest_baseline_region_matches(
    selected_contexts: list[dict],
    baseline_recording: list,
    *,
    top_k: int = 5,
) -> list[dict]:
    """
    Match selected comparison region contexts to the nearest baseline regions.

    "Nearest" is Jaccard overlap over the union of region data nodes and check
    nodes.  This distinguishes a genuinely new region grouping from a region
    that exists in the baseline policy but is inactive or messaged differently.
    """
    if not selected_contexts:
        return []
    shot = baseline_recording[-1]
    if "gbp" not in shot:
        raise ValueError("baseline recording has no GBP metadata; use log_type='gbp'")

    active_counts: dict[int, int] = {}
    first_active: dict[int, int] = {}
    for it, iteration in enumerate(shot.get("iterations", [])):
        for idx in _as_int_list(iteration.get("active_regions", [])):
            idx = int(idx)
            active_counts[idx] = active_counts.get(idx, 0) + 1
            first_active.setdefault(idx, it)

    baseline_regions = []
    for region in shot["gbp"]["regions"]:
        data = set(_as_int_list(region["data"]))
        checks = set(_as_int_list(region["cycle_checks"]))
        checks.update(_as_int_list(region["internal_check_indices"]))
        baseline_regions.append(
            {
                "region": region,
                "index": int(region["index"]),
                "data": data,
                "checks": checks,
            }
        )

    rows = []
    for selected in selected_contexts:
        success_data = set(int(v) for v in selected.get("data", []))
        success_checks = set(int(c) for c in selected.get("cycle_checks", []))
        success_checks.update(int(c) for c in selected.get("internal_checks", []))
        success_signature = (tuple(sorted(success_data)), tuple(sorted(success_checks)))

        scored = []
        for candidate in baseline_regions:
            data_intersection = len(success_data & candidate["data"])
            data_union = len(success_data | candidate["data"])
            check_intersection = len(success_checks & candidate["checks"])
            check_union = len(success_checks | candidate["checks"])
            total_intersection = data_intersection + check_intersection
            total_union = data_union + check_union
            jaccard = float(total_intersection / total_union) if total_union else 1.0
            exact_match = (
                success_signature
                == (tuple(sorted(candidate["data"])), tuple(sorted(candidate["checks"])))
            )
            scored.append(
                {
                    "success_region": int(selected["success_region"]),
                    "baseline_region": candidate["index"],
                    "jaccard": jaccard,
                    "data_intersection": data_intersection,
                    "data_union": data_union,
                    "check_intersection": check_intersection,
                    "check_union": check_union,
                    "total_intersection": total_intersection,
                    "total_union": total_union,
                    "exact_match": exact_match,
                    "baseline_active_count": int(active_counts.get(candidate["index"], 0)),
                    "baseline_first_active_iteration": first_active.get(candidate["index"], None),
                    "success_data": sorted(success_data),
                    "success_checks": sorted(success_checks),
                    "baseline_data": sorted(candidate["data"]),
                    "baseline_checks": sorted(candidate["checks"]),
                }
            )
        scored.sort(
            key=lambda row: (
                not row["exact_match"],
                -row["jaccard"],
                -row["baseline_active_count"],
                row["baseline_region"],
            )
        )
        for rank, row in enumerate(scored[:max(1, int(top_k))], start=1):
            out = dict(row)
            out["rank"] = rank
            rows.append(out)
    return rows


def _cover_mask(rows: list[dict]) -> int:
    mask = 0
    for row in rows:
        mask |= int(row.get("mask", 0))
    return mask


def _minimum_set_cover(candidates: list[dict], full_mask: int) -> list[dict]:
    if full_mask == 0:
        return []
    if not candidates:
        return []

    suffix = [0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix[i] = suffix[i + 1] | int(candidates[i]["mask"])

    best: list[dict] | None = None

    def search(i: int, covered: int, chosen: list[dict]) -> None:
        nonlocal best
        if covered == full_mask:
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if i >= len(candidates):
            return
        if best is not None and len(chosen) >= len(best):
            return
        if (covered | suffix[i]) != full_mask:
            return

        # Choose a still-uncovered target and branch only over candidates that cover it.
        missing = full_mask & ~covered
        target_bit = missing & -missing
        covering = [
            j for j in range(i, len(candidates))
            if int(candidates[j]["mask"]) & target_bit
        ]
        covering.sort(
            key=lambda j: (
                -int(candidates[j]["mask"] & ~covered).bit_count(),
                -_candidate_weight(candidates[j]),
            )
        )
        for j in covering:
            row = candidates[j]
            chosen.append(row)
            search(j + 1, covered | int(row["mask"]), chosen)
            chosen.pop()

    search(0, 0, [])
    return best or _greedy_set_cover(candidates, full_mask)


def _greedy_set_cover(candidates: list[dict], full_mask: int) -> list[dict]:
    selected = []
    covered = 0
    remaining = list(candidates)
    while covered != full_mask and remaining:
        best = max(
            remaining,
            key=lambda row: (
                int(int(row["mask"]) & ~covered).bit_count(),
                _candidate_weight(row),
            ),
        )
        if (int(best["mask"]) & ~covered) == 0:
            break
        selected.append(best)
        covered |= int(best["mask"])
        remaining.remove(best)
    return selected


def _candidate_weight(row: dict) -> int:
    return int(row.get("delta_count", row.get("count", 1)))


def write_delta_edge_cover_csv(
    selected_rows: list[dict],
    candidate_rows: list[dict],
    selected_path: str | Path,
    candidate_path: str | Path,
) -> None:
    selected_path = Path(selected_path)
    candidate_path = Path(candidate_path)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["selected", "check", "data", "cover", "regions", "delta_count"]
    with selected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            out = dict(row)
            out["selected"] = True
            writer.writerow(out)
    with candidate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(row)


def write_delta_region_context_cover_csv(
    selected_rows: list[dict],
    candidate_rows: list[dict],
    selected_path: str | Path,
    candidate_path: str | Path,
) -> None:
    selected_path = Path(selected_path)
    candidate_path = Path(candidate_path)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selected",
        "success_region",
        "cover",
        "count",
        "first_iteration",
        "data",
        "cycle_checks",
        "internal_checks",
    ]
    with selected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            out = dict(row)
            out["selected"] = True
            writer.writerow(out)
    with candidate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(row)


def write_nearest_baseline_region_matches_csv(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "success_region",
        "rank",
        "baseline_region",
        "jaccard",
        "data_intersection",
        "data_union",
        "check_intersection",
        "check_union",
        "total_intersection",
        "total_union",
        "exact_match",
        "baseline_active_count",
        "baseline_first_active_iteration",
        "success_data",
        "success_checks",
        "baseline_data",
        "baseline_checks",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_policy_delta_graph(
    decoder,
    syndrome,
    node_rows: list[dict],
    output_path: str | Path,
    layout: dict | None = None,
    show_labels: bool = True,
) -> None:
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    figsize = layout.get("figsize") or (
        8.0,
        min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0),
    )
    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]

    data_scores = np.zeros(num_vars, dtype=np.float64)
    check_scores = np.zeros(num_checks, dtype=np.float64)
    for row in node_rows:
        if row["kind"] == "data":
            data_scores[int(row["index"])] = float(row["score"])
        else:
            check_scores[int(row["index"])] = float(row["score"])
    vmax = max(float(data_scores.max(initial=0)), float(check_scores.max(initial=0)), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges_from_H(H):
        important = data_scores[v] > 0 or check_scores[d] > 0
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=ML_CORRECTION if important else FAINT_EDGE,
            linewidth=1.2 if important else 0.45,
            alpha=0.45 if important else 0.45,
            zorder=1,
        )

    var_colors = [
        cmap(norm(data_scores[v])) if data_scores[v] > 0 else "white"
        for v in range(num_vars)
    ]
    var_edges = [
        ML_CORRECTION if data_scores[v] > 0 else FAINT_NODE
        for v in range(num_vars)
    ]
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_colors,
        edgecolors=var_edges,
        linewidths=[1.8 if data_scores[v] > 0 else 0.6 for v in range(num_vars)],
        zorder=3,
    )

    check_colors = []
    check_edges = []
    for d in range(num_checks):
        if syndrome[d]:
            check_colors.append(ACTIVE_CHECK)
            check_edges.append("black")
        elif check_scores[d] > 0:
            check_colors.append(cmap(norm(check_scores[d])))
            check_edges.append(ML_CORRECTION)
        else:
            check_colors.append("white")
            check_edges.append(FAINT_NODE)
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_colors,
        edgecolors=check_edges,
        linewidths=[1.8 if check_scores[d] > 0 or syndrome[d] else 0.6 for d in range(num_checks)],
        marker="s",
        zorder=3,
    )

    if show_labels:
        label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
        for v in range(num_vars):
            ax.annotate(str(v), xy=var_pos[v], xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=label_fontsize, color="black", zorder=4)
        for d in range(num_checks):
            ax.annotate(str(d), xy=check_pos[d], xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=label_fontsize,
                        color="white" if syndrome[d] else "black", zorder=4)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("success-only active-region coverage count")
    ax.set_title("GBP policy delta: nodes covered only by the comparison decoder")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_delta_edge_cover_graph(
    decoder,
    syndrome,
    selected_rows: list[dict],
    summary: dict,
    output_path: str | Path,
    layout: dict | None = None,
    show_labels: bool = True,
) -> None:
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    figsize = layout.get("figsize") or (
        8.0,
        min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0),
    )
    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    selected_edges = {(int(row["check"]), int(row["data"])) for row in selected_rows}
    target_checks = set(int(c) for c in summary.get("target_checks", []))
    target_data = set(int(v) for v in summary.get("target_data", []))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges_from_H(H):
        chosen = (d, v) in selected_edges
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=BP_CORRECTION if chosen else FAINT_EDGE,
            linewidth=2.6 if chosen else 0.45,
            alpha=0.95 if chosen else 0.35,
            zorder=3 if chosen else 1,
        )

    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=[ML_CORRECTION if v in target_data else "white" for v in range(num_vars)],
        edgecolors=[ML_CORRECTION if v in target_data else FAINT_NODE for v in range(num_vars)],
        linewidths=[2.0 if v in target_data else 0.6 for v in range(num_vars)],
        zorder=4,
    )
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=[
            BP_CORRECTION if d in target_checks
            else (ACTIVE_CHECK if syndrome[d] else "white")
            for d in range(num_checks)
        ],
        edgecolors=["black" if syndrome[d] or d in target_checks else FAINT_NODE
                    for d in range(num_checks)],
        linewidths=[2.0 if d in target_checks else 1.0 for d in range(num_checks)],
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
                        color="white" if syndrome[d] or d in target_checks else "black",
                        zorder=5)

    ax.set_title(
        "GBP delta edge cover    "
        f"selected={summary.get('num_selected', 0)}  "
        f"complete={summary.get('complete', False)}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_delta_region_context_cover_graph(
    decoder,
    syndrome,
    selected_rows: list[dict],
    summary: dict,
    output_path: str | Path,
    layout: dict | None = None,
    show_labels: bool = True,
) -> None:
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    figsize = layout.get("figsize") or (
        8.0,
        min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0),
    )
    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]

    selected_edges = set()
    selected_data = set()
    selected_checks = set()
    for row in selected_rows:
        data = set(int(v) for v in row.get("data", []))
        checks = set(int(c) for c in row.get("cycle_checks", []))
        checks.update(int(c) for c in row.get("internal_checks", []))
        selected_data.update(data)
        selected_checks.update(checks)
        for c in checks:
            if c < 0 or c >= num_checks:
                continue
            for v in data:
                if 0 <= v < num_vars and H[c, v]:
                    selected_edges.add((c, v))

    target_checks = set(int(c) for c in summary.get("target_checks", []))
    target_data = set(int(v) for v in summary.get("target_data", []))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for d, v in edges_from_H(H):
        chosen = (d, v) in selected_edges
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=BP_CORRECTION if chosen else FAINT_EDGE,
            linewidth=2.0 if chosen else 0.45,
            alpha=0.75 if chosen else 0.35,
            zorder=3 if chosen else 1,
        )

    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=[
            ML_CORRECTION if v in target_data
            else ("#cfe8ff" if v in selected_data else "white")
            for v in range(num_vars)
        ],
        edgecolors=[
            ML_CORRECTION if v in target_data or v in selected_data else FAINT_NODE
            for v in range(num_vars)
        ],
        linewidths=[2.0 if v in target_data else (1.2 if v in selected_data else 0.6)
                    for v in range(num_vars)],
        zorder=4,
    )
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=[
            BP_CORRECTION if d in target_checks
            else ("#cfe8ff" if d in selected_checks else (ACTIVE_CHECK if syndrome[d] else "white"))
            for d in range(num_checks)
        ],
        edgecolors=["black" if syndrome[d] or d in target_checks or d in selected_checks else FAINT_NODE
                    for d in range(num_checks)],
        linewidths=[2.0 if d in target_checks else (1.2 if d in selected_checks else 0.6)
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
                        color="white" if syndrome[d] or d in target_checks else "black",
                        zorder=5)

    ax.set_title(
        "GBP delta region-context cover    "
        f"selected={summary.get('num_selected', 0)}  "
        f"complete={summary.get('complete', False)}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
