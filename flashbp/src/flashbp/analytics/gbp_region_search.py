from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from flashbp.animation.layout import bipartite_layout, edges_from_H
from .style import ACTIVE_CHECK, BP_CORRECTION, FAINT_EDGE, FAINT_NODE, ML_CORRECTION


def _as_int_list(value) -> list[int]:
    return [int(x) for x in np.asarray(value, dtype=np.int64).ravel()]


def _rank_binary_masks(masks: list[int]) -> int:
    rows = [int(m) for m in masks if int(m)]
    rank = 0
    bit = max((m.bit_length() for m in rows), default=0) - 1
    while bit >= 0:
        pivot = next((r for r in range(rank, len(rows)) if (rows[r] >> bit) & 1), -1)
        if pivot >= 0:
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            for r in range(len(rows)):
                if r != rank and ((rows[r] >> bit) & 1):
                    rows[r] ^= rows[rank]
            rank += 1
        bit -= 1
    return rank


def _region_complexity(H: np.ndarray, data: list[int], checks: list[int]) -> dict:
    data = sorted(set(int(v) for v in data))
    checks = sorted(set(int(c) for c in checks))
    axis = {v: i for i, v in enumerate(data)}
    masks = []
    for c in checks:
        support = [int(v) for v in np.flatnonzero(H[c])]
        if support and all(v in axis for v in support):
            mask = 0
            for v in support:
                mask |= 1 << axis[v]
            masks.append(mask)
    rank = _rank_binary_masks(masks)
    num_axes = len(data)
    dense = 1 << num_axes if num_axes < 63 else 2**63 - 1
    valid = 1 << max(0, num_axes - rank) if num_axes < 63 else 2**63 - 1
    return {
        "num_axes": num_axes,
        "num_checks": len(checks),
        "internal_check_rank": rank,
        "dense_state_count": dense,
        "valid_state_count": valid,
        "log2_valid_states": max(0, num_axes - rank),
    }


def _candidate_key(data: list[int], checks: list[int], activation: str) -> tuple:
    return (tuple(sorted(set(data))), tuple(sorted(set(checks))), activation)


def _add_candidate(
    candidates: dict[tuple, dict],
    H: np.ndarray,
    data: list[int],
    checks: list[int],
    *,
    source: str,
    activation: str = "always",
    priority: float = 0.0,
    source_region: int | None = None,
    metadata: dict | None = None,
) -> None:
    data = sorted(set(int(v) for v in data))
    checks = sorted(set(int(c) for c in checks))
    if not data or not checks:
        return
    key = _candidate_key(data, checks, activation)
    complexity = _region_complexity(H, data, checks)
    row = candidates.get(key)
    if row is None:
        row = {
            "candidate": len(candidates),
            "data": data,
            "checks": checks,
            "activation": activation,
            "sources": [],
            "priority": float(priority),
            "source_regions": [],
            **complexity,
        }
        candidates[key] = row
    row["priority"] = max(float(row["priority"]), float(priority))
    if source not in row["sources"]:
        row["sources"].append(source)
    if source_region is not None and int(source_region) not in row["source_regions"]:
        row["source_regions"].append(int(source_region))
    if metadata:
        for k, v in metadata.items():
            row.setdefault(k, v)


def _component_candidates_from_seed_sets(
    H: np.ndarray,
    seed_data: set[int],
    seed_checks: set[int],
) -> list[tuple[list[int], list[int]]]:
    seed_data = set(int(v) for v in seed_data)
    seed_checks = set(int(c) for c in seed_checks)
    seen_data: set[int] = set()
    seen_checks: set[int] = set()
    components = []

    for start in sorted(seed_data):
        if start in seen_data:
            continue
        data = {start}
        checks = set()
        queue = [("v", start)]
        seen_data.add(start)
        while queue:
            kind, idx = queue.pop()
            if kind == "v":
                for c in np.flatnonzero(H[:, idx]):
                    c = int(c)
                    if c not in seed_checks or c in seen_checks:
                        continue
                    seen_checks.add(c)
                    checks.add(c)
                    queue.append(("c", c))
            else:
                for v in np.flatnonzero(H[idx]):
                    v = int(v)
                    if v not in seed_data or v in seen_data:
                        continue
                    seen_data.add(v)
                    data.add(v)
                    queue.append(("v", v))
        if data and checks:
            components.append((sorted(data), sorted(checks)))

    for start in sorted(seed_checks):
        if start in seen_checks:
            continue
        checks = {start}
        data = set()
        queue = [("c", start)]
        seen_checks.add(start)
        while queue:
            kind, idx = queue.pop()
            if kind == "c":
                for v in np.flatnonzero(H[idx]):
                    v = int(v)
                    if v not in seed_data or v in seen_data:
                        continue
                    seen_data.add(v)
                    data.add(v)
                    queue.append(("v", v))
            else:
                for c in np.flatnonzero(H[:, idx]):
                    c = int(c)
                    if c not in seed_checks or c in seen_checks:
                        continue
                    seen_checks.add(c)
                    checks.add(c)
                    queue.append(("c", c))
        if data and checks:
            components.append((sorted(data), sorted(checks)))
    return components


def _add_detection_candidates(candidates: dict[tuple, dict], H: np.ndarray, syndrome) -> None:
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    active_checks = [int(c) for c in np.flatnonzero(syndrome)]
    active_data = set()
    for c in active_checks:
        data = [int(v) for v in np.flatnonzero(H[c])]
        active_data.update(data)
        _add_candidate(
            candidates,
            H,
            data,
            [c],
            source="detection_active_check_neighborhood",
            priority=750.0,
            source_region=c,
        )

    for data, checks in _component_candidates_from_seed_sets(
        H, active_data, set(active_checks)
    ):
        _add_candidate(
            candidates,
            H,
            data,
            checks,
            source="detection_active_component",
            priority=800.0 + len(checks),
        )


def _add_truth_candidates(candidates: dict[tuple, dict], H: np.ndarray, true_errors) -> None:
    if true_errors is None:
        return
    true_errors = np.asarray(true_errors, dtype=np.uint8)
    true_data = {int(v) for v in np.flatnonzero(true_errors)}
    if not true_data:
        return
    adjacent_checks = {
        int(c)
        for v in true_data
        for c in np.flatnonzero(H[:, v])
    }
    _add_candidate(
        candidates,
        H,
        sorted(true_data),
        sorted(adjacent_checks),
        source="truth_error_support",
        priority=1200.0,
        metadata={"oracle": True},
    )
    for data, checks in _component_candidates_from_seed_sets(
        H, true_data, adjacent_checks
    ):
        _add_candidate(
            candidates,
            H,
            data,
            checks,
            source="truth_error_component",
            priority=1100.0 + len(data),
            metadata={"oracle": True},
        )


def select_gbp_region_candidates(
    H,
    *,
    syndrome=None,
    true_errors=None,
    candidate_source: str = "delta",
    selected_contexts: list[dict] | None = None,
    region_rows: list[dict] | None = None,
    nearest_baseline_rows: list[dict] | None = None,
    max_candidates: int = 48,
    max_log2_valid_states: int | None = None,
    include_success_contexts: bool = True,
    include_inactive_twins: bool = True,
    include_substitutes: bool = True,
    include_top_delta_regions: bool = True,
) -> list[dict]:
    """
    Build a compact candidate list for manual GBP grouping experiments.

    Candidates come from the diagnostic artifacts produced by the GBP policy
    delta analysis, active detections, oracle true errors, or a union of these.
    """
    H = np.asarray(H, dtype=np.uint8)
    if candidate_source not in ("delta", "detections", "truth", "all"):
        raise ValueError(
            "candidate_source must be one of: delta, detections, truth, all"
        )
    candidates: dict[tuple, dict] = {}
    use_delta = candidate_source in ("delta", "all")
    use_detections = candidate_source in ("detections", "all")
    use_truth = candidate_source in ("truth", "all")

    if use_detections and syndrome is not None:
        _add_detection_candidates(candidates, H, syndrome)

    if use_truth and true_errors is not None:
        _add_truth_candidates(candidates, H, true_errors)

    if use_delta and include_success_contexts:
        for row in selected_contexts or []:
            checks = sorted(set(_as_int_list(row.get("cycle_checks", []))).union(
                _as_int_list(row.get("internal_checks", []))
            ))
            _add_candidate(
                candidates,
                H,
                _as_int_list(row.get("data", [])),
                checks,
                source="selected_success_context",
                priority=1000.0 + float(row.get("count", 0)),
                source_region=int(row.get("success_region", -1)),
                metadata={"cover": row.get("cover", [])},
            )

    if use_delta and include_top_delta_regions:
        for row in region_rows or []:
            checks = sorted(set(_as_int_list(row.get("cycle_checks", []))).union(
                _as_int_list(row.get("internal_checks", []))
            ))
            _add_candidate(
                candidates,
                H,
                _as_int_list(row.get("data", [])),
                checks,
                source="top_success_only_region",
                priority=500.0 + float(row.get("count", 0)),
                source_region=int(row.get("success_region", -1)),
            )

    for row in ((nearest_baseline_rows or []) if use_delta else []):
        active_count = int(row.get("baseline_active_count", 0))
        exact = bool(row.get("exact_match", False))
        jaccard = float(row.get("jaccard", 0.0))
        if include_inactive_twins and exact and active_count == 0:
            _add_candidate(
                candidates,
                H,
                _as_int_list(row.get("baseline_data", [])),
                _as_int_list(row.get("baseline_checks", [])),
                source="exact_inactive_baseline_twin",
                priority=900.0 + jaccard,
                source_region=int(row.get("baseline_region", -1)),
                metadata={"nearest_jaccard": jaccard},
            )
        if include_substitutes and active_count > 0 and not exact:
            _add_candidate(
                candidates,
                H,
                _as_int_list(row.get("baseline_data", [])),
                _as_int_list(row.get("baseline_checks", [])),
                source="active_baseline_substitute",
                priority=100.0 + active_count + jaccard,
                source_region=int(row.get("baseline_region", -1)),
                metadata={"nearest_jaccard": jaccard},
            )

    rows = list(candidates.values())
    if max_log2_valid_states is not None:
        rows = [
            row for row in rows
            if int(row["log2_valid_states"]) <= int(max_log2_valid_states)
        ]
    rows.sort(
        key=lambda row: (
            -float(row["priority"]),
            int(row["log2_valid_states"]),
            int(row["num_axes"]),
            row["candidate"],
        )
    )
    rows = rows[:max_candidates]
    for i, row in enumerate(rows):
        row["candidate"] = i
    return rows


def _success(outcome: dict, require_correct: bool) -> bool:
    if require_correct and "correct" in outcome:
        return bool(outcome.get("converged", False)) and bool(outcome.get("correct", False))
    return bool(outcome.get("converged", False))


def _outcome_score(outcome: dict, require_correct: bool) -> tuple:
    return (
        int(_success(outcome, require_correct)),
        int(bool(outcome.get("correct", False))),
        int(bool(outcome.get("converged", False))),
        -int(outcome.get("residual_weight", 10**9)),
        -int(outcome.get("iterations", 10**9)),
    )


def _total_complexity(groups: list[dict]) -> int:
    return int(sum(int(row.get("valid_state_count", 1)) for row in groups))


def search_minimal_gbp_groups(
    candidates: list[dict],
    evaluate: Callable[[list[dict]], dict],
    *,
    max_selected: int = 6,
    require_correct: bool = True,
) -> dict:
    """
    Greedy-add/prune search for a small sufficient manual region set.

    `evaluate(groups)` should run the manual GBP decoder and return at least
    `converged`, `residual_weight`, and optionally `correct`/`iterations`.
    """
    individual = []
    for cand in candidates:
        outcome = evaluate([cand])
        individual.append({
            "candidate": int(cand["candidate"]),
            "outcome": outcome,
            "succeeds": _success(outcome, require_correct),
        })

    selected: list[dict] = []
    remaining = list(candidates)
    history = []
    best_seen = None
    best_seen_score = None
    final_outcome = evaluate([])
    if _success(final_outcome, require_correct):
        success_nonoptimal = [row for row in individual if row["succeeds"]]
        failed = [row for row in individual if not row["succeeds"]]
        success_nonoptimal.sort(key=lambda row: (
            int(candidates[row["candidate"]].get("valid_state_count", 1)),
            row["candidate"],
        ))
        failed.sort(key=lambda row: (
            _outcome_score(row["outcome"], require_correct),
            -int(candidates[row["candidate"]].get("valid_state_count", 1)),
        ), reverse=True)
        return {
            "selected": [],
            "selected_ids": [],
            "final_outcome": final_outcome,
            "succeeded": True,
            "history": [{
                "step": 0,
                "action": "empty_succeeds",
                "candidate": -1,
                "outcome": final_outcome,
                "total_valid_state_count": 0,
            }],
            "individual": individual,
            "success_nonoptimal": success_nonoptimal,
            "failed": failed,
            "best_seen_candidate_ids": [],
            "best_seen_score": list(_outcome_score(final_outcome, require_correct)),
        }
    for step in range(max_selected):
        trials = []
        for cand in remaining:
            groups = selected + [cand]
            outcome = evaluate(groups)
            score = _outcome_score(outcome, require_correct)
            total_complexity = _total_complexity(groups)
            trials.append((cand, outcome, score, total_complexity))
            if best_seen is None or score > best_seen_score:
                best_seen = groups
                best_seen_score = score
        if not trials:
            break

        successful = [t for t in trials if _success(t[1], require_correct)]
        if successful:
            chosen = min(
                successful,
                key=lambda t: (t[3], len(selected) + 1, int(t[1].get("iterations", 10**9))),
            )
        else:
            chosen = max(
                trials,
                key=lambda t: (
                    t[2],
                    -int(t[0].get("valid_state_count", 1)),
                    float(t[0].get("priority", 0.0)),
                ),
            )

        cand, outcome, score, total_complexity = chosen
        selected.append(cand)
        remaining = [row for row in remaining if row["candidate"] != cand["candidate"]]
        final_outcome = outcome
        history.append({
            "step": step,
            "action": "add",
            "candidate": int(cand["candidate"]),
            "outcome": outcome,
            "total_valid_state_count": total_complexity,
        })
        if _success(outcome, require_correct):
            break

    if _success(final_outcome, require_correct):
        changed = True
        while changed and len(selected) > 1:
            changed = False
            for cand in list(selected):
                trial = [row for row in selected if row["candidate"] != cand["candidate"]]
                outcome = evaluate(trial)
                if _success(outcome, require_correct):
                    selected = trial
                    final_outcome = outcome
                    history.append({
                        "step": len(history),
                        "action": "prune",
                        "candidate": int(cand["candidate"]),
                        "outcome": outcome,
                        "total_valid_state_count": _total_complexity(trial),
                    })
                    changed = True
                    break

    selected_ids = {int(row["candidate"]) for row in selected}
    individual_by_id = {int(row["candidate"]): row for row in individual}
    success_nonoptimal = [
        row for row in individual
        if row["succeeds"] and int(row["candidate"]) not in selected_ids
    ]
    failed = [
        row for row in individual
        if not row["succeeds"] and int(row["candidate"]) not in selected_ids
    ]
    success_nonoptimal.sort(key=lambda row: (
        int(candidates[row["candidate"]].get("valid_state_count", 1)),
        row["candidate"],
    ))
    failed.sort(key=lambda row: (
        _outcome_score(row["outcome"], require_correct),
        -int(candidates[row["candidate"]].get("valid_state_count", 1)),
    ), reverse=True)

    return {
        "selected": selected,
        "selected_ids": sorted(selected_ids),
        "final_outcome": final_outcome,
        "succeeded": _success(final_outcome, require_correct),
        "history": history,
        "individual": individual,
        "success_nonoptimal": success_nonoptimal,
        "failed": failed,
        "best_seen_candidate_ids": (
            [int(row["candidate"]) for row in best_seen] if best_seen else []
        ),
        "best_seen_score": list(best_seen_score) if best_seen_score else None,
    }


def manual_groups_from_candidates(candidates: list[dict]) -> list[dict]:
    return [
        {
            "data": _as_int_list(row["data"]),
            "checks": _as_int_list(row["checks"]),
            "activation": row.get("activation", "always"),
        }
        for row in candidates
    ]


def write_region_search_csvs(
    candidates: list[dict],
    search: dict,
    candidate_path: str | Path,
    history_path: str | Path,
) -> None:
    candidate_path = Path(candidate_path)
    history_path = Path(history_path)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    selected_ids = set(int(i) for i in search.get("selected_ids", []))
    individual = {
        int(row["candidate"]): row
        for row in search.get("individual", [])
    }
    fieldnames = [
        "candidate",
        "selected",
        "individual_succeeds",
        "individual_converged",
        "individual_correct",
        "individual_residual_weight",
        "individual_iterations",
        "priority",
        "sources",
        "source_regions",
        "activation",
        "num_axes",
        "num_checks",
        "internal_check_rank",
        "dense_state_count",
        "valid_state_count",
        "log2_valid_states",
        "data",
        "checks",
    ]
    with candidate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            outcome = individual.get(int(row["candidate"]), {}).get("outcome", {})
            writer.writerow({
                "candidate": row["candidate"],
                "selected": int(row["candidate"]) in selected_ids,
                "individual_succeeds": individual.get(int(row["candidate"]), {}).get("succeeds", False),
                "individual_converged": outcome.get("converged", None),
                "individual_correct": outcome.get("correct", None),
                "individual_residual_weight": outcome.get("residual_weight", None),
                "individual_iterations": outcome.get("iterations", None),
                "priority": row["priority"],
                "sources": row["sources"],
                "source_regions": row["source_regions"],
                "activation": row["activation"],
                "num_axes": row["num_axes"],
                "num_checks": row["num_checks"],
                "internal_check_rank": row["internal_check_rank"],
                "dense_state_count": row["dense_state_count"],
                "valid_state_count": row["valid_state_count"],
                "log2_valid_states": row["log2_valid_states"],
                "data": row["data"],
                "checks": row["checks"],
            })

    with history_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "action",
            "candidate",
            "converged",
            "correct",
            "residual_weight",
            "iterations",
            "total_valid_state_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in search.get("history", []):
            outcome = row.get("outcome", {})
            writer.writerow({
                "step": row["step"],
                "action": row["action"],
                "candidate": row["candidate"],
                "converged": outcome.get("converged", None),
                "correct": outcome.get("correct", None),
                "residual_weight": outcome.get("residual_weight", None),
                "iterations": outcome.get("iterations", None),
                "total_valid_state_count": row.get("total_valid_state_count", None),
            })


def plot_region_candidate_samples(
    decoder,
    syndrome,
    candidates: list[dict],
    search: dict,
    output_path: str | Path,
    *,
    layout: dict | None = None,
    show_labels: bool = True,
    max_per_category: int = 3,
) -> None:
    H = np.asarray(decoder.H, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    base_figsize = layout.get("figsize") or (
        8.0,
        min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0),
    )
    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    candidate_by_id = {int(row["candidate"]): row for row in candidates}

    sample_rows = []
    for cand in search.get("selected", [])[:max_per_category]:
        sample_rows.append(("optimal selected", cand))
    for row in search.get("success_nonoptimal", [])[:max_per_category]:
        sample_rows.append(("succeeds not optimal", candidate_by_id[int(row["candidate"])]))
    for row in search.get("failed", [])[:max_per_category]:
        sample_rows.append(("does not succeed", candidate_by_id[int(row["candidate"])]))
    if not sample_rows and candidates:
        sample_rows.append(("candidate", candidates[0]))
    if not sample_rows:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "no candidate samples available",
            ha="center",
            va="center",
            fontsize=12,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return

    n = len(sample_rows)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(base_figsize[0], max(3.0, base_figsize[1] * 0.75) * n),
        squeeze=False,
    )
    label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))

    for ax, (category, cand) in zip(axes.ravel(), sample_rows):
        data = set(_as_int_list(cand.get("data", [])))
        checks = set(_as_int_list(cand.get("checks", [])))
        selected_edges = {
            (d, v) for d, v in edges_from_H(H)
            if d in checks and v in data
        }
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for d, v in edges_from_H(H):
            chosen = (d, v) in selected_edges
            ax.plot(
                [var_pos[v][0], check_pos[d][0]],
                [var_pos[v][1], check_pos[d][1]],
                color=BP_CORRECTION if chosen else FAINT_EDGE,
                linewidth=2.2 if chosen else 0.45,
                alpha=0.9 if chosen else 0.35,
                zorder=2 if chosen else 1,
            )
        ax.scatter(
            [var_pos[v][0] for v in range(num_vars)],
            [var_pos[v][1] for v in range(num_vars)],
            s=base_size,
            c=[ML_CORRECTION if v in data else "white" for v in range(num_vars)],
            edgecolors=[ML_CORRECTION if v in data else FAINT_NODE for v in range(num_vars)],
            linewidths=[1.8 if v in data else 0.6 for v in range(num_vars)],
            zorder=3,
        )
        ax.scatter(
            [check_pos[d][0] for d in range(num_checks)],
            [check_pos[d][1] for d in range(num_checks)],
            s=base_size,
            c=[
                BP_CORRECTION if d in checks
                else (ACTIVE_CHECK if syndrome[d] else "white")
                for d in range(num_checks)
            ],
            edgecolors=["black" if syndrome[d] or d in checks else FAINT_NODE for d in range(num_checks)],
            linewidths=[1.8 if d in checks else 0.8 for d in range(num_checks)],
            marker="s",
            zorder=3,
        )
        if show_labels:
            for v in range(num_vars):
                ax.annotate(str(v), xy=var_pos[v], xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=label_fontsize, color="black", zorder=4)
            for d in range(num_checks):
                ax.annotate(str(d), xy=check_pos[d], xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=label_fontsize,
                            color="white" if syndrome[d] or d in checks else "black",
                            zorder=4)
        ax.set_title(
            f"{category}: candidate {cand['candidate']}  "
            f"K={cand['num_axes']}  valid=2^{cand['log2_valid_states']}  "
            f"sources={','.join(cand.get('sources', []))}",
            fontsize=11,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
