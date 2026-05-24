"""
Search for a low-complexity set of manual GBP regions that repairs a failure.

The script first compares a baseline GBP policy against a comparison GBP policy
on one syndrome, then forms a compact pool of candidate manual regions from the
policy delta diagnostics.  It searches that pool with a greedy-add/prune loop:
add the best region until manual GBP succeeds, then remove unnecessary regions.

Outputs:
    candidates.csv              candidate groups and individual outcomes
    search_history.csv          greedy/prune actions
    selected_groups.json        manual groups for the final selected set
    candidate_samples.png       examples of selected, successful nonoptimal,
                                and unsuccessful candidates
    summary.json                decode outcomes and search summary
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import (
    gbp_delta_region_context_cover,
    gbp_nearest_baseline_region_matches,
    gbp_policy_delta,
    manual_groups_from_candidates,
    plot_region_candidate_samples,
    search_minimal_gbp_groups,
    select_gbp_region_candidates,
    write_region_search_csvs,
)

from _common import (
    CODES,
    layout_for_code,
    load_cached_shot,
    p_token,
    parse_decoder_spec,
    prepare_output_dir,
    resolve_code_and_p,
    sample_or_cached_shot,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--code", choices=CODES.keys(), default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--syndrome", type=str, default=None)
    parser.add_argument("--baseline-decoder", type=str, default="gbp-cycles-all:8")
    parser.add_argument("--comparison-decoder", type=str, default="gbp-cycles-any:8")
    parser.add_argument("--manual-backend", type=str, default=None,
                        help="override GBP backend for manual-group trials")
    parser.add_argument("--candidate-source", choices=("delta", "detections", "truth", "all"),
                        default="all",
                        help="where candidate manual regions come from")
    parser.add_argument("--no-manual-add-single-checks", action="store_true",
                        help="start manual GBP from no fallback single-check regions")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--max-selected", type=int, default=6)
    parser.add_argument("--max-log2-valid-states", type=int, default=None)
    parser.add_argument("--nearest-baseline-k", type=int, default=5)
    parser.add_argument("--allow-logical-failure", action="store_true",
                        help="treat convergence as success even if true_obs is known and wrong")
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    return parser.parse_args()


def make_gbp(dem, spec, output_dir: Path, label: str, *, log: bool = True):
    if spec.config is None or spec.config.decoder != "gbp":
        raise ValueError(f"{label} must be a GBP decoder spec")
    cfg = copy.deepcopy(spec.config)
    cfg.log = bool(log)
    cfg.log_type = "gbp" if log else "simple"
    cfg.log_console = False
    cfg.log_file = str(output_dir / f"{label}.log.txt") if log else None
    cfg.log_level = 3
    cfg.record_dir = str(output_dir)
    return flashbp.FlashBP(dem, cfg)


def decode_once(bp, syndrome, true_obs, max_iter: int) -> dict:
    correction = bp.decode(syndrome, max_iter)
    pred_obs = (bp.L @ correction.astype(np.int32)) % 2
    residual = ((bp.H @ correction.astype(np.int32)) % 2).astype(np.uint8) ^ syndrome.astype(np.uint8)
    stats = bp.last_decode_stats() if hasattr(bp, "last_decode_stats") else {}
    return {
        "correction": correction,
        "pred_obs": pred_obs,
        "correct": bool(true_obs is None or np.array_equal(pred_obs, true_obs)),
        "converged": bool(stats.get("converged", int(residual.sum()) == 0)),
        "iterations": int(stats.get("iterations", max_iter)),
        "residual_weight": int(residual.sum()),
        "recording": bp.get_recording() if hasattr(bp, "get_recording") else None,
    }


def make_manual_evaluator(
    dem,
    syndrome,
    true_obs,
    max_iter,
    template_cfg,
    backend_override,
    add_single_checks: bool,
):
    cache: dict[tuple, dict] = {}

    def evaluate(candidate_groups: list[dict]) -> dict:
        key = tuple(int(row["candidate"]) for row in candidate_groups)
        if key in cache:
            return dict(cache[key])
        cfg = copy.deepcopy(template_cfg)
        cfg.decoder = "gbp"
        cfg.region_policy = "manual_groups"
        cfg.gbp_manual_groups = manual_groups_from_candidates(candidate_groups)
        cfg.gbp_manual_add_single_checks = bool(add_single_checks)
        if backend_override:
            cfg.gbp_backend = backend_override
        cfg.log = False
        cfg.log_console = False
        bp = flashbp.FlashBP(dem, cfg)
        outcome = decode_once(bp, syndrome, true_obs, max_iter)
        compact = {
            "converged": outcome["converged"],
            "correct": outcome["correct"],
            "residual_weight": outcome["residual_weight"],
            "iterations": outcome["iterations"],
        }
        cache[key] = compact
        return dict(compact)

    return evaluate


def main():
    args = parse_args()
    baseline_spec = parse_decoder_spec(args.baseline_decoder)
    comparison_spec = parse_decoder_spec(args.comparison_decoder)
    if baseline_spec.config is None or comparison_spec.config is None:
        raise ValueError("baseline and comparison decoders must be FlashBP GBP specs")

    baseline_token = baseline_spec.label.replace(":", "_")
    comparison_token = comparison_spec.label.replace(":", "_")

    cache_path = Path(args.cache) if args.cache else None
    shot = {}
    metadata = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)

    output_dir = Path(
        args.output_dir
        or (
            f"results/gbp_region_search/{code_name}_{p_token(p)}."
            f"{cache_path.stem}_{args.shot_index}_{baseline_token}_vs_{comparison_token}"
            if cache_path is not None
            else f"results/gbp_region_search/{code_name}_{p_token(p)}."
                 f"{baseline_token}_vs_{comparison_token}"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    syndrome, true_obs, true_errors = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if args.candidate_source in ("truth", "all") and true_errors is None:
        if args.candidate_source == "truth":
            raise ValueError(
                "--candidate-source truth requires cached/sampled true errors; "
                "use a cache with true_errors or omit --syndrome."
            )
        print("WARNING: true errors unavailable; truth-seeded candidates disabled.")
    if true_obs is None:
        probe = make_gbp(dem, comparison_spec, output_dir, "probe", log=False)
        true_obs = np.zeros(probe.num_observables, dtype=np.uint8)

    baseline_bp = make_gbp(dem, baseline_spec, output_dir, "baseline", log=True)
    comparison_bp = make_gbp(dem, comparison_spec, output_dir, "comparison", log=True)
    baseline = decode_once(baseline_bp, syndrome, true_obs, args.max_iter)
    comparison = decode_once(comparison_bp, syndrome, true_obs, args.max_iter)

    region_rows, node_rows = gbp_policy_delta(
        baseline["recording"],
        comparison["recording"],
        comparison_bp.num_errors,
        comparison_bp.num_detectors,
    )
    selected_contexts, context_candidates, context_summary = (
        gbp_delta_region_context_cover(
            comparison_bp,
            syndrome,
            baseline["correction"],
            comparison["correction"],
            region_rows,
        )
    )
    nearest_rows = gbp_nearest_baseline_region_matches(
        selected_contexts,
        baseline["recording"],
        top_k=args.nearest_baseline_k,
    )

    candidates = select_gbp_region_candidates(
        comparison_bp.H,
        syndrome=syndrome,
        true_errors=true_errors,
        candidate_source=args.candidate_source,
        selected_contexts=selected_contexts,
        region_rows=region_rows,
        nearest_baseline_rows=nearest_rows,
        max_candidates=args.max_candidates,
        max_log2_valid_states=args.max_log2_valid_states,
    )

    template_cfg = copy.deepcopy(comparison_spec.config)
    evaluator = make_manual_evaluator(
        dem,
        syndrome,
        true_obs,
        args.max_iter,
        template_cfg,
        args.manual_backend,
        not args.no_manual_add_single_checks,
    )
    search = search_minimal_gbp_groups(
        candidates,
        evaluator,
        max_selected=args.max_selected,
        require_correct=not args.allow_logical_failure,
    )

    candidates_path = output_dir / "candidates.csv"
    history_path = output_dir / "search_history.csv"
    selected_groups_path = output_dir / "selected_groups.json"
    samples_path = output_dir / "candidate_samples.png"
    summary_path = output_dir / "summary.json"

    write_region_search_csvs(candidates, search, candidates_path, history_path)
    selected_groups_path.write_text(
        json.dumps(manual_groups_from_candidates(search["selected"]), indent=2),
        encoding="utf-8",
    )
    plot_region_candidate_samples(
        comparison_bp,
        syndrome,
        candidates,
        search,
        samples_path,
        layout=layout_for_code(code),
        show_labels=not args.no_labels,
    )

    summary = {
        "code": code_name,
        "p": float(p),
        "syndrome_weight": int(syndrome.sum()),
        "baseline_decoder": baseline_spec.label,
        "comparison_decoder": comparison_spec.label,
        "baseline": {
            "converged": baseline["converged"],
            "correct": baseline["correct"],
            "residual_weight": baseline["residual_weight"],
            "iterations": baseline["iterations"],
            "pred_obs": baseline["pred_obs"].astype(int).tolist(),
        },
        "comparison": {
            "converged": comparison["converged"],
            "correct": comparison["correct"],
            "residual_weight": comparison["residual_weight"],
            "iterations": comparison["iterations"],
            "pred_obs": comparison["pred_obs"].astype(int).tolist(),
        },
        "context_cover": context_summary,
        "num_candidates": len(candidates),
        "candidate_source": args.candidate_source,
        "manual_add_single_checks": not args.no_manual_add_single_checks,
        "search": {
            "succeeded": search["succeeded"],
            "selected_ids": search["selected_ids"],
            "final_outcome": search["final_outcome"],
            "best_seen_candidate_ids": search["best_seen_candidate_ids"],
            "best_seen_score": search["best_seen_score"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Code        : {code}")
    print(f"Noise p     : {float(p):.4g}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    print(
        "Baseline    : "
        f"{baseline_spec.label} converged={baseline['converged']} "
        f"correct={baseline['correct']} residual={baseline['residual_weight']} "
        f"iters={baseline['iterations']}"
    )
    print(
        "Comparison  : "
        f"{comparison_spec.label} converged={comparison['converged']} "
        f"correct={comparison['correct']} residual={comparison['residual_weight']} "
        f"iters={comparison['iterations']}"
    )
    print(
        "Candidates  : "
        f"{len(candidates)}  source={args.candidate_source}  "
        f"manual_single_checks={not args.no_manual_add_single_checks}  "
        f"context_targets={context_summary.get('universe', [])}"
    )
    print(
        "Search      : "
        f"succeeded={search['succeeded']} selected={search['selected_ids']} "
        f"final={search['final_outcome']}"
    )
    for cand in search["selected"]:
        print(
            f"  c{cand['candidate']} K={cand['num_axes']} "
            f"valid=2^{cand['log2_valid_states']} sources={cand['sources']} "
            f"data={cand['data']} checks={cand['checks']}"
        )
    print(f"Candidates  : {candidates_path}")
    print(f"History     : {history_path}")
    print(f"Groups JSON : {selected_groups_path}")
    print(f"Samples     : {samples_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
