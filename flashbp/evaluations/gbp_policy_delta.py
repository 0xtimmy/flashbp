"""
Compare two GBP policies on the same syndrome and show success-only regions.

This is meant for cases such as `gbp-cycles-all` failing while `gbp-cycles-any`
converges.  It scores the data/check nodes covered by regions that are active
in the comparison decoder but not in the baseline decoder.

Outputs:
    region_delta.csv   success-only active regions, counted across iterations
    node_scores.csv    data/check coverage score from those regions
    edge_cover.csv     minimum selected Tanner edges covering the analytic target
    edge_candidates.csv
                       all candidate Tanner edges from the delta regions
    region_context_cover.csv
                       minimum success-only region groupings covering the target
    region_context_candidates.csv
                       all success-only region grouping candidates
    nearest_baseline_regions.csv
                       closest baseline regions for each selected comparison context
    policy_delta.png   Tanner graph shaded by success-only coverage
    edge_cover.png     selected edge-cover hypothesis
    region_context_cover.png
                       selected region-context cover hypothesis
    summary.json       decode outcomes and top rows

Examples:
    python evaluations/gbp_policy_delta.py --code surface_5 --p 0.02 \\
        --baseline-decoder gbp-cycles-all:8 --comparison-decoder gbp-cycles-any:8

    python evaluations/gbp_policy_delta.py --cache results/errors/surface_5_0p02.50.npz \\
        --shot-index 3 --baseline-decoder gbp-union-cycles-all:8 \\
        --comparison-decoder gbp-union-cycles-any:8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import flashbp
from flashbp.analytics import (
    gbp_delta_edge_cover,
    gbp_delta_region_context_cover,
    gbp_nearest_baseline_region_matches,
    gbp_policy_delta,
    plot_delta_edge_cover_graph,
    plot_delta_region_context_cover_graph,
    plot_policy_delta_graph,
    write_delta_edge_cover_csv,
    write_delta_region_context_cover_csv,
    write_nearest_baseline_region_matches_csv,
    write_policy_delta_csvs,
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
    parser.add_argument("--baseline-decoder", type=str, default="gbp-cycles-all:8",
                        help="usually the failing/more restrictive GBP policy")
    parser.add_argument("--comparison-decoder", type=str, default="gbp-cycles-any:8",
                        help="usually the successful/more permissive GBP policy")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--nearest-baseline-k", type=int, default=5,
                        help="number of nearest baseline regions to report for each selected context")
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite output_dir without prompting")
    parser.add_argument("--no-labels", action="store_true")
    return parser.parse_args()


def make_gbp(dem, spec, output_dir: Path, label: str):
    if spec.config is None or spec.config.decoder != "gbp":
        raise ValueError(f"{label} must be a GBP decoder spec")
    cfg = spec.config
    cfg.log = True
    cfg.log_type = "gbp"
    cfg.log_console = False
    cfg.log_file = str(output_dir / f"{label}.log.txt")
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
        "correct": bool(np.array_equal(pred_obs, true_obs)),
        "converged": bool(stats.get("converged", int(residual.sum()) == 0)),
        "iterations": int(stats.get("iterations", max_iter)),
        "residual_weight": int(residual.sum()),
        "recording": bp.get_recording(),
    }


def main():
    args = parse_args()
    baseline_spec = parse_decoder_spec(args.baseline_decoder)
    comparison_spec = parse_decoder_spec(args.comparison_decoder)
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
            f"results/gbp_policy_delta/{code_name}_{p_token(p)}."
            f"{cache_path.stem}_{args.shot_index}_{baseline_token}_vs_{comparison_token}"
            if cache_path is not None
            else f"results/gbp_policy_delta/{code_name}_{p_token(p)}."
                 f"{baseline_token}_vs_{comparison_token}"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    syndrome, true_obs, _ = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_obs is None:
        tmp_spec = comparison_spec if comparison_spec.config is not None else baseline_spec
        tmp_bp = make_gbp(dem, tmp_spec, output_dir, "tmp")
        true_obs = np.zeros(tmp_bp.num_observables, dtype=np.uint8)

    baseline_bp = make_gbp(dem, baseline_spec, output_dir, "baseline")
    comparison_bp = make_gbp(dem, comparison_spec, output_dir, "comparison")
    baseline = decode_once(baseline_bp, syndrome, true_obs, args.max_iter)
    comparison = decode_once(comparison_bp, syndrome, true_obs, args.max_iter)

    region_rows, node_rows = gbp_policy_delta(
        baseline["recording"],
        comparison["recording"],
        comparison_bp.num_errors,
        comparison_bp.num_detectors,
    )

    region_path = output_dir / "region_delta.csv"
    node_path = output_dir / "node_scores.csv"
    edge_cover_path = output_dir / "edge_cover.csv"
    edge_candidates_path = output_dir / "edge_candidates.csv"
    context_cover_path = output_dir / "region_context_cover.csv"
    context_candidates_path = output_dir / "region_context_candidates.csv"
    nearest_baseline_path = output_dir / "nearest_baseline_regions.csv"
    plot_path = output_dir / "policy_delta.png"
    edge_plot_path = output_dir / "edge_cover.png"
    context_plot_path = output_dir / "region_context_cover.png"
    summary_path = output_dir / "summary.json"
    write_policy_delta_csvs(region_rows, node_rows, region_path, node_path)
    selected_edges, edge_candidates, edge_summary = gbp_delta_edge_cover(
        comparison_bp,
        syndrome,
        baseline["correction"],
        comparison["correction"],
        region_rows,
        baseline_recording=baseline["recording"],
        comparison_recording=comparison["recording"],
    )
    write_delta_edge_cover_csv(
        selected_edges,
        edge_candidates,
        edge_cover_path,
        edge_candidates_path,
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
    write_delta_region_context_cover_csv(
        selected_contexts,
        context_candidates,
        context_cover_path,
        context_candidates_path,
    )
    nearest_baseline_rows = gbp_nearest_baseline_region_matches(
        selected_contexts,
        baseline["recording"],
        top_k=args.nearest_baseline_k,
    )
    write_nearest_baseline_region_matches_csv(
        nearest_baseline_rows,
        nearest_baseline_path,
    )
    plot_policy_delta_graph(
        comparison_bp,
        syndrome,
        node_rows,
        plot_path,
        layout=layout_for_code(code),
        show_labels=not args.no_labels,
    )
    plot_delta_edge_cover_graph(
        comparison_bp,
        syndrome,
        selected_edges,
        edge_summary,
        edge_plot_path,
        layout=layout_for_code(code),
        show_labels=not args.no_labels,
    )
    plot_delta_region_context_cover_graph(
        comparison_bp,
        syndrome,
        selected_contexts,
        context_summary,
        context_plot_path,
        layout=layout_for_code(code),
        show_labels=not args.no_labels,
    )

    summary = {
        "code": code_name,
        "p": float(p),
        "syndrome_weight": int(syndrome.sum()),
        "true_obs": true_obs.astype(int).tolist(),
        "baseline_decoder": baseline_spec.label,
        "comparison_decoder": comparison_spec.label,
        "baseline": {
            "correct": baseline["correct"],
            "converged": baseline["converged"],
            "iterations": baseline["iterations"],
            "residual_weight": baseline["residual_weight"],
            "pred_obs": baseline["pred_obs"].astype(int).tolist(),
        },
        "comparison": {
            "correct": comparison["correct"],
            "converged": comparison["converged"],
            "iterations": comparison["iterations"],
            "residual_weight": comparison["residual_weight"],
            "pred_obs": comparison["pred_obs"].astype(int).tolist(),
        },
        "top_regions": region_rows[:20],
        "top_nodes": node_rows[:40],
        "edge_cover": edge_summary,
        "selected_edges": selected_edges,
        "region_context_cover": context_summary,
        "selected_region_contexts": selected_contexts,
        "nearest_baseline_regions": nearest_baseline_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Code        : {code}")
    print(f"Noise p     : {float(p):.4g}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    print(
        "Baseline    : "
        f"{baseline_spec.label}  converged={baseline['converged']}  "
        f"correct={baseline['correct']}  residual={baseline['residual_weight']}  "
        f"iters={baseline['iterations']}"
    )
    print(
        "Comparison  : "
        f"{comparison_spec.label}  converged={comparison['converged']}  "
        f"correct={comparison['correct']}  residual={comparison['residual_weight']}  "
        f"iters={comparison['iterations']}"
    )
    print(f"Delta regs  : {len(region_rows)} success-only active region signatures")
    if region_rows:
        print("Top regions :")
        for row in region_rows[:8]:
            print(
                f"  r{row['success_region']} count={row['count']} "
                f"first={row['first_iteration']} data={row['data']} "
                f"checks={sorted(set(row['cycle_checks']).union(row['internal_checks']))}"
            )
    if node_rows:
        print("Top nodes   :")
        for row in node_rows[:12]:
            print(f"  {row['kind']} {row['index']} score={row['score']}")
    print(
        "Edge cover  : "
        f"selected={edge_summary['num_selected']}  "
        f"candidates={edge_summary['num_candidates']}  "
        f"complete={edge_summary['complete']}  "
        f"targets={edge_summary['universe']}"
    )
    if selected_edges:
        for row in selected_edges[:12]:
            print(
                f"  c{row['check']}-v{row['data']} "
                f"covers={row['cover']} regions={row['regions']}"
            )
    print(
        "Context cov : "
        f"selected={context_summary['num_selected']}  "
        f"candidates={context_summary['num_candidates']}  "
        f"complete={context_summary['complete']}  "
        f"targets={context_summary['universe']}"
    )
    if selected_contexts:
        for row in selected_contexts[:8]:
            checks = sorted(set(row["cycle_checks"]).union(row["internal_checks"]))
            print(
                f"  r{row['success_region']} count={row['count']} "
                f"covers={row['cover']} data={row['data']} checks={checks}"
            )
    if nearest_baseline_rows:
        print("Nearest base:")
        for row in nearest_baseline_rows[: min(12, len(nearest_baseline_rows))]:
            first = row["baseline_first_active_iteration"]
            first_text = "never" if first is None else str(first)
            print(
                f"  r{row['success_region']} -> base r{row['baseline_region']} "
                f"rank={row['rank']} J={row['jaccard']:.3f} "
                f"exact={row['exact_match']} active={row['baseline_active_count']} "
                f"first={first_text} "
                f"data={row['data_intersection']}/{row['data_union']} "
                f"checks={row['check_intersection']}/{row['check_union']}"
            )
    print(f"Regions CSV : {region_path}")
    print(f"Nodes CSV   : {node_path}")
    print(f"Edges CSV   : {edge_cover_path}")
    print(f"Candidates  : {edge_candidates_path}")
    print(f"Context CSV : {context_cover_path}")
    print(f"Context cand: {context_candidates_path}")
    print(f"Nearest base: {nearest_baseline_path}")
    print(f"Graph       : {plot_path}")
    print(f"Edge graph  : {edge_plot_path}")
    print(f"Context gr. : {context_plot_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
