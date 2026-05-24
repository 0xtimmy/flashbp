"""
Render GBP region sparsity visualizations for one decoded syndrome.

Outputs:
    region_summary.png      bar plot of region sparsity
    region_heatmap_rN.png   valid/invalid state heatmap for one region
    region_heatmaps/*.png   valid/invalid state heatmaps when --region all
    region_overlay.png      Tanner graph with active region hulls by sparsity

Examples:
    python evaluations/gbp_sparsity.py --code steane --p 0.05 --decoder gbp-cycles:8
    python evaluations/gbp_sparsity.py --cache results/errors/steane_0p05.20.npz --decoder gbp-union-cycles-any:8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp.analytics import (
    choose_region,
    gbp_sparsity_stats,
    plot_gbp_region_heatmap,
    plot_gbp_sparsity_graph,
    plot_gbp_sparsity_summary,
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
    parser.add_argument("--decoder", type=str, default="gbp-cycles:8",
                        help="GBP decoder spec, e.g. gbp-cycles:8, "
                             "gbp-cycles:8:sparse, gbp-union-cycles-any:8")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--region", type=str, default=None,
                        help="region index for heatmap, or 'all'; defaults to sparsest active region")
    parser.add_argument("--iteration-index", type=int, default=-1)
    parser.add_argument("--summary-sort",
                        choices=("region", "sparsity", "compression"),
                        default="region")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite output_dir without prompting")
    parser.add_argument("--no-labels", action="store_true")
    return parser.parse_args()


def parse_region_arg(region: str | None) -> int | str | None:
    if region is None:
        return None
    if region.lower() == "all":
        return "all"
    try:
        return int(region)
    except ValueError as exc:
        raise ValueError("--region must be an integer region index or 'all'") from exc


def main():
    args = parse_args()
    requested_region = parse_region_arg(args.region)
    spec = parse_decoder_spec(args.decoder)
    if spec.config is None or spec.config.decoder != "gbp":
        raise ValueError("--decoder must be a GBP decoder spec")
    label_token = spec.label.replace(":", "_")

    cache_path = Path(args.cache) if args.cache else None
    shot = {}
    metadata = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)

    output_dir = Path(
        args.output_dir
        or (
            f"results/gbp_sparsity/{code_name}_{p_token(p)}.{cache_path.stem}_{args.shot_index}_{label_token}"
            if cache_path is not None
            else f"results/gbp_sparsity/{code_name}_{p_token(p)}.{label_token}"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    cfg = spec.config
    cfg.log = True
    cfg.log_type = "gbp"
    cfg.log_console = False
    cfg.log_file = str(output_dir / "log.txt")
    cfg.log_level = 3
    cfg.record_dir = str(output_dir)

    bp = flashbp.FlashBP(dem, cfg)
    syndrome, true_obs, true_errors = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_obs is None:
        true_obs = np.zeros(bp.num_observables, dtype=np.uint8)

    result = bp.decode(syndrome, args.max_iter)
    pred_obs = (bp.L @ result.astype(np.int32)) % 2
    correct = bool(np.array_equal(pred_obs, true_obs))
    recording = bp.get_recording()
    selected_region = None
    if requested_region != "all":
        selected_region = choose_region(
            recording,
            shot_index=-1,
            iteration_index=args.iteration_index,
            region_index=requested_region,
        )
    layout = layout_for_code(code)

    summary_path = output_dir / "region_summary.png"
    overlay_path = output_dir / "region_overlay.png"

    stats = plot_gbp_sparsity_summary(
        recording,
        summary_path,
        shot_index=-1,
        iteration_index=args.iteration_index,
        sort_by=args.summary_sort,
    )

    heatmap_paths: list[Path] = []
    if requested_region == "all":
        heatmap_dir = output_dir / "region_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for row in sorted(stats, key=lambda item: int(item["index"])):
            region_index = int(row["index"])
            heatmap_path = heatmap_dir / f"region_heatmap_r{region_index}.png"
            plot_gbp_region_heatmap(
                recording,
                heatmap_path,
                region_index=region_index,
                shot_index=-1,
                iteration_index=args.iteration_index,
            )
            heatmap_paths.append(heatmap_path)
    else:
        heatmap_path = output_dir / f"region_heatmap_r{selected_region}.png"
        plot_gbp_region_heatmap(
            recording,
            heatmap_path,
            region_index=selected_region,
            shot_index=-1,
            iteration_index=args.iteration_index,
        )
        heatmap_paths.append(heatmap_path)

    plot_gbp_sparsity_graph(
        bp,
        recording,
        overlay_path,
        shot_index=-1,
        iteration_index=args.iteration_index,
        layout=layout,
        show_labels=not args.no_labels,
    )

    print(f"Code        : {code}")
    print(f"Noise p     : {float(p):.4g}")
    print(f"Decoder     : {spec.label}")
    print(f"Policy      : {cfg.region_policy}")
    print(f"Backend     : {cfg.gbp_backend}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    print(f"Correct     : {correct}  true_obs={true_obs}  pred_obs={pred_obs}")
    if requested_region == "all":
        print(f"Regions     : all ({len(heatmap_paths)} heatmaps)")
    else:
        selected_stats = next(row for row in stats if row["index"] == selected_region)
        print(
            "Region      : "
            f"{selected_region}  K={selected_stats['num_axes']}  "
            f"valid={selected_stats['valid_state_count']}/"
            f"{selected_stats['dense_state_count']}  "
            f"compression={selected_stats['compression']:.3g}x"
        )
    print(f"Summary     : {summary_path}")
    if requested_region == "all":
        print(f"Heatmaps    : {output_dir / 'region_heatmaps'}")
    else:
        print(f"Heatmap     : {heatmap_paths[0]}")
    print(f"Overlay     : {overlay_path}")


if __name__ == "__main__":
    main()
