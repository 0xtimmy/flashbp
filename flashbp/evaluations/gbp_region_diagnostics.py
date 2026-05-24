"""
Diagnose whether GBP regions reject the decoder's current hard decision.

For each active region at a chosen iteration, this script recomputes the local
valid-state table, checks the global hard decision restricted to region.data,
finds the best valid local state under the region costs, and records outgoing
region LLR signs.

Outputs:
    diagnostics.csv     compact table for spreadsheet inspection
    diagnostics.json    full rows including bit vectors and output LLRs
    diagnostics.png     summary plot of invalid regions and Hamming distances

Examples:
    python evaluations/gbp_region_diagnostics.py --code surface_5 --p 0.02 --decoder gbp-cycles:8
    python evaluations/gbp_region_diagnostics.py --cache results/errors/surface_5_0p02.50.npz --shot-index 3 --decoder gbp-union-cycles-any:8:sparse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import flashbp
from flashbp.analytics import (
    gbp_region_diagnostics,
    plot_gbp_region_diagnostics,
    write_gbp_region_diagnostics_csv,
)

from _common import (
    CODES,
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
    parser.add_argument("--iteration-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--max-dense-states", type=int, default=1 << 22,
                        help="skip exact diagnostics for regions wider than this dense table size")
    parser.add_argument("--all-regions", action="store_true",
                        help="diagnose inactive regions too")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite output_dir without prompting")
    return parser.parse_args()


def main():
    args = parse_args()
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
            f"results/gbp_region_diagnostics/{code_name}_{p_token(p)}."
            f"{cache_path.stem}_{args.shot_index}_{label_token}"
            if cache_path is not None
            else f"results/gbp_region_diagnostics/{code_name}_{p_token(p)}.{label_token}"
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
    syndrome, true_obs, _ = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_obs is None:
        true_obs = np.zeros(bp.num_observables, dtype=np.uint8)

    result = bp.decode(syndrome, args.max_iter)
    pred_obs = (bp.L @ result.astype(np.int32)) % 2
    correct = bool(np.array_equal(pred_obs, true_obs))
    recording = bp.get_recording()

    rows = gbp_region_diagnostics(
        bp,
        recording,
        shot_index=-1,
        iteration_index=args.iteration_index,
        only_active=not args.all_regions,
        max_dense_states=args.max_dense_states,
    )

    csv_path = output_dir / "diagnostics.csv"
    json_path = output_dir / "diagnostics.json"
    plot_path = output_dir / "diagnostics.png"
    write_gbp_region_diagnostics_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    try:
        plot_gbp_region_diagnostics(rows, plot_path)
    except ValueError as exc:
        print(f"WARNING: skipping diagnostics plot: {exc}")

    exact = [row for row in rows if not row.get("skipped")]
    invalid = [row for row in exact if row.get("current_valid") is False]
    skipped = [row for row in rows if row.get("skipped")]
    worst = sorted(
        invalid,
        key=lambda row: (
            -1 if row.get("hamming_to_best") is None else row["hamming_to_best"],
            row["region"],
        ),
        reverse=True,
    )[:8]

    print(f"Code        : {code}")
    print(f"Noise p     : {float(p):.4g}")
    print(f"Decoder     : {spec.label}")
    print(f"Policy      : {cfg.region_policy}")
    print(f"Backend     : {cfg.gbp_backend}")
    print(f"Iteration   : {args.iteration_index}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    print(f"Correct     : {correct}  true_obs={true_obs}  pred_obs={pred_obs}")
    print(f"Regions     : {len(rows)} diagnosed  exact={len(exact)}  skipped={len(skipped)}")
    print(f"Invalid     : {len(invalid)} exact regions reject the current hard decision")
    if worst:
        print("Worst       :")
        for row in worst:
            print(
                f"  r{row['region']}  K={row['num_axes']}  "
                f"valid={row['valid_state_count']}/{row['dense_state_count']}  "
                f"hamming={row['hamming_to_best']}  "
                f"current={row['current_bits']}  best={row['best_bits']}"
            )
    print(f"CSV         : {csv_path}")
    print(f"JSON        : {json_path}")
    if plot_path.exists():
        print(f"Plot        : {plot_path}")


if __name__ == "__main__":
    main()
