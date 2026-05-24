"""
Detect and plot simple-BP oscillations for trapping-set study.

Examples:
    python evaluations/bp_oscillations.py --code surface_5 --p 0.02
    python evaluations/bp_oscillations.py --cache results/errors/surface_5_0p02.50.npz --shot-index 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import (
    detect_bp_oscillation,
    plot_bp_oscillation_graph,
    plot_bp_oscillation_trace,
)

from _common import (
    CODES,
    layout_for_code,
    load_cached_shot,
    p_token,
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
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--key", choices=("decision", "residual"), default="decision",
                        help="state used for repeat detection")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite output_dir without prompting")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--until-oscillation", action="store_true",
                        help="resample until a repeated BP state is detected")
    parser.add_argument("--max-attempts", type=int, default=1000,
                        help="give up after this many samples when --until-oscillation")
    return parser.parse_args()


def make_bp(dem, output_dir: Path):
    cfg = DecoderConfig(
        decoder="simple",
        log=True,
        log_type="record",
        log_file=str(output_dir / "log.txt"),
        log_level=3,
        log_console=False,
        record_dir=str(output_dir),
    )
    return flashbp.FlashBP(dem, cfg)


def main():
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else None
    shot = {}
    metadata = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)

    output_dir = Path(
        args.output_dir
        or (
            f"results/bp_oscillations/{code_name}_{p_token(p)}."
            f"{cache_path.stem}_{args.shot_index}"
            if cache_path is not None
            else f"results/bp_oscillations/{code_name}_{p_token(p)}.simple"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    layout = layout_for_code(code)

    print(f"Code        : {code}")
    print(f"Noise p     : {float(p):.4g}")
    print(f"Key         : {args.key}")

    attempts = 0
    final_bp = None
    final_recording = None
    final_syndrome = None
    final_true_obs = None
    final_true_errors = None
    final_decision = None
    final_oscillation = None

    while True:
        attempts += 1
        shot_args = argparse.Namespace(**vars(args))
        if cache_path is None and args.syndrome is None and args.seed is not None:
            shot_args.seed = int(args.seed) + attempts - 1
        bp = make_bp(dem, output_dir)
        syndrome, true_obs, true_errors = sample_or_cached_shot(
            dem, shot_args, cache_path, shot, want_errors=True)
        if true_obs is None:
            true_obs = np.zeros(bp.num_observables, dtype=np.uint8)
        if true_errors is None:
            true_errors = np.zeros(bp.num_errors, dtype=np.uint8)

        decision = bp.decode(syndrome, args.max_iter)
        recording = bp.get_recording()
        oscillation = detect_bp_oscillation(
            bp, recording, shot_index=-1, key=args.key)

        final_bp = bp
        final_recording = recording
        final_syndrome = syndrome
        final_true_obs = true_obs
        final_true_errors = true_errors
        final_decision = decision
        final_oscillation = oscillation

        if (
            cache_path is not None
            or args.syndrome is not None
            or not args.until_oscillation
            or oscillation.found
            or attempts >= args.max_attempts
        ):
            break

    pred_obs = (final_bp.L @ final_decision.astype(np.int32)) % 2
    correct = bool(np.array_equal(pred_obs, final_true_obs))

    graph_path = output_dir / "oscillation_graph.png"
    trace_path = output_dir / "oscillation_trace.png"
    summary_path = output_dir / "summary.json"

    final_oscillation = plot_bp_oscillation_graph(
        final_bp,
        final_recording,
        graph_path,
        shot_index=-1,
        key=args.key,
        layout=layout,
        show_labels=not args.no_labels,
    )
    plot_bp_oscillation_trace(final_oscillation, trace_path)

    summary = {
        "code": code_name,
        "p": float(p),
        "attempts": attempts,
        "max_iter": args.max_iter,
        "key": args.key,
        "found": final_oscillation.found,
        "start": final_oscillation.start,
        "end": final_oscillation.end,
        "period": final_oscillation.period,
        "syndrome_weight": int(final_syndrome.sum()),
        "decision_weight": int(final_decision.sum()),
        "correct": correct,
        "true_obs": final_true_obs.astype(int).tolist(),
        "pred_obs": pred_obs.astype(int).tolist(),
        "flipping_data": final_oscillation.flipping_data,
        "active_data": final_oscillation.active_data,
        "unsatisfied_checks": final_oscillation.unsatisfied_checks,
        "residual_weights": final_oscillation.residual_weights,
        "decision_weights": final_oscillation.decision_weights,
        "true_error_weight": int(final_true_errors.sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Attempts    : {attempts}")
    print(f"Syndrome wt : {int(final_syndrome.sum())}")
    print(f"Decision wt : {int(final_decision.sum())}")
    print(f"Correct     : {correct}  true_obs={final_true_obs}  pred_obs={pred_obs}")
    if final_oscillation.found:
        print(
            "Oscillation : "
            f"period={final_oscillation.period}  "
            f"start={final_oscillation.start}  end={final_oscillation.end}"
        )
    else:
        print("Oscillation : no repeated state detected")
    print(f"Flipping    : {final_oscillation.flipping_data}")
    print(f"Unsatisfied : {final_oscillation.unsatisfied_checks}")
    print(f"Graph       : {graph_path}")
    print(f"Trace       : {trace_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
