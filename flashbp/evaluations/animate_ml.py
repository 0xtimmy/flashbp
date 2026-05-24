"""
Animate the maximum-likelihood coset-sum contraction.

Usage:
    python evaluations/ml_contraction_animation.py --code steane --force
    python evaluations/ml_contraction_animation.py --code surface_3 --force
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import plot_detector_distance_graph, plot_ml_surprise_graph
from flashbp.animation import (
    animate_cycles,
    animate_ml_contraction_recording,
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
    parser.add_argument("--code", choices=CODES.keys(), default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--syndrome", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--framerate", type=float, default=2.0)
    parser.add_argument("--max-bars", type=int, default=64,
                        help="maximum accumulated-state bars to draw per frame")
    parser.add_argument("--posterior-max-bits", type=int, default=22,
                        help="exact node-likelihood enumeration limit")
    parser.add_argument("--cycle-max-dist", type=int, default=6,
                        help="max cycle length for the cycle animation "
                             "(set to 0 to skip)")
    parser.add_argument("--no-distance-map", action="store_true",
                        help="skip the detector-distance analytics PNG")
    parser.add_argument("--no-surprise-map", action="store_true",
                        help="skip the ML branch-surprise analytics PNG")
    parser.add_argument("--surprise-metric", type=str, default="js_divergence",
                        choices=["js_divergence", "kl_0_to_1", "kl_1_to_0", "sym_kl"],
                        help="branch-divergence metric used for the surprise map")
    parser.add_argument("--log-file", type=str, default=None,
                        help="C++ ML decoder log file; set to empty string to disable file logging")
    parser.add_argument("--log-level", type=int, default=2,
                        help="C++ ML decoder log level")
    parser.add_argument("--no-log-console", action="store_true",
                        help="disable C++ decoder log output on stdout")
    parser.add_argument("--log-buffered", action="store_true",
                        help="buffer C++ decoder logs until flush")
    parser.add_argument("--force", action="store_true",
                        help="overwrite output_dir without prompting")
    return parser.parse_args()


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
            f"results/recordings/ml/{code_name}_{p_token(p)}.{cache_path.stem}_{args.shot_index}"
            if cache_path is not None
            else f"results/recordings/ml/{code_name}_{p_token(p)}.contraction"
        )
    )
    prepare_output_dir(output_dir, args.force)

    code = CODES[code_name]()
    dem = code.to_dem(p)
    log_file = args.log_file if args.log_file is not None else str(output_dir / "log.txt")
    bp = flashbp.FlashBP(
        dem,
        DecoderConfig(
            decoder="ml",
            log=True,
            log_type="ml",
            log_level=args.log_level,
            log_console=not args.no_log_console,
            log_file=log_file,
            log_buffered=args.log_buffered,
        ),
    )

    syndrome, true_obs, true_errors = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_obs is None:
        true_obs = np.zeros(bp.num_observables, dtype=np.uint8)
    if true_errors is None:
        true_errors = np.zeros(bp.num_errors, dtype=np.uint8)

    result = None
    pred_obs = None
    decode_warning = None
    try:
        result = bp.decode(syndrome, 1)
        bp.flush()
        pred_obs = (bp.L @ result.astype(np.int32)) % 2
    except RuntimeError as exc:
        bp.flush()
        decode_warning = str(exc).splitlines()[0]

    if decode_warning is not None:
        print(f"Decode note : {decode_warning}")
        sys.exit(1)

    recording = bp.get_recording()
    layout = layout_for_code(code)
    video = animate_ml_contraction_recording(
        bp,
        recording,
        output_dir,
        framerate=args.framerate,
        layout=layout,
        max_bars=args.max_bars,
        posterior_max_bits=args.posterior_max_bits,
        true_errors=true_errors,
        predicted_errors=result,
    )

    cycle_video = None
    cycle_warning = None
    if args.cycle_max_dist >= 4:
        try:
            cycle_video = animate_cycles(
                bp,
                output_dir / "cycles",
                max_dist=args.cycle_max_dist,
                framerate=args.framerate,
                layout=layout,
                syndrome=syndrome,
            )
        except ValueError as exc:
            cycle_warning = str(exc)

    distance_map = None
    if not args.no_distance_map:
        distance_map = output_dir / "detector_distance.png"
        plot_detector_distance_graph(
            bp,
            output_path=distance_map,
            syndrome=syndrome,
            layout=layout,
            show_labels=True,
        )

    surprise_map = None
    if not args.no_surprise_map:
        surprise_map = output_dir / "ml_surprise.png"
        plot_ml_surprise_graph(
            bp,
            recording,
            output_path=surprise_map,
            metric=args.surprise_metric,
            layout=layout,
            syndrome=syndrome,
            show_labels=True,
        )

    print(f"Code        : {code}")
    print("Decoder     : ml")
    print("Contraction : C++ MLLogger recording")
    print(f"C++ logging : level={args.log_level}  console={not args.no_log_console}  file={log_file}")
    print(f"Noise p     : {p:.3%}")
    print(f"Error axes  : {bp.num_errors}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    print(f"True err wt : {int(true_errors.sum())}")
    print(f"True obs    : {true_obs}    Pred obs: {pred_obs}")
    if decode_warning is not None:
        print(f"Decode note : {decode_warning}")
    print(f"Video       : {video}")
    if distance_map is not None:
        print(f"Distance map: {distance_map}")
    if surprise_map is not None:
        print(f"Surprise map: {surprise_map}")
    if cycle_video is not None:
        print(f"Cycle video : {cycle_video}")
    elif cycle_warning is not None:
        print(f"Cycle note  : {cycle_warning}")


if __name__ == "__main__":
    main()
