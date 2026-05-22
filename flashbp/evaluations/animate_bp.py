"""
Decode one shot with simple BP/RecordLogger and animate the BP iterations.

Usage:
    python evaluations/animate_bp.py --code steane --p 0.15
    python evaluations/animate_bp.py --cache results/errors/steane_0p15.5.npz
"""
import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp           import DecoderConfig
from flashbp.animation import animate, animate_cycles
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
    parser.add_argument("--cache",      type=str, default=None)
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--code",       choices=CODES.keys(), default=None)
    parser.add_argument("--p",          type=float, default=None)
    parser.add_argument("--syndrome",   type=str, default=None)
    parser.add_argument("--max-iter",   type=int,   default=20)
    parser.add_argument("--seed",       type=int,   default=650)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--framerate",  type=float, default=2.0)
    parser.add_argument("--force",      action="store_true",
                        help="overwrite output_dir without prompting")
    parser.add_argument("--until-fail", action="store_true",
                        help="resample shots until BP gives a wrong answer")
    parser.add_argument("--max-attempts", type=int, default=1000,
                        help="give up after this many samples when --until-fail")
    parser.add_argument("--cycle-max-dist", type=int, default=6,
                        help="max cycle length for the cycle animation "
                             "(set to 0 to skip)")
    return parser.parse_args()


def main():
    args       = parse_args()
    cache_path = Path(args.cache) if args.cache else None
    shot = {}
    metadata = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)
    output_dir = Path(
        args.output_dir
        or (
            f"results/recordings/bp/{cache_path.stem}_{args.shot_index}"
            if cache_path is not None
            else f"results/recordings/bp/{code_name}_p{p_token(p)}"
        )
    )

    prepare_output_dir(output_dir, args.force)

    code = CODES[code_name]()
    dem  = code.to_dem(p)

    cfg = DecoderConfig(
        decoder="simple",
        log=True,
        log_type="record",
        log_file=str(output_dir / "log.txt"),
        log_level=5,
        log_console=True,
        record_dir=str(output_dir),
    )
    bp = flashbp.FlashBP(dem, cfg)
 
    print(f"Code        : {code}")
    print(f"Noise p     : {p:.1%}")

    attempts = 0
    while True:
        attempts += 1
        syndrome, true_obs, true_errors = sample_or_cached_shot(
            dem, args, cache_path, shot, want_errors=True)
        if true_obs is None:
            true_obs = np.zeros(bp.num_observables, dtype=np.uint8)
        if true_errors is None:
            true_errors = np.zeros(bp.num_errors, dtype=np.uint8)

        result   = bp.decode(syndrome, args.max_iter)
        pred_obs = (bp.L @ result.astype(np.int32)) % 2
        correct  = bool(np.array_equal(pred_obs, true_obs))

        if cache_path is not None or args.syndrome is not None or not args.until_fail or not correct:
            break
        if attempts >= args.max_attempts:
            print(f"No failure found in {args.max_attempts} attempts; "
                  f"animating the last (correct) decode.")
            break

    print(f"Attempts    : {attempts}")
    print(f"Syndrome    : {syndrome}  (weight {syndrome.sum()})")
    print(f"Decision    : {result}  (weight {result.sum()})")
    print(f"True obs    : {true_obs}    Pred obs: {pred_obs}    Correct: {correct}")

    recording = bp.get_recording()
    n_iter    = len(recording[-1]["iterations"])
    layout    = layout_for_code(code)
    print(f"\nRendering {n_iter} frames to {output_dir / 'frames'} ...")
    video = animate(bp, recording, output_dir,
                    shot_index=-1, framerate=args.framerate, layout=layout,
                    true_errors=true_errors)
    print(f"Video       : {video}")

    if args.cycle_max_dist >= 4:
        cycle_dir = output_dir / "cycles"
        print(f"\nRendering cycles (length <= {args.cycle_max_dist}) "
              f"to {cycle_dir / 'frames'} ...")
        cycle_video = animate_cycles(bp, cycle_dir,
                                     max_dist=args.cycle_max_dist,
                                     framerate=args.framerate,
                                     layout=layout,
                                     syndrome=syndrome)
        print(f"Cycle video : {cycle_video}")


if __name__ == "__main__":
    main()
