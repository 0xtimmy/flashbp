"""
Decode one shot with a GBP policy and animate the active regions over iterations.

Examples:
    python evaluations/animate_gbp.py --code steane --p 0.08 --decoder gbp-cycles:8
    python evaluations/animate_gbp.py --cache results/errors/steane_0p05.20.npz --decoder gbp-union-cycles-any:8
"""
import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp.animation import animate_gbp_recording

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
                        help="GBP decoder spec, e.g. gbp-check:2, "
                             "gbp-cycles:8:sparse, gbp-union-cycles-any:8")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=650)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--framerate", type=float, default=2.0)
    parser.add_argument("--max-region-frames", type=int, default=24,
                        help="maximum active regions rendered per iteration")
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
            f"results/recordings/gbp/{code_name}_{p_token(p)}.{cache_path.stem}_{args.shot_index}_{label_token}"
            if cache_path is not None
            else f"results/recordings/gbp/{code_name}_{p_token(p)}.{label_token}"
        )
    )
    prepare_output_dir(output_dir, args.force)

    code = CODES[code_name]()
    dem = code.to_dem(p)

    cfg = spec.config
    cfg.log = True
    cfg.log_type = "gbp"
    cfg.log_file = str(output_dir / "log.txt")
    cfg.log_level = 5
    cfg.log_console = True
    cfg.record_dir = str(output_dir)

    bp = flashbp.FlashBP(dem, cfg)
    syndrome, true_obs, true_errors = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_obs is None:
        true_obs = np.zeros(bp.num_observables, dtype=np.uint8)
    if true_errors is None:
        true_errors = np.zeros(bp.num_errors, dtype=np.uint8)

    result = bp.decode(syndrome, args.max_iter)
    pred_obs = (bp.L @ result.astype(np.int32)) % 2
    correct = bool(np.array_equal(pred_obs, true_obs))

    print(f"Code        : {code}")
    print(f"Noise p     : {p:.4g}")
    print(f"Decoder     : {spec.label}")
    print(f"Policy      : {cfg.region_policy}")
    print(f"Degree      : {cfg.degree}")
    print(f"Syndrome    : {syndrome}  (weight {int(syndrome.sum())})")
    print(f"Decision    : {result}  (weight {int(result.sum())})")
    print(f"True obs    : {true_obs}    Pred obs: {pred_obs}    Correct: {correct}")

    recording = bp.get_recording()
    n_iter = len(recording[-1]["iterations"])
    layout = layout_for_code(code)
    print(f"\nRendering GBP policy animation to {output_dir / 'frames'} ...")
    video = animate_gbp_recording(
        bp,
        recording,
        output_dir,
        policy=cfg.region_policy,
        degree=cfg.degree,
        shot_index=-1,
        framerate=args.framerate,
        video_name="gbp.mp4",
        layout=layout,
        true_errors=true_errors,
        max_region_frames=args.max_region_frames,
    )
    print(f"Iterations  : {n_iter}")
    print(f"Video       : {video}")


if __name__ == "__main__":
    main()
