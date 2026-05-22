"""
Enumerate and animate cycles in a code's Tanner graph.

Each cycle is rendered as one frame, in ascending order of length, then
stitched into an mp4 via ffmpeg.

Usage:
    python evaluations/cycles.py --code steane --max-dist 6
    python evaluations/cycles.py --code gross  --max-dist 6 --framerate 4
"""
import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp           import DecoderConfig
from flashbp.animation import animate_cycles
from _common import (
    CODES,
    layout_for_code,
    load_cached_shot,
    prepare_output_dir,
    resolve_code_and_p,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code",       choices=CODES.keys(), default=None)
    parser.add_argument("--cache",      type=str, default=None)
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--syndrome",   type=str, default=None,
                        help="bit string of active detectors; filters cycles")
    parser.add_argument("--max-dist",   type=int,   default=6,
                        help="maximum cycle length (must be >= 4)")
    parser.add_argument("--p",          type=float, default=None,
                        help="error rate for DEM construction (doesn't "
                             "affect topology — only the FlashBP build)")
    parser.add_argument("--output-dir", type=str,   default=None)
    parser.add_argument("--framerate",  type=float, default=2.0)
    parser.add_argument("--force",      action="store_true",
                        help="overwrite output_dir without prompting")
    return parser.parse_args()


def main():
    args       = parse_args()
    cache_path = Path(args.cache) if args.cache else None
    metadata = {}
    shot = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)
    output_dir = Path(
        args.output_dir
        or (
            f"results/cycles/{cache_path.stem}_{args.shot_index}_all"
            if cache_path is not None
            else f"results/cycles/{code_name}_all"
        )
    )

    prepare_output_dir(output_dir, args.force)

    code   = CODES[code_name]()
    dem    = code.to_dem(p)
    bp     = flashbp.FlashBP(dem, DecoderConfig())
    layout = layout_for_code(code)
    syndrome = None
    if args.syndrome is not None:
        syndrome = np.asarray([1 if ch == "1" else 0 for ch in args.syndrome.strip()],
                              dtype=np.uint8)
    elif "syndrome" in shot:
        syndrome = shot["syndrome"]

    print(f"Code     : {code}")
    print(f"Max dist : {args.max_dist}")
    print("Searching for cycles ...")

    video = animate_cycles(bp, output_dir, max_dist=args.max_dist,
                           framerate=args.framerate, layout=layout,
                           syndrome=syndrome)
    print(f"Video    : {video}")


if __name__ == "__main__":
    main()
