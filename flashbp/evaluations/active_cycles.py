"""
Render Tanner-graph cycles that touch syndrome-active parity checks.

Typical workflow:
    python evaluations/cache_bp_ml_failures.py --code steane --p 0.15 --target 5
    python evaluations/active_cycles.py --cache results/errors/steane_0p15.5.npz

The script writes:
    active_cycles.png          all active-check cycles overlaid in one graph
    cycles/cycles.mp4          optional one-cycle-per-frame video
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import plot_active_check_cycles
from flashbp.animation import animate_cycles
from _common import CODES, layout_for_code, p_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=None,
                        help="NPZ cache from cache_bp_ml_failures.py")
    parser.add_argument("--shot-index", type=int, default=0,
                        help="which cached shot to render")
    parser.add_argument("--code", choices=CODES.keys(), default=None,
                        help="code name; inferred from cache metadata when possible")
    parser.add_argument("--p", type=float, default=None,
                        help="DEM physical error rate; inferred from cache metadata when possible")
    parser.add_argument("--syndrome", type=str, default=None,
                        help="explicit syndrome bits, e.g. 100101")
    parser.add_argument("--max-dist", type=int, default=8,
                        help="maximum cycle length")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--framerate", type=float, default=2.0)
    parser.add_argument("--no-video", action="store_true",
                        help="only write the combined PNG")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="overwrite output_dir without prompting")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        reply = input(f"Output dir '{output_dir}' already exists. Overwrite? [y/N]: ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)
    shutil.rmtree(output_dir)


def parse_syndrome_bits(text: str) -> np.ndarray:
    clean = "".join(ch for ch in text if ch in "01")
    if not clean:
        raise ValueError("explicit --syndrome must contain at least one 0/1 bit")
    return np.asarray([int(ch) for ch in clean], dtype=np.uint8)


def load_cache(path: Path, shot_index: int) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    syndromes = data["syndromes"]
    if shot_index < 0:
        shot_index += syndromes.shape[0]
    if shot_index < 0 or shot_index >= syndromes.shape[0]:
        raise IndexError(
            f"shot_index {shot_index} outside cache with {syndromes.shape[0]} shots"
        )
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    return syndromes[shot_index].astype(np.uint8), metadata


def main():
    args = parse_args()

    metadata = {}
    cache_path = Path(args.cache) if args.cache else None
    if args.syndrome is not None:
        syndrome = parse_syndrome_bits(args.syndrome)
    elif cache_path is not None:
        syndrome, metadata = load_cache(cache_path, args.shot_index)
    else:
        raise ValueError("provide either --cache or --syndrome")

    code_name = args.code or metadata.get("code")
    p = args.p if args.p is not None else metadata.get("p", 0.05)
    if code_name not in CODES:
        raise ValueError(
            "could not infer code; pass --code explicitly "
            f"(known: {', '.join(CODES)})"
        )

    output_dir = Path(
        args.output_dir
        or (
            f"results/cycles/{code_name}_{p_token(float(p))}.active"
            if cache_path is None
            else f"results/cycles/{code_name}_{p_token(float(p))}.{cache_path.stem}_{args.shot_index}"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    bp = flashbp.FlashBP(dem, DecoderConfig())
    if syndrome.shape[0] != bp.num_detectors:
        raise ValueError(
            f"syndrome has length {syndrome.shape[0]}, "
            f"but {code_name} has {bp.num_detectors} detectors"
        )

    layout = layout_for_code(code)
    png_path = output_dir / "active_cycles.png"
    cycles = plot_active_check_cycles(
        bp,
        syndrome,
        output_path=png_path,
        max_length=args.max_dist,
        layout=layout,
        show_labels=not args.no_labels,
    )

    video_path = None
    if not args.no_video:
        video_path = animate_cycles(
            bp,
            output_dir / "cycles",
            max_dist=args.max_dist,
            framerate=args.framerate,
            layout=layout,
            syndrome=syndrome,
        )

    lengths = [len(c) for c in cycles]
    length_text = "none" if not lengths else f"{min(lengths)}..{max(lengths)}"
    print(f"Code          : {code}")
    print(f"Noise p       : {float(p):.3%}")
    print(f"Syndrome wt   : {int(syndrome.sum())}")
    print(f"Max dist      : {args.max_dist}")
    print(f"Active cycles : {len(cycles)}  lengths={length_text}")
    print(f"PNG           : {png_path}")
    if video_path is not None:
        print(f"Video         : {video_path}")


if __name__ == "__main__":
    main()
