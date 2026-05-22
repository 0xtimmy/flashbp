"""
Render a syndrome on the Tanner graph.

Active detections are filled black.  If true errors are available, all Tanner
edges incident to erroring data nodes are drawn red.

Examples:
    python evaluations/render_syndrome.py --cache results/errors/steane_0p15.5.npz
    python evaluations/render_syndrome.py --code steane --p 0.15 --seed 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import plot_syndrome_graph
from _common import CODES, layout_for_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=None,
                        help="NPZ cache from cache_bp_ml_failures.py")
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--code", choices=CODES.keys(), default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--syndrome", type=str, default=None,
                        help="explicit syndrome bit string")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-labels", action="store_true")
    return parser.parse_args()


def parse_bits(text: str) -> np.ndarray:
    clean = "".join(ch for ch in text if ch in "01")
    if not clean:
        raise ValueError("bit string must contain at least one 0/1 bit")
    return np.asarray([int(ch) for ch in clean], dtype=np.uint8)


def load_cache(path: Path, shot_index: int) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    syndromes = data["syndromes"]
    if shot_index < 0:
        shot_index += syndromes.shape[0]
    if shot_index < 0 or shot_index >= syndromes.shape[0]:
        raise IndexError(
            f"shot_index {shot_index} outside cache with {syndromes.shape[0]} shots"
        )
    shot = {"syndrome": syndromes[shot_index].astype(np.uint8)}
    if "true_errors" in data:
        shot["true_errors"] = data["true_errors"][shot_index].astype(np.uint8)
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    return shot, metadata


def main():
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else None
    metadata = {}
    shot = {}

    if cache_path is not None:
        shot, metadata = load_cache(cache_path, args.shot_index)

    code_name = args.code or metadata.get("code") or "steane"
    p = args.p if args.p is not None else metadata.get("p", 0.05)
    if code_name not in CODES:
        raise ValueError(f"unknown code {code_name!r}")

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    bp = flashbp.FlashBP(dem, DecoderConfig())

    if args.syndrome is not None:
        syndrome = parse_bits(args.syndrome)
        true_errors = None
    elif "syndrome" in shot:
        syndrome = shot["syndrome"]
        true_errors = shot.get("true_errors")
    else:
        if args.seed is not None:
            np.random.seed(args.seed)
        sampler = dem.compile_sampler()
        det, _, err = sampler.sample(shots=1, return_errors=True)
        syndrome = det[0].astype(np.uint8)
        true_errors = err[0].astype(np.uint8)

    if syndrome.shape[0] != bp.num_detectors:
        raise ValueError(
            f"syndrome has length {syndrome.shape[0]}, expected {bp.num_detectors}"
        )

    output = Path(
        args.output
        or (
            f"results/syndromes/{cache_path.stem}_{args.shot_index}.png"
            if cache_path is not None
            else f"results/syndromes/{code_name}_p{str(float(p)).replace('.', 'p')}.png"
        )
    )
    layout = layout_for_code(code)
    plot_syndrome_graph(
        bp,
        syndrome,
        output_path=output,
        error_vector=true_errors,
        layout=layout,
        show_labels=not args.no_labels,
    )

    print(f"Code        : {code}")
    print(f"Noise p     : {float(p):.3%}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    if true_errors is not None:
        print(f"Error wt    : {int(true_errors.sum())}")
    print(f"Output      : {output}")


if __name__ == "__main__":
    main()
