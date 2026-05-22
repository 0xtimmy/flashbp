"""
Run exact BP on severed detection-rooted trees for each active syndrome bit.

Typical workflow:
    python evaluations/cache_bp_ml_failures.py --code steane --p 0.15 --target 5
    python evaluations/tree_bp_detections.py --cache results/errors/steane_0p15.5.npz

The default output directory is:
    results/tree_bp/{cache_stem}_{shot_index}/
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
from flashbp.analytics import plot_tree_bp_marginals
from _common import CODES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=None,
                        help="NPZ cache from cache_bp_ml_failures.py")
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--code", choices=CODES.keys(), default=None,
                        help="code name; inferred from cache metadata when possible")
    parser.add_argument("--p", type=float, default=None,
                        help="DEM physical error rate; inferred from cache metadata when possible")
    parser.add_argument("--syndrome", type=str, default=None,
                        help="explicit syndrome bits, e.g. 100101")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="optional BFS tree depth cutoff")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="simple BP iterations for comparison")
    parser.add_argument("--output-dir", type=str, default=None)
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


def load_cache(path: Path, shot_index: int) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    syndromes = data["syndromes"]
    if shot_index < 0:
        shot_index += syndromes.shape[0]
    if shot_index < 0 or shot_index >= syndromes.shape[0]:
        raise IndexError(
            f"shot_index {shot_index} outside cache with {syndromes.shape[0]} shots"
        )
    shot = {
        "syndrome": syndromes[shot_index].astype(np.uint8),
    }
    for key in ("true_errors", "bp_corrections", "ml_corrections"):
        if key in data:
            shot[key] = data[key][shot_index].astype(np.uint8)
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    return shot, metadata


def main():
    args = parse_args()

    metadata = {}
    cache_path = Path(args.cache) if args.cache else None
    shot: dict = {}
    if args.syndrome is not None:
        syndrome = parse_syndrome_bits(args.syndrome)
    elif cache_path is not None:
        shot, metadata = load_cache(cache_path, args.shot_index)
        syndrome = shot["syndrome"]
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
            f"results/tree_bp/{code_name}_syndrome"
            if cache_path is None
            else f"results/tree_bp/{cache_path.stem}_{args.shot_index}"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = CODES[code_name]()
    dem = code.to_dem(float(p))
    bp = flashbp.FlashBP(dem, DecoderConfig(decoder="simple"))
    if syndrome.shape[0] != bp.num_detectors:
        raise ValueError(
            f"syndrome has length {syndrome.shape[0]}, "
            f"but {code_name} has {bp.num_detectors} detectors"
        )

    active_detections = np.flatnonzero(syndrome).astype(int).tolist()
    if not active_detections:
        print("Syndrome has no active detections; no tree BP runs rendered.")
        return

    bp_correction = shot.get("bp_corrections")
    if bp_correction is None:
        bp_correction = bp.decode(syndrome, args.max_iter)
    ml_correction = shot.get("ml_corrections")
    true_errors = shot.get("true_errors")

    marginals = []
    decisions = []
    roots = []
    for detection in active_detections:
        path = output_dir / f"tree_bp_detection_{detection:04d}.png"
        result = plot_tree_bp_marginals(
            bp,
            syndrome,
            root_check=detection,
            output_path=path,
            max_depth=args.max_depth,
            bp_correction=bp_correction,
            ml_correction=ml_correction,
            true_errors=true_errors,
            show_labels=not args.no_labels,
        )
        roots.append(detection)
        marginals.append(result["marginals"])
        decisions.append(result["decision"])

    marginals_arr = np.stack(marginals, axis=0)
    decisions_arr = np.stack(decisions, axis=0)
    data_path = output_dir / "tree_bp_results.npz"
    np.savez_compressed(
        data_path,
        roots=np.asarray(roots, dtype=np.int64),
        marginals=marginals_arr,
        decisions=decisions_arr,
        syndrome=syndrome.astype(np.uint8),
        bp_correction=np.asarray(bp_correction, dtype=np.uint8),
        ml_correction=(
            np.asarray(ml_correction, dtype=np.uint8)
            if ml_correction is not None else np.asarray([], dtype=np.uint8)
        ),
        true_errors=(
            np.asarray(true_errors, dtype=np.uint8)
            if true_errors is not None else np.asarray([], dtype=np.uint8)
        ),
        metadata_json=np.asarray(json.dumps({
            "code": code_name,
            "p": float(p),
            "shot_index": args.shot_index,
            "max_depth": args.max_depth,
            "max_iter": args.max_iter,
            "source_cache": str(cache_path) if cache_path is not None else None,
        }, sort_keys=True)),
    )

    visible_counts = np.sum(np.isfinite(marginals_arr), axis=1)
    finite_masks = np.isfinite(marginals_arr)
    disagreement_counts = np.sum(
        (decisions_arr != bp_correction[None, :]) & finite_masks,
        axis=1,
    )
    print(f"Code              : {code}")
    print(f"Noise p           : {float(p):.3%}")
    print(f"Syndrome wt       : {int(syndrome.sum())}")
    print(f"Active detections : {active_detections}")
    print(f"Max depth         : {args.max_depth if args.max_depth is not None else 'full'}")
    print(f"Output dir        : {output_dir}")
    print(f"Data              : {data_path}")
    for root, visible, disagrees in zip(roots, visible_counts, disagreement_counts):
        print(
            f"  root d{root}: visible_data={int(visible)}  "
            f"tree_vs_simple_bp_bits={int(disagrees)}"
        )


if __name__ == "__main__":
    main()
