"""
Render Tanner distance from selected detector nodes to data/error nodes.

Usage:
    python evaluations/detector_distance.py --code steane --detector 0
    python evaluations/detector_distance.py --code steane --p 0.15
    python evaluations/detector_distance.py --cache results/errors/steane_0p15.5.npz
    python evaluations/detector_distance.py --code hbb --detector 3 --detector 7
"""

import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import plot_detector_distance_graph
from _common import CODES, layout_for_code, load_cached_shot, resolve_code_and_p


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", choices=CODES.keys(), default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed used when sampling a syndrome")
    parser.add_argument("--max-attempts", type=int, default=100,
                        help="resample this many times looking for an on parity node")
    parser.add_argument("--cache", type=str, default=None,
                        help="NPZ cache from cache_bp_ml_failures.py")
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--detector", type=int, action="append", default=None,
                        help="source detector index; repeat for multiple")
    parser.add_argument("--syndrome", type=str, default=None,
                        help="bit string of active detectors; used as sources "
                             "when --detector is omitted")
    parser.add_argument("--output", type=str, default=None,
                        help="output PNG path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory when --output is not provided")
    parser.add_argument("--no-labels", action="store_true",
                        help="hide data-node index/distance labels")
    return parser.parse_args()


def parse_syndrome(text: str | None, n: int) -> np.ndarray | None:
    if text is None:
        return None
    bits = np.array([1 if ch == "1" else 0 for ch in text.strip()], dtype=np.uint8)
    if bits.size != n:
        raise ValueError(f"syndrome length {bits.size} does not match {n} detectors")
    return bits


def main():
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else None
    metadata = {}
    shot = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)
    code = CODES[code_name]()
    dem = code.to_dem(p)
    bp = flashbp.FlashBP(dem, DecoderConfig())
    syndrome = parse_syndrome(args.syndrome, bp.num_detectors)
    if syndrome is None and "syndrome" in shot:
        syndrome = shot["syndrome"]
    if syndrome is None and args.detector is None:
        if args.seed is not None:
            np.random.seed(args.seed)
        sampler = dem.compile_sampler()
        for _ in range(args.max_attempts):
            det, _, _ = sampler.sample(shots=1)
            candidate = det[0].astype(np.uint8)
            if candidate.sum():
                syndrome = candidate
                break
        if syndrome is None:
            raise ValueError(
                "sampled no nonzero syndrome; pass --detector, --syndrome, "
                "or increase --max-attempts / p"
            )
    if syndrome is not None and args.detector is None and not syndrome.sum():
        raise ValueError(
            "syndrome has no on parity nodes; pass --detector explicitly"
        )
    layout = layout_for_code(code)
    output = (
        Path(args.output)
        if args.output
        else Path(
            args.output_dir
            or (
                f"results/distance/{cache_path.stem}_{args.shot_index}"
                if cache_path is not None
                else f"results/distance/{code_name}_p{str(p).replace('.', 'p')}"
            )
        ) / "detector_distance.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    distances = plot_detector_distance_graph(
        bp,
        output_path=output,
        detectors=args.detector,
        syndrome=syndrome,
        layout=layout,
        show_labels=not args.no_labels,
    )

    finite = distances[distances >= 0]
    print(f"Code      : {code}")
    print(f"Detectors : {bp.num_detectors} check  +  {bp.num_errors} variable")
    if args.detector is not None:
        sources = args.detector
    elif syndrome is not None:
        sources = np.flatnonzero(syndrome).astype(int).tolist()
    else:
        sources = "default"
    print(f"Sources   : {sources}")
    if finite.size:
        print(f"Distance  : {int(finite.min())}..{int(finite.max())} Tanner hops")
    print(f"Output    : {output}")


if __name__ == "__main__":
    main()
