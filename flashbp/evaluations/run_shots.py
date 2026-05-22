"""
Run N shots of BP decoding and report the logical error rate.

Usage:
    python evaluations/run_shots.py --code steane --p 0.05 --shots 1000
    python evaluations/run_shots.py --code gross  --p 0.01 --shots 5000 --decoder simple
"""

import argparse
import numpy as np
from _common import (
    CODES,
    logical_prediction,
    make_decoder_runner,
    parse_decoder_spec,
)

DECODERS = [
    "simple",
    "tensor",
    "gbp",
    "gbp-check",
    "gbp-cycles",
    "gbp-cycles-any",
    "gbp-cycles-all",
    "gbp-union-cycles",
    "gbp-union-cycles-any",
    "gbp-union-cycles-all",
    "degree",
    "ml",
    "bp-osd",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code",     choices=CODES.keys(),    default="steane")
    parser.add_argument("--decoder",  type=str,                default="simple",
                        help="decoder spec, e.g. simple, ml, gbp-cycles:8, bp-osd:0")
    parser.add_argument("--p",        type=float,              default=0.05,
                        help="physical error rate")
    parser.add_argument("--shots",    type=int,                default=1000)
    parser.add_argument("--max-iter", type=int,                default=100,
                        help="BP max iterations")
    return parser.parse_args()


def main():
    args = parse_args()

    code   = CODES[args.code]()
    dem    = code.to_dem(args.p)
    spec = parse_decoder_spec(args.decoder)
    bp = make_decoder_runner(dem, spec, args.max_iter)

    print(f"Code        : {code}")
    print(f"Decoder     : {spec.label}")
    print(f"Detectors   : {bp.num_detectors}  ({code.H_X.shape[0]} X + {code.H_Z.shape[0]} Z)")
    print(f"Error mechs : {bp.num_errors}")
    print(f"Noise p     : {args.p:.3%}")
    print(f"Shots       : {args.shots}")

    sampler = dem.compile_sampler()
    det_data, obs_data, _ = sampler.sample(shots=args.shots)

    n_correct = 0
    for syndrome, true_obs in zip(det_data, obs_data):
        error_vec = bp.decode(syndrome.astype(np.uint8), args.max_iter)
        if hasattr(bp, "flush"):
            bp.flush()
        pred_obs = logical_prediction(bp, error_vec)
        if np.array_equal(pred_obs, true_obs.astype(np.int32)):
            n_correct += 1

    logical_error_rate = 1.0 - n_correct / args.shots
    print(f"\nLogical error rate : {logical_error_rate:.4%}")


if __name__ == "__main__":
    main()
