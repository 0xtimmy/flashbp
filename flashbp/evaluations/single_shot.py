"""
Decode one shot of the Steane code with RecordLogger and print per-iteration data.

Usage:
    python evaluations/single_shot.py
    python evaluations/single_shot.py --p 0.15 --max-iter 20
"""

import argparse
import numpy as np
import flashbp
from flashbp import DecoderConfig
from flashbp.codes import SteaneCode
from _common import load_cached_shot, sample_or_cached_shot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p",        type=float, default=None)
    parser.add_argument("--cache",    type=str, default=None)
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--syndrome", type=str, default=None)
    parser.add_argument("--max-iter", type=int,   default=20)
    parser.add_argument("--seed",     type=int,   default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    shot = {}
    metadata = {}
    cache_path = args.cache
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    p = args.p if args.p is not None else metadata.get("p", 0.1)

    code = SteaneCode()
    dem  = code.to_dem(p)
    cfg = DecoderConfig(
        log=True,
        log_type="record",
        log_level=3,
        log_console=True,
    )
    bp = flashbp.FlashBP(dem, cfg)

    syndrome, true_obs, _ = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=False)
    if true_obs is None:
        true_obs = np.zeros(bp.num_observables, dtype=np.uint8)

    print(f"Code        : {code}")
    print(f"Noise p     : {p:.1%}")
    print(f"Syndrome    : {syndrome}  (weight {syndrome.sum()})")

    result   = bp.decode(syndrome, args.max_iter)
    pred_obs = (bp.L @ result.astype(np.int32)) % 2

    print(f"Decision    : {result}  (weight {result.sum()})")
    print(f"True obs    : {true_obs}")
    print(f"Pred obs    : {pred_obs}")
    print(f"Correct     : {np.array_equal(pred_obs, true_obs)}")

    recording = bp.get_recording()
    shot = recording[0]
    print(f"\nIterations recorded: {len(shot['iterations'])}")
    for it in shot["iterations"]:
        n      = it["iteration"]
        dec_w  = int(it["decision"].sum())
        v2c_l1 = float(np.abs(it["msg_v2c"]).mean())
        c2v_l1 = float(np.abs(it["msg_c2v"]).mean())
        print(f"  iter={n}  decision_weight={dec_w}"
              f"  |v2c|_mean={v2c_l1:.3f}  |c2v|_mean={c2v_l1:.3f}")


if __name__ == "__main__":
    main()
