"""
Render the final BP recorded iteration for one cached or sampled syndrome.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.animation import bipartite_layout, render_frame
from _common import (
    CODES,
    layout_for_code,
    load_cached_shot,
    p_token,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else None
    shot = {}
    metadata = {}
    if cache_path is not None:
        shot, metadata = load_cached_shot(cache_path, args.shot_index)
    code_name, p = resolve_code_and_p(args, metadata)
    code = CODES[code_name]()
    dem = code.to_dem(p)
    bp = flashbp.FlashBP(
        dem,
        DecoderConfig(
            decoder="simple",
            log=True,
            log_type="record",
            log_level=0,
            log_console=False,
        ),
    )

    syndrome, true_obs, true_errors = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_errors is None:
        true_errors = np.zeros(bp.num_errors, dtype=np.uint8)

    result = bp.decode(syndrome, args.max_iter)
    bp.flush()
    recording = bp.get_recording()
    final_iter = recording[-1]["iterations"][-1]
    H = np.asarray(bp.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    layout = layout_for_code(code) or bipartite_layout(num_vars, num_checks)
    output = Path(
        args.output
        or (
            f"results/final_bp/{code_name}_{p_token(p)}.{cache_path.stem}_{args.shot_index}.png"
            if cache_path is not None
            else f"results/final_bp/{code_name}_{p_token(p)}.final_bp.png"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    render_frame(final_iter, H, layout, output, true_errors=true_errors)

    pred_obs = (bp.L @ result.astype(np.int32)) % 2
    print(f"Code        : {code}")
    print(f"Noise p     : {p:.3%}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    if true_obs is not None:
        print(f"True obs    : {true_obs}    Pred obs: {pred_obs}")
    print(f"Output      : {output}")


if __name__ == "__main__":
    main()
