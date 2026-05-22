"""
Render the final ML contraction frame for one cached or sampled syndrome.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.animation.contraction import (
    _ml_recording_class_distributions,
    _ml_recording_heatmap_distributions,
    _ml_recording_prefix_distributions,
    _posterior_error_probs_for_syndrome,
    render_ml_contraction_frame,
)
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
    parser.add_argument("--posterior-max-bits", type=int, default=22)
    parser.add_argument("--max-bars", type=int, default=64)
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
            decoder="ml",
            log=True,
            log_type="ml",
            log_level=0,
            log_console=False,
        ),
    )

    syndrome, true_obs, true_errors = sample_or_cached_shot(
        dem, args, cache_path, shot, want_errors=True)
    if true_errors is None:
        true_errors = np.zeros(bp.num_errors, dtype=np.uint8)

    result = bp.decode(syndrome, 1)
    bp.flush()
    recording = bp.get_recording()
    ml_shot = recording[-1]
    steps = ml_shot["steps"]

    H = np.asarray(bp.H, dtype=np.uint8)
    L = np.asarray(bp.L, dtype=np.uint8)
    error_probs = np.asarray(bp.error_probs, dtype=np.float64)
    layout = layout_for_code(code)
    if layout is None:
        from flashbp.animation import bipartite_layout
        num_checks, num_vars = H.shape
        layout = bipartite_layout(num_vars, num_checks)

    prefix_distributions = _ml_recording_prefix_distributions(ml_shot)
    step_class_log_probs = _ml_recording_class_distributions(ml_shot)
    heatmap_distributions = _ml_recording_heatmap_distributions(
        H, L, ml_shot, prefix_distributions
    )
    class_log_probs = np.asarray(ml_shot.get("class_log_probs", []), dtype=np.float64)
    if class_log_probs.size == 0:
        class_log_probs = None
    order = [int(step["error_idx"]) for step in steps[1:]]
    posterior_probs = _posterior_error_probs_for_syndrome(
        H, error_probs, syndrome, max_bits=args.posterior_max_bits
    )

    output = Path(
        args.output
        or (
            f"results/final_ml/{cache_path.stem}_{args.shot_index}.png"
            if cache_path is not None
            else f"results/final_ml/{code_name}_p{p_token(p)}.png"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    final_idx = len(prefix_distributions) - 1
    render_ml_contraction_frame(
        H,
        L,
        error_probs,
        syndrome,
        layout,
        output,
        contracted_count=final_idx,
        order=order,
        prefix_distributions=prefix_distributions,
        posterior_probs=posterior_probs,
        max_bars=args.max_bars,
        resolved_device=str(ml_shot.get("device", "cpp")),
        true_errors=true_errors,
        predicted_errors=result,
        class_log_probs=class_log_probs,
        step_class_log_probs=step_class_log_probs[final_idx],
        heatmap_distributions=heatmap_distributions[final_idx],
    )

    pred_obs = (bp.L @ result.astype(np.int32)) % 2
    print(f"Code        : {code}")
    print(f"Noise p     : {p:.3%}")
    print(f"Syndrome wt : {int(syndrome.sum())}")
    if true_obs is not None:
        print(f"True obs    : {true_obs}    Pred obs: {pred_obs}")
    print(f"Output      : {output}")


if __name__ == "__main__":
    main()
