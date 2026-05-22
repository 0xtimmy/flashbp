"""
Find shots where one decoder fails and another decoder succeeds, then cache
them for reuse.

Examples:
    python evaluations/cache_bp_ml_failures.py --code steane --p 0.05 --target 20
    python evaluations/cache_bp_ml_failures.py --code smbb --p 0.08 --target 50 --force
    python evaluations/cache_bp_ml_failures.py --code steane --p 0.05 \\
        --fail-decoder simple --success-decoder gbp-cycles:8

The cache is a compressed NPZ file containing:
    syndromes, true_obs, true_errors, fail_corrections, success_corrections,
    fail_pred_obs, success_pred_obs, and attempt_indices.

For the default simple-vs-ML case, legacy aliases are also written:
    bp_corrections, ml_corrections, bp_pred_obs, ml_pred_obs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (
    CODES,
    logical_prediction,
    make_decoder_runner,
    parse_decoder_spec,
)


ARRAY_KEYS = (
    "syndromes",
    "true_obs",
    "true_errors",
    "fail_corrections",
    "success_corrections",
    "fail_pred_obs",
    "success_pred_obs",
    "fail_converged",
    "success_converged",
    "bp_corrections",
    "ml_corrections",
    "bp_pred_obs",
    "ml_pred_obs",
    "attempt_indices",
)


def safe_label(text: str) -> str:
    return (
        text.replace(":", "-")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", choices=CODES.keys(), default="steane")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--target", type=int, default=25,
                        help="number of fail-decoder-fail/success-decoder-success shots to cache")
    parser.add_argument("--mode", choices=("logical", "convergence"),
                        default="logical",
                        help="logical: fail/succeed by observable correctness; "
                             "convergence: fail/succeed by syndrome satisfaction")
    parser.add_argument("--max-attempts", type=int, default=10000,
                        help="stop after this many sampled shots")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="sample shots in batches of this size")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="default max iterations for iterative decoders")
    parser.add_argument("--fail-decoder", type=str, default="simple",
                        help="decoder that must fail logically; supports threshold GBP keys")
    parser.add_argument("--success-decoder", type=str, default="ml",
                        help="decoder that must succeed logically; supports threshold GBP keys")
    parser.add_argument("--fail-max-iter", type=int, default=None,
                        help="max iterations for --fail-decoder; defaults to --max-iter")
    parser.add_argument("--success-max-iter", type=int, default=None,
                        help="max iterations for --success-decoder; defaults to 1 for ML, else --max-iter")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="NPZ output path; defaults under results/errors/")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing cache instead of appending")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="print progress after this many attempts")
    return parser.parse_args()


def default_output_path(code: str, p: float, target: int) -> Path:
    p_text = f"{p:.6g}".replace(".", "p").replace("-", "m")
    return Path("results") / "errors" / f"{code}_{p_text}.{target}.npz"


def generic_output_path(
    code: str,
    p: float,
    target: int,
    fail_label: str,
    success_label: str,
) -> Path:
    if fail_label == "simple" and success_label == "ml":
        return default_output_path(code, p, target)
    p_text = f"{p:.6g}".replace(".", "p").replace("-", "m")
    return (
        Path("results")
        / "errors"
        / f"{code}_{p_text}.{safe_label(fail_label)}_fail."
          f"{safe_label(success_label)}_success.{target}.npz"
    )


def output_path_for_mode(
    code: str,
    p: float,
    target: int,
    fail_label: str,
    success_label: str,
    mode: str,
) -> Path:
    path = generic_output_path(code, p, target, fail_label, success_label)
    if mode == "logical":
        return path
    return path.with_name(path.stem + f".{mode}" + path.suffix)


def empty_cache() -> dict[str, list]:
    return {key: [] for key in ARRAY_KEYS}


def load_existing(path: Path) -> tuple[dict[str, list], dict]:
    if not path.exists():
        return empty_cache(), {}
    data = np.load(path, allow_pickle=False)
    cache = empty_cache()
    for key in ARRAY_KEYS:
        if key in data:
            cache[key] = [row.copy() for row in data[key]]

    # Backfill generic fields when appending to older simple-vs-ML caches.
    aliases = {
        "fail_corrections": "bp_corrections",
        "success_corrections": "ml_corrections",
        "fail_pred_obs": "bp_pred_obs",
        "success_pred_obs": "ml_pred_obs",
    }
    for key, old_key in aliases.items():
        if not cache[key] and cache[old_key]:
            cache[key] = [row.copy() for row in cache[old_key]]
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    return cache, metadata


def append_case(
    cache: dict[str, list],
    syndrome: np.ndarray,
    true_obs: np.ndarray,
    true_errors: np.ndarray,
    fail_correction: np.ndarray,
    success_correction: np.ndarray,
    fail_pred_obs: np.ndarray,
    success_pred_obs: np.ndarray,
    fail_converged: bool,
    success_converged: bool,
    attempt_index: int,
    write_legacy_aliases: bool,
) -> None:
    cache["syndromes"].append(syndrome.astype(np.uint8, copy=True))
    cache["true_obs"].append(true_obs.astype(np.uint8, copy=True))
    cache["true_errors"].append(true_errors.astype(np.uint8, copy=True))
    cache["fail_corrections"].append(fail_correction.astype(np.uint8, copy=True))
    cache["success_corrections"].append(success_correction.astype(np.uint8, copy=True))
    cache["fail_pred_obs"].append(fail_pred_obs.astype(np.uint8, copy=True))
    cache["success_pred_obs"].append(success_pred_obs.astype(np.uint8, copy=True))
    cache["fail_converged"].append(np.asarray(fail_converged, dtype=np.uint8))
    cache["success_converged"].append(np.asarray(success_converged, dtype=np.uint8))
    if write_legacy_aliases:
        cache["bp_corrections"].append(fail_correction.astype(np.uint8, copy=True))
        cache["ml_corrections"].append(success_correction.astype(np.uint8, copy=True))
        cache["bp_pred_obs"].append(fail_pred_obs.astype(np.uint8, copy=True))
        cache["ml_pred_obs"].append(success_pred_obs.astype(np.uint8, copy=True))
    cache["attempt_indices"].append(np.asarray(attempt_index, dtype=np.int64))


def stack_cache(cache: dict[str, list]) -> dict[str, np.ndarray]:
    arrays = {}
    for key in ARRAY_KEYS:
        values = cache[key]
        if values:
            arrays[key] = np.stack(values, axis=0)
        else:
            dtype = np.int64 if key == "attempt_indices" else np.uint8
            arrays[key] = np.asarray([], dtype=dtype)
    return arrays


def save_cache(path: Path, cache: dict[str, list], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = stack_cache(cache)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(
        tmp_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    tmp_npz = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if tmp_npz.exists():
        tmp_npz.replace(path)
    else:
        tmp_path.replace(path)


def forced_converged(label: str, spec) -> bool:
    if spec.backend == "ldpc_bp_osd":
        return True
    if spec.config is not None and spec.config.decoder in ("ml", "maximum_likelihood"):
        return True
    return False


def syndrome_satisfied(decoder, syndrome: np.ndarray, correction: np.ndarray) -> bool:
    predicted = (decoder.H @ correction.astype(np.int32)) % 2
    return bool(np.array_equal(predicted.astype(np.uint8), syndrome.astype(np.uint8)))


def main():
    args = parse_args()
    fail_spec = parse_decoder_spec(args.fail_decoder)
    success_spec = parse_decoder_spec(args.success_decoder)
    fail_label = fail_spec.label
    success_label = success_spec.label
    fail_max_iter = args.fail_max_iter if args.fail_max_iter is not None else args.max_iter
    if args.success_max_iter is not None:
        success_max_iter = args.success_max_iter
    elif success_spec.config is not None and success_spec.config.decoder in ("ml", "maximum_likelihood"):
        success_max_iter = 1
    else:
        success_max_iter = args.max_iter

    output_path = (
        Path(args.output)
        if args.output
        else output_path_for_mode(
            args.code, args.p, args.target, fail_label, success_label, args.mode)
    )

    code = CODES[args.code]()
    dem = code.to_dem(args.p)
    sampler = dem.compile_sampler()
    rng = np.random.default_rng(args.seed)

    if output_path.exists() and not args.force:
        cache, previous_metadata = load_existing(output_path)
        print(f"Appending to existing cache: {output_path}")
        if previous_metadata:
            old_code = previous_metadata.get("code")
            old_p = previous_metadata.get("p")
            old_fail = previous_metadata.get(
                "fail_decoder", previous_metadata.get("bp_decoder"))
            old_success = previous_metadata.get(
                "success_decoder", previous_metadata.get("ml_decoder"))
            if (
                old_code != args.code
                or float(old_p) != float(args.p)
                or old_fail != fail_label
                or old_success != success_label
                or previous_metadata.get("mode", "logical") != args.mode
            ):
                raise ValueError(
                    "Existing cache metadata does not match requested code/p/decoders. "
                    "Use --force or choose a different --output."
                )
    else:
        cache = empty_cache()

    fail_decoder = make_decoder_runner(dem, fail_spec, fail_max_iter)
    success_decoder = make_decoder_runner(dem, success_spec, success_max_iter)
    write_legacy_aliases = fail_label == "simple" and success_label == "ml"
    fail_always_converged = forced_converged(fail_label, fail_spec)
    success_always_converged = forced_converged(success_label, success_spec)

    found_start = len(cache["syndromes"])
    attempts = 0
    fail_decoder_failures = 0
    success_decoder_successes_on_failure = 0
    t0 = time.time()

    print(f"Code        : {code}")
    print(f"Noise p     : {args.p:.3%}")
    print(f"Fail decoder    : {fail_label}  max_iter={fail_max_iter}")
    print(f"Success decoder : {success_label}  max_iter={success_max_iter}")
    print(f"Mode            : {args.mode}")
    print(f"Target      : {args.target} cached cases")
    print(f"Starting    : {found_start} existing cases")
    print(f"Output      : {output_path}")

    while len(cache["syndromes"]) < args.target and attempts < args.max_attempts:
        batch = min(args.batch_size, args.max_attempts - attempts)
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        np.random.seed(seed)
        det_data, obs_data, err_data = sampler.sample(
            shots=batch,
            return_errors=True,
        )

        for row in range(batch):
            attempts += 1
            syndrome = det_data[row].astype(np.uint8)
            true_obs = obs_data[row].astype(np.uint8)
            true_errors = err_data[row].astype(np.uint8)

            fail_correction = fail_decoder.decode(syndrome, fail_max_iter)
            fail_pred_obs = logical_prediction(
                fail_decoder, fail_correction).astype(np.uint8)
            fail_converged = (
                True
                if fail_always_converged
                else syndrome_satisfied(fail_decoder, syndrome, fail_correction)
            )
            if args.mode == "logical":
                fail_ok = np.array_equal(fail_pred_obs, true_obs)
            else:
                fail_ok = fail_converged
            if fail_ok:
                continue
            fail_decoder_failures += 1

            success_correction = success_decoder.decode(syndrome, success_max_iter)
            success_pred_obs = logical_prediction(
                success_decoder, success_correction).astype(np.uint8)
            success_converged = (
                True
                if success_always_converged
                else syndrome_satisfied(success_decoder, syndrome, success_correction)
            )
            if args.mode == "logical":
                success_ok = np.array_equal(success_pred_obs, true_obs)
            else:
                success_ok = success_converged
            if not success_ok:
                continue
            success_decoder_successes_on_failure += 1

            append_case(
                cache,
                syndrome,
                true_obs,
                true_errors,
                fail_correction,
                success_correction,
                fail_pred_obs,
                success_pred_obs,
                fail_converged,
                success_converged,
                attempts,
                write_legacy_aliases,
            )
            print(
                f"  cached {len(cache['syndromes'])}/{args.target} "
                f"at attempt {attempts} "
                f"(syndrome_wt={int(syndrome.sum())}, "
                f"true_err_wt={int(true_errors.sum())})"
            )
            if len(cache["syndromes"]) >= args.target:
                break

        if args.progress_every > 0 and attempts % args.progress_every < batch:
            elapsed = time.time() - t0
            print(
                f"attempts={attempts}  cached={len(cache['syndromes'])}  "
                f"{fail_label}_failures={fail_decoder_failures}  "
                f"fail_success_hits={success_decoder_successes_on_failure}  "
                f"elapsed={elapsed:.1f}s"
            )

    metadata = {
        "code": args.code,
        "code_repr": str(code),
        "p": args.p,
        "fail_decoder": fail_label,
        "success_decoder": success_label,
        "mode": args.mode,
        "fail_decoder_config": fail_spec.config.__dict__ if fail_spec.config else None,
        "success_decoder_config": success_spec.config.__dict__ if success_spec.config else None,
        "fail_max_iter": fail_max_iter,
        "success_max_iter": success_max_iter,
        "bp_decoder": "simple" if write_legacy_aliases else None,
        "ml_decoder": "ml" if write_legacy_aliases else None,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "target": args.target,
        "max_attempts": args.max_attempts,
        "attempts_this_run": attempts,
        "existing_cases_at_start": found_start,
        "num_cases": len(cache["syndromes"]),
        "num_detectors": int(dem.num_detectors),
        "num_observables": int(dem.num_observables),
        "num_errors": int(fail_decoder.num_errors),
        "created_by": "evaluations/cache_bp_ml_failures.py",
    }
    save_cache(output_path, cache, metadata)

    elapsed = time.time() - t0
    print()
    print(f"Attempts this run      : {attempts}")
    print(f"{fail_label} {args.mode} failures : {fail_decoder_failures}")
    print(f"Failure + {success_label} {args.mode} success : {success_decoder_successes_on_failure}")
    print(f"Cached total           : {len(cache['syndromes'])}")
    print(f"Elapsed                : {elapsed:.1f}s")
    print(f"Wrote                  : {output_path}")


if __name__ == "__main__":
    main()
