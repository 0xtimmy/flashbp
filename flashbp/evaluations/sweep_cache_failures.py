"""
Run cache_failures.py over a grid of fail/success decoder pairs.

Defaults compare simple + GBP variants as the failing decoder against ML and
BP+OSD as the succeeding decoder.

Example:
    python evaluations/sweep_cache_failures.py --code steane --p 0.05 --target 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import parse_decoder_spec, p_token


DEFAULT_FAIL_DECODERS = (
    "simple",
    "gbp-check:2",
    "gbp-cycles:8",
    "gbp-cycles-any:8",
    "gbp-cycles-all:8",
    "gbp-union-cycles:8",
    "gbp-union-cycles-any:8",
    "gbp-union-cycles-all:8",
)

DEFAULT_SUCCESS_DECODERS = (
    "ml",
    "bp-osd:0",
)


def safe_label(text: str) -> str:
    return (
        text.replace(":", "-")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
    )


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def cached_case_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if "syndromes" not in data:
            return 0
        return int(data["syndromes"].shape[0])
    except Exception as exc:
        print(f"WARNING: could not inspect existing cache {path}: {exc}")
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--target", type=int, default=25)
    parser.add_argument("--mode", choices=("logical", "convergence"),
                        default="logical")
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-decoders", type=str,
                        default=",".join(DEFAULT_FAIL_DECODERS),
                        help="comma-separated fail decoder specs")
    parser.add_argument("--success-decoders", type=str,
                        default=",".join(DEFAULT_SUCCESS_DECODERS),
                        help="comma-separated success decoder specs")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="pass --force to each cache_failures.py call")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without running them")
    return parser.parse_args()


def main():
    args = parse_args()
    fail_decoders = parse_csv(args.fail_decoders)
    success_decoders = parse_csv(args.success_decoders)
    output_dir = Path(
        args.output_dir
        or (Path("results") / "errors" / "sweeps" / f"{args.code}_p{p_token(args.p)}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).with_name("cache_failures.py")
    total = len(fail_decoders) * len(success_decoders)
    print(f"Code        : {args.code}")
    print(f"Noise p     : {args.p:.3%}")
    print(f"Fail set    : {fail_decoders}")
    print(f"Success set : {success_decoders}")
    print(f"Jobs        : {total}")
    print(f"Output dir  : {output_dir}")

    failures = 0
    skipped = 0
    job = 0
    for fail in fail_decoders:
        fail_label = parse_decoder_spec(fail).label
        for success in success_decoders:
            success_label = parse_decoder_spec(success).label
            if fail_label == success_label:
                print(f"\nSkipping identical pair: {fail_label}")
                continue

            job += 1
            output = (
                output_dir
                / f"{safe_label(fail_label)}_fail."
                  f"{safe_label(success_label)}_success.{args.target}.npz"
            )
            cmd = [
                sys.executable,
                str(script),
                "--code", args.code,
                "--p", str(args.p),
                "--target", str(args.target),
                "--mode", args.mode,
                "--max-attempts", str(args.max_attempts),
                "--batch-size", str(args.batch_size),
                "--max-iter", str(args.max_iter),
                "--seed", str(args.seed),
                "--fail-decoder", fail,
                "--success-decoder", success,
                "--output", str(output),
            ]
            if args.force:
                cmd.append("--force")

            print(f"\n[{job}/{total}] {fail_label} fails, {success_label} succeeds")
            existing_count = cached_case_count(output)
            if existing_count is not None and not args.force:
                if existing_count >= args.target:
                    skipped += 1
                    print(
                        f"Skipping existing complete cache: {output} "
                        f"({existing_count}/{args.target} cases)"
                    )
                    continue
                print(
                    f"Resuming partial cache: {output} "
                    f"({existing_count}/{args.target} cases)"
                )
            print(" ".join(cmd))
            if args.dry_run:
                continue

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                failures += 1
                print(f"  failed with exit code {result.returncode}")

    if failures:
        raise SystemExit(f"{failures} cache sweep job(s) failed")
    if skipped:
        print(f"\nSkipped {skipped} complete cache file(s).")


if __name__ == "__main__":
    main()
