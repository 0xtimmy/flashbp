"""
Sweep the logical error rate of a code across a range of physical error
rates, for one or more decoders, and plot the result.

Decoder specs (CLI ``--decoders``) use a compact syntax, comma-separated:

    simple              SimpleDecoder
    degree:2            DegreeDecoder with degree=2
    tensor:3            TensorDecoder with degree=3
    bp-osd:0            Roffe ldpc BP+OSD with OSD order 0
    gbp-check:2         GBPDecoder with check-neighborhood regions of degree=2
    gbp-cycles:8        GBPDecoder with all cycles of length <= 8
    gbp-cycles-any:8    GBPDecoder with cycles where any check is active
    gbp-cycles-all:8    GBPDecoder with cycles where all checks are active
    gbp-union-cycles:8  GBPDecoder with overlapping cycles unioned into regions
    gbp:8:<policy>      Explicit GBP region policy, e.g. gbp:8:short_cycles
    ml                  MaximumLikelihoodDecoder

Examples:
    python evaluations/threshold_sweep.py \\
        --code steane \\
        --decoders simple,gbp-check:2,gbp-cycles:8,gbp-cycles-any:8,ml \\
        --p-min 0.005 --p-max 0.1 --num-p 8 --shots 500

    python evaluations/threshold_sweep.py \\
        --code surface_5 \\
        --decoders simple,degree:2,degree:3 \\
        --p-min 5e-4 --p-max 0.03 --num-p 8 --shots 2000 --log-x --log-y

Outputs:
    <output-dir>/sweep.png          log-log plot
    <output-dir>/time_per_shot.png  average decoder time per shot plot
    <output-dir>/sweep.csv          resumable raw rows
    <output-dir>/samples.npz        sampled syndromes/observables reused by runs
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import CODES, make_decoder_runner, parse_decoder_spec

try:
    from tqdm.auto import tqdm
except ImportError:  # keep the evaluation usable in minimal environments
    tqdm = None


CSV_COLUMNS = [
    "decoder",
    "p",
    "logical_err",
    "logical_err_stderr",
    "shots",
    "time_per_shot_s",
    "converged_shots",
    "converged_time_per_shot_s",
]


def confirm_existing_path(path: Path, description: str, force: bool) -> bool:
    if force or not path.exists():
        return True
    reply = input(f"{description} exists at '{path}'. Overwrite/update it? [y/N]: ")
    return reply.strip().lower() in ("y", "yes")


def decoder_always_converged(spec) -> bool:
    if spec.backend == "ldpc_bp_osd":
        return True
    if spec.config is not None and spec.config.decoder in ("ml", "maximum_likelihood"):
        return True
    return False


def syndrome_satisfied(decoder, syndrome: np.ndarray, correction: np.ndarray) -> bool:
    predicted = (decoder.H @ correction.astype(np.int32)) % 2
    return bool(np.array_equal(predicted.astype(np.uint8), syndrome.astype(np.uint8)))


def p_key(p: float) -> str:
    return f"{float(p):.12g}"


def logical_err_stderr(err: float, shots: int) -> float:
    if not np.isfinite(err) or shots <= 0:
        return float("nan")
    # Continuity correction keeps zero-failure runs visibly less certain than
    # infinite data, especially on log plots.
    p_hat = float(np.clip(err, 0.5 / shots, 1.0 - 0.5 / shots))
    return float(np.sqrt(p_hat * (1.0 - p_hat) / shots))


def read_sweep_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "logical_err_stderr" not in row or row["logical_err_stderr"] == "":
            try:
                err = float(row["logical_err"])
                shots = int(row["shots"])
                row["logical_err_stderr"] = f"{logical_err_stderr(err, shots):.9g}"
            except (KeyError, TypeError, ValueError):
                row["logical_err_stderr"] = ""
        for column in CSV_COLUMNS:
            row.setdefault(column, "")
    return rows


def write_sweep_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda r: (
        r.get("decoder", ""),
        float(r.get("p", "nan")),
        int(r.get("shots", "0") or 0),
    ))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def completed_row_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        try:
            keys.add((row["decoder"], p_key(float(row["p"])), int(row["shots"])))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def load_sample_cache(path: Path, code_name: str, ps: np.ndarray) -> dict[str, object] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    cached_code = metadata.get("code")
    if cached_code is not None and cached_code != code_name:
        raise ValueError(
            f"sample cache {path} was generated for code {cached_code!r}, "
            f"not {code_name!r}"
        )
    cached_ps = np.asarray(data["p_values"], dtype=np.float64)
    if cached_ps.shape != ps.shape or not np.allclose(cached_ps, ps, rtol=1e-12, atol=1e-15):
        raise ValueError(
            f"sample cache {path} uses a different p grid; use a different "
            "output directory or rerun with the same p-min/p-max/num-p"
        )
    return {
        "metadata": metadata,
        "p_values": cached_ps,
        "det": np.asarray(data["det"], dtype=np.uint8),
        "obs": np.asarray(data["obs"], dtype=np.uint8),
    }


def save_sample_cache(
    path: Path,
    code_name: str,
    seed: int,
    ps: np.ndarray,
    det_by_p: list[np.ndarray],
    obs_by_p: list[np.ndarray],
) -> None:
    if not det_by_p:
        return
    max_shots = max(det.shape[0] for det in det_by_p)
    num_detectors = det_by_p[0].shape[1]
    num_observables = obs_by_p[0].shape[1]
    det = np.zeros((len(ps), max_shots, num_detectors), dtype=np.uint8)
    obs = np.zeros((len(ps), max_shots, num_observables), dtype=np.uint8)
    shot_counts = np.zeros(len(ps), dtype=np.int64)
    for i, (det_i, obs_i) in enumerate(zip(det_by_p, obs_by_p)):
        shot_counts[i] = det_i.shape[0]
        det[i, :det_i.shape[0], :] = det_i
        obs[i, :obs_i.shape[0], :] = obs_i
    metadata = {
        "code": code_name,
        "seed": int(seed),
        "shot_counts": shot_counts.tolist(),
    }
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata, sort_keys=True),
        p_values=np.asarray(ps, dtype=np.float64),
        det=det,
        obs=obs,
        shot_counts=shot_counts,
    )


def sample_for_p(dem, shots: int, seed: int, p_index: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    # Stim's sampler follows NumPy's global seed here, so make extensions
    # deterministic without replacing the existing cached prefix.
    np.random.seed(int(seed) + 1000003 * int(p_index) + int(offset))
    det, obs, _ = dem.compile_sampler().sample(shots=shots)
    return det.astype(np.uint8), obs.astype(np.uint8)


def ensure_sample_cache(
    output_dir: Path,
    code_name: str,
    code,
    ps: np.ndarray,
    seed: int,
    shots: int,
    force: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], Path]:
    cache_path = output_dir / "samples.npz"
    cache_existed = cache_path.exists()
    cached = load_sample_cache(cache_path, code_name, ps)
    if cached is None:
        det_by_p = []
        obs_by_p = []
    else:
        shot_counts = np.asarray(cached["metadata"].get("shot_counts", []), dtype=np.int64)
        det_array = np.asarray(cached["det"], dtype=np.uint8)
        obs_array = np.asarray(cached["obs"], dtype=np.uint8)
        if shot_counts.size != len(ps):
            shot_counts = np.full(len(ps), det_array.shape[1], dtype=np.int64)
        det_by_p = [det_array[i, :int(shot_counts[i]), :].copy() for i in range(len(ps))]
        obs_by_p = [obs_array[i, :int(shot_counts[i]), :].copy() for i in range(len(ps))]

    changed = cached is None
    for i, p in enumerate(ps):
        dem = code.to_dem(float(p))
        current = det_by_p[i].shape[0] if i < len(det_by_p) else 0
        if i >= len(det_by_p):
            det_i, obs_i = sample_for_p(dem, shots, seed, i, 0)
            det_by_p.append(det_i)
            obs_by_p.append(obs_i)
            changed = True
        elif current < shots:
            det_more, obs_more = sample_for_p(dem, shots - current, seed, i, current)
            det_by_p[i] = np.concatenate([det_by_p[i], det_more], axis=0)
            obs_by_p[i] = np.concatenate([obs_by_p[i], obs_more], axis=0)
            changed = True

    if changed:
        if cache_existed and not confirm_existing_path(
            cache_path,
            "Sample cache",
            force,
        ):
            print("Aborted before updating sample cache.")
            sys.exit(0)
        save_sample_cache(cache_path, code_name, seed, ps, det_by_p, obs_by_p)
    return det_by_p, obs_by_p, cache_path


def rows_by_decoder(rows: list[dict[str, str]]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        try:
            parsed = {
                "p": float(row["p"]),
                "logical_err": float(row["logical_err"]),
                "logical_err_stderr": float(row.get("logical_err_stderr", "nan")),
                "shots": int(row["shots"]),
                "time_per_shot_s": float(row["time_per_shot_s"]),
                "converged_shots": int(row["converged_shots"]),
                "converged_time_per_shot_s": float(row["converged_time_per_shot_s"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(row["decoder"], []).append(parsed)
    for pts in grouped.values():
        pts.sort(key=lambda pt: (pt["p"], pt["shots"]))
    return grouped


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--code",       choices=CODES.keys(), default="steane")
    p.add_argument("--decoders",   default="simple,ml",
                   help="comma-separated decoder specs")
    p.add_argument("--p-min",      type=float, default=0.0005)
    p.add_argument("--p-max",      type=float, default=0.1)
    p.add_argument("--num-p",      type=int,   default=8,
                   help="number of physical error rates (geometric spacing)")
    p.add_argument("--shots",      type=int,   default=1000,
                   help="shots per (decoder, p) point")
    p.add_argument("--max-iter",   type=int,   default=50)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--output-dir", type=str,   default=None)
    p.add_argument("--log-x",      action="store_true", default=True)
    p.add_argument("--log-y",      action="store_true", default=True)
    p.add_argument("--linear",     action="store_true",
                   help="override --log-x/--log-y to linear axes")
    p.add_argument("--no-progress", action="store_true",
                   help="disable tqdm progress bars")
    p.add_argument("--force",      action="store_true",
                   help="update/overwrite existing output files without prompting")
    return p.parse_args()


def main():
    args = parse_args()

    code_factory = CODES[args.code]
    code         = code_factory()

    decoder_specs = [parse_decoder_spec(s.strip())
                     for s in args.decoders.split(",") if s.strip()]

    ps = np.geomspace(args.p_min, args.p_max, args.num_p)

    output_dir = Path(args.output_dir or f"results/threshold/{args.code}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Code        : {code}")
    print(f"Decoders    : {[spec.label for spec in decoder_specs]}")
    print(f"p sweep     : {args.num_p} points  {args.p_min:g} .. {args.p_max:g}")
    print(f"Shots / pt  : {args.shots}")
    print(f"Output dir  : {output_dir}")
    print()

    csv_path = output_dir / "sweep.csv"
    existing_rows = read_sweep_rows(csv_path)
    done = completed_row_keys(existing_rows)
    had_sample_cache = (output_dir / "samples.npz").exists()
    det_by_p, obs_by_p, sample_cache_path = ensure_sample_cache(
        output_dir,
        args.code,
        code,
        ps,
        args.seed,
        args.shots,
        args.force,
    )
    print(f"Samples    : {sample_cache_path}")
    if existing_rows:
        print(f"Resume     : loaded {len(existing_rows)} existing sweep rows")
        if not had_sample_cache:
            print("WARNING: existing CSV had no samples.npz; old rows cannot be tied to saved syndromes.")
    print()

    new_rows: list[dict[str, str]] = []

    use_progress = tqdm is not None and not args.no_progress
    if not use_progress:
        if args.no_progress:
            reason = "--no-progress was passed"
        elif tqdm is None:
            reason = "tqdm is not installed"
        else:
            reason = "stderr is not an interactive terminal"
        print(f"WARNING: tqdm progress bars disabled ({reason}).")
    p_iter = tqdm(ps, desc="p sweep", unit="p") if use_progress else ps

    for p_index, p in enumerate(p_iter):
        dem     = code.to_dem(float(p))
        det = det_by_p[p_index][:args.shots]
        obs = obs_by_p[p_index][:args.shots]

        if use_progress:
            tqdm.write(f"p = {p:.4g}")
        else:
            print(f"p = {p:.4g}")
        for spec in decoder_specs:
            label = spec.label
            row_key = (label, p_key(float(p)), int(args.shots))
            if row_key in done:
                msg = f"  {label:18s}  cached ({args.shots} shots)"
                if use_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
                continue
            try:
                bp = make_decoder_runner(dem, spec, args.max_iter)
            except (ImportError, RuntimeError, TypeError) as e:
                msg = f"  {label:18s}  skipped ({str(e).split(chr(10))[0][:80]})"
                if use_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
                err = float("nan")
                new_rows.append({
                    "decoder": label,
                    "p": f"{float(p):.12g}",
                    "logical_err": f"{err:.9g}",
                    "logical_err_stderr": f"{logical_err_stderr(err, args.shots):.9g}",
                    "shots": str(args.shots),
                    "time_per_shot_s": f"{float('nan'):.9g}",
                    "converged_shots": "0",
                    "converged_time_per_shot_s": f"{float('nan'):.9g}",
                })
                done.add(row_key)
                continue

            always_converged = decoder_always_converged(spec)
            total_decode_time = 0.0
            converged_decode_time = 0.0
            converged_shots = 0
            correct = 0
            shot_iter = zip(det, obs)
            if use_progress:
                shot_iter = tqdm(
                    shot_iter,
                    total=args.shots,
                    desc=f"{label} @ p={p:.4g}",
                    unit="shot",
                    leave=False,
                )
            for syn, ob in shot_iter:
                shot_t0 = time.perf_counter()
                r = bp.decode(syn.astype(np.uint8), args.max_iter)
                shot_dt = time.perf_counter() - shot_t0
                total_decode_time += shot_dt
                if always_converged or syndrome_satisfied(bp, syn, r):
                    converged_decode_time += shot_dt
                    converged_shots += 1
                pred = (bp.L @ r.astype(np.int32)) % 2
                if np.array_equal(pred, ob.astype(np.int32)):
                    correct += 1
            err = 1.0 - correct / args.shots
            dt = total_decode_time
            time_per_shot = dt / max(args.shots, 1)
            converged_time_per_shot = (
                converged_decode_time / converged_shots
                if converged_shots
                else float("nan")
            )
            msg = (
                f"  {label:18s}  err={err:.3%}   "
                f"t={dt:5.1f}s   shot={time_per_shot:.3e}s   "
                f"conv={converged_shots}/{args.shots}"
            )
            if use_progress:
                tqdm.write(msg)
            else:
                print(msg)
            new_rows.append({
                "decoder": label,
                "p": f"{float(p):.12g}",
                "logical_err": f"{err:.9g}",
                "logical_err_stderr": f"{logical_err_stderr(err, args.shots):.9g}",
                "shots": str(args.shots),
                "time_per_shot_s": f"{time_per_shot:.9g}",
                "converged_shots": str(converged_shots),
                "converged_time_per_shot_s": f"{converged_time_per_shot:.9g}",
            })
            done.add(row_key)
        if use_progress:
            tqdm.write("")
        else:
            print()

    # ── write CSV ───────────────────────────────────────────────────────────
    all_rows = existing_rows + new_rows
    if not confirm_existing_path(csv_path, "Sweep CSV", args.force):
        print("Aborted before updating sweep CSV.")
        sys.exit(0)
    write_sweep_rows(csv_path, all_rows)
    print(f"CSV    : {csv_path}")
    print(f"Added  : {len(new_rows)} new sweep rows")

    results = rows_by_decoder(all_rows)

    # ── plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    log_x = args.log_x and not args.linear
    log_y = args.log_y and not args.linear

    # y = x reference line: above it the decoder is doing worse than no-code
    ax.plot([args.p_min, args.p_max], [args.p_min, args.p_max],
            color="lightgray", linestyle="--", linewidth=1,
            zorder=0, label="y = x")

    for label, pts in results.items():
        xs = np.array([pt["p"] for pt in pts])
        ys = np.array([pt["logical_err"] for pt in pts])
        yerr = np.array([pt["logical_err_stderr"] for pt in pts])
        shots = np.array([pt["shots"] for pt in pts])
        # clip zeros so log-y doesn't choke
        if log_y:
            ys = np.where(ys > 0, ys, 0.5 / np.maximum(shots, 1))
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=label,
        )
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    ax.set_title(f"{code} threshold sweep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "sweep.png"
    if confirm_existing_path(plot_path, "Threshold plot", args.force):
        fig.savefig(plot_path, dpi=150)
        print(f"Plot   : {plot_path}")
    else:
        print(f"Plot   : skipped existing {plot_path}")
    plt.close(fig)

    # ── time-per-shot plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, pts in results.items():
        xs = np.array([pt["p"] for pt in pts])
        ys = np.array([pt["time_per_shot_s"] for pt in pts])
        conv_ys = np.array([pt["converged_time_per_shot_s"] for pt in pts])
        (line,) = ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            label=f"{label} all",
        )
        ax.plot(
            xs,
            conv_ys,
            marker="o",
            linestyle=":",
            linewidth=1.8,
            color=line.get_color(),
            label=f"{label} converged",
        )
    if log_x:
        ax.set_xscale("log")
    finite_times = [
        t
        for pts in results.values()
        for pt in pts
        for t in (pt["time_per_shot_s"], pt["converged_time_per_shot_s"])
        if np.isfinite(t) and t > 0
    ]
    if finite_times:
        ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Average decode time per shot (s)")
    ax.set_title(f"{code} average decoder time per shot")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    time_plot_path = output_dir / "time_per_shot.png"
    if confirm_existing_path(time_plot_path, "Timing plot", args.force):
        fig.savefig(time_plot_path, dpi=150)
        print(f"Timing : {time_plot_path}")
    else:
        print(f"Timing : skipped existing {time_plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
