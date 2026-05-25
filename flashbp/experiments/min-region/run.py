from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Editable scikit-build normally checks/rebuilds the CMake extension at import
# time. In multiprocessing runs every worker imports this module, and concurrent
# MSBuild/CMake checks race with each other on Windows. The experiment assumes
# the extension has already been built, so skip the editable rebuild hook unless
# the caller explicitly opted into it.
if os.environ.get("FLASHBP_ALLOW_EDITABLE_REBUILD", "").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("SKBUILD_EDITABLE_SKIP", str(ROOT / "build"))

import flashbp
from flashbp import DecoderConfig
from flashbp.analytics import (
    data_detector_distances,
    gbp_delta_region_context_cover,
    gbp_nearest_baseline_region_matches,
    gbp_policy_delta,
    manual_groups_from_candidates,
    search_minimal_gbp_groups,
    select_gbp_region_candidates,
)

from _common import (
    CODES,
    logical_prediction,
    make_decoder_runner,
    parse_decoder_spec,
    p_token,
    prepare_output_dir,
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
    parser.add_argument("--shots", type=int, default=50,
                        help="number of sampled syndromes to examine")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-decoders", nargs="+",
                        default=["simple", "bp-osd:0"],
                        help="decoders to score on every sampled shot")
    parser.add_argument("--opt-decoder", type=str, default="ml",
                        help="decoder treated as optimal; if it fails, skip region search")
    parser.add_argument("--search-trigger",
                        choices=("opt_correct", "nonzero_syndrome",
                                 "baseline_failure", "baseline_failure_nonzero"),
                        default="baseline_failure_nonzero",
                        help="which ML-correct shots are worth minimal-region search")
    parser.add_argument("--trigger-mode", choices=("logical", "convergence", "either"),
                        default="logical",
                        help="what counts as a baseline failure for search triggering")
    parser.add_argument("--trigger-decoders", nargs="*", default=None,
                        help="baseline decoders used for --search-trigger baseline_failure*; "
                             "defaults to all baseline decoders except opt")
    parser.add_argument("--gbp-fail-decoder", type=str, default="gbp-cycles-all:8",
                        help="GBP policy used as the failing side for delta candidates")
    parser.add_argument("--gbp-success-decoder", type=str, default="gbp-cycles-any:8",
                        help="GBP policy used as the succeeding side for delta candidates")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--gbp-max-iter", type=int, default=None)
    parser.add_argument("--ml-max-iter", type=int, default=1)
    parser.add_argument("--candidate-source",
                        choices=("delta", "detections", "truth", "all"),
                        default="all")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-selected", type=int, default=6)
    parser.add_argument("--max-log2-valid-states", type=int, default=None)
    parser.add_argument("--nearest-baseline-k", type=int, default=5)
    parser.add_argument("--manual-backend", type=str, default=None)
    parser.add_argument("--no-manual-add-single-checks", action="store_true")
    parser.add_argument("--allow-logical-failure", action="store_true",
                        help="count manual convergence as success even if logical obs are wrong")
    parser.add_argument("--top-k", type=int, default=25,
                        help="number of low-complexity successful region sets to retain")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of worker processes for per-shot analysis")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable tqdm progress bar")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="JSON progress file; defaults to <output-dir>/progress.json")
    parser.add_argument("--progress-write-every", type=float, default=5.0,
                        help="minimum seconds between progress-file rewrites")
    parser.add_argument("--plot-every", type=int, default=64,
                        help="refresh shot-index plots every K completed shots; use 0 to disable")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def max_iter_for(label: str, spec, args) -> int:
    if spec.config is not None and spec.config.decoder in ("ml", "maximum_likelihood"):
        return args.ml_max_iter
    if spec.config is not None and spec.config.decoder == "gbp":
        return args.gbp_max_iter if args.gbp_max_iter is not None else args.max_iter
    return args.max_iter


def decoder_forced_converged(spec) -> bool:
    if spec.backend == "ldpc_bp_osd":
        return True
    if spec.config is not None and spec.config.decoder in ("ml", "maximum_likelihood"):
        return True
    return False


def decode_runner(decoder, syndrome, true_obs, max_iter: int, *, force_converged: bool = False) -> dict:
    t0 = time.perf_counter()
    correction = decoder.decode(syndrome, max_iter)
    duration_s = time.perf_counter() - t0
    pred_obs = logical_prediction(decoder, correction).astype(np.uint8)
    residual = ((decoder.H @ correction.astype(np.int32)) % 2).astype(np.uint8) ^ syndrome
    stats = decoder.last_decode_stats() if hasattr(decoder, "last_decode_stats") else {}
    converged = (
        True
        if force_converged
        else bool(stats.get("converged", int(residual.sum()) == 0))
    )
    return {
        "correction": correction.astype(np.uint8),
        "pred_obs": pred_obs,
        "correct": bool(np.array_equal(pred_obs, true_obs)),
        "converged": converged,
        "iterations": int(stats.get("iterations", max_iter)),
        "residual_weight": int(residual.sum()),
        "duration_s": float(duration_s),
    }


def make_gbp_logger(dem, spec, label: str):
    if spec.config is None or spec.config.decoder != "gbp":
        raise ValueError(f"{label} must be a GBP decoder")
    cfg = copy.deepcopy(spec.config)
    cfg.log = True
    cfg.log_type = "gbp"
    cfg.log_console = False
    cfg.log_file = None
    cfg.log_level = 1
    return flashbp.FlashBP(dem, cfg)


def decode_gbp_recording(bp, syndrome, true_obs, max_iter: int) -> dict:
    out = decode_runner(bp, syndrome, true_obs, max_iter)
    out["recording"] = bp.get_recording()
    return out


def make_manual_evaluator(
    dem,
    syndrome,
    true_obs,
    max_iter,
    template_cfg,
    backend_override,
    add_single_checks: bool,
):
    cache: dict[tuple, dict] = {}

    def evaluate(candidate_groups: list[dict]) -> dict:
        key = tuple(
            (
                int(row.get("candidate", -1)),
                tuple(int(v) for v in row.get("data", [])),
                tuple(int(c) for c in row.get("checks", [])),
                row.get("activation", "always"),
            )
            for row in candidate_groups
        )
        if key in cache:
            return dict(cache[key])
        cfg = copy.deepcopy(template_cfg)
        cfg.decoder = "gbp"
        cfg.region_policy = "manual_groups"
        cfg.gbp_manual_groups = manual_groups_from_candidates(candidate_groups)
        cfg.gbp_manual_add_single_checks = bool(add_single_checks)
        if backend_override:
            cfg.gbp_backend = backend_override
        cfg.log = False
        cfg.log_console = False
        bp = flashbp.FlashBP(dem, cfg)
        outcome = decode_runner(bp, syndrome, true_obs, max_iter)
        compact = {
            "converged": outcome["converged"],
            "correct": outcome["correct"],
            "residual_weight": outcome["residual_weight"],
            "iterations": outcome["iterations"],
            "duration_s": outcome["duration_s"],
        }
        cache[key] = compact
        return dict(compact)

    return evaluate


def group_distance_stats(H, syndrome, groups: list[dict]) -> dict:
    if not groups:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    H = np.asarray(H, dtype=np.uint8)
    try:
        distances = data_detector_distances(H, syndrome=syndrome)
    except ValueError:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    values = []
    for group in groups:
        for v in group.get("data", []):
            d = int(distances[int(v)])
            if d >= 0:
                values.append(d)
    if not values:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "count": int(len(values)),
    }


def total_valid_state_count(groups: list[dict]) -> int:
    return int(sum(int(row.get("valid_state_count", 1)) for row in groups))


def update_top_sets(top_sets: list[dict], row: dict, groups: list[dict], args) -> None:
    if not groups or not row.get("search_succeeded"):
        return
    entry = {
        "shot": row["shot"],
        "complexity": int(row["selected_total_valid_state_count"]),
        "log2_complexity": int(row["selected_total_log2_valid_states"]),
        "selected_region_count": int(row["selected_region_count"]),
        "selected_max_axes": int(row["selected_max_axes"]),
        "selected_mean_distance": row["selected_mean_distance"],
        "syndrome": row["syndrome"],
        "true_obs": row["true_obs"],
        "true_errors": row["true_errors"],
        "groups": manual_groups_from_candidates(groups),
        "candidate_ids": row["selected_ids"],
    }
    top_sets.append(entry)
    top_sets.sort(key=lambda x: (
        x["log2_complexity"],
        x["selected_max_axes"],
        x["selected_region_count"],
        x["complexity"],
        x["shot"],
    ))
    del top_sets[args.top_k:]


def summarize(rows: list[dict], decoder_labels: list[str], args) -> dict:
    summary = {
        "num_shots": len(rows),
        "searched_shots": int(sum(bool(r.get("searched")) for r in rows)),
        "skipped_opt_failures": int(sum(r.get("skip_reason") == "opt_failed" for r in rows)),
        "skipped_no_search_trigger": int(sum(r.get("skip_reason") == "no_search_trigger" for r in rows)),
        "search_success_rate": None,
        "nonempty_search_success_rate": None,
        "decoders": {},
        "selected_region_count": {},
        "nonempty_selected_region_count": {},
        "selected_log2_valid_states": {},
        "selected_detector_distance": {},
    }
    for label in decoder_labels:
        key = safe_label(label)
        if not rows:
            continue
        summary["decoders"][label] = {
            "logical_error_rate": float(np.mean([not bool(r[f"{key}_correct"]) for r in rows])),
            "convergence_rate": float(np.mean([bool(r[f"{key}_converged"]) for r in rows])),
            "mean_iterations": float(np.mean([int(r[f"{key}_iterations"]) for r in rows])),
            "mean_duration_s": float(np.mean([float(r[f"{key}_duration_s"]) for r in rows])),
        }
    searched = [r for r in rows if r.get("searched")]
    if searched:
        summary["search_success_rate"] = float(np.mean([bool(r["search_succeeded"]) for r in searched]))
        nonempty_hits = [
            bool(r["search_succeeded"]) and int(r.get("selected_region_count", 0)) > 0
            for r in searched
        ]
        summary["nonempty_search_success_rate"] = float(np.mean(nonempty_hits))
    successful = [r for r in searched if r.get("search_succeeded")]
    nonempty_successful = [
        r for r in successful
        if int(r.get("selected_region_count", 0)) > 0
    ]
    if successful:
        counts = np.asarray([r["selected_region_count"] for r in successful], dtype=np.float64)
        log2s = np.asarray([r["selected_total_log2_valid_states"] for r in successful], dtype=np.float64)
        distances = np.asarray([
            r["selected_mean_distance"]
            for r in successful
            if r["selected_mean_distance"] is not None
        ], dtype=np.float64)
        summary["selected_region_count"] = {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "histogram": {
                str(int(v)): int(np.sum(counts == v))
                for v in sorted(set(counts.astype(int)))
            },
        }
        summary["selected_log2_valid_states"] = {
            "mean": float(log2s.mean()),
            "median": float(np.median(log2s)),
            "max": int(log2s.max()),
        }
        if distances.size:
            summary["selected_detector_distance"] = {
                "mean": float(distances.mean()),
                "median": float(np.median(distances)),
                "max": int(distances.max()),
            }
    if nonempty_successful:
        counts = np.asarray([r["selected_region_count"] for r in nonempty_successful], dtype=np.float64)
        summary["nonempty_selected_region_count"] = {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "histogram": {
                str(int(v)): int(np.sum(counts == v))
                for v in sorted(set(counts.astype(int)))
            },
        }
    summary["args"] = vars(args)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def running_mean(values: list[bool | int | float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    return np.cumsum(arr) / np.arange(1, arr.size + 1)


def write_placeholder_plot(output_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boolean_rates(rows: list[dict], decoder_labels: list[str], output_dir: Path) -> None:
    if not rows:
        return
    x = np.asarray([int(r["shot"]) for r in rows], dtype=np.int64)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for label in decoder_labels:
        key = safe_label(label)
        correct = [bool(r.get(f"{key}_correct", False)) for r in rows]
        converged = [bool(r.get(f"{key}_converged", False)) for r in rows]
        axes[0].plot(x, running_mean(correct), label=label, linewidth=1.8)
        axes[1].plot(x, running_mean(converged), label=label, linewidth=1.8)
    axes[0].set_ylabel("running logical success rate")
    axes[1].set_ylabel("running convergence rate")
    axes[1].set_xlabel("shot index")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Baseline decoder outcomes over sampled shots")
    fig.savefig(output_dir / "baseline_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_manual_search_rates(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    x = np.asarray([int(r["shot"]) for r in rows], dtype=np.int64)
    searched = [bool(r.get("searched", False)) for r in rows]
    search_success = [
        bool(r.get("search_succeeded", False)) if bool(r.get("searched", False)) else False
        for r in rows
    ]
    nonempty_success = [
        bool(r.get("search_succeeded", False))
        and int(r.get("selected_region_count", 0) or 0) > 0
        for r in rows
    ]
    opt_skipped = [r.get("skip_reason") == "opt_failed" for r in rows]
    untriggered = [r.get("skip_reason") == "no_search_trigger" for r in rows]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, running_mean(searched), label="searched", linewidth=1.8)
    ax.plot(x, running_mean(search_success), label="manual succeeds", linewidth=1.8)
    ax.plot(x, running_mean(nonempty_success), label="manual succeeds with regions", linewidth=2.2)
    ax.plot(x, running_mean(opt_skipped), label="ML/opt failed skip", linewidth=1.2, linestyle="--")
    ax.plot(x, running_mean(untriggered), label="not triggered", linewidth=1.2, linestyle="--")
    ax.set_xlabel("shot index")
    ax.set_ylabel("running fraction of all shots")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Minimal-region search outcomes over sampled shots")
    fig.savefig(output_dir / "search_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_region_distributions(rows: list[dict], output_dir: Path) -> None:
    successful = [
        r for r in rows
        if r.get("searched")
        and r.get("search_succeeded")
        and int(r.get("selected_region_count", 0) or 0) > 0
    ]
    if not successful:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "no nonempty successful region sets", ha="center", va="center", fontsize=12)
        fig.savefig(output_dir / "region_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    counts = np.asarray([int(r["selected_region_count"]) for r in successful], dtype=np.int64)
    log2s = np.asarray([int(r["selected_total_log2_valid_states"]) for r in successful], dtype=np.int64)
    axes_values = np.asarray([int(r["selected_max_axes"]) for r in successful], dtype=np.int64)
    distances = np.asarray([
        float(r["selected_mean_distance"])
        for r in successful
        if r.get("selected_mean_distance") is not None
    ], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    axes[0].hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5, color="#1f77b4", alpha=0.8)
    axes[0].set_xlabel("selected region count")
    axes[0].set_ylabel("shots")
    axes[1].hist(log2s, bins=min(30, max(5, len(set(log2s)))), color="#ff7f0e", alpha=0.8)
    axes[1].set_xlabel("sum log2 valid states")
    axes[2].hist(axes_values, bins=np.arange(axes_values.min(), axes_values.max() + 2) - 0.5, color="#2ca02c", alpha=0.8)
    axes[2].set_xlabel("max axes in selected region")
    if distances.size:
        axes[3].hist(distances, bins=min(30, max(5, len(set(distances)))), color="#9467bd", alpha=0.8)
    else:
        axes[3].text(0.5, 0.5, "no finite detector distances", ha="center", va="center")
    axes[3].set_xlabel("mean data-node distance to nearest active detector")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Nonempty successful minimal-region set distributions")
    fig.savefig(output_dir / "region_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_manual_vs_baseline(rows: list[dict], decoder_labels: list[str], output_dir: Path) -> None:
    searched = [r for r in rows if r.get("searched")]
    if not searched:
        write_placeholder_plot(
            output_dir / "manual_vs_baselines.png",
            "no searched shots",
        )
        return
    x = np.asarray([int(r["shot"]) for r in searched], dtype=np.int64)
    manual = [bool(r.get("manual_final_correct", False)) for r in searched]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, running_mean(manual), label="manual minimal-region", linewidth=2.4, color="#1f77b4")
    for label in decoder_labels:
        key = safe_label(label)
        vals = [bool(r.get(f"{key}_correct", False)) for r in searched]
        ax.plot(x, running_mean(vals), label=label, linewidth=1.4, alpha=0.8)
    ax.set_xlabel("shot index")
    ax.set_ylabel("running logical success rate on searched shots")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Manual minimal-region decoder versus baselines on searched shots")
    fig.savefig(output_dir / "manual_vs_baselines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_plots(rows: list[dict], decoder_labels: list[str], output_dir: Path) -> None:
    plot_boolean_rates(rows, decoder_labels, output_dir)
    plot_manual_search_rates(rows, output_dir)
    plot_region_distributions(rows, output_dir)
    plot_manual_vs_baseline(rows, decoder_labels, output_dir)


def write_shot_index_plots(rows: list[dict], decoder_labels: list[str], output_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: int(row["shot"]))
    plot_boolean_rates(ordered, decoder_labels, output_dir)
    plot_manual_search_rates(ordered, output_dir)
    plot_manual_vs_baseline(ordered, decoder_labels, output_dir)


WORKER = {}


def init_worker(args_dict: dict) -> None:
    args = argparse.Namespace(**args_dict)
    code = CODES[args.code]()
    dem = code.to_dem(float(args.p))

    requested_specs = {label: parse_decoder_spec(label) for label in args.baseline_decoders}
    opt_spec = parse_decoder_spec(args.opt_decoder)
    if args.opt_decoder not in requested_specs:
        requested_specs[args.opt_decoder] = opt_spec

    decoder_specs = {}
    decoders = {}
    skipped_decoders = {}
    for label, spec in requested_specs.items():
        try:
            decoders[label] = make_decoder_runner(
                dem, spec, max_iter_for(label, spec, args))
            decoder_specs[label] = spec
        except (ImportError, TypeError) as exc:
            if label == args.opt_decoder:
                raise
            skipped_decoders[label] = str(exc)

    gbp_fail_spec = parse_decoder_spec(args.gbp_fail_decoder)
    gbp_success_spec = parse_decoder_spec(args.gbp_success_decoder)
    gbp_fail = make_gbp_logger(dem, gbp_fail_spec, "gbp_fail")
    gbp_success = make_gbp_logger(dem, gbp_success_spec, "gbp_success")

    WORKER.clear()
    WORKER.update({
        "args": args,
        "code": code,
        "dem": dem,
        "decoder_specs": decoder_specs,
        "decoder_labels": list(decoder_specs.keys()),
        "decoders": decoders,
        "skipped_decoders": skipped_decoders,
        "gbp_fail_spec": gbp_fail_spec,
        "gbp_success_spec": gbp_success_spec,
        "gbp_fail": gbp_fail,
        "gbp_success": gbp_success,
        "gbp_iter": args.gbp_max_iter if args.gbp_max_iter is not None else args.max_iter,
    })


def process_one_shot(payload: dict) -> dict:
    args = WORKER["args"]
    dem = WORKER["dem"]
    decoder_specs = WORKER["decoder_specs"]
    decoder_labels = WORKER["decoder_labels"]
    decoders = WORKER["decoders"]
    gbp_fail = WORKER["gbp_fail"]
    gbp_success = WORKER["gbp_success"]
    gbp_success_spec = WORKER["gbp_success_spec"]
    gbp_iter = WORKER["gbp_iter"]

    shot_idx = int(payload["shot"])
    syndrome = np.asarray(payload["syndrome"], dtype=np.uint8)
    true_obs = np.asarray(payload["true_obs"], dtype=np.uint8)
    true_errors = np.asarray(payload["true_errors"], dtype=np.uint8)

    row = {
        "shot": shot_idx,
        "syndrome_weight": int(syndrome.sum()),
        "true_error_weight": int(true_errors.sum()),
        "syndrome": syndrome.astype(int).tolist(),
        "true_obs": true_obs.astype(int).tolist(),
        "true_errors": true_errors.astype(int).tolist(),
        "searched": False,
        "skip_reason": "",
    }

    outcomes = {}
    for label, decoder in decoders.items():
        spec = decoder_specs[label]
        outcome = decode_runner(
            decoder,
            syndrome,
            true_obs,
            max_iter_for(label, spec, args),
            force_converged=decoder_forced_converged(spec),
        )
        outcomes[label] = outcome
        key = safe_label(label)
        row[f"{key}_correct"] = outcome["correct"]
        row[f"{key}_converged"] = outcome["converged"]
        row[f"{key}_residual_weight"] = outcome["residual_weight"]
        row[f"{key}_iterations"] = outcome["iterations"]
        row[f"{key}_duration_s"] = outcome["duration_s"]

    opt = outcomes[args.opt_decoder]
    if not opt["correct"]:
        row["skip_reason"] = "opt_failed"
        return {"row": row, "region_record": None, "selected": []}

    if not search_triggered(row, decoder_labels, args):
        row["skip_reason"] = "no_search_trigger"
        return {"row": row, "region_record": None, "selected": []}

    fail = decode_gbp_recording(gbp_fail, syndrome, true_obs, gbp_iter)
    success = decode_gbp_recording(gbp_success, syndrome, true_obs, gbp_iter)
    region_rows, _ = gbp_policy_delta(
        fail["recording"],
        success["recording"],
        gbp_success.num_errors,
        gbp_success.num_detectors,
    )
    selected_contexts, _, context_summary = gbp_delta_region_context_cover(
        gbp_success,
        syndrome,
        fail["correction"],
        success["correction"],
        region_rows,
    )
    nearest_rows = gbp_nearest_baseline_region_matches(
        selected_contexts,
        fail["recording"],
        top_k=args.nearest_baseline_k,
    )
    candidates = select_gbp_region_candidates(
        gbp_success.H,
        syndrome=syndrome,
        true_errors=true_errors,
        candidate_source=args.candidate_source,
        selected_contexts=selected_contexts,
        region_rows=region_rows,
        nearest_baseline_rows=nearest_rows,
        max_candidates=args.max_candidates,
        max_log2_valid_states=args.max_log2_valid_states,
    )

    evaluator = make_manual_evaluator(
        dem,
        syndrome,
        true_obs,
        gbp_iter,
        gbp_success_spec.config or DecoderConfig(decoder="gbp"),
        args.manual_backend,
        not args.no_manual_add_single_checks,
    )
    search = search_minimal_gbp_groups(
        candidates,
        evaluator,
        max_selected=args.max_selected,
        require_correct=not args.allow_logical_failure,
        H=gbp_success.H,
    )
    selected = search["selected"]
    distance_stats = group_distance_stats(gbp_success.H, syndrome, selected)
    row.update({
        "searched": True,
        "search_succeeded": bool(search["succeeded"]),
        "selected_ids": search["selected_ids"],
        "selected_region_count": len(selected),
        "selected_total_valid_state_count": total_valid_state_count(selected),
        "selected_total_log2_valid_states": int(sum(
            int(g.get("log2_valid_states", 0)) for g in selected
        )),
        "selected_max_axes": int(max(
            [int(g.get("num_axes", 0)) for g in selected] or [0]
        )),
        "selected_mean_distance": distance_stats["mean"],
        "selected_min_distance": distance_stats["min"],
        "selected_max_distance": distance_stats["max"],
        "num_candidates": len(candidates),
        "context_targets": context_summary.get("universe", []),
        "manual_final_converged": search["final_outcome"].get("converged"),
        "manual_final_correct": search["final_outcome"].get("correct"),
        "manual_final_residual_weight": search["final_outcome"].get("residual_weight"),
        "manual_final_iterations": search["final_outcome"].get("iterations"),
    })

    region_record = {
        "shot": shot_idx,
        "syndrome": row["syndrome"],
        "true_obs": row["true_obs"],
        "true_errors": row["true_errors"],
        "baseline_outcomes": {
            label: {
                "correct": outcomes[label]["correct"],
                "converged": outcomes[label]["converged"],
                "residual_weight": outcomes[label]["residual_weight"],
                "iterations": outcomes[label]["iterations"],
            }
            for label in decoder_labels
        },
        "context_summary": context_summary,
        "selected_groups": manual_groups_from_candidates(selected),
        "selected_candidates": selected,
        "search": {
            "succeeded": search["succeeded"],
            "selected_ids": search["selected_ids"],
            "final_outcome": search["final_outcome"],
            "history": search["history"],
        },
        "candidate_summaries": [
            {
                "candidate": c["candidate"],
                "sources": c["sources"],
                "source_regions": c["source_regions"],
                "num_axes": c["num_axes"],
                "log2_valid_states": c["log2_valid_states"],
                "data": c["data"],
                "checks": c["checks"],
            }
            for c in candidates
        ],
    }
    return {"row": row, "region_record": region_record, "selected": selected}


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def trigger_decoder_labels(decoder_labels: list[str], args) -> list[str]:
    if args.trigger_decoders:
        return list(args.trigger_decoders)
    return [label for label in decoder_labels if label != args.opt_decoder]


def decoder_trigger_failed(row: dict, label: str, mode: str) -> bool:
    key = safe_label(label)
    correct = bool(row.get(f"{key}_correct", False))
    converged = bool(row.get(f"{key}_converged", False))
    if mode == "logical":
        return not correct
    if mode == "convergence":
        return not converged
    return (not correct) or (not converged)


def search_triggered(row: dict, decoder_labels: list[str], args) -> bool:
    if args.search_trigger == "opt_correct":
        return True
    nonzero = int(row.get("syndrome_weight", 0)) > 0
    if args.search_trigger == "nonzero_syndrome":
        return nonzero
    labels = trigger_decoder_labels(decoder_labels, args)
    baseline_failed = any(
        decoder_trigger_failed(row, label, args.trigger_mode)
        for label in labels
        if safe_label(label) + "_correct" in row
    )
    if args.search_trigger == "baseline_failure":
        return baseline_failed
    return nonzero and baseline_failed


class ProgressBar:
    def __init__(self, total: int, enabled: bool):
        self.total = total
        self.enabled = enabled
        self.bar = None
        if enabled:
            try:
                from tqdm import tqdm
                self.bar = tqdm(total=total, unit="shot")
            except Exception:
                self.bar = None

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def main():
    args = parse_args()
    code = CODES[args.code]()
    dem = code.to_dem(float(args.p))
    output_dir = Path(
        args.output_dir
        or (
            ROOT / "results" / "experiments" / "min-region"
            / f"{args.code}_{p_token(args.p)}.{args.candidate_source}.{args.shots}"
        )
    )
    prepare_output_dir(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args).copy()
    init_worker(args_dict)
    decoder_labels = WORKER["decoder_labels"]
    for label, reason in WORKER["skipped_decoders"].items():
        print(f"WARNING: skipping decoder {label!r}: {reason}")

    sampler = dem.compile_sampler()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    top_sets: list[dict] = []
    region_sets_path = output_dir / "region_sets.jsonl"
    region_sets_file = region_sets_path.open("w", encoding="utf-8")
    t0 = time.time()
    progress_path = Path(args.progress_file) if args.progress_file else output_dir / "progress.json"

    print(f"Code        : {code}")
    print(f"Noise p     : {args.p:.6g}")
    print(f"Shots       : {args.shots}")
    print(f"Workers     : {args.workers}")
    print(f"Decoders    : {', '.join(decoder_labels)}")
    print(f"Opt decoder : {args.opt_decoder}")
    print(f"Candidates  : {args.candidate_source}")
    print(f"Output      : {output_dir}")

    payloads = []
    shot_idx = 0
    while shot_idx < args.shots:
        batch = min(args.batch_size, args.shots - shot_idx)
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        np.random.seed(seed)
        det_data, obs_data, err_data = sampler.sample(shots=batch, return_errors=True)
        for local in range(batch):
            payloads.append({
                "shot": shot_idx,
                "syndrome": det_data[local].astype(np.uint8),
                "true_obs": obs_data[local].astype(np.uint8),
                "true_errors": err_data[local].astype(np.uint8),
            })
            shot_idx += 1

    progress = ProgressBar(args.shots, enabled=not args.no_progress)
    last_progress_write = 0.0

    def consume_result(result: dict) -> None:
        nonlocal last_progress_write
        row = result["row"]
        selected = result.get("selected", [])
        region_record = result.get("region_record")
        rows.append(row)
        if region_record is not None:
            region_sets_file.write(json.dumps(region_record) + "\n")
            region_sets_file.flush()
        update_top_sets(top_sets, row, selected, args)
        progress.update(1)

        now = time.time()
        should_print = args.progress_every > 0 and len(rows) % args.progress_every == 0
        should_write = (now - last_progress_write) >= args.progress_write_every or len(rows) == args.shots
        if should_print or should_write:
            elapsed = now - t0
            searched = sum(bool(r.get("searched")) for r in rows)
            succ = sum(bool(r.get("search_succeeded")) for r in rows)
            skipped = sum(r.get("skip_reason") == "opt_failed" for r in rows)
            untriggered = sum(r.get("skip_reason") == "no_search_trigger" for r in rows)
            payload = {
                "completed": len(rows),
                "total": args.shots,
                "searched": searched,
                "search_success": succ,
                "skipped_opt_failures": skipped,
                "skipped_no_search_trigger": untriggered,
                "elapsed_s": elapsed,
                "shots_per_second": len(rows) / elapsed if elapsed > 0 else None,
                "output_dir": str(output_dir),
                "updated_at_unix": now,
            }
            if should_write:
                write_progress(progress_path, payload)
                last_progress_write = now
            if should_print:
                print(
                    f"shot={len(rows)}/{args.shots} searched={searched} "
                    f"search_success={succ} skipped_opt={skipped} "
                    f"untriggered={untriggered} elapsed={elapsed:.1f}s"
                )
        if args.plot_every > 0 and len(rows) % args.plot_every == 0:
            try:
                write_shot_index_plots(rows, decoder_labels, output_dir)
            except Exception as exc:
                print(
                    f"WARNING: failed to refresh shot-index plots at "
                    f"{len(rows)} shots: {exc}"
                )

    try:
        if args.workers <= 1:
            for payload in payloads:
                consume_result(process_one_shot(payload))
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(args_dict,),
            ) as pool:
                futures = [pool.submit(process_one_shot, payload) for payload in payloads]
                for future in as_completed(futures):
                    consume_result(future.result())
    finally:
        progress.close()
        region_sets_file.close()

    rows.sort(key=lambda row: int(row["shot"]))

    write_csv(output_dir / "shots.csv", rows)
    (output_dir / "top_region_sets.json").write_text(
        json.dumps(top_sets, indent=2),
        encoding="utf-8",
    )
    summary = summarize(rows, decoder_labels, args)
    summary.update({
        "code": args.code,
        "code_repr": str(code),
        "p": float(args.p),
        "elapsed_s": float(time.time() - t0),
        "output_dir": str(output_dir),
        "region_sets_path": str(region_sets_path),
    })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_plots(rows, decoder_labels, output_dir)
    print(f"Wrote shots : {output_dir / 'shots.csv'}")
    print(f"Wrote sets  : {region_sets_path}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")
    print(f"Wrote plots : {output_dir}")


if __name__ == "__main__":
    main()
