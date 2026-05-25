from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

if os.environ.get("FLASHBP_ALLOW_EDITABLE_REBUILD", "").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("SKBUILD_EDITABLE_SKIP", str(ROOT / "build"))

from flashbp.animation.layout import bipartite_layout, edges_from_H
from _common import CODES, layout_for_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render offline analysis plots for an existing min-region experiment."
    )
    parser.add_argument("run_dir", type=str,
                        help="existing results/experiments/min-region/... directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="defaults to <run_dir>/analysis")
    parser.add_argument("--top-k", type=int, default=20,
                        help="number of smallest/largest cases to write")
    parser.add_argument("--cooccurrence-max-nodes", type=int, default=80,
                        help="maximum data nodes shown in the co-occurrence heatmap")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def safe_label(text: str) -> str:
    return (
        text.replace(":", "-")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
    )


def parse_literal(value):
    if value is None or value == "":
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def parse_int(value, default=0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def parse_float(value, default=np.nan) -> float:
    if value is None or value == "":
        return default
    return float(value)


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "shots.csv").open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["shot"] = parse_int(row.get("shot"))
            row["searched"] = parse_bool(row.get("searched"))
            row["search_succeeded"] = parse_bool(row.get("search_succeeded"))
            row["selected_region_count"] = parse_int(row.get("selected_region_count"), 0)
            row["selected_total_log2_valid_states"] = parse_int(
                row.get("selected_total_log2_valid_states"), 0)
            row["selected_total_valid_state_count"] = parse_int(
                row.get("selected_total_valid_state_count"), 0)
            row["selected_max_axes"] = parse_int(row.get("selected_max_axes"), 0)
            row["selected_mean_distance"] = parse_float(row.get("selected_mean_distance"))
            row["syndrome_weight"] = parse_int(row.get("syndrome_weight"), 0)
            row["true_error_weight"] = parse_int(row.get("true_error_weight"), 0)
            row["syndrome"] = parse_literal(row.get("syndrome")) or []
            row["true_errors"] = parse_literal(row.get("true_errors")) or []
            row["selected_ids"] = parse_literal(row.get("selected_ids")) or []
            row["bucket"] = bucket_row(row)
            rows.append(row)
    rows.sort(key=lambda r: int(r["shot"]))
    return rows


def load_region_records(run_dir: Path) -> list[dict]:
    path = run_dir / "region_sets.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bucket_row(row: dict) -> str:
    if not row.get("searched"):
        reason = row.get("skip_reason") or "unsearched"
        return str(reason)
    if not row.get("search_succeeded"):
        return "not repaired"
    if int(row.get("selected_region_count", 0)) == 0:
        return "vacuous"
    return "repaired"


def decoder_labels_from_summary(summary: dict, rows: list[dict]) -> list[str]:
    args = summary.get("args", {})
    labels = list(args.get("baseline_decoders", []) or [])
    opt = args.get("opt_decoder")
    if opt and opt not in labels:
        labels.append(opt)
    if labels:
        return labels
    suffix = "_correct"
    found = []
    if rows:
        for key in rows[0].keys():
            if key.endswith(suffix) and key not in ("manual_final_correct",):
                found.append(key[:-len(suffix)])
    return found


def css_dem_H(code) -> np.ndarray:
    r_x, n = code.H_X.shape
    r_z = code.H_Z.shape[0]
    H = np.zeros((r_x + r_z, 2 * n), dtype=np.uint8)
    H[:r_x, :n] = code.H_X
    H[r_x:, n:] = code.H_Z
    return H


def infer_code(summary: dict):
    args = summary.get("args", {})
    code_name = summary.get("code") or args.get("code")
    if not code_name:
        raise ValueError("Could not infer code from summary.json; pass a complete run directory.")
    if code_name not in CODES:
        raise ValueError(f"Unknown code {code_name!r}; known: {', '.join(CODES)}")
    return code_name, CODES[code_name]()


def successful_nonempty(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if r.get("searched")
        and r.get("search_succeeded")
        and int(r.get("selected_region_count", 0)) > 0
    ]


def plot_complexity_vs_syndrome(rows: list[dict], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "repaired": "#1f77b4",
        "vacuous": "#2ca02c",
        "not repaired": "#d62728",
    }
    for bucket, color in colors.items():
        subset = [r for r in rows if r.get("bucket") == bucket]
        if not subset:
            continue
        x = [r["syndrome_weight"] for r in subset]
        y = [r["selected_total_log2_valid_states"] for r in subset]
        ax.scatter(x, y, label=bucket, alpha=0.75, s=42, color=color)
    ax.set_xlabel("syndrome weight")
    ax.set_ylabel("selected sum log2 valid states")
    ax.set_title("Minimal-region complexity versus syndrome weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_dir / "complexity_vs_syndrome.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_vs_distance(rows: list[dict], output_dir: Path, dpi: int) -> None:
    subset = [
        r for r in successful_nonempty(rows)
        if np.isfinite(float(r.get("selected_mean_distance", np.nan)))
    ]
    if not subset:
        write_placeholder(output_dir / "complexity_vs_distance.png",
                          "no nonempty successful repairs with finite distances", dpi)
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    x = [r["selected_mean_distance"] for r in subset]
    y = [r["selected_total_log2_valid_states"] for r in subset]
    c = [r["selected_region_count"] for r in subset]
    sc = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.8, s=48)
    fig.colorbar(sc, ax=ax, label="selected region count")
    ax.set_xlabel("mean grouped data-node distance to nearest active detector")
    ax.set_ylabel("selected sum log2 valid states")
    ax.set_title("Complexity versus detector distance")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "complexity_vs_distance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_region_count_vs_axes(rows: list[dict], output_dir: Path, dpi: int) -> None:
    subset = successful_nonempty(rows)
    if not subset:
        write_placeholder(output_dir / "region_count_vs_axes.png",
                          "no nonempty successful repairs", dpi)
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    x = [r["selected_region_count"] for r in subset]
    y = [r["selected_max_axes"] for r in subset]
    c = [r["selected_total_log2_valid_states"] for r in subset]
    sc = ax.scatter(x, y, c=c, cmap="plasma", alpha=0.8, s=55)
    fig.colorbar(sc, ax=ax, label="sum log2 valid states")
    ax.set_xlabel("selected region count")
    ax.set_ylabel("max axes in any selected region")
    ax.set_title("Many small regions versus one large region")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "region_count_vs_axes.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_repair_buckets(rows: list[dict], output_dir: Path, dpi: int) -> None:
    counts = Counter(r.get("bucket", "unknown") for r in rows)
    order = ["vacuous", "repaired", "not repaired", "no_search_trigger", "opt_failed", "unsearched"]
    labels = [k for k in order if counts.get(k)] + [
        k for k in sorted(counts) if k not in order
    ]
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="#4c78a8", alpha=0.85)
    ax.set_ylabel("shots")
    ax.set_title("Shot buckets")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_dir / "repair_buckets.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def candidate_source_counts(records: list[dict]) -> tuple[Counter, Counter]:
    pool_counts = Counter()
    selected_counts = Counter()
    for rec in records:
        for cand in rec.get("candidate_summaries", []):
            for source in cand.get("sources", []):
                if source != "shrunk":
                    pool_counts[source] += 1
        if not rec.get("search", {}).get("succeeded"):
            continue
        for cand in rec.get("selected_candidates", []):
            for source in cand.get("sources", []):
                if source != "shrunk":
                    selected_counts[source] += 1
    return pool_counts, selected_counts


def plot_selected_candidate_sources(records: list[dict], output_dir: Path, dpi: int) -> None:
    pool_counts, selected_counts = candidate_source_counts(records)
    if not pool_counts and not selected_counts:
        write_placeholder(output_dir / "selected_candidate_sources.png",
                          "no selected candidate source counts", dpi)
        return
    labels = [
        label for label, _ in
        (pool_counts + selected_counts).most_common()
    ]
    x = np.arange(len(labels))
    pool = np.asarray([pool_counts.get(label, 0) for label in labels], dtype=np.float64)
    selected = np.asarray([selected_counts.get(label, 0) for label in labels], dtype=np.float64)
    rate = np.divide(selected, pool, out=np.zeros_like(selected), where=pool > 0)

    fig, ax = plt.subplots(figsize=(13, 7))
    width = 0.38
    ax.bar(x - width / 2, pool, width=width, label="candidate pool", color="#bab0ab", alpha=0.8)
    ax.bar(x + width / 2, selected, width=width, label="selected repairs", color="#59a14f", alpha=0.85)
    ax.set_ylabel("candidate-source memberships")
    ax.set_title("Candidate source pool versus selected repairs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, rate, color="#d62728", marker="o", linewidth=1.6, label="selected / pool")
    ax2.set_ylabel("selected / pool")
    ax2.set_ylim(0.0, max(1.0, float(rate.max(initial=0)) * 1.15))
    ax2.legend(loc="upper right")
    fig.savefig(output_dir / "selected_candidate_sources.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def selected_node_counts(records: list[dict], num_vars: int, num_checks: int) -> tuple[np.ndarray, np.ndarray]:
    data_counts = np.zeros(num_vars, dtype=np.int64)
    check_counts = np.zeros(num_checks, dtype=np.int64)
    for rec in records:
        if not rec.get("search", {}).get("succeeded"):
            continue
        if not rec.get("selected_candidates"):
            continue
        data_seen = set()
        check_seen = set()
        for cand in rec.get("selected_candidates", []):
            data_seen.update(int(v) for v in cand.get("data", []))
            check_seen.update(int(c) for c in cand.get("checks", []))
        for v in data_seen:
            if 0 <= v < num_vars:
                data_counts[v] += 1
        for c in check_seen:
            if 0 <= c < num_checks:
                check_counts[c] += 1
    return data_counts, check_counts


def plot_selected_node_frequency(
    H: np.ndarray,
    layout: dict,
    records: list[dict],
    output_dir: Path,
    dpi: int,
) -> None:
    num_checks, num_vars = H.shape
    data_counts, check_counts = selected_node_counts(records, num_vars, num_checks)
    if data_counts.max(initial=0) == 0 and check_counts.max(initial=0) == 0:
        write_placeholder(output_dir / "selected_node_frequency.png",
                          "no selected nodes in successful repairs", dpi)
        return

    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    figsize = layout.get("figsize", (12, 8))
    node_size = layout.get("node_size", max(30.0, 4500.0 / max(num_vars, num_checks)))
    edges = edges_from_H(H)
    vmax = max(int(data_counts.max(initial=0)), int(check_counts.max(initial=0)), 1)
    cmap = plt.get_cmap("magma")
    norm = plt.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    for c, v in edges:
        ax.plot(
            [check_pos[c][0], var_pos[v][0]],
            [check_pos[c][1], var_pos[v][1]],
            color="#d0d0d0",
            linewidth=0.45,
            zorder=1,
        )
    data_colors = [cmap(norm(data_counts[v])) for v in range(num_vars)]
    check_colors = [cmap(norm(check_counts[c])) for c in range(num_checks)]
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=node_size,
        c=data_colors,
        edgecolors="#222222",
        linewidths=0.55,
        zorder=3,
    )
    ax.scatter(
        [check_pos[c][0] for c in range(num_checks)],
        [check_pos[c][1] for c in range(num_checks)],
        s=node_size * 0.95,
        c=check_colors,
        edgecolors="#222222",
        linewidths=0.55,
        marker="s",
        zorder=4,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="successful selected-region count")
    ax.set_title("How often each Tanner node appears in selected repairs")
    ax.set_axis_off()
    fig.savefig(output_dir / "selected_node_frequency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def data_cooccurrence(records: list[dict], num_vars: int) -> np.ndarray:
    mat = np.zeros((num_vars, num_vars), dtype=np.int64)
    for rec in records:
        if not rec.get("search", {}).get("succeeded"):
            continue
        for cand in rec.get("selected_candidates", []):
            data = sorted(set(int(v) for v in cand.get("data", []) if 0 <= int(v) < num_vars))
            for i, a in enumerate(data):
                mat[a, a] += 1
                for b in data[i + 1:]:
                    mat[a, b] += 1
                    mat[b, a] += 1
    return mat


def plot_data_cooccurrence(
    records: list[dict],
    num_vars: int,
    output_dir: Path,
    dpi: int,
    max_nodes: int,
) -> None:
    mat = data_cooccurrence(records, num_vars)
    freq = np.diag(mat)
    active = np.flatnonzero(freq > 0)
    if active.size == 0:
        write_placeholder(output_dir / "selected_data_cooccurrence.png",
                          "no selected data-node co-occurrences", dpi)
        return
    order = active[np.argsort(freq[active])[::-1]]
    order = order[:max_nodes]
    sub = mat[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="same selected region count")
    ax.set_title("Selected data-node co-occurrence")
    ax.set_xlabel("data node")
    ax.set_ylabel("data node")
    ticks = np.arange(len(order))
    step = max(1, len(order) // 30)
    ax.set_xticks(ticks[::step])
    ax.set_yticks(ticks[::step])
    ax.set_xticklabels([str(int(v)) for v in order[::step]], rotation=90)
    ax.set_yticklabels([str(int(v)) for v in order[::step]])
    fig.savefig(output_dir / "selected_data_cooccurrence.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_placeholder(path: Path, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def rank_key(row: dict) -> tuple:
    return (
        int(row.get("selected_total_log2_valid_states", 0)),
        int(row.get("selected_max_axes", 0)),
        int(row.get("selected_region_count", 0)),
        int(row.get("shot", 0)),
    )


def write_case_tables(rows: list[dict], records: list[dict], output_dir: Path, top_k: int) -> None:
    records_by_shot = {int(rec.get("shot", -1)): rec for rec in records}
    cases = []
    for row in successful_nonempty(rows):
        rec = records_by_shot.get(int(row["shot"]), {})
        selected_sources = sorted(set(
            source
            for cand in rec.get("selected_candidates", [])
            for source in cand.get("sources", [])
            if source != "shrunk"
        ))
        selected_data = sorted(set(
            int(v)
            for cand in rec.get("selected_candidates", [])
            for v in cand.get("data", [])
        ))
        selected_checks = sorted(set(
            int(c)
            for cand in rec.get("selected_candidates", [])
            for c in cand.get("checks", [])
        ))
        cases.append({
            "shot": row["shot"],
            "syndrome_weight": row["syndrome_weight"],
            "true_error_weight": row["true_error_weight"],
            "selected_region_count": row["selected_region_count"],
            "selected_total_log2_valid_states": row["selected_total_log2_valid_states"],
            "selected_max_axes": row["selected_max_axes"],
            "selected_mean_distance": row.get("selected_mean_distance"),
            "manual_final_iterations": row.get("manual_final_iterations"),
            "selected_ids": row.get("selected_ids"),
            "selected_sources": selected_sources,
            "selected_data": selected_data,
            "selected_checks": selected_checks,
        })
    cases.sort(key=rank_key)
    write_dict_csv(output_dir / "repair_cases_ranked.csv", cases)
    write_dict_csv(output_dir / "repair_cases_smallest.csv", cases[:top_k])
    write_dict_csv(output_dir / "repair_cases_largest.csv", list(reversed(cases[-top_k:])))


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(run_dir)
    rows = load_rows(run_dir)
    records = load_region_records(run_dir)
    code_name, code = infer_code(summary)
    H = css_dem_H(code)
    layout = layout_for_code(code)

    plot_repair_buckets(rows, output_dir, args.dpi)
    plot_complexity_vs_syndrome(rows, output_dir, args.dpi)
    plot_complexity_vs_distance(rows, output_dir, args.dpi)
    plot_region_count_vs_axes(rows, output_dir, args.dpi)
    plot_selected_candidate_sources(records, output_dir, args.dpi)
    plot_selected_node_frequency(H, layout, records, output_dir, args.dpi)
    plot_data_cooccurrence(
        records,
        H.shape[1],
        output_dir,
        args.dpi,
        max_nodes=args.cooccurrence_max_nodes,
    )
    write_case_tables(rows, records, output_dir, args.top_k)

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "code": code_name,
        "num_rows": len(rows),
        "num_region_records": len(records),
        "plots": [
            "repair_buckets.png",
            "complexity_vs_syndrome.png",
            "complexity_vs_distance.png",
            "region_count_vs_axes.png",
            "selected_candidate_sources.png",
            "selected_node_frequency.png",
            "selected_data_cooccurrence.png",
        ],
        "tables": [
            "repair_cases_ranked.csv",
            "repair_cases_smallest.csv",
            "repair_cases_largest.csv",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Analyzed {len(rows)} shots from {run_dir}")
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
