# Evaluation And Graph Style

This repo uses a shared visual language for decoder diagnostics.  The canonical
colors live in `src/flashbp/analytics/style.py`.

## Output Layout

Generated evaluation artifacts should go under `results/` by default.

- `results/errors/`: cached BP-fail / ML-success syndrome NPZ files.
- `results/syndromes/`: single-shot syndrome graph PNGs.
- `results/bp/`: BP animation outputs and BP logs.
- `results/ml/`: ML contraction animation outputs and ML logs.
- `results/cycles/`: cycle overlays and cycle videos.
- `results/trees/`: treeified Tanner graph views.
- `results/tree_bp/`: severed-tree BP marginal diagnostics.
- `results/distance/`: detector-distance maps.
- `results/threshold/`: threshold sweeps.

When an evaluation script accepts `--output`, `--output-dir`, or `--log-file`,
an explicit user path should be respected exactly.  Otherwise, use the relevant
`results/...` default.

## Cache Convention

Cached syndrome files are compressed NPZ files produced by:

```bash
python evaluations/cache_bp_ml_failures.py --code steane --p 0.15 --target 5
```

Default cache path:

```text
results/errors/{code}_{p}.{target}.npz
```

The decimal point in `p` is replaced with `p`, for example:

```text
results/errors/steane_0p15.5.npz
```

Cache-aware scripts should accept:

- `--cache`
- `--shot-index`
- optional `--code` / `--p` overrides

If a cache is provided, scripts should infer `code` and `p` from
`metadata_json` unless the user explicitly overrides them.

## Graph Semantics

Use the same meaning for colors and outlines everywhere.

- **Black filled square**: active detection / parity-check node.
- **White square**: inactive parity-check node.
- **White circle**: data/error node with no active highlighted state.
- **Red**: true sampled error.
  - Red data-node outline or fill means the sampled error bit is `1`.
  - Red Tanner edge means the edge is incident to a true-error data node.
- **Orange**: simple BP-related decision or state.
  - Orange data-node ring means simple BP correction bit is `1`.
  - Orange check outline means currently unsatisfied check in a BP frame.
  - Orange contraction outline/edge means already contracted or selected by a BP-style process.
- **Blue**: ML/root/source emphasis.
  - Blue data-node ring means ML correction bit is `1`.
  - Blue check/root marker means selected source/root/ML-highlighted syndrome.
- **Green**: cycle membership.
  - Green edges/nodes belong to a rendered cycle or active-check cycle family.
- **Faint gray**: background structure.
  - Non-highlighted Tanner edges, non-tree closure edges, and inactive context.

## Plot Types

### Syndrome Graph

`evaluations/render_syndrome.py`

- Active detections are black.
- True-error data nodes are lightly red.
- Edges incident to true-error data nodes are red.

### BP Animation

`evaluations/animate_bp.py`

- Data node fill shows current BP hard decision.
- True-error data nodes use red outlines.
- Active checks are black.
- Unsatisfied checks use orange outlines.

### ML Animation

`evaluations/animate_ml.py`

- Data-node shading shows current/posterior error likelihood.
- True errors use red outlines.
- Predicted/ML overlays use blue when shown.
- Contracted axes use orange outlines/edges.

### Cycles

`evaluations/cycles.py` and `evaluations/active_cycles.py`

- Cycle nodes and edges are green.
- Active checks remain black.
- Faint gray edges provide context.

### Treeified Graphs

`evaluations/treeify_detections.py`

- Root/source check is blue.
- BFS tree edges are blue.
- Severed/non-tree closure edges are faint dashed gray.
- Active checks remain black.

### Tree BP

`evaluations/tree_bp_detections.py`

- Data-node color shows severed-tree BP `Pr(error = 1)`.
- Orange ring is simple BP correction.
- Blue ring is ML correction.
- Red dot is true sampled error.
- Active checks are black; root/source check is blue.

### Detector Distance

`evaluations/detector_distance.py`

- Source detectors are blue.
- Other active detectors are black.
- Data-node shading encodes Tanner-hop distance from selected sources.

## Script Behavior

Evaluation scripts should be able to run in two modes when possible:

1. **Cached mode**: use a saved syndrome from `--cache`.
2. **Sample mode**: sample a fresh shot from `--code`, `--p`, and `--seed`.

Aggregate scripts such as threshold sweeps and bulk logical-error-rate runs do
not use cached single-shot syndromes.
