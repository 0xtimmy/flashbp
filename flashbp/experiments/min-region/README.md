# Minimal GBP Region Search Experiment

This experiment samples syndromes for one code/noise point, measures several
baseline decoders, and searches for small manual GBP region sets that repair or
match an oracle decoder.

The runner is intentionally diagnostic.  `--candidate-source truth` uses
ground-truth error support and is therefore an oracle/debug mode.  It is useful
for discovering what the decoder should have grouped before turning those
patterns into a runtime heuristic based on visible detections or oscillations.

## Example

```powershell
python experiments/min-region/run.py `
  --code surface_5 `
  --p 0.02 `
  --shots 100 `
  --candidate-source all `
  --no-manual-add-single-checks `
  --workers 4 `
  --max-candidates 64 `
  --max-selected 6
```

## Outputs

By default outputs are written under:

```text
results/experiments/min-region/{code}_{p}.{candidate_source}.{shots}/
```

The main artifacts are:

- `shots.csv`: one row per sampled syndrome with baseline decoder outcomes and
  minimal-region search results.
- `region_sets.jsonl`: one JSON object per searched shot with selected groups,
  candidate summaries, syndrome, observables, and true errors.
- `top_region_sets.json`: the lowest-complexity successful selected sets.
- `summary.json`: aggregate logical error rates, convergence rates, search
  success rate, selected-region size distributions, and detector-distance stats.
- `progress.json`: periodically rewritten progress snapshot for long runs.

## Candidate Sources

- `delta`: candidates from GBP policy-delta diagnostics.
- `detections`: active detector neighborhoods/components only.
- `truth`: oracle true-error support/components only.
- `all`: union of the above.

## Manual Baseline

By default manual GBP trials include single-check fallback regions.  Pass
`--no-manual-add-single-checks` to start from no regions and test only the
candidate groups.

## Progress And Multiprocessing

Use `--workers N` to analyze shots in parallel. Each worker constructs its own
decoders once and reuses them across assigned shots, so this is most useful for
larger runs where worker startup is amortized.

The runner shows a tqdm progress bar when `tqdm` is installed. Use
`--no-progress` to disable it. Independently of the terminal display, it writes
`progress.json` under the output directory every few seconds. You can override
that path with `--progress-file` and control the write cadence with
`--progress-write-every`.

The shot-index plots are also refreshed during long runs. By default this
happens every 64 completed shots; use `--plot-every K` to change the cadence or
`--plot-every 0` to disable periodic plot writes. The final run still writes the
complete plot set.

## Search Trigger

By default the expensive minimal-region search is only run when the optimal
decoder is correct, the syndrome is nonzero, and at least one non-opt baseline
decoder has a logical failure:

```text
--search-trigger baseline_failure_nonzero
--trigger-mode logical
```

This avoids counting trivial zero-syndrome shots as "minimal region" successes.
Use `--search-trigger opt_correct` to search every ML-correct shot, or
`--search-trigger nonzero_syndrome` to search all nonzero syndromes where the
optimal decoder succeeds.
