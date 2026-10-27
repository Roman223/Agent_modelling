# Engineering improvements: validation record

## Research parameter controls: task 1 (2026-10-27)

Implemented on `feat/research-parameter-controls`, branched from `6bd3467`.
Existing untracked research files were preserved. No dependencies were added,
and no research sweeps were started. Tasks 2–7 remain pending.

Changed files and responsibilities:

| Files | Change |
| --- | --- |
| `src/scenarios.py`, `experiments/run_scenario.py` | Five validated keyword-only controls, numeric scalar normalization, immutable CLI overrides, resolved run header |
| `src/asset.py`, `src/order_book.py`, `src/simulation.py` | Dividend and closing-noise controls, day-order expiry, initial/evolved trader parameter propagation |
| `src/bundles.py` | Effective control values in the existing schema-1 configuration; unchanged strict replay checks |
| `tests/test_core.py`, `tests/test_logging.py`, `tests/test_bundles.py` | Control semantics, compatibility, logging neutrality, custom replay, and CLI validation |
| `README.md`, `VALIDATION.md` | Usage, behavior, compatibility limits, and this handoff |

Validation commands and results:

| Command | Result |
| --- | --- |
| `.venv/bin/python -m unittest discover -s tests -p 'test_core.py'` | 19/19 passed |
| `.venv/bin/python tests/test_bundles.py` | 9/9 passed |
| `.venv/bin/python tests/test_logging.py` | 11/11 passed |
| `.venv/bin/python tests/test_legacy_equivalence.py` | 25/25 passed, 250 days, exact results and final RNG states against `84fa2a6` |

Checks ran as background tool sessions with output redirected to
`/tmp/abm-task1-core.log`, `/tmp/abm-task1-bundles.log`,
`/tmp/abm-task1-logging.log`, and `/tmp/abm-task1-legacy.log`.
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` were each set to
`1`; `MPLCONFIGDIR=/tmp/abm-task1-matplotlib` provided a writable plotting cache.
Existing nolds/scikit-learn estimator warnings remained visible. No estimator or
legacy expectation was changed. The extended 1,000-day suite was not run for this task.

The focused checks cover parameter validation before logs/RNG use, positional
constructor compatibility, frozen scenarios, initial and evolved trader controls,
known order quantities, closing-price arithmetic/floors/halts, both dividend shock
branches, constant dividends during warmup and halts, and partial-order expiry.
Scripted two-day markets confirm that expired orders cannot fill on the next day
and that expiry events reconcile with pre-expiry summaries. Logging modes agree
for all eight default scenarios and three combined-control scenarios. Bundle
checks exercise each control separately, all controls with evolution, NumPy scalar
inputs, custom CLI output, exact replay, caller RNG restoration, and rejection of
tampered effective settings.

Task 2 can use `dataclasses.replace(SCENARIOS[name], ...)` with `chartist_alpha`,
`fundamentalist_gamma`, `closing_noise_scale`, `dividend_mode`, and `order_lifetime`,
then call the unchanged `simulate`, `analyze_result`, `save_bundle`, and
`replay_bundle` interfaces. See the README for ranges, defaults, and CLI flags.
Day-order events contain counts and remaining quantities by side; daily book
summaries still describe the state after matching and before expiry.

Retained limits: the global NumPy RNG remains shared, and equal seeds do not
guarantee aligned future draws across interventions. Constant dividends and zero
closing noise retain their original draws. Evolved chartists keep their legacy
20-observation window. Existing circuit-breaker/evolution behavior remains intact.
Historical bundles require their recorded source and runtime; source hashes do
not archive dirty changes. The experiment runner, atlas, hard and smooth reductions,
bifurcation toolkit, and multifractal pipeline are still pending.

## Earlier engineering validation (2026-10-21)

Validated on 2026-10-21 on branch `improvements/reproducibility-and-structure`.
The branch starts at `450c168`; numerical comparisons use the original notebook
at `84fa2a6`. No model rules or regression expectations were revised.

## Structured logging extension

The subsequent logging change was checked separately on the same branch:

| Check | Result |
| --- | --- |
| Legacy suite after logging changes, 250 days | 25/25 passed, including RNG states |
| Existing matching, simulation, metric checks | 10/10 passed |
| Existing bundle/replay/CLI checks | 5/5 passed |
| New logging checks (`tests/test_logging.py`) | 9/9 passed |

Logging checks run all eight scenarios for 102 days in each of the four modes
(`off`, `summary`, `trades`, `decisions`), with seed 0 except circuit-breaker seed
18. Prices, dividends, balances, activity flags, individual/group wealth histories,
legacy events, analysis diagnostics, and simulation/analysis RNG states match
exactly between modes. Separate checks reconcile daily/strategy summaries with
decisions and fills, check immediate transaction balance changes and persistent
partial fills, verify warm-up halts and random evolution links, and test filters,
record/byte caps, streamed export, identical replayed log records, CLI arguments,
temporary-file cleanup on failure, and incomplete-export detection.

The default legacy suite was rerun after this extension. The 1,000-day suite in
the earlier section below describes the preceding engineering implementation;
it was not repeated for the logging extension. The existing estimator warnings
remain visible and do not indicate a logging regression.

A sequential storage smoke run used `more_noise`, 250 days, seed 0. Every mode
ended at price `55.32145065402109`. Only compressed log files are counted below:

| Mode | Compressed bytes | Detail coverage |
| --- | ---: | --- |
| `off` | 0 | Disabled |
| `summary` | 64,894 | All 270 warm-up/trading steps and 750 strategy rows |
| `trades` | 5,765,299 | All 90,785 fills |
| `decisions` | 5,009,527 | 100,000 detail records; cap explicitly reported |

The capped decision run retained 60,500 decisions and 39,500 fills; metadata
reported 77,000 omitted decisions and 51,285 omitted fills. Its summaries remained
complete. Detail is streamed to gzip files and bounded by both record and byte
limits; no population-size detail table is retained in memory. All jobs used one
native thread per library and wrote their output to `/tmp/abm-logging-*.log`.

## Compatibility and correctness

| Check | Result |
| --- | --- |
| Before implementation: default legacy suite | 25/25 passed |
| Result/API refactor: default legacy suite | 25/25 passed, including final RNG states |
| Diagnostics/validation: default legacy suite | 25/25 passed, including final RNG states |
| Optimized implementation: default suite, 250 days | 25/25 passed |
| Optimized implementation: extended suite, 1,000 days | 25/25 passed |
| Focused matching, simulation, metric tests | 10/10 passed |
| Bundle round trips, replay, failure handling, CLI validation | 5/5 passed |
| Fresh pinned environment | Installation, `pip check`, imports through focused tests, and baseline legacy check passed |
| CLI smoke runs and plot inspection | Passed: baseline, evolution, circuit breaker |

Both regression suites cover all eight scenarios at seeds 0, 1, and 2, plus
circuit-breaker seed 18. They compare prices, dividends, trader balances and
activity, every trader's wealth history, sampled group wealth, metric values,
and final NumPy RNG state. The original notebook does not expose structured
event histories; separate focused tests check halt and evolution events and
population counts. Event comparisons are also supported when a reference
provides those attributes.

Bundle checks cover full individual histories, final trader states, seeded and
unseeded replay, deliberately delayed analysis, cached analysis, caller RNG
restoration, unique directories, strict JSON for non-finite metrics, incomplete
writes, absent Git metadata, and source mismatch rejection. CLI validation checks
invalid seeds/days and conflicting output flags before any output directory is
created. An additional metric test preserves explicitly supplied arguments and
the zero-variance Sample Entropy tolerance.

Existing nolds warnings for infinite Sample Entropy estimates and scikit-learn
RANSAC warnings remain visible. These occurred in passing checks and were not
suppressed or used to revise expected values.

## Commands and environment

Runtime: CPython 3.12.9, Linux x86_64; exact runtime dependency closure is recorded
in `requirements-lock.txt`. The lock was installed in a fresh temporary virtual
environment; its focused tests and baseline legacy check (120 days, seed 0)
passed. The existing project environment ran the complete suites.

Long jobs ran in the background with output redirected to logs. Native threading
was bounded with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for
the post-refactor suites, smoke runs, and benchmarks. No process pool was used.

```bash
.venv/bin/python tests/test_core.py
.venv/bin/python tests/test_bundles.py
.venv/bin/python tests/test_legacy_equivalence.py
.venv/bin/python tests/test_legacy_equivalence.py --full
.venv/bin/python experiments/benchmark_order_book.py
```

CLI smoke runs used 120 days: baseline seed 0 with the existing `--save` option;
evolution seed 0 and circuit breaker seed 18 with `--bundle`; all used `--no-show`.
Scenario listing also passed. Price, group wealth, wealth histograms, and trader
population plots were visually inspected. Existing price observation indexing
and approximate halt-marker placement were deliberately retained.

## Order-book benchmark

The final comparison runs the historical and current order books in separate,
sequential processes. The simulation workload is `more_noise`, 250 days, seed 0,
including final metric analysis. The insertion workload adds 10,000 buy orders
with repeated prices. Timings are a single local measurement, not a universal
performance guarantee.

| Measurement | Historical | Stable binary insertion |
| --- | ---: | ---: |
| Insert 10,000 orders | 1.4544 s | 0.0115 s |
| Simulation plus analysis | 14.2458 s | 1.0764 s |
| Process peak RSS (Linux KiB) | 186,828 | 186,908 |
| Final price | 55.32145065402109 | 55.32145065402109 |

This workload was approximately 13.2 times faster end to end with essentially
unchanged peak memory. Equal-price insertion order, partial fills, retained
orders, execution arithmetic, and RNG consumption remain covered by focused
and full legacy checks. Front removal and persistent-order policy were retained.

## Deliberate boundaries

The legacy API still uses the global NumPy stream and updates the shared history
dictionary in place. New result objects keep independent histories; analysis is
cached and assumes those objects are not manually mutated afterward. Replay
requires matching source hashes, Python runtime, and runtime package versions;
bundles record dirty-source status but do not archive source changes.

Research changes, index chronology corrections, trader lineage/accounting
corrections, and new estimators remain deferred. Local `CONTEXT.md` and `CLAUDE.md`
were reconciled with the implementation; they remain Git-ignored. Shared usage
and limitations are documented in the tracked README.
