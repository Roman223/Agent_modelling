# Agent-Based Financial Market Model

A Python simulation of a single-asset market with noise traders, chartists, and
fundamentalists. Explore how trader composition, trend-following windows, trading
halts, and strategy changes affect prices and trader wealth.

The model is available through a command-line runner and a Jupyter notebook.
Scenarios share one implementation and can be reproduced with an explicit seed.

## Quick start

Run commands from the repository root. If you already have the project's `.venv`,
skip creating it.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/run_scenario.py --list
.venv/bin/python experiments/run_scenario.py baseline --days 250 --seed 0 --save results --no-show
```

The last command prints simulation statistics and saves price, mean-wealth, and
wealth-distribution plots under `results/`. Evolution runs also save trader-count
plots. Omit `--no-show` to display the figures interactively.

The default run lasts 1,000 trading days, following a 20-step price warm-up.
Always pass `--seed` when you need reproducible results; runs are unseeded otherwise.
Plots use scenario-based filenames, so saving the same scenario again overwrites
its previous plots.

For a full run that does not overwrite previous output:

```bash
.venv/bin/python experiments/run_scenario.py evolution --days 250 --seed 0 --bundle results --no-show
```

`--bundle` and `--save` are mutually exclusive. A bundle gets a unique folder with
configuration, RNG states, environment/source provenance, analysis diagnostics,
compressed numerical histories, final trader states, events, and PNG plots.
Every trader's wealth history is included. A `COMPLETE` file marks a successful
export; interrupted exports remain incomplete. Bundle naming uses no NumPy
random draws. Without `--no-show`, the bundle is completed before plots are shown.

For the recorded numerical environment, install `requirements-lock.txt` instead
of `requirements.txt`. The lock records runtime dependencies and their transitive
dependencies from CPython 3.12.9 on Linux x86_64. `requirements.txt` remains the
minimum-version dependency list. Exact reproduction also requires the same source,
configuration, and RNG state; a seed alone does not guarantee results across versions.

## Scenarios

| Scenario | Noise | Chartists | Fundamentalists | Variation |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 50 | 20 | 30 | 20-step chartist window |
| `more_noise` | 500 | 20 | 30 | More noise traders |
| `more_chartists` | 25 | 130 | 15 | Chartist-heavy composition |
| `more_fundamentalists` | 50 | 20 | 300 | More fundamentalists |
| `circuit_breaker` | 50 | 20 | 30 | Trading halt mechanism |
| `evolution` | 50 | 20 | 30 | Strategy changes every 100 days |
| `window_5` | 50 | 20 | 30 | 5-step chartist window |
| `window_50` | 50 | 20 | 30 | 50-step chartist window |

Counts are initial populations. Configurations and shared endowments live in
[`src/scenarios.py`](src/scenarios.py). Add a `Scenario` there to expose it through
the runner automatically.

## Research parameter controls

The following keyword-only `Scenario` fields expose existing model parameters
and optional interventions. Defaults retain the original model and random-draw
ordering. Existing positional `Scenario` arguments are unchanged.

| Python field | CLI flag | Default | Accepted values |
| --- | --- | --- | --- |
| `chartist_alpha` | `--chartist-alpha` | `0.1` | Finite real, `0 < value <= 1` |
| `fundamentalist_gamma` | `--fundamentalist-gamma` | `2` | Finite real, `value > 0` |
| `closing_noise_scale` | `--closing-noise-scale` | `0.05` | Finite real, `value >= 0` |
| `dividend_mode` | `--dividend-mode` | `stochastic` | `stochastic`, `constant` |
| `order_lifetime` | `--order-lifetime` | `persistent` | `persistent`, `day` |

Booleans, nonnumeric values, NaN, infinity, and invalid choices are rejected
before logs, RNG draws, or output directories are created. Numeric controls from
NumPy and other real scalar types are normalized to native Python numbers so
their arithmetic precision is retained on JSON replay.

```bash
.venv/bin/python experiments/run_scenario.py baseline --days 250 --seed 0 \
  --chartist-alpha 0.3 --fundamentalist-gamma 4 --closing-noise-scale 0 \
  --dividend-mode constant --order-lifetime day --bundle results --no-show
```

Omitted flags retain the selected scenario's settings. The run header prints all
five resolved controls. To create a configuration in Python without changing the
shared scenario registry:

```python
from dataclasses import replace
import numpy as np
from src import SCENARIOS, simulation

scenario = replace(SCENARIOS["baseline"], chartist_alpha=0.3,
                   fundamentalist_gamma=4, closing_noise_scale=0,
                   dividend_mode="constant", order_lifetime="day")
np.random.seed(0)
result = simulation.simulate(scenario, days=250)
```

Allocation and risk parameters also apply to traders created by evolution.
Evolved chartists retain the legacy 20-observation window, even when initial
chartists use a different window. Closing noise changes only the coefficient in
`VWAP * (1 + closing_noise_scale * liquidity_factor * normal_draw)`; the price
floor and circuit-breaker behavior remain in effect.

Constant dividends are `0.016` throughout initialization, warmup, trading, and
halts. The stochastic dividend draws (including conditional shocks) still occur
before the proposed dividend is replaced. A zero closing-noise coefficient also
retains the normal draw after executions; no executions still means no closing
draw or price update. Warmup prices and trader decisions/quotes remain random.
Equal seeds do not guarantee identical later shock paths across interventions:
changed decisions and executions can change subsequent draw consumption.

Day orders expire at the end of each trading day, after the existing daily
summaries, including any partial-fill remainders. Nothing carries into the next
day. An `orders_expired` event reports remaining `buy_orders`, `sell_orders`,
`buy_quantity`, and `sell_quantity` when there are residual orders. Expiry also
occurs with logging disabled. Persistent mode has no expiry.

Bundles retain schema version 1 and their existing file layout. The scenario
configuration includes all five controls; `shared_parameters` reports the actual
alpha/gamma, and `fixed_rules` records the resolved dividend mode, closing noise,
and order lifetime. Custom controls replay under the same strict source/runtime
checks as default runs. Older bundles still require their recorded source and
runtime; this change does not bypass provenance checks.

These controls complete task 1 of the research plan. The task 2 experiment runner
is described below. Parameter sweeps, deterministic reductions, bifurcation
analysis, and multifractal analysis remain pending (tasks 3–7).
The branch and PR merge sequence is tracked in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Resumable research experiments

The Linux experiment runner executes explicit, seeded cells in isolated processes.
It preserves the simulation loop and uses its streamed logs to retain every
trading day, including days without trades or price updates. It does not run the
legacy nonlinear estimators or create plots. No additional dependency is needed.

```bash
.venv/bin/python experiments/run_experiment.py validate experiments/specs/baseline_pilot.json
mkdir -p results
nohup .venv/bin/python experiments/run_experiment.py run experiments/specs/baseline_pilot.json results/task2-pilot > results/task2-pilot-console.log 2>&1 &
.venv/bin/python experiments/run_experiment.py status results/task2-pilot
nohup .venv/bin/python experiments/run_experiment.py resume results/task2-pilot --workers 2 --minutes 120 >> results/task2-pilot-console.log 2>&1 &
```

`run` requires a new destination. `resume` uses the immutable specification already
stored there. It validates and skips completed runs and restarts interrupted runs
from their seeds; it does not checkpoint a simulation partway through. Failed or
corrupt runs require `resume --retry-failed`. A lock prevents concurrent writers.
Exit codes are 0 for all runs complete, 2 for an incomplete batch (budget, signal,
or run failure), and 1 for an invalid request or supervisor error. `status` reads
the ledger; full artifact validation happens on resume or through `read_compact`.

The version-1 JSON specification has `schema_version`, `name`, `days`, `burn_in`,
and `cells`. Each cell has a unique `name`, an existing `scenario`, an explicit
`seeds` list, and optional `overrides` for any Scenario field except its name/title.
Unknown fields are rejected. For example:

```json
{
  "schema_version": 1,
  "name": "closing-noise-pilot",
  "days": 250,
  "burn_in": 50,
  "cells": [
    {"name": "reference", "scenario": "baseline", "seeds": [100, 101]},
    {"name": "zero-close-noise", "scenario": "baseline", "seeds": [100, 101],
     "overrides": {"closing_noise_scale": 0}}
  ]
}
```

Shared resolved scenarios, seeds, durations, and burn-in settings are deduplicated
across cells. Cell labels and worker counts do not determine run identity. Exact
execution source hashes, Python/platform information, and the installed runtime
dependency closure do. Resume rejects a changed execution fingerprint; unrelated
untracked documents and Git status changes alone do not invalidate results. Do not
edit execution sources or change dependencies during an experiment. Git revision,
dirty status, effective model configuration, and pre/post simulation RNG states
are recorded. Source hashes identify dirty source but do not archive it. Equal
seed integers across variants do not guarantee aligned later random shocks.

The Python interfaces use absolute imports:

```python
from src.experiments import Limits, execute_experiment, expand_spec, read_compact

# Blocks until its bounded coordinator exits: launch long calls in the background.
code = execute_experiment("results/pilot", spec="experiments/specs/baseline_pilot.json",
                          limits=Limits(workers=2), log_path="results/pilot-console.log")
# spec=None resumes; retry_failed=True explicitly retries failed/corrupt runs.
metadata, arrays, summary = read_compact("results/pilot/runs/<run_id>")
```

### Compact artifact format

Experiment manifests and compact run artifacts have their own version-1 schemas;
they are separate from schema-1 replay bundles and cannot be passed to
`replay_bundle`. The original bundle interface and strict replay checks remain.

| Location | Contents |
| --- | --- |
| `manifest.json`, `manifest.sha256.json` | Immutable input, resolved runs/cell membership, provenance and integrity hash |
| `ledger.sqlite` | Run states, attempts, invocation budgets, benchmark identity, errors, timings and peak RSS |
| `invocations/*.jsonl` | Bounded progress logs; console output also reports each transition |
| `runs/<run_id>/daily.npz` | Compressed numeric daily arrays; no pickle/object arrays |
| `runs/<run_id>/metadata.json`, `summary.json` | Effective settings, RNG states, array schema/units, summary definitions and results |
| `runs/<run_id>/logs/` | Complete compressed warm-up/trading summaries, strategy summaries, and events when present |
| `runs/<run_id>/complete.json` | File sizes and SHA-256 hashes, written after all run artifacts |
| `staging/<run_id>/` | Unpublished work and temporary logs; cleaned before retry |

Arrays include day, price, previous price, dividend, natural-log return, volume,
trade count, halt/skip flags, and bid/ask order counts after matching and before
expiry. Strategy arrays have columns noise/chartist/fundamentalist and retain
active counts, active-group cash, shares and wealth, and buy/sell volume. These
preserve existing active-trader accounting; they are not cohort performance or
liquidation values. Event trader IDs remain run-local indices without retained
individual snapshots. Detailed decisions and trader wealth histories are omitted.

All trading days remain in the arrays. Summaries select `[burn_in, days)`, and
day-zero returns use the actual price immediately before trading. Volatility is
the daily log-return sample standard deviation (`ddof=1`), without annualization.
Maximum drawdown is the nonnegative fractional loss from the running price peak,
including the price immediately before the selected window. Stalls are consecutive
zero-volume days, including halts; both longest and terminal streaks are reported.
Undefined summaries use JSON null with a reason. Raw daily values are never
jittered, stitched, or filtered to remove no-trade days.

### Resource limits and recovery

Defaults are two workers, one native numerical thread each, 120 minutes per
invocation, 6 GiB memory, 5 GiB experiment output, and a 10 GiB free-disk floor.
CLI flags are `--workers`, `--minutes`, `--memory-gib`, `--output-gib`, and
`--min-free-gib`. Workers must be 1–6; memory/output can only be lowered and the
free-disk floor can only be raised. Each invocation benchmarks its first pending
run serially and retains that result before admitting concurrent work.

Linux address-space limits reserve 768 MiB for the coordinator and 256 MiB for
probes; the remaining memory is divided among the configured workers, even during
the serial benchmark. Imports count against these conservative virtual-memory
limits. A run that cannot fit fails rather than falling back to an unbounded mode;
using fewer workers increases each worker's allowance. The supervisor also checks
RSS and stops when system `MemAvailable` falls below 1 GiB.

Output reservations count temporary files as well as published artifacts. Each
worker reserves at most 512 MiB, divided over 16 bounded file slots enforced by
`RLIMIT_FSIZE`; this intentionally can reject an unusually large individual file
before exhausting the aggregate quota. The fixed worker layout uses at most 14
files simultaneously. A 64 MiB coordinator reserve covers the bounded SQLite
database/journal, manifest, and invocation log. The resolved manifest is limited
to 4 MiB of canonical JSON; split larger specifications into smaller batches.
Invocation logs retain at most 64 KiB each; full progress is also written to the
caller-selected console log, which is outside the runner's managed quota.

The runner reserves disk headroom before dispatch and monitors free space during
execution; unrelated processes can still consume disk between checks. It stops
dispatching near the wall-time deadline using measured run duration, terminates
unfinished workers at the deadline, and leaves completed artifacts intact. A new
resume invocation gets a fresh time budget. SIGINT/SIGTERM leave recoverable
state, and Linux parent-death signals kill workers if the coordinator is killed.
Runs publish by same-filesystem rename only after validation and file syncing;
resume recovers publication that preceded a ledger update. Startup failure before
the immutable manifest is fully written requires a new destination.

Use one-run readers or streaming aggregation for ensembles. The runner itself
never loads logs or ensemble tables into pandas. The simulation still retains its
existing in-memory trader histories; the memory cap bounds them but does not
change that model behavior.

## How the model works

- **Noise traders** place randomized buy and sell orders.
- **Chartists** compare the current price with a moving average and adjust orders
  using recent volatility.
- **Fundamentalists** use dividends and excess returns to choose their holdings.

Each trading day updates dividends, applies scenario rules, collects orders, and
matches the best bids and asks. Trades update cash and holdings; executed volume
and a random liquidity adjustment determine the next price. Wealth is then
recorded for each trader. Group mean wealth is sampled every 100 days and on the
final day.

## Use from Python

```python
import numpy as np

from src import SCENARIOS, simulation
from src.plotting import plot_price

np.random.seed(0)
asset, traders = simulation.run_simulation(
    SCENARIOS["baseline"], days=250, verbose=False
)
figure = plot_price(asset, save_path="results/baseline_price.png")
```

`asset.price_history` and `asset.dividend_history` contain the asset paths.
Each trader exposes cash, holdings, activity status, and a wealth history.
`simulation.wealth_history` holds sampled group means and is overwritten by the
next run; copy it if you need to retain several runs.

For multiple runs, use the independent result API:

```python
from src.plotting import plot_wealth_history

np.random.seed(0)
result = simulation.simulate(SCENARIOS["baseline"], days=250)
analysis = simulation.analyze_result(result)
simulation.print_analysis(analysis)
figure = plot_wealth_history(result.wealth_history)
```

`simulate()` does not compute final metrics or update the compatibility global.
Call `analyze_result()` immediately to reproduce legacy metric values: RANSAC
uses the current global RNG stream. Analysis records its own starting/ending RNG
states and is cached, so repeated analysis and printing consume no extra random
draws. Treat a result's objects as read-only after analysis; manual edits do not
invalidate the cache. `run_simulation()` retains its tuple return, global history,
and historical RNG consumption, even with `verbose=False`. Quiet runs suppress
model messages, including circuit-breaker messages; estimator warnings remain visible.

Days and chartist windows must be positive integers. Trader counts must be
nonnegative integers, with at least one trader. The CLI accepts seeds from 0 to
4294967295. Invalid inputs fail before simulation or output creation.

To save or replay from Python:

```python
from src.bundles import save_bundle, replay_bundle

folder = save_bundle(result, "results", seed=0)
replayed = replay_bundle(folder)
```

Replay uses both recorded RNG states, including for unseeded runs or delayed
analysis, and restores the caller's RNG state afterward. It requires matching
source hashes, Python runtime, and runtime dependency versions. Restore the
recorded revision and any recorded dirty-source changes before replay. Source
hashes identify changes but do not archive source code; keep source and installed
packages unchanged between running and exporting.

Load numerical data with `np.load(folder / "histories.npz", allow_pickle=False)`.
`price` and `dividend` retain their historical observation indexing, including
warm-up; they are not guaranteed daily series. Group histories use `days`,
`noise_avg`, `chartist_avg`, and `fundamentalist_avg`. Population series use
`population_days` and `population_<type>`. For trader index `i`, take
`trader_wealth[trader_wealth_offsets[i]:trader_wealth_offsets[i + 1]]`.
Trader indices are local to the run. Structured evolution events link the old and
new indices when one trader is replaced.

## Structured logging

Every `simulate()` run records compact summaries and events by default, independently
of `verbose`. Bundles retain them under `logs/`; the console keeps its progress and
final report. `--save` still saves plots only. Metric failures now include their
status and reason in the console; full estimator settings and diagnostics remain
in `analysis.json`.

| File in a bundle | Contents |
| --- | --- |
| `logs/steps.jsonl.gz` | Each warm-up step and trading day: price, simple return, dividend; trading rows also contain trade count/volume/notional, closing spread and outstanding orders, halt state, and price formation |
| `logs/strategies.jsonl.gz` | Each trading day/type: submitted buy/sell order counts and quantities, executed buy/sell volume and notional, active population, total cash/holdings/wealth, average cash/wealth |
| `logs/events.jsonl.gz` | Circuit-breaker triggers and resumptions, skipped trading days, evolution links and selection rule, resource-based deactivations, day-order expiry; absent if no events occurred |
| `logs/trades.jsonl.gz` | Optional: each fill's buyer/seller IDs and types, execution price/quantity, cash and holdings immediately before and after |
| `logs/decisions.jsonl.gz` | Optional: each evaluated trader's inputs, decision reason, intended order, cash/holdings, and whether an order was submitted |
| `logs/metadata.json` | Log schema version, options, filenames, written counts, caps, and omitted detail counts |
| `analysis.json` | Metric value/status/reason, estimator, parameters, library defaults, input counts, library version, and final run summary |

Detailed modes require a bundle in the CLI:

```bash
# Default: complete daily summaries and events
.venv/bin/python experiments/run_scenario.py baseline --days 250 --seed 0 --bundle results --no-show

# Include individual transactions
.venv/bin/python experiments/run_scenario.py baseline --days 250 --seed 0 --bundle results --log-level trades --no-show

# Include decisions and trades for two traders during days 90–110
.venv/bin/python experiments/run_scenario.py evolution --days 250 --seed 0 --bundle results --log-level decisions --log-trader 0 --log-trader 50 --log-start-day 90 --log-end-day 110 --no-show
```

`--log-level` accepts `off`, `summary` (default), `trades`, or `decisions` (includes
trades). Repeating `--log-trader ID` selects multiple traders. A trade matches if
either participant is selected. Day bounds are inclusive and zero-based; filters
affect detail only. Trader IDs match `traders.json` indices, remain stable within
the run, and append as evolution creates traders. Unknown IDs produce no matching
detail. A halted day has a skip event rather than fabricated trader decisions.

Detail defaults to **100,000 records or 64 MiB of uncompressed JSON**, whichever
limit is reached first, shared across trades and decisions in execution order.
Adjust with `--log-max-records` and `--log-max-bytes`. Once capped, remaining eligible
detail records are counted as omitted. `detail_truncated` and `omitted_by_cap` in
metadata identify this explicitly; verbose runs also print a notice. Filtered-out
records are outside the requested scope and are not counted as cap omissions.
Summaries and events remain complete regardless of detail caps or filters.

Logs stream to compressed temporary files, rather than accumulating records in
RAM. They persist with the result until exported or released. In Python:

```python
from src import LogOptions, simulation
from src.bundles import save_bundle

result = simulation.simulate(days=120, log_options=LogOptions(
    level="decisions", trader_ids=(0, 50), start_day=0, end_day=9,
))
analysis = simulation.analyze_result(result)  # retain legacy metric RNG timing
for row in result.run_log.iter_records("steps"):
    if row["phase"] == "trading" and row["day"] == 0:
        print(row)
folder = save_bundle(result, "results")
result.run_log.cleanup()  # optional immediate release; cannot read/export logs again
```

Temporary files are also released when the log object is garbage-collected. The
legacy tuple API exposes the same log through `asset.run_log`. Use
`LogOptions(level="off")` to disable structured collection. Replay restores the
recorded logging options as well as the RNG states. Read exported files without
loading a whole table:

```python
import gzip
import json

with gzip.open(folder / "logs" / "steps.jsonl.gz", "rt") as stream:
    for line in stream:
        row = json.loads(line)
        # Process one observation at a time.
```

Interpretation details:

- `phase` distinguishes warm-up from trading in step/event records. The other
  streams refer only to trading days. Step `return` is the simple price change
  against that step's starting price; nonlinear metrics still use log-returns.
- Trading summaries describe the book **after matching and before expiry**.
  Outstanding quantities can include persistent, possibly unaffordable orders.
  Day-order expiry events reconcile to these quantities. The spread is `null` if either
  side is empty; it can remain crossed when matching cannot proceed.
- `price_formation` records VWAP, liquidity factor, the proposed noisy price before
  and after the floor, and whether the asset accepted it. No executions means
  `price_formation: null`; a rejected update can have positive volume and zero price
  change. Trade volume counts each fill once. Buy/sell strategy volumes count each
  side separately; adding both doubles market volume.
- Strategy balance/wealth aggregates cover **active traders at day end**. Activity
  covers every submitted/executed order, including persistent orders belonging to
  traders who later became inactive. Empty active groups have `null` averages.
- Evolution selection is uniform random among active traders of the source type;
  `selection_score` is `null` because the model has no performance-based selection.
  Events give the active counts after each replacement. No rule is inferred from
  a trader's profitability.
- Decisions record already-computed values: noise signal draws, chartist moving
  averages and buy volatility/correction, fundamentalist excess returns and target
  holdings. Reasons follow the implemented branches: chartists' `sell_fallback`
  may also occur above the moving average when there is no cash. Inputs are present
  only when their branch evaluated them. `submitted` does not imply execution.
- Logs add no random draws or model decisions. They explain observed activity;
  causal strategy claims still require controlled comparisons. There are no unique
  order IDs or a complete order amendment/removal ledger in this logging schema.

[`Multiagent_Modelling_notebook.ipynb`](Multiagent_Modelling_notebook.ipynb) uses
the same package. Select a Jupyter kernel with the project dependencies installed;
configure `SEED` and `DAYS` in its setup cell.

## Metrics and index data

Metrics operate on **log-returns**, computed from the supplied price series:

| Output | Estimator | Settings |
| --- | --- | --- |
| Hurst exponent | `nolds.hurst_rs` | R/S analysis |
| Lyapunov estimate | `nolds.lyap_r` | Rosenstein; runner uses embedding dimension 10, lag 1 |
| Sample Entropy (nats) | `nolds.sampen` | Embedding dimension 3; tolerance 0.2 × return standard deviation |

The public Lyapunov helper defaults to dimension 5; simulation and index reports
explicitly use 10. Sample Entropy uses tolerance 0.01 if return variance is zero.
Short series or estimator failures return `NaN`. The historical function name
`calculate_kolmogorov_entropy` remains a compatibility alias for
`calculate_sample_entropy`; it does not estimate Kolmogorov–Sinai entropy.

`analysis.metrics` contains structured `MetricResult` records: value, status,
reason, estimator, explicit parameters, library defaults, observation counts,
and nolds version. Statuses are `ok`, `insufficient_input`, `estimator_error`, and
`non_finite`. Short inputs skip preprocessing, so their return count is `None`.
The scalar helpers keep their historical returns, including infinite estimates;
bundles store non-finite JSON values as `null` alongside diagnostics. For an
individual diagnostic, call `metric_result("lyapunov", prices, emb_dim=10, lag=1)`.

`data/` contains CSV exports for RTS, MOEX, S&P 500, DJIA, and NASDAQ.
`src/index_data.py` parses their Russian headers and localized numeric format.

## Reproducibility and model caveats

Default settings preserve the original notebook's numerical behavior. In particular:

- Random draws use the global NumPy stream. RANSAC metric fitting also uses
  randomness; seed before each run or standalone metric calculation. Dependency
  versions can affect results.
- Prices include a 20-step exogenous geometric Brownian motion warm-up.
- Unmatched orders persist between days by default. Large populations and long runs can be
  slow as the order book grows.
- Group wealth plots show means over active traders, rather than total wealth.
  Evolved traders are omitted from the original-trader profitability report.
- Evolution retains the notebook's inherited circuit-breaker asset without halt
  checks. This is a legacy behavior, not an independently designed control.
- Index rows retain the exports' newest-first order for legacy reproducibility.
  Chronological comparisons require a deliberate change to that convention.

The historical result tables are not presented here as validated benchmarks.
The code audit and research backlog are recorded in [`CONTEXT.md`](CONTEXT.md)
and [`CLAUDE.md`](CLAUDE.md).

## Repository layout

```text
src/
    asset.py          Asset prices, dividends, and trading halts
    traders.py        Trader state and decision rules
    order_book.py     Order storage and matching
    scenarios.py      Scenario configurations and shared parameters
    simulation.py     Result objects, simulation loop, cached analysis and reporting
    bundles.py        Compressed run export and RNG-state replay
    run_logging.py    Streaming structured logs, detail filters and limits
    metrics.py        Log-return estimators
    plotting.py       Reusable plotting functions
    index_data.py     Market-index CSV loading and reports
experiments/
    run_scenario.py   Command-line entry point
    benchmark_order_book.py  Sequential legacy/current performance comparison
tests/
    test_legacy_equivalence.py
    test_core.py
    test_bundles.py
    test_logging.py
data/                Market-index CSV inputs
Multiagent_Modelling_notebook.ipynb
```

## Development and validation

Research controls are opt-in and preserve the original defaults. Experiment
infrastructure, ensemble studies, reductions, and new estimators remain separate work.
Keep reusable logic in `src/`, scenarios in the configuration module, and preserve
random-draw order when refactoring.

The standalone regression runner compares prices, dividends, trader balances,
wealth histories, and metrics against the original notebook in Git commit
`84fa2a6`. Keep that commit available in your local history.

```bash
# Focused correctness and artifact checks
.venv/bin/python tests/test_core.py
.venv/bin/python tests/test_bundles.py
.venv/bin/python tests/test_logging.py

# Quick check: baseline, 120 days, seed 0
.venv/bin/python tests/test_legacy_equivalence.py --scenario baseline --days 120 --seeds 1

# Default suite: all scenarios, 250 days, seeds 0–2, plus a circuit-breaker case
.venv/bin/python tests/test_legacy_equivalence.py

# Extended suite: run in the background and inspect the log
.venv/bin/python -u tests/test_legacy_equivalence.py --full > /tmp/abm-regression.log 2>&1 &
```

`--seeds N` runs seeds `0` through `N - 1`. The default suite can take several
minutes; the extended suite takes substantially longer. Numerical equivalence
checks preserve behavior, rather than validate the model's scientific claims.

The regression suite also compares final NumPy RNG states before reseeding for
standalone metric checks. Focused tests exercise matching conservation and order
priority, halts, independent results, diagnostics, and bundle replay/failure cases.
For background benchmarks and long regression runs, set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` to bound native threading. The benchmark
prints timings and the process peak RSS (KiB on Linux); it compares the historical
order book with the current implementation on the same seeded workload.
