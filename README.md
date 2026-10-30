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

## Parameter atlas (task 3)

The atlas crosses chartist alpha and fundamentalist gamma with all eight
combinations of closing noise `{0, 0.05}`, dividends `{constant, stochastic}`,
and order lifetime `{day, persistent}`. The stochastic/persistent/noise-0.05
variant is the named legacy stochastic reference. Composition and endowments
are baseline, the SMA window is 20, and evolution and both breaker switches are
off. Removing these stochastic inputs does not make trader decisions deterministic.

The design has a 17-point geometric alpha sweep from 0.025 to 0.4 at gamma 2,
plus a 9-by-7 map with gamma from 0.5 to 8. Grid alpha values are taken directly
from every second sweep value. Five seeds (100–104), 2,000 trading days, and a
1,000-day burn-in give 568 configurations and 2,840 unique runs after overlap
deduplication. Seeds are the outer execution loop; variants alternate at each
coordinate. Equal seeds do not guarantee aligned shocks across variants.

Run the complete study in the background, redirecting driver output to a log:

```bash
.venv/bin/python experiments/run_atlas.py study results/task3_atlas_20261027 \
  > /tmp/task3-study.log 2>&1 &
```

Repeating that command resumes the study. The driver benchmarks the first
simulation and analysis in each invocation, repeats only time-limited batches
(120 minutes by default), and stops on failures or resource limits. Defaults are
two workers and one native numerical thread each. `--workers` accepts 1–6 and
`--minutes` accepts up to 120. The simulation gets 4.5 GiB of the total 5 GiB disk
budget; analysis and figures share the remainder. Child coordinators get a
5.5 GiB memory budget, leaving headroom inside the 6 GiB total for the small
driver. All stages preserve a 10 GiB free-disk floor. Do not run simulation and
analysis concurrently; both take the experiment lock.

Individual stages and Python interfaces are also available:

```bash
.venv/bin/python experiments/run_atlas.py generate /tmp/atlas-spec.json
.venv/bin/python experiments/run_experiment.py run /tmp/atlas-spec.json \
  results/my_atlas/simulation --output-gib 4.5
.venv/bin/python experiments/run_atlas.py analyze results/my_atlas/simulation
.venv/bin/python experiments/run_atlas.py render results/my_atlas/simulation
.venv/bin/python experiments/run_atlas.py validate-estimators /tmp/atlas-controls.json
.venv/bin/python tests/test_atlas.py
```

`src.atlas.make_spec`, `analyze_arrays`, and `aggregate` are reusable numerical
interfaces. `src.atlas_runner.analyze_experiment` is a CLI coordinator entry
after process-limit bootstrap; use the CLI from an ordinary Python application.
`src.atlas_plotting.render_atlas` creates figures. Custom small specifications can
be generated using `make_spec(seeds=..., days=..., burn_in=...)` and validated
using the existing `expand_spec`. Run directories must be nested under a
dedicated atlas root, as shown above: the parent contains analysis and reports
and is the scope of the total disk quota.

Analysis reads one validated compact run at a time and never loads daily
ensemble tables into pandas. It uses the stored chronological daily log returns
in `[burn_in, days)`, including zero-volume days and the actual preceding price
for drawdown. These research metrics use calendar trading days; legacy report
helpers retain their price-observation inputs. Volatility uses `ddof=1`;
positive-volume volatility needs two trading days. Streak lengths count days
within the retained window. Terminal
stall incidence is the proportion of runs with at least 100 terminal no-trade
days. Other outputs include final price, window log return, mean daily volume,
drawdown, zero-volume fraction, and longest/terminal no-trade streaks.

The separate research wrapper reuses installed `nolds` without changing the
legacy metrics API. Sample Entropy (nats) uses dimension 3, lag 1, Chebyshev
distance, an open tolerance interval, and tolerance `0.2 * std(returns, ddof=0)`.
The Rosenstein estimate (per trading day) uses dimension 10, lag 1, temporal
exclusion 20, trajectory length 20, fit offset 0, `tau=1`, and RANSAC. A separate
fit seed is the first eight hex digits of SHA256 of canonical JSON
`[simulation_seed, "rosenstein"]`; global RNG state is restored after fitting.
Settings, fit trajectories, warnings (including any library fit fallback), and
runtime/source fingerprints accompany results. Constant/nonfinite/short inputs
and nonfinite results have explicit statuses. No jitter, sample removal, or
segment stitching is used. Each worker has a five-minute timeout.

Analysis records are checksummed and published atomically under
`analysis/<analysis-id>/runs/`. Their identity includes settings, analysis source
and dependencies, simulation run ID, and artifact completion-manifest hash.
Invalid cache records are recomputed; corrupt simulation input is rejected.
Operational failures stop completion and can be retried with `analyze --retry-failed`.
Changing analysis settings or source creates a separate cache. Changing any
`src` file also changes the existing simulation fingerprint: freeze execution
sources before a study, and restore recorded source to resume historical runs.
Historical rendering accepts `--analysis-id` without requiring current model source.

Reports contain `summary.json` (per-cell quartiles, valid/expected seed counts,
and status counts), `per_run.jsonl.gz`, `tables.npz` (load with
`allow_pickle=False`), and 24 figures in both PNG and PDF. Heatmaps share scales
across interventions and mask incomplete/undefined cells; sweep bands show
quartiles and mark partial estimates. Stall plots use frequencies rather than
medians of binary indicators. `render.json` records output hashes and coverage.

Five seeds support exploration, not confirmatory inference. Quartile bands are
between-seed dispersion, not confidence intervals. Rosenstein estimates on these
short stochastic return series are diagnostics, not evidence of deterministic
chaos. Known-system validation uses logistic-map **levels** and an explicitly
separate short-fit configuration; production settings are also checked on
periodic, chaotic, constant, and Gaussian controls. See
[nolds methods](https://cschoel.github.io/nolds/nolds.html) and
[validation examples](https://nolds.readthedocs.io/en/stable/examples.html).

The [completed task 3 study](VALIDATION.md#research-parameter-atlas-task-3-2026-10-27)
records all 2,840 simulations and analyses, findings, limitations, and task 4
anchors. Representative [volatility](experiments/figures/task3_20261027/log_return_volatility_map.png)
and [terminal-stall](experiments/figures/task3_20261027/terminal_stall_map.png) maps
are retained in Git; the full 24-figure set and compact data remain under `results/`.

## Portfolio detail and structure discovery (task 5)

Task 5 tests how much individual portfolio information the full ABM needs.
It compresses portfolios **once at saved checkpoints**, then resumes ordinary
individual trading. The population, independent decisions, ordered book, ownership,
and accepted-price memory remain. This is an experiment on initial information;
it does not yet provide a permanently reduced simulator or a speedup. No governing
market equations are derived, and the observed structures are not chaos claims.

`src.market_checkpoints` provides `MarketCheckpoint`, `run_checkpointed`,
`continue_checkpoint`, and gzip checkpoint I/O. Snapshots preserve stable trader
indices, ordered queues, asset/trader state and histories, elapsed days, and the
NumPy RNG. NumPy scalar types in transition inputs are preserved: converting them
to Python floats changes Python 3.12 summation rounding. Historical wealth values
are stored as ordinary floats because they are not transition inputs. Research
entry points restore the caller's RNG. Evolution and circuit-breaker checkpoints
are rejected; the ordinary APIs and schema-1 bundles remain unchanged.

`src.heterogeneity.compress_checkpoint` accepts a `CompressionConfig(level=...)`.
Levels are 1, 2, 4, 8, 16, or `full` by default. Within each strategy, Ward clustering
uses cash and shares divided by the initial per-agent endowments. Nested tree cuts
preserve joint portfolio information. Inactive/empty portfolios stay unchanged.
Cluster cash is averaged with a floating-point residual correction; integer shares
are distributed as floor/ceiling values, with extra shares assigned by descending
original holdings and then stable index. Hence a cluster may retain two share
profiles; requested and realized counts are reported separately. Queues and their
owners remain unchanged. No compression is performed at `full`.

Retained dispersion is the ratio of summed squared normalized deviations after
and before compression, separately by strategy; zero original dispersion is
undefined. Joint Wasserstein-1 error uses SciPy's minimum-cost assignment between
equal-sized, uniformly weighted individual portfolio samples. This is the exact
empirical transport distance and avoids a separate LP solver. Compression records
also retain memberships, assignment error, and changes in resource availability.

```bash
.venv/bin/python experiments/run_heterogeneity.py generate results/task5_heterogeneity_20261030 \
  --atlas results/task3_atlas_20261027
.venv/bin/python experiments/run_heterogeneity.py run results/task5_heterogeneity_20261030 \
  --workers 2 --minutes 120 > results/task5_heterogeneity_20261030/execution.log 2>&1
.venv/bin/python experiments/run_heterogeneity.py status results/task5_heterogeneity_20261030
.venv/bin/python experiments/run_heterogeneity.py resume results/task5_heterogeneity_20261030 \
  --workers 2 --minutes 120 >> results/task5_heterogeneity_20261030/execution.log 2>&1
.venv/bin/python experiments/run_heterogeneity.py render results/task5_heterogeneity_20261030
.venv/bin/python tests/test_heterogeneity.py
```

Run long commands in a background terminal session. The default pilot selects five
stored atlas configurations: reference variants 0/6/7 and liquidity variants 6/7.
It uses development seeds 300–309 and held-out seeds 310–359, 2,000-day reference
runs, checkpoints after 100/500/1,000 completed days, and 1,000-day continuations.
There are 300 references and 900 checkpoint jobs containing up to 5,400
continuations; identical compressed snapshots are deduplicated within each job.
Every full-detail continuation must match reference daily outputs and its final
physical-state/RNG digest before that job can be published. Equal starting RNG
states do not imply aligned future shocks after decisions consume different draws.

Artifacts and the two phase ledgers are separate from existing experiment/bundle
formats. Checkpoints are deterministic gzip JSON with typed numerical scalars;
daily and thinned feature arrays use compressed NPZ without pickle. Completion
manifests validate every payload. Resume verifies source/runtime and input identity;
corrupted outputs require `--retry-failed`. Each phase benchmarks serially before
using two workers. Workers have address-space and file-size limits; the study
retains the 6 GiB memory, 5 GiB output, 10 GiB free-disk, six-worker maximum, and
120-minute batch limits. Repeated bounded resumes complete the fixed pilot.

Fidelity is assessed at 100 and 1,000 future days. Primary tolerances are 10
percentage points in terminal-stall frequency; Wasserstein distance 0.05 in
no-trade fractions and each strategy's final resource fractions; and `log(1.2)`
in log(1 + mean volume), log(return volatility + 0.0001), and cumulative log return.
Offsets define comparison scales; no jitter is added. Terminal stalls use 100
future no-trade days, without borrowing pre-checkpoint history. Secondary outputs
include 1/20-day diagnostics, return quantiles, positive-volume volatility, and
within-strategy/same-owner volume. Full-ABM split-seed comparisons describe sampling
variability and do not define equivalence tolerances.

The predeclared criteria use 2,000 matched seed-bundle bootstrap samples and
conservative paired-binomial stall bounds, including zero-disagreement cases.
Results are within tolerance, outside tolerance, or inconclusive. A candidate
minimum level must pass along with every finer tested level. These are exploratory
per-cell/horizon intervals, not simultaneous certification across the entire study.

Structure discovery fits scaling, PCA (2/4/8 components), and K-means (2–6 clusters,
20 initializations) on development reference runs only. Clusters are selected by
development silhouette score. Current features are compared with lags of 1/5/20
trading days; rows are sampled every tenth day after a 20-day lag window. Portfolio
quantiles, resource availability, book summaries, trading activity, and accepted-price
signals form the features. Validation covers reconstruction, seed-bootstrap cluster
stability, occupancy, and subsequent 20-day outcomes. Compressed continuations are
projected using those frozen reference-trained models. The report contains numeric
records, transform arrays, and PNG/PDF figures, with independently recorded rendering
provenance. See VALIDATION.md for completed checks and pilot findings.

The current feature vector already includes accepted-price memory signals and the
no-trade streak. The lagged variant appends earlier feature vectors to test whether
they contain useful additional information; it leaves the trader decision rules
unchanged. Cluster stability is conditional on the frozen development scaler/PCA;
five seed-bootstrap repetitions refit K-means, not the complete representation pipeline.
Reconstruction errors describe each representation's own feature space. The
subsequent 20-day targets are shared across modes and provide the direct comparison
of predictive usefulness. The forecast baseline is the development cohort's global
mean; stronger forecasting baselines remain a subsequent validation step.

The [completed pilot and validation record](VALIDATION.md#retained-individuality-and-structure-discovery-task-5-2026-10-30)
contains 300 references and 5,400 continuations across 900 checkpoint jobs.
All full-detail continuations reproduced their references exactly. Eleven of
30 tested conditions have candidate compression thresholds; the pilot does not
support a universal cutoff. Both representations find five stable clusters,
with no consistent predictive advantage from adding lags. These findings concern
one-off portfolio information, while all individual trading remains in place.

The recorded pilot used four workers after an initial two-worker benchmark and
user approval; CLI defaults remain two. Its execution source is `6db78d5`.
Historical resume requires that source/runtime; the current renderer can read
the completed artifacts with separately recorded rendering provenance.

## Hard-rule deterministic reduction (task 4)

The reduction is an explicit research variant, separate from the unchanged
stochastic ABM. `src.reduction` exposes `ReductionConfig`, `ReductionState`,
`initial_state`, `step_hard(state, config)`, and `simulate_reduction`. A step
returns a new state and pre-expiry diagnostics without consuming randomness or
modifying its input. `simulate_reduction` returns an independent result with
daily numeric arrays and initial/final states. Seeds identify recorded initial
histories; they do not drive the deterministic dynamics.

The default state has six strategy-resource coordinates (cash and shares for
noise/chartist/fundamentalist groups) and 20 accepted-price observations. Price is
the final observation. The day-order map has an empty book at each day boundary;
the persistent companion additionally retains every outstanding price-priority,
FIFO-tie order parcel and its strategy owner. The full persistent state can grow
even with constant prices. A bounded price trace does not establish a bounded
attractor. Future continuation of the day map must restrict the six resource
coordinates to their two conserved-total constraints.

Both maps use constant dividends `0.016`, zero closing noise, fixed population,
and no circuit breaker or evolution. Dividends and the risk-free rate enter the
fundamentalist signal only; they do not pay cash into the system. The initial
strategy resources are `[5000000, 3000000, 30000000]` cash and
`[50000, 20000, 15000]` shares. Total cash `38000000` and shares `85000` are
conserved, subject to floating-point tolerance.

### Closure and equations

For strategy `s`, decisions use per-agent means `c_s = C_s/N_s` and
`a_s = A_s/N_s`. A conditional legacy order at quote draw `u_k` contributes
`N_s * pi_branch * w_k * quantity(c_s, a_s, history, u_k)` shares at its original
limit price. Noise buy/sell probabilities are each `0.4`; the remaining `0.2`
holds. Chartists follow the original SMA branch and sell fallback. Fundamentalists
use `P/P_previous - 1 + 0.016/P_previous - 0.0003`, the strict absolute `0.05`
activation threshold, and allocation clipped to `[0.1, 0.9]`.

Uniform quote draws are replaced with Gauss–Legendre nodes and weights mapped
to `[0, 1]`: 16 points by default, with 8/32-point sensitivity runs. The original
integer decisions, minimum-one quantities, holdings limits, and affordability
conditions are applied **before** weighting. Fundamentalists need only one
deterministic quote. Absent strategies submit nothing.

Orders are submitted in noise/chartist/fundamentalist order, noise buys before
sells, then ascending node order. Matching retains best-price priority, stable
arrival ties, midpoint execution, and bid-price integer cash clipping on the
group resource pools. Quantities and fills can be fractional after weighting.
Within-strategy fills count toward volume and VWAP but have zero net resource
transfer. The closing price is `max(VWAP, 0.01)`. Its observation is appended only
when volume is positive, even if the closing price is unchanged. No-trade days
freeze the entire accepted-price memory. Day expiry occurs after diagnostics.

This is a pooled-portfolio, weighted-order closure. It discards within-strategy
resource dispersion and allows old orders to draw on group resources. Matching
weighted orders is not the expected matched transition of the stochastic ABM.
Fractional inventories can sustain positive flows below one share and advance
the signal clock after discrete agents would stop. The report therefore keeps
exact zero-volume stalls and `0 < volume < 1` days separate, without epsilon
cutoffs or forced rounding. Quadrature convergence does not establish ABM
fidelity; order-parcel counts also depend on quadrature resolution.

[Mesa's activation facilities](https://mesa.readthedocs.io/latest/tutorials/2_agent_activation.html)
and [PyDSTool's hybrid-system facilities](https://pydstool.github.io/PyDSTool/HybridSystems.html)
were assessed. The model-specific closure is implemented directly; existing
[NumPy quadrature](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html),
artifact utilities, and bounded process infrastructure are reused. No dependency
was added. There is no smoothing, stability calculation, or chaos classification
in this task.

### Comparison workflow

```bash
.venv/bin/python experiments/run_reduction.py generate results/task4_reduction_20261027 \
  --atlas results/task3_atlas_20261027
.venv/bin/python experiments/run_reduction.py run results/task4_reduction_20261027 \
  --workers 2 --minutes 120 > results/task4_reduction_20261027/execution.log 2>&1
.venv/bin/python experiments/run_reduction.py status results/task4_reduction_20261027
.venv/bin/python experiments/run_reduction.py resume results/task4_reduction_20261027 \
  --workers 2 --minutes 120 >> results/task4_reduction_20261027/execution.log 2>&1
.venv/bin/python experiments/run_reduction.py render results/task4_reduction_20261027
.venv/bin/python tests/test_reduction.py
```

Launch long execution/validation commands in a background terminal session.
Generation selects Task 3's stored grid `(4, 3)` reference and all eight variants,
plus grid `(6, 0)` variants 6/7 for the exploratory liquidity contrast. It retains
the exact stored alpha value rather than retyping `0.2`. The default pilot reuses
50 historical ABMs (seeds 100–104), runs 50 fresh ABMs (110–114), and evaluates
four reduction configurations under ten recorded warm-up histories and three
quote resolutions: 120 deterministic runs. All runs have 2,000 trading days;
summary metrics use `[1000, 2000)`, while path figures show the full run.

Zero-noise, constant-dividend cases are the primary fidelity comparisons. Other
reference controls quantify differences when forcing is removed. Historical and
fresh cohorts remain separate. Quartile bands describe dispersion across seeds
or initial histories, not confidence intervals. Fresh five-seed results are still
exploratory. Equal seeds do not guarantee aligned later ABM shocks.

The driver runs the existing ABM coordinator first, then isolated reduction
workers. Each phase benchmarks one run before enabling two workers by default.
Native numerical threads are capped at one; never exceed six workers. Combined
process budgets stay within 6 GiB, output within 5 GiB, and free disk above 10 GiB.
The default batch lasts at most 120 minutes. Exit code 2 indicates interrupted,
limited, or incomplete work; inspect status and logs before resuming. Failed
reduction artifacts require `resume --retry-failed`. Corrupt scientific inputs
are rejected; restore their original recorded files or create a new study.

Completed reductions are atomically published under `reduction/runs/` and tracked
in SQLite. Interrupted individual runs restart from their recorded initial state;
validated completed runs are retained. A separately versioned format contains
`daily.npz`, metadata, a compressed final state including the persistent queue,
summary statistics, and a checksum manifest. Read numeric archives with
`allow_pickle=False`; `read_reduction` validates identity, layout, chronology,
conservation, and summaries. Existing schema-1 bundles and compact ABM formats
are unchanged. Execution source/runtime fingerprints gate resume, while historical
input artifacts can be read under newer code. Freeze execution sources before a
study; changing any `src` file changes its identity.

Rendering requires complete, valid inputs. It writes per-run compressed JSON,
numeric tables, per-group coverage and quartiles, and six figures in PNG/PDF:
reference prices, liquidity paths, two resource-allocation panels, trading
continuity, and quadrature sensitivity. Price axes are logarithmic; activity axes
retain zero. Report identity includes source/runtime and every input completion
hash. The [completed pilot and validation record](VALIDATION.md#hard-rule-deterministic-reduction-task-4-2026-10-27)
documents a material closure failure: fresh liquidity-anchor ABMs stall in 5/5
persistent-order runs, versus 1/5 primary reductions, with large volume and price
gaps. Read these findings before interpreting the reduced dynamics.

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
is described below; the task 3 atlas and task 4 hard reduction are described above.
Smoothing, bifurcation analysis, and multifractal analysis remain later tasks.
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
