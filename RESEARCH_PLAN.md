# Emergent dynamics research: PR integration plan

Integration branch: `research/emergent-dynamics`.
Integration PR: [#3](https://github.com/Roman223/Agent_modelling/pull/3).
Final target: `main`, through a draft integration PR that remains open until the
selected research plan is complete. The scope covers parameter discovery,
ABM reduction fidelity, data-driven structure discovery, and multifractal
analysis. Deterministic bifurcation verification is conditional on finding a
suitable, validated model. Basin/boundary geometry (research phase 4) is deferred.
The approved 2026-10-30 revision replaces task 5's smoothing study with a study
of retained individual differences; see the revised scope below. The approved
2026-11-17 priority revision put Task 7 next, including all five existing market
indices. Its fixed pilot is now validated. The approved 2026-11-24 Task 6
specification studies market inactivity, its mechanisms, and controlled recovery.

## Sequential PRs

Execution task numbers below are distinct from the research phase numbers.
Each task has its own branch and PR targeting `research/emergent-dynamics`.
All seven tasks, including Task 6's completed inactivity pilot and final audit,
are integrated. The combined result awaits the user's final validation in PR #3.
Create each subsequent task branch from
the updated integration branch after its predecessor has merged; this keeps its
PR focused on that task's changes. Use merge commits to retain the task history.

| Task | Branch | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | `feat/research-parameter-controls` | Python/CLI controls, compatibility checks, logging and bundle replay | Merged via [PR #4](https://github.com/Roman223/Agent_modelling/pull/4) |
| 2 | `feat/research-experiment-runner` | Resumable experiments, compact artifacts, provenance and resource limits | Merged via [PR #5](https://github.com/Roman223/Agent_modelling/pull/5) |
| 3 | `feat/research-parameter-atlas` | One-dimensional sweeps and coarse two-dimensional parameter maps | Merged via [PR #6](https://github.com/Roman223/Agent_modelling/pull/6) |
| 4 | `feat/research-hard-reduction` | Hard-rule deterministic reduction and comparisons with controlled ABM runs | Merged via [PR #7](https://github.com/Roman223/Agent_modelling/pull/7) |
| 5 | `feat/research-heterogeneity-study` | Representative portfolios, minimum retained individuality for ABM fidelity, and data-driven structure discovery | Merged via [PR #8](https://github.com/Roman223/Agent_modelling/pull/8) |
| 7 | `feat/research-multifractal-analysis` | MF-DFA, surrogate/GARCH controls, validation and pilot results | Merged via [PR #9](https://github.com/Roman223/Agent_modelling/pull/9) |
| 6 | `feat/research-inactivity-dynamics` | Inactivity onset, decision/resource/book diagnostics, and resource-conserving factorial recovery | Merged via [PR #10](https://github.com/Roman223/Agent_modelling/pull/10); full pilot and final audit validated |

Future PRs are opened when their changes are available; pending rows do not imply
implemented code or completed experiments. Each PR records its predecessor, scope,
validation results, limitations, and handoff to the next task. Update this table
and the integration PR checklist after each merge. Completed task PRs are
integrated after their required checks and implementation verification; they do
not wait for individual user review. The user will validate the combined result
when all seven tasks are finished. Keep the integration PR as a draft until then,
and merge it into `main` only after the required checks and that final validation.

## Compatibility requirements for every task

- Keep the original stochastic ABM as a named reference. Default results and random
  draw ordering must remain identical to notebook commit `84fa2a6`.
- Preserve the legacy tuple API, shared `wealth_history` dictionary identity,
  independent result objects, and cached analysis behavior.
- Preserve existing scenario definitions, order priority, rounding, thresholds,
  bankruptcy rules, and inherited circuit-breaker/evolution behavior. Research
  interventions and deterministic reductions must be explicit variants.
- Apply configured alpha/gamma to initial and evolved traders; evolved chartists
  retain their legacy 20-observation window. Constant dividends and zero closing
  noise retain the original draws. Equal seeds do not guarantee aligned later
  shocks across variants.
- Preserve daily book summaries after matching and before optional day-order
  expiry. Expiry must be independent of logging, with residual counts/quantities
  recorded in `orders_expired` events when logging is enabled.
- Keep the existing schema-1 bundle layout and strict source/runtime/dependency/
  effective-configuration replay checks. Historical bundles require their recorded
  source/runtime. Later compact experiment artifacts are a separate interface.
- Use absolute Python imports, explicit seeds, correctly named estimators, and
  reproducible provenance. Do not claim deterministic chaos from noisy time-series
  estimates or visual complexity alone.

## Research and execution defaults

- Python only; reuse existing numerical tools and assess open-source solutions
  before implementing substantial new numerical methods. No new dependency is
  required by task 1.
- Deliver a toolkit and pilot results. Default experiment batches to a configurable
  120-minute budget; larger confirmatory studies require a later scope decision.
- Default to two workers and one native numerical thread per worker; never exceed
  six workers. Bound experiment memory to 6 GiB and output to 5 GiB, and stop before
  free disk falls below 10 GiB. Benchmark one run before scaling and run long jobs
  in the background with logs. Stream/aggregate large data rather than loading
  multi-million-row tables into pandas.
- For task 3, start from baseline composition/endowments and SMA window 20, with
  evolution and the breaker disabled. Scan alpha at 17 geometric values from
  0.025 to 0.4 with gamma 2, plus a 9-by-7 alpha/gamma grid with gamma from 0.5 to 8.
  Use seeds 100–104 and 2,000 days, analyzing the last 1,000; deduplicate shared runs.
- Preserve task 4's hard-rule reductions and their recorded fidelity failures as
  comparison baselines. Task 5 now varies the retained portfolio detail and
  discovers structure from simulations; it does not derive governing market
  equations or smooth the existing pooled reduction. Its first target is the
  full individual-trader ABM, not historical market-index data.
- Reassess task 6 after task 5 establishes what representation and dynamics are
  supported. If a suitable deterministic map is validated, its analysis requires
  discrete-map multipliers of `DF`, continuation, periods 2/4/8, converged tangent
  Lyapunov estimates, and known-answer tests. A stochastic or predictive model
  does not automatically justify that workflow. Smoothing scales 0.01, 0.003,
  and 0.001 belong to the deferred proposal, not the revised task 5 pilot.
- For task 7, use selected anchors and seeds 200–202, with 10,240-day runs, discard
  2,048 days, and retain 8,192. Compare nested lengths and chronological index
  returns using MF-DFA plus shuffled, IAAFT, and Student-t GARCH controls. Treat
  stalled/zero-variance segments explicitly; do not add jitter or stitch segments.

## Task 1 handoff

Implementation commit: `58ab641`; integration merge commit: `725ffdf`.
The five controls are documented in the [README](README.md#research-parameter-controls).
The [validation record](VALIDATION.md#research-parameter-controls-task-1-2026-10-27)
lists changed files, commands, results, and limitations. All 39 focused checks and
25 default legacy-regression cases passed. Task 1 did not launch a new research
sweep or implement any later task.

## Task 2 handoff

Implementation commit: `1ae3195`; integration merge commit: `02e96f6` via
[PR #5](https://github.com/Roman223/Agent_modelling/pull/5).
The runner, compact artifact format, resource controls, and CLI/Python interfaces
are documented in the [README](README.md#resumable-research-experiments).
The [validation record](VALIDATION.md#research-experiment-runner-task-2-2026-10-27)
records 57 passing focused checks, 25 passing default legacy cases, and the
three-seed baseline pilot with verified interruption/resume. Task 2 does not add
parameter sweeps or implement tasks 3–7. Its branch started at `8fe91d6`.
Create the task 3 branch from the updated integration branch, including this
merge and its handoff documentation.

## Task 3 handoff

Implementation commit: `a480c8d`, based on integration commit `f56003d`.
Validated task head: `90e0d79`; integration merge commit: `da3ec4f` via
[PR #6](https://github.com/Roman223/Agent_modelling/pull/6). The merged tree was
identical to the validated task head before this documentation update.
The approved task 3 plan extends the initial pilot to all eight combinations of
closing noise `{0, 0.05}`, dividends `{constant, stochastic}`, and order lifetime
`{day, persistent}`, at every alpha/gamma coordinate. With seeds 100–104 this is
568 configurations and 2,840 unique 2,000-day runs. The final 1,000 daily returns
are retained for analysis. Repeated bounded batches are authorized until the
full atlas is complete; each batch retains the 120-minute limit and machine caps.
The execution default remains two workers; after the serial benchmark, the user
authorized four workers for this full study. The existing ledger was resumed
with that limit and already completed artifacts were retained.

Market diagnostics are accompanied by Sample Entropy (nats) and fixed-setting
Rosenstein estimates, with separate fit seeds, known-system checks, and explicit
undefined results. These are exploratory stochastic-series diagnostics. They do
not replace task 6's deterministic-map stability and tangent-Lyapunov validation.
The original stochastic ABM remains the named reference. No model rules or
legacy metrics were changed. See the [atlas interface](README.md#parameter-atlas-task-3).

All 2,840 simulations and analyses completed, with zero operational failures and
no missing runs. Both nonlinear estimators produced finite values for 2,557 runs;
283 entirely inactive analysis windows were explicitly marked constant. All
67 focused checks and 25 default legacy cases passed. The complete report has
24 figures in PNG and PDF, per-run records, per-cell coverage/dispersion, and
compressed numeric tables. The [validation and findings](VALIDATION.md#research-parameter-atlas-task-3-2026-10-27)
record execution resources, warnings, selected figures, and limitations.

For task 4, retain the alpha=0.1/gamma=2 reference and its eight controls. Add the
explicitly exploratory alpha=0.2/gamma=0.5, zero-noise/constant-dividend/persistent
cell (5/5 terminal stalls) and its day-order counterpart as a liquidity contrast.
Select the stored coordinate by index rather than retyping rounded floating-point
values. Preserve resource state and accepted-price memory in the reduction, and
test trading continuity as well as prices. These are comparison anchors, not
verified bifurcations or deterministic attractors. Create task 4 from the updated
integration branch, including the PR #6 merge and this handoff documentation.
The task 4 and task 5 handoffs below record the subsequent implementations.
Task 6 remains deferred; the 2026-11-17 revision prioritizes Task 7.

## Task 4 handoff

Simulation execution commit: `a1e3f3d`, based on integration commit `60b765d`.
Validated task head: `dab21ed`; merged on 2026-10-28 through
[PR #7](https://github.com/Roman223/Agent_modelling/pull/7) as `d3f3a60`.
The merged tree was identical to the validated task head before this handoff update.
The implementation supplies a deterministic day-order map and a persistent-order
queue companion. Both retain strategy cash/shares and accepted-price memory.
Conditional quotes use NumPy quadrature; pooled resources, fractional quantities,
and matching the weighted order measure are explicit approximations. The
[README](README.md#hard-rule-deterministic-reduction-task-4) defines the equations,
interfaces, execution workflow, and artifact format. The
[validation record](VALIDATION.md#hard-rule-deterministic-reduction-task-4-2026-10-27)
contains checks, resources, figures, findings, and reproduction details.

All 83 focused checks and 25 default legacy cases passed. The pilot reused 50
historical ABMs, added 50 fresh controls (seeds 110–114), and completed all 120
reductions at 8/16/32 quote points with no operational failures. Reports retain
historical/fresh cohorts separately, exact zero-volume and sub-share activity,
within-strategy volume, resource paths, and quadrature sensitivity. A later
plotting-only axis adjustment is separately fingerprinted; simulation artifacts
retain their original execution identity.

The tested closure fails ABM fidelity: at the fresh liquidity anchor, persistent
ABMs stall in 5/5 cases versus 1/5 primary reductions; day-order ABMs have 27.7%
median no-trade days versus zero in the reduction. Continued reduced volume is
predominantly within-strategy circulation, and prices/resource redistribution
also diverge. Stall classifications change with quote resolution. These failures
are retained as results, not tuned away or treated as verified dynamics.

The original handoff proposed a smoothing study. The approved revision below
supersedes that next step while preserving these results. Further closure
validation is needed before interpreting reduced attractors as ABM explanations.
Keep the persistent companion outside fixed-size continuation; the day map has
two conserved-resource constraints. Task 5 was branched from integration commit
`abc1324`, including the PR #7 merge and this scope revision.

## Revised task 5: retained individuality and structure discovery (2026-10-30)

The user approved studying representative portfolios and discovering structure
from simulation data, with the full original ABM as the first comparison target.
Do not write direct governing market equations. This replaces the original
regularized-companion task. Its implementation and fixed pilot are now complete;
see the Task 5 handoff below. Task 4 passed implementation checks, but its
scientific fidelity and the combined project remain subject to user review.

### Approved implementation protocol

The implementation plan narrows the first pilot to **one-off checkpoint
compression**, preserving all individual traders and their independent trading.
This measures how much initial portfolio information is needed over a specified
horizon; it does not establish a permanently compressed model or a speedup.

Use reference variants 0/6/7 and liquidity variants 6/7; development seeds 300–309
and held-out seeds 310–359; 2,000-day references; checkpoints after 100/500/1,000
completed trading days; and 1,000-day continuations. Report horizons 1/20/100/1,000,
with threshold assessment at 100/1,000. A full-detail continuation must reproduce
the uninterrupted ABM exactly. Collect bounded observations and freeze the
moderate tolerances before held-out analysis: 10 percentage points for stalls,
5 for inactivity/resource fractions, and log(1.2) distribution distances for
volume, volatility, and cumulative returns under the documented transforms.

Group portfolios by nested Ward cuts; conserve cash and integer shares, record
realized profiles after integer balancing, and retain original order ownership.
Fit current/lagged PCA and clustering on development runs only. Keep uncertainty
and inconclusive outcomes explicit. Finish the fixed pilot through bounded,
resumable batches under the existing machine caps. The README specifies the
interfaces, exact transformations, and workflow.

### Research question and controlled comparison

Find how much individual portfolio variation must be retained before a simpler
representation reproduces the full ABM's behavior. A threshold may depend on
the regime or observable, and there may be no useful compression before full
individual detail is required. Do not assume a sharp or monotone transition.

- Explore 1, 2, 4, 8, and 16 representative portfolios per strategy, capped by
  that strategy's population, plus a full individual-detail endpoint. Keep the
  represented population, initial total cash/shares, strategy parameters, and
  price memory fixed. This varies representation detail, not population size
  or the strength of newly invented behavioral differences.
- Traders within a strategy start with identical portfolios in the current ABM;
  differences emerge through trading. Retain that emergence rather than merely
  duplicating identical deterministic pools. Preserve the joint distribution of
  cash and shares; equal wealth does not imply equal ability to buy or sell.
- Keep order lifetime and random forcing as explicitly matched controls, using
  the existing reference and liquidity anchors initially. The named original
  stochastic ABM remains the reference, with zero-noise/constant-dividend controls
  separately labeled. Preserve the accepted-price clock: no-trade days freeze
  its memory, so 20 price observations need not mean 20 calendar days.
- Define and validate grouping, portfolio updates, resource conservation, and
  order ownership before running the hierarchy. The full-detail endpoint must
  reproduce the individual-trader ABM, including its seeded behavior. Increasing
  the number of deterministic weighted pools alone does not guarantee that
  endpoint. Task 4 remains a separate baseline, not an assumed equivalent of
  the new one-profile setting.

### Measures, threshold assessment, and discovery

- Report the number of profiles alongside retained within-strategy portfolio
  dispersion and normalized joint cash/share distribution error. Specify the
  normalization and zero-dispersion handling explicitly; a profile count alone
  is not a universal measure of individuality.
- Compare ensemble distributions of trading stalls, no-trade fractions, volume,
  returns, and strategy resource allocation. Retain exact-zero and sub-share
  activity separately. Use full-ABM versus full-ABM ensemble differences to
  characterize sampling variability, not to claim equivalence from a failed
  significance test or visually similar price traces.
- Freeze quantitative similarity tolerances and primary outcomes before held-out
  validation. Identify the smallest tested representation that meets those
  tolerances on separate runs and remains acceptable at finer resolutions.
  Report uncertainty, per-regime differences, and unresolved cases; do not
  force a single threshold when different outcomes require different detail.
- Discover compact structure using portfolio distributions, resource availability,
  trading activity, order-book summaries, and price history. Start with clustering
  and dimensionality reduction; compare representations with and without lagged
  observations. Fit scaling and representations on development runs only and
  evaluate on independent runs. Predictive structure does not by itself establish
  a closed deterministic model, a causal explanation, or chaos.
- Existing compact atlas artifacts omit individual portfolio snapshots. Collect
  bounded, reproducible observations where needed rather than infer individual
  variation from group totals. Keep simulation and analysis seeds separate, and
  retain source/configuration provenance and the existing resource limits.

### Reuse, deliverables, and next-task decision

Reuse the existing simulation/artifact infrastructure, NumPy, and
[scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)
and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
Assess [SciPy distribution distances](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance_nd.html)
for small portfolio distributions. [HARK](https://docs.econ-ark.org/docs/overview/ARKitecture.html)
provides useful distribution-representation precedents, but replacing the market
engine with that framework is not planned. Adapt established tools; implement
only the model-specific grouping and validation needed here.

Deliver a bounded, resumable pilot; plots of retained individuality versus ABM
disagreement; threshold estimates or explicit non-convergence; and validated
structure-discovery results. Preserve the hard-reduction failures in comparisons.
No historical-index realism claim or hand-derived market equations are in scope.

Use the findings to revise task 6's specification before implementing it. The
previous smooth-map continuation route is not automatic; smoothing remains
deferred unless the evidence supports a suitable model. Task 7's multifractal
scope is unchanged and does not require a successful smooth reduction. Keep the
integration PR draft and retain final user validation before merging into `main`.

## Task 5 handoff

Validated task head `fcd40a6` merged through
[PR #8](https://github.com/Roman223/Agent_modelling/pull/8) as `287bd13` on
2026-10-30. The merged tree matched the validated head before this handoff update.

Task 5 implements exact market checkpoints, conservative one-off portfolio
compression, retained-detail measures, a bounded resumable study, and current/
lagged structure discovery. Simulation execution is reproducible at `6db78d5`;
later plotting changes are separately fingerprinted. All 96 focused checks and
25 default legacy cases passed. The fixed pilot completed 300 references and
5,400 continuations across 900 checkpoint jobs. All full-detail checks were exact.
Execution switched from two to four workers with user approval and stayed within
the existing resource limits. Detailed evidence is in
[VALIDATION.md](VALIDATION.md#retained-individuality-and-structure-discovery-task-5-2026-10-30).

The pilot supports 11 candidate thresholds across 30 held-out conditions, not a
universal individuality cutoff. Ten candidates concern a 100-day horizon. The
remaining candidate uses one portfolio group per strategy in the original-forcing
ABM after checkpoint 1,000, evaluated for the following 1,000 days. Most other
original-forcing comparisons are inconclusive; the controlled configurations
show substantial longer-horizon disagreement. Retaining approximately 99.9% of
portfolio dispersion at 16 groups does not guarantee fidelity in every case.
All traders and individual trading remain, so this is not a permanent reduced
simulator or a demonstrated speedup.

Current and lagged representations each find five stable market-state clusters
and improve simple future-outcome predictions over a global-mean baseline.
Additional lags have no consistent advantage across the three targets. These
results support further state/prediction validation, but no autonomous transition
map has been validated. Keep Task 6's deterministic continuation route deferred.
Its next specification should assess stronger forecasting baselines and
within-configuration generalization before claiming a closed reduced model.
Specify that follow-up before implementation. Task 7 is unchanged. Keep
integration PR #3 draft until the combined work receives final user validation.

## Task 7 protocol: multifractal scaling (2026-11-17)

Research question: do simulated and real stock-index returns exhibit comparable
multifractal scaling, and how much survives stochastic and finite-length controls?
The user selected a complete pilot, inclusion of existing empirical indices, and
fixed-regime comparison without ABM parameter calibration. Fractal regime-boundary
geometry remains part of the broader agenda and is deferred independently.

Use the exact five Task 5 atlas configurations and seeds 200–202: 15 full ABMs,
10,240 trading days, discard 2,048, retain 8,192. The primary comparison uses the
last 2,048 returns of each simulation and each of the five index files. End-aligned
lengths are 1,024/2,048/4,096/8,192 for ABMs and 1,024/2,048 for indices: 70 windows.
Snapshot inputs, sort empirical observations chronologically, record gaps without
filling them, and retain ABM no-trade days. Preserve the legacy loader and model.

Freeze q=-3,-2.5,...,3, including the logarithmic q=0 limit; linear detrending is
primary and quadratic detrending is a sensitivity check. Use quarter-octave scales
from 16 through N/8 and OLS log-log fits; also retain the common 16–128 scale fit.
The primary statistic is h(-3)-h(3); retain h(q), tau(q), the Legendre spectrum,
finite-q spectrum width, residuals, segment counts, and R-squared. Flag R-squared
below 0.95. Zero/unresolved segments invalidate affected nonpositive moments;
never add jitter, remove inactivity, or stitch active segments. Full spectrum width
requires a complete q grid. No adaptive selection of favorable scaling ranges.

Generate 99 shuffled, IAAFT, and fitted constant-mean Student-t GARCH(1,1) controls
per window. Use separate hash-derived generators with base analysis seed 20261117.
IAAFT preserves amplitudes exactly, allows at most 1,000 iterations, and records
rank stability and relative Fourier-amplitude error; errors above 1% invalidate
that control comparison. Require converged, stationary GARCH fits with finite
admissible parameters and degrees of freedom above two; require persistence below
1 minus 1e-8 so a boundary solution cannot pass through floating-point rounding.
Discard 2,048 simulated
GARCH burn-in steps. Keep unavailable fits explicit. Reuse control realizations
across detrending settings. Retain finite weak-scaling null estimates, and report
incomplete control ensembles as inconclusive rather than replacing realizations.

Validate against MFDFA 0.4.3, Gaussian/fractional Gaussian noise, and a lognormal
cascade before market analysis. The accepted implementation plan specifies the
numerical tolerances, recovery tests, fixed default legacy checks, and outputs.
The immutable study manifest stores the full machine-readable protocol and hashes.
Run through bounded background batches until the fixed pilot is complete; default
to two workers with one native thread each and the existing 6 GiB memory/5 GiB
output/10 GiB free-disk limits. Time a full simulation and a longest-window control
chunk before parallel execution. Save chunks of at most 25 control realizations.

Deliver plots, compressed numeric tables, a findings report, validation evidence,
and the Task 7 integration handoff. Report all three seeds separately. Nested
windows are dependent, and null reference intervals are not population confidence
intervals. Multifractal scaling alone does not establish deterministic chaos,
causal mechanisms, or empirical equivalence. A negative or inconclusive scientific
result is valid; unresolved operational failures are not completion.

## Task 7 handoff

Validated task head `2424f90` merged through
[PR #9](https://github.com/Roman223/Agent_modelling/pull/9) as `bdb0122` on
2026-11-17. The merged tree matched the validated task head before this handoff
update. Integration PR #3 remains draft.

The full fixed pilot is complete: 15 simulations, five chronological index inputs,
70 analysis windows, and 980 completed analysis jobs with no operational failures.
All 114 focused checks and 25 default legacy cases passed. Simulation source is
`78147c0`; final analysis source is `933b84f`, with final rendering separately
fingerprinted. A numerical stationarity-boundary guard was added after detecting
a rounding edge case; verified historical simulation reuse preserved every ABM
trajectory, and all 70 observed analysis payloads are unchanged. The final study
is `results/task7_multifractal_final_20261117/`. Full evidence and representative
figures are in [VALIDATION.md](VALIDATION.md#multifractal-scaling-task-7-2026-11-17).

The primary windows of all 12 controlled ABMs are inactive. The three original-
forcing ABMs have finite but weak and length-sensitive scaling fits. NASDAQ alone
exceeds the shuffled and IAAFT reference bounds for primary delta-h; none of the
five indices exceeds its GARCH bound. These exploratory comparisons establish
neither robust ABM–index multifractal equivalence nor deterministic chaos.

Task 6's deterministic route remains deferred. This pilot does not validate an
autonomous map, and another scaling study would require a prespecified treatment
of transient windows, trading continuity, and independent seeds. Do not launch
that follow-up or revive deterministic continuation automatically. Keep integration
PR #3 draft for the final combined user validation; do not merge it into `main`.

## Approved Task 6: inactivity and controlled recovery (2026-11-24)

This specification supersedes the deferred bifurcation task. Task 4 failed ABM
closure fidelity; Task 5 did not establish an autonomous transition map; Task 7
found terminally inactive controlled ABMs. Study onset and mechanisms of inactivity
and test recovery in the original individual-trader ABM. Do not perform deterministic
continuation, another MF-DFA study, or an empirical-index comparison here.

The fixed design has 800 references: seeds 600–649 at both stored atlas coordinates
`grid-4-3` (reference) and `grid-6-0` (liquidity), each with all eight combinations of
closing noise `{0, 0.05}`, dividends `{constant, stochastic}`, and orders
`{persistent, day}`. Use the stored floating-point coordinates, baseline populations
and endowments, window 20, no evolution, and no breaker. Run 10,240 trading days;
exclude only the exogenous warm-up. Never pool these fresh references with earlier
studies or treat failed/incomplete jobs as scientific censoring.

A stall is confirmed after 100 consecutive exact zero executed-volume days. Retain
all spells, their onset and confirmation, recovery, and terminal right censoring.
Thresholds 20 and 250 are descriptive sensitivity analyses, not extra intervention
cohorts. Report finite-horizon incidence and time to first confirmation, fixed
100/500-day windows, and onset-aligned diagnostics with contributing counts. Capture
actual decision reasons, submissions, self versus distinct-trader trades, strategy
resources, accepted-price memory, and post-matching/pre-expiry book conditions.

Capture one immutable checkpoint per eligible reference after its first confirmation
and after day-order expiry. Apply all eight combinations of three binary interventions:

1. Clear both outstanding queues once, retaining the subsequent lifetime policy.
2. Reset the last 20 accepted prices to the current price, retaining history length
   and dividends. This shared memory affects chartists and fundamentalists.
3. Equalize active traders' target invested fractions at the current price, preserving
   each trader's marked-to-market wealth and global cash/integer shares. Allocate
   target shares proportional to wealth, floor, then assign largest fractional
   remainders subject to affordability caps, breaking ties by stable trader ID.
   Leave inactive traders unchanged and record transfers and conservation residuals.

Each arm receives five reproducible future RNG seeds and 1,000 continuation days:
at most 32,000 logical continuations before identical-state deduplication. Retain all
eight labels after deduplication. Corresponding arms share the initial future RNG
state; branch-dependent draw consumption prevents a claim of identical later shocks.
Verify an additional untouched original-RNG continuation against the uninterrupted
reference's daily observations and final state/RNG. Extend a late reference only
enough for this validation tail; exclude that tail from onset statistics.

The primary outcome is the distinct-trader exchange-day fraction on continuation
days 901–1,000. Secondary outcomes include first execution/exchange, volume, and
recurrent stalls. Average futures within checkpoints; use reference seeds as the
inference unit for factorial main effects/interactions, with 2,000 paired bootstrap
resamples and pointwise 95% intervals. Flag small samples and degenerate empirical
intervals. Recovery is conditional on selected stalled states; finite nonrecovery
does not prove permanent absorption. No-stall configurations have N/A recovery.

Reuse checkpoint/provenance/worker infrastructure and NumPy/SciPy survival, binomial,
and bootstrap functions; add no dependency. Freeze settings and source/runtime
identity before the pilot. Benchmark a reference and recovery job before scaling;
use two workers, one native thread each, 6 GiB total memory, 5 GiB output, a 10 GiB
free-disk floor, and logged background batches no longer than 120 minutes each.
Continue bounded resumable batches until the fixed design completes, stopping on
resource limits or failures. Validate exact replay, conservation, censoring, and
known survival answers; interruption/resume, corruption and provenance guards;
all existing focused suites and the 25 default legacy cases. Keep PR #3 draft for
the final combined user validation.

### Task 6 implementation handoff

Initial source `40848ff` implements the approved protocol. Optimized source `e07f286`
passed 132 focused checks plus all 25 default notebook comparisons. See the
[interface](README.md#inactivity-and-recovery-task-6) and
[validation record](VALIDATION.md#inactivity-and-recovery-task-6-2026-11-24).
[PR #10](https://github.com/Roman223/Agent_modelling/pull/10) merged into the
integration branch on 2026-11-26. The initial pilot was stopped to remove repeated book scans;
full-length comparisons retained identical observations and final market/RNG state.
Its artifacts remain separate. The final pilot uses
`results/task6_inactivity_final_20261124`, with output in
`results/task6_inactivity_final_20261124.log`. It started at 13:37 UTC on 2026-11-24.
Its driver completed all eligible recovery jobs, analysis, and rendering after
the references finished. The full scientific pilot and final artifact/figure audit
passed on 2026-11-26. Historical pauses and restarts are recorded below.

Paused at the user's request on 2026-11-24 at 14:46 UTC. The coordinator and workers
exited with 428 references complete, two interrupted and restartable, and 370
pending. Recovery experiments had not started.

The user requested resumption on 2026-11-25. Source/runtime identity and all 428
completed artifacts were verified, and the same study resumed at 13:09 UTC with
two workers and the original limits. All 800 references completed at 14:08 UTC,
and 522 eligible recovery checkpoint jobs were scheduled.

Paused again at the user's explicit request on 2026-11-25 at 15:53 UTC. The
coordinator and workers exited cleanly. All 800 references and 121 recovery batches
are complete; two recovery batches are interrupted and 399 are pending. No jobs
failed. The pause required an explicit user request before resumption. Preserve
the frozen study and execution source.

Resume with the same `study` command from the README, using
`results/task6_inactivity_final_20261124` and two workers. Completed references and
recovery batches are verified and reused; only the interrupted batches restart.
Analysis, rendering, the final audit, and integration remain pending.

The user requested resumption on 2026-11-26. Source/runtime identity and completed
artifacts were verified. Recovery resumed at 10:52 UTC with 401 checkpoint jobs
remaining, two worker slots, and unchanged limits. The first job runs alone as
the scheduler's benchmark before both slots are used. Logs are appended to the
same file. A preceding restart attempt exited before launching recovery workers
with a transient SQLite lock, apparently coinciding with a status query. Monitor
the append-only log during execution; the ledger writer has a zero-second busy
timeout. No simulation source or frozen study settings were changed.

The resumed run stopped automatically at 11:08 UTC on 2026-11-26 with
`free_disk_limit`: 138/522 recovery jobs complete, two interrupted, and 382 pending.
The coordinator exited with code 2. At 12:01 UTC the filesystem had only 2.6 GiB
free; the study itself still occupied about 2.0 GiB. Restore sufficient free space
for the unchanged 10 GiB floor and active-worker reservations before restarting.
Completed artifacts remain preserved. About 6–7 hours of running time remain at
the observed pace, excluding any time needed to resolve the disk shortage.

The user requested continuation later on 2026-11-26. At 12:13 UTC, available disk
space was 60 GiB and available memory about 9.2 GiB. Source/runtime identity passed
again, and the same study was restarted with two workers and all original limits.
It resumes from 138 completed recovery jobs, with 384 remaining. The existing log
is retained and appended; analysis, rendering, and the final audit remain pending.

### Task 6 completed pilot and integration handoff (2026-11-26)

All 800 references and 522 recovery checkpoint jobs are complete. Recovery finished
at 17:39 UTC; analysis and all eight PNG/PDF figure families finished by 17:48 UTC.
The coordinator exited successfully. Every checkpoint passed exact original-RNG
replay. State deduplication retained all 20,880 logical arm/future labels while
executing 14,900 distinct futures. There are no failed or incomplete jobs.

The final audit verified all 1,322 input artifacts, 36 execution-source hashes,
six analysis payloads, and sixteen figure payloads. It independently reconciled
reference stall thresholds and primary recovery outcomes with daily arrays and
checked arm means and factorial estimates. All eight figure families were visually
inspected. See [the complete validation and findings](VALIDATION.md#completed-task-6-pilot-2026-11-26).

Every zero-closing-noise reference stalled; 122/400 positive-noise references did.
Recovery depends strongly on portfolios, accepted-price memory, and the subsequent
forcing/order-lifetime policy. Queue clearing alone does not improve the primary
outcome in this pilot. Conditional finite-horizon recovery does not establish a
deterministic transition map or permanent absorption. No further simulations are
required for the approved Task 6 pilot.

Validated task head `1c836c4` merged through PR #10 as `1ff3c71` at 18:19 UTC on
2026-11-26. The merged tree was verified identical to the validated head before
this documentation handoff. All seven task PRs are now integrated. Integration
PR #3 remains draft and `main` is unchanged, pending the user's combined final
validation. Broader confirmatory studies, additional model changes, and
deterministic bifurcation analysis remain outside this completed pilot's scope.
