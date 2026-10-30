# Emergent dynamics research: PR integration plan

Integration branch: `research/emergent-dynamics`.
Integration PR: [#3](https://github.com/Roman223/Agent_modelling/pull/3).
Final target: `main`, through a draft integration PR that remains open until the
selected research plan is complete. The scope covers parameter discovery,
ABM reduction fidelity, data-driven structure discovery, and multifractal
analysis. Deterministic bifurcation verification is conditional on finding a
suitable, validated model. Basin/boundary geometry (research phase 4) is deferred.
The approved 2026-10-30 revision replaces task 5's smoothing study with a study
of retained individual differences; see the revised scope below.

## Sequential PRs

Execution task numbers below are distinct from the research phase numbers.
Each task has its own branch and PR targeting `research/emergent-dynamics`.
Merge completed tasks in the order below. Create each subsequent task branch from
the updated integration branch after its predecessor has merged; this keeps its
PR focused on that task's changes. Use merge commits to retain the task history.

| Task | Branch | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | `feat/research-parameter-controls` | Python/CLI controls, compatibility checks, logging and bundle replay | Merged via [PR #4](https://github.com/Roman223/Agent_modelling/pull/4) |
| 2 | `feat/research-experiment-runner` | Resumable experiments, compact artifacts, provenance and resource limits | Merged via [PR #5](https://github.com/Roman223/Agent_modelling/pull/5) |
| 3 | `feat/research-parameter-atlas` | One-dimensional sweeps and coarse two-dimensional parameter maps | Merged via [PR #6](https://github.com/Roman223/Agent_modelling/pull/6) |
| 4 | `feat/research-hard-reduction` | Hard-rule deterministic reduction and comparisons with controlled ABM runs | Merged via [PR #7](https://github.com/Roman223/Agent_modelling/pull/7) |
| 5 | `feat/research-heterogeneity-study` | Representative portfolios, minimum retained individuality for ABM fidelity, and data-driven structure discovery | Implemented; fixed pilot in progress |
| 6 | `feat/research-bifurcation-analysis` | Reassess the appropriate dynamics analysis after task 5; deterministic continuation/stability only if justified | Deferred pending task 5 findings |
| 7 | `feat/research-multifractal-analysis` | MF-DFA, surrogate/GARCH controls, validation and pilot results | Pending |

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
The task 4 handoff below records the next implementation. Tasks 5–7 remain unimplemented.

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
two conserved-resource constraints. Create the revised task 5 branch from the
updated integration branch, including the PR #7 merge and this scope revision.

## Revised task 5: retained individuality and structure discovery (2026-10-30)

The user approved studying representative portfolios and discovering structure
from simulation data, with the full original ABM as the first comparison target.
Do not write direct governing market equations. This replaces the original
regularized-companion task; it is a research scope revision, not an implemented
model or a completed experiment. Task 4 passed implementation checks, but its
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
