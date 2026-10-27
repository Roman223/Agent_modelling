# Emergent dynamics research: PR integration plan

Integration branch: `research/emergent-dynamics`.
Integration PR: [#3](https://github.com/Roman223/Agent_modelling/pull/3).
Final target: `main`, through a draft integration PR that remains open until the
selected research plan is complete. The scope is research phases 1, 2, 3, and 5:
parameter discovery, deterministic reductions, bifurcation verification, and
multifractal analysis. Basin/boundary geometry (research phase 4) is deferred.

## Sequential PRs

Execution task numbers below are distinct from the research phase numbers.
Each task has its own branch and PR targeting `research/emergent-dynamics`.
Merge completed tasks in the order below. Create each subsequent task branch from
the updated integration branch after its predecessor has merged; this keeps its
PR focused on that task's changes. Use merge commits to retain the task history.

| Task | Branch | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | `feat/research-parameter-controls` | Python/CLI controls, compatibility checks, logging and bundle replay | Merged via [PR #4](https://github.com/Roman223/Agent_modelling/pull/4) |
| 2 | `feat/research-experiment-runner` | Resumable experiments, compact artifacts, provenance and resource limits | Implemented and validated; awaiting review/merge |
| 3 | `feat/research-parameter-atlas` | One-dimensional sweeps and coarse two-dimensional parameter maps | Pending |
| 4 | `feat/research-hard-reduction` | Hard-rule deterministic reduction and comparisons with controlled ABM runs | Pending |
| 5 | `feat/research-smooth-reduction` | Regularized companion and sensitivity to smoothing | Pending |
| 6 | `feat/research-bifurcation-analysis` | Continuation, stability/Lyapunov checks, known-answer tests and pilot results | Pending |
| 7 | `feat/research-multifractal-analysis` | MF-DFA, surrogate/GARCH controls, validation and pilot results | Pending |

Future PRs are opened when their changes are available; pending rows do not imply
implemented code or completed experiments. Each PR records its predecessor, scope,
validation results, limitations, and handoff to the next task. Update this table
and the integration PR checklist after each merge. The final integration PR is
merged into `main` only after the remaining tasks and their required checks pass.

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
- For tasks 4–5, retain strategy cash/shares and the accepted-price memory in an
  ABM-derived aggregate reduction. Compare hard rules with a smooth companion;
  assess smoothing scales 0.01, 0.003, and 0.001. Validate the reduction against
  explicit ABM controls before interpreting its attractors.
- For task 6, use discrete-map stability multipliers of `DF`, continuation, periods
  2/4/8, and converged tangent Lyapunov estimates. Validate against known systems;
  distinguish verified bifurcations, candidates, and nonsmooth transitions.
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

The runner, compact artifact format, resource controls, and CLI/Python interfaces
are documented in the [README](README.md#resumable-research-experiments).
The [validation record](VALIDATION.md#research-experiment-runner-task-2-2026-10-27)
records 57 passing focused checks, 25 passing default legacy cases, and the
three-seed baseline pilot with verified interruption/resume. Task 2 does not add
parameter sweeps or implement tasks 3–7. Its branch starts at `8fe91d6`; merge it
before creating the task 3 branch from the updated integration branch.
