# ABM Market Model — Revision & Extension Project

## Current mission (2026-10-21)

Make the repository clear and presentable: finish the behavior-preserving refactor,
improve code readability, and provide an English README and user-facing output.
Preserve numerical behavior and random-draw ordering, and verify changes against
the original notebook. Tracks A and B below are the research backlog, not the
current implementation scope. Do not start model corrections, ensemble studies,
or new estimators as part of this cleanup.

Cleanup checkpoint: the README, scenario titles, reports, and plot labels are in
English. Shared return preparation and trader initialization have been extracted;
Sample Entropy has a correctly named public helper with the old name retained as
an alias. scikit-learn is declared as a dependency, and the regression CLI rejects
zero seeds. The default legacy-equivalence suite passed all 25 cases (250 days,
seeds 0–2 across eight scenarios, plus circuit-breaker seed 18). An evolution CLI
smoke run also saved all four plot types. Global RNG and wealth-history behavior
remain preserved for compatibility.

## Context

Working on the codebase behind Belozerov & Kovantsev, "A Multi-Agent Financial Market Model for
Understanding Chaotic Dynamics" (rejected from Chaos Theory and Applications; revision + follow-up
paper planned). Source: Jupyter notebook `Мультиагентное_моделирование_1_1.ipynb` — a stylized
stock market ABM with three trader classes (noise / chartist / fundamentalist) trading one asset
through a simplified order book. Scenarios: baseline, 10x each class, circuit breaker, evolutionary
type-switching; nonlinear metrics (Hurst, Lyapunov, entropy) computed on simulated and real index
data (RTS, MOEX, S&P500, DJIA, NASDAQ csv files).

Two work tracks:

- **Track A** — fix the notebook/paper for resubmission (reviewer demanded: ensembles, reproducible
  methods, consistent numbers).
- **Track B** — new experiments toward a second paper (does a low-dimensional deterministic
  skeleton emerge from the ecology, or are the stylized facts finite-size fluctuations?).

## Ground truth from code audit (paper text is WRONG where they differ)

| Topic                      | Code (authoritative)                                                             | Paper text (to fix)                          |
| -------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------- |
| Fundamentalists (baseline) | `fundamentalists=30`                                                             | Table 1 says 300 — typo                      |
| Metrics input              | log-returns: `np.diff(np.log(price))`                                            | Eqs. 11–16 describe prices/levels            |
| "KS entropy"               | `nolds.sampen`, m=3, tol=0.2·std → **Sample Entropy**, nats                      | Eqs. 21–23 claim h_KS; "бит/шаг" wrong units |
| Lyapunov                   | `nolds.lyap_r` (Rosenstein)                                                      | Paper claims Wolf method                     |
| Lyapunov emb_dim           | **5 for simulations, 10 for real indices** — comparability bug                   | not disclosed                                |
| Wealth in Table 2          | per-agent mean over *active* traders (`np.mean`)                                 | called "group wealth"; survivorship bias     |
| Price initialization       | 20-day exogenous GBM warm-up (drift 3e-4, vol 0.02)                              | claims purely endogenous prices              |
| Chartist rule              | buy markup `0.02*(1+10*vol)*rand` (volatility-amplified)                         | plain SMA rule, Eq. 3                        |
| Fundamentalist rule        | trigger `abs(r_excess)>0.05`, `x_t=clip(r_ex/(γ*0.1), 0.1, 0.9)`                 | mean-variance Eq. 6 with γσ²                 |
| Bankruptcy                 | cash≤0 and assets≤0 → permanently inactive                                       | not mentioned                                |
| Risk-free rate             | 0.0003 passed in call                                                            | Table 1 says 0.0004                          |
| Seeds                      | none set; single run per scenario; global `wealth_history` dict mutated by cells | —                                            |

Retracted earlier critique: Table 3 does NOT violate the Ruelle inequality — the entropy column is
SampEn (stochastic-process metric), not h_KS, so no inequality connects it to λ. The problem is
labeling only. Do not "fix" the SampEn computation itself; it is fine.

## Track A — refactor & fix (priority order)

1. **Refactor notebook → package.** `market/` with `asset.py`, `orderbook.py`, `traders.py`, `simulate.py`, `metrics.py`; scenarios as configs, not copy-pasted 8KB cells (cells 18/24/30/36/43/47
   are near-duplicates). Kill the global `wealth_history`; return results objects. Seed everything
   (`numpy.random.default_rng(seed)` threaded through).
2. **Ensemble harness.** ≥100 seeds per scenario; outputs: median + IQR bands for price paths,
   final price, per-class wealth. Report group TOTAL wealth and per-agent mean separately; track
   bankruptcies explicitly (survivorship).
3. **Metrics module fixes.** Keep SampEn but rename everywhere (`sample_entropy`); unify `emb_dim` between simulated and real data (pick 5, or better report both); unify series length
   (subsample indices to 1000 pts to match sims, or state the difference); nats not bits.
   Add permutation entropy (ordinal patterns, m=4–5) as a second, rank-based metric.
4. **Surrogate controls.** Shuffled + AAFT (IAAFT) surrogates for every reported nonlinear metric;
   report z-scores vs surrogate ensembles. GARCH(1,1) fitted to each index as stochastic null.
5. **Docs-vs-code reconciliation file.** Autogenerate a parameter table from the config so paper
   Table 1 can never drift from code again.

## Track B — experiments (after A1–A2 infrastructure exists)

1. **N-scaling (first, decides the follow-up paper's direction).** Scale all class counts ×1, ×10,
   ×100 at fixed composition and per-capita endowments; ensemble each. Measure vs N: return
   volatility, kurtosis, ACF of |r|, SampEn, permutation entropy. Two hypotheses:
   convergence to structured aperiodic dynamics (Brock–Hommes-like chaotic skeleton) vs
   ~1/sqrt(N) decay of stylized facts (Lux–Marchesi-like finite-size effects).
2. **Multifractal module.** MF-DFA on log-returns, q ∈ [−3,3]; ζ(q), H(q), f(α) width Δα with
   shuffled/AAFT corrections. Validate the implementation on a log-normal cascade (closed-form ζ(q))
   before applying. Compare: scenarios vs indices vs GARCH null. Prediction to test:
   chartist-heavy composition → widest Δα.
3. **Mean-field skeleton (analytical, code assists).** Replace 20-day SMA with EMA; derive/simulate
   deterministic per-class expected-demand map in (P, EMA, wealth shares); bifurcation scan over
   chartist fraction, α, γ. Numeric Lyapunov/bifurcation diagrams for the reduced map (h_KS and λ
   are legitimate THERE).
4. **(Later) EDMD/Koopman on ensemble data** — spectral gap, slow eigenfunctions as emergent
   coordinates.

## Conventions

- All analysis on log-returns unless explicitly stated.
- Every stochastic result: seed-controlled, ensemble-reported (median, IQR), n_seeds stated.
- Every nonlinear metric: named by its actual estimator (SampEn, PermEn, Rosenstein λ), parameters
  (m, tolerance, lag, length) logged next to the value.
- Known-answer validation before first use of any estimator (logistic map for entropy/λ,
  log-normal cascade for MF-DFA).
- Russian comments in the original notebook are fine to keep; new code and docs in English.
