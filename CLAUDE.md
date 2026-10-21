# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Current scope (2026-10-21):** repository cleanup, behavior-preserving refactoring,
clean code, and English documentation/output. The research tracks below are
deferred; see the current-mission section in `CONTEXT.md`.

Code behind Belozerov & Kovantsev, "A Multi-Agent Financial Market Model for Understanding Chaotic
Dynamics" — rejected, now being revised (Track A) and extended toward a second paper (Track B).
Agent-based model of a single-asset market: noise, chartist and fundamentalist traders trade through a
simplified order book; nonlinear metrics on the simulated log-returns are compared against real indices
in `data/` (RTS, MOEX/ММВБ, S&P 500, DJIA, NASDAQ, 2016-05-09 → 2026-05-09).

**The research plan, the code-vs-paper audit, and the analysis conventions live in `CONTEXT.md`, imported
below — read it before changing model behaviour or metrics.** Key consequences:

- **Code is authoritative over the paper text** wherever they disagree.
- The historical "Kolmogorov–Sinai entropy" is really **Sample Entropy**
  (`nolds.sampen`) in **nats**, not h_KS in bits. The Lyapunov exponent is **Rosenstein** (`nolds.lyap_r`).
  Name new code and outputs by the actual estimator; don't change the SampEn computation itself.
- New code and docs in English; the existing Russian comments/log strings can stay.

@CONTEXT.md

## Running

```bash
.venv/bin/pip install -r requirements.txt   # .venv (gitignored) already has nolds; the base conda env does NOT
.venv/bin/python experiments/run_scenario.py --list
.venv/bin/python experiments/run_scenario.py baseline --days 1000 --seed 0 --save results --no-show
.venv/bin/python tests/test_legacy_equivalence.py          # ~5 min; --full for 1000 days (~30+ min, run in background)
```

`experiments/run_scenario.py` runs any entry of `src/scenarios.py::SCENARIOS` (`baseline`, `more_noise`,
`more_chartists`, `more_fundamentalists`, `circuit_breaker`, `evolution`, `window_5`, `window_50`). Runs are
unseeded unless `--seed` is given. `more_noise` is slow (~200 s per 1000 days; the order book grows unbounded).

`Multiagent_Modelling_notebook.ipynb` imports the package: one run cell + plot cells per scenario, each
re-seeded with `np.random.seed(SEED)`. Existing notebook outputs may predate the cleanup; use a kernel
with the project dependencies installed when rerunning it.
The **original** notebook, where every scenario cell redefined the classes inline and which produced the
published numbers, is at git commit `84fa2a6` (CONTEXT.md calls it `Мультиагентное_моделирование_1_1.ipynb`).

## Current state of the package

`src/` is a behaviour-preserving extraction of the notebook — not yet the planned `market/` package
(Track A1). It still uses the global `np.random` stream, returns `(asset, traders)`, and writes the
module-level `simulation.wealth_history` dict (plus `asset.halt_events`, `asset.evolution_log`,
`asset.trader_counts_history`).

**`tests/test_legacy_equivalence.py` is the safety net:** it execs the original notebook cells from git
`84fa2a6` and asserts bit-identical prices, dividends, trader balances, `wealth_history` and metric values
per scenario under the same seed. Any change that alters the RNG call order or the model (including Track A1's
`default_rng` threading and the bug fixes) will intentionally break it — at that point retire or re-baseline it.

## Architecture

Per-day loop (`src/simulation.py::run_simulation`, configured by a `Scenario`):

1. 20-day GBM warm-up on the asset before trading starts.
2. `asset.update_dividend()` — multiplicative random walk with rare shocks.
3. Optional regime hooks: evolution (every 100 days, deactivate 2 noise → spawn chartists, 1 chartist →
   fundamentalist, copying cash/assets/`wealth_history`) and circuit-breaker halt check.
4. Each active trader's `make_decision(asset)` returns `(order_type, price, qty)` or `(None, None, None)`;
   orders go into the shared `OrderBook`.
5. `OrderBook.match_orders(traders, asset)` crosses best bid/ask at the midpoint, clips quantities to the
   counterparty's cash/holdings, mutates trader balances directly, then sets the new price as
   VWAP × (1 + 5% noise scaled by a liquidity factor) via `asset.update_price`.
6. `trader.update_wealth(price)`; every 100 days, per-type averages over *active* traders are appended to
   a module-level `wealth_history` dict that scripts read for plots.

Non-obvious behaviour (beyond the audit table in CONTEXT.md):

- **Traders are identified by `id(trader)`** in the order book and in `initial_wealths`. Evolved traders
  are new objects, so they're absent from `initial_wealths` and skipped in profitability stats.
- **The order book is never cleared** between days; unmatched orders persist and accumulate.
- **Circuit breaker:** with `Asset(circuit_breaker=True)` a move >20% vs. `last_check_price` is rejected and
  sets `trading_halted`. `last_check_price` is only refreshed by `check_halt_status()`, which the loop calls
  only when `scenario.halt_check`. **`SCENARIOS['evolution']` uses the breaker Asset without halt checks**
  (in the notebook, cell 42's `Asset` redefinition leaked into the evolution cell), so the first >20% drift from
  100 is dropped and the flag then stays set. In `circuit_breaker` runs, halts in practice only come from the
  20-day warm-up drifting >20% (e.g. seed 18); daily moves essentially never exceed 20%.
- `Trader.wealth_history[0]` assumes a price of 100 regardless of the actual asset price.
- Metrics (`src/metrics.py`) silently return `NaN` on short series or any `nolds` exception.
- **Lyapunov `emb_dim`:** CONTEXT.md says simulations use 5 and indices use 10. In the code, 5 is only the
  function default; every simulation call site (original notebook cells 18–47, now `src/simulation.py`)
  passes `emb_dim=10`, as does the index cell. The saved notebook has no simulation metric outputs
  to confirm which setting produced the paper's Table 3 — verify before treating it as a bug.
- **`nolds.lyap_r` and `hurst_rs` fit by RANSAC only if scikit-learn is importable**; otherwise they warn and
  fall back to `poly`. scikit-learn is included in `requirements.txt`. The historical README's index λ values
  (0.003–0.013) are RANSAC values; `poly` gives ≈0.041 for every index *and* every simulated scenario.
- **RANSAC is stochastic:** `RANSACRegressor` has no `random_state`, so it draws from the global `np.random`
  stream. Index λ varies across RNG states by up to ~3× (e.g. RTS 0.0014–0.0034); seed before computing
  metrics and report the spread. The historical README's simulated λ (baseline 0.086, evolution 0.12) are not
  reproduced by either fit over seeds 0–9 (RANSAC median ≈0.006, poly ≈0.041).

## Data

`data/*.csv` are Investing.com-style exports: Russian headers (`Дата`, `Цена`, `Откр.`, ...),
`DD.MM.YYYY` dates, `49.609,16` number format (`.` thousands, `,` decimal), volume like `456,51M`, and
rows **newest-first**. The notebook strips `.` then swaps `,`→`.` to parse `Цена`, and does **not** re-sort
by date — its index log-returns are time-reversed and sign-flipped. `src/index_data.py::load_index` keeps
that behaviour on purpose (reproducibility); fix it together with the other Track A3 metric changes.
