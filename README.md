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

`data/` contains CSV exports for RTS, MOEX, S&P 500, DJIA, and NASDAQ.
`src/index_data.py` parses their Russian headers and localized numeric format.

## Reproducibility and model caveats

This refactor preserves the original notebook's numerical behavior. In particular:

- Random draws use the global NumPy stream. RANSAC metric fitting also uses
  randomness; seed before each run or standalone metric calculation. Dependency
  versions can affect results.
- Prices include a 20-step exogenous geometric Brownian motion warm-up.
- Unmatched orders persist between days. Large populations and long runs can be
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
    simulation.py     Shared simulation loop and reporting
    metrics.py        Log-return estimators
    plotting.py       Reusable plotting functions
    index_data.py     Market-index CSV loading and reports
experiments/
    run_scenario.py   Command-line entry point
tests/
    test_legacy_equivalence.py
data/                Market-index CSV inputs
Multiagent_Modelling_notebook.ipynb
```

## Development and validation

The current focus is code organization, readability, and English documentation.
Model changes, ensemble studies, and new estimators are separate research work.
Keep reusable logic in `src/`, scenarios in the configuration module, and preserve
random-draw order when refactoring.

The standalone regression runner compares prices, dividends, trader balances,
wealth histories, and metrics against the original notebook in Git commit
`84fa2a6`. Keep that commit available in your local history.

```bash
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
