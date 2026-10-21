# Repository Guidelines

## Project Structure & Module Organization

This Python project models a single-asset financial market with noise, chartist, and fundamentalist traders.
- `src/`: assets, order matching, traders, simulation, scenario definitions, metrics, plotting, and index loading.
- `experiments/run_scenario.py`: shared command-line runner; add scenario configurations in `src/scenarios.py`.
- `tests/test_legacy_equivalence.py`: regression comparisons against the original notebook.
- `data/`: market-index CSV inputs; `Multiagent_Modelling_notebook.ipynb` imports the package for interactive analysis.
- `CONTEXT.md` and `CLAUDE.md`: research conventions and model caveats; read before changing model behavior or metrics.

## Build, Test, and Development Commands

Run from the repository root using the existing virtual environment:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/run_scenario.py --list
.venv/bin/python experiments/run_scenario.py baseline --days 250 --seed 0 --save results --no-show
.venv/bin/python tests/test_legacy_equivalence.py --scenario baseline --days 120 --seeds 1
```

These install dependencies, list scenarios, save simulation plots, and run a focused regression check. There is no separate build step. Use explicit seeds for reproducible runs.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Write new code and documentation in English; existing Russian comments may remain. Keep reusable logic in `src/`. No formatter or linter is configured.

Preserve random-draw ordering during refactors. Name new entropy outputs Sample Entropy (nats), matching the actual estimator.

## Testing Guidelines

Tests use a standalone Python assertion runner, not a configured pytest suite. Keep test files named `test_*.py`. The runner compares prices, dividends, balances, wealth histories, and metrics against Git commit `84fa2a6`; retain access to that history.

Run the script without arguments for the default suite; `--full` extends simulations to 1,000 days. Background long runs and redirect output to a log. No coverage threshold is configured. Explain intentional model changes and update regression expectations explicitly.

## Commit & Pull Request Guidelines

History is sparse: mostly “Initial commit,” plus “Add Python .gitignore.” Use short, imperative subjects. PRs should describe behavior changes, affected scenarios, seeds, and validation commands/results; link relevant issues and include plots when visual results change.

## Resource Limits

Cap workers at six; never use `n_jobs=-1`. RAM is 15 GiB; swap is not headroom. Never load multi-million-row tables into pandas. Aggregate large data with DuckDB or Arrow batches; cap engine memory at 6 GB and temporary spill. Process large tables sequentially and encode string keys before Python conversion. Check disk space before large writes, prefer compressed Parquet, and time one shard before scaling.
