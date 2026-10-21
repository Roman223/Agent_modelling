"""Check that the package reproduces the original notebook's simulations bit-for-bit.

The reference is the notebook as committed at LEGACY_COMMIT, read straight from git, so no copy of the old
code lives in the working tree. Its class/metric cells and the scenario's `run_simulation` cell are exec'd in a
fresh namespace (replaying cell 42's `Asset` redefinition where the notebook's execution order applied it),
then both implementations run under the same global `np.random.seed` and every trader balance, price,
dividend and wealth-history value is compared exactly.

Usage:
    python tests/test_legacy_equivalence.py            # 250 days, seeds 0-2, all scenarios
    python tests/test_legacy_equivalence.py --full     # 1000 days
    python tests/test_legacy_equivalence.py --scenario evolution --days 120 --seeds 5
"""
import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import time
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import numpy as np

# Emitted by nolds when scikit-learn is absent (it then fits with 'poly'); identical on both sides.
warnings.filterwarnings('ignore', message="fitting mode 'RANSAC' requires the package sklearn")

LEGACY_COMMIT = '84fa2a6'
NOTEBOOK = 'Multiagent_Modelling_notebook.ipynb'

# Imports, Asset, OrderBook, Trader, NoiseTrader, ChartistTrader, FundamentalistTrader, metric functions.
DEFINITION_CELLS = [2, 4, 6, 8, 10, 12, 14, 16]
CB_ASSET_CELL = 42

# scenario -> (extra cells exec'd before the runner, runner cell, optional (old, new) source substitution).
# Cell 42 redefines Asset with a circuit breaker and stays in effect for every later cell, so the
# evolution cell (47) ran on that Asset too.
LEGACY_RUNNERS = {
    'baseline': ([], 18, None),
    'more_noise': ([], 24, None),
    'more_chartists': ([], 30, None),
    'more_fundamentalists': ([], 36, None),
    'circuit_breaker': ([CB_ASSET_CELL], 43, None),
    'evolution': ([CB_ASSET_CELL], 47, None),
    # Not in the notebook; the old experiments/run_window_*.py were cell 18 with a different chartist window.
    'window_5': ([], 18, ('window=20', 'window=5')),
    'window_50': ([], 18, ('window=20', 'window=50')),
}

# Seeds 0-2 never trip the circuit breaker; with seed 18 the warm-up drifts >20% and trading halts on days 0-3.
EXTRA_SEEDS = {'circuit_breaker': [18]}

_legacy_cells = None


def legacy_cells():
    global _legacy_cells
    if _legacy_cells is None:
        raw = subprocess.run(['git', 'show', f'{LEGACY_COMMIT}:{NOTEBOOK}'], cwd=REPO_ROOT,
                             check=True, capture_output=True).stdout
        _legacy_cells = [''.join(c['source']) for c in json.loads(raw)['cells']]
        # Guard against the indices above silently pointing at the wrong cells.
        assert _legacy_cells[4].startswith('class Asset'), 'cell 4 is not Asset'
        assert _legacy_cells[CB_ASSET_CELL].startswith('class Asset'), 'cell 42 is not the circuit-breaker Asset'
        for _, runner, _ in LEGACY_RUNNERS.values():
            assert 'def run_simulation(' in _legacy_cells[runner], f'cell {runner} does not define run_simulation'
    return _legacy_cells


def legacy_namespace(scenario):
    extra, runner, substitution = LEGACY_RUNNERS[scenario]
    cells = legacy_cells()
    ns = {'__name__': f'legacy_{scenario}'}
    for i in DEFINITION_CELLS + extra:
        exec(compile(cells[i], f'<notebook cell {i}>', 'exec'), ns)
    source = cells[runner]
    if substitution:
        old, new = substitution
        assert source.count(old) == 1, f'{old!r} must occur exactly once in cell {runner}'
        source = source.replace(old, new)
    exec(compile(source, f'<notebook cell {runner}>', 'exec'), ns)
    return ns


def run_legacy(scenario, days, seed):
    ns = legacy_namespace(scenario)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        asset, traders = ns['run_simulation'](days=days)
    return snapshot(asset, traders, ns['wealth_history']), ns


def run_package(scenario, days, seed):
    from src import simulation
    from src.scenarios import SCENARIOS
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        asset, traders = simulation.run_simulation(SCENARIOS[scenario], days=days)
    return snapshot(asset, traders, simulation.wealth_history)


def snapshot(asset, traders, wealth_history):
    return {
        'rng_state': np.random.get_state(),
        'events': {name: copy.deepcopy(getattr(asset, name)) for name in
                   ('halt_events', 'evolution_log', 'trader_counts_history') if hasattr(asset, name)},
        'price_history': np.array(asset.price_history),
        'dividend_history': np.array(asset.dividend_history),
        'traders': [(t.type, t.cash, t.assets, t.active, np.array(t.wealth_history)) for t in traders],
        'wealth_history': copy.deepcopy(wealth_history),
    }


def differences(legacy, new):
    diffs = []
    if any(not np.array_equal(a, b) for a, b in zip(legacy['rng_state'], new['rng_state'])):
        diffs.append('final NumPy RNG state differs')
    for name, events in legacy['events'].items():
        if events != new['events'].get(name):
            diffs.append(f'{name} differs')
    for key in ('price_history', 'dividend_history'):
        if not np.array_equal(legacy[key], new[key]):
            diffs.append(f'{key}: first mismatch at index {_first_mismatch(legacy[key], new[key])}')
    if len(legacy['traders']) != len(new['traders']):
        diffs.append(f"trader count {len(legacy['traders'])} != {len(new['traders'])}")
    else:
        for i, (a, b) in enumerate(zip(legacy['traders'], new['traders'])):
            if a[:4] != b[:4] or not np.array_equal(a[4], b[4]):
                diffs.append(f'trader {i}: {a[:4]} != {b[:4]}')
                break
    if legacy['wealth_history'] != new['wealth_history']:
        diffs.append('wealth_history dict differs')
    return diffs


def _first_mismatch(a, b):
    n = min(len(a), len(b))
    idx = np.flatnonzero(a[:n] != b[:n])
    return int(idx[0]) if idx.size else f'length {len(a)} vs {len(b)}'


def check_metrics(legacy_ns, prices):
    """The package's metric functions must give the notebook's values on the same series."""
    from src import metrics
    pairs = [
        ('calculate_hurst_exponent', {}),
        ('calculate_lyapunov_exponent', {'emb_dim': 10, 'lag': 1}),
        ('calculate_kolmogorov_entropy', {'emb_dim': 3}),
    ]
    diffs = []
    for name, kwargs in pairs:
        # With scikit-learn installed, nolds fits by RANSAC, which draws from the global np.random stream.
        np.random.seed(0)
        old = legacy_ns[name](prices, **kwargs)
        np.random.seed(0)
        new = getattr(metrics, name)(prices, **kwargs)
        if not (old == new or (np.isnan(old) and np.isnan(new))):
            diffs.append(f'{name}: {old!r} != {new!r}')
    return diffs


def check(scenarios, days, seeds, verbose=True):
    failures = []
    for scenario in scenarios:
        for seed in list(seeds) + EXTRA_SEEDS.get(scenario, []):
            t0 = time.perf_counter()
            legacy, legacy_ns = run_legacy(scenario, days, seed)
            new = run_package(scenario, days, seed)
            diffs = differences(legacy, new)
            if not diffs:
                diffs = check_metrics(legacy_ns, legacy['price_history'])
            status = 'OK  ' if not diffs else 'FAIL'
            if verbose:
                print(f'{status} {scenario:22s} seed={seed} days={days} '
                      f'final_price={new["price_history"][-1]:.4f} ({time.perf_counter() - t0:.1f}s)', flush=True)
                for d in diffs:
                    print(f'       {d}', flush=True)
            if diffs:
                failures.append((scenario, seed, diffs))
    return failures


def test_all_scenarios_match_notebook():
    failures = check(list(LEGACY_RUNNERS), days=250, seeds=[0, 1, 2], verbose=False)
    assert not failures, failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario', action='append', choices=list(LEGACY_RUNNERS),
                        help='repeatable; default: all scenarios')
    parser.add_argument('--days', type=int, default=250)
    parser.add_argument('--full', action='store_true', help='shorthand for --days 1000')
    parser.add_argument('--seeds', type=int, default=3, help='runs seeds 0..N-1')
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error('--seeds must be a positive count (use 1 to run seed 0)')
    if args.days <= 0:
        parser.error('--days must be a positive integer')
    failures = check(args.scenario or list(LEGACY_RUNNERS), 1000 if args.full else args.days, range(args.seeds))
    print(f'\n{len(failures)} failing run(s)' if failures else '\nall runs identical to the notebook')
    sys.exit(1 if failures else 0)
