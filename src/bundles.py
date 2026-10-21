"""Portable, compressed run artifacts. Bundles never pickle Python objects."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from uuid import uuid4
import zipfile

import numpy as np

from . import scenarios
from .simulation import analyze_result, simulate, TRADER_TYPES
from .run_logging import LogOptions

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(_json_safe(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def rng_state_from_json(state):
    """Restore the numeric array in a JSON-serialized NumPy RandomState tuple."""
    return (state[0], np.asarray(state[1], dtype=np.uint32), int(state[2]),
            int(state[3]), float(state[4]))


def _provenance():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL, text=True)
        git = {'revision': revision, 'dirty': bool(status), 'status': status}
    except (OSError, subprocess.CalledProcessError):
        git = {'revision': None, 'dirty': None, 'status': 'Git metadata unavailable'}
    hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((REPO_ROOT / 'src').glob('*.py'))}
    return {
        'git': git, 'source_sha256': hashes, 'python': sys.version,
        'platform': platform.platform(),
        'packages': {dist.metadata['Name']: dist.version for dist in metadata.distributions()
                     if dist.metadata['Name']},
    }


def effective_configuration(result):
    """Record scenario overrides and fixed legacy settings beside the source hash.

    These fixed settings document the implementation, not additional model knobs.
    Replay also verifies source hashes so fixed-rule changes cannot go unnoticed.
    """
    return {
        'scenario': asdict(result.scenario), 'days': result.days,
        'shared_parameters': {key: value for key, value in vars(scenarios).items()
                              if key.isupper() and key not in ('SCENARIOS',)},
        'fixed_rules': {
            'initial_price': 100, 'initial_dividend': 0.016,
            'warmup': {'steps': 20, 'drift': 0.0003, 'volatility': 0.02},
            'dividends': {'drift': 0.0008, 'volatility': 0.0008,
                          'shock_probability': 0.01, 'shock_scale': 0.005, 'floor': 0},
            'circuit_breaker': {'threshold': 0.20, 'duration': 5},
            'evolution': {'interval': 100, 'noise_to_chartist': 2,
                          'chartist_to_fundamentalist': 1, 'chartist_window': 20},
            'noise': {'rn': 0.05, 'buy_threshold': 0.6, 'sell_threshold': 0.4,
                      'quantity_fraction': 0.1},
            'chartist': {'markup': 0.02, 'volatility_multiplier': 10},
            'fundamentalist': {'excess_return_threshold': 0.05, 'risk_scale': 0.1,
                               'allocation_min': 0.1, 'allocation_max': 0.9,
                               'buy_multiplier': 1.01, 'sell_multiplier': 0.99},
            'matching': {'noise_scale': 0.05, 'liquidity_scale': 1000, 'price_floor': 0.01},
            'status_interval': 100, 'initial_wealth_price': 100,
        },
    }


def _write_histories(path, result):
    # Stream one numeric array at a time into NPZ. Individual wealth histories
    # are flattened with offsets without building a second full population copy.
    def write_array(archive, name, value):
        with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
            np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)

    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        write_array(archive, 'price', result.asset.price_history)
        write_array(archive, 'dividend', result.asset.dividend_history)
        for key, values in result.wealth_history.items():
            write_array(archive, key, values)
        write_array(archive, 'population_days', [r['day'] for r in result.trader_counts_history])
        for kind in TRADER_TYPES:
            write_array(archive, 'population_' + kind,
                        [r[kind] for r in result.trader_counts_history])
        offsets = np.zeros(len(result.traders) + 1, dtype=np.int64)
        for i, trader in enumerate(result.traders):
            offsets[i + 1] = offsets[i] + len(trader.wealth_history)
        write_array(archive, 'trader_wealth_offsets', offsets)
        with archive.open('trader_wealth.npy', 'w', force_zip64=True) as stream:
            np.lib.format.write_array_header_2_0(stream, {
                'descr': np.dtype('<f8').str, 'fortran_order': False, 'shape': (int(offsets[-1]),),
            })
            for trader in result.traders:
                stream.write(np.asarray(trader.wealth_history, dtype='<f8').tobytes())


def save_bundle(result, directory, *, seed=None, invocation=None, show=False):
    """Save a full run and plots in a new folder; COMPLETE is written last.

    Analyze immediately after simulate for legacy metric values. Saving an
    already analyzed result does not advance NumPy's random stream. Source files
    and installed packages must remain unchanged between running and exporting.
    """
    analysis = analyze_result(result)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    scenario_name = re.sub(r'[^A-Za-z0-9_.-]', '_', result.scenario.name) or 'scenario'
    seed_name = str(seed) if seed is not None else 'unseeded'
    parent = Path(directory)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f'{scenario_name}_{seed_name}_{stamp}_{uuid4().hex[:12]}'
    destination.mkdir()
    manifest = {
        'schema_version': SCHEMA_VERSION, 'created_utc': stamp,
        'configuration': effective_configuration(result), 'seed': seed,
        'invocation': list(invocation) if invocation is not None else None,
        'rng_state_before': result.rng_state_before, 'rng_state_after': result.rng_state_after,
        'analysis_rng_state_before': analysis.rng_state_before,
        'analysis_rng_state_after': analysis.rng_state_after,
        'provenance': _provenance(),
        'history_indexing': 'price/dividend are legacy observation sequences, including warm-up; not daily alignment',
        'logging': result.run_log.metadata() if result.run_log is not None else None,
    }
    _write_json(destination / 'manifest.json', manifest)
    _write_json(destination / 'analysis.json', {
        'summary': analysis.summary,
        'metrics': {key: asdict(value) for key, value in analysis.metrics.items()},
        'report_lines': analysis.report_lines,
    })
    _write_histories(destination / 'histories.npz', result)
    _write_json(destination / 'traders.json', [
        {'index': i, 'type': trader.type, 'cash': trader.cash, 'assets': trader.assets,
         'active': trader.active, 'initial_wealth': result.initial_wealths.get(id(trader))}
        for i, trader in enumerate(result.traders)
    ])
    _write_json(destination / 'events.json', {
        'halt_events': result.halt_events, 'evolution_log': result.evolution_log,
        'trader_counts_history': result.trader_counts_history,
    })
    if result.run_log is not None:
        result.run_log.export(destination / 'logs')
    from .plotting import plot_price, plot_wealth_history, plot_wealth_histograms, plot_trader_counts
    import matplotlib.pyplot as plt

    figures_before = set(plt.get_fignums())
    try:
        plot_price(result.asset, title=f'{result.scenario.title}: asset price',
                   show_halts=result.scenario.halt_check, save_path=destination / 'price.png')
        plot_wealth_history(result.wealth_history, save_path=destination / 'wealth.png')
        plot_wealth_histograms(result.traders, save_path=destination / 'wealth_hist.png')
        if result.scenario.evolution:
            plot_trader_counts(result.asset, save_path=destination / 'trader_counts.png')
        # All artifacts are complete before waiting for an interactive window.
        (destination / 'COMPLETE').write_text(f'schema_version={SCHEMA_VERSION}\n', encoding='utf-8')
        if show:
            plt.show()
    finally:
        for figure in set(plt.get_fignums()) - figures_before:
            plt.close(figure)
    return destination


def replay_bundle(directory):
    """Replay with recorded simulation and analysis RNG states, preserving caller RNG.

    Requires matching source files and runtime package versions. A Git dirty flag
    alone does not prevent replay when the recorded file hashes still match.
    """
    directory = Path(directory)
    if not (directory / 'COMPLETE').is_file():
        raise ValueError('Bundle is incomplete (missing COMPLETE marker)')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported bundle schema version')
    current = _provenance()
    expected = manifest['provenance']
    if current['python'] != expected['python']:
        raise ValueError('Python runtime differs from the recorded run')
    if current['source_sha256'] != expected['source_sha256']:
        raise ValueError('Source files differ from the recorded run; restore the recorded source before replay')
    # Compare only the runtime closure, not unrelated tools installed in the environment.
    from packaging.requirements import Requirement
    pending = ['nolds', 'numpy', 'pandas', 'matplotlib', 'scikit-learn']
    checked = set()
    while pending:
        dist = metadata.distribution(pending.pop())
        name = dist.metadata['Name']
        if name in checked:
            continue
        checked.add(name)
        if current['packages'].get(name) != expected['packages'].get(name):
            raise ValueError(f'Runtime dependency differs: {name}')
        for dependency in dist.requires or []:
            requirement = Requirement(dependency)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    state = np.random.get_state()
    try:
        configuration = manifest['configuration']
        np.random.set_state(rng_state_from_json(manifest['rng_state_before']))
        logging = manifest.get('logging')
        options = LogOptions(**logging['options']) if logging else LogOptions(level='off')
        result = simulate(scenarios.Scenario(**configuration['scenario']), days=configuration['days'],
                          log_options=options)
        if _json_safe(effective_configuration(result)) != configuration:
            raise ValueError('Effective model settings differ from the recorded run')
        np.random.set_state(rng_state_from_json(manifest['analysis_rng_state_before']))
        analyze_result(result)
        return result
    finally:
        np.random.set_state(state)
