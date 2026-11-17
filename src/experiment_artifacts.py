"""Versioned compact experiment outputs; independent of schema-1 replay bundles."""

from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from src.bundles import effective_configuration
from src.run_logging import json_safe
from src.scenarios import SCENARIOS, Scenario

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ARTIFACT_ERRORS = (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile)
STRATEGIES = ('noise', 'chartist', 'fundamentalist')
DAILY_FIELDS = {
    'price': ('float64', 'currency/share'),
    'previous_price': ('float64', 'currency/share'),
    'dividend': ('float64', 'currency/share'),
    'volume': ('float64', 'shares'),
    'trade_count': ('int64', 'fills'),
    'trading_skipped': ('bool', 'circuit-breaker skip'),
    'halt_flag': ('bool', 'asset halt flag'),
    'outstanding_buy_orders': ('int64', 'orders after matching, before expiry'),
    'outstanding_sell_orders': ('int64', 'orders after matching, before expiry'),
}
GROUP_FIELDS = {
    'active_count': ('int64', 'active traders'),
    'total_cash': ('float64', 'currency, active traders'),
    'total_assets': ('float64', 'shares, active traders'),
    'total_wealth': ('float64', 'currency, active traders'),
    'buy_volume': ('float64', 'shares'),
    'sell_volume': ('float64', 'shares'),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    """Exclusive create and fsync; publication is handled at directory level."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def execution_fingerprint():
    pending = ['numpy', 'nolds', 'pandas', 'matplotlib', 'scikit-learn', 'packaging']
    packages = {}
    while pending:
        distribution = metadata.distribution(pending.pop())
        name = canonicalize_name(distribution.metadata['Name'])
        if name in packages:
            continue
        packages[name] = distribution.version
        for dependency in distribution.requires or []:
            requirement = Requirement(dependency)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    sources = sorted((ROOT / 'src').glob('*.py')) + [ROOT / 'experiments/run_experiment.py']
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sources}}


def provenance():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        git = {'revision': revision, 'dirty': bool(status), 'status': status}
    except (OSError, subprocess.CalledProcessError):
        git = {'revision': None, 'dirty': None, 'status': 'unavailable'}
    return {'git': git, 'execution': execution_fingerprint()}


def _keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f'Missing fields: {sorted(missing)}; unknown fields: {sorted(unknown)}')


def _integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum if maximum is not None else "unbounded"}]')


def expand_spec(spec, fingerprint=None):
    """Resolve cells into unique run dictionaries without consuming any RNG draws.

    Labels and cell order do not determine identity. Resolved Scenario fields,
    seed, days, burn-in and execution fingerprint do.
    """
    _keys(spec, ('schema_version', 'name', 'days', 'burn_in', 'cells'))
    if type(spec['schema_version']) is not int or spec['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported experiment specification schema')
    if not isinstance(spec['name'], str) or not spec['name'].strip():
        raise ValueError('Experiment name must be a nonempty string')
    _integer(spec['days'], 'days', 1)
    _integer(spec['burn_in'], 'burn_in', 0, spec['days'] - 1)
    if not isinstance(spec['cells'], list) or not spec['cells']:
        raise ValueError('cells must be a nonempty list')
    fingerprint = fingerprint if fingerprint is not None else execution_fingerprint()
    names, runs = set(), {}
    for cell in spec['cells']:
        _keys(cell, ('name', 'scenario', 'seeds'), ('overrides',))
        name = cell['name']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Cell names must be nonempty and unique')
        names.add(name)
        if not isinstance(cell['scenario'], str) or cell['scenario'] not in SCENARIOS:
            raise ValueError(f'Unknown scenario: {cell["scenario"]}')
        values = asdict(SCENARIOS[cell['scenario']])
        overrides = cell.get('overrides', {})
        _keys(overrides, (), set(values) - {'name', 'title'})
        values.update(overrides)
        for key in ('circuit_breaker_asset', 'halt_check', 'evolution'):
            if type(values[key]) is not bool:
                raise ValueError(f'{key} must be boolean')
        scenario = Scenario(**values)
        scenario.validate()
        if not isinstance(cell['seeds'], list) or not cell['seeds']:
            raise ValueError('Each cell requires an explicit nonempty seed list')
        for seed in cell['seeds']:
            _integer(seed, 'seed', 0, 2**32 - 1)
            run = {'scenario': asdict(scenario), 'seed': seed,
                   'days': spec['days'], 'burn_in': spec['burn_in']}
            run_id = digest({'run': run, 'execution': fingerprint})
            if run_id not in runs:
                runs[run_id] = {'run_id': run_id, **run, 'cells': []}
            if name not in runs[run_id]['cells']:
                runs[run_id]['cells'].append(name)
    return list(runs.values())


def daily_arrays(run_log, days):
    """Stream the logs into numeric arrays, never into a dataframe."""
    arrays = {key: np.empty(days, dtype=dtype) for key, (dtype, _) in DAILY_FIELDS.items()}
    arrays.update({key: np.empty((days, 3), dtype=dtype)
                   for key, (dtype, _) in GROUP_FIELDS.items()})
    arrays['day'] = np.arange(days, dtype=np.int64)
    count = 0
    for record in run_log.iter_records('steps'):
        if record['phase'] != 'trading':
            continue
        if record['day'] != count or count >= days:
            raise ValueError('Trading log is not contiguous')
        for key in DAILY_FIELDS:
            arrays[key][count] = record[key]
        count += 1
    if count != days:
        raise ValueError('Incomplete daily log')
    count = 0
    for record in run_log.iter_records('strategies'):
        day, column = divmod(count, 3)
        if day >= days or record['day'] != day or record['type'] != STRATEGIES[column]:
            raise ValueError('Strategy log is not contiguous')
        for key in GROUP_FIELDS:
            arrays[key][day, column] = record[key]
        count += 1
    if count != days * 3:
        raise ValueError('Incomplete strategy log')
    arrays['log_return'] = np.log(arrays['price']) - np.log(arrays['previous_price'])
    return arrays


def summarize(arrays, burn_in):
    price, previous = arrays['price'][burn_in:], arrays['previous_price'][burn_in:]
    returns, volume = arrays['log_return'][burn_in:], arrays['volume'][burn_in:]
    longest = terminal = 0
    for empty in volume == 0:
        terminal = terminal + 1 if empty else 0
        longest = max(longest, terminal)
    path = np.concatenate(([previous[0]], price))
    values = {
        'final_price': float(price[-1]),
        'log_return_volatility': float(np.std(returns, ddof=1)) if len(returns) > 1 else None,
        'maximum_drawdown': float(np.max(1 - path / np.maximum.accumulate(path))),
        'zero_volume_fraction': float(np.mean(volume == 0)),
        'longest_no_trade_streak': longest, 'terminal_no_trade_streak': terminal,
    }
    reasons = {}
    for name, value in values.items():
        if value is None or not math.isfinite(value):
            values[name] = None
            reasons[name] = ('requires at least two daily returns' if len(returns) == 1
                             and name == 'log_return_volatility' else 'non-finite input or result')
    return {'burn_in': burn_in, 'observations': len(price), 'values': values, 'undefined': reasons}


def save_compact(result, run, destination, fingerprint):
    """Write to a new staging directory. No plotting or legacy metric estimation."""
    destination = Path(destination)
    destination.mkdir()
    arrays = daily_arrays(result.run_log, run['days'])
    with (destination / 'daily.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_json(destination / 'summary.json', summarize(arrays, run['burn_in']))
    write_json(destination / 'metadata.json', {
        'artifact_schema_version': SCHEMA_VERSION, 'run': run,
        'execution': fingerprint, 'configuration': effective_configuration(result),
        'rng_state_before': [json_safe(x.tolist() if isinstance(x, np.ndarray) else x)
                             for x in result.rng_state_before],
        'rng_state_after': [json_safe(x.tolist() if isinstance(x, np.ndarray) else x)
                            for x in result.rng_state_after],
        'analysis': 'lightweight daily summaries only; no fitted estimators or estimator seed',
        'day_indexing': 'zero-based trading days; warm-up only in logs',
        'pre_trading_price': float(arrays['previous_price'][0]),
        'strategy_columns': STRATEGIES,
        'arrays': {key: {'dtype': str(value.dtype), 'shape': list(value.shape),
                         'units': (DAILY_FIELDS | GROUP_FIELDS).get(key, (None,
                                   'trading day' if key == 'day' else 'natural log ratio'))[1]}
                   for key, value in arrays.items()},
        'summary_conventions': {'window': '[burn_in, days)', 'volatility_ddof': 1,
                                'drawdown': 'nonnegative fractional loss from running peak; includes pre-window price',
                                'stalls': 'consecutive zero-volume trading days, including halts'},
    })
    result.run_log.export(destination / 'logs')
    # Compact runs have no trader snapshots. IDs remain useful within event logs.
    log_meta_path = destination / 'logs/metadata.json'
    log_meta = read_json(log_meta_path)
    log_meta['trader_ids'] = 'run-local indices; initial noise/chartist/fundamentalist order; evolution appends indices; no trader snapshots retained'
    with log_meta_path.open('w', encoding='utf-8') as stream:
        json.dump(log_meta, stream, indent=2, allow_nan=False)
        stream.write('\n')
    files = {str(path.relative_to(destination)): {'sha256': file_hash(path), 'bytes': path.stat().st_size}
             for path in sorted(destination.rglob('*')) if path.is_file()}
    write_json(destination / 'complete.json', {'artifact_schema_version': SCHEMA_VERSION,
                                             'run_id': run['run_id'], 'files': files})


def validate_compact(directory, expected_run=None, fingerprint=None):
    """Validate hashes, numeric layout, scientific identity and daily chronology."""
    directory = Path(directory)
    complete = read_json(directory / 'complete.json')
    if not isinstance(complete, dict) or not isinstance(complete.get('files'), dict):
        raise ValueError('Invalid completion manifest')
    if complete['artifact_schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported compact artifact schema')
    required = {'daily.npz', 'metadata.json', 'summary.json', 'logs/metadata.json',
                'logs/steps.jsonl.gz', 'logs/strategies.jsonl.gz'}
    names = set(complete['files'])
    if not required <= names or names - required - {'logs/events.jsonl.gz'}:
        raise ValueError('Unexpected or missing compact artifact files')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != names | {'complete.json'}:
        raise ValueError('Unmanifested compact artifact files')
    for name, entry in complete['files'].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError(f'Corrupt artifact: {name}')
    meta = read_json(directory / 'metadata.json')
    run = meta['run']
    scientific = {key: run[key] for key in ('scenario', 'seed', 'days', 'burn_in')}
    if (meta['artifact_schema_version'] != SCHEMA_VERSION
            or digest({'run': scientific, 'execution': meta['execution']}) != run['run_id']
            or complete['run_id'] != run['run_id'] or (expected_run is not None and run != expected_run)
            or (fingerprint is not None and meta['execution'] != fingerprint)):
        raise ValueError('Artifact run identity/provenance mismatch')
    if (meta['configuration']['scenario'] != run['scenario'] or meta['configuration']['days'] != run['days']):
        raise ValueError('Artifact effective configuration mismatch')
    # Readers can inspect historical data under a newer source tree. Resume and
    # publication additionally require the current effective rules to agree.
    if fingerprint is not None:
        expected_config = effective_configuration(SimpleNamespace(scenario=Scenario(**run['scenario']), days=run['days']))
        if meta['configuration'] != expected_config:
            raise ValueError('Artifact effective configuration mismatch')
    with np.load(directory / 'daily.npz', allow_pickle=False) as archive:
        if set(archive.files) != set(DAILY_FIELDS) | set(GROUP_FIELDS) | {'day', 'log_return'}:
            raise ValueError('Invalid numeric array fields')
        for key in archive.files:
            array = archive[key]
            expected_shape = (run['days'], 3) if key in GROUP_FIELDS else (run['days'],)
            dtype = (DAILY_FIELDS | GROUP_FIELDS).get(key, ('int64' if key == 'day' else 'float64',))[0]
            if array.dtype != np.dtype(dtype) or array.shape != expected_shape:
                raise ValueError(f'Invalid array layout: {key}')
        if not np.array_equal(archive['day'], np.arange(run['days'])):
            raise ValueError('Invalid daily chronology')
        if not np.array_equal(archive['previous_price'][1:], archive['price'][:-1], equal_nan=True):
            raise ValueError('Daily prices are not contiguous')
    return meta


def read_compact(directory):
    """Read one validated run; callers should aggregate ensembles incrementally."""
    meta = validate_compact(directory)
    with np.load(Path(directory) / 'daily.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return meta, arrays, read_json(Path(directory) / 'summary.json')
