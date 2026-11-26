"""Resumable Task 6 references, exact replay, and factorial recovery study."""

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from src.atlas_runner import manifest_at
from src.experiment_artifacts import (ROOT, digest, execution_fingerprint, file_hash, provenance,
                                      read_json, write_json)
from src.experiments import (Limits, MIB, _fsync_tree, directory_bytes, disk_guard,
                             experiment_lock, now, status)
from src.heterogeneity_study import _phase, publish_manifest, validate_seeds
from src.inactivity import (ARMS, REASONS, MATCH_REASONS, MATCH_COUNTS, GROUP_FIELDS,
                            SCALAR_FIELDS, PRIMARY_FIELDS, positive_integer, run_reference,
                            run_continuation, intervene, with_future, spells)
from src.inactivity_analysis import SETTINGS, reference_summary, recovery_summary
from src.market_checkpoints import read_checkpoint, write_checkpoint, validate_scenario
from src.scenarios import Scenario

CLI = ROOT / 'experiments/run_inactivity.py'


def fingerprint():
    result = execution_fingerprint()
    result['source_sha256'][str(CLI.relative_to(ROOT))] = file_hash(CLI)
    return result


def future_seeds(cell, seed, count):
    return [int(digest({'cell': cell, 'seed': seed, 'future': f, 'task': 6})[:8], 16)
            for f in range(count)]


def select_configurations(atlas):
    atlas = Path(atlas).resolve()
    if not (atlas / 'manifest.json').exists():
        atlas /= 'simulation'
    manifest = manifest_at(atlas)
    wanted = {(x, y, f'v{v}'): f'{name}-v{v}' for x, y, name in
              (('4', '3', 'reference'), ('6', '0', 'liquidity')) for v in range(8)}
    selected = {}
    for run in manifest['runs']:
        for cell in run['cells']:
            parts = cell.split('-')
            if len(parts) == 5 and parts[0] == 'grid' and tuple(parts[1:4]) in wanted:
                name, scenario = wanted[tuple(parts[1:4])], run['scenario']
                validate_scenario(Scenario(**scenario))
                if (scenario['noise'], scenario['chartists'], scenario['fundamentalists'], scenario['chartist_window']) != (50, 20, 30, 20):
                    raise ValueError('Atlas anchor has unexpected population/window')
                if name in selected and selected[name] != scenario:
                    raise ValueError('Atlas anchor configurations disagree')
                selected[name] = scenario
    if set(selected) != set(wanted.values()):
        raise ValueError('Task 6 requires all sixteen stored atlas controls')
    expected = {(n, d, l) for n in (0, .05) for d in ('constant', 'stochastic') for l in ('day', 'persistent')}
    for anchor in ('reference', 'liquidity'):
        configs = [s for k, s in selected.items() if k.startswith(anchor)]
        if {(s['closing_noise_scale'], s['dividend_mode'], s['order_lifetime']) for s in configs} != expected:
            raise ValueError('Missing forcing-control combinations')
        if len({(s['chartist_alpha'], s['fundamentalist_gamma']) for s in configs}) != 1:
            raise ValueError('Anchor coordinates differ between controls')
    return selected, {'path': str(atlas / 'manifest.json'), 'sha256': digest(manifest)}


def generate_study(atlas, root, *, seeds=tuple(range(600, 650)), days=10240,
                   threshold=100, continuation_days=1000, futures=5, tail_days=100,
                   resamples=2000):
    validate_seeds(seeds)
    for name, value in dict(days=days, threshold=threshold, continuation_days=continuation_days,
                            futures=futures, tail_days=tail_days, resamples=resamples).items():
        positive_integer(value, name)
    if threshold > days or tail_days > continuation_days or futures > 20 or days > 20000:
        raise ValueError('Invalid or unbounded study dimensions')
    configs, source = select_configurations(atlas)
    future_map = {cell: {str(seed): future_seeds(cell, seed, futures) for seed in seeds} for cell in sorted(configs)}
    all_futures = [s for cell in future_map.values() for row in cell.values() for s in row]
    if len(set(all_futures)) != len(all_futures):
        raise ValueError('Future seed collision in the requested study')
    settings = {**SETTINGS, 'primary_tail_days': tail_days, 'bootstrap_resamples': resamples,
                'thresholds': sorted(set(SETTINGS['thresholds'] + [threshold]))}
    study = {'schema_version': 1, 'kind': 'inactivity-study', 'created': now(),
             'execution': fingerprint(), 'git': provenance()['git'], 'atlas': source,
             'configurations': configs, 'seeds': list(seeds), 'days': days,
             'threshold': threshold, 'continuation_days': continuation_days,
             'futures': futures, 'future_seeds': future_map, 'arms': list(ARMS), 'analysis_settings': settings,
             'observation_schema': {'day_index': 'one-based trading only', 'decision_reasons': list(REASONS),
                 'match_reasons': list(MATCH_REASONS), 'match_counts': list(MATCH_COUNTS),
                 'groups': ['noise', 'chartist', 'fundamentalist'],
                 'submissions': ['buy_orders', 'sell_orders', 'buy_quantity', 'sell_quantity'],
                 'book_timing': 'after matching before expiry', 'checkpoint_timing': 'after expiry'},
             'protocol': {'stall': '100 exact zero-volume days' if threshold == 100 else f'{threshold} exact zero-volume days',
                 'warmup': 'excluded; no trading burn-in', 'checkpoint': 'first confirmation only',
                 'memory_reset': 'last 20 accepted prices set to current price; shared across strategies',
                 'portfolio_reset': 'active traders; equal target asset/wealth fraction; largest remainders with affordability caps',
                 'future_rng': 'uint32 SHA256 seeds keyed by cell/reference seed/future; common initial state across arms',
                 'conservation': 'integer global shares, global cash and individual marked-to-market wealth',
                 'cash_tolerance': '1e-8 absolute + 1e-12 relative',
                 'replay': 'untouched original RNG; extend reference only for validation tail',
                 'negative_findings': 'finite-horizon nonrecovery is not proof of absorption'}}
    root = Path(root).resolve()
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    if disk_guard(parent, Limits(), reserved=4*MIB) == 'free_disk_limit':
        raise ValueError('Insufficient free disk')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'manifest.json', study)
    write_json(root / 'manifest.sha256.json', {'sha256': digest(study)})
    return study


def load_study(root, *, current=False):
    study = manifest_at(root)
    if study.get('schema_version') != 1 or study.get('kind') != 'inactivity-study':
        raise ValueError('Unsupported inactivity study')
    if current and study['execution'] != fingerprint():
        raise ValueError('Study source/runtime changed; restore recorded sources or create a new study')
    if digest(manifest_at(Path(study['atlas']['path']).parent)) != study['atlas']['sha256']:
        raise ValueError('Atlas source changed')
    return study


def reference_jobs(study):
    jobs = []
    for cell, scenario in sorted(study['configurations'].items()):
        for seed in study['seeds']:
            job = dict(phase='reference', cell=cell, seed=seed, scenario=scenario,
                       days=study['days'], threshold=study['threshold'],
                       continuation_days=study['continuation_days'],
                       thresholds=study['analysis_settings']['thresholds'])
            job['run_id'] = digest({'job': job, 'execution': study['execution']})
            jobs.append(job)
    return jobs


def continuation_jobs(root, study, refs):
    jobs = []
    for ref in refs:
        path = Path(root) / 'reference/runs' / ref['run_id']
        validate_artifact(path, ref, study['execution'])
        summary = read_json(path / 'summary.json')
        if summary['checkpoint_day'] is None:
            continue
        seeds = study['future_seeds'][ref['cell']][str(ref['seed'])]
        if seeds != future_seeds(ref['cell'], ref['seed'], study['futures']):
            raise ValueError('Frozen future seeds differ from the protocol')
        job = dict(phase='continuation', cell=ref['cell'], seed=ref['seed'], source=str(path),
                   source_sha256=file_hash(path / 'complete.json'), checkpoint_day=summary['checkpoint_day'],
                   days=study['continuation_days'], threshold=study['threshold'],
                   tail_days=study['analysis_settings']['primary_tail_days'], future_seeds=seeds)
        job['run_id'] = digest({'job': job, 'execution': study['execution']})
        jobs.append(job)
    return jobs


def read_arrays(path, *, full=False):
    expected = (set(SCALAR_FIELDS) | set(GROUP_FIELDS) | {'day', 'decision_counts', 'submissions', 'match_counts'}
                if full else set(PRIMARY_FIELDS))
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != expected:
            raise ValueError('Invalid inactivity array fields')
        arrays = {k: z[k] for k in z.files}
    n = len(arrays['day'])
    for key, values in arrays.items():
        shape = ((n, 3) if key in GROUP_FIELDS else (n, 3, len(REASONS)) if key == 'decision_counts' else
                 (n, 3, 4) if key == 'submissions' else (n, len(MATCH_COUNTS)) if key == 'match_counts' else (n,))
        if values.shape != shape or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError(f'Invalid numeric field: {key}')
    if not n or arrays['day'][0] < 1 or np.any(np.diff(arrays['day']) != 1):
        raise ValueError('Invalid trading-day chronology')
    if any(np.any(arrays[k] < 0) for k in ('total_cash', 'total_assets', 'volume', 'distinct_volume', 'same_owner_volume')):
        raise ValueError('Negative resources or volume')
    if np.any(arrays['price'] <= 0) or np.any(arrays['previous_price'] <= 0):
        raise ValueError('Nonpositive price')
    if not np.array_equal(arrays['volume'], arrays['distinct_volume'] + arrays['same_owner_volume']):
        raise ValueError('Volume decomposition mismatch')
    if not np.array_equal(arrays['previous_price'][1:], arrays['price'][:-1]):
        raise ValueError('Price chronology mismatch')
    for key in ('total_cash', 'total_assets'):
        sums = arrays[key].sum(axis=1)
        if not np.allclose(sums, sums[0], rtol=1e-11, atol=1e-8):
            raise ValueError('Nonconserved resources')
    if np.any(arrays['total_assets'] != np.floor(arrays['total_assets'])):
        raise ValueError('Noninteger shares')
    return arrays


def validate_artifact(path, job=None, execution=None, *, deep=False):
    # Share byte-level integrity checks; Task 6 has its own scientific schema.
    from src.heterogeneity_study import validate_artifact as validate_bytes
    meta = validate_bytes(path, job, execution)
    path = Path(path)
    job = meta['job']
    if job['phase'] == 'reference':
        summary = read_json(path / 'summary.json')
        expected = {'metadata.json', 'daily.npz', 'summary.json', 'spells.json'}
        if summary['checkpoint_day'] is not None:
            expected |= {'checkpoint.json.gz', 'replay.npz'}
        if deep:
            arrays = read_arrays(path / 'daily.npz', full=True)
            if len(arrays['day']) != job['days'] or arrays['day'][0] != 1:
                raise ValueError('Incomplete reference')
            first = summary['thresholds'][str(job['threshold'])]['first_confirmation']
            if first != summary['checkpoint_day']:
                raise ValueError('Wrong checkpoint selection')
            if first is not None:
                checkpoint = read_checkpoint(path / 'checkpoint.json.gz')
                replay = read_arrays(path / 'replay.npz', full=True)
                if (checkpoint.completed_days != first or len(replay['day']) != job['continuation_days']
                        or replay['day'][0] != first+1):
                    raise ValueError('Invalid replay chronology')
    elif job['phase'] == 'continuation':
        recovery = read_json(path / 'recovery.json')
        if set(recovery['arms']) != set(ARMS) or not recovery['original_rng_replay_verified']:
            raise ValueError('Incomplete recovery or unverified replay')
        expected = {'metadata.json', 'recovery.json'} | {s['file'] for s in recovery['states'].values()}
        if deep:
            for state in recovery['states'].values():
                arrays = read_arrays(path / state['file'])
                if len(arrays['day']) != job['days'] or arrays['day'][0] != job['checkpoint_day'] + 1:
                    raise ValueError('Incomplete continuation')
            for arm in ARMS:
                row = recovery['arms'][arm]
                if len(row['futures']) != len(job['future_seeds']) or any(s not in recovery['states'] for s in row['futures']):
                    raise ValueError('Missing logical recovery futures')
    else:
        raise ValueError('Unknown phase')
    if set(read_json(path / 'complete.json')['files']) != expected:
        raise ValueError('Unexpected artifact files')
    return meta


def materialize(job, destination, execution):
    destination = Path(destination)
    destination.mkdir()
    if job['phase'] == 'reference':
        result = run_reference(Scenario(**job['scenario']), seed=job['seed'], days=job['days'],
                               threshold=job['threshold'], continuation_days=job['continuation_days'])
        np.savez_compressed(destination / 'daily.npz', **result.arrays)
        summary = reference_summary(result.arrays, thresholds=job['thresholds'])
        summary.update(checkpoint_day=result.checkpoint.completed_days if result.checkpoint else None,
                       replay_digest=result.replay_digest, final_digest=result.final_digest)
        write_json(destination / 'summary.json', summary)
        write_json(destination / 'spells.json', spells(result.arrays['volume'], threshold=job['threshold']))
        if result.checkpoint:
            write_checkpoint(destination / 'checkpoint.json.gz', result.checkpoint)
            np.savez_compressed(destination / 'replay.npz', **result.replay_arrays)
    else:
        source = Path(job['source'])
        validate_artifact(source, execution=execution)
        if file_hash(source / 'complete.json') != job['source_sha256']:
            raise ValueError('Reference dependency changed')
        checkpoint = read_checkpoint(source / 'checkpoint.json.gz')
        replay = run_continuation(checkpoint, days=job['days'])
        expected = read_arrays(source / 'replay.npz', full=True)
        for key, values in expected.items():
            if not np.array_equal(values, replay.arrays[key]):
                raise AssertionError(f'Original RNG replay differs: {key}')
        if replay.final_digest != read_json(source / 'summary.json')['replay_digest']:
            raise AssertionError('Original RNG replay final dynamics/RNG differ')
        del replay, expected
        arms, states = {}, {}
        for arm in ARMS:
            transformed, receipt = intervene(checkpoint, arm)
            ids = []
            for seed in job['future_seeds']:
                state = with_future(transformed, seed)
                state_id = state.sha256
                ids.append(state_id)
                if state_id in states:
                    continue
                result = run_continuation(state, days=job['days'])
                filename = f'state_{state_id}.npz'
                np.savez_compressed(destination / filename, **{k: result.arrays[k] for k in PRIMARY_FIELDS})
                states[state_id] = {'file': filename, 'future_seed': seed,
                    'summary': recovery_summary(result.arrays, threshold=job['threshold'], tail_days=job['tail_days']),
                    'final_digest': result.final_digest}
                del result
            arms[arm] = {'receipt': receipt, 'futures': ids}
        write_json(destination / 'recovery.json', {'original_rng_replay_verified': True,
                   'arms': arms, 'states': states, 'logical_continuations': 8*len(job['future_seeds']),
                   'unique_continuations': len(states)})
    publish_manifest(destination, job, execution)
    validate_artifact(destination, job, execution, deep=True)


def worker(request_path):
    request_path = Path(request_path)
    request = read_json(request_path)
    start = time.monotonic()
    try:
        if fingerprint() != request['execution']:
            raise ValueError('Execution changed before worker')
        materialize(request['job'], request_path.parent / 'artifact', request['execution'])
        if fingerprint() != request['execution']:
            raise ValueError('Execution changed during worker')
        _fsync_tree(request_path.parent / 'artifact')
        record = {'state': 'complete', 'seconds': time.monotonic()-start,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'artifact_bytes': directory_bytes(request_path.parent / 'artifact')}
        code = 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        record, code = {'state': 'failed', 'error': f'{type(error).__name__}: {error}'}, 1
    write_json(request_path.parent / 'result.json', record)
    return code


def complete_study(root, *, limits=None, retry_failed=False, repeat=True):
    root, limits = Path(root).resolve(), limits or Limits()
    if limits.minutes > 120:
        raise ValueError('Batches cannot exceed 120 minutes')
    with experiment_lock(root):
        study = load_study(root, current=True)
        refs = reference_jobs(study)
        for phase in ('reference', 'continuation'):
            jobs = refs if phase == 'reference' else continuation_jobs(root, study, refs)
            while True:
                code = _phase(root, phase, jobs, study, limits, time.monotonic()+limits.minutes*60,
                              retry_failed, backend=sys.modules[__name__])
                if not code:
                    break
                if not repeat or status(root / phase)['latest_invocation']['stop_reason'] != 'time_limit':
                    return code
        analyze_study(root, limits=limits, locked=True)
        from src.inactivity_plotting import render_study
        render_study(root, limits=limits, locked=True)
    return 0


def analyze_study(root, *, limits=None, locked=False):
    from src.inactivity_reporting import analyze
    if locked:
        return analyze(Path(root).resolve(), limits or Limits())
    with experiment_lock(root):
        return analyze(Path(root).resolve(), limits or Limits())


def study_status(root):
    root = Path(root).resolve()
    study = load_study(root)
    result = {'study': str(root), 'expected_references': len(reference_jobs(study)),
              'maximum_logical_continuations': len(reference_jobs(study))*8*study['futures'],
              'output_bytes': directory_bytes(root)}
    for phase in ('reference', 'continuation'):
        result[phase] = status(root / phase) if (root / phase / 'ledger.sqlite').exists() else None
    return result
