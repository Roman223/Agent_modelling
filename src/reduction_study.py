"""Task-4 inputs, separately versioned artifacts, and bounded run supervision."""

from dataclasses import asdict, replace
import gzip
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np

from src.atlas_runner import atomic_json, manifest_at
from src.experiment_artifacts import (ARTIFACT_ERRORS, ROOT, canonical, digest, execution_fingerprint,
                                      file_hash, provenance, read_compact, read_json, summarize,
                                      validate_compact, write_json)
from src.experiments import (GIB, MIB, Limits, _connect, _fsync_directory, _fsync_tree, _remove_stage,
                             _terminate, directory_bytes, disk_guard, experiment_lock, memory_available,
                             now, resident_bytes, status)
from src.reduction import (DIVIDEND, STRATEGIES, Order, ReductionConfig, ReductionState, initial_state,
                           quote_grid, simulate_reduction, validate_state)

CLI = ROOT / 'experiments/run_reduction.py'
SCHEMA = 1
FILES = {'daily.npz', 'metadata.json', 'final_state.json.gz', 'summary.json'}
GROUP_FIELDS = {'total_cash', 'total_assets', 'total_wealth', 'buy_volume', 'sell_volume',
                'submitted_buy_mass', 'submitted_sell_mass'}
INT_FIELDS = {'day', 'trade_count', 'outstanding_buy_orders', 'outstanding_sell_orders'}
SCALAR_FIELDS = {'price', 'previous_price', 'volume', 'notional', 'within_strategy_volume',
                 'outstanding_buy_quantity', 'outstanding_sell_quantity', 'log_return'}
ARRAY_FIELDS = GROUP_FIELDS | INT_FIELDS | SCALAR_FIELDS | {'accepted_price_update'}


def fingerprint():
    value = execution_fingerprint()
    value['source_sha256']['experiments/run_reduction.py'] = file_hash(CLI)
    return value


def state_dict(state):
    return asdict(state)


def state_from_dict(value):
    if set(value) != {'cash', 'shares', 'memory', 'bids', 'asks'}:
        raise ValueError('Invalid serialized state fields')
    return ReductionState(value['cash'], value['shares'], value['memory'],
                          tuple(Order(**x) for x in value['bids']), tuple(Order(**x) for x in value['asks']))


def select_anchors(manifest, seeds):
    """Select by stored coordinate labels; never reconstruct rounded alpha."""
    selected = {}
    for run in manifest['runs']:
        if run['seed'] not in seeds:
            continue
        for label in run['cells']:
            fields = label.split('-')
            if len(fields) != 5 or fields[0] != 'grid':
                continue
            _, i, j, variant, _ = fields
            coordinate = ('reference' if (i, j) == ('4', '3') else
                          'liquidity' if (i, j) == ('6', '0') and variant in ('v6', 'v7') else None)
            if coordinate:
                key = (f'{coordinate}-{variant}', run['seed'])
                if key in selected:
                    raise ValueError('Duplicate anchor coordinate')
                selected[key] = run
    expected = {(f'reference-v{v}', seed) for seed in seeds for v in range(8)}
    expected |= {(f'liquidity-v{v}', seed) for seed in seeds for v in (6, 7)}
    if selected.keys() != expected:
        raise ValueError('Historical manifest is missing the requested anchor cells/seeds')
    return selected


def _input(directory, run, cell, cohort):
    artifact = Path(directory).resolve() / 'runs' / run['run_id']
    validate_compact(artifact, expected_run=run)
    return {'artifact': str(artifact), 'complete_sha256': file_hash(artifact / 'complete.json'),
            'run': run, 'cell': cell, 'cohort': cohort}


def validate_input(entry):
    artifact = Path(entry['artifact'])
    validate_compact(artifact, expected_run=entry['run'])
    if file_hash(artifact / 'complete.json') != entry['complete_sha256']:
        raise ValueError('Comparison input changed')


def warmup_memory(entry):
    prices = []
    with gzip.open(Path(entry['artifact']) / 'logs/steps.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] != 'warmup':
                break
            if row['price_update_accepted']:
                prices.append(row['price'])
    if len(prices) != 20:
        raise ValueError('Expected the complete 20-step accepted warm-up')
    return prices


def generate_study(atlas, root, *, historical_seeds=(100, 101, 102, 103, 104),
                   fresh_seeds=(110, 111, 112, 113, 114), points=(16, 8, 32)):
    atlas, root = Path(atlas).resolve(), Path(root).resolve()
    if not (atlas / 'manifest.json').exists():
        atlas = atlas / 'simulation'
    for name, seeds in (('historical', historical_seeds), ('fresh', fresh_seeds)):
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
            raise ValueError(f'{name} seeds must be unique nonnegative uint32 integers')
    if set(historical_seeds) & set(fresh_seeds):
        raise ValueError('Fresh and historical seed sets must be disjoint')
    if not points or len(set(points)) != len(points):
        raise ValueError('Quote resolutions must be nonempty and unique')
    for k in points:
        ReductionConfig(quadrature_points=k)
    manifest = manifest_at(atlas)
    selected = select_anchors(manifest, historical_seeds)
    entries = [_input(atlas, r, cell, 'historical') for (cell, _), r in sorted(selected.items())]
    horizons = {(r['days'], r['burn_in']) for r in selected.values()}
    if len(horizons) != 1:
        raise ValueError('Anchor horizons disagree')
    days, burn_in = horizons.pop()
    cells = []
    for cell in sorted({c for c, _ in selected}):
        configurations = {canonical(r['scenario']) for (c, _), r in selected.items() if c == cell}
        if len(configurations) != 1:
            raise ValueError('Anchor configurations differ across seeds')
        scenario = json.loads(configurations.pop())
        if any(scenario[k] for k in ('evolution', 'circuit_breaker_asset', 'halt_check')) or scenario['chartist_window'] != 20:
            raise ValueError('Task 4 requires window 20 without evolution or breaker controls')
        cells.append({'name': cell, 'scenario': 'baseline', 'seeds': list(fresh_seeds),
                      'overrides': {k: v for k, v in scenario.items() if k not in ('name', 'title')}})
    fresh_spec = {'schema_version': 1, 'name': 'task4-fresh-controls', 'days': days,
                  'burn_in': burn_in, 'cells': cells}
    study = {'schema_version': SCHEMA, 'kind': 'hard-reduction-study', 'created': now(),
             'execution': fingerprint(), 'provenance': provenance()['git'],
             'historical_manifest': str(atlas / 'manifest.json'), 'historical_manifest_sha256': digest(manifest),
             'historical_inputs': entries, 'fresh_spec': fresh_spec, 'quadrature_points': list(points),
             'days': days, 'burn_in': burn_in}
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    if disk_guard(parent, Limits(), reserved=2*MIB) == 'free_disk_limit':
        raise ValueError('Insufficient free disk')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'manifest.json', study)
    write_json(root / 'manifest.sha256.json', {'sha256': digest(study)})
    write_json(root / 'fresh_spec.json', fresh_spec)
    return study


def study_inputs(root, study):
    fresh = manifest_at(Path(root) / 'fresh_abm')
    if fresh['specification'] != study['fresh_spec']:
        raise ValueError('Fresh control specification differs from the study')
    inputs = list(study['historical_inputs'])
    for run in fresh['runs']:
        if len(run['cells']) != 1:
            raise ValueError('Unexpected fresh control cell mapping')
        inputs.append(_input(Path(root) / 'fresh_abm', run, run['cells'][0], 'fresh'))
    for entry in inputs:
        validate_input(entry)
    expected = len(study['historical_inputs']) + 10 * len(study['fresh_spec']['cells'][0]['seeds'])
    if len(inputs) != expected:
        raise ValueError('Missing comparison inputs')
    return inputs


def run_identity(run, execution):
    return digest({'run': {k: v for k, v in run.items() if k != 'run_id'}, 'execution': execution})


def make_reduction_manifest(root, study):
    inputs = study_inputs(root, study)
    memories, runs = {}, []
    # Retain all input identities, but create models only for constant/zero-noise controls.
    for entry in inputs:
        key = entry['cohort'], entry['run']['seed']
        memory = warmup_memory(entry)
        if key in memories and memory != memories[key]:
            raise ValueError('Controls do not share the expected warm-up history')
        memories[key] = memory
        if entry['cell'].split('-')[1] not in ('v6', 'v7'):
            continue
        s = entry['run']['scenario']
        if s['closing_noise_scale'] != 0 or s['dividend_mode'] != 'constant':
            raise ValueError('Stored primary control is not zero-noise/constant-dividend')
        for k in study['quadrature_points']:
            config = ReductionConfig((s['noise'], s['chartists'], s['fundamentalists']),
                                     s['chartist_alpha'], s['fundamentalist_gamma'], s['chartist_window'], k,
                                     s['order_lifetime'])
            run = {'config': asdict(config), 'initial_state': state_dict(initial_state(memory, config)),
                   'initial_condition_seed': entry['run']['seed'], 'cohort': entry['cohort'],
                   'coordinate': entry['cell'].split('-')[0], 'source': entry,
                   'days': study['days'], 'burn_in': study['burn_in']}
            run['run_id'] = run_identity(run, study['execution'])
            runs.append(run)
    # Benchmark the selected persistent liquidity contrast at the primary resolution.
    runs.sort(key=lambda r: (r['config']['quadrature_points'] != study['quadrature_points'][0],
                             r['config']['order_lifetime'] != 'persistent', r['coordinate'] != 'liquidity',
                             r['initial_condition_seed'], r['config']['quadrature_points']))
    result = {'schema_version': SCHEMA, 'study_sha256': digest(study), 'execution': study['execution'],
              'inputs': inputs, 'runs': runs}
    # Normalize tuples once: stored and in-memory scientific identities must agree.
    return json.loads(canonical(result))


def comparison_summary(arrays, burn_in):
    result = summarize(arrays, burn_in)
    v, r = arrays['volume'][burn_in:], arrays['log_return'][burn_in:]
    positive = r[v > 0]
    result['values'].update(mean_volume=float(np.mean(v)),
                            positive_volume_volatility=float(np.std(positive, ddof=1)) if len(positive) > 1 else None,
                            cumulative_log_return=float(np.sum(r)),
                            terminal_stall=bool(result['values']['terminal_no_trade_streak'] >= 100),
                            subshare_volume_fraction=float(np.mean((v > 0) & (v < 1))),
                            max_outstanding_orders=int(np.max(arrays['outstanding_buy_orders'][burn_in:]
                                                              + arrays['outstanding_sell_orders'][burn_in:])))
    if len(positive) <= 1:
        result['undefined']['positive_volume_volatility'] = 'requires two positive-volume days'
    result['final_cash'] = arrays['total_cash'][-1].tolist()
    result['final_shares'] = arrays['total_assets'][-1].tolist()
    if 'within_strategy_volume' in arrays:
        total = float(np.sum(v))
        result['values']['within_strategy_volume_fraction'] = float(np.sum(arrays['within_strategy_volume'][burn_in:]) / total) if total else None
    return result


def save_reduction(directory, result, run, execution):
    directory = Path(directory)
    directory.mkdir()
    with (directory / 'daily.npz').open('xb') as stream:
        np.savez_compressed(stream, **result.arrays)
    with gzip.open(directory / 'final_state.json.gz', 'wt') as stream:
        stream.write(canonical(state_dict(result.final_state)))
    write_json(directory / 'summary.json', comparison_summary(result.arrays, run['burn_in']))
    write_json(directory / 'metadata.json', {
        'schema_version': SCHEMA, 'model': 'pooled-hard-rule', 'run': run, 'execution': execution,
        'strategy_columns': STRATEGIES, 'quote_grid': quote_grid(result.config.quadrature_points),
        'forcing': {'dividend': DIVIDEND, 'closing_noise_scale': 0, 'evolution': False, 'circuit_breaker': False},
        'closure': 'per-agent mean portfolios; weighted conditional quotes; pooled sequential matching; fractional flows',
        'units': {'prices': 'currency/share', 'resources': 'currency and shares', 'volume': 'shares',
                  'returns': 'natural log ratio per calendar trading day', 'submitted_mass': 'expected agents'},
        'memory_clock': 'accepted closing prices, including unchanged prices after positive volume',
        'book_measurement': 'after matching, before expiry', 'initial_state_in': 'run.initial_state',
        'day_indexing': 'zero-based trading days, excluding the recorded warm-up',
        'summary_conventions': {'window': '[burn_in, days)', 'volatility_ddof': 1,
                                'stall': 'at least 100 terminal exact zero-volume days',
                                'subshare': '0 < volume < 1, no rounding or epsilon cutoff'},
    })
    write_json(directory / 'complete.json', {
        'schema_version': SCHEMA, 'run_id': run['run_id'],
        'files': {p: {'sha256': file_hash(directory / p), 'bytes': (directory / p).stat().st_size} for p in sorted(FILES)}})


def read_reduction(directory, expected_run=None, execution=None):
    directory = Path(directory)
    complete = read_json(directory / 'complete.json')
    if complete.get('schema_version') != SCHEMA or set(complete.get('files', {})) != FILES:
        raise ValueError('Invalid reduction completion manifest')
    if {p.name for p in directory.iterdir()} != FILES | {'complete.json'}:
        raise ValueError('Unmanifested reduction files')
    for name, record in complete['files'].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError(f'Corrupt reduction file: {name}')
    meta = read_json(directory / 'metadata.json')
    run = meta['run']
    if (meta['schema_version'] != SCHEMA or run_identity(run, meta['execution']) != run['run_id']
            or complete['run_id'] != run['run_id'] or (expected_run is not None and run != expected_run)
            or (execution is not None and meta['execution'] != execution)):
        raise ValueError('Reduction provenance mismatch')
    config = ReductionConfig(**run['config'])
    first = state_from_dict(run['initial_state'])
    with gzip.open(directory / 'final_state.json.gz', 'rt') as stream:
        last = state_from_dict(json.load(stream))
    validate_state(first, config)
    validate_state(last, config)
    with np.load(directory / 'daily.npz', allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_FIELDS:
            raise ValueError('Invalid reduction array fields')
        arrays = {k: archive[k] for k in archive.files}
    for name, value in arrays.items():
        dtype = np.dtype('bool' if name == 'accepted_price_update' else 'int64' if name in INT_FIELDS else 'float64')
        shape = (run['days'], 3) if name in GROUP_FIELDS else (run['days'],)
        if value.shape != shape or value.dtype != dtype or not np.all(np.isfinite(value)):
            raise ValueError(f'Invalid reduction array: {name}')
    if (not np.array_equal(arrays['day'], np.arange(run['days']))
            or arrays['previous_price'][0] != first.price
            or not np.array_equal(arrays['previous_price'][1:], arrays['price'][:-1])
            or last.price != arrays['price'][-1]
            or not np.array_equal(last.cash, arrays['total_cash'][-1])
            or not np.array_equal(last.shares, arrays['total_assets'][-1])
            or not np.array_equal(arrays['accepted_price_update'], arrays['volume'] > 0)):
        raise ValueError('Invalid reduction state/chronology')
    for name, total in (('total_cash', sum(first.cash)), ('total_assets', sum(first.shares))):
        if np.any(arrays[name] < 0) or not np.allclose(arrays[name].sum(axis=1), total, rtol=1e-11, atol=1e-8):
            raise ValueError('Invalid reduction resource accounting')
    summary = read_json(directory / 'summary.json')
    if summary != comparison_summary(arrays, run['burn_in']):
        raise ValueError('Reduction summary disagrees with daily arrays')
    return meta, arrays, summary


def worker(request_path):
    request_path = Path(request_path)
    request = read_json(request_path)
    started = time.monotonic()
    try:
        if fingerprint() != request['execution']:
            raise ValueError('Reduction execution sources/runtime changed')
        run = request['run']
        validate_input(run['source'])
        result = simulate_reduction(state_from_dict(run['initial_state']), ReductionConfig(**run['config']), days=run['days'])
        save_reduction(request_path.parent / 'artifact', result, run, request['execution'])
        read_reduction(request_path.parent / 'artifact', run, request['execution'])
        if fingerprint() != request['execution']:
            raise ValueError('Reduction execution changed during run')
        _fsync_tree(request_path.parent / 'artifact')
        record = {'state': 'complete', 'seconds': time.monotonic() - started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'artifact_bytes': directory_bytes(request_path.parent / 'artifact')}
        write_json(request_path.parent / 'result.json', record)
        return 0
    except Exception as error:
        record = {'state': 'failed', 'error': f'{type(error).__name__}: {error}'}
        print(record['error'], flush=True)
        try:
            write_json(request_path.parent / 'result.json', record)
        except (OSError, MemoryError):
            pass
        return 1


def _fresh_controls(root, study, limits, deadline, retry_failed=False):
    directory = root / 'fresh_abm'
    if read_json(root / 'fresh_spec.json') != study['fresh_spec']:
        raise ValueError('Fresh specification was modified')
    if directory.exists() and status(directory)['runs'] == {'complete': 10 * len(study['fresh_spec']['cells'][0]['seeds'])}:
        if manifest_at(directory)['provenance']['execution'] != execution_fingerprint():
            raise ValueError('Fresh ABM runtime/source differs')
        return 0
    outside = directory_bytes(root) - directory_bytes(directory)
    output = min(4.5, limits.output_gib - outside / GIB - .25)
    if output <= .125:
        return 2
    minutes = max(.0001, (deadline - time.monotonic()) / 60)
    command = [sys.executable, str(ROOT / 'experiments/run_experiment.py')]
    command += ['resume', str(directory)] if directory.exists() else ['run', str(root / 'fresh_spec.json'), str(directory)]
    command += ['--workers', str(limits.workers), '--minutes', str(minutes), '--memory-gib', str(limits.memory_gib - .5),
                '--output-gib', str(output), '--min-free-gib', str(limits.min_free_gib)]
    if retry_failed and directory.exists():
        command.append('--retry-failed')
    with (root / 'fresh_abm.log').open('ab') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    old, stopped = {}, []
    def stop(signum, frame):
        stopped.append(signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, stop)
    try:
        while process.poll() is None:
            if stopped or time.monotonic() >= deadline or disk_guard(root, limits):
                _terminate([process])
                return 2
            time.sleep(.2)
        return process.returncode
    finally:
        _terminate([process])
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _recover(directory, db, manifest, retry_failed):
    expected = {r['run_id']: r for r in manifest['runs']}
    rows = list(db.execute('SELECT * FROM runs'))
    if any(r['run_id'] not in expected or json.loads(r['spec']) != expected[r['run_id']] for r in rows):
        raise ValueError('Reduction ledger and manifest disagree')
    with db:
        for run in manifest['runs']:
            db.execute('INSERT OR IGNORE INTO runs(run_id,spec,state) VALUES(?,?,?)', (run['run_id'], canonical(run), 'pending'))
    for row in list(db.execute('SELECT * FROM runs')):
        run_id, state, error = row['run_id'], row['state'], row['error']
        target = directory / 'runs' / run_id
        if target.exists():
            try:
                read_reduction(target, expected[run_id], manifest['execution'])
                state, error = 'complete', None
            except ARTIFACT_ERRORS as problem:
                state, error = 'failed', f'Corrupt artifact: {problem}'
        elif state == 'complete':
            state, error = 'failed', 'Completed artifact is missing'
        elif state == 'running':
            state, error = 'interrupted', 'Coordinator exited'
        _remove_stage(directory / 'staging' / run_id)
        if state == 'failed' and retry_failed:
            _remove_stage(target)
            state, error = 'pending', None
        with db:
            db.execute('UPDATE runs SET state=?, error=? WHERE run_id=?', (state, error, run_id))
            db.execute("UPDATE attempts SET state='interrupted', ended=?, error='Coordinator exited' WHERE run_id=? AND state='running'", (now(), run_id))
    with db:
        db.execute("UPDATE invocations SET ended=?, stop_reason='coordinator_exited' WHERE ended IS NULL", (now(),))


def _supervise(root, directory, db, manifest, limits, deadline):
    invocation = uuid4().hex
    with db:
        db.execute('INSERT INTO invocations(id,started,limits) VALUES(?,?,?)', (invocation, now(), canonical(asdict(limits))))
    pending = [json.loads(row[0]) for row in db.execute("SELECT spec FROM runs WHERE state IN ('pending','interrupted') ORDER BY rowid")]
    active, stopped, old = {}, [], {}
    benchmark = False
    reason = 'complete'
    def stop(signum, frame):
        stopped.append(f'signal_{signum}')
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, stop)
    def emit(event, **fields):
        line = canonical({'time': now(), 'event': event, **fields})
        print(line, flush=True)
        with (directory / 'invocations' / f'{invocation}.jsonl').open('a') as stream:
            stream.write(line + '\n')
    try:
        emit('started', pending=len(pending), limits=asdict(limits))
        while pending or active:
            if stopped or time.monotonic() >= deadline:
                reason = stopped[0] if stopped else 'time_limit'
                break
            failure = disk_guard(root, limits, reserved=len(active) * 64*MIB)
            if failure or memory_available() < GIB:
                reason = failure or 'system_memory_pressure'
                break
            if resident_bytes(os.getpid()) + sum(resident_bytes(e['process'].pid) for e in active.values()) > limits.memory_gib * GIB:
                reason = 'memory_limit'
                break
            for run_id, entry in list(active.items()):
                process, stage, run = entry['process'], entry['stage'], entry['run']
                if process.poll() is None:
                    continue
                record = {}
                try:
                    record = read_json(stage / 'result.json')
                except (ValueError, OSError):
                    pass
                state, error = 'failed', record.get('error', f'Worker exited {process.returncode}; see staging worker.log')
                if process.returncode == 0 and record.get('state') == 'complete':
                    try:
                        read_reduction(stage / 'artifact', run, manifest['execution'])
                        if fingerprint() != manifest['execution']:
                            raise ValueError('Execution changed before publication')
                        os.rename(stage / 'artifact', directory / 'runs' / run_id)
                        _fsync_directory(directory / 'runs')
                        state, error = 'complete', None
                    except ARTIFACT_ERRORS as problem:
                        error = str(problem)
                seconds = time.monotonic() - entry['started']
                with db:
                    db.execute('UPDATE runs SET state=?,error=? WHERE run_id=?', (state, error, run_id))
                    db.execute('UPDATE attempts SET state=?,ended=?,seconds=?,peak_rss_bytes=?,artifact_bytes=?,error=? WHERE id=?',
                               (state, now(), seconds, record.get('peak_rss_bytes'), record.get('artifact_bytes'), error, entry['attempt']))
                emit('finished', run_id=run_id, state=state, seconds=seconds, error=error,
                     peak_rss_bytes=record.get('peak_rss_bytes'), artifact_bytes=record.get('artifact_bytes'))
                del active[run_id]
                if state == 'complete':
                    _remove_stage(stage)
                else:
                    stopped.append('worker_failed')
                if not benchmark and state == 'complete':
                    benchmark = True
                    with db:
                        db.execute('UPDATE invocations SET benchmark_run=? WHERE id=?', (run_id, invocation))
                    emit('benchmark_complete', run_id=run_id, seconds=seconds)
            capacity = limits.workers if benchmark else 1
            while pending and not stopped and len(active) < capacity and time.monotonic() < deadline:
                if fingerprint() != manifest['execution']:
                    stopped.append('provenance_changed')
                    break
                failure = disk_guard(root, limits, reserved=(len(active) + 1)*64*MIB)
                if failure:
                    stopped.append(failure)
                    break
                run = pending.pop(0)
                run_id = run['run_id']
                stage = directory / 'staging' / run_id
                stage.mkdir()
                write_json(stage / 'request.json', {'run': run, 'execution': manifest['execution']})
                memory = (int(limits.memory_gib * GIB) - 768*MIB) // limits.workers
                with db:
                    attempt = db.execute('INSERT INTO attempts(run_id,invocation,state,started,memory_limit,file_limit) VALUES(?,?,?,?,?,?)',
                                         (run_id, invocation, 'running', now(), memory, 8*MIB)).lastrowid
                    db.execute("UPDATE runs SET state='running',attempts=attempts+1,error=NULL WHERE run_id=?", (run_id,))
                with (stage / 'worker.log').open('wb') as stream:
                    process = subprocess.Popen([sys.executable, str(CLI), '_worker', str(stage / 'request.json'),
                                                '--memory-bytes', str(memory), '--parent-pid', str(os.getpid())],
                                               stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
                with db:
                    db.execute('UPDATE attempts SET pid=? WHERE id=?', (process.pid, attempt))
                active[run_id] = {'run': run, 'stage': stage, 'process': process, 'attempt': attempt, 'started': time.monotonic()}
                emit('launched', run_id=run_id, pid=process.pid, benchmark=not benchmark)
            if active:
                time.sleep(.1)
        if stopped:
            reason = stopped[0]
        if reason == 'complete' and db.execute("SELECT COUNT(*) FROM runs WHERE state!='complete'").fetchone()[0]:
            reason = 'failed_runs'
    except BaseException:
        reason = 'supervisor_error'
        raise
    finally:
        _terminate([e['process'] for e in active.values()])
        with db:
            for run_id, entry in active.items():
                db.execute("UPDATE runs SET state='interrupted',error=? WHERE run_id=?", (reason, run_id))
                db.execute("UPDATE attempts SET state='interrupted',ended=?,error=? WHERE id=?", (now(), reason, entry['attempt']))
            db.execute('UPDATE invocations SET ended=?,stop_reason=? WHERE id=?', (now(), reason, invocation))
        for sig, handler in old.items():
            signal.signal(sig, handler)
        emit('stopped', reason=reason, **status(directory))
    return 0 if reason == 'complete' else 2


def execute_study(root, *, limits=None, retry_failed=False):
    """CLI entry after process bootstrap; one bounded, resumable pilot batch."""
    root = Path(root).resolve()
    limits = limits or Limits()
    if limits.memory_gib <= 1.5:
        raise ValueError('Study needs more than 1.5 GiB including its driver')
    deadline = time.monotonic() + limits.minutes * 60
    with experiment_lock(root):
        study = manifest_at(root)
        if study['kind'] != 'hard-reduction-study' or study['schema_version'] != SCHEMA:
            raise ValueError('Unsupported study manifest')
        if study['execution'] != fingerprint():
            raise ValueError('Study execution sources/runtime changed; restore recorded source or create a new study')
        if disk_guard(root, limits):
            raise ValueError('Study disk/output limit reached')
        for entry in study['historical_inputs']:
            validate_input(entry)
        code = _fresh_controls(root, study, limits, deadline, retry_failed)
        if code:
            return code
        manifest = make_reduction_manifest(root, study)
        directory = root / 'reduction'
        directory.mkdir(exist_ok=True)
        for name in ('runs', 'staging', 'invocations'):
            (directory / name).mkdir(exist_ok=True)
        if (directory / 'manifest.json').exists():
            if manifest_at(directory) != manifest:
                raise ValueError('Reduction manifest/inputs changed')
        else:
            write_json(directory / 'manifest.json', manifest)
            write_json(directory / 'manifest.sha256.json', {'sha256': digest(manifest)})
        db = _connect(directory)
        try:
            _recover(directory, db, manifest, retry_failed)
            return _supervise(root, directory, db, manifest, limits, deadline)
        finally:
            db.close()


def study_status(root):
    root = Path(root)
    study = manifest_at(root)
    return {'study': str(root), 'historical_runs': len(study['historical_inputs']),
            'fresh_abm': status(root / 'fresh_abm') if (root / 'fresh_abm/ledger.sqlite').exists() else None,
            'reduction': status(root / 'reduction') if (root / 'reduction/ledger.sqlite').exists() else None}
