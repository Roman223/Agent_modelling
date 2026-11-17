"""Bounded, resumable checkpoint-compression experiments and strict artifacts."""

from dataclasses import asdict
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

from src.atlas_runner import manifest_at
from src.experiment_artifacts import (ROOT, canonical, digest, execution_fingerprint, file_hash,
                                      provenance, read_json, write_json, ARTIFACT_ERRORS)
from src.experiments import (Limits, GIB, MIB, _connect, _remove_stage, _terminate, _fsync_tree,
                             _fsync_directory, directory_bytes, disk_guard, experiment_lock,
                             memory_available, resident_bytes, status, now)
from src.market_checkpoints import (run_checkpointed, continue_checkpoint, read_checkpoint,
                                    write_checkpoint, dynamics_digest, validate_scenario)
from src.heterogeneity import CompressionConfig, compress_checkpoint, FEATURE_NAMES, LAGS
from src.scenarios import Scenario

CLI = ROOT / 'experiments/run_heterogeneity.py'
LEVELS = (1, 2, 4, 8, 16, 'full')
PRIMARY_FIELDS = ('price', 'previous_price', 'volume', 'within_strategy_volume', 'same_owner_volume',
                  'log_return', 'total_cash', 'total_assets', 'day', 'no_trade_streak')
ARRAY_FIELDS = set(PRIMARY_FIELDS) | {'feature_days', 'features'}


def fingerprint():
    result = execution_fingerprint()
    result['source_sha256'][str(CLI.relative_to(ROOT))] = file_hash(CLI)
    return result


def validate_seeds(values):
    if not values or len(set(values)) != len(values) or any(type(s) is not int or not 0 <= s < 2**32 for s in values):
        raise ValueError('Seed cohorts must be nonempty, unique uint32 integers')


def select_configurations(atlas):
    atlas = Path(atlas).resolve()
    if not (atlas / 'manifest.json').exists():
        atlas = atlas / 'simulation'
    manifest = manifest_at(atlas)
    wanted = {('4', '3', 'v0'): 'reference-v0', ('4', '3', 'v6'): 'reference-v6',
              ('4', '3', 'v7'): 'reference-v7', ('6', '0', 'v6'): 'liquidity-v6',
              ('6', '0', 'v7'): 'liquidity-v7'}
    selected = {}
    for run in manifest['runs']:
        for cell in run['cells']:
            parts = cell.split('-')
            if len(parts) == 5 and parts[0] == 'grid' and tuple(parts[1:4]) in wanted:
                label = wanted[tuple(parts[1:4])]
                value = run['scenario']
                validate_scenario(Scenario(**value))
                if label in selected and selected[label] != value:
                    raise ValueError('Atlas configurations disagree')
                selected[label] = value
    if set(selected) != set(wanted.values()):
        raise ValueError('Missing Task 5 atlas anchors')
    return selected, {'path': str(atlas / 'manifest.json'), 'sha256': digest(manifest)}


def generate_study(atlas, root, *, development=tuple(range(300, 310)), validation=tuple(range(310, 360)),
                   days=2000, checkpoints=(100, 500, 1000), continuation_days=1000,
                   levels=LEVELS, horizons=(1, 20, 100, 1000)):
    validate_seeds(development)
    validate_seeds(validation)
    if set(development) & set(validation):
        raise ValueError('Development and validation seeds must be disjoint')
    if type(days) is not int or type(continuation_days) is not int or continuation_days < 1:
        raise ValueError('Invalid duration')
    if (not checkpoints or len(checkpoints) > 8 or len(set(checkpoints)) != len(checkpoints)
            or any(type(d) is not int or d < 1 or d + continuation_days > days for d in checkpoints)):
        raise ValueError('Checkpoint continuations must fit within the reference run')
    if not horizons or any(type(h) is not int or not 1 <= h <= continuation_days for h in horizons):
        raise ValueError('Invalid observation horizons')
    if not levels or len(levels) > 8 or len(set(levels)) != len(levels) or 'full' not in levels:
        raise ValueError('Unique levels must include full')
    for level in levels:
        CompressionConfig(level)
    configs, source = select_configurations(atlas)
    from src.heterogeneity_analysis import SETTINGS
    study = {'schema_version': 1, 'kind': 'heterogeneity-study', 'created': now(),
             'execution': fingerprint(), 'git': provenance()['git'], 'atlas': source,
             'configurations': configs, 'development_seeds': list(development), 'validation_seeds': list(validation),
             'days': days, 'checkpoints': list(checkpoints), 'continuation_days': continuation_days,
             'levels': list(levels), 'horizons': sorted(set(horizons)), 'analysis_settings': SETTINGS,
             'features': FEATURE_NAMES, 'lags': list(LAGS)}
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


def reference_jobs(study):
    jobs = []
    for cell, scenario in sorted(study['configurations'].items()):
        for cohort, seeds in (('development', study['development_seeds']), ('validation', study['validation_seeds'])):
            for seed in seeds:
                job = {'phase': 'reference', 'cell': cell, 'scenario': scenario, 'cohort': cohort, 'seed': seed,
                       'days': study['days'], 'checkpoints': study['checkpoints'],
                       'check_days': sorted({d + study['continuation_days'] for d in study['checkpoints']})}
                job['run_id'] = digest({'job': job, 'execution': study['execution']})
                jobs.append(job)
    return jobs


def continuation_jobs(root, study, references):
    jobs = []
    for ref in references:
        path = root / 'reference/runs' / ref['run_id']
        validate_artifact(path, ref, study['execution'])
        for day in study['checkpoints']:
            job = {'phase': 'continuation', 'cell': ref['cell'], 'cohort': ref['cohort'], 'seed': ref['seed'],
                   'checkpoint_day': day, 'days': study['continuation_days'], 'levels': study['levels'],
                   'horizons': study['horizons'], 'source': str(path),
                   'source_sha256': file_hash(path / 'complete.json')}
            job['run_id'] = digest({'job': job, 'execution': study['execution']})
            jobs.append(job)
    return jobs


def publish_manifest(path, job, execution):
    write_json(path / 'metadata.json', {'schema_version': 1, 'job': job, 'execution': execution})
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}
    write_json(path / 'complete.json', {'schema_version': 1, 'run_id': job['run_id'], 'files': files})


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != ARRAY_FIELDS:
            raise ValueError('Invalid research array layout')
        arrays = {key: z[key] for key in z.files}
    n = len(arrays['day'])
    for key, values in arrays.items():
        shape = ((n, 3) if key in ('total_cash', 'total_assets') else
                 (len(arrays['feature_days']), len(FEATURE_NAMES)*len(LAGS)) if key == 'features' else
                 (len(arrays['feature_days']),) if key == 'feature_days' else (n,))
        if values.shape != shape or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError(f'Invalid array {key}')
    if (not n or np.any(np.diff(arrays['day']) != 1) or np.any(arrays['price'] <= 0)
            or np.any(arrays['previous_price'] <= 0) or np.any(arrays['volume'] < 0)
            or np.any(arrays['total_cash'] < 0) or np.any(arrays['total_assets'] < 0)
            or np.any(arrays['total_assets'] != np.floor(arrays['total_assets']))):
        raise ValueError('Invalid research state chronology/resources')
    for name in ('total_cash', 'total_assets'):
        sums = arrays[name].sum(axis=1)
        if not np.allclose(sums, sums[0], rtol=1e-11, atol=1e-8):
            raise ValueError('Resource totals changed')
    if not np.array_equal(arrays['previous_price'][1:], arrays['price'][:-1]):
        raise ValueError('Broken price chronology')
    return arrays


def validate_artifact(path, job=None, execution=None, *, deep=False):
    path = Path(path)
    complete, meta = read_json(path / 'complete.json'), read_json(path / 'metadata.json')
    if complete.get('schema_version') != 1 or meta.get('schema_version') != 1:
        raise ValueError('Invalid artifact schema')
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()} - {'complete.json'}
    if actual != set(complete['files']) or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('Unmanifested artifact contents')
    for name, checksum in complete['files'].items():
        if file_hash(path / name) != checksum:
            raise ValueError(f'Corrupt artifact: {name}')
    if complete['run_id'] != meta['job']['run_id'] or (job is not None and meta['job'] != job):
        raise ValueError('Artifact identity mismatch')
    if execution is not None and meta['execution'] != execution:
        raise ValueError('Artifact provenance mismatch')
    if deep:
        for name in complete['files']:
            if name.endswith('.npz'):
                read_arrays(path / name)
            elif name.startswith('checkpoint_'):
                read_checkpoint(path / name)
    return meta


def materialize(job, destination, execution):
    from src.heterogeneity_analysis import outcome_summary
    destination.mkdir()
    if job['phase'] == 'reference':
        result = run_checkpointed(Scenario(**job['scenario']), seed=job['seed'], days=job['days'],
                                 checkpoint_days=job['checkpoints'], check_days=job['check_days'])
        np.savez_compressed(destination / 'daily.npz', **result.arrays)
        for day, checkpoint in result.checkpoints.items():
            write_checkpoint(destination / f'checkpoint_{day}.json.gz', checkpoint)
        write_json(destination / 'checks.json', {str(k): v for k, v in result.checks.items()})
    else:
        source = Path(job['source'])
        validate_artifact(source, execution=execution)
        if file_hash(source / 'complete.json') != job['source_sha256']:
            raise ValueError('Reference artifact changed')
        checkpoint = read_checkpoint(source / f"checkpoint_{job['checkpoint_day']}.json.gz")
        reference = read_arrays(source / 'daily.npz')
        checks = read_json(source / 'checks.json')
        unique, levels = {}, {}
        # Full endpoint first: verify the restoration before accepting interventions.
        for level in sorted(job['levels'], key=lambda x: (x != 'full', str(x))):
            state, compression = compress_checkpoint(checkpoint, CompressionConfig(level))
            state_id = state.sha256
            if state_id not in unique:
                result = continue_checkpoint(state, days=job['days'])
                if level == 'full':
                    offset = job['checkpoint_day']
                    for field in PRIMARY_FIELDS:
                        if not np.array_equal(result.arrays[field], reference[field][offset:offset + job['days']]):
                            raise AssertionError(f'Full-detail daily mismatch: {field}')
                    if dynamics_digest(result.final) != checks[str(offset + job['days'])]:
                        raise AssertionError('Full-detail final-state/RNG mismatch')
                name = f'state_{state_id}.npz'
                np.savez_compressed(destination / name, **result.arrays)
                unique[state_id] = {'file': name, 'summaries': {str(h): outcome_summary(result.arrays, h)
                                                              for h in job['horizons']},
                                    'final_dynamics_sha256': dynamics_digest(result.final)}
            levels[str(level)] = {'state': state_id, 'compression': compression}
        write_json(destination / 'levels.json', {'levels': levels, 'states': unique,
                   'checkpoint_stalled': checkpoint.data()['no_trade_streak'] >= 100, 'full_detail_verified': True})
    publish_manifest(destination, job, execution)
    validate_artifact(destination, job, execution, deep=True)


def worker(request_path):
    request_path = Path(request_path)
    request = read_json(request_path)
    start = time.monotonic()
    try:
        if fingerprint() != request['execution']:
            raise ValueError('Execution sources/runtime changed')
        materialize(request['job'], request_path.parent / 'artifact', request['execution'])
        if fingerprint() != request['execution']:
            raise ValueError('Execution changed during worker')
        _fsync_tree(request_path.parent / 'artifact')
        record = {'state': 'complete', 'seconds': time.monotonic() - start,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'artifact_bytes': directory_bytes(request_path.parent / 'artifact')}
        code = 0
    except Exception as error:
        record = {'state': 'failed', 'error': f'{type(error).__name__}: {error}'}
        import traceback
        traceback.print_exc()
        code = 1
    write_json(request_path.parent / 'result.json', record)
    return code


def _phase(root, name, jobs, study, limits, deadline, retry_failed):
    directory = root / name
    directory.mkdir(exist_ok=True)
    for sub in ('runs', 'staging', 'invocations'):
        (directory / sub).mkdir(exist_ok=True)
    manifest = {'schema_version': 1, 'study_sha256': digest(study), 'jobs': jobs}
    if (directory / 'manifest.json').exists():
        if manifest_at(directory) != manifest:
            raise ValueError('Phase manifest changed')
    else:
        write_json(directory / 'manifest.json', manifest)
        write_json(directory / 'manifest.sha256.json', {'sha256': digest(manifest)})
    db = _connect(directory)
    try:
        expected = {j['run_id']: j for j in jobs}
        for row in db.execute('SELECT * FROM runs'):
            if row['run_id'] not in expected or json.loads(row['spec']) != expected[row['run_id']]:
                raise ValueError('Ledger differs from immutable job manifest')
        with db:
            for job in jobs:
                db.execute('INSERT OR IGNORE INTO runs(run_id,spec,state) VALUES(?,?,?)', (job['run_id'], canonical(job), 'pending'))
        for row in list(db.execute('SELECT * FROM runs')):
            run_id, state, error = row['run_id'], row['state'], row['error']
            path = directory / 'runs' / run_id
            if path.exists():
                try:
                    validate_artifact(path, expected[run_id], study['execution'])
                    state, error = 'complete', None
                except ARTIFACT_ERRORS as problem:
                    state, error = 'failed', str(problem)
            elif state == 'complete':
                state, error = 'failed', 'Published artifact missing'
            elif state == 'running':
                state, error = 'interrupted', 'Coordinator exited'
            _remove_stage(directory / 'staging' / run_id)
            if state == 'failed' and retry_failed:
                _remove_stage(path)
                state, error = 'pending', None
            with db:
                db.execute('UPDATE runs SET state=?,error=? WHERE run_id=?', (state, error, run_id))
                db.execute("UPDATE attempts SET state='interrupted',ended=? WHERE run_id=? AND state='running'", (now(), run_id))
        with db:
            db.execute("UPDATE invocations SET ended=?,stop_reason='coordinator_exited' WHERE ended IS NULL", (now(),))
        return _supervise(root, directory, db, study, limits, deadline)
    finally:
        db.close()


def _supervise(root, directory, db, study, limits, deadline):
    invocation, active, stopped, handlers = uuid4().hex, {}, [], {}
    pending = [json.loads(r[0]) for r in db.execute("SELECT spec FROM runs WHERE state IN ('pending','interrupted') ORDER BY rowid")]
    with db:
        db.execute('INSERT INTO invocations(id,started,limits) VALUES(?,?,?)', (invocation, now(), canonical(asdict(limits))))
    benchmark, reason = False, 'complete'
    def emit(event, **fields):
        line = canonical({'time': now(), 'phase': directory.name, 'event': event, **fields})
        print(line, flush=True)
        with (directory / 'invocations' / f'{invocation}.jsonl').open('a') as stream:
            stream.write(line + '\n')
    def stop(sig, frame):
        stopped.append(f'signal_{sig}')
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, stop)
    try:
        emit('started', pending=len(pending), limits=asdict(limits))
        while pending or active:
            reason_now = disk_guard(root, limits, reserved=len(active)*512*MIB)
            if stopped or time.monotonic() >= deadline or reason_now or memory_available() < GIB:
                reason = stopped[0] if stopped else reason_now or ('time_limit' if time.monotonic() >= deadline else 'memory_pressure')
                break
            if resident_bytes(os.getpid()) + sum(resident_bytes(e['process'].pid) for e in active.values()) > limits.memory_gib*GIB:
                reason = 'memory_limit'
                break
            for run_id, entry in list(active.items()):
                process, stage, job = entry['process'], entry['stage'], entry['job']
                if process.poll() is None:
                    continue
                try:
                    record = read_json(stage / 'result.json')
                except (ValueError, OSError):
                    record = {}
                state, error = 'failed', record.get('error', f'Worker exit {process.returncode}; inspect staging worker.log')
                if process.returncode == 0 and record.get('state') == 'complete':
                    try:
                        validate_artifact(stage / 'artifact', job, study['execution'])
                        if fingerprint() != study['execution']:
                            raise ValueError('Execution changed before publication')
                        os.rename(stage / 'artifact', directory / 'runs' / run_id)
                        _fsync_directory(directory / 'runs')
                        state, error = 'complete', None
                    except ARTIFACT_ERRORS as problem:
                        error = str(problem)
                with db:
                    db.execute('UPDATE runs SET state=?,error=? WHERE run_id=?', (state, error, run_id))
                    db.execute('UPDATE attempts SET state=?,ended=?,seconds=?,peak_rss_bytes=?,artifact_bytes=?,error=? WHERE id=?',
                               (state, now(), time.monotonic()-entry['started'], record.get('peak_rss_bytes'),
                                record.get('artifact_bytes'), error, entry['attempt']))
                emit('finished', run_id=run_id, state=state, error=error, seconds=time.monotonic()-entry['started'],
                     peak_rss_bytes=record.get('peak_rss_bytes'), artifact_bytes=record.get('artifact_bytes'))
                del active[run_id]
                if state != 'complete':
                    stopped.append('worker_failed')
                else:
                    _remove_stage(stage)
                    if not benchmark:
                        benchmark = True
                        with db:
                            db.execute('UPDATE invocations SET benchmark_run=? WHERE id=?', (run_id, invocation))
                        emit('benchmark_complete', run_id=run_id)
            capacity = limits.workers if benchmark else 1
            while pending and not stopped and len(active) < capacity and time.monotonic() < deadline:
                if fingerprint() != study['execution']:
                    stopped.append('provenance_changed')
                    break
                failure = disk_guard(root, limits, reserved=(len(active)+1)*512*MIB)
                if failure:
                    stopped.append(failure)
                    break
                job = pending.pop(0)
                run_id = job['run_id']
                stage = directory / 'staging' / run_id
                stage.mkdir()
                write_json(stage / 'request.json', {'job': job, 'execution': study['execution']})
                with db:
                    attempt = db.execute('INSERT INTO attempts(run_id,invocation,state,started,memory_limit,file_limit) VALUES(?,?,?,?,?,?)',
                                         (run_id, invocation, 'running', now(), limits.worker_memory, 32*MIB)).lastrowid
                    db.execute("UPDATE runs SET state='running',attempts=attempts+1,error=NULL WHERE run_id=?", (run_id,))
                with (stage / 'worker.log').open('wb') as stream:
                    process = subprocess.Popen([sys.executable, str(CLI), '_worker', str(stage / 'request.json'),
                                                '--memory-bytes', str(limits.worker_memory), '--parent-pid', str(os.getpid())],
                                               stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
                with db:
                    db.execute('UPDATE attempts SET pid=? WHERE id=?', (process.pid, attempt))
                active[run_id] = {'process': process, 'job': job, 'stage': stage, 'attempt': attempt, 'started': time.monotonic()}
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
            for run_id, e in active.items():
                db.execute("UPDATE runs SET state='interrupted',error=? WHERE run_id=?", (reason, run_id))
                db.execute("UPDATE attempts SET state='interrupted',ended=?,error=? WHERE id=?", (now(), reason, e['attempt']))
            db.execute('UPDATE invocations SET ended=?,stop_reason=? WHERE id=?', (now(), reason, invocation))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        emit('stopped', reason=reason, **status(directory))
    return 0 if reason == 'complete' else 2


def execute_study(root, *, limits=None, retry_failed=False):
    root, limits = Path(root).resolve(), limits or Limits()
    deadline = time.monotonic() + limits.minutes*60
    with experiment_lock(root):
        study = manifest_at(root)
        if study.get('kind') != 'heterogeneity-study' or study.get('schema_version') != 1:
            raise ValueError('Unsupported study')
        if study['execution'] != fingerprint():
            raise ValueError('Execution sources/runtime changed; restore source or start a new study')
        if digest(manifest_at(Path(study['atlas']['path']).parent)) != study['atlas']['sha256']:
            raise ValueError('Source atlas manifest changed')
        refs = reference_jobs(study)
        code = _phase(root, 'reference', refs, study, limits, deadline, retry_failed)
        if code:
            return code
        jobs = continuation_jobs(root, study, refs)
        return _phase(root, 'continuation', jobs, study, limits, deadline, retry_failed)


def study_status(root):
    root = Path(root)
    study = manifest_at(root)
    return {'study': str(root), 'expected_references': len(reference_jobs(study)),
            'expected_checkpoint_jobs': len(reference_jobs(study))*len(study['checkpoints']),
            **{phase: status(root / phase) if (root / phase / 'ledger.sqlite').exists() else None
               for phase in ('reference', 'continuation')}}
