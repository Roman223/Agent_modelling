"""Bounded, resumable analysis of task-2 artifacts; no model execution here."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

from src.atlas import OPERATIONAL, SCHEMA, SETTINGS, aggregate, analyze_arrays
from src.experiment_artifacts import (ARTIFACT_ERRORS, ROOT, digest, execution_fingerprint,
                                      file_hash, read_compact, read_json, validate_compact, write_json)
from src.experiments import (GIB, MIB, Limits, _terminate, directory_bytes, experiment_lock,
                             memory_available, now, resident_bytes)

CLI = ROOT / 'experiments/run_atlas.py'
ANALYSIS_RESERVE = 256 * MIB


def analysis_fingerprint(settings=None):
    runtime = execution_fingerprint()
    paths = ('src/atlas.py', 'src/atlas_runner.py', 'src/experiment_artifacts.py',
             'src/experiments.py', 'experiments/run_experiment.py', 'experiments/run_atlas.py')
    return {'settings': SETTINGS if settings is None else settings,
            'python': runtime['python'], 'platform': runtime['platform'], 'packages': runtime['packages'],
            'source_sha256': {name: file_hash(ROOT / name) for name in paths}}


def manifest_at(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if digest(manifest) != read_json(directory / 'manifest.sha256.json')['sha256']:
        raise ValueError('Immutable experiment manifest was modified')
    return manifest


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        write_json(temporary, data)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def save_record(path, record):
    atomic_json(path, {'sha256': digest(record), 'record': record})


def cached_record(path, run, artifact, fingerprint, *, validate=True):
    """Corrupt cache is recomputed; corrupt scientific input is never reused."""
    if validate:
        validate_compact(artifact, expected_run=run)
    try:
        envelope = read_json(path)
        record = envelope['record']
        expected = {'run_id': run['run_id'], 'analysis_id': digest(fingerprint),
                    'artifact_sha256': file_hash(Path(artifact) / 'complete.json')}
        if envelope['sha256'] != digest(record) or any(record.get(k) != v for k, v in expected.items()):
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def has_failure(record):
    return (record['state'] != 'complete' or
            any(item['status'] in OPERATIONAL for item in record['result']['metrics'].values()))


def analyze_worker(request_path):
    request = read_json(request_path)
    began = time.monotonic()
    record = {'schema_version': SCHEMA, 'run_id': request['run']['run_id'],
              'analysis_id': digest(request['fingerprint']), 'artifact_sha256': request['artifact_sha256']}
    try:
        if analysis_fingerprint(request['fingerprint']['settings']) != request['fingerprint']:
            raise RuntimeError('Analysis sources/runtime changed')
        meta, arrays, _ = read_compact(request['artifact'])
        if meta['run'] != request['run'] or file_hash(Path(request['artifact']) / 'complete.json') != request['artifact_sha256']:
            raise ValueError('Artifact identity changed')
        record.update(state='complete', result=analyze_arrays(arrays, meta['run']['burn_in'],
                                                             meta['run']['seed'], request['fingerprint']['settings']))
    except Exception as error:
        record.update(state='worker_error', error=f'{type(error).__name__}: {error}')
    record.update(seconds=time.monotonic() - began,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    save_record(request['destination'], record)
    return 0


def analyze_experiment(directory, *, limits=None, timeout=300, retry_failed=False):
    """CLI entry after bootstrap. Analysis and simulation never run concurrently.

    A 256 MiB analysis reservation sits inside the atlas's total 5 GiB quota.
    Each analysis worker can write at most its request/log/result file allowance.
    """
    limits = limits or Limits()
    if not 0 < timeout <= 300:
        raise ValueError('Analysis timeout must be in (0, 300] seconds')
    directory = Path(directory).resolve()
    root = directory.parent
    manifest = manifest_at(directory)
    fingerprint = analysis_fingerprint()
    output = root / 'analysis' / digest(fingerprint)
    with experiment_lock(directory):
        output.mkdir(parents=True, exist_ok=True)
        for name in ('runs', 'staging', 'invocations'):
            (output / name).mkdir(exist_ok=True)
        atomic_json(output / 'analysis.json', {'schema_version': SCHEMA, 'fingerprint': fingerprint,
                                              'manifest_sha256': digest(manifest)})
        return _analyze(directory, root, output, manifest, fingerprint, limits, timeout, retry_failed)


def _analyze(directory, root, output, manifest, fingerprint, limits, timeout, retry_failed):
    started = time.monotonic()
    deadline = started + limits.minutes * 60
    active, stop, old_handlers = {}, [], {}
    invocation = {'started': now(), 'limits': asdict(limits), 'timeout_seconds': timeout,
                  'completed': 0, 'reused': 0, 'failures': 0, 'missing': 0, 'benchmarks': []}
    iterator = iter(manifest['runs'])
    exhausted, benchmark_done = False, False
    next_guard = 0
    reason = 'complete'

    def interrupted(signum, frame):
        stop.append(f'signal_{signum}')

    def guard():
        if shutil.disk_usage(root).free < limits.min_free_gib * GIB + 32 * MIB:
            return 'free_disk_limit'
        if directory_bytes(root) + 32 * MIB > limits.output_gib * GIB:
            return 'output_limit'
        if directory_bytes(root / 'analysis') + 32 * MIB > ANALYSIS_RESERVE:
            return 'analysis_output_limit'
        if memory_available() < GIB:
            return 'system_memory_pressure'
        if resident_bytes(os.getpid()) + sum(resident_bytes(e['process'].pid) for e in active.values()) > limits.memory_gib * GIB:
            return 'memory_limit'
        return None

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)
    print(json.dumps({'event': 'analysis_started', 'analysis_id': digest(fingerprint), **invocation}), flush=True)
    try:
        while active or not exhausted:
            if stop or time.monotonic() >= deadline:
                reason = stop[0] if stop else 'time_limit'
                break
            if time.monotonic() >= next_guard:
                reason_now = guard()
                if reason_now:
                    reason = reason_now
                    break
                next_guard = time.monotonic() + 2
            for run_id, entry in list(active.items()):
                process = entry['process']
                timed_out = time.monotonic() - entry['started'] >= timeout
                if timed_out and process.poll() is None:
                    _terminate([process])
                if process.poll() is None:
                    continue
                record = cached_record(entry['stage'] / 'result.json', entry['run'], entry['artifact'],
                                       fingerprint, validate=False)
                if timed_out or record is None:
                    record = {'schema_version': SCHEMA, 'run_id': run_id, 'analysis_id': digest(fingerprint),
                              'artifact_sha256': entry['artifact_hash'],
                              'state': 'timeout' if timed_out else 'worker_error',
                              'error': f'Worker exit {process.returncode}; timeout {timeout}s',
                              'seconds': time.monotonic() - entry['started']}
                if analysis_fingerprint() != fingerprint:
                    stop.append('provenance_changed')
                    break
                save_record(output / 'runs' / f'{run_id}.json', record)
                invocation['completed'] += 1
                invocation['failures'] += int(has_failure(record))
                if not benchmark_done:
                    invocation['benchmarks'].append(record)
                    print(json.dumps({'event': 'analysis_benchmark', **record}), flush=True)
                    benchmark_done = True
                print(json.dumps({'event': 'analyzed', 'run_id': run_id, 'state': record['state'],
                                  'seconds': record['seconds']}), flush=True)
                shutil.rmtree(entry['stage'])
                del active[run_id]
            capacity = limits.workers if benchmark_done else 1
            while not exhausted and not stop and len(active) < capacity and time.monotonic() < deadline:
                try:
                    run = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                artifact = directory / 'runs' / run['run_id']
                if not artifact.exists():
                    invocation['missing'] += 1
                    continue
                try:
                    record = cached_record(output / 'runs' / f'{run["run_id"]}.json', run, artifact, fingerprint)
                except ARTIFACT_ERRORS as error:
                    invocation['failures'] += 1
                    print(json.dumps({'event': 'artifact_error', 'run_id': run['run_id'], 'error': str(error)}), flush=True)
                    continue
                if record and (not has_failure(record) or not retry_failed):
                    invocation['reused'] += 1
                    invocation['failures'] += int(has_failure(record))
                    continue
                stage = output / 'staging' / run['run_id']
                if stage.exists():
                    shutil.rmtree(stage)
                stage.mkdir()
                artifact_hash = file_hash(artifact / 'complete.json')
                request = {'run': run, 'artifact': str(artifact), 'artifact_sha256': artifact_hash,
                           'fingerprint': fingerprint, 'destination': str(stage / 'result.json')}
                write_json(stage / 'request.json', request)
                with (stage / 'worker.log').open('wb') as stream:
                    process = subprocess.Popen([sys.executable, str(CLI), '_worker', str(stage / 'request.json'),
                                                '--memory-bytes', str(limits.worker_memory),
                                                '--parent-pid', str(os.getpid())],
                                               stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT,
                                               start_new_session=True)
                active[run['run_id']] = {'process': process, 'started': time.monotonic(), 'run': run,
                                         'stage': stage, 'artifact': artifact, 'artifact_hash': artifact_hash}
            if active:
                time.sleep(0.05)
        if reason == 'complete' and (invocation['failures'] or invocation['missing']):
            reason = 'failed_or_missing_runs'
    finally:
        _terminate([entry['process'] for entry in active.values()])
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        invocation.update(ended=now(), seconds=time.monotonic() - started, stop_reason=reason)
        atomic_json(output / 'invocations' / f'{uuid4().hex}.json', invocation)
        atomic_json(output / 'last_invocation.json', invocation)
        print(json.dumps({'event': 'analysis_finished', **invocation}), flush=True)
    return 0 if reason == 'complete' else 2


def collect(directory, analysis_id=None):
    directory = Path(directory)
    manifest = manifest_at(directory)
    output = directory.parent / 'analysis' / (analysis_id or digest(analysis_fingerprint()))
    metadata = read_json(output / 'analysis.json')
    if metadata['manifest_sha256'] != digest(manifest) or digest(metadata['fingerprint']) != output.name:
        raise ValueError('Analysis manifest identity mismatch')
    records = {}
    for run in manifest['runs']:
        artifact = directory / 'runs' / run['run_id']
        if not artifact.exists():
            continue
        try:
            record = cached_record(output / 'runs' / f'{run["run_id"]}.json', run, artifact, metadata['fingerprint'])
        except ARTIFACT_ERRORS:
            record = {'state': 'artifact_error'}
        if record:
            # Keep only scalar metrics for ensemble aggregation, not fit trajectories.
            records[run['run_id']] = record
    summary = aggregate(manifest, records)
    summary.update(analysis_id=output.name, fingerprint=metadata['fingerprint'],
                   manifest_sha256=digest(manifest),
                   operational_failures=sum(has_failure(r) for r in records.values()),
                   missing_analysis=len(manifest['runs']) - len(records))
    return summary, records
