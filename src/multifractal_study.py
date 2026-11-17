"""Bounded Task 7 orchestration, immutable inputs, and resumable control chunks."""

from copy import deepcopy
from dataclasses import asdict
import gzip
from importlib import metadata
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

import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from src.atlas_runner import atomic_json, manifest_at
from src.experiment_artifacts import (ROOT, digest, execution_fingerprint, file_hash,
                                      read_compact, read_json, write_json, provenance, validate_compact,
                                      expand_spec)
from src.experiments import (Limits, GIB, MIB, _remove_stage, _terminate, directory_bytes,
                             disk_guard, experiment_lock,
                             memory_available, now, resident_bytes, status)
from src.heterogeneity_study import select_configurations
from src.index_data import INDEX_FILES, load_chronological_index
from src.multifractal import SETTINGS, analyze_series, validate_settings
from src.multifractal_controls import control, fit_garch
from src.run_logging import json_safe

CLI = ROOT / 'experiments/run_multifractal.py'
FAMILIES = ('shuffled', 'iaaft', 'garch')
INDEX_NAMES = {'РТС': 'RTS', 'ММВБ': 'MOEX', 'S&P500': 'SP500', 'DJIA': 'DJIA', 'NASDAQ': 'NASDAQ'}


def analysis_fingerprint(settings=None):
    runtime = execution_fingerprint()
    packages, pending = dict(runtime['packages']), ['MFDFA', 'arch']
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
    paths = ['src/multifractal.py', 'src/multifractal_controls.py', 'src/multifractal_validation.py',
             'src/multifractal_study.py', 'src/index_data.py', 'src/experiment_artifacts.py',
             'src/experiments.py', 'src/atlas_runner.py', 'src/run_logging.py',
             'experiments/run_multifractal.py', 'experiments/run_experiment.py',
             'requirements-lock.txt', 'THIRD_PARTY_NOTICES.md']
    return {'settings': SETTINGS if settings is None else settings,
            'python': runtime['python'], 'platform': runtime['platform'], 'packages': packages,
            'source_sha256': {p: file_hash(ROOT / p) for p in paths}}


def verified_simulations(path, specification):
    path = Path(path).resolve()
    manifest = manifest_at(path)
    if manifest['specification'] != specification:
        raise ValueError('Reused simulation specification differs from the study')
    execution = manifest['provenance']['execution']
    if expand_spec(specification, execution) != manifest['runs']:
        raise ValueError('Reused simulation run inventory differs from the specification')
    for run in manifest['runs']:
        validate_compact(path/'runs'/run['run_id'], expected_run=run,
                         fingerprint=execution)
    return {'source': str(path), 'manifest_sha256': file_hash(path/'manifest.json'),
            'runs': len(manifest['runs']), 'bytes': directory_bytes(path)}


def generate_study(root, atlas, *, indices=None, seeds=(200, 201, 202), days=10240,
                   burn_in=2048, settings=None, reuse_simulations=None):
    settings = deepcopy(SETTINGS if settings is None else settings)
    validate_settings(settings)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Seeds must be unique uint32 integers')
    if (type(days) is not int or type(burn_in) is not int or burn_in < 0
            or days-burn_in < max(settings['simulation_lengths'])):
        raise ValueError('Simulation must contain all specified post-burn-in windows')
    configurations, source = select_configurations(atlas)
    inputs = []
    for label, filename in INDEX_FILES.items():
        path = Path(indices or ROOT / 'data') / filename
        data = load_chronological_index(path)
        if len(data['returns']) < max(settings['index_lengths']):
            raise ValueError(f'Insufficient observations in {path}')
        inputs.append({'name': INDEX_NAMES[label], 'filename': filename,
                       'source': str(path.resolve()), 'sha256': file_hash(path),
                       'returns': len(data['returns']), 'first_price_date': data['first_price_date'],
                       'last_price_date': data['last_price_date'],
                       'maximum_interval_days': int(data['interval_days'].max())})
    root = Path(root).resolve()
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 10*GIB + 4*MIB:
        raise ValueError('Insufficient free disk')
    fingerprint = analysis_fingerprint(settings)
    manifest = {'schema_version': 1, 'kind': 'multifractal-study', 'created': now(),
                'settings': settings, 'configurations': configurations, 'atlas': source,
                'indices': inputs, 'seeds': list(seeds), 'days': days, 'burn_in': burn_in,
                'analysis_fingerprint': fingerprint, 'analysis_id': digest(fingerprint),
                'provenance': provenance()}
    cells = [{'name': name, 'scenario': 'baseline', 'seeds': list(seeds),
              'overrides': {k: v for k, v in scenario.items() if k not in ('name', 'title')}}
             for name, scenario in configurations.items()]
    spec = {'schema_version': 1, 'name': 'task7-multifractal-pilot', 'days': days,
            'burn_in': burn_in, 'cells': cells}
    reused = verified_simulations(reuse_simulations, spec) if reuse_simulations is not None else None
    if reused:
        if reused['bytes'] > 4*GIB or shutil.disk_usage(ancestor).free < 10*GIB+reused['bytes']+4*MIB:
            raise ValueError('Reused simulation copy exceeds the study disk budget')
        manifest['reused_simulation'] = reused
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    for item in inputs:
        destination = root / 'inputs' / item['filename']
        shutil.copyfile(item['source'], destination)
        if file_hash(destination) != item['sha256']:
            raise ValueError('Index changed while snapshotting')
    write_json(root / 'spec.json', spec)
    if reused:
        shutil.copytree(reuse_simulations, root/'simulation')
        copied = verified_simulations(root/'simulation', spec)
        if copied['manifest_sha256'] != reused['manifest_sha256']:
            raise ValueError('Reused simulation manifest changed during copying')
    manifest['spec_sha256'] = file_hash(root / 'spec.json')
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'manifest.sha256.json', {'sha256': digest(manifest)})
    return manifest


def load_study(root, *, current=False):
    root = Path(root).resolve()
    manifest = manifest_at(root)
    if manifest.get('kind') != 'multifractal-study' or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported multifractal manifest')
    validate_settings(manifest['settings'])
    if digest(manifest['analysis_fingerprint']) != manifest['analysis_id']:
        raise ValueError('Analysis identity mismatch')
    if file_hash(root / 'spec.json') != manifest['spec_sha256']:
        raise ValueError('Simulation specification changed')
    if ('reused_simulation' in manifest and file_hash(root/'simulation/manifest.json')
            != manifest['reused_simulation']['manifest_sha256']):
        raise ValueError('Reused simulation manifest changed')
    for item in manifest['indices']:
        path = root / 'inputs' / item['filename']
        if path.is_symlink() or file_hash(path) != item['sha256']:
            raise ValueError(f'Corrupt index snapshot: {item["name"]}')
    if current and analysis_fingerprint(manifest['settings']) != manifest['analysis_fingerprint']:
        raise ValueError('Analysis source/runtime/settings changed; restore the recorded version or generate a new study')
    return manifest


def analysis_directory(root, manifest):
    return Path(root) / 'analysis' / manifest['analysis_id']


def save_record(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    payload = json_safe(payload)
    try:
        with temporary.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                compressed.write(json.dumps({'sha256': digest(payload), 'payload': payload},
                                             allow_nan=False, separators=(',', ':')).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def cached_record(path, job, analysis_id):
    try:
        with gzip.open(path, 'rt') as stream:
            envelope = json.load(stream)
        payload = envelope['payload']
        if (digest(payload) != envelope['sha256'] or payload['job'] != job
                or payload['analysis_id'] != analysis_id):
            return None
        return payload
    except (OSError, EOFError, ValueError, TypeError, KeyError):
        return None


def windows_for(root, manifest):
    root = Path(root).resolve()
    simulation = manifest_at(root / 'simulation')
    if simulation['specification'] != read_json(root / 'spec.json'):
        raise ValueError('Simulation manifest does not match study')
    windows = []
    # Reference-v0's longest record is benchmarked before parallel work.
    runs = sorted(simulation['runs'], key=lambda r: (r['cells'][0] != 'reference-v0',
                                                    r['cells'][0], r['seed']))
    for run in runs:
        artifact = root / 'simulation/runs' / run['run_id']
        meta, arrays, _ = read_compact(artifact)
        if meta['run'] != run:
            raise ValueError('Simulation identity mismatch')
        retained = len(arrays['log_return']) - run['burn_in']
        for length in sorted(manifest['settings']['simulation_lengths'], reverse=True):
            if retained < length:
                raise ValueError('Insufficient retained simulation days')
            window = {'kind': 'simulation', 'source_id': run['run_id'],
                      'label': run['cells'][0], 'seed': run['seed'], 'length': length,
                      'start': run['days']-length, 'artifact': str(artifact),
                      'input_sha256': file_hash(artifact / 'complete.json')}
            window['window_id'] = digest(window)
            windows.append(window)
    for item in manifest['indices']:
        for length in sorted(manifest['settings']['index_lengths'], reverse=True):
            window = {'kind': 'index', 'source_id': item['name'], 'label': item['name'],
                      'seed': None, 'length': length, 'start': item['returns']-length,
                      'artifact': str(root / 'inputs' / item['filename']), 'input_sha256': item['sha256']}
            window['window_id'] = digest(window)
            windows.append(window)
    return windows


def load_window(window):
    path, length = Path(window['artifact']), window['length']
    if window['kind'] == 'simulation':
        if file_hash(path / 'complete.json') != window['input_sha256']:
            raise ValueError('Simulation input identity changed')
        _, arrays, _ = read_compact(path)
        values, volume = arrays['log_return'][-length:], arrays['volume'][-length:]
        longest = terminal = 0
        for empty in volume == 0:
            terminal = terminal+1 if empty else 0
            longest = max(longest, terminal)
        diagnostics = {'zero_volume_fraction': float(np.mean(volume == 0)),
                       'longest_no_trade_streak': longest, 'terminal_no_trade_streak': terminal,
                       'first_trading_day': int(arrays['day'][-length]),
                       'last_trading_day': int(arrays['day'][-1])}
    else:
        if path.is_symlink() or file_hash(path) != window['input_sha256']:
            raise ValueError('Index input identity changed')
        arrays = load_chronological_index(path)
        values = arrays['returns'][-length:]
        diagnostics = {'first_return_date': str(arrays['dates'][-length]),
                       'last_return_date': str(arrays['dates'][-1]),
                       'maximum_interval_days': int(arrays['interval_days'][-length:].max()),
                       'clock': 'consecutive recorded trading observations; no gap filling'}
    if len(values) != length:
        raise ValueError('Incomplete analysis window')
    return values, diagnostics


def make_job(window, phase, **extra):
    job = {'phase': phase, 'window': window, **extra}
    job['job_id'] = digest(job)
    return job


def jobs_for(windows, phase, settings):
    if phase in ('observed', 'fit'):
        return [make_job(w, phase) for w in windows]
    # Benchmark an IAAFT chunk on the longest original-forcing reference first.
    return [make_job(w, 'controls', family=family, first=first,
                     stop=min(first+settings['chunk_size'], settings['replicates']))
            for w in windows for family in ('iaaft', 'shuffled', 'garch')
            for first in range(0, settings['replicates'], settings['chunk_size'])]


def result_path(output, job):
    return Path(output) / 'records' / (job['job_id'] + '.json.gz')


def compute_job(root, manifest, job):
    settings, window = manifest['settings'], job['window']
    values, diagnostics = load_window(window)
    if job['phase'] == 'observed':
        return {'analysis': analyze_series(values, settings=settings), 'window_diagnostics': diagnostics}
    if job['phase'] == 'fit':
        return {'fit': fit_garch(values, settings)}
    fit = None
    if job['family'] == 'garch':
        fit_job = make_job(window, 'fit')
        record = cached_record(result_path(analysis_directory(root, manifest), fit_job),
                               fit_job, manifest['analysis_id'])
        if record is None or record['state'] != 'complete':
            raise ValueError('Missing or invalid GARCH fit prerequisite')
        fit = record['result']['fit']
    records = []
    for replicate in range(job['first'], job['stop']):
        values_control, diagnostic = control(values, job['family'], window['window_id'],
                                              replicate, fit=fit, settings=settings)
        records.append({'replicate': replicate, 'diagnostic': diagnostic,
                        'analysis': analyze_series(values_control, settings=settings)
                        if values_control is not None else None})
    return {'replicates': records}


def worker(request_path):
    request = read_json(request_path)
    began = time.monotonic()
    payload = {'job': request['job'], 'analysis_id': request['analysis_id']}
    try:
        manifest = load_study(request['root'], current=True)
        if manifest['analysis_id'] != request['analysis_id']:
            raise ValueError('Worker analysis identity mismatch')
        result = compute_job(request['root'], manifest, request['job'])
        load_study(request['root'], current=True)
        payload.update(state='complete', result=result)
    except Exception as error:
        payload.update(state='worker_error', error=f'{type(error).__name__}: {error}')
    payload.update(seconds=time.monotonic()-began,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save_record(request['destination'], payload)
    return 0 if payload['state'] == 'complete' else 1


def phase_jobs(root, manifest, jobs, limits, *, timeout=300, retry_failed=False):
    if not 0 < timeout <= 300:
        raise ValueError('Analysis worker timeout must be in (0, 300] seconds')
    output = analysis_directory(root, manifest)
    for name in ('records', 'staging', 'invocations'):
        (output / name).mkdir(parents=True, exist_ok=True)
    began, deadline = time.monotonic(), time.monotonic()+limits.minutes*60
    invocation = {'started': now(), 'phase': jobs[0]['phase'] if jobs else 'empty',
                  'limits': asdict(limits), 'timeout_seconds': timeout, 'completed': 0,
                  'reused': 0, 'failures': 0, 'benchmarks': []}
    pending = []
    for job in jobs:
        record = cached_record(result_path(output, job), job, manifest['analysis_id'])
        if record and (record['state'] == 'complete' or not retry_failed):
            invocation['reused'] += 1
            invocation['failures'] += int(record['state'] != 'complete')
        else:
            pending.append(job)
    active, stopped, benchmark = {}, [], False
    iterator, exhausted = iter(pending), False
    previous = {s: signal.signal(s, lambda signum, frame: stopped.append(f'signal_{signum}'))
                for s in (signal.SIGINT, signal.SIGTERM)}
    reason, next_guard = 'complete', 0.
    try:
        while active or not exhausted:
            if stopped or time.monotonic() >= deadline:
                reason = stopped[0] if stopped else 'time_limit'
                break
            if time.monotonic() >= next_guard:
                problem = disk_guard(root, limits, reserved=64*MIB+32*MIB*len(active))
                if problem or memory_available() < GIB:
                    reason = problem or 'system_memory_pressure'
                    break
                if resident_bytes(os.getpid()) + sum(resident_bytes(e['process'].pid) for e in active.values()) > limits.memory_gib*GIB:
                    reason = 'memory_limit'
                    break
                next_guard = time.monotonic()+1
            for job_id, entry in list(active.items()):
                process = entry['process']
                expired = time.monotonic()-entry['started'] >= timeout
                if expired and process.poll() is None:
                    _terminate([process])
                if process.poll() is None:
                    continue
                record = cached_record(entry['stage'] / 'result.json.gz', entry['job'], manifest['analysis_id'])
                if expired or record is None:
                    record = {'job': entry['job'], 'analysis_id': manifest['analysis_id'],
                              'state': 'timeout' if expired else 'worker_error',
                              'error': f'Worker exit {process.returncode}',
                              'seconds': time.monotonic()-entry['started']}
                load_study(root, current=True)
                save_record(result_path(output, entry['job']), record)
                invocation['completed'] += 1
                invocation['failures'] += int(record['state'] != 'complete')
                if not benchmark:
                    invocation['benchmarks'].append({k: record.get(k) for k in
                                                    ('seconds', 'peak_rss_bytes', 'state')})
                    print(json.dumps({'event': 'analysis_benchmark', 'phase': invocation['phase'],
                                      **invocation['benchmarks'][-1]}), flush=True)
                    benchmark = True
                    if record['state'] != 'complete':
                        stopped.append('benchmark_failed')
                print(json.dumps({'event': 'analyzed', 'job_id': job_id, 'phase': invocation['phase'],
                                  'state': record['state'], 'seconds': record.get('seconds')}), flush=True)
                _remove_stage(entry['stage'])
                del active[job_id]
            capacity = limits.workers if benchmark else 1
            while not stopped and not exhausted and len(active) < capacity:
                try:
                    job = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                stage = output / 'staging' / job['job_id']
                _remove_stage(stage)
                stage.mkdir()
                request = {'root': str(root), 'analysis_id': manifest['analysis_id'], 'job': job,
                           'destination': str(stage / 'result.json.gz')}
                write_json(stage / 'request.json', request)
                with (stage / 'worker.log').open('wb') as log:
                    process = subprocess.Popen([sys.executable, str(CLI), '_worker', str(stage / 'request.json'),
                                                '--memory-bytes', str(limits.worker_memory),
                                                '--parent-pid', str(os.getpid())], cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job['job_id']] = {'job': job, 'stage': stage, 'process': process,
                                         'started': time.monotonic()}
            if active:
                time.sleep(.05)
        if stopped:
            reason = stopped[0]
        if reason == 'complete' and invocation['failures']:
            reason = 'failed_jobs'
    finally:
        _terminate([e['process'] for e in active.values()])
        for s, handler in previous.items():
            signal.signal(s, handler)
        invocation.update(ended=now(), seconds=time.monotonic()-began, stop_reason=reason)
        atomic_json(output / 'invocations' / (uuid4().hex+'.json'), invocation)
        atomic_json(output / 'last_invocation.json', invocation)
        print(json.dumps({'event': 'analysis_phase_finished', **invocation}), flush=True)
    return 0 if reason == 'complete' else 2


def validate_study(root, manifest=None):
    from src.multifractal_validation import validate_estimators
    manifest = manifest or load_study(root, current=True)
    path = Path(root) / 'estimator_validation.json'
    if path.exists():
        try:
            saved = read_json(path)
            payload = saved['payload']
            if saved['sha256'] == digest(payload) and payload['analysis_id'] == manifest['analysis_id'] and payload['validation']['passed']:
                return payload['validation']
        except (OSError, ValueError, KeyError, TypeError):
            pass
    result = validate_estimators(manifest['settings'])
    payload = {'analysis_id': manifest['analysis_id'], 'validation': json_safe(result)}
    atomic_json(path, {'sha256': digest(payload), 'payload': payload})
    return result


def analyze_study(root, *, limits=None, timeout=300, retry_failed=False, repeat=False, locked=False):
    root, limits = Path(root).resolve(), limits or Limits()
    if limits.minutes > 120:
        raise ValueError('Analysis batches cannot exceed 120 minutes')
    if not locked:
        with experiment_lock(root):
            return analyze_study(root, limits=limits, timeout=timeout, retry_failed=retry_failed,
                                 repeat=repeat, locked=True)
    manifest = load_study(root, current=True)
    if not validate_study(root, manifest)['passed']:
        raise ValueError('Known-answer estimator validation failed')
    windows = windows_for(root, manifest)
    output = analysis_directory(root, manifest)
    output.mkdir(parents=True, exist_ok=True)
    inventory = {'analysis_id': manifest['analysis_id'], 'windows': windows}
    atomic_json(output / 'inventory.json', {'sha256': digest(inventory), 'payload': inventory})
    for phase in ('observed', 'fit', 'controls'):
        jobs = jobs_for(windows, phase, manifest['settings'])
        while True:
            code = phase_jobs(root, manifest, jobs, limits, timeout=timeout, retry_failed=retry_failed)
            if code == 0:
                break
            reason = read_json(output / 'last_invocation.json')['stop_reason']
            if not repeat or reason != 'time_limit':
                return code
    return 0


def complete_study(root, *, limits=None, retry_failed=False):
    root, limits = Path(root).resolve(), limits or Limits()
    if limits.minutes > 120:
        raise ValueError('Study batches cannot exceed 120 minutes')
    with experiment_lock(root):
        manifest = load_study(root, current=True)
        if not validate_study(root, manifest)['passed']:
            raise ValueError('Known-answer estimator validation failed')
        simulation = root / 'simulation'
        while True:
            if 'reused_simulation' in manifest:
                verified_simulations(simulation, read_json(root/'spec.json'))
                break
            if disk_guard(root, limits, reserved=128*MIB):
                raise ValueError('Study disk budget prevents simulation')
            outside = directory_bytes(root) - (directory_bytes(simulation) if simulation.exists() else 0)
            simulation_limits = Limits(workers=limits.workers, minutes=min(120, limits.minutes),
                                       memory_gib=limits.memory_gib-.75,
                                       output_gib=min(4.5, limits.output_gib-outside/GIB-.125),
                                       min_free_gib=limits.min_free_gib)
            code = launch_simulation(root, simulation_limits, retry_failed)
            if code == 0:
                break
            state = status(simulation)
            reason = state['latest_invocation']['stop_reason']
            if reason != 'time_limit':
                print(json.dumps({'event': 'simulation_stopped', 'status': state}), flush=True)
                return code
        code = analyze_study(root, limits=limits, retry_failed=retry_failed, repeat=True, locked=True)
        if code:
            return code
        from src.multifractal_plotting import render_study
        render_study(root, limits=limits, locked=True)
    return 0


def simulation_worker(request_path):
    """Parent-death protected adapter around the unchanged simulation supervisor."""
    from src.experiments import _execute
    request = read_json(request_path)
    root = Path(request['root'])
    load_study(root, current=True)
    simulation = root / 'simulation'
    return _execute(simulation, Limits(**request['limits']),
                    spec=None if simulation.exists() else read_json(root/'spec.json'),
                    retry_failed=request['retry_failed'] if simulation.exists() else False)


def launch_simulation(root, limits, retry_failed):
    request = root / 'simulation_request.json'
    atomic_json(request, {'root': str(root), 'limits': asdict(limits), 'retry_failed': retry_failed})
    stopped = []
    previous = {s: signal.signal(s, lambda signum, frame: stopped.append(signum))
                for s in (signal.SIGINT, signal.SIGTERM)}
    process = None
    try:
        with (root/'simulation.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, str(CLI), '_simulation', str(request),
                                        '--memory-bytes', str(int(limits.memory_gib*GIB)), '--parent-pid', str(os.getpid())],
                                       stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
            while process.poll() is None:
                if stopped:
                    _terminate([process])
                    return 2
                time.sleep(.1)
            return process.returncode
    finally:
        if process is not None and process.poll() is None:
            _terminate([process])
        for s, handler in previous.items():
            signal.signal(s, handler)


def study_status(root):
    root = Path(root).resolve()
    manifest = load_study(root)
    output = analysis_directory(root, manifest)
    counts = {}
    for path in (output / 'records').glob('*.json.gz'):
        try:
            with gzip.open(path, 'rt') as stream:
                envelope = json.load(stream)
            record = envelope['payload']
            if digest(record) != envelope['sha256'] or record['analysis_id'] != manifest['analysis_id']:
                raise ValueError('Corrupt analysis record')
            key = record['job']['phase'] + ':' + record['state']
        except (OSError, EOFError, ValueError, KeyError):
            key = 'corrupt'
        counts[key] = counts.get(key, 0)+1
    return {'root': str(root), 'analysis_id': manifest['analysis_id'], 'analysis_records': counts,
            'simulation': status(root/'simulation') if (root/'simulation/ledger.sqlite').exists() else None,
            'last_invocation': read_json(output/'last_invocation.json') if (output/'last_invocation.json').exists() else None,
            'report': str(root/'reports'/manifest['analysis_id'])}
