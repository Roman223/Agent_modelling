"""Single-host experiment supervisor, durable ledger, and isolated worker entrypoint.

Use execute_experiment from Python or experiments/run_experiment.py from a shell.
The supervisor runs in a separate, bounded process even for the Python interface.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from uuid import uuid4

from src.experiment_artifacts import (
    ARTIFACT_ERRORS, ROOT, canonical, digest, execution_fingerprint, expand_spec, provenance,
    read_compact, read_json, save_compact, validate_compact, write_json,
)

GIB = 1024**3
MIB = 1024**2
COORDINATOR_MEMORY = 768 * MIB
PROBE_RESERVE = 256 * MIB
COORDINATOR_OUTPUT = 64 * MIB
# At peak: request, worker log, result; 3 temporary RunLog files; 8 artifact files.
# Two spare slots cover failure bookkeeping. Every worker file has RLIMIT_FSIZE.
WORKER_FILE_SLOTS = 16
CLI = ROOT / 'experiments/run_experiment.py'


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Limits:
    workers: int = 2
    minutes: float = 120
    memory_gib: float = 6
    output_gib: float = 5
    min_free_gib: float = 10

    def __post_init__(self):
        if type(self.workers) is not int or not 1 <= self.workers <= 6:
            raise ValueError('workers must be between 1 and 6')
        for name in ('minutes', 'memory_gib', 'output_gib', 'min_free_gib'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number')
        if not 1 < self.memory_gib <= 6:
            raise ValueError('memory_gib must be greater than 1 and at most 6')
        if self.output_gib > 5:
            raise ValueError('output_gib cannot exceed 5')
        if self.min_free_gib < 10:
            raise ValueError('min_free_gib cannot be less than 10')

    @property
    def worker_memory(self):
        return (int(self.memory_gib * GIB) - COORDINATOR_MEMORY - PROBE_RESERVE) // self.workers


@contextmanager
def experiment_lock(directory):
    with (Path(directory) / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another coordinator holds the experiment lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def directory_bytes(directory):
    total = 0
    for path in Path(directory).rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Workers may remove their temporary logs during a scan.
    return total


def disk_guard(directory, limits, reserved=0):
    usage = shutil.disk_usage(directory)
    if usage.free - reserved - COORDINATOR_OUTPUT < int(limits.min_free_gib * GIB):
        return 'free_disk_limit'
    if directory_bytes(directory) + reserved + COORDINATOR_OUTPUT > int(limits.output_gib * GIB):
        return 'output_limit'
    return None


def resident_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux MemAvailable accounting is unavailable')


def _connect(directory):
    db = sqlite3.connect(Path(directory) / 'ledger.sqlite', timeout=0)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA max_page_count=4096')  # <=16 MiB database, <=16 MiB rollback journal
    db.executescript('''
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
        CREATE TABLE IF NOT EXISTS invocations (
            id TEXT PRIMARY KEY, started TEXT NOT NULL, ended TEXT,
            limits TEXT NOT NULL, stop_reason TEXT, benchmark_run TEXT);
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
            invocation TEXT NOT NULL REFERENCES invocations(id), state TEXT NOT NULL,
            started TEXT NOT NULL, ended TEXT, pid INTEGER, seconds REAL,
            peak_rss_bytes INTEGER, artifact_bytes INTEGER, error TEXT,
            memory_limit INTEGER, file_limit INTEGER);
    ''')
    return db


def status(directory):
    """Read the ledger without taking the writer lock or changing any state."""
    path = (Path(directory) / 'ledger.sqlite').resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        counts = dict(db.execute('SELECT state, COUNT(*) FROM runs GROUP BY state'))
        latest = db.execute('SELECT id, started, ended, stop_reason FROM invocations ORDER BY rowid DESC LIMIT 1').fetchone()
        return {'directory': str(path.parent), 'runs': counts,
                'latest_invocation': dict(zip(('id', 'started', 'ended', 'stop_reason'), latest)) if latest else None}


def _remove_stage(path):
    if path.is_symlink():
        raise ValueError('Refusing a symlink as a staging directory')
    if path.exists():
        shutil.rmtree(path)


def _fsync_tree(directory):
    for path in Path(directory).rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for path in [p for p in Path(directory).rglob('*') if p.is_dir()] + [Path(directory)]:
        _fsync_directory(path)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def recover(directory, db, manifest, retry_failed):
    """Reconcile the ledger with atomically published artifacts under the lock."""
    execution = manifest['provenance']['execution']
    expected = {r['run_id']: r for r in manifest['runs']}
    rows = list(db.execute('SELECT * FROM runs'))
    existing = {r['run_id']: json.loads(r['spec']) for r in rows}
    if any(run_id not in expected or run != expected[run_id] for run_id, run in existing.items()):
        raise ValueError('Ledger and immutable experiment manifest disagree')
    # A crash can occur after manifest publication but before ledger initialization.
    with db:
        db.executemany('INSERT INTO runs(run_id,spec,state) VALUES(?,?,?)',
                       [(run_id, canonical(run), 'pending') for run_id, run in expected.items()
                        if run_id not in existing])
    rows = list(db.execute('SELECT * FROM runs'))
    for row in rows:
        run_id, state, error = row['run_id'], row['state'], row['error']
        target = directory / 'runs' / run_id
        stage = directory / 'staging' / run_id
        if target.exists():
            try:
                validate_compact(target, expected[run_id], execution)
                state, error = 'complete', None
            except ARTIFACT_ERRORS as problem:
                state, error = 'failed', f'Corrupt completed artifact: {problem}'
        elif state == 'complete':
            state, error = 'failed', 'Completed artifact is missing'
        elif state == 'running':
            state, error = 'interrupted', 'Coordinator exited before publication'
        # A completed staging directory was not published: rerun from its seed.
        # This also removes temporary logs left by SIGKILL.
        _remove_stage(stage)
        if retry_failed and state == 'failed':
            _remove_stage(target)
            state, error = 'pending', None
        with db:
            db.execute('UPDATE runs SET state=?, error=? WHERE run_id=?', (state, error, run_id))
            db.execute("UPDATE attempts SET state=?, ended=?, error=? WHERE run_id=? AND state='running'",
                       ('complete' if state == 'complete' else 'interrupted', now(),
                        None if state == 'complete' else 'Coordinator exited', run_id))
    with db:
        db.execute("UPDATE invocations SET ended=?, stop_reason='coordinator_exited' WHERE ended IS NULL", (now(),))


def _worker(request_path):
    """All model allocation and array compression happen in this capped process."""
    import numpy as np
    from src.run_logging import LogOptions
    from src.scenarios import Scenario
    from src.simulation import simulate

    request_path = Path(request_path)
    stage = request_path.parent
    request = read_json(request_path)
    started = time.monotonic()
    result = None
    try:
        if execution_fingerprint() != request['execution']:
            raise ValueError('Execution sources or runtime changed before worker startup')
        run = request['run']
        scenario = Scenario(**run['scenario'])
        np.random.seed(run['seed'])
        result = simulate(scenario, days=run['days'], log_options=LogOptions(level='summary'))
        save_compact(result, run, stage / 'artifact', request['execution'])
        result.run_log.cleanup()
        result = None
        if execution_fingerprint() != request['execution']:
            raise ValueError('Execution sources or runtime changed during simulation')
        validate_compact(stage / 'artifact', run, request['execution'])
        _fsync_tree(stage / 'artifact')
        write_json(stage / 'result.json', {
            'state': 'complete', 'seconds': time.monotonic() - started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'artifact_bytes': directory_bytes(stage / 'artifact'),
        })
        return 0
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'[:4000]
        # A MemoryError/file-size failure may leave no room for this diagnostic.
        try:
            write_json(stage / 'result.json', {'state': 'failed', 'error': reason,
                       'seconds': time.monotonic() - started,
                       'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        except (OSError, MemoryError):
            pass
        print(reason, file=sys.stderr)
        return 1
    finally:
        if result is not None:
            result.run_log.cleanup()


def _terminate(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 1
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait()


def _supervise(directory, db, manifest, limits, deadline):
    invocation = uuid4().hex
    with db:
        db.execute('INSERT INTO invocations(id, started, limits) VALUES(?,?,?)',
                   (invocation, now(), canonical(asdict(limits))))
    log_path = directory / 'invocations' / f'{invocation}.jsonl'
    log_bytes = 0

    def emit(event, **fields):
        nonlocal log_bytes
        line = canonical({'time': now(), 'event': event, **fields}) + '\n'
        if log_bytes + len(line.encode()) <= 64 * 1024:
            with log_path.open('a', encoding='utf-8') as stream:
                stream.write(line)
            log_bytes += len(line.encode())
        print(line, end='', flush=True)

    active, stopped, previous_handlers = {}, [], {}
    pending = [dict(r) for r in db.execute("SELECT * FROM runs WHERE state IN ('pending','interrupted') ORDER BY rowid")]
    execution = manifest['provenance']['execution']
    benchmark_done = False
    estimate = 0.0
    reason = 'complete'

    def stop(signum, frame):
        stopped.append(f'signal_{signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop)
    emit('invocation_started', invocation=invocation, pending=len(pending), limits=asdict(limits))
    try:
        while pending or active:
            if stopped:
                reason = stopped[0]
                break
            if time.monotonic() >= deadline:
                reason = 'time_limit'
                break
            # Reservations bound writes between polls; monitoring also catches external disk usage.
            reason_now = disk_guard(directory, limits)
            if reason_now:
                reason = reason_now
                break
            if resident_bytes(os.getpid()) + sum(resident_bytes(p.pid) for p, *_ in active.values()) > limits.memory_gib * GIB:
                reason = 'memory_limit'
                break
            if memory_available() < GIB:
                reason = 'system_memory_pressure'
                break

            for run_id, (process, attempt_id, stage, quota, launched, peak) in list(active.items()):
                peak = max(peak, resident_bytes(process.pid))
                active[run_id] = (process, attempt_id, stage, quota, launched, peak)
                code = process.poll()
                if code is None:
                    continue
                try:
                    record = read_json(stage / 'result.json') if (stage / 'result.json').exists() else {}
                except (OSError, ValueError):
                    record = {}
                if not isinstance(record, dict):
                    record = {}
                with (stage / 'worker.log').open('rb') as stream:
                    stream.seek(max(0, (stage / 'worker.log').stat().st_size - 3000))
                    diagnostic = stream.read(3000).decode('utf-8', errors='replace')
                error = record.get('error', f'Worker exited with code {code}: {diagnostic}')
                state = 'failed'
                if code == 0 and record.get('state') == 'complete':
                    try:
                        run = json.loads(db.execute('SELECT spec FROM runs WHERE run_id=?', (run_id,)).fetchone()[0])
                        validate_compact(stage / 'artifact', run, execution)
                        if execution_fingerprint() != execution:
                            raise ValueError('Execution provenance changed before publication')
                        os.rename(stage / 'artifact', directory / 'runs' / run_id)
                        _fsync_directory(directory / 'runs')
                        state, error = 'complete', None
                    except ARTIFACT_ERRORS as problem:
                        error = str(problem)
                seconds = time.monotonic() - launched
                peak = max(peak, record.get('peak_rss_bytes', 0))
                size = record.get('artifact_bytes', 0)
                with db:
                    db.execute('UPDATE runs SET state=?, error=? WHERE run_id=?', (state, error, run_id))
                    db.execute('UPDATE attempts SET state=?, ended=?, seconds=?, peak_rss_bytes=?, artifact_bytes=?, error=? WHERE id=?',
                               (state, now(), seconds, peak, size, error, attempt_id))
                emit('run_finished', run_id=run_id, state=state, seconds=seconds,
                     peak_rss_bytes=peak, artifact_bytes=size, error=error)
                del active[run_id]
                _remove_stage(stage)
                if not benchmark_done:
                    if state != 'complete':
                        reason = 'benchmark_failed'
                        stopped.append(reason)
                        break
                    benchmark_done = True
                    estimate = seconds
                    with db:
                        db.execute('UPDATE invocations SET benchmark_run=? WHERE id=?', (run_id, invocation))
                    emit('benchmark_complete', run_id=run_id, seconds=seconds,
                         peak_rss_bytes=peak, artifact_bytes=size)
                else:
                    estimate = max(estimate, seconds)
                if code in (-signal.SIGXFSZ, -signal.SIGKILL) or (error and ('MemoryError' in error or 'File too large' in error)):
                    stopped.append('worker_resource_limit')
            if stopped:
                continue

            capacity = limits.workers if benchmark_done else 1
            while pending and not stopped and len(active) < capacity and (benchmark_done or not active):
                remaining = deadline - time.monotonic()
                if remaining <= max(0.05, estimate * 1.1):
                    if not active:
                        stopped.append('time_limit')
                    break
                if execution_fingerprint() != execution:
                    stopped.append('provenance_changed')
                    break
                if stopped or time.monotonic() >= deadline:
                    break
                # Conservative: count current staging bytes AND full outstanding reservations.
                reserved = sum(entry[3] for entry in active.values())
                available = min(int(limits.output_gib * GIB) - directory_bytes(directory) - COORDINATOR_OUTPUT - reserved,
                                shutil.disk_usage(directory).free - int(limits.min_free_gib * GIB)
                                - COORDINATOR_OUTPUT - reserved)
                quota = min(512 * MIB, available // max(1, capacity - len(active)))
                if quota < 256 * 1024:
                    if not active:
                        stopped.append('output_or_free_disk_limit')
                    break
                row = pending.pop(0)
                run_id, run = row['run_id'], json.loads(row['spec'])
                stage = directory / 'staging' / run_id
                stage.mkdir()
                (stage / 'tmp').mkdir()
                request = {'run': run, 'execution': execution}
                if len(canonical(request).encode()) * 2 > quota // WORKER_FILE_SLOTS:
                    raise ValueError('Worker reservation too small for request metadata')
                write_json(stage / 'request.json', request)
                file_limit = quota // WORKER_FILE_SLOTS
                env = os.environ.copy()
                env.update(TMPDIR=str(stage / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
                with db:
                    cursor = db.execute('INSERT INTO attempts(run_id,invocation,state,started,memory_limit,file_limit) VALUES(?,?,?,?,?,?)',
                                        (run_id, invocation, 'running', now(), limits.worker_memory, file_limit))
                    attempt_id = cursor.lastrowid
                    db.execute("UPDATE runs SET state='running', attempts=attempts+1, error=NULL WHERE run_id=?", (run_id,))
                with (stage / 'worker.log').open('wb') as worker_log:
                    process = subprocess.Popen([sys.executable, str(CLI), '_worker', str(stage / 'request.json'),
                                                '--memory-bytes', str(limits.worker_memory), '--file-bytes', str(file_limit),
                                                '--parent-pid', str(os.getpid())], stdout=worker_log, stderr=subprocess.STDOUT,
                                               env=env, cwd=ROOT, start_new_session=True)
                active[run_id] = (process, attempt_id, stage, quota, time.monotonic(), 0)
                with db:
                    db.execute('UPDATE attempts SET pid=? WHERE id=?', (process.pid, attempt_id))
                emit('run_started', run_id=run_id, seed=run['seed'], pid=process.pid, benchmark=not benchmark_done)
                if not benchmark_done:
                    break
            if active:
                time.sleep(0.1)
        if stopped:
            reason = stopped[0]
        if reason == 'complete' and db.execute("SELECT COUNT(*) FROM runs WHERE state != 'complete'").fetchone()[0]:
            reason = 'failed_runs'
    except BaseException:
        reason = 'supervisor_error'
        raise
    finally:
        _terminate([entry[0] for entry in active.values()])
        for run_id, (_, attempt_id, _, _, launched, peak) in active.items():
            with db:
                db.execute("UPDATE runs SET state='interrupted', error=? WHERE run_id=?", (reason, run_id))
                db.execute("UPDATE attempts SET state='interrupted', ended=?, seconds=?, peak_rss_bytes=?, error=? WHERE id=?",
                           (now(), time.monotonic() - launched, peak, reason, attempt_id))
        with db:
            db.execute('UPDATE invocations SET ended=?, stop_reason=? WHERE id=?', (now(), reason, invocation))
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        emit('invocation_finished', reason=reason, **status(directory))
    return 0 if reason == 'complete' else 2


def _execute(directory, limits, spec=None, retry_failed=False):
    """Internal supervisor entry: the CLI has already installed process limits."""
    deadline = time.monotonic() + limits.minutes * 60
    directory = Path(directory).resolve()
    if spec is not None:
        recorded = provenance()
        runs = expand_spec(spec, recorded['execution'])
        manifest = {'experiment_schema_version': 1, 'created': now(), 'specification': spec,
                    'provenance': recorded, 'runs': runs}
        if len(canonical(manifest).encode()) > 4 * MIB:
            raise ValueError('Resolved manifest exceeds 4 MiB; split into smaller experiments')
        # Validate limits/free space before creating output.
        parent = directory.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < limits.min_free_gib * GIB + COORDINATOR_OUTPUT:
            raise ValueError('Insufficient free disk for experiment')
        if limits.output_gib * GIB <= COORDINATOR_OUTPUT:
            raise ValueError('Output budget must exceed the 64 MiB coordinator reserve')
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise ValueError('Experiment directory does not exist')
    with experiment_lock(directory):
        if spec is not None:
            for name in ('runs', 'staging', 'invocations'):
                (directory / name).mkdir()
            write_json(directory / 'manifest.json', manifest)
            write_json(directory / 'manifest.sha256.json', {'sha256': digest(manifest)})
        else:
            manifest = read_json(directory / 'manifest.json')
            if digest(manifest) != read_json(directory / 'manifest.sha256.json')['sha256']:
                raise ValueError('Immutable manifest was modified')
            if manifest['provenance']['execution'] != execution_fingerprint():
                raise ValueError('Execution sources/runtime differ; restore recorded versions or start a new experiment')
            if expand_spec(manifest['specification'], manifest['provenance']['execution']) != manifest['runs']:
                raise ValueError('Resolved scientific configuration differs')
        if disk_guard(directory, limits):
            raise ValueError('Insufficient disk/output capacity for coordinator reserve')
        db = _connect(directory)
        try:
            if spec is not None:
                with db:
                    db.executemany('INSERT INTO runs(run_id,spec,state) VALUES(?,?,?)',
                                   [(r['run_id'], canonical(r), 'pending') for r in manifest['runs']])
            recover(directory, db, manifest, retry_failed)
            return _supervise(directory, db, manifest, limits, deadline)
        finally:
            db.close()


def execute_experiment(directory, *, spec=None, limits=None, retry_failed=False, log_path=None):
    """Run a bounded coordinator subprocess. spec=None resumes an existing batch.

    Pass a JSON specification path for a new batch. Launch long calls in the
    background; log_path redirects both output streams. Returns the CLI exit code.
    """
    limits = limits or Limits()
    command = [sys.executable, str(CLI), 'run' if spec is not None else 'resume']
    if spec is not None:
        command.append(str(spec))
    command.append(str(directory))
    for name, value in asdict(limits).items():
        command.extend(['--' + name.replace('_', '-'), str(value)])
    if retry_failed:
        if spec is not None:
            raise ValueError('retry_failed only applies to resume')
        command.append('--retry-failed')
    if log_path is not None:
        with Path(log_path).open('a') as stream:
            return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
    return subprocess.call(command)


def cli_main(args):
    if args.command == '_worker':
        return _worker(args.request)
    if args.command == 'validate':
        runs = expand_spec(read_json(args.spec))
        print(json.dumps({'unique_runs': len(runs), 'runs': runs}, indent=2))
        return 0
    if args.command == 'status':
        print(json.dumps(status(args.directory), indent=2))
        return 0
    limits = Limits(**{name: getattr(args, name) for name in asdict(Limits())})
    return _execute(args.directory, limits, spec=read_json(args.spec) if args.command == 'run' else None,
                    retry_failed=getattr(args, 'retry_failed', False))
