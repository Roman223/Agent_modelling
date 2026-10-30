"""Generate, analyze/resume, validate, and render the task 3 parameter atlas."""

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys


def complete_study(root, minutes=120, workers=2):
    """Small stdlib-only driver; repeat time-limited batches, stop on other failures."""
    if not 0 < minutes <= 120 or not 1 <= workers <= 6:
        raise ValueError('Use 1–6 workers and a batch budget in (0, 120] minutes')
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cli = Path(__file__).resolve()
    simulation_cli = cli.with_name('run_experiment.py')
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR', '/tmp/abm-task3-matplotlib')
    active, interrupted = [], []

    def stop(signum, frame):
        interrupted.append(signum)
        for process in active:
            process.send_signal(signum)

    def call(command, label):
        if interrupted:
            return 2
        command = list(map(str, command))
        print(json.dumps({'stage': label, 'command': command}), flush=True)
        with (root / f'{label}.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, *map(str, command)], env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            active.append(process)
            try:
                return process.wait()
            finally:
                active.remove(process)

    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        spec = root / 'spec.json'
        if not spec.exists() and call([cli, 'generate', spec], 'generate'):
            return 1
        controls = root / 'estimator_validation.json'
        if call([cli, 'validate-estimators', controls], 'validation'):
            return 1
        simulation = root / 'simulation'
        while not interrupted:
            command = ([simulation_cli, 'resume', simulation] if simulation.exists()
                       else [simulation_cli, 'run', spec, simulation])
            # Reserve 0.5 GiB disk for analysis/reports and 0.5 GiB RAM for this driver.
            code = call([*command, '--workers', workers, '--minutes', minutes,
                         '--memory-gib', 5.5, '--output-gib', 4.5], 'simulation')
            if code == 0:
                break
            if code != 2 or interrupted:
                return code
            with sqlite3.connect((simulation / 'ledger.sqlite').as_uri() + '?mode=ro', uri=True) as db:
                reason = db.execute('SELECT stop_reason FROM invocations ORDER BY rowid DESC LIMIT 1').fetchone()[0]
            if reason != 'time_limit':
                print(f'Stopped simulation: {reason}', flush=True)
                return code
        while not interrupted:
            code = call([cli, 'analyze', simulation, '--workers', workers, '--minutes', minutes,
                         '--memory-gib', 5.5], 'analysis')
            if code == 0:
                break
            reports = sorted((root / 'analysis').glob('*/last_invocation.json'), key=lambda p: p.stat().st_mtime)
            if code != 2 or interrupted or not reports or json.loads(reports[-1].read_text())['stop_reason'] != 'time_limit':
                return code
        return call([cli, 'render', simulation], 'render') if not interrupted else 2
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    study = sub.add_parser('study', help='complete all simulations, analysis, and figures in resumable batches')
    study.add_argument('directory', type=Path, help='dedicated atlas root; simulation lives underneath it')
    study.add_argument('--minutes', type=float, default=120)
    study.add_argument('--workers', type=int, default=2)
    generate = sub.add_parser('generate', help='write the complete schema-1 experiment specification')
    generate.add_argument('destination', type=Path)
    analyze = sub.add_parser('analyze', help='resume bounded analysis of completed simulation artifacts')
    analyze.add_argument('directory', type=Path, help='task-2 experiment directory, normally ATLAS/simulation')
    analyze.add_argument('--workers', type=int, default=2)
    analyze.add_argument('--minutes', type=float, default=120)
    analyze.add_argument('--memory-gib', type=float, default=6)
    analyze.add_argument('--output-gib', type=float, default=5)
    analyze.add_argument('--min-free-gib', type=float, default=10)
    analyze.add_argument('--timeout', type=float, default=300)
    analyze.add_argument('--retry-failed', action='store_true')
    render = sub.add_parser('render', help='write coverage, per-run results, summaries, and figures')
    render.add_argument('directory', type=Path)
    render.add_argument('--analysis-id', help='explicit historical analysis identity')
    controls = sub.add_parser('validate-estimators', help='save known-system and stochastic-control diagnostics')
    controls.add_argument('destination', type=Path)
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('request', type=Path)
    worker.add_argument('--memory-bytes', type=int, required=True)
    worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.run_experiment import bootstrap
    bootstrap(args.memory_bytes if args.command == '_worker' else 768 * 1024**2,
              2 * 1024**2 if args.command == '_worker' else None,
              getattr(args, 'parent_pid', None))
    if args.command == 'study':
        return complete_study(args.directory, args.minutes, args.workers)
    from src.atlas import make_spec, validation_controls
    from src.experiment_artifacts import expand_spec, write_json
    from src.atlas_runner import analysis_fingerprint, analyze_experiment, analyze_worker, atomic_json
    from src.experiments import Limits
    if args.command == 'generate':
        spec = make_spec()
        count = len(expand_spec(spec))
        write_json(args.destination, spec)
        print(json.dumps({'cells': len(spec['cells']), 'unique_runs': count, 'path': str(args.destination)}))
        return 0
    if args.command == 'validate-estimators':
        atomic_json(args.destination, {'fingerprint': analysis_fingerprint(), **validation_controls()})
        return 0
    if args.command == '_worker':
        return analyze_worker(args.request)
    if args.command == 'analyze':
        limits = Limits(**{key: getattr(args, key) for key in ('workers', 'minutes', 'memory_gib', 'output_gib', 'min_free_gib')})
        return analyze_experiment(args.directory, limits=limits, timeout=args.timeout, retry_failed=args.retry_failed)
    from src.atlas_plotting import render_atlas
    render_atlas(args.directory, args.analysis_id)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
