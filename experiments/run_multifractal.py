"""Generate, validate, execute/resume, inspect, and render the Task 7 pilot."""

import argparse
import json
import os
from pathlib import Path
import resource
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('generate')
    generate.add_argument('directory', type=Path)
    generate.add_argument('--atlas', type=Path, default=Path('results/task3_atlas_20261027'))
    generate.add_argument('--indices', type=Path, default=Path('data'))
    for name in ('study', 'analyze'):
        command = sub.add_parser(name)
        command.add_argument('directory', type=Path)
        command.add_argument('--workers', type=int, default=2)
        command.add_argument('--minutes', type=float, default=120)
        command.add_argument('--memory-gib', type=float, default=6)
        command.add_argument('--output-gib', type=float, default=5)
        command.add_argument('--min-free-gib', type=float, default=10)
        command.add_argument('--retry-failed', action='store_true')
    for name in ('validate-estimators', 'status', 'render'):
        sub.add_parser(name).add_argument('directory', type=Path)
    for name in ('_worker', '_simulation'):
        worker = sub.add_parser(name, help=argparse.SUPPRESS)
        worker.add_argument('request', type=Path)
        worker.add_argument('--memory-bytes', type=int, required=True)
        worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-task7-matplotlib')
    from experiments.run_experiment import bootstrap
    bootstrap(args.memory_bytes if args.command.startswith('_') else 768*1024**2,
              16*1024**2 if args.command == '_worker' else None,
              getattr(args, 'parent_pid', None))
    if args.command == '_simulation':
        # Keep the coordinator small while allowing its workers to raise their
        # inherited soft limit within the enclosing simulation budget.
        resource.setrlimit(resource.RLIMIT_AS, (768*1024**2, args.memory_bytes))
    from src.multifractal_study import (analyze_study, complete_study, generate_study,
                                       simulation_worker, study_status, validate_study, worker)
    from src.experiments import Limits, experiment_lock
    if args.command == '_worker':
        return worker(args.request)
    if args.command == '_simulation':
        return simulation_worker(args.request)
    if args.command == 'generate':
        manifest = generate_study(args.directory, args.atlas, indices=args.indices)
        print(json.dumps({'analysis_id': manifest['analysis_id'], 'simulations': 15, 'indices': 5}))
        return 0
    if args.command == 'validate-estimators':
        with experiment_lock(args.directory):
            result = validate_study(args.directory)
        print(json.dumps({'passed': result['passed'], 'checks': result['checks']}))
        return 0 if result['passed'] else 1
    if args.command == 'status':
        print(json.dumps(study_status(args.directory), indent=2))
        return 0
    if args.command == 'render':
        from src.multifractal_plotting import render_study
        render_study(args.directory)
        return 0
    limits = Limits(**{name: getattr(args, name) for name in
                      ('workers', 'minutes', 'memory_gib', 'output_gib', 'min_free_gib')})
    if limits.minutes > 120:
        raise ValueError('Batch duration cannot exceed 120 minutes')
    function = complete_study if args.command == 'study' else analyze_study
    return function(args.directory, limits=limits, retry_failed=args.retry_failed)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
