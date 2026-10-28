"""Generate, execute/resume, inspect, and render the bounded Task 4 pilot."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('generate')
    generate.add_argument('directory', type=Path)
    generate.add_argument('--atlas', type=Path, required=True)
    generate.add_argument('--historical-seeds', type=int, nargs='+', default=list(range(100, 105)))
    generate.add_argument('--fresh-seeds', type=int, nargs='+', default=list(range(110, 115)))
    generate.add_argument('--points', type=int, nargs='+', default=[16, 8, 32])
    for name in ('run', 'resume'):
        command = sub.add_parser(name)
        command.add_argument('directory', type=Path)
        command.add_argument('--workers', type=int, default=2)
        command.add_argument('--minutes', type=float, default=120)
        command.add_argument('--memory-gib', type=float, default=6)
        command.add_argument('--output-gib', type=float, default=5)
        command.add_argument('--min-free-gib', type=float, default=10)
        command.add_argument('--retry-failed', action='store_true')
    sub.add_parser('status').add_argument('directory', type=Path)
    sub.add_parser('render').add_argument('directory', type=Path)
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('request', type=Path)
    worker.add_argument('--memory-bytes', type=int, required=True)
    worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-reduction-matplotlib')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.run_experiment import bootstrap
    bootstrap(args.memory_bytes if args.command == '_worker' else 512*1024**2,
              8*1024**2 if args.command == '_worker' else None, getattr(args, 'parent_pid', None))
    from src.reduction_study import execute_study, generate_study, study_status, worker
    if args.command == 'generate':
        study = generate_study(args.atlas, args.directory, historical_seeds=args.historical_seeds,
                               fresh_seeds=args.fresh_seeds, points=args.points)
        print(json.dumps({'historical_runs': len(study['historical_inputs']), 'directory': str(args.directory)}))
        return 0
    if args.command == 'status':
        print(json.dumps(study_status(args.directory), indent=2))
        return 0
    if args.command == '_worker':
        return worker(args.request)
    if args.command == 'render':
        from src.reduction_plotting import render_study
        print(json.dumps(render_study(args.directory)))
        return 0
    from dataclasses import asdict
    from src.experiments import Limits
    return execute_study(args.directory, limits=Limits(**{key: getattr(args, key) for key in asdict(Limits())}),
                         retry_failed=args.retry_failed)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
