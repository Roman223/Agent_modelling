"""Run Task 5's bounded checkpoint portfolio-detail study."""

import argparse
from dataclasses import asdict
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
    generate.add_argument('--development-seeds', type=int, nargs='+', default=list(range(300, 310)))
    generate.add_argument('--validation-seeds', type=int, nargs='+', default=list(range(310, 360)))
    generate.add_argument('--days', type=int, default=2000)
    generate.add_argument('--checkpoints', type=int, nargs='+', default=[100, 500, 1000])
    generate.add_argument('--continuation-days', type=int, default=1000)
    generate.add_argument('--levels', nargs='+', default=['1', '2', '4', '8', '16', 'full'])
    generate.add_argument('--horizons', type=int, nargs='+', default=[1, 20, 100, 1000])
    for name in ('run', 'resume'):
        cmd = sub.add_parser(name)
        cmd.add_argument('directory', type=Path)
        cmd.add_argument('--workers', type=int, default=2)
        cmd.add_argument('--minutes', type=float, default=120)
        cmd.add_argument('--memory-gib', type=float, default=6)
        cmd.add_argument('--output-gib', type=float, default=5)
        cmd.add_argument('--min-free-gib', type=float, default=10)
        cmd.add_argument('--retry-failed', action='store_true')
    sub.add_parser('status').add_argument('directory', type=Path)
    render = sub.add_parser('render')
    render.add_argument('directory', type=Path)
    render.add_argument('--bootstrap-resamples', type=int, default=2000)
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('request', type=Path)
    worker.add_argument('--memory-bytes', type=int, required=True)
    worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-heterogeneity-matplotlib')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.run_experiment import bootstrap
    bootstrap(args.memory_bytes if args.command == '_worker' else (4*1024**3 if args.command == 'render' else 768*1024**2),
              32*1024**2 if args.command == '_worker' else None, getattr(args, 'parent_pid', None))
    from src.heterogeneity_study import generate_study, execute_study, study_status, worker as run_worker
    from src.experiments import Limits
    if args.command == 'generate':
        study = generate_study(args.atlas, args.directory, development=args.development_seeds,
                               validation=args.validation_seeds, days=args.days, checkpoints=args.checkpoints,
                               continuation_days=args.continuation_days, horizons=args.horizons,
                               levels=[x if x == 'full' else int(x) for x in args.levels])
        print(json.dumps({'directory': str(args.directory), 'configurations': len(study['configurations'])}))
        return 0
    if args.command == 'status':
        print(json.dumps(study_status(args.directory), indent=2))
        return 0
    if args.command == '_worker':
        return run_worker(args.request)
    if args.command == 'render':
        from src.heterogeneity_plotting import render_study
        print(json.dumps(render_study(args.directory, resamples=args.bootstrap_resamples)))
        return 0
    return execute_study(args.directory, limits=Limits(**{k: getattr(args, k) for k in asdict(Limits())}),
                         retry_failed=args.retry_failed)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
