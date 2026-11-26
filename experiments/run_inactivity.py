"""Task 6: bounded, resumable inactivity and factorial recovery study."""

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
    generate.add_argument('--seeds', type=int, nargs='+', default=list(range(600, 650)))
    generate.add_argument('--days', type=int, default=10240)
    generate.add_argument('--threshold', type=int, default=100)
    generate.add_argument('--continuation-days', type=int, default=1000)
    generate.add_argument('--futures', type=int, default=5)
    generate.add_argument('--tail-days', type=int, default=100)
    generate.add_argument('--resamples', type=int, default=2000)
    for name in ('study', 'analyze', 'render'):
        command = sub.add_parser(name)
        command.add_argument('directory', type=Path)
        command.add_argument('--workers', type=int, default=2)
        command.add_argument('--minutes', type=float, default=120)
        command.add_argument('--memory-gib', type=float, default=6)
        command.add_argument('--output-gib', type=float, default=5)
        command.add_argument('--min-free-gib', type=float, default=10)
        if name == 'study':
            command.add_argument('--retry-failed', action='store_true')
            command.add_argument('--one-batch', action='store_true')
    sub.add_parser('status').add_argument('directory', type=Path)
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('request', type=Path)
    worker.add_argument('--memory-bytes', type=int, required=True)
    worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-inactivity-matplotlib')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.run_experiment import bootstrap
    bootstrap(args.memory_bytes if args.command == '_worker' else 768*1024**2,
              32*1024**2 if args.command == '_worker' else None, getattr(args, 'parent_pid', None))
    from src.inactivity_study import generate_study, complete_study, analyze_study, study_status, worker as run_worker
    from src.experiments import Limits
    if args.command == 'generate':
        study = generate_study(args.atlas, args.directory, seeds=args.seeds, days=args.days,
                               threshold=args.threshold, continuation_days=args.continuation_days,
                               futures=args.futures, tail_days=args.tail_days, resamples=args.resamples)
        print(json.dumps({'directory': str(args.directory), 'references': 16*len(study['seeds'])}))
        return 0
    if args.command == 'status':
        print(json.dumps(study_status(args.directory), indent=2))
        return 0
    if args.command == '_worker':
        return run_worker(args.request)
    limits = Limits(**{key: getattr(args, key) for key in asdict(Limits())})
    if args.command == 'study':
        return complete_study(args.directory, limits=limits, retry_failed=args.retry_failed, repeat=not args.one_batch)
    if args.command == 'analyze':
        result = analyze_study(args.directory, limits=limits)
    else:
        from src.inactivity_plotting import render_study
        result = render_study(args.directory, limits=limits)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
