"""Run bounded, resumable research experiments (Linux only)."""

import argparse
import ctypes
import os
from pathlib import Path
import resource
import signal
import sqlite3
import sys


def bootstrap(memory_bytes, file_bytes=None, parent_pid=None):
    """Apply limits before src/NumPy imports or model allocation."""
    if sys.platform != 'linux' or not hasattr(resource, 'RLIMIT_AS'):
        raise RuntimeError('Experiment execution requires Linux RLIMIT_AS and /proc')
    if parent_pid is not None:
        # Kill workers even when the coordinator dies without running its finally block.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Could not set parent-death signal')
        if os.getppid() != parent_pid:
            raise RuntimeError('Experiment coordinator exited during worker startup')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    # The coordinator's children need to raise their inherited soft limit.
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes if parent_pid else hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if file_bytes is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate', help='resolve and deduplicate a JSON specification')
    validate.add_argument('spec', type=Path)
    for name in ('run', 'resume'):
        command = sub.add_parser(name)
        if name == 'run':
            command.add_argument('spec', type=Path)
        command.add_argument('directory', type=Path)
        command.add_argument('--workers', type=int, default=2)
        command.add_argument('--minutes', type=float, default=120)
        command.add_argument('--memory-gib', type=float, default=6)
        command.add_argument('--output-gib', type=float, default=5)
        command.add_argument('--min-free-gib', type=float, default=10)
        if name == 'resume':
            command.add_argument('--retry-failed', action='store_true')
    sub.add_parser('status').add_argument('directory', type=Path)
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('request', type=Path)
    worker.add_argument('--memory-bytes', type=int, required=True)
    worker.add_argument('--file-bytes', type=int, required=True)
    worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    try:
        bootstrap(args.memory_bytes if args.command == '_worker' else 768 * 1024**2,
                  getattr(args, 'file_bytes', None), getattr(args, 'parent_pid', None))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.experiments import cli_main
        return cli_main(args)
    except (ValueError, RuntimeError, OSError, MemoryError, sqlite3.Error) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
