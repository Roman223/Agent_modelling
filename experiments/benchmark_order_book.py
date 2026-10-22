"""Sequential order-book benchmark; run in the background with output redirected.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/benchmark_order_book.py

The legacy implementation is loaded from Git without changing the working tree.
Both simulation timings include the same legacy-compatible metric analysis.
"""

import contextlib
import argparse
import io
import json
import os
import resource
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))

import numpy as np

from src import simulation
from src.order_book import OrderBook
from src.scenarios import SCENARIOS
from test_legacy_equivalence import legacy_namespace


def benchmark(implementation):
    legacy = legacy_namespace('baseline')['OrderBook']
    orders = [(float(90 + (i * 37) % 21), 1, i) for i in range(10000)]
    for name, book_class in [(implementation, legacy if implementation == 'legacy' else OrderBook)]:
        start = time.perf_counter()
        book = book_class()
        for price, quantity, trader_id in orders:
            book.add_order('buy', price, quantity, trader_id)
        print(json.dumps({'implementation': name, 'insertion_seconds': time.perf_counter() - start}), flush=True)
        old_class = simulation.OrderBook
        simulation.OrderBook = book_class
        try:
            np.random.seed(0)
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                asset, _ = simulation.run_simulation(SCENARIOS['more_noise'], days=250, verbose=False)
            print(json.dumps({'implementation': name, 'scenario': 'more_noise', 'seed': 0,
                              'days': 250, 'seconds': time.perf_counter() - start,
                              'final_price': asset.current_price,
                              'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        finally:
            simulation.OrderBook = old_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=('legacy', 'current'))
    args = parser.parse_args()
    if args.implementation:
        benchmark(args.implementation)
    else:
        # Separate sequential processes make peak RSS comparable, without workers.
        for implementation in ('legacy', 'current'):
            subprocess.run([sys.executable, __file__, '--implementation', implementation], check=True)
