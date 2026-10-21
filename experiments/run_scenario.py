"""Run one market scenario, print the analysis, and plot price and wealth.

    python experiments/run_scenario.py baseline
    python experiments/run_scenario.py evolution --days 500 --seed 42 --save results --no-show
    python experiments/run_scenario.py --list
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib

# Select a noninteractive backend before pyplot is imported, including by src.plotting.
if '--no-show' in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from src import simulation
from src.plotting import plot_price, plot_trader_counts, plot_wealth_histograms, plot_wealth_history
from src.scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenario', nargs='?', choices=list(SCENARIOS))
    parser.add_argument('--list', action='store_true', help='list scenarios and exit')
    parser.add_argument('--days', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None, help='np.random.seed; unseeded if omitted')
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--save', metavar='DIR', default=None,
                        help='save PNGs as DIR/<scenario>_*.png (relative paths are under the repo root)')
    output.add_argument('--bundle', metavar='DIR', default=None,
                        help='save a unique full run bundle under DIR (relative to the repo root)')
    parser.add_argument('--no-show', action='store_true', help="don't open matplotlib windows")
    args = parser.parse_args()

    if args.days <= 0:
        parser.error('--days must be a positive integer')
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error('--seed must be an integer between 0 and 4294967295')

    if args.list:
        for s in SCENARIOS.values():
            print(f'{s.name:22s} {s.noise}/{s.chartists}/{s.fundamentalists}  {s.title}')
        return
    if args.scenario is None:
        parser.error('scenario is required')

    scenario = SCENARIOS[args.scenario]
    try:
        scenario.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.seed is not None:
        np.random.seed(args.seed)
    print(f'SCENARIO: {scenario.title} ({scenario.noise} / {scenario.chartists} / {scenario.fundamentalists})')
    if args.bundle:
        from src.bundles import save_bundle
        result = simulation.simulate(scenario, days=args.days, verbose=True)
        simulation.print_analysis(simulation.analyze_result(result))
        directory = save_bundle(result, os.path.join(REPO_ROOT, args.bundle), seed=args.seed,
                                invocation=sys.argv, show=not args.no_show)
        print(f'Run bundle: {directory}')
        return
    asset, traders = simulation.run_simulation(scenario, days=args.days)

    save_dir = os.path.join(REPO_ROOT, args.save) if args.save else None

    def path(kind):
        return os.path.join(save_dir, f'{scenario.name}_{kind}.png') if save_dir else None

    plot_price(asset, title=f'{scenario.title}: asset price', show_halts=scenario.halt_check,
               save_path=path('price'))
    plot_wealth_history(simulation.wealth_history, save_path=path('wealth'))
    plot_wealth_histograms(traders, save_path=path('wealth_hist'))
    if scenario.evolution:
        plot_trader_counts(asset, save_path=path('trader_counts'))
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
