"""Focused correctness checks: .venv/bin/python tests/test_core.py."""

import contextlib
import copy
import io
import os
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import simulation
from src.asset import Asset
from src.order_book import OrderBook
from src.scenarios import SCENARIOS
from src.traders import Trader
from test_legacy_equivalence import legacy_namespace


def assert_rng_equal(test, before, after):
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


class MatchingTests(unittest.TestCase):
    def test_partial_fill_conservation_and_priority(self):
        buyers = [Trader('noise', 1000, 0), Trader('noise', 1000, 0)]
        seller = Trader('noise', 0, 7)
        traders = buyers + [seller]
        book = OrderBook()
        for buyer in buyers:
            book.add_order('buy', 101, 5, id(buyer))
        book.add_order('sell', 99, 7, id(seller))
        trades = book.match_orders(traders, Asset())
        self.assertEqual([t[1] for t in trades], [5, 2])
        self.assertIs(trades[0][2], buyers[0])
        self.assertEqual(book.buy_orders, [(101, 3, id(buyers[1]))])
        self.assertEqual(book.sell_orders, [])
        self.assertAlmostEqual(sum(t.cash for t in traders), 2000)
        self.assertEqual(sum(t.assets for t in traders), 7)

    def test_insufficient_resources_and_persistence(self):
        buyer, seller = Trader('noise', 250, 0), Trader('noise', 0, 1)
        book = OrderBook()
        book.add_order('buy', 100, 5, id(buyer))
        book.add_order('sell', 100, 5, id(seller))
        trades = book.match_orders([buyer, seller], Asset())
        self.assertEqual([t[1] for t in trades], [1])
        self.assertEqual((buyer.cash, buyer.assets, seller.cash, seller.assets), (150, 1, 100, 0))
        book.add_order('sell', 200, 1, id(seller))
        remaining = copy.deepcopy((book.buy_orders, book.sell_orders))
        self.assertEqual(book.match_orders([buyer, seller], Asset()), [])
        self.assertEqual((book.buy_orders, book.sell_orders), remaining)

    def test_historical_order_stream(self):
        legacy = legacy_namespace('baseline')['OrderBook']
        traces = []
        for cls in (legacy, OrderBook):
            np.random.seed(7)
            traders = [Trader('noise', 5000, 30) for _ in range(4)]
            ids = {id(t): i for i, t in enumerate(traders)}
            book, asset, trace = cls(), Asset(), []
            for day in range(5):
                for i in range(20):
                    book.add_order('buy' if i % 2 else 'sell', 98 + (i % 5),
                                   1 + i % 7, id(traders[i % 4]))
                trades = book.match_orders(traders, asset)
                trace.append(([(p, q, ids[id(b)], ids[id(s)]) for p, q, b, s in trades],
                              [(p, q, ids[t]) for p, q, t in book.buy_orders],
                              [(p, q, ids[t]) for p, q, t in book.sell_orders],
                              [(t.cash, t.assets) for t in traders], list(asset.price_history)))
            traces.append((trace, np.random.get_state()))
        self.assertEqual(traces[0][0], traces[1][0])
        assert_rng_equal(self, traces[0][1], traces[1][1])


class SimulationTests(unittest.TestCase):
    def test_result_isolation_and_cached_analysis(self):
        np.random.seed(0)
        global_before = copy.deepcopy(simulation.wealth_history)
        first = simulation.simulate(days=120)
        saved = copy.deepcopy(first.wealth_history)
        analysis = simulation.analyze_result(first)
        state = np.random.get_state()
        self.assertIs(simulation.analyze_result(first), analysis)
        with contextlib.redirect_stdout(io.StringIO()):
            simulation.print_analysis(analysis)
        assert_rng_equal(self, state, np.random.get_state())
        simulation.simulate(days=2)
        self.assertEqual(first.wealth_history, saved)
        self.assertEqual(simulation.wealth_history, global_before)

    def test_compatibility_global_identity_and_quiet_halt(self):
        from src import wealth_history
        np.random.seed(18)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            asset, _ = simulation.run_simulation(SCENARIOS['circuit_breaker'], days=10, verbose=False)
        self.assertEqual(output.getvalue(), '')
        self.assertIs(wealth_history, simulation.wealth_history)
        self.assertEqual(asset.halt_events, [0, 1, 2, 3])
        self.assertEqual(wealth_history['days'], [0, 9])

    def test_halt_countdown(self):
        messages = []
        asset = Asset(circuit_breaker=True, log=messages.append)
        self.assertFalse(asset.update_price(130))
        self.assertEqual(asset.current_price, 100)
        self.assertEqual([asset.check_halt_status() for _ in range(5)], [True, True, True, True, False])
        self.assertEqual(len(messages), 2)

    def test_evolution_and_event_histories(self):
        np.random.seed(0)
        result = simulation.simulate(SCENARIOS['evolution'], days=102)
        self.assertEqual([e['day'] for e in result.evolution_log], [100, 100, 100])
        self.assertEqual(len(result.traders), 103)
        self.assertEqual(result.trader_counts_history[-1],
                         {'day': 101, 'noise': 48, 'chartist': 21, 'fundamentalist': 31})

    def test_invalid_inputs_do_not_consume_randomness(self):
        cases = [(SCENARIOS['baseline'], d) for d in (0, -1, True, 1.5)]
        for fields in ({'noise': -1}, {'chartists': True}, {'chartist_window': 0},
                       {'fundamentalists': 1.5}, {'noise': 0, 'chartists': 0, 'fundamentalists': 0}):
            # Invalid scenarios must be rejected at construction or at the API boundary.
            state = np.random.get_state()
            with self.assertRaises(ValueError):
                simulation.simulate(replace(SCENARIOS['baseline'], **fields), days=1)
            assert_rng_equal(self, state, np.random.get_state())
        for scenario, days in cases:
            state = np.random.get_state()
            with self.assertRaises(ValueError):
                simulation.simulate(scenario, days=days)
            assert_rng_equal(self, state, np.random.get_state())


class MetricTests(unittest.TestCase):
    def test_explicit_arguments_and_constant_returns(self):
        from src import metrics
        prices = np.full(120, 100, dtype=np.float32)
        with patch.object(metrics, 'sampen', return_value=0.0) as estimator:
            result = metrics.metric_result('sampen', prices)
            self.assertEqual(result.parameters['tolerance'], 0.01)
            self.assertEqual(result.return_count, 119)
            self.assertEqual(estimator.call_args.kwargs, {'emb_dim': 3, 'tolerance': 0.01})
        with patch.object(metrics, 'lyap_r', return_value=0.0) as estimator:
            metrics.calculate_lyapunov_exponent(prices, emb_dim=None)
            self.assertIsNone(estimator.call_args.kwargs['emb_dim'])

    def test_diagnostics_and_scalar_compatibility(self):
        from src import metrics
        short = metrics.metric_result('sampen', [100.0] * 10)
        self.assertEqual(short.status, 'insufficient_input')
        self.assertTrue(np.isnan(short.value))
        prices = np.exp(np.linspace(4.0, 4.1, 120))
        with patch.object(metrics, 'hurst_rs', side_effect=RuntimeError('controlled failure')) as estimator:
            failed = metrics.metric_result('hurst', prices)
        self.assertEqual(estimator.call_count, 1)
        self.assertEqual(failed.status, 'estimator_error')
        self.assertIn('controlled failure', failed.reason)
        with patch.object(metrics, 'sampen', return_value=np.inf):
            result = metrics.metric_result('sampen', prices)
            self.assertEqual(result.status, 'non_finite')
            self.assertEqual(metrics.calculate_sample_entropy(prices), np.inf)
        self.assertIs(metrics.calculate_kolmogorov_entropy, metrics.calculate_sample_entropy)


if __name__ == '__main__':
    unittest.main()
