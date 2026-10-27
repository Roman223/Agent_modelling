"""Focused correctness checks: .venv/bin/python tests/test_core.py."""

import contextlib
import copy
import io
import os
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import simulation
from src.asset import Asset
from src.order_book import OrderBook
from src.scenarios import SCENARIOS, Scenario
from src.traders import Trader
from test_legacy_equivalence import legacy_namespace


def assert_rng_equal(test, before, after):
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


class MatchingTests(unittest.TestCase):
    def test_closing_noise_formula_floor_and_breaker(self):
        for scale, shock, price, breaker in ((0, 2, 100, False), (0.2, 0.5, 100, False),
                                             (0.2, -20, 100, False), (0, 2, 0.005, False),
                                             (1, 1, 100, True)):
            with self.subTest(scale=scale, shock=shock, price=price, breaker=breaker):
                buyer, seller = Trader('noise', 1000, 0), Trader('noise', 0, 2)
                book = OrderBook(closing_noise_scale=scale)
                book.add_order('buy', price, 2, id(buyer))
                book.add_order('sell', price, 2, id(seller))
                asset = Asset(circuit_breaker=breaker, log=lambda *args: None)
                formation = {}
                with patch('src.order_book.np.random.randn', return_value=shock) as draw:
                    trades = book.match_orders([buyer, seller], asset, on_price=formation.update)
                    draw.assert_called_once_with()
                self.assertEqual(len(trades), 1)
                proposal = price * (1 + scale * (1 / (1 + 2 / 1000)) * shock)
                self.assertEqual(formation['proposed_price_before_floor'], proposal)
                self.assertEqual(formation['proposed_price'], max(proposal, 0.01))
                self.assertEqual(formation['accepted'], not breaker)
                self.assertEqual(asset.current_price, 100 if breaker else max(proposal, 0.01))

    def test_noise_control_preserves_draw_count_and_no_fill_behavior(self):
        states = []
        for scale in (0, 0.05, 0.2):
            np.random.seed(37)
            buyer, seller = Trader('noise', 1000, 0), Trader('noise', 0, 2)
            book = OrderBook(closing_noise_scale=scale)
            asset = Asset()
            book.add_order('buy', 100, 2, id(buyer))
            before = np.random.get_state()
            with patch.object(asset, 'update_price', wraps=asset.update_price) as update:
                self.assertEqual(book.match_orders([buyer, seller], asset), [])
                update.assert_not_called()
            assert_rng_equal(self, before, np.random.get_state())
            book.add_order('sell', 100, 2, id(seller))
            book.match_orders([buyer, seller], asset)
            states.append(np.random.get_state())
        np.random.seed(37)
        np.random.randn()
        for state in states:
            assert_rng_equal(self, np.random.get_state(), state)

    def test_day_order_expiry_clears_partial_and_unmatched_orders(self):
        for lifetime in ('persistent', 'day'):
            with self.subTest(lifetime=lifetime):
                buyer, seller = Trader('noise', 2000, 0), Trader('noise', 0, 20)
                book = OrderBook(order_lifetime=lifetime)
                self.assertIsNone(book.expire_day_orders())
                book.add_order('buy', 101, 5, id(buyer))
                book.add_order('buy', 98, 4, id(buyer))
                book.add_order('sell', 99, 2, id(seller))
                book.add_order('sell', 105, 6, id(seller))
                book.match_orders([buyer, seller], Asset())
                before = copy.deepcopy((book.buy_orders, book.sell_orders))
                state = np.random.get_state()
                expired = book.expire_day_orders()
                assert_rng_equal(self, state, np.random.get_state())
                if lifetime == 'day':
                    self.assertEqual(expired, {'buy_orders': 2, 'sell_orders': 1,
                                               'buy_quantity': 7, 'sell_quantity': 6})
                    self.assertEqual((book.buy_orders, book.sell_orders), ([], []))
                    self.assertIsNone(book.expire_day_orders())
                else:
                    self.assertIsNone(expired)
                    self.assertEqual((book.buy_orders, book.sell_orders), before)

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


class ParameterTests(unittest.TestCase):
    def test_legacy_positional_constructor_and_keyword_only_controls(self):
        scenario = Scenario('custom', 'Custom', 2, 3, 4, 5, True, False, True)
        self.assertEqual(scenario.chartist_window, 5)
        self.assertTrue(scenario.evolution)
        self.assertEqual((scenario.chartist_alpha, scenario.fundamentalist_gamma,
                          scenario.closing_noise_scale, scenario.dividend_mode, scenario.order_lifetime),
                         (0.1, 2, 0.05, 'stochastic', 'persistent'))
        with self.assertRaises(TypeError):
            Scenario('custom', 'Custom', 2, 3, 4, 5, True, False, True, 0.2)
        with self.assertRaises(FrozenInstanceError):
            scenario.chartist_alpha = 0.2

    def test_control_boundaries_and_invalid_inputs_before_logs_or_rng(self):
        for values in ({'chartist_alpha': 1}, {'chartist_alpha': np.float64(1e-12)},
                       {'fundamentalist_gamma': np.int64(1)}, {'fundamentalist_gamma': 1e-12},
                       {'closing_noise_scale': 0}, {'closing_noise_scale': 1.5},
                       {'dividend_mode': 'constant'}, {'order_lifetime': 'day'}):
            replace(SCENARIOS['baseline'], **values).validate()
        invalid = []
        for name in ('chartist_alpha', 'fundamentalist_gamma', 'closing_noise_scale'):
            invalid.extend({name: value} for value in
                           (True, False, np.bool_(True), '0.1', None, 1j, [],
                            np.nan, np.inf, -np.inf, -0.1, 10**400))
        invalid.extend(({'chartist_alpha': 0}, {'chartist_alpha': 1.01},
                        {'fundamentalist_gamma': 0}))
        for name in ('dividend_mode', 'order_lifetime'):
            invalid.extend({name: value} for value in ('invalid', '', True, 1, None, [], np.array(['day'])))
        for values in invalid:
            with self.subTest(values=values), patch('src.simulation.RunLog') as run_log:
                before = np.random.get_state()
                with self.assertRaises(ValueError):
                    simulation.simulate(replace(SCENARIOS['baseline'], **values), days=1)
                run_log.assert_not_called()
                assert_rng_equal(self, before, np.random.get_state())

    def test_initial_and_evolved_traders_inherit_controls(self):
        scenario = replace(SCENARIOS['evolution'], chartist_window=5,
                           chartist_alpha=0.3, fundamentalist_gamma=4)
        np.random.seed(0)
        result = simulation.simulate(scenario, days=102)
        self.addCleanup(result.run_log.cleanup)
        self.assertEqual(len(result.traders), 103)
        for trader in result.traders:
            if trader.type == 'chartist':
                self.assertEqual(trader.alpha, 0.3)
            elif trader.type == 'fundamentalist':
                self.assertEqual(trader.gamma, 4)
        self.assertTrue(all(t.window == 5 for t in result.traders[:100] if t.type == 'chartist'))
        self.assertEqual([t.window for t in result.traders[100:] if t.type == 'chartist'], [20, 20])
        self.assertEqual([t.type for t in result.traders[100:]], ['chartist', 'chartist', 'fundamentalist'])
        self.assertEqual(SCENARIOS['evolution'].chartist_alpha, 0.1)
        self.assertEqual(SCENARIOS['evolution'].fundamentalist_gamma, 2)

    def test_configured_parameters_change_order_quantities(self):
        asset = Asset()
        asset.price_history = [90] * 19 + [100]
        asset.dividend_history = [0.016, 0.016]
        quantities = []
        for alpha, gamma in ((0.1, 2), (0.3, 4)):
            traders = simulation._create_traders(replace(
                SCENARIOS['baseline'], noise=0, chartists=1, fundamentalists=1,
                chartist_alpha=alpha, fundamentalist_gamma=gamma))
            with patch('src.traders.np.random.rand', return_value=0):
                orders = [trader.make_decision(asset) for trader in traders]
            self.assertEqual([order[0] for order in orders], ['buy', 'buy'])
            quantities.append([order[2] for order in orders])
        self.assertEqual(quantities, [[150, 5326], [450, 2413]])

    def test_constant_dividends_keep_both_random_draw_branches(self):
        for mode in ('stochastic', 'constant'):
            asset = Asset(dividend_mode=mode)
            with patch('src.asset.np.random.rand', side_effect=[0.005, 0.5]) as uniform, \
                    patch('src.asset.np.random.randn', side_effect=[2, 3, 4]) as normal:
                values = [asset.update_dividend(), asset.update_dividend()]
            self.assertEqual(uniform.call_count, 2)
            self.assertEqual(normal.call_count, 3)
            if mode == 'constant':
                self.assertEqual(values, [0.016, 0.016])
            else:
                first = 0.016 * (1 + 0.0008 + 0.0008 * 3 + 2 * 0.005)
                self.assertEqual(values, [first, first * (1 + 0.0008 + 0.0008 * 4)])
        states = []
        for mode in ('stochastic', 'constant'):
            np.random.seed(0)
            asset = Asset(dividend_mode=mode)
            for _ in range(200):
                asset.update_dividend()
            states.append(np.random.get_state())
        assert_rng_equal(self, *states)

    def test_constant_dividends_in_warmup_trading_and_halts(self):
        np.random.seed(18)
        result = simulation.simulate(replace(SCENARIOS['circuit_breaker'], dividend_mode='constant'), days=6)
        self.addCleanup(result.run_log.cleanup)
        self.assertEqual(result.asset.dividend_history, [0.016] * 27)
        self.assertEqual(result.halt_events, [0, 1, 2, 3])
        rows = list(result.run_log.iter_records('steps'))
        self.assertEqual(len(rows), 26)
        self.assertTrue(all(row['dividend'] == 0.016 for row in rows))


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
