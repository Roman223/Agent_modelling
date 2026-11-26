"""Logging neutrality, reconciliation, bounded export, and replay checks."""

from collections import defaultdict
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import numpy as np

from src import LogOptions, SCENARIOS, simulation
from src.asset import Asset
from src.bundles import save_bundle, replay_bundle, _json_safe
from src.order_book import OrderBook
from src.run_logging import RunLog
from src.traders import Trader, NoiseTrader, ChartistTrader, FundamentalistTrader
from test_core import assert_rng_equal


class LoggingTests(unittest.TestCase):
    def test_modes_preserve_all_scenarios_and_analysis_rng(self):
        cases = list(SCENARIOS.items()) + [
            (name, replace(SCENARIOS[name], chartist_alpha=0.3, fundamentalist_gamma=4,
                           closing_noise_scale=0, dividend_mode='constant', order_lifetime='day'))
            for name in ('baseline', 'evolution', 'circuit_breaker')
        ]
        for name, scenario in cases:
            baseline = None
            for level in ('off', 'summary', 'trades', 'decisions'):
                with self.subTest(scenario=name, level=level):
                    np.random.seed(18 if name == 'circuit_breaker' else 0)
                    result = simulation.simulate(scenario, days=102, log_options=LogOptions(level=level))
                    analysis = simulation.analyze_result(result)
                    snapshot = (result.asset.price_history, result.asset.dividend_history,
                                result.wealth_history, result.halt_events, result.evolution_log,
                                [(t.cash, t.assets, t.active, t.wealth_history) for t in result.traders],
                                _json_safe(asdict(analysis)))
                    if baseline is None:
                        baseline, state = snapshot, np.random.get_state()
                        sim_state = result.rng_state_after
                    else:
                        self.assertEqual(snapshot, baseline)
                        assert_rng_equal(self, state, np.random.get_state())
                        assert_rng_equal(self, sim_state, result.rng_state_after)
                    result.run_log.cleanup()

    def test_day_expiry_follows_summaries_and_prevents_next_day_fills(self):
        for lifetime in ('persistent', 'day'):
            for level in ('off', 'summary', 'decisions'):
                with self.subTest(lifetime=lifetime, level=level):
                    buyer, seller = NoiseTrader(2000, 10), NoiseTrader(0, 10)
                    buyer.make_decision = Mock(side_effect=[('buy', 101, 5), (None, None, None)])
                    seller.make_decision = Mock(side_effect=[('sell', 99, 2), ('sell', 99, 3)])
                    scenario = replace(SCENARIOS['baseline'], noise=2, chartists=0, fundamentalists=0,
                                       order_lifetime=lifetime, closing_noise_scale=0)
                    np.random.seed(0)
                    with patch('src.simulation._create_traders', return_value=[buyer, seller]):
                        result = simulation.simulate(scenario, days=2, log_options=LogOptions(level=level))
                    self.addCleanup(result.run_log.cleanup)
                    self.assertEqual(buyer.assets, 12 if lifetime == 'day' else 15)
                    if level == 'off':
                        self.assertEqual(list(result.run_log.iter_records('events')), [])
                        continue
                    steps = [r for r in result.run_log.iter_records('steps') if r['phase'] == 'trading']
                    self.assertEqual([r['volume'] for r in steps], [2, 0] if lifetime == 'day' else [2, 3])
                    self.assertEqual(steps[0]['price_formation']['proposed_price'], 100)
                    self.assertEqual(steps[0]['outstanding_buy_quantity'], 3)
                    expired = [r for r in result.run_log.iter_records('events') if r['event'] == 'orders_expired']
                    if lifetime == 'persistent':
                        self.assertEqual(expired, [])
                        continue
                    self.assertEqual(len(expired), 2)
                    for event, step in zip(expired, steps):
                        self.assertEqual(event['phase'], 'trading')
                        self.assertEqual(event['day'], step['day'])
                        for name in ('buy_orders', 'sell_orders', 'buy_quantity', 'sell_quantity'):
                            self.assertEqual(event[name], step['outstanding_' + name])

    def test_day_expiry_emits_no_event_when_all_orders_fill(self):
        buyer, seller = NoiseTrader(2000, 10), NoiseTrader(0, 10)
        buyer.make_decision = Mock(return_value=('buy', 101, 2))
        seller.make_decision = Mock(return_value=('sell', 99, 2))
        scenario = replace(SCENARIOS['baseline'], noise=2, chartists=0, fundamentalists=0,
                           order_lifetime='day')
        with patch('src.simulation._create_traders', return_value=[buyer, seller]):
            result = simulation.simulate(scenario, days=1)
        self.addCleanup(result.run_log.cleanup)
        self.assertEqual(list(result.run_log.iter_records('events')), [])

    def test_summaries_reconcile_to_trades_decisions_and_wealth(self):
        np.random.seed(4)
        result = simulation.simulate(days=12, log_options=LogOptions(level='decisions'))
        rows = lambda kind: list(result.run_log.iter_records(kind))
        steps = rows('steps')
        self.assertEqual(len(steps), 32)
        trades = defaultdict(list)
        decisions = defaultdict(list)
        for row in rows('trades'):
            trades[row['day']].append(row)
            before, after = row['before'], row['after']
            if row['buyer_id'] != row['seller_id']:
                self.assertAlmostEqual(before['buyer_cash'] - after['buyer_cash'], row['price'] * row['quantity'])
                self.assertAlmostEqual(after['seller_cash'] - before['seller_cash'], row['price'] * row['quantity'])
                self.assertEqual(after['buyer_assets'] - before['buyer_assets'], row['quantity'])
                self.assertEqual(before['seller_assets'] - after['seller_assets'], row['quantity'])
        for row in rows('decisions'):
            decisions[row['day']].append(row)
            self.assertTrue(row['reason'])
        for row in steps[20:]:
            fills = trades[row['day']]
            self.assertEqual(row['trade_count'], len(fills))
            self.assertEqual(row['volume'], sum(t['quantity'] for t in fills))
            self.assertAlmostEqual(row['notional'], sum(t['price'] * t['quantity'] for t in fills))
            self.assertEqual(row['return'], row['price'] / row['previous_price'] - 1)
            if fills:
                self.assertAlmostEqual(row['price_formation']['vwap'], row['notional'] / row['volume'])
        for row in rows('strategies'):
            day, kind = row['day'], row['type']
            self.assertEqual(row['buy_volume'], sum(t['quantity'] for t in trades[day] if t['buyer_type'] == kind))
            self.assertEqual(row['sell_volume'], sum(t['quantity'] for t in trades[day] if t['seller_type'] == kind))
            for side in ('buy', 'sell'):
                submitted = [d for d in decisions[day] if d['type'] == kind and d['submitted'] and d['order_type'] == side]
                self.assertEqual(row[side + '_orders'], len(submitted))
                self.assertEqual(row[side + '_quantity'], sum(d['quantity'] for d in submitted))
            if day == 11:
                group = [t for t in result.traders if t.active and t.type == kind]
                self.assertEqual(row['total_wealth'], sum(t.wealth_history[-1] for t in group))
        result.run_log.cleanup()

    def test_filters_caps_and_replay_export(self):
        np.random.seed(0)
        options = LogOptions(level='decisions', trader_ids=(0, 50), start_day=2, end_day=4,
                             max_detail_records=5)
        result = simulation.simulate(days=6, log_options=options)
        metadata = result.run_log.metadata()
        self.assertTrue(metadata['detail_truncated'])
        self.assertGreater(sum(metadata['omitted_by_cap'].values()), 0)
        self.assertEqual(metadata['records'].get('trades', 0) + metadata['records'].get('decisions', 0), 5)
        self.assertEqual(metadata['records']['steps'], 26)
        for kind in ('trades', 'decisions'):
            for row in result.run_log.iter_records(kind):
                self.assertTrue(2 <= row['day'] <= 4)
                if kind == 'decisions':
                    self.assertIn(row['trader_id'], (0, 50))
                else:
                    self.assertTrue({row['buyer_id'], row['seller_id']} & {0, 50})
        with tempfile.TemporaryDirectory() as root:
            directory = save_bundle(result, root, seed=0)
            replay = replay_bundle(directory)
            self.assertEqual(replay.run_log.metadata(), metadata)
            for kind in metadata['records']:
                with gzip.open(directory / 'logs' / f'{kind}.jsonl.gz', 'rt') as stream:
                    exported = [json.loads(line) for line in stream]
                self.assertEqual(exported, list(result.run_log.iter_records(kind)))
                self.assertEqual(exported, list(replay.run_log.iter_records(kind)))
            replay.run_log.cleanup()
        result.run_log.cleanup()

    def test_byte_cap_and_invalid_options(self):
        result = simulation.simulate(days=1, log_options=LogOptions(level='decisions', max_detail_bytes=1))
        self.assertTrue(result.run_log.metadata()['detail_truncated'])
        self.assertEqual(list(result.run_log.iter_records('decisions')), [])
        self.assertEqual(result.run_log.metadata()['detail_bytes_uncompressed'], 0)
        result.run_log.cleanup()
        for values in ({'level': 'debug'}, {'start_day': -1}, {'start_day': 3, 'end_day': 2},
                       {'trader_ids': (-1,)}, {'max_detail_records': True}, {'max_detail_bytes': -1}):
            with self.assertRaises(ValueError):
                LogOptions(**values)

    def test_warmup_halts_and_random_evolution(self):
        np.random.seed(18)
        result = simulation.simulate(SCENARIOS['circuit_breaker'], days=6)
        events = list(result.run_log.iter_records('events'))
        trigger = next(e for e in events if e['event'] == 'circuit_breaker_trigger')
        self.assertEqual(trigger['phase'], 'warmup')
        self.assertGreater(trigger['absolute_return'], trigger['threshold'])
        self.assertEqual([e['day'] for e in events if e['event'] == 'trading_skipped'], [0, 1, 2, 3])
        self.assertEqual(next(e['day'] for e in events if e['event'] == 'circuit_breaker_resume'), 4)
        result.run_log.cleanup()
        np.random.seed(0)
        result = simulation.simulate(SCENARIOS['evolution'], days=102)
        changes = [e for e in result.run_log.iter_records('events') if e['event'] == 'evolution']
        self.assertEqual(len(changes), 3)
        for event in changes:
            old, new = result.traders[event['old_trader_id']], result.traders[event['new_trader_id']]
            self.assertFalse(old.active)
            self.assertEqual(event['old_type'], old.type)
            self.assertEqual(event['new_type'], new.type)
            self.assertEqual(event['selection'], 'uniform_random_among_active_source_type')
            self.assertIsNone(event['selection_score'])
        self.assertEqual(changes[-1]['active_counts'], {'noise': 48, 'chartist': 21, 'fundamentalist': 31})
        result.run_log.cleanup()

    def test_decision_explanations(self):
        asset = Asset()
        noise, chartist, fundamentalist = NoiseTrader(0, 0), ChartistTrader(10, 1), FundamentalistTrader(100, 1)
        trace = {}
        with patch('src.traders.np.random.rand', return_value=0.9) as draw:
            self.assertEqual(noise.make_decision(asset, trace=trace), (None, None, None))
            draw.assert_called_once()
        self.assertEqual(trace['reason'], 'no_cash')
        trace = {}
        chartist.make_decision(asset, trace=trace)
        self.assertEqual(trace['reason'], 'insufficient_history')
        asset.update_price(100)
        asset.dividend_history.append(0.016)
        trace = {}
        fundamentalist.make_decision(asset, trace=trace)
        self.assertEqual(trace['reason'], 'excess_return_below_threshold')
        self.assertAlmostEqual(trace['excess_return'], 0.016 / 100 - fundamentalist.rf)

    def test_matching_records_actual_partial_fills_and_persistence(self):
        log = RunLog(LogOptions(level='trades'))
        log.phase = 'trading'
        buyer, seller = Trader('noise', 1000, 0), Trader('chartist', 0, 7)
        log.register([buyer, seller])
        book, asset = OrderBook(), Asset()
        book.add_order('buy', 101, 10, id(buyer))
        book.add_order('sell', 99, 4, id(seller))
        book.match_orders([buyer, seller], asset, on_trade=log.trade)
        log.day = 1
        book.add_order('sell', 99, 3, id(seller))
        book.match_orders([buyer, seller], asset, on_trade=log.trade)
        log.close()
        records = list(log.iter_records('trades'))
        self.assertEqual([(r['day'], r['quantity']) for r in records], [(0, 4), (1, 3)])
        self.assertEqual(records[1]['after']['buyer_assets'], 7)
        log.cleanup()

    def test_failure_cleans_temp_logs_and_prevents_complete_bundle(self):
        cleanup = RunLog.cleanup
        cleaned = []

        def spy(log):
            cleaned.append(log._directory.name)
            cleanup(log)

        with patch.object(RunLog, 'cleanup', spy), patch.object(NoiseTrader, 'make_decision', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                simulation.simulate(days=1)
        self.assertEqual(len(cleaned), 1)
        self.assertFalse(Path(cleaned[0]).exists())
        result = simulation.simulate(days=1)
        with tempfile.TemporaryDirectory() as root, patch.object(RunLog, 'export', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                save_bundle(result, root)
            self.assertFalse(next(Path(root).iterdir()).joinpath('COMPLETE').exists())
        result.run_log.cleanup()

    def test_cli_filters_and_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            base = [sys.executable, 'experiments/run_scenario.py', 'baseline', '--days', '3',
                    '--seed', '0', '--no-show']
            for flags in (['--log-level', 'trades'], ['--bundle', root, '--log-trader', '0'],
                          ['--bundle', root, '--log-level', 'decisions', '--log-end-day', '-1']):
                process = subprocess.run(base + flags, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertEqual(list(Path(root).iterdir()), [])
            process = subprocess.run(base + ['--bundle', root, '--log-level', 'decisions',
                                             '--log-trader', '0', '--log-start-day', '1', '--log-end-day', '1'],
                                     cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            folder = next(Path(root).iterdir())
            self.assertTrue((folder / 'COMPLETE').exists())
            with gzip.open(folder / 'logs' / 'decisions.jsonl.gz', 'rt') as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual([(r['day'], r['trader_id']) for r in records], [(1, 0)])


if __name__ == '__main__':
    unittest.main()
