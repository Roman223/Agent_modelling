"""Standalone known-answer and operational checks for Task 6."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.asset import Asset
from src.order_book import OrderBook
from src.traders import NoiseTrader
from src.scenarios import Scenario, SCENARIOS
from src.simulation import simulate
from src.market_checkpoints import MarketCheckpoint, run_checkpointed, decode_rng
from src.inactivity import (ARMS, PRIMARY_FIELDS, MATCH_REASONS, REASONS, spells,
                            run_reference, run_continuation, intervene, with_future, InactivityObserver)
from src.inactivity_analysis import survival, incidence, factorial_effects, recovery_summary
from src.inactivity_study import (generate_study, reference_jobs, continuation_jobs, materialize,
                                  validate_artifact, read_arrays, fingerprint, load_study, complete_study, future_seeds)
from src.experiment_artifacts import write_json, read_json, digest, file_hash, expand_spec
from src.atlas import make_spec
from src.experiments import Limits


def rng_equal(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


class ScientificTests(unittest.TestCase):
    def test_exact_spell_boundaries_and_censoring(self):
        self.assertEqual(spells([1, 1]), [])
        p = spells([0, 0, 1, 0, 0, 0], threshold=2)
        self.assertEqual(p, [dict(onset=1, last_zero_day=2, duration=2, confirmation=2, recovery_day=3, right_censored=False),
                             dict(onset=4, last_zero_day=6, duration=3, confirmation=5, recovery_day=None, right_censored=True)])
        self.assertIsNone(spells(np.zeros(99))[0]['confirmation'])
        self.assertEqual(spells(np.zeros(100))[0]['confirmation'], 100)
        self.assertEqual(spells(np.zeros(250), threshold=250)[0]['confirmation'], 250)
        for volume in ([np.nan], [-1], [[0]]):
            with self.assertRaises(ValueError):
                spells(volume)

    def test_known_kaplan_meier_and_incidence(self):
        result = survival([1, 2, 2, 3], [True, True, False, True])
        np.testing.assert_allclose(result['survival'], [.75, .5, 0])
        self.assertEqual(result['at_risk'], [4, 3, 1])
        self.assertEqual(survival([5, 5], [False, False])['survival'], [1])
        self.assertEqual(survival([], [])['status'], 'not_applicable')
        self.assertGreater(incidence([False]*50)['high'], 0)
        self.assertLess(incidence([True]*50)['low'], 1)

    def test_factorial_pairing_and_future_replication(self):
        rng = np.random.default_rng(60)
        base = rng.uniform(0, .1, size=(20, 1, 1))
        bits = np.array([[int(b) for b in arm] for arm in ARMS])
        y = base + (bits @ np.array([.1, .2, .3]))[None, :, None]
        y = np.repeat(y, 5, axis=2)
        effect = factorial_effects(y, resamples=100)
        main = list(effect['factorial'].values())
        np.testing.assert_allclose([r['estimate'] for r in main], [.1, .2, .3, 0, 0, 0, 0], atol=1e-15)
        repeated = factorial_effects(np.repeat(y, 4, axis=2), resamples=100)
        self.assertAlmostEqual(effect['arms']['000']['low'], repeated['arms']['000']['low'])
        self.assertEqual(effect['arms']['000']['n'], 20)
        self.assertIn('degenerate_empirical_distribution', factorial_effects(np.zeros((5, 8, 5)))['arms']['000']['flags'])
        self.assertIsNone(factorial_effects(np.zeros((0, 8, 5)))['arms']['000']['estimate'])

    def test_self_trades_and_flat_prices_are_not_distinct_exchange(self):
        t = NoiseTrader(1000, 10)
        book = OrderBook(closing_noise_scale=0)
        book.add_order('buy', 100, 1, id(t))
        book.add_order('sell', 100, 1, id(t))
        a = Asset()
        trades = book.match_orders([t], a)
        self.assertEqual(len(trades), 1)
        self.assertIs(trades[0][2], trades[0][3])
        self.assertEqual(a.current_price, 100)
        outcome = recovery_summary({'volume': np.ones(100), 'distinct_volume': np.zeros(100)})
        self.assertEqual(outcome['tail_distinct_fraction'], 0)
        self.assertEqual(outcome['first_execution'], 1)
        self.assertIsNone(outcome['first_distinct_exchange'])
        self.assertEqual(spells(np.ones(100)), [])

    def test_matcher_reports_actual_discard_and_stop(self):
        buyer, seller = NoiseTrader(0, 1), NoiseTrader(1000, 1)
        book = OrderBook()
        book.add_order('buy', 101, 1, id(buyer))
        book.add_order('sell', 100, 1, id(seller))
        observed = {}
        book.match_orders([buyer, seller], Asset(), on_match=observed.update)
        self.assertEqual(observed['dropped_bids'], 1)
        self.assertEqual(observed['stop_reason'], 'empty_bids')


class DynamicsTests(unittest.TestCase):
    def test_observed_original_matches_all_eight_controls(self):
        for life in ('persistent', 'day'):
            for noise in (0, .05):
                for dividend in ('constant', 'stochastic'):
                    scenario = replace(SCENARIOS['baseline'], order_lifetime=life,
                                       closing_noise_scale=noise, dividend_mode=dividend)
                    np.random.seed(602)
                    ordinary = simulate(scenario, days=120)
                    end_rng = np.random.get_state()
                    np.random.seed(42)
                    before = np.random.get_state()
                    observed = run_reference(scenario, seed=602, days=120, continuation_days=20)
                    rng_equal(before, np.random.get_state())
                    legacy = run_checkpointed(scenario, seed=602, days=120)
                    np.testing.assert_array_equal(observed.arrays['price'], legacy.arrays['price'])
                    np.testing.assert_array_equal(observed.arrays['volume'], legacy.arrays['volume'])
                    np.testing.assert_array_equal(ordinary.asset.price_history, legacy.final.data()['asset']['price_history'])
                    rng_equal(end_rng, decode_rng(legacy.final.data()['rng_state']))
                    np.testing.assert_array_equal(observed.arrays['decision_counts'].sum(axis=2), np.tile([50, 20, 30], (120, 1)))

    def test_confirmed_checkpoint_and_validation_tail_replay(self):
        # Constant-dividend, fundamentalist-only market never submits an order.
        s = Scenario('idle', 'Idle', noise=0, chartists=0, fundamentalists=2, dividend_mode='constant', order_lifetime='day')
        result = run_reference(s, seed=600, days=12, threshold=10, continuation_days=8)
        self.assertEqual(result.checkpoint.completed_days, 10)
        self.assertEqual(len(result.arrays['day']), 12)  # excludes extended validation tail
        self.assertEqual(result.replay_arrays['day'].tolist(), list(range(11, 19)))
        self.assertEqual(result.checkpoint.data()['bids'], [])
        replay = run_continuation(result.checkpoint, days=8)
        self.assertEqual(replay.final_digest, result.replay_digest)
        for key in replay.arrays:
            np.testing.assert_array_equal(replay.arrays[key], result.replay_arrays[key])

    def test_rng_restoration_on_error(self):
        np.random.seed(3)
        before = np.random.get_state()
        with patch('src.inactivity.InactivityObserver.decision', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                run_reference(SCENARIOS['baseline'], seed=4, days=10)
        rng_equal(before, np.random.get_state())


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = run_checkpointed(Scenario('tiny', 'Tiny', noise=4, chartists=0, fundamentalists=0),
                                          seed=1, days=3).final

    def state(self, portfolios, *, price=100., inactive=()):
        data = self.checkpoint.data()
        data['asset']['current_price'] = price
        data['asset']['price_history'][-1] = price
        for i, (t, (cash, shares)) in enumerate(zip(data['traders'], portfolios)):
            t['cash'], t['assets'], t['active'] = float(cash), shares, i not in inactive
        return MarketCheckpoint.from_data(data)

    def test_largest_remainder_known_answer_and_stable_ties(self):
        cp = self.state([(100, 0), (100, 0), (0, 1), (0, 1)])
        result, receipt = intervene(cp, '001')
        data = result.data()
        self.assertEqual([t['assets'] for t in data['traders']], [1, 1, 0, 0])
        self.assertEqual([t['cash'] for t in data['traders']], [0, 0, 100, 100])
        self.assertEqual(receipt['cash_residual'], 0)
        self.assertEqual(receipt['max_wealth_residual'], 0)

    def test_all_arms_conserve_and_do_not_mutate_checkpoint_or_rng(self):
        cp = self.state([(1, 2), (9999, 0), (0, 20), (100, 1)], price=3.14159, inactive=(3,))
        original = cp.payload
        np.random.seed(2)
        rng = np.random.get_state()
        for arm in ARMS:
            result, receipt = intervene(cp, arm)
            a, b = cp.data(), result.data()
            self.assertEqual(a['traders'][3], b['traders'][3])
            self.assertEqual(a['rng_state'], b['rng_state'])
            self.assertEqual(a['asset']['dividend_history'], b['asset']['dividend_history'])
            self.assertEqual(len(a['asset']['price_history']), len(b['asset']['price_history']))
            self.assertEqual(sum(t['assets'] for t in a['traders']), sum(t['assets'] for t in b['traders']))
            for ta, tb in zip(a['traders'], b['traders']):
                self.assertGreaterEqual(tb['cash'], 0)
                self.assertEqual(tb['assets'], int(tb['assets']))
                self.assertAlmostEqual(ta['cash']+ta['assets']*3.14159, tb['cash']+tb['assets']*3.14159, places=8)
                self.assertEqual(ta['wealth_history'], tb['wealth_history'])
            if arm[0] == '1':
                self.assertFalse(b['bids'] or b['asks'])
            if arm[1] == '1':
                self.assertTrue(all(p == 3.14159 for p in b['asset']['price_history'][-20:]))
        self.assertEqual(cp.payload, original)
        rng_equal(rng, np.random.get_state())
        self.assertEqual(intervene(cp, '000')[0], cp)

    def test_empty_inactive_no_cash_no_shares_and_affordability(self):
        for p in ([(0, 0)]*4, [(10, 0)]*4, [(0, 3)]*4,
                  [(0, 5), (.01, 0), (.02, 0), (.03, 0)]):
            cp = self.state(p)
            result, _ = intervene(cp, '001')
            self.assertEqual(result, cp)
        cp = self.state([(1, 2), (30, 4), (500, 6), (7000, 8)], inactive=(0, 1, 2, 3))
        self.assertEqual(intervene(cp, '001')[0], cp)
        # Fuzz only small portfolios; checks conservation and exact cash caps.
        rng = np.random.default_rng(10)
        for _ in range(100):
            price = float(10**rng.uniform(-4, 4))
            cp = self.state(list(zip(rng.random(4)*price*10, rng.integers(0, 20, 4).tolist())), price=price)
            intervene(cp, '001')

    def test_future_rng_is_local_and_common_across_arms(self):
        before = np.random.get_state()
        a = with_future(self.checkpoint, 10)
        b = with_future(intervene(self.checkpoint, '111')[0], 10)
        self.assertEqual(a.data()['rng_state'], b.data()['rng_state'])
        rng_equal(before, np.random.get_state())


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='task6-tests-')
        cls.root = Path(cls.temp.name)
        cls.atlas = cls.root / 'atlas'
        cls.atlas.mkdir()
        manifest = {'runs': expand_spec(make_spec(seeds=(100,), days=20, burn_in=0), fingerprint())}
        write_json(cls.atlas / 'manifest.json', manifest)
        write_json(cls.atlas / 'manifest.sha256.json', {'sha256': digest(manifest)})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self, *args, code=0):
        p = subprocess.run([sys.executable, str(ROOT / 'experiments/run_inactivity.py'), *map(str, args)],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, code, p.stdout[-3000:] + p.stderr[-3000:])
        return p.stdout

    def test_default_design_and_invalid_inputs(self):
        study = generate_study(self.atlas, self.root/'default')
        self.assertEqual(len(reference_jobs(study)), 800)
        self.assertEqual(study['seeds'], list(range(600, 650)))
        self.assertEqual(study['futures'], 5)
        for kwargs in ({'seeds': (1, 1)}, {'days': 0}, {'futures': True}, {'tail_days': 1001}, {'threshold': 10241}):
            with self.assertRaises(ValueError):
                generate_study(self.atlas, self.root/'invalid', **kwargs)
        self.assertFalse((self.root/'invalid').exists())

    def test_cli_bounded_resume_no_eligible_corruption_and_plots(self):
        root = self.root/'smoke'
        study = generate_study(self.atlas, root, seeds=(600,), days=20, threshold=20,
                               continuation_days=10, tail_days=5, futures=2, resamples=50)
        self.call('study', root, '--one-batch', '--minutes', .00001, code=2)
        self.call('study', root, '--workers', 2)
        self.assertEqual(continuation_jobs(root, study, reference_jobs(study)), [])
        records = {p: p.stat().st_mtime_ns for p in root.glob('*/runs/*/complete.json')}
        self.call('study', root)
        self.assertEqual(records, {p: p.stat().st_mtime_ns for p in records})
        path = next(root.glob('reference/runs/*/daily.npz'))
        with path.open('ab') as stream:
            stream.write(b'corrupt')
        self.call('study', root, code=2)
        self.call('study', root, '--retry-failed')
        self.call('status', root)
        report_path = next(root.glob('analysis/*/summary.json'))
        report = read_json(report_path)
        self.assertEqual(report['reference_count'], 16)
        self.assertEqual(report['eligible_checkpoints'], 0)
        self.assertTrue(all(v['status'] == 'not_applicable' for v in report['recovery'].values()))
        with patch('src.inactivity_study.fingerprint', return_value={}):
            with self.assertRaises(ValueError):
                load_study(root, current=True)

    def test_stalled_artifact_replay_dedup_and_corruption(self):
        root = self.root/'idle'
        root.mkdir()
        execution = fingerprint()
        ref = dict(phase='reference', cell='idle', seed=600, days=12, threshold=10,
                   continuation_days=8, thresholds=[10, 20, 100, 250],
                   scenario=asdict(Scenario('idle', 'Idle', noise=0, chartists=0, fundamentalists=2,
                                           dividend_mode='constant', order_lifetime='day')))
        ref['run_id'] = digest(ref)
        path = root/'reference/runs'/ref['run_id']
        path.parent.mkdir(parents=True)
        materialize(ref, path, execution)
        study = {'execution': execution, 'futures': 2, 'continuation_days': 8, 'threshold': 10,
                 'analysis_settings': {'primary_tail_days': 5}, 'future_seeds': {'idle': {'600': future_seeds('idle', 600, 2)}}}
        job = continuation_jobs(root, study, [ref])[0]
        destination = root/'continuation'
        materialize(job, destination, execution)
        validate_artifact(destination, job, execution, deep=True)
        record = read_json(destination/'recovery.json')
        self.assertTrue(record['original_rng_replay_verified'])
        self.assertLess(record['unique_continuations'], record['logical_continuations'])
        self.assertEqual(set(record['arms']), set(ARMS))
        (destination/'recovery.json').write_text('{}')
        with self.assertRaises(ValueError):
            validate_artifact(destination, job, execution)

    def test_resource_limits_stop_without_censoring_or_publishing(self):
        for label, target, returned, reason in (
            ('disk', 'src.heterogeneity_study.disk_guard', 'output_limit', 'output_limit'),
            ('memory', 'src.heterogeneity_study.memory_available', 0, 'memory_pressure')):
            root = self.root/label
            generate_study(self.atlas, root, seeds=(600,), days=20, threshold=20,
                           continuation_days=10, tail_days=5, futures=1)
            with patch(target, return_value=returned):
                self.assertEqual(complete_study(root, repeat=False), 2)
            with sqlite3.connect(root/'reference/ledger.sqlite') as db:
                self.assertEqual(db.execute('SELECT stop_reason FROM invocations').fetchone()[0], reason)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs WHERE state='pending'").fetchone()[0], 16)
            self.assertFalse(list(root.glob('reference/runs/*/complete.json')))
            self.assertFalse((root/'analysis').exists())
        for kwargs in ({'workers': 7}, {'memory_gib': 7}, {'output_gib': 6}, {'min_free_gib': 9}):
            with self.assertRaises(ValueError):
                Limits(**kwargs)

    def test_signal_terminates_worker_and_resume_keeps_completed_work(self):
        root = self.root/'interrupted'
        generate_study(self.atlas, root, seeds=(600,), days=20, threshold=20,
                       continuation_days=10, tail_days=5, futures=1, resamples=50)
        with (self.root/'signal.log').open('w') as stream:
            process = subprocess.Popen([sys.executable, str(ROOT/'experiments/run_inactivity.py'), 'study', str(root)],
                                       stdout=stream, stderr=subprocess.STDOUT)
            worker_pid = None
            try:
                deadline = time.monotonic()+20
                while process.poll() is None and time.monotonic() < deadline:
                    for path in (root/'reference/invocations').glob('*.jsonl'):
                        for line in path.read_text().splitlines():
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if event.get('event') == 'launched':
                                worker_pid = event['pid']
                    if worker_pid:
                        break
                    time.sleep(.01)
                self.assertIsNotNone(worker_pid)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=15), 2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)
        self.call('study', root)
        with sqlite3.connect(root/'reference/ledger.sqlite') as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runs WHERE state!='complete'").fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
