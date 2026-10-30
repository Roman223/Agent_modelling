"""Standalone checkpoint, compression, inference, and study recovery checks."""

from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-heterogeneity-matplotlib')

import numpy as np

from src.scenarios import SCENARIOS, Scenario
from src.simulation import simulate
from src.market_checkpoints import (MarketCheckpoint, run_checkpointed, continue_checkpoint, dynamics_digest,
                                    read_checkpoint, write_checkpoint)
from src.heterogeneity import CompressionConfig, compress_checkpoint, STRATEGIES
from src.heterogeneity_analysis import compare_outcomes, outcome_summary, paired_stall_interval, select_threshold
from src.heterogeneity_study import (generate_study, reference_jobs, continuation_jobs, materialize,
                                     validate_artifact, read_arrays, fingerprint)
from src.experiment_artifacts import canonical, digest, write_json
from src.atlas import make_spec


def rng_equal(test, a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


class CheckpointTests(unittest.TestCase):
    def test_matches_legacy_and_uninterrupted_for_all_controls(self):
        for life in ('persistent', 'day'):
            for noise in (0, .05):
                for dividend in ('constant', 'stochastic'):
                    s = replace(SCENARIOS['baseline'], order_lifetime=life, closing_noise_scale=noise, dividend_mode=dividend)
                    np.random.seed(301)
                    ordinary = simulate(s, days=120)
                    expected_rng = np.random.get_state()
                    np.random.seed(42)
                    caller = np.random.get_state()
                    r = run_checkpointed(s, seed=301, days=120, checkpoint_days=(50,), check_days=(120,))
                    rng_equal(self, caller, np.random.get_state())
                    np.testing.assert_array_equal(r.final.data()['asset']['price_history'], ordinary.asset.price_history)
                    np.testing.assert_array_equal(r.final.data()['asset']['dividend_history'], ordinary.asset.dividend_history)
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / 'checkpoint.gz'
                        write_checkpoint(path, r.checkpoints[50])
                        restored = read_checkpoint(path)
                    self.assertEqual(restored, r.checkpoints[50])
                    c = continue_checkpoint(restored, days=70)
                    rng_equal(self, caller, np.random.get_state())
                    self.assertEqual(dynamics_digest(c.final), r.checks[120])
                    for field in ('price', 'volume', 'total_cash', 'total_assets', 'log_return'):
                        np.testing.assert_array_equal(c.arrays[field], r.arrays[field][50:])
                    for t, actual in zip(ordinary.traders, c.final.data()['traders']):
                        self.assertEqual((t.cash, t.assets, t.active), (actual['cash'], actual['assets'], actual['active']))
                        np.testing.assert_array_equal(t.wealth_history, actual['wealth_history'])

    def test_rng_restored_on_failure_and_unsupported_inputs(self):
        np.random.seed(13)
        before = np.random.get_state()
        for s in (SCENARIOS['evolution'], SCENARIOS['circuit_breaker']):
            with self.assertRaises(ValueError):
                run_checkpointed(s, seed=0, days=10)
        for kwargs in ({'seed': True}, {'seed': 1, 'days': 0}, {'seed': 1, 'days': 5, 'checkpoint_days': (6,)}):
            with self.assertRaises(ValueError):
                run_checkpointed(SCENARIOS['baseline'], **kwargs)
        rng_equal(self, before, np.random.get_state())

    def test_scalar_types_survive_roundtrip(self):
        r = run_checkpointed(SCENARIOS['baseline'], seed=300, days=20)
        data = r.final.data()
        self.assertIsInstance(data['asset']['current_price'], np.float64)
        self.assertEqual(MarketCheckpoint.from_data(data), r.final)

    def test_checkpoint_immutable_and_validated(self):
        cp = run_checkpointed(SCENARIOS['baseline'], seed=2, days=5).final
        data = cp.data()
        data['traders'][0]['cash'] = -1
        with self.assertRaises(ValueError):
            MarketCheckpoint.from_data(data)
        self.assertGreaterEqual(cp.data()['traders'][0]['cash'], 0)
        data = cp.data()
        data['bids'] = [[10, 1, 100000]]
        with self.assertRaises(ValueError):
            MarketCheckpoint.from_data(data)


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = run_checkpointed(SCENARIOS['baseline'], seed=300, days=100).final

    def test_conservation_nested_partitions_and_full_identity(self):
        before = self.cp.data()
        previous = None
        np.random.seed(7)
        rng = np.random.get_state()
        for k in (1, 2, 4, 8, 16, 'full'):
            result, details = compress_checkpoint(self.cp, CompressionConfig(k))
            data = result.data()
            for key in ('asset', 'bids', 'asks', 'rng_state', 'completed_days'):
                self.assertEqual(data[key], before[key])
            for kind in STRATEGIES:
                a = [t for t in before['traders'] if t['type'] == kind]
                b = [t for t in data['traders'] if t['type'] == kind]
                self.assertAlmostEqual(sum(t['cash'] for t in a), sum(t['cash'] for t in b), places=6)
                self.assertEqual(sum(t['assets'] for t in a), sum(t['assets'] for t in b))
                self.assertTrue(all(t['cash'] >= 0 and type(t['assets']) is int for t in b))
                if previous:
                    for group in details['groups'][kind]['members']:
                        self.assertTrue(any(set(group) <= set(old) for old in previous['groups'][kind]['members']))
            previous = details
            if k == 'full':
                self.assertIs(result, self.cp)
                self.assertFalse(details['changed'])
        rng_equal(self, rng, np.random.get_state())

    def test_integer_balancing_and_empty_portfolios(self):
        r = run_checkpointed(Scenario('tiny', 'Tiny', noise=4, chartists=0, fundamentalists=0), seed=0, days=1)
        data = r.final.data()
        for i, trader in enumerate(data['traders']):
            trader['cash'], trader['assets'] = float((i+1)*10), i
        data['traders'][0]['cash'] = 0.
        cp = MarketCheckpoint.from_data(data)
        result, info = compress_checkpoint(cp, CompressionConfig(1))
        t = result.data()['traders']
        self.assertEqual((t[0]['cash'], t[0]['assets']), (0, 0))
        self.assertEqual(sum(x['assets'] for x in t), 6)
        self.assertTrue(all(x['assets'] == 2 for x in t[1:]))
        self.assertEqual(info['groups']['chartist']['cluster_count'], 0)

    def test_identical_portfolios_are_not_changed(self):
        cp = run_checkpointed(Scenario('fund', 'Fund', noise=0, chartists=0, fundamentalists=5), seed=0, days=1).final
        result, info = compress_checkpoint(cp, CompressionConfig(1))
        self.assertEqual(cp, result)
        self.assertIsNone(info['groups']['fundamentalist']['retained_dispersion'])
        self.assertEqual(info['groups']['fundamentalist']['joint_wasserstein'], 0)
        for k in (0, -1, True, 2.5, 'unknown'):
            with self.assertRaises(ValueError):
                CompressionConfig(k)


class AnalysisTests(unittest.TestCase):
    def rows(self, n=50):
        result = []
        for i in range(n):
            result.append({'terminal_stall': False, 'zero_volume_fraction': 0., 'mean_volume': 100. + i,
                           'log_return_volatility': .01, 'cumulative_log_return': i/1000,
                           **{f'{resource}_fraction_{kind}': 1/3 for resource in ('cash', 'share') for kind in STRATEGIES}})
        return result

    def test_known_pass_fail_inconclusive(self):
        rows = self.rows()
        same = compare_outcomes(rows, rows, seed=1)
        self.assertEqual(same['status'], 'within_tolerance')
        shifted = [{**r, 'mean_volume': 10000} for r in rows]
        self.assertEqual(compare_outcomes(shifted, rows, seed=1)['status'], 'outside_tolerance')
        self.assertEqual(compare_outcomes(rows[:2], rows[:2], seed=1)['status'], 'inconclusive')
        self.assertEqual(compare_outcomes(rows[:2], rows[:2], seed=1, exact=True)['status'], 'within_tolerance')
        lo, hi = paired_stall_interval(np.zeros(50, bool), np.zeros(50, bool))
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_nonmonotone_threshold(self):
        states = {str(k): {'status': s} for k, s in ((1, 'within_tolerance'), (2, 'outside_tolerance'),
                                                    (4, 'inconclusive'), (8, 'within_tolerance'), ('full', 'within_tolerance'))}
        self.assertEqual(select_threshold(states, (1, 2, 4, 8, 'full'))['level'], 8)
        states['8']['status'] = 'outside_tolerance'
        self.assertEqual(select_threshold(states, (1, 2, 4, 8, 'full'))['status'], 'no_supported_compression')

    def test_stalls_do_not_borrow_precheckpoint_history(self):
        arrays = {k: np.zeros(100) for k in ('volume', 'log_return', 'within_strategy_volume', 'same_owner_volume')}
        arrays.update(total_cash=np.ones((100, 3)), total_assets=np.ones((100, 3)))
        self.assertIsNone(outcome_summary(arrays, 20)['terminal_stall'])
        self.assertTrue(outcome_summary(arrays, 100)['terminal_stall'])
        arrays['volume'][:] = .1
        summary = outcome_summary(arrays, 100)
        self.assertFalse(summary['terminal_stall'])
        self.assertEqual(summary['subshare_volume_fraction'], 1)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='task5-tests-')
        cls.root = Path(cls.temp.name)
        cls.atlas = cls.root / 'atlas'
        cls.atlas.mkdir()
        spec = make_spec(seeds=(100,), days=80, burn_in=20)
        from src.experiment_artifacts import expand_spec, execution_fingerprint
        runs = expand_spec(spec, execution_fingerprint())
        manifest = {'runs': runs}
        write_json(cls.atlas / 'manifest.json', manifest)
        write_json(cls.atlas / 'manifest.sha256.json', {'sha256': digest(manifest)})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cli_recovery_corruption_and_render(self):
        root = self.root / 'study'
        study = generate_study(self.atlas, root, development=(300,), validation=(310,), days=120,
                               checkpoints=(20,), continuation_days=100, levels=(1, 'full'), horizons=(1, 20, 100))
        cli = [sys.executable, str(ROOT / 'experiments/run_heterogeneity.py')]
        def call(args, expected=0):
            result = subprocess.run(cli+args, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stdout[-4000:] + result.stderr[-4000:])
            return result.stdout
        call(['run', str(root), '--minutes', '0.00001'], 2)
        call(['resume', str(root), '--workers', '2'])
        refs = reference_jobs(study)
        jobs = continuation_jobs(root, study, refs)
        for job in jobs:
            path = root / 'continuation/runs' / job['run_id']
            validate_artifact(path, job, study['execution'], deep=True)
            self.assertTrue(json.loads((path / 'levels.json').read_text())['full_detail_verified'])
        before = {p: p.stat().st_mtime_ns for p in root.glob('*/runs/*/complete.json')}
        call(['resume', str(root), '--workers', '1'])
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        path = root / 'continuation/runs' / jobs[0]['run_id'] / 'levels.json'
        path.write_text(path.read_text()+' ')
        call(['resume', str(root)], 2)
        call(['resume', str(root), '--retry-failed'])
        # Retrying a damaged reference must recreate identical bytes so that
        # already published continuation dependencies remain valid.
        reference_path = root / 'reference/runs' / refs[0]['run_id'] / 'daily.npz'
        original_hash = (reference_path.parent / 'complete.json').read_bytes()
        with reference_path.open('ab') as stream:
            stream.write(b'corrupt')
        call(['resume', str(root)], 2)
        call(['resume', str(root), '--retry-failed'])
        self.assertEqual(original_hash, (reference_path.parent / 'complete.json').read_bytes())
        rendered = json.loads(call(['render', str(root), '--bootstrap-resamples', '100']).strip().splitlines()[-1])
        report = json.loads((Path(rendered['directory']) / 'summary.json').read_text())
        self.assertEqual(len(report['thresholds']), 10)
        self.assertTrue(all(row['score'] == 0 for row in report['comparisons'] if row['level'] == 'full'))
        discovery = report['discovery']
        if discovery.get('status') == 'ok':
            for mode in ('current', 'lagged'):
                if discovery.get(mode, {}).get('status') == 'ok':
                    self.assertEqual(discovery[mode]['training_seeds'], [300])
                    self.assertEqual(discovery[mode]['validation_seeds'], [310])
        self.assertTrue(json.loads(call(['render', str(root), '--bootstrap-resamples', '100']).strip().splitlines()[-1])['reused'])
        artifact = root / 'reference/runs' / refs[0]['run_id']
        with self.assertRaises(ValueError):
            validate_artifact(artifact, refs[0], {'changed': True})

    def test_signal_stops_active_worker_and_resume_recovers(self):
        root = self.root / 'signal-study'
        generate_study(self.atlas, root, development=(300,), validation=(310,), days=80,
                       checkpoints=(20,), continuation_days=60, levels=('full',), horizons=(60,))
        cli = [sys.executable, str(ROOT / 'experiments/run_heterogeneity.py')]
        with (self.root / 'signal.log').open('w') as stream:
            process = subprocess.Popen(cli + ['run', str(root)], stdout=stream, stderr=subprocess.STDOUT)
            worker_pid = None
            try:
                deadline = time.monotonic() + 20
                while process.poll() is None and time.monotonic() < deadline:
                    ledger = root / 'reference/ledger.sqlite'
                    if ledger.exists():
                        try:
                            with sqlite3.connect(ledger) as db:
                                row = db.execute("SELECT pid FROM attempts WHERE state='running' AND pid IS NOT NULL").fetchone()
                            if row:
                                worker_pid = row[0]
                                break
                        except sqlite3.OperationalError:
                            pass
                    time.sleep(.01)
                self.assertIsNotNone(worker_pid, 'An active worker was not observed')
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=15), 2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)
        with sqlite3.connect(root / 'reference/ledger.sqlite') as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runs WHERE state='interrupted'").fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT stop_reason FROM invocations').fetchone()[0], 'signal_15')
        resumed = subprocess.run(cli + ['resume', str(root)], capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, resumed.stdout[-2000:] + resumed.stderr[-2000:])
        with sqlite3.connect(root / 'reference/ledger.sqlite') as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runs WHERE state!='complete'").fetchone()[0], 0)

    def test_invalid_design_rejected_before_writes(self):
        root = self.root / 'invalid'
        with self.assertRaises(ValueError):
            generate_study(self.atlas, root, development=(1,), validation=(1,))
        self.assertFalse(root.exists())
        with self.assertRaises(ValueError):
            generate_study(self.atlas, root, days=30, checkpoints=(20,), continuation_days=20)
        self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
