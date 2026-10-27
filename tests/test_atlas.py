"""Standalone atlas design, scientific conventions, and recovery checks."""

from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-task3-matplotlib')

import numpy as np

from src.atlas import (SETTINGS, aggregate, analyze_arrays, axes, make_spec, nonlinear,
                       validation_controls, variants)
from src.atlas_runner import (analysis_fingerprint, cached_record, collect, manifest_at,
                              save_record)
from src.experiment_artifacts import canonical, digest, expand_spec, file_hash, read_json, write_json
from src.experiments import Limits, execute_experiment


class DesignTests(unittest.TestCase):
    def test_full_design_and_overlap(self):
        state = np.random.get_state()
        spec = make_spec()
        runs = expand_spec(spec, {})
        self.assertEqual(len(spec['cells']), 3200)
        self.assertEqual(len(runs), 2840)
        self.assertEqual(sum(len(r['cells']) == 2 for r in runs), 360)
        self.assertEqual(len({digest(r['scenario']) for r in runs}), 568)
        self.assertEqual([r['seed'] for r in runs], sorted(r['seed'] for r in runs))
        self.assertEqual([r['scenario']['order_lifetime'] for r in runs[:8]], ['persistent', 'day'] * 4)
        self.assertLess(len(canonical({'runs': runs, 'specification': spec}).encode()), 4 * 1024**2)
        for before, after in zip(state, np.random.get_state()):
            np.testing.assert_array_equal(before, after)
        a, grid, g = axes()
        self.assertEqual(grid, a[::2])
        self.assertEqual((a[0], a[8], a[-1], g[0], g[3], g[-1]), (0.025, 0.1, 0.4, 0.5, 2, 8))

    def test_bad_configuration_rejected_by_existing_interface(self):
        for kwargs in ({'seeds': [-1]}, {'days': 0}, {'burn_in': 2000}):
            with self.assertRaises(ValueError):
                expand_spec(make_spec(**kwargs), {})


class DiagnosticTests(unittest.TestCase):
    def test_daily_window_and_stall_boundary(self):
        price = np.array([100., 80., 80., 80., 40., 40.])
        previous = np.array([100., 100., 80., 80., 80., 40.])
        arrays = {'price': price, 'previous_price': previous, 'log_return': np.log(price / previous),
                  'volume': np.array([1., 2., 0., 0., 3., 0.])}
        result = analyze_arrays(arrays, 2, 100)
        m = result['metrics']
        self.assertEqual(result['observations'], 4)
        self.assertAlmostEqual(m['cumulative_log_return']['value'], np.log(0.5))
        self.assertEqual(m['maximum_drawdown']['value'], 0.5)
        self.assertEqual(m['longest_no_trade_streak']['value'], 2)
        self.assertEqual(m['terminal_no_trade_streak']['value'], 1)
        self.assertEqual(m['positive_volume_volatility']['status'], 'insufficient_input')
        self.assertEqual(m['zero_volume_fraction']['value'], 0.75)

    def test_constant_nonfinite_short_and_rng(self):
        for values, status in ((np.full(100, 0.1), 'constant_series'), (np.full(100, np.nan), 'non_finite_input'),
                               (np.ones(3), 'insufficient_input')):
            self.assertEqual({r['status'] for r in nonlinear(values, 100).values()}, {status})
        state = np.random.get_state()
        data = np.random.default_rng(123).normal(size=150)
        first = nonlinear(data, 100)
        self.assertEqual(first, nonlinear(data, 100))
        for a, b in zip(state, np.random.get_state()):
            np.testing.assert_array_equal(a, b)

    def test_entropy_independent_match_count(self):
        x = np.tile([0., 1., 0., 0., 1., 1., 0.], 6)
        tolerance = 0.2 * np.std(x)
        counts = []
        for length in (3, 4):
            templates = [x[i:i + length] for i in range(len(x) - 3)]
            counts.append(sum(max(abs(a - b)) < tolerance for a, b in combinations(templates, 2)))
        expected = -np.log(counts[1] / counts[0])
        self.assertAlmostEqual(nonlinear(x, 100)['sample_entropy']['value'], expected)

    def test_known_system_validation(self):
        controls = validation_controls()
        self.assertLess(abs(controls['logistic_short_fit']['value'] - np.log(2)), 0.12)
        self.assertEqual(controls['production_results']['constant']['rosenstein']['status'], 'constant_series')
        self.assertIn('gaussian_noise', controls['production_results'])

    def test_aggregation_counts_undefined_and_binary_incidence(self):
        spec = {'schema_version': 1, 'name': 'fixture', 'days': 120, 'burn_in': 0,
                'cells': [{'name': 'a', 'scenario': 'baseline', 'seeds': [100, 101, 102, 103, 104]}]}
        runs = expand_spec(spec, {})
        records = {r['run_id']: {'state': 'complete', 'result': {'metrics': {
            'terminal_stall': {'value': int(i == 0), 'status': 'ok'},
            'rosenstein': {'value': None, 'status': 'constant_series'}}}} for i, r in enumerate(runs)}
        summary = aggregate({'runs': runs}, records)
        stats = summary['cells'][0]['metrics']
        self.assertEqual(stats['terminal_stall']['frequency'], 0.2)
        self.assertEqual(stats['terminal_stall']['median'], 0)
        self.assertEqual(stats['rosenstein']['valid'], 0)
        self.assertEqual(stats['rosenstein']['statuses'], {'constant_series': 5})
        del records[runs[0]['run_id']]
        self.assertEqual(aggregate({'runs': runs}, records)['analyzed_runs'], 4)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.directory = cls.root / 'simulation'
        spec = {'schema_version': 1, 'name': 'atlas-test', 'days': 120, 'burn_in': 20,
                'cells': [{'name': 'fixture', 'scenario': 'baseline', 'seeds': [100, 101, 102]}]}
        write_json(cls.root / 'spec.json', spec)
        code = execute_experiment(cls.directory, spec=cls.root / 'spec.json', limits=Limits(minutes=2),
                                  log_path=cls.root / 'simulation.log')
        if code:
            raise AssertionError((cls.root / 'simulation.log').read_text())
        cls.manifest = manifest_at(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def command(self, *extra):
        return [sys.executable, str(ROOT / 'experiments/run_atlas.py'), 'analyze', str(self.directory),
                '--minutes', '2', *extra]

    def test_resume_identity_corruption_and_settings(self):
        with (self.root / 'analysis.log').open('w') as log:
            self.assertEqual(subprocess.call(self.command(), stdout=log, stderr=subprocess.STDOUT), 0)
        summary, records = collect(self.directory)
        self.assertEqual(summary['analyzed_runs'], 3)
        self.assertEqual(summary['operational_failures'], 0)
        fingerprint = analysis_fingerprint()
        output = self.root / 'analysis' / digest(fingerprint)
        times = {p.name: p.stat().st_mtime_ns for p in (output / 'runs').iterdir()}
        with (self.root / 'resume.log').open('w') as log:
            self.assertEqual(subprocess.call(self.command(), stdout=log, stderr=subprocess.STDOUT), 0)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in (output / 'runs').iterdir()})
        run = self.manifest['runs'][0]
        cache = output / 'runs' / f'{run["run_id"]}.json'
        artifact = self.directory / 'runs' / run['run_id']
        other = deepcopy(SETTINGS)
        other['rosenstein']['lag'] = 2
        self.assertIsNone(cached_record(cache, run, artifact, analysis_fingerprint(other)))
        cache.write_text('{}')
        self.assertIsNone(cached_record(cache, run, artifact, fingerprint))
        with (self.root / 'repair.log').open('w') as log:
            self.assertEqual(subprocess.call(self.command(), stdout=log, stderr=subprocess.STDOUT), 0)
        self.assertEqual(collect(self.directory)[0]['analyzed_runs'], 3)
        source = artifact / 'summary.json'
        original = source.read_bytes()
        try:
            source.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                cached_record(cache, run, artifact, fingerprint)
            self.assertEqual(collect(self.directory)[0]['operational_failures'], 1)
        finally:
            source.write_bytes(original)

    def test_plot_coverage_and_mask(self):
        from src.atlas_plotting import plot_metric
        import matplotlib.pyplot as plt
        summary = aggregate(self.manifest, {})
        heat, curve = plot_metric(summary, 'rosenstein')
        self.assertTrue(heat.axes[0].collections[0].get_array().mask.all())
        self.assertTrue(any('0/3' == text.get_text() for text in heat.axes[0].texts))
        self.assertIn('undefined', heat._suptitle.get_text())
        plt.close(heat)
        plt.close(curve)

    def test_interruption_and_timeout_recovery(self):
        output = self.root / 'analysis' / digest(analysis_fingerprint())
        if output.exists():
            shutil.rmtree(output)
        with (self.root / 'interrupt.log').open('w') as log:
            process = subprocess.Popen(self.command('--workers', '1'), stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 20
                while len(list((output / 'runs').glob('*.json'))) < 1 and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail('Analysis completed before interruption checkpoint')
                    time.sleep(0.01)
                retained = {p.name: p.stat().st_mtime_ns for p in (output / 'runs').glob('*.json')}
                self.assertGreaterEqual(len(retained), 1)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=10), 2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        with (self.root / 'interrupt-resume.log').open('w') as log:
            self.assertEqual(subprocess.call(self.command(), stdout=log, stderr=subprocess.STDOUT), 0)
        self.assertEqual(collect(self.directory)[0]['analyzed_runs'], 3)
        for name, timestamp in retained.items():
            self.assertEqual((output / 'runs' / name).stat().st_mtime_ns, timestamp)
        shutil.rmtree(output)
        with (self.root / 'timeout.log').open('w') as log:
            self.assertEqual(subprocess.call(self.command('--timeout', '0.001'), stdout=log, stderr=subprocess.STDOUT), 2)
        self.assertEqual(collect(self.directory)[0]['operational_failures'], 3)
        with (self.root / 'timeout-retry.log').open('w') as log:
            self.assertEqual(subprocess.call(self.command('--retry-failed'), stdout=log, stderr=subprocess.STDOUT), 0)
        self.assertEqual(collect(self.directory)[0]['operational_failures'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
