"""Standalone numerical, input, provenance, and live recovery checks for Task 7."""

from copy import deepcopy
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-task7-tests-matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np

from src.atlas_runner import atomic_json
from src.experiment_artifacts import digest, read_json
from src.experiments import Limits
from src.index_data import load_chronological_index, load_index
from src.multifractal import SETTINGS, analyze_series, log_fluctuations, scales_for, validate_settings
from src.multifractal_controls import admissible_garch, control, generator, iaaft
from src.multifractal_study import (analysis_directory, cached_record, generate_study, jobs_for,
                                    load_study, load_window, phase_jobs, result_path, study_status,
                                    windows_for, complete_study, verified_simulations)
from src.multifractal_plotting import collect_study, control_summary
from src.multifractal_validation import validate_estimators


def same_rng(test, before, after):
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


class NumericalTests(unittest.TestCase):
    def test_garch_stationarity_boundary(self):
        parameters = {'mu': 0., 'omega': .1, 'alpha[1]': .1, 'beta[1]': .8, 'nu': 8.}
        self.assertTrue(admissible_garch(parameters))
        for persistence in (1., np.nextafter(1., 0.), 1-1e-10, 1.01):
            parameters.update({'alpha[1]': 0., 'beta[1]': persistence})
            self.assertFalse(admissible_garch(parameters), persistence)
        parameters.update({'beta[1]': .8, 'nu': 2.})
        self.assertFalse(admissible_garch(parameters))

    def test_known_answers_and_rng(self):
        before = np.random.get_state()
        result = validate_estimators()
        self.assertTrue(result['passed'], result['checks'])
        same_rng(self, before, np.random.get_state())

    def test_q_zero_and_zero_segment_moments(self):
        result = log_fluctuations(np.array([1., 4., 16.]), np.array([-1e-5, 0., 1e-5]))
        self.assertAlmostEqual(result[1], np.log(2.))
        self.assertAlmostEqual(result[[0, 2]].mean(), result[1], places=8)
        zero = log_fluctuations(np.array([0., 4.]), np.array([-2., 0., 2.]))
        self.assertTrue(np.isnan(zero[:2]).all())
        self.assertAlmostEqual(zero[2], np.log(np.sqrt(2.)))

    def test_scales_and_invalid_settings(self):
        self.assertEqual((scales_for(2048)[0], scales_for(2048)[-1]), (16, 256))
        self.assertEqual(scales_for(1024).tolist(), scales_for(8192)[:13].tolist())
        for change in ({'chunk_size': 26}, {'orders': [1]}, {'q': [0, -1, 1]}, {'replicates': 0}):
            settings = dict(SETTINGS, **change)
            with self.assertRaises(ValueError):
                validate_settings(settings)

    def test_zero_and_unresolved_series(self):
        self.assertEqual(analyze_series(np.ones(2048))['status'], 'constant_series')
        self.assertEqual(analyze_series([0., np.inf])['status'], 'non_finite_input')
        self.assertEqual(analyze_series(np.arange(32.))['status'], 'insufficient_scales')
        with self.assertRaises(ValueError):
            analyze_series(np.ones((3, 2)))
        x = np.r_[np.random.default_rng(8).normal(size=1024), np.zeros(1024)]
        part = analyze_series(x)['orders']['1']
        self.assertIsNone(part['fit']['delta_h'])
        self.assertIsNone(part['fit']['spectrum_width'])
        self.assertGreater(sum(d['zero_segments'] for d in part['segment_diagnostics']), 0)
        self.assertTrue(np.isfinite(np.array(part['log_fluctuations'])[:, -1]).all())

    def test_controls_rng_identity_and_ties(self):
        before = np.random.get_state()
        x = np.round(np.random.default_rng(3).normal(size=2048), 1)
        for family in ('shuffled', 'iaaft'):
            a, da = control(x, family, 'source-window', 1)
            b, db = control(x, family, 'source-window', 1)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(np.sort(a), np.sort(x))
            self.assertEqual(da, db)
            self.assertNotEqual(da['seed_token'], control(x, family, 'source-window', 2)[1]['seed_token'])
        same_rng(self, before, np.random.get_state())
        _, diagnostic = iaaft(x, generator('bad-fit', 'iaaft', 0)[0],
                              dict(SETTINGS, iaaft_max_spectral_error=1e-15, iaaft_iterations=1))
        self.assertEqual(diagnostic['status'], 'spectral_mismatch')

    def test_weak_null_fits_are_retained_and_incomplete_is_explicit(self):
        observed = {'delta_h': .5, 'spectrum_width': .7}
        fit = {'h': [.5]*13, 'delta_h': .2, 'spectrum_width': .3, 'status': 'weak_scaling'}
        records = [{'diagnostic': {'status': 'ok'}, 'analysis': {'orders': {'1': {'fit': fit}}}}]*2
        summary = control_summary(records, '1', 'fit', observed, 2)
        self.assertEqual(summary['weak_scaling_count'], 2)
        self.assertEqual(summary['statistics']['delta_h']['valid'], 2)
        self.assertEqual(summary['statistics']['delta_h']['status'], 'complete')
        partial = control_summary(records[:1], '1', 'fit', observed, 2)
        self.assertEqual(partial['statistics']['delta_h']['status'], 'incomplete_controls')
        self.assertIsNone(partial['statistics']['delta_h']['above_reference_interval'])


class InputTests(unittest.TestCase):
    def test_chronology_and_legacy_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'index.csv'
            path.write_text('Дата,Цена\n03.01.2026,"121,0"\n02.01.2026,"110,0"\n01.01.2026,"100,0"\n')
            original = load_index(path)
            chronological = load_chronological_index(path)
            np.testing.assert_allclose(chronological['returns'], [np.log(1.1), np.log(1.1)])
            self.assertEqual(str(chronological['dates'][0]), '2026-01-02')
            self.assertEqual(original['Цена'].tolist(), [121, 110, 100])
            self.assertEqual(load_index(path)['Цена'].tolist(), original['Цена'].tolist())

    def test_invalid_dates_and_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'index.csv'
            for rows in ('01.01.2026,1\n01.01.2026,2\n', '01.01.2026,0\n02.01.2026,2\n',
                         '01.01.2026,inf\n02.01.2026,2\n', 'invalid,1\n02.01.2026,2\n'):
                path.write_text('Дата,Цена\n'+rows)
                with self.assertRaises(ValueError):
                    load_chronological_index(path)

    def test_fixed_design_and_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'study'
            manifest = generate_study(root, ROOT/'results/task3_atlas_20261027')
            self.assertEqual(len(manifest['configurations'])*len(manifest['seeds']), 15)
            self.assertEqual(len(manifest['indices']), 5)
            self.assertEqual(manifest['configurations']['liquidity-v6']['chartist_alpha'], .19999999999999998)
            self.assertEqual(load_study(root, current=True), manifest)
            path = root/'inputs'/manifest['indices'][0]['filename']
            with path.open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Corrupt index'):
                load_study(root)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='abm-task7-tests-')
        cls.root = Path(cls.temporary.name)/'study'
        cls.settings = deepcopy(SETTINGS)
        cls.settings.update(replicates=2, chunk_size=1, simulation_lengths=[1024],
                            index_lengths=[1024], primary_length=1024)
        cls.manifest = generate_study(cls.root, ROOT/'results/task3_atlas_20261027', seeds=(900,),
                                      days=1024, burn_in=0, settings=cls.settings)
        cls.command = [sys.executable, str(ROOT/'experiments/run_multifractal.py')]
        cls.log = Path(cls.temporary.name)/'workflow.log'
        # Exercise live termination after a published result, then resume with a
        # different worker count. The remaining tests inspect the completed run.
        with cls.log.open('w') as stream:
            process = subprocess.Popen(cls.command+['study', str(cls.root), '--workers', '1'],
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            output = analysis_directory(cls.root, cls.manifest)
            deadline = time.monotonic()+120
            while process.poll() is None and time.monotonic() < deadline:
                records = list((output/'records').glob('*.json.gz'))
                if len(records) >= 2:
                    break
                time.sleep(.05)
            cls.retained = {p.name: p.stat().st_mtime_ns for p in (output/'records').glob('*.json.gz')}
            if not cls.retained:
                process.terminate()
                process.wait(timeout=10)
                raise AssertionError(cls.log.read_text())
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=15)
        with cls.log.open('a') as stream:
            result = subprocess.run(cls.command+['study', str(cls.root), '--workers', '2'],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise AssertionError(cls.log.read_text())
        cls.resume_preserved = all((output/'records'/name).stat().st_mtime_ns == timestamp
                                   for name, timestamp in cls.retained.items())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_resume_coverage_and_published_reuse(self):
        output = analysis_directory(self.root, self.manifest)
        self.assertTrue(self.resume_preserved)
        state = study_status(self.root)
        self.assertEqual(state['simulation']['runs'], {'complete': 5})
        self.assertEqual(state['analysis_records'], {'observed:complete': 10, 'fit:complete': 10,
                                                   'controls:complete': 60})
        manifest, rows = collect_study(self.root)
        self.assertEqual(len(rows), 10)
        self.assertEqual(manifest['analysis_id'], self.manifest['analysis_id'])

    def test_window_data_and_provenance(self):
        window = windows_for(self.root, self.manifest)[0]
        values, diagnostics = load_window(window)
        self.assertEqual(len(values), 1024)
        self.assertEqual(diagnostics['first_trading_day'], 0)
        with patch('src.multifractal_study.analysis_fingerprint', return_value={}):
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_study(self.root, current=True)

    def test_corrupt_cache_recomputed_with_identical_science(self):
        output = analysis_directory(self.root, self.manifest)
        job = jobs_for(windows_for(self.root, self.manifest), 'observed', self.settings)[0]
        path = result_path(output, job)
        original = cached_record(path, job, self.manifest['analysis_id'])
        path.write_bytes(b'corrupt cache')
        self.assertIsNone(cached_record(path, job, self.manifest['analysis_id']))
        with self.log.open('a') as stream:
            result = subprocess.run(self.command+['analyze', str(self.root), '--workers', '1'],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(result.returncode, 0, self.log.read_text())
        repaired = cached_record(path, job, self.manifest['analysis_id'])
        self.assertEqual(repaired['result'], original['result'])

    def test_resource_stop_does_not_launch_workers(self):
        job = jobs_for(windows_for(self.root, self.manifest), 'observed', self.settings)[0]
        with patch('src.multifractal_study.disk_guard', return_value='free_disk_limit'):
            code = phase_jobs(self.root, self.manifest, [job], Limits())
        self.assertEqual(code, 2)
        output = analysis_directory(self.root, self.manifest)
        self.assertEqual(read_json(output/'last_invocation.json')['stop_reason'], 'free_disk_limit')
        with patch('src.multifractal_study.memory_available', return_value=0):
            self.assertEqual(phase_jobs(self.root, self.manifest, [job], Limits()), 2)
        with patch('src.multifractal_study.resident_bytes', return_value=7*1024**3):
            self.assertEqual(phase_jobs(self.root, self.manifest, [job], Limits()), 2)

    def test_timeout_and_retry(self):
        window = dict(windows_for(self.root, self.manifest)[0], window_id='timeout-fixture')
        job = jobs_for([window], 'observed', self.settings)[0]
        self.assertEqual(phase_jobs(self.root, self.manifest, [job], Limits(), timeout=.001), 2)
        output = analysis_directory(self.root, self.manifest)
        self.assertEqual(cached_record(result_path(output, job), job, self.manifest['analysis_id'])['state'], 'timeout')
        self.assertEqual(phase_jobs(self.root, self.manifest, [job], Limits(), retry_failed=True), 0)
        # This isolated supervisor fixture is not a member of the immutable study.
        result_path(output, job).unlink()

    def test_report_artifacts(self):
        report = self.root/'reports'/self.manifest['analysis_id']
        self.assertTrue((report/'findings.md').exists())
        self.assertEqual(len(list(report.glob('*.png'))), 5)
        self.assertEqual(len(list(report.glob('*.pdf'))), 5)
        self.assertEqual(read_json(report/'summary.json')['operational_failures'], 0)
        groups = read_json(report/'summary.json')['primary_simulation_groups']
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(group['expected'] == 1 for group in groups))
        with np.load(report/'tables.npz', allow_pickle=False) as arrays:
            self.assertEqual(len(arrays['length']), 10)

    def test_verified_simulation_reuse(self):
        root = Path(self.temporary.name)/'reused'
        manifest = generate_study(root, ROOT/'results/task3_atlas_20261027', seeds=(900,),
                                  days=1024, burn_in=0, settings=self.settings,
                                  reuse_simulations=self.root/'simulation')
        self.assertEqual(manifest['reused_simulation']['runs'], 5)
        load_study(root, current=True)
        with patch('src.multifractal_study.launch_simulation') as launch:
            with patch('src.multifractal_study.validate_study', return_value={'passed': True}):
                with patch('src.multifractal_study.analyze_study', return_value=2):
                    self.assertEqual(complete_study(root), 2)
            launch.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'specification differs'):
            generate_study(Path(self.temporary.name)/'wrong-spec', ROOT/'results/task3_atlas_20261027',
                           seeds=(900,), days=2048, burn_in=0, settings=self.settings,
                           reuse_simulations=self.root/'simulation')
        next((root/'simulation/runs').glob('*/daily.npz')).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt artifact'):
            verified_simulations(root/'simulation', read_json(root/'spec.json'))

class ParentDeathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='abm-task7-parent-test-')
        self.settings = dict(SETTINGS, simulation_lengths=[1024], index_lengths=[1024], primary_length=1024)
        self.command = [sys.executable, str(ROOT/'experiments/run_multifractal.py')]

    def tearDown(self):
        self.temporary.cleanup()

    def test_parent_death_stops_simulation_descendants(self):
        root = Path(self.temporary.name)/'parent-death'
        generate_study(root, ROOT/'results/task3_atlas_20261027', seeds=(901,),
                       days=10240, burn_in=0, settings=self.settings)
        process, descendants = None, []
        def alive(pid):
            try:
                return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
            except FileNotFoundError:
                return False
        try:
            with (root/'driver.log').open('w') as stream:
                process = subprocess.Popen(self.command+['study', str(root), '--workers', '1'],
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+60
                while process.poll() is None and time.monotonic() < deadline:
                    path = root/'simulation/ledger.sqlite'
                    if path.exists():
                        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
                            workers = [r[0] for r in db.execute("SELECT pid FROM attempts WHERE state='running' AND pid IS NOT NULL")]
                        if workers:
                            children = Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()
                            descendants = list(map(int, children))+workers
                            break
                    time.sleep(.03)
                self.assertTrue(descendants, (root/'driver.log').read_text())
                process.kill()
                process.wait(timeout=10)
            deadline = time.monotonic()+10
            while any(alive(pid) for pid in descendants) and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertFalse(any(alive(pid) for pid in descendants), descendants)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for pid in descendants:
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
    unittest.main(verbosity=2)
