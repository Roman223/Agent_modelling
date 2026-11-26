"""Standalone correctness/recovery tests for the bounded experiment runner."""

from copy import deepcopy
from dataclasses import replace
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
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from src.experiment_artifacts import (daily_arrays, digest, execution_fingerprint, expand_spec,
                                      read_compact, read_json, summarize, write_json)
from src.experiments import (CLI, GIB, Limits, disk_guard, execute_experiment,
                             experiment_lock, status)
from src.scenarios import SCENARIOS, Scenario
from src.simulation import simulate


def specification(seeds=(100, 101, 102), days=12, overrides=None, scenario='baseline'):
    return {'schema_version': 1, 'name': 'test', 'days': days, 'burn_in': min(2, days - 1),
            'cells': [{'name': 'base', 'scenario': scenario, 'overrides': overrides or {}, 'seeds': list(seeds)}]}


class SpecTests(unittest.TestCase):
    def test_dedup_and_identity(self):
        spec = specification(seeds=(100, 100, 101))
        spec['cells'].append({**deepcopy(spec['cells'][0]), 'name': 'overlap'})
        state = np.random.get_state()
        runs = expand_spec(spec, {'version': 1})
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]['cells'], ['base', 'overlap'])
        changed_label = deepcopy(spec)
        changed_label['cells'][0]['name'] = 'renamed'
        self.assertEqual(runs[0]['run_id'], expand_spec(changed_label, {'version': 1})[0]['run_id'])
        self.assertNotEqual(runs[0]['run_id'], expand_spec(spec, {'version': 2})[0]['run_id'])
        for key, value in zip(state, np.random.get_state()):
            np.testing.assert_array_equal(key, value)

    def test_rejects_invalid_specs(self):
        cases = []
        for name, value in [('schema_version', True), ('days', 0), ('days', True),
                            ('burn_in', 12), ('burn_in', -1), ('cells', []), ('name', '')]:
            cases.append({**specification(), name: value})
        cases.append({**specification(), 'unknown': 1})
        for changes in ({'seeds': []}, {'seeds': [True]}, {'seeds': [-1]}, {'seeds': [2**32]},
                        {'scenario': 'missing'}, {'overrides': {'typo': 1}},
                        {'overrides': {'evolution': 1}}, {'overrides': {'chartist_alpha': float('nan')}}):
            value = specification()
            value['cells'][0].update(changes)
            cases.append(value)
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                expand_spec(spec, {})

    def test_limits_reject_machine_oversubscription(self):
        for kwargs in ({'workers': 0}, {'workers': 7}, {'workers': True}, {'minutes': float('inf')},
                       {'memory_gib': 7}, {'output_gib': 6}, {'min_free_gib': 9}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Limits(**kwargs)

    def test_disk_and_output_guards(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('src.experiments.shutil.disk_usage', return_value=type('Usage', (), {'free': 9 * GIB})()):
                self.assertEqual(disk_guard(root, Limits()), 'free_disk_limit')
            with patch('src.experiments.directory_bytes', return_value=5 * GIB), patch(
                    'src.experiments.shutil.disk_usage', return_value=type('Usage', (), {'free': 30 * GIB})()):
                self.assertEqual(disk_guard(root, Limits()), 'output_limit')

    def test_summary_definitions_and_undefined(self):
        arrays = {'price': np.array([80., 100., 50., 50.]), 'previous_price': np.array([100., 80., 100., 50.]),
                  'volume': np.array([0., 1., 0., 0.])}
        arrays['log_return'] = np.log(arrays['price']) - np.log(arrays['previous_price'])
        summary = summarize(arrays, 0)
        self.assertEqual(summary['values']['maximum_drawdown'], 0.5)
        self.assertEqual(summary['values']['zero_volume_fraction'], 0.75)
        self.assertEqual(summary['values']['longest_no_trade_streak'], 2)
        self.assertEqual(summary['values']['terminal_no_trade_streak'], 2)
        single = summarize(arrays, 3)
        self.assertIsNone(single['values']['log_return_volatility'])
        self.assertIn('log_return_volatility', single['undefined'])
        self.assertEqual(single['values']['maximum_drawdown'], 0)

    def test_no_trade_and_halted_days_remain_in_arrays(self):
        for scenario, seed in [(replace(SCENARIOS['baseline'], noise=1, chartists=0, fundamentalists=0, order_lifetime='day'), 1),
                               (SCENARIOS['circuit_breaker'], 18)]:
            np.random.seed(seed)
            result = simulate(scenario, days=12)
            try:
                arrays = daily_arrays(result.run_log, 12)
                self.assertEqual(len(arrays['price']), 12)
                if scenario.halt_check:
                    self.assertTrue(np.any(arrays['trading_skipped']))
                else:
                    self.assertTrue(np.all(arrays['volume'] == 0))
                    self.assertTrue(np.all(arrays['log_return'] == 0))
                self.assertEqual(arrays['previous_price'][0], next(
                    r['previous_price'] for r in result.run_log.iter_records('steps') if r['phase'] == 'trading'))
            finally:
                result.run_log.cleanup()

    def test_kernel_memory_and_file_limits(self):
        prefix = f"import runpy; bootstrap=runpy.run_path({str(CLI)!r})['bootstrap']; "
        memory = subprocess.run([sys.executable, '-c', prefix +
                                 'bootstrap(128*1024**2); bytearray(256*1024**2)'], capture_output=True, text=True)
        self.assertNotEqual(memory.returncode, 0)
        self.assertIn('MemoryError', memory.stderr)
        with tempfile.TemporaryDirectory() as root:
            output = str(Path(root) / 'limited')
            file_run = subprocess.run([sys.executable, '-c', prefix +
                                       f'bootstrap(128*1024**2, 1024); f=open({output!r}, "wb"); f.write(b"a"*2048); f.close()'],
                                      capture_output=True, text=True)
            self.assertNotEqual(file_run.returncode, 0)
            self.assertLessEqual(Path(output).stat().st_size, 1024)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='experiment-tests-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec_path = self.root / 'spec.json'

    def run_batch(self, directory='batch', spec=None, workers=2):
        write_json(self.spec_path, spec or specification())
        destination = self.root / directory
        code = execute_experiment(destination, spec=self.spec_path, limits=Limits(workers=workers),
                                  log_path=self.root / 'test.log')
        self.assertEqual(code, 0, (self.root / 'test.log').read_text())
        return destination

    def query(self, directory, sql, parameters=()):
        with sqlite3.connect(directory / 'ledger.sqlite') as db:
            return db.execute(sql, parameters).fetchall()

    def start(self, directory, spec):
        write_json(self.spec_path, spec)
        log = (self.root / 'test.log').open('w')
        self.addCleanup(log.close)
        process = subprocess.Popen([sys.executable, str(CLI), 'run', str(self.spec_path), str(directory)],
                                   stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        return process

    def wait_for(self, process, condition):
        end = time.monotonic() + 20
        while time.monotonic() < end:
            if condition():
                return
            if process.poll() is not None:
                self.fail((self.root / 'test.log').read_text())
            time.sleep(0.01)
        self.fail('Timed out waiting for runner state')

    def test_serial_parallel_and_direct_logs_match(self):
        serial = self.run_batch('serial', workers=1)
        self.spec_path.unlink()
        parallel = self.run_batch('parallel', workers=2)
        serial_manifest = read_json(serial / 'manifest.json')
        self.assertEqual(status(serial)['runs'], {'complete': 3})
        for run in serial_manifest['runs']:
            meta, arrays, summary = read_compact(serial / 'runs' / run['run_id'])
            other_meta, other_arrays, other_summary = read_compact(parallel / 'runs' / run['run_id'])
            self.assertEqual(summary, other_summary)
            self.assertEqual(meta, other_meta)
            np.random.seed(run['seed'])
            result = simulate(Scenario(**run['scenario']), days=run['days'])
            try:
                expected = daily_arrays(result.run_log, run['days'])
                for name, array in arrays.items():
                    np.testing.assert_array_equal(array, other_arrays[name])
                    np.testing.assert_array_equal(array, expected[name])
                self.assertEqual(meta['rng_state_after'][1], result.rng_state_after[1].tolist())
            finally:
                result.run_log.cleanup()
        attempts = self.query(parallel, 'SELECT memory_limit,file_limit,peak_rss_bytes,artifact_bytes FROM attempts')
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(memory > peak > 0 and size > 0 and file_limit > 0 for memory, file_limit, peak, size in attempts))

    def test_resume_skips_completed_and_recovers_publication_crash(self):
        directory = self.run_batch()
        # Simulate a crash after atomic publication but before the ledger transaction.
        with sqlite3.connect(directory / 'ledger.sqlite') as db:
            db.execute("UPDATE runs SET state='running'")
            db.execute("UPDATE attempts SET state='running', ended=NULL")
        before = {p: p.stat().st_mtime_ns for p in (directory / 'runs').rglob('*') if p.is_file()}
        code = execute_experiment(directory, log_path=self.root / 'test.log')
        self.assertEqual(code, 0)
        self.assertEqual(self.query(directory, 'SELECT attempts FROM runs'), [(1,), (1,), (1,)])
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        self.assertEqual(status(directory)['runs'], {'complete': 3})

    def test_corruption_requires_explicit_retry(self):
        directory = self.run_batch(spec=specification(seeds=(100,)))
        run_id = read_json(directory / 'manifest.json')['runs'][0]['run_id']
        (directory / 'runs' / run_id / 'daily.npz').write_bytes(b'broken')
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 2)
        self.assertEqual(status(directory)['runs'], {'failed': 1})
        self.assertEqual(execute_experiment(directory, retry_failed=True, log_path=self.root / 'test.log'), 0)
        self.assertEqual(self.query(directory, 'SELECT attempts FROM runs'), [(2,)])
        read_compact(directory / 'runs' / run_id)
        (directory / 'runs' / run_id / 'complete.json').write_text('[]')
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 2)
        self.assertEqual(status(directory)['runs'], {'failed': 1})

    def test_missing_ledger_rows_recover_from_immutable_manifest(self):
        directory = self.run_batch(spec=specification(seeds=(100,)))
        with sqlite3.connect(directory / 'ledger.sqlite') as db:
            db.execute('DELETE FROM attempts')
            db.execute('DELETE FROM runs')
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 0)
        self.assertEqual(status(directory)['runs'], {'complete': 1})
        self.assertEqual(self.query(directory, 'SELECT COUNT(*) FROM attempts'), [(0,)])

    def test_changed_manifest_and_provenance_rejected(self):
        directory = self.run_batch(spec=specification(seeds=(100,)))
        original = read_json(directory / 'manifest.json')
        mutated = deepcopy(original)
        mutated['specification']['days'] = 13
        (directory / 'manifest.json').write_text(json.dumps(mutated))
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 1)
        mutated = deepcopy(original)
        mutated['provenance']['execution']['python'] = 'different'
        (directory / 'manifest.json').write_text(json.dumps(mutated))
        (directory / 'manifest.sha256.json').write_text(json.dumps({'sha256': digest(mutated)}))
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 1)
        self.assertEqual(self.query(directory, 'SELECT attempts FROM runs'), [(1,)])

    def test_reader_does_not_require_current_execution_rules(self):
        directory = self.run_batch(spec=specification(seeds=(100,)))
        run = read_json(directory / 'manifest.json')['runs'][0]
        with patch('src.experiment_artifacts.effective_configuration', return_value={'changed': True}):
            meta, arrays, summary = read_compact(directory / 'runs' / run['run_id'])
        self.assertEqual(meta['run'], run)
        self.assertEqual(len(arrays['day']), run['days'])

    def test_exclusive_lock(self):
        directory = self.run_batch(spec=specification(seeds=(100,)))
        with experiment_lock(directory):
            self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 1)
            self.assertEqual(status(directory)['runs'], {'complete': 1})
        self.assertIn('experiment lock', (self.root / 'test.log').read_text())

    def test_signal_resume_and_staging_cleanup(self):
        directory = self.root / 'interrupted'
        process = self.start(directory, specification(days=250))
        self.wait_for(process, lambda: (self.root / 'test.log').exists() and
                      '"seed":101' in (self.root / 'test.log').read_text())
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 2)
        self.assertEqual(status(directory)['runs'].get('complete'), 1)
        self.assertIn('interrupted', status(directory)['runs'])
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 0)
        self.assertEqual(status(directory)['runs'], {'complete': 3})
        rows = self.query(directory, 'SELECT spec,attempts FROM runs')
        self.assertEqual(next(attempts for spec, attempts in rows if json.loads(spec)['seed'] == 100), 1)
        self.assertFalse(list((directory / 'staging').iterdir()))

    def test_coordinator_sigkill_does_not_leave_live_worker(self):
        directory = self.root / 'killed'
        process = self.start(directory, specification(seeds=(100,), days=1000))
        self.wait_for(process, lambda: (self.root / 'test.log').exists() and
                      'run_started' in (self.root / 'test.log').read_text())
        pid = self.query(directory, 'SELECT pid FROM attempts')[0][0]
        process.kill()
        process.wait(timeout=10)
        for _ in range(100):
            path = Path(f'/proc/{pid}/stat')
            if not path.exists() or path.read_text().split()[2] == 'Z':
                break
            time.sleep(0.02)
        else:
            self.fail('Orphan worker is still running')
        self.assertEqual(execute_experiment(directory, log_path=self.root / 'test.log'), 0)
        self.assertEqual(self.query(directory, 'SELECT attempts FROM runs'), [(2,)])

    def test_deadline_and_small_output_budget(self):
        write_json(self.spec_path, specification(seeds=(100,), days=1000))
        directory = self.root / 'deadline'
        self.assertEqual(execute_experiment(directory, spec=self.spec_path, limits=Limits(minutes=0.002),
                                           log_path=self.root / 'test.log'), 2)
        self.assertEqual(status(directory)['latest_invocation']['stop_reason'], 'time_limit')
        self.assertNotIn('complete', status(directory)['runs'])
        self.assertEqual(execute_experiment(self.root / 'too-small', spec=self.spec_path,
                                           limits=Limits(output_gib=0.001), log_path=self.root / 'test.log'), 1)
        self.assertFalse((self.root / 'too-small').exists())

    def test_worker_log_limit_is_bounded_and_stops_batch(self):
        write_json(self.spec_path, specification(seeds=(100, 101), days=1000))
        directory = self.root / 'limited-output'
        budget = (64 * 1024**2 + 512 * 1024) / GIB
        self.assertEqual(execute_experiment(directory, spec=self.spec_path, limits=Limits(output_gib=budget),
                                           log_path=self.root / 'test.log'), 2)
        self.assertEqual(status(directory)['latest_invocation']['stop_reason'], 'benchmark_failed')
        self.assertEqual(status(directory)['runs'], {'failed': 1, 'pending': 1})
        error = self.query(directory, 'SELECT error FROM runs WHERE state="failed"')[0][0]
        self.assertIn('File too large', error)
        self.assertLess(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()), budget * GIB)


if __name__ == '__main__':
    unittest.main(verbosity=2)
