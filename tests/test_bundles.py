"""Bundle round-trip, replay and CLI checks; run as a standalone script."""

from dataclasses import asdict, replace
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import numpy as np

from src import simulation, SCENARIOS
from src.bundles import save_bundle, replay_bundle, _json_safe
from test_core import assert_rng_equal


class BundleTests(unittest.TestCase):
    def test_custom_controls_round_trip_including_combined_evolution(self):
        controls = {'chartist_alpha': np.float32(0.3), 'fundamentalist_gamma': np.int64(4),
                    'closing_noise_scale': 0, 'dividend_mode': 'constant', 'order_lifetime': 'day'}
        cases = [{name: value} for name, value in controls.items()] + [controls]
        with tempfile.TemporaryDirectory() as root:
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    combined = len(overrides) > 1
                    scenario = replace(SCENARIOS['evolution' if combined else 'baseline'], **overrides)
                    np.random.seed(0)
                    result = simulation.simulate(scenario, days=102 if combined else 8)
                    self.addCleanup(result.run_log.cleanup)
                    directory = save_bundle(result, root, seed=0)
                    manifest = json.loads((directory / 'manifest.json').read_text())
                    config = manifest['configuration']
                    self.assertEqual(manifest['schema_version'], 1)
                    self.assertEqual(config['scenario'], asdict(scenario))
                    self.assertEqual(config['shared_parameters']['CHARTIST_ALPHA'], scenario.chartist_alpha)
                    self.assertEqual(config['shared_parameters']['FUNDAMENTALIST_GAMMA'], scenario.fundamentalist_gamma)
                    self.assertEqual(config['fixed_rules']['matching']['noise_scale'], scenario.closing_noise_scale)
                    self.assertEqual(config['fixed_rules']['matching']['order_lifetime'], scenario.order_lifetime)
                    self.assertEqual(config['fixed_rules']['dividends']['mode'], scenario.dividend_mode)
                    np.random.random(3)
                    before = np.random.get_state()
                    replay = replay_bundle(directory)
                    self.addCleanup(replay.run_log.cleanup)
                    assert_rng_equal(self, before, np.random.get_state())
                    self.assertEqual(replay.scenario, scenario)
                    self.assertEqual(replay.asset.price_history, result.asset.price_history)
                    self.assertEqual(replay.asset.dividend_history, result.asset.dividend_history)
                    self.assertEqual(replay.wealth_history, result.wealth_history)
                    self.assertEqual(replay.evolution_log, result.evolution_log)
                    self.assertEqual(replay.trader_counts_history, result.trader_counts_history)
                    self.assertEqual(replay.halt_events, result.halt_events)
                    self.assertEqual(_json_safe(asdict(replay.analysis)), _json_safe(asdict(result.analysis)))
                    for old, new in zip(result.traders, replay.traders):
                        self.assertEqual((old.type, old.cash, old.assets, old.active, old.wealth_history),
                                         (new.type, new.cash, new.assets, new.active, new.wealth_history))
                        if old.type == 'chartist':
                            self.assertEqual(old.alpha, new.alpha)
                        elif old.type == 'fundamentalist':
                            self.assertEqual(old.gamma, new.gamma)
                    assert_rng_equal(self, result.rng_state_after, replay.rng_state_after)
                    self.assertEqual(result.run_log.metadata(), replay.run_log.metadata())
                    for kind in result.run_log.metadata()['records']:
                        self.assertEqual(list(result.run_log.iter_records(kind)),
                                         list(replay.run_log.iter_records(kind)))

    def test_effective_configuration_mismatch_still_rejected(self):
        scenario = replace(SCENARIOS['baseline'], closing_noise_scale=0, order_lifetime='day')
        result = simulation.simulate(scenario, days=2)
        self.addCleanup(result.run_log.cleanup)
        with tempfile.TemporaryDirectory() as root:
            directory = save_bundle(result, root)
            manifest = json.loads((directory / 'manifest.json').read_text())
            manifest['configuration']['fixed_rules']['matching']['noise_scale'] = 0.05
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            before = np.random.get_state()
            with self.assertRaisesRegex(ValueError, 'Effective model settings differ'):
                replay_bundle(directory)
            assert_rng_equal(self, before, np.random.get_state())

    def test_full_round_trip_and_replay(self):
        with tempfile.TemporaryDirectory() as root:
            for seed in (0, None):
                if seed is not None:
                    np.random.seed(seed)
                result = simulation.simulate(SCENARIOS['evolution'], days=102)
                # Exercise delayed analysis with a different RNG state, too.
                np.random.random(3)
                analysis = simulation.analyze_result(result)
                state = np.random.get_state()
                directory = save_bundle(result, root, seed=seed)
                assert_rng_equal(self, state, np.random.get_state())
                self.assertTrue((directory / 'COMPLETE').is_file())
                self.assertEqual(len(list(directory.glob('*.png'))), 4)
                manifest = json.loads((directory / 'manifest.json').read_text())
                self.assertEqual(manifest['seed'], seed)
                with np.load(directory / 'histories.npz', allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['price'], result.asset.price_history)
                    np.testing.assert_array_equal(arrays['dividend'], result.asset.dividend_history)
                    for key, values in result.wealth_history.items():
                        np.testing.assert_array_equal(arrays[key], values)
                    offsets, flat = arrays['trader_wealth_offsets'], arrays['trader_wealth']
                    for i, trader in enumerate(result.traders):
                        np.testing.assert_array_equal(flat[offsets[i]:offsets[i + 1]], trader.wealth_history)
                states = json.loads((directory / 'traders.json').read_text())
                for i, (state_row, trader) in enumerate(zip(states, result.traders)):
                    self.assertEqual(state_row, _json_safe({
                        'index': i, 'type': trader.type, 'cash': trader.cash, 'assets': trader.assets,
                        'active': trader.active, 'initial_wealth': result.initial_wealths.get(id(trader)),
                    }))
                replay = replay_bundle(directory)
                assert_rng_equal(self, state, np.random.get_state())
                self.assertEqual(replay.asset.price_history, result.asset.price_history)
                self.assertEqual(replay.asset.dividend_history, result.asset.dividend_history)
                self.assertEqual(replay.wealth_history, result.wealth_history)
                self.assertEqual(replay.evolution_log, result.evolution_log)
                for old, new in zip(result.traders, replay.traders):
                    self.assertEqual((old.cash, old.assets, old.active, old.wealth_history),
                                     (new.cash, new.assets, new.active, new.wealth_history))
                self.assertEqual(_json_safe(asdict(replay.analysis)), _json_safe(asdict(analysis)))
                assert_rng_equal(self, replay.rng_state_after, result.rng_state_after)

    def test_unique_directories_and_nonfinite_json(self):
        result = simulation.simulate(days=1)
        simulation.analyze_result(result)
        with tempfile.TemporaryDirectory() as root:
            first = save_bundle(result, root)
            second = save_bundle(result, root)
            self.assertNotEqual(first, second)
            analysis = json.loads((first / 'analysis.json').read_text(),
                                  parse_constant=lambda token: self.fail(f'Nonstandard JSON: {token}'))
            self.assertIsNone(analysis['metrics']['sampen']['value'])
            self.assertEqual(analysis['metrics']['sampen']['status'], 'insufficient_input')

    def test_interrupted_bundle_and_provenance_mismatch(self):
        result = simulation.simulate(days=1)
        with tempfile.TemporaryDirectory() as root:
            with patch('src.bundles._write_histories', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_bundle(result, root)
            directory = next(Path(root).iterdir())
            self.assertFalse((directory / 'COMPLETE').exists())
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                replay_bundle(directory)
            good = save_bundle(result, root)
            manifest = json.loads((good / 'manifest.json').read_text())
            manifest['provenance']['source_sha256'] = {}
            (good / 'manifest.json').write_text(json.dumps(manifest))
            before = np.random.get_state()
            with self.assertRaisesRegex(ValueError, 'Source files differ'):
                replay_bundle(good)
            assert_rng_equal(self, before, np.random.get_state())

    def test_missing_git_metadata(self):
        from src.bundles import _provenance
        with patch('src.bundles.subprocess.check_output', side_effect=FileNotFoundError):
            self.assertIsNone(_provenance()['git']['revision'])


class CliTests(unittest.TestCase):
    def test_omitted_controls_preserve_selected_scenario_without_mutating_registry(self):
        from experiments import run_scenario
        scenario = replace(SCENARIOS['baseline'], chartist_alpha=0.25, fundamentalist_gamma=5,
                           closing_noise_scale=0, dividend_mode='constant', order_lifetime='day')
        argv = ['run_scenario.py', 'baseline', '--days', '1', '--seed', '0',
                '--chartist-alpha', '0.4', '--bundle', 'unused', '--no-show']
        with patch.dict(SCENARIOS, baseline=scenario), patch.object(sys, 'argv', argv), \
                patch('src.simulation.simulate') as simulate, \
                patch('src.simulation.analyze_result'), patch('src.simulation.print_analysis'), \
                patch('src.bundles.save_bundle'), contextlib.redirect_stdout(io.StringIO()) as output:
            run_scenario.main()
            resolved = simulate.call_args.args[0]
            self.assertEqual(resolved, replace(scenario, chartist_alpha=0.4))
            self.assertIs(SCENARIOS['baseline'], scenario)
            self.assertIn('chartist_alpha=0.4', output.getvalue())
            self.assertIn('order_lifetime=day', output.getvalue())

    def test_custom_controls_cli_bundle_replays(self):
        expected = {'chartist_alpha': 0.3, 'fundamentalist_gamma': 4.0, 'closing_noise_scale': 0.0,
                    'dividend_mode': 'constant', 'order_lifetime': 'day'}
        with tempfile.TemporaryDirectory() as root:
            flags = [argument for key, value in expected.items()
                     for argument in ('--' + key.replace('_', '-'), str(value))]
            process = subprocess.run([sys.executable, 'experiments/run_scenario.py', 'baseline',
                                      '--days', '6', '--seed', '0', '--bundle', root, '--no-show', *flags],
                                     cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('CONTROLS:', process.stdout)
            directory = next(Path(root).iterdir())
            replay = replay_bundle(directory)
            self.addCleanup(replay.run_log.cleanup)
            for name, value in expected.items():
                self.assertEqual(getattr(replay.scenario, name), value)
            self.assertEqual(replay.asset.dividend_history, [0.016] * 27)
            with np.load(directory / 'histories.npz', allow_pickle=False) as histories:
                np.testing.assert_array_equal(histories['price'], replay.asset.price_history)

    def test_invalid_arguments_create_no_output(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / 'must-not-exist'
            for flags in (['--seed', '-1'], ['--seed', str(2**32)], ['--days', '0'],
                          ['--save', root], ['--chartist-alpha', '0'], ['--chartist-alpha', '1.1'],
                          ['--chartist-alpha', 'nan'], ['--chartist-alpha', 'True'],
                          ['--fundamentalist-gamma', '0'], ['--fundamentalist-gamma', 'inf'],
                          ['--closing-noise-scale', '-0.1'], ['--closing-noise-scale', 'nan'],
                          ['--closing-noise-scale', 'inf'], ['--closing-noise-scale', 'nope'],
                          ['--dividend-mode', 'fixed'], ['--order-lifetime', 'week']):
                process = subprocess.run([sys.executable, 'experiments/run_scenario.py', 'baseline',
                                          '--bundle', str(destination), '--no-show', *flags],
                                         cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
