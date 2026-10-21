"""Bundle round-trip, replay and CLI checks; run as a standalone script."""

import copy
from dataclasses import asdict
import json
import os
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
    def test_invalid_arguments_create_no_output(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / 'must-not-exist'
            for flags in (['--seed', '-1'], ['--seed', str(2**32)], ['--days', '0'],
                          ['--save', root]):
                process = subprocess.run([sys.executable, 'experiments/run_scenario.py', 'baseline',
                                          '--bundle', str(destination), '--no-show', *flags],
                                         cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
