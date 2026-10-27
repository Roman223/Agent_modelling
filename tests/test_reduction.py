"""Standalone rule, invariant, provenance, and recovery checks for task 4."""

from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import signal
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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/abm-reduction-matplotlib')

import numpy as np

from src.asset import Asset
from src.order_book import OrderBook
from src.reduction import (DIVIDEND, Order, ReductionConfig, ReductionState, _match, _submit,
                           conditional_order, initial_state, quote_grid, simulate_reduction,
                           step_hard, validate_state)
from src.scenarios import RISK_FREE_RATE
from src.traders import ChartistTrader, FundamentalistTrader, NoiseTrader, Trader
from src.atlas import make_spec
from src.experiment_artifacts import (digest, execution_fingerprint, expand_spec, read_json,
                                      save_compact, write_json)
from src.reduction_study import (comparison_summary, fingerprint, generate_study, make_reduction_manifest,
                                 read_reduction, run_identity, save_reduction, select_anchors, state_dict,
                                 state_from_dict, study_status)
from src.reduction_plotting import aggregate_records, collect_study
from src.scenarios import Scenario
from src.simulation import simulate


class RuleTests(unittest.TestCase):
    def test_conditional_rules_match_original_traders(self):
        config = ReductionConfig()
        memories = [tuple([100.] * 20), tuple(np.linspace(95, 105, 20)),
                    tuple(np.linspace(105, 95, 20)), tuple([100.] * 19 + [110.]),
                    tuple([100.] * 19 + [90.])]
        for kind in range(3):
            for cash, shares in ((100000., 1000.), (0., 10.), (10., 0.), (0., 0.),
                                 (100000., 0.), (10000., 0.3), (105., 1.)):
                for memory in memories:
                    for u in (0., 0.2, 0.5, 0.999):
                        for branch in (('buy', 'sell') if kind == 0 else ('signal',)):
                            with self.subTest(kind=kind, cash=cash, shares=shares, u=u, branch=branch, p=memory[-1]):
                                trader = (NoiseTrader(cash, shares) if kind == 0 else
                                          ChartistTrader(cash, shares, alpha=config.alpha) if kind == 1 else
                                          FundamentalistTrader(cash, shares, gamma=config.gamma, rf=RISK_FREE_RATE))
                                asset = Asset()
                                asset.price_history = list(memory)
                                asset.current_price = memory[-1]
                                asset.dividend_history = [DIVIDEND, DIVIDEND]
                                draws = [0.8 if branch == 'buy' else 0.2, u] if kind == 0 else [u]
                                with patch('numpy.random.rand', side_effect=draws):
                                    expected = trader.make_decision(asset)
                                actual = conditional_order(kind, cash, shares, memory, config, u, branch)
                                self.assertEqual(actual or (None, None, None), expected)

    def test_noise_participation_and_quote_measure(self):
        config = ReductionConfig(counts=(50, 0, 0))
        state = initial_state([100.] * 20, config)
        bids, asks, masses = _submit(state, config)
        np.testing.assert_allclose(masses[0], [20, 20])
        self.assertEqual(len(bids), 16)
        self.assertEqual(len(asks), 16)
        for points in (8, 16, 32):
            grid = quote_grid(points)
            self.assertAlmostEqual(sum(w for _, w in grid), 1)
            self.assertAlmostEqual(sum(u * w for u, w in grid), .5)
            self.assertAlmostEqual(sum(u*u * w for u, w in grid), 1/3)

    def test_switch_equality_and_fundamentalist_clipping(self):
        config = ReductionConfig()
        self.assertEqual(conditional_order(1, 1000, 10, [100.] * 20, config)[0], 'sell')
        self.assertIsNone(conditional_order(2, 1000, 10, [100.] * 20, config))
        self.assertEqual(conditional_order(2, 1000, 10, [100.] * 19 + [90.], config)[0], 'sell')

    def test_match_matches_legacy_resource_clipping(self):
        for cash, shares in ((250., 1.), (1000., 7.), (0., 1.), (1000., 0.)):
            buyer, seller = Trader('noise', cash, 0), Trader('chartist', 0, shares)
            book = OrderBook(closing_noise_scale=0)
            book.add_order('buy', 101, 5, id(buyer))
            book.add_order('sell', 99, 7, id(seller))
            asset = Asset()
            trades = book.match_orders([buyer, seller], asset)
            c, s = [cash, 0., 0.], [0., shares, 0.]
            out = _match(c, s, [Order(101, 5, 0)], [Order(99, 7, 1)])
            np.testing.assert_allclose(c, [buyer.cash, seller.cash, 0])
            np.testing.assert_allclose(s, [buyer.assets, seller.assets, 0])
            self.assertEqual(out[2], sum(t[1] for t in trades))
            self.assertEqual([(o.price, o.quantity) for o in out[0]], [(o[0], o[1]) for o in book.buy_orders])
            self.assertEqual([(o.price, o.quantity) for o in out[1]], [(o[0], o[1]) for o in book.sell_orders])

    def test_price_time_priority_partial_fills_and_within_strategy(self):
        c, s = [1000., 1000., 0.], [0., 0., 7.]
        out = _match(c, s, [Order(101, 5, 0), Order(101, 5, 1)], [Order(99, 7, 2)])
        self.assertEqual(s, [5, 2, 0])
        self.assertEqual(out[0], (Order(101, 3, 1),))
        c, s = [1000., 0., 0.], [10., 0., 0.]
        out = _match(c, s, [Order(101, 5, 0)], [Order(99, 5, 0)])
        self.assertEqual(c, [1000, 0, 0])
        self.assertEqual(s, [10, 0, 0])
        self.assertEqual(out[5], 5)

    def test_expiry_is_after_diagnostics(self):
        config = ReductionConfig(counts=(0, 20, 0))
        state = initial_state([100.] * 20, config)
        after, diag = step_hard(state, config)
        self.assertEqual(len(after.asks), 0)
        self.assertGreater(diag['outstanding_sell_orders'], 0)
        persistent = replace(config, order_lifetime='persistent')
        after, diag = step_hard(state, persistent)
        self.assertEqual(len(after.asks), diag['outstanding_sell_orders'])

    def test_no_trade_freezes_memory(self):
        config = ReductionConfig(counts=(0, 20, 0))
        state = initial_state(np.linspace(90, 100, 20), config)
        after, diag = step_hard(state, config)
        self.assertEqual(after.memory, state.memory)
        self.assertEqual(diag['volume'], 0)
        self.assertFalse(diag['accepted_price_update'])

    def test_unchanged_price_trade_advances_memory(self):
        config = ReductionConfig(counts=(1, 0, 0), quadrature_points=1)
        state = ReductionState((1000., 0., 0.), (10., 0., 0.), tuple([90.] * 19 + [100.]))
        after, diag = step_hard(state, config)
        self.assertEqual(after.price, state.price)
        self.assertGreater(diag['volume'], 0)
        self.assertEqual(after.memory[-2:], (100., 100.))

    def test_tiny_fractional_flow_is_not_rounded_to_stall(self):
        config = ReductionConfig(counts=(1, 0, 0), quadrature_points=1)
        state = ReductionState((1000., 0., 0.), (.01, 0., 0.), tuple([100.] * 20))
        _, diag = step_hard(state, config)
        self.assertGreater(diag['volume'], 0)
        self.assertLess(diag['volume'], 1)

    def test_pure_deterministic_conservative_runs(self):
        for lifetime in ('day', 'persistent'):
            config = ReductionConfig(order_lifetime=lifetime, quadrature_points=8)
            state = initial_state(np.linspace(98, 100, 20), config)
            snapshot = copy.deepcopy(state)
            np.random.seed(3)
            before = np.random.get_state()
            one = simulate_reduction(state, config, days=100)
            after = np.random.get_state()
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)
            np.random.seed(9)
            two = simulate_reduction(state, config, days=100)
            self.assertEqual(state, snapshot)
            for key in one.arrays:
                np.testing.assert_array_equal(one.arrays[key], two.arrays[key])
            np.testing.assert_allclose(one.arrays['total_cash'].sum(axis=1), 38_000_000, rtol=1e-12)
            np.testing.assert_allclose(one.arrays['total_assets'].sum(axis=1), 85_000, rtol=1e-12)
            np.testing.assert_array_equal(one.arrays['previous_price'][1:], one.arrays['price'][:-1])

    def test_invalid_inputs_and_absent_strategy(self):
        for kwargs in ({'counts': (0, 0, 0)}, {'counts': (True, 0, 1)}, {'alpha': 0},
                       {'gamma': float('nan')}, {'window': 0}, {'quadrature_points': 0},
                       {'order_lifetime': 'forever'}):
            with self.assertRaises(ValueError):
                ReductionConfig(**kwargs)
        with self.assertRaises(ValueError):
            initial_state([100.], ReductionConfig())
        with self.assertRaises(ValueError):
            step_hard(ReductionState((-1, 0, 0), (0, 0, 0), (100.,) * 20))
        state = initial_state([100.] * 20, ReductionConfig(counts=(0, 1, 0)))
        self.assertEqual(state.cash[0], 0)
        with self.assertRaises(ValueError):
            simulate_reduction(state, days=True)


class ArtifactTests(unittest.TestCase):
    def run_record(self):
        config = ReductionConfig(quadrature_points=8, order_lifetime='persistent')
        state = initial_state(np.linspace(98, 100, 20), config)
        run = json.loads(json.dumps({'config': asdict(config), 'initial_state': state_dict(state),
                                    'days': 12, 'burn_in': 6, 'initial_condition_seed': 100, 'cohort': 'historical'}))
        execution = fingerprint()
        run['run_id'] = run_identity(run, execution)
        return state, config, run, execution

    def test_roundtrip_and_corruption(self):
        state, config, run, execution = self.run_record()
        result = simulate_reduction(state, config, days=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run'
            save_reduction(path, result, run, execution)
            meta, arrays, summary = read_reduction(path, run, execution)
            for key in result.arrays:
                np.testing.assert_array_equal(arrays[key], result.arrays[key])
            self.assertEqual(state_from_dict(json.loads(json.dumps(state_dict(result.final_state)))), result.final_state)
            with self.assertRaises(ValueError):
                read_reduction(path, run, {'changed': True})
            with (path / 'daily.npz').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaises(ValueError):
                read_reduction(path)

    def test_state_and_identity_validation(self):
        state, config, run, execution = self.run_record()
        self.assertNotEqual(run_identity({**run, 'days': 13}, execution), run['run_id'])
        self.assertNotEqual(run_identity(run, {'changed': True}), run['run_id'])
        with self.assertRaises(ValueError):
            state_from_dict({'cash': []})
        invalid = replace(state, bids=(Order(100, 1, 0), Order(101, 1, 0)))
        with self.assertRaises(ValueError):
            validate_state(invalid, config)
        with self.assertRaises(ValueError):
            validate_state(replace(state, asks=(Order(100, -1, 0),)), config)

    def test_subshare_and_stall_summaries(self):
        state, config, _, _ = self.run_record()
        result = simulate_reduction(state, config, days=120)
        arrays = result.arrays
        arrays['volume'][:] = 0
        summary = comparison_summary(arrays, 0)
        self.assertTrue(summary['values']['terminal_stall'])
        self.assertIsNone(summary['values']['positive_volume_volatility'])
        arrays['volume'][:] = .1
        summary = comparison_summary(arrays, 0)
        self.assertFalse(summary['values']['terminal_stall'])
        self.assertEqual(summary['values']['subshare_volume_fraction'], 1)


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='abm-reduction-tests-')
        cls.root = Path(cls.temporary.name)
        cls.atlas = cls.root / 'atlas'
        cls.atlas.mkdir()
        spec = make_spec(seeds=(100,), days=12, burn_in=6)
        spec['cells'] = [c for c in spec['cells'] if c['name'].startswith(('grid-4-3-', 'grid-6-0-v6-', 'grid-6-0-v7-'))]
        execution = execution_fingerprint()
        runs = expand_spec(spec, execution)
        manifest = {'specification': spec, 'provenance': {'execution': execution}, 'runs': runs}
        write_json(cls.atlas / 'manifest.json', manifest)
        write_json(cls.atlas / 'manifest.sha256.json', {'sha256': digest(manifest)})
        (cls.atlas / 'runs').mkdir()
        for run in runs:
            np.random.seed(run['seed'])
            result = simulate(Scenario(**run['scenario']), days=run['days'])
            try:
                save_compact(result, run, cls.atlas / 'runs' / run['run_id'], execution)
            finally:
                result.run_log.cleanup()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_coordinate_selection_and_no_rounded_alpha(self):
        manifest = read_json(self.atlas / 'manifest.json')
        selected = select_anchors(manifest, [100])
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected['liquidity-v6', 100]['scenario']['chartist_alpha'], .19999999999999998)
        with self.assertRaises(ValueError):
            select_anchors(manifest, [999])
        with self.assertRaises(ValueError):
            generate_study(self.atlas, self.root / 'overlap', historical_seeds=[100], fresh_seeds=[100])

    def test_cli_interrupt_resume_and_published_reuse(self):
        root = self.root / 'study'
        generate_study(self.atlas, root, historical_seeds=[100], fresh_seeds=[110], points=[8])
        cli = [sys.executable, str(ROOT / 'experiments/run_reduction.py')]
        with (self.root / 'execution.log').open('w') as log:
            process = subprocess.Popen([*cli, 'run', str(root), '--workers', '1', '--minutes', '.004'],
                                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            self.assertEqual(process.wait(timeout=30), 2)
            code = subprocess.call([*cli, 'resume', str(root), '--workers', '2', '--minutes', '2'],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            if code:
                self.fail((self.root / 'execution.log').read_text())
            status = study_status(root)
            self.assertEqual(status['fresh_abm']['runs'], {'complete': 10})
            self.assertEqual(status['reduction']['runs'], {'complete': 8})
            manifest = make_reduction_manifest(root, read_json(root / 'manifest.json'))
            before = {r['run_id']: (root / 'reduction/runs' / r['run_id'] / 'complete.json').stat().st_mtime_ns for r in manifest['runs']}
            self.assertEqual(subprocess.call([*cli, 'resume', str(root), '--workers', '1'], stdout=log, stderr=subprocess.STDOUT), 0)
            for run in manifest['runs']:
                path = root / 'reduction/runs' / run['run_id']
                self.assertEqual((path / 'complete.json').stat().st_mtime_ns, before[run['run_id']])
                _, arrays, _ = read_reduction(path)
                direct = simulate_reduction(state_from_dict(run['initial_state']), ReductionConfig(**run['config']), days=12)
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], direct.arrays[key])
            _, _, records, _ = collect_study(root)
            self.assertEqual(len(records), 28)
            groups = aggregate_records(records)
            self.assertEqual(sum(len(g['seeds']) for g in groups), len(records))
            # Interrupted reduction work is recovered from its initial state.
            import sqlite3
            import shutil
            run = manifest['runs'][0]
            shutil.rmtree(root / 'reduction/runs' / run['run_id'])
            with sqlite3.connect(root / 'reduction/ledger.sqlite') as db:
                db.execute("UPDATE runs SET state='running' WHERE run_id=?", (run['run_id'],))
            stage = root / 'reduction/staging' / run['run_id']
            stage.mkdir()
            (stage / 'partial').write_text('incomplete')
            self.assertEqual(subprocess.call([*cli, 'resume', str(root)], stdout=log, stderr=subprocess.STDOUT), 0)
            self.assertFalse(stage.exists())
            read_reduction(root / 'reduction/runs' / run['run_id'])
            # Corruption is a failure until explicit retry, never silently reused.
            path = root / 'reduction/runs' / run['run_id'] / 'summary.json'
            path.write_text('{}')
            self.assertEqual(subprocess.call([*cli, 'resume', str(root)], stdout=log, stderr=subprocess.STDOUT), 2)
            self.assertEqual(subprocess.call([*cli, 'resume', str(root), '--retry-failed'], stdout=log, stderr=subprocess.STDOUT), 0)
            self.assertEqual(subprocess.call([*cli, 'render', str(root)], stdout=log, stderr=subprocess.STDOUT), 0)
            self.assertEqual(len(list((root / 'reports').rglob('*.png'))), 6)
            # A modified immutable study is rejected before execution.
            study = read_json(root / 'manifest.json')
            study['execution']['python'] = 'changed'
            (root / 'manifest.json').write_text(json.dumps(study))
            write = root / 'manifest.sha256.json'
            write.write_text(json.dumps({'sha256': digest(study)}))
            self.assertEqual(subprocess.call([*cli, 'resume', str(root)], stdout=log, stderr=subprocess.STDOUT), 1)


if __name__ == '__main__':
    unittest.main()
