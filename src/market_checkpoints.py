"""Immutable market snapshots and exact research continuation of the original ABM.

Snapshots retain individual portfolios and ordered owner-linked queues. Research
entry points restore the caller's RNG; the legacy simulation API is unchanged.
"""

from dataclasses import asdict, dataclass
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from src.asset import Asset
from src.experiment_artifacts import canonical, digest
from src.order_book import OrderBook
from src.run_logging import LogOptions, RunLog, json_safe
from src.scenarios import Scenario
from src.simulation import _Market, _initialize_market, _advance_market
from src.traders import NoiseTrader, ChartistTrader, FundamentalistTrader

CLASSES = {'noise': NoiseTrader, 'chartist': ChartistTrader, 'fundamentalist': FundamentalistTrader}


def validate_scenario(scenario):
    scenario.validate()
    if scenario.evolution or scenario.circuit_breaker_asset or scenario.halt_check:
        raise ValueError('Checkpoints require fixed populations without circuit breakers')


def encode_rng(state):
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def decode_rng(state):
    if (not isinstance(state, list) or len(state) != 5 or state[0] != 'MT19937'
            or len(state[1]) != 624 or any(type(x) is not int or not 0 <= x < 2**32 for x in state[1])
            or type(state[2]) is not int or not 0 <= state[2] <= 624
            or state[3] not in (0, 1) or not np.isfinite(state[4])):
        raise ValueError('Invalid checkpoint RNG state')
    return state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]


def _pack(value):
    # Python 3.12 sums Python floats differently from NumPy scalars. Preserve
    # scalar types as well as bit values, including outstanding limit prices.
    if isinstance(value, np.generic):
        return ['__numpy_scalar__', str(value.dtype), value.item()]
    if isinstance(value, dict):
        return {key: ([float(x) for x in item] if key == 'wealth_history' else _pack(item))
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_pack(item) for item in value]
    return value


def _unpack(value):
    if isinstance(value, list):
        if len(value) == 3 and isinstance(value[0], str) and value[0] == '__numpy_scalar__':
            if value[1] not in ('float64', 'float32', 'int64', 'int32', 'bool'):
                raise ValueError('Unsupported checkpoint scalar dtype')
            return np.dtype(value[1]).type(value[2])
        return [_unpack(item) for item in value]
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class MarketCheckpoint:
    """Canonical JSON is immutable; data() returns an independent decoded copy."""

    payload: str

    def __post_init__(self):
        encoded = json.loads(self.payload)
        validate_checkpoint(_unpack(encoded))
        object.__setattr__(self, 'payload', canonical(encoded))

    def data(self):
        return _unpack(json.loads(self.payload))

    @classmethod
    def from_data(cls, data):
        return cls(canonical(_pack(data)))

    @property
    def sha256(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    @property
    def completed_days(self):
        return self.data()['completed_days']


def validate_checkpoint(data):
    required = {'schema_version', 'scenario', 'completed_days', 'asset', 'traders', 'bids', 'asks',
                'history', 'initial_wealths', 'rng_state_before', 'rng_state', 'no_trade_streak'}
    if set(data) != required or data['schema_version'] != 1:
        raise ValueError('Invalid checkpoint layout')
    scenario = Scenario(**data['scenario'])
    validate_scenario(scenario)
    day = data['completed_days']
    if type(day) is not int or day < 0 or type(data['no_trade_streak']) is not int or not 0 <= data['no_trade_streak'] <= day:
        raise ValueError('Invalid checkpoint chronology')
    traders, asset = data['traders'], data['asset']
    expected = (['noise'] * scenario.noise + ['chartist'] * scenario.chartists
                + ['fundamentalist'] * scenario.fundamentalists)
    if [t.get('type') for t in traders] != expected or len(data['initial_wealths']) != len(traders):
        raise ValueError('Checkpoint population mismatch')
    for t in traders:
        if (not np.isfinite([t['cash'], t['assets']]).all() or t['cash'] < 0 or t['assets'] < 0
                or t['assets'] != int(t['assets']) or type(t['active']) is not bool
                or len(t['wealth_history']) != day + 1 or not np.isfinite(t['wealth_history']).all()):
            raise ValueError('Invalid trader state or noninteger shares')
    if (not asset['price_history'] or not np.isfinite(asset['price_history']).all()
            or min(asset['price_history']) <= 0 or asset['current_price'] != asset['price_history'][-1]
            or len(asset['dividend_history']) != day + 21
            or not np.isfinite(asset['dividend_history']).all() or min(asset['dividend_history']) < 0
            or len(asset['price_history']) > day + 21):
        raise ValueError('Invalid asset history')
    for side, reverse in (('bids', True), ('asks', False)):
        orders = data[side]
        if scenario.order_lifetime == 'day' and orders:
            raise ValueError('Day-order checkpoint must have empty queues')
        for order in orders:
            if (len(order) != 3 or not np.isfinite(order[:2]).all() or min(order[:2]) <= 0
                    or type(order[2]) is not int or not 0 <= order[2] < len(traders)):
                raise ValueError('Invalid checkpoint order')
        if any((a[0] < b[0] if reverse else a[0] > b[0]) for a, b in zip(orders, orders[1:])):
            raise ValueError('Checkpoint queue priority changed')
    decode_rng(data['rng_state'])
    decode_rng(data['rng_state_before'])


def capture_market(market, completed_days, no_trade_streak=0):
    ids = {id(t): i for i, t in enumerate(market.traders)}
    data = {'schema_version': 1, 'scenario': asdict(market.scenario), 'completed_days': completed_days,
            'asset': {k: v for k, v in vars(market.asset).items() if k not in ('log', 'event', 'run_log')},
            'traders': [vars(t) for t in market.traders],
            'bids': [(p, q, ids[owner]) for p, q, owner in market.book.buy_orders],
            'asks': [(p, q, ids[owner]) for p, q, owner in market.book.sell_orders],
            'history': market.history, 'initial_wealths': [market.initial_wealths[id(t)] for t in market.traders],
            'rng_state_before': encode_rng(market.rng_state_before), 'rng_state': encode_rng(np.random.get_state()),
            'no_trade_streak': no_trade_streak}
    return MarketCheckpoint.from_data(data)


def restore_market(checkpoint, run_log):
    data = checkpoint.data()
    scenario = Scenario(**data['scenario'])
    asset = Asset(circuit_breaker=False, dividend_mode=scenario.dividend_mode, log=lambda *args: None)
    vars(asset).update(data['asset'])
    asset.run_log = run_log
    traders = []
    for values in data['traders']:
        trader = CLASSES[values['type']].__new__(CLASSES[values['type']])
        vars(trader).update(values)
        traders.append(trader)
    book = OrderBook(closing_noise_scale=scenario.closing_noise_scale, order_lifetime=scenario.order_lifetime)
    book.buy_orders = [(p, q, id(traders[owner])) for p, q, owner in data['bids']]
    book.sell_orders = [(p, q, id(traders[owner])) for p, q, owner in data['asks']]
    run_log.register(traders)
    initial = {id(t): w for t, w in zip(traders, data['initial_wealths'])}
    market = _Market(scenario, asset, traders, book, data['history'], initial, decode_rng(data['rng_state_before']))
    return market


def dynamics_digest(checkpoint):
    """Exclude reporting-only final-day samples; retain every decision/RNG input."""
    data = checkpoint.data()
    data.pop('history')
    data['asset'].pop('trader_counts_history', None)
    return digest(_pack(data))


def write_checkpoint(path, checkpoint):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
            stream.write(checkpoint.payload.encode())


def read_checkpoint(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return MarketCheckpoint(stream.read())


@dataclass
class CheckpointRun:
    final: MarketCheckpoint
    checkpoints: dict
    checks: dict
    arrays: dict


def run_checkpointed(scenario, *, seed, days=2000, checkpoint_days=(), check_days=()):
    """Run an unchanged ABM with observations, preserving the caller's RNG."""
    validate_scenario(scenario)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    return _research_run(scenario, seed, days, checkpoint_days, check_days)


def continue_checkpoint(checkpoint, *, days=1000):
    """Resume from its exact RNG and queues; compression is a separate operation."""
    if not isinstance(checkpoint, MarketCheckpoint):
        raise ValueError('Expected a MarketCheckpoint')
    return _research_run(None, None, days, (), (), checkpoint)


def _research_run(scenario, seed, days, checkpoint_days, check_days, checkpoint=None):
    from src.heterogeneity import ObservationCollector
    if type(days) is not int or days < 1:
        raise ValueError('days must be positive')
    start = checkpoint.completed_days if checkpoint else 0
    requested = set(checkpoint_days) | set(check_days)
    if any(type(d) is not int or not start < d <= start + days for d in requested):
        raise ValueError('Checkpoint/check days must lie in the run interval')
    caller = np.random.get_state()
    log = RunLog(LogOptions())
    try:
        if checkpoint is None:
            np.random.seed(seed)
            market = _initialize_market(scenario, False, log)
            streak = 0
        else:
            market = restore_market(checkpoint, log)
            np.random.set_state(decode_rng(checkpoint.data()['rng_state']))
            streak = checkpoint.data()['no_trade_streak']
        observer = ObservationCollector(days, start, streak)
        checkpoints, checks = {}, {}
        for day in range(start, start + days):
            _advance_market(market, day, start + days, False, log, observer)
            if day + 1 in requested:
                current = capture_market(market, day + 1, observer.streak)
                if day + 1 in checkpoint_days:
                    checkpoints[day + 1] = current
                if day + 1 in check_days:
                    checks[day + 1] = dynamics_digest(current)
        final = capture_market(market, start + days, observer.streak)
        return CheckpointRun(final, checkpoints, checks, observer.arrays())
    finally:
        log.cleanup()
        np.random.set_state(caller)
