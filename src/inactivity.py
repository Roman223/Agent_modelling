"""Observed inactivity and resource-conserving interventions on the original ABM.

Trading days in this interface are one-based. Zero executions, not zero returns,
define inactivity. Observations precede day-order expiry; snapshots follow it.
"""

from dataclasses import dataclass
from math import fsum

import numpy as np

from src.market_checkpoints import (MarketCheckpoint, capture_market, restore_market,
                                    encode_rng, decode_rng, dynamics_digest, validate_scenario)
from src.run_logging import LogOptions, RunLog
from src.simulation import _initialize_market, _advance_market, TRADER_TYPES

REASONS = ('inactive', 'no_cash_and_no_holdings', 'neutral_draw', 'no_cash', 'no_holdings',
           'insufficient_cash', 'random_buy', 'random_sell', 'insufficient_history',
           'allocation_below_one_share', 'price_above_average', 'sell_fallback',
           'excess_return_below_threshold', 'target_met_or_resources_unavailable',
           'below_target_holdings', 'above_target_holdings')
MATCH_REASONS = ('empty_both', 'empty_bids', 'empty_asks', 'uncrossed', 'missing_owner',
                 'nonpositive_quantity', 'execution_constraint')
MATCH_COUNTS = ('dropped_bids', 'dropped_asks', 'rationed_bids', 'rationed_asks')
ARMS = tuple(f'{q}{m}{p}' for q in (0, 1) for m in (0, 1) for p in (0, 1))
PRIMARY_FIELDS = ('day', 'price', 'previous_price', 'volume', 'same_owner_volume',
                  'distinct_volume', 'no_trade_streak', 'total_cash', 'total_assets')
GROUP_FIELDS = ('total_cash', 'total_assets', 'total_wealth', 'active_count',
                'no_holdings_count', 'can_buy_spot_count')
SCALAR_FIELDS = ('price', 'previous_price', 'volume', 'same_owner_volume', 'distinct_volume',
                 'dividend', 'accepted_price_count', 'decision_sma', 'decision_previous_price',
                 'best_bid', 'best_ask', 'bid_count', 'ask_count', 'bid_quantity', 'ask_quantity',
                 'no_trade_streak', 'trade_count', 'match_stop')


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def spells(volume, *, threshold=100, start=1):
    """All zero-volume spells; right censor only at the observation boundary."""
    positive_integer(threshold, 'threshold')
    positive_integer(start, 'start')
    volume = np.asarray(volume)
    if volume.ndim != 1 or not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError('Expected a finite nonnegative one-dimensional volume array')
    zero = volume == 0
    edges = np.diff(np.r_[False, zero, False].astype(np.int8))
    result = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        length = int(b - a)
        result.append({'onset': int(start + a), 'last_zero_day': int(start + b - 1),
                       'duration': length, 'confirmation': int(start + a + threshold - 1)
                       if length >= threshold else None,
                       'recovery_day': int(start + b) if b < len(volume) else None,
                       'right_censored': bool(b == len(volume))})
    return result


class InactivityObserver:
    """Bounded numeric daily observations, including actual branch reasons."""

    def __init__(self, capacity, start=0, streak=0):
        self.start, self.streak, self.count = start, streak, 0
        self.values = {key: np.zeros(capacity) for key in SCALAR_FIELDS}
        self.values['day'] = np.arange(start + 1, start + capacity + 1, dtype=np.int32)
        self.values.update({key: np.zeros((capacity, 3)) for key in GROUP_FIELDS})
        self.values['decision_counts'] = np.zeros((capacity, 3, len(REASONS)), dtype=np.uint16)
        self.values['submissions'] = np.zeros((capacity, 3, 4))  # buys, sells, buy qty, sell qty
        self.values['match_counts'] = np.zeros((capacity, len(MATCH_COUNTS)), dtype=np.int32)

    def begin(self, market, day):
        self.index = day - self.start
        a = market.asset
        self.values['decision_sma'][self.index] = np.mean(a.price_history[-20:])
        self.values['decision_previous_price'][self.index] = a.price_history[-2]

    def decision(self, trader, trace, order, accepted):
        group = TRADER_TYPES.index(trader.type)
        self.values['decision_counts'][self.index, group, REASONS.index(trace['reason'])] += 1
        if accepted:
            side = 0 if order[0] == 'buy' else 1
            self.values['submissions'][self.index, group, side] += 1
            self.values['submissions'][self.index, group, side + 2] += order[2]

    def match(self, record):
        self.values['match_stop'][self.index] = MATCH_REASONS.index(record['stop_reason'])
        self.values['match_counts'][self.index] = [record[k] for k in MATCH_COUNTS]

    def __call__(self, market, day, previous_price, trades):
        i, a, book = day - self.start, market.asset, market.book
        volume = sum(q for _, q, _, _ in trades)
        same = sum(q for _, q, b, s in trades if b is s)
        self.streak = self.streak + 1 if volume == 0 else 0
        row = dict(price=a.current_price, previous_price=previous_price, volume=volume,
                   same_owner_volume=same, distinct_volume=volume - same,
                   dividend=a.dividend_history[-1], accepted_price_count=len(a.price_history),
                   no_trade_streak=self.streak, trade_count=len(trades),
                   best_bid=book.buy_orders[0][0] if book.buy_orders else 0,
                   best_ask=book.sell_orders[0][0] if book.sell_orders else 0,
                   bid_count=len(book.buy_orders), ask_count=len(book.sell_orders),
                   bid_quantity=sum(q for _, q, _ in book.buy_orders),
                   ask_quantity=sum(q for _, q, _ in book.sell_orders))
        for key, value in row.items():
            self.values[key][i] = value
        for k, kind in enumerate(TRADER_TYPES):
            traders = [t for t in market.traders if t.type == kind]
            cash, shares = fsum(t.cash for t in traders), sum(t.assets for t in traders)
            for key, value in dict(total_cash=cash, total_assets=shares,
                                   total_wealth=cash + shares * a.current_price,
                                   active_count=sum(t.active for t in traders),
                                   no_holdings_count=sum(t.active and t.assets == 0 for t in traders),
                                   can_buy_spot_count=sum(t.active and t.cash >= a.current_price for t in traders)).items():
                self.values[key][i, k] = value
        self.count = i + 1

    def arrays(self):
        return {key: value[:self.count] for key, value in self.values.items()}


@dataclass
class ObservedRun:
    arrays: dict
    checkpoint: MarketCheckpoint | None
    replay_arrays: dict | None
    replay_digest: str | None
    final_digest: str


def run_reference(scenario, *, seed, days=10240, threshold=100, continuation_days=1000):
    validate_scenario(scenario)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    for name, value in (('days', days), ('threshold', threshold), ('continuation_days', continuation_days)):
        positive_integer(value, name)
    return _run(scenario, seed, days, threshold, continuation_days)


def run_continuation(checkpoint, *, days=1000):
    positive_integer(days, 'days')
    return _run(None, None, days, 0, 0, checkpoint)


def _run(scenario, seed, days, threshold, continuation_days, checkpoint=None):
    caller, log = np.random.get_state(), RunLog(LogOptions())
    try:
        if checkpoint is None:
            np.random.seed(seed)
            market = _initialize_market(scenario, False, log)
            start, streak = 0, 0
        else:
            data = checkpoint.data()
            market = restore_market(checkpoint, log)
            np.random.set_state(decode_rng(data['rng_state']))
            start, streak = data['completed_days'], data['no_trade_streak']
        observer = InactivityObserver(days + continuation_days, start, streak)
        captured, replay_end, replay_digest = None, None, None
        end = start + days
        day = start
        while day < end:
            _advance_market(market, day, end, False, log, observer)
            day += 1
            if threshold and captured is None and day <= days and observer.streak >= threshold:
                captured = capture_market(market, day, observer.streak)
                replay_end = day + continuation_days
                end = max(end, replay_end)
            if day == replay_end:
                replay_digest = dynamics_digest(capture_market(market, day, observer.streak))
        arrays = observer.arrays()
        replay = ({k: v[captured.completed_days:replay_end].copy() for k, v in arrays.items()}
                  if captured else None)
        return ObservedRun({k: v[:days] for k, v in arrays.items()}, captured, replay, replay_digest,
                           dynamics_digest(capture_market(market, day, observer.streak)))
    finally:
        log.cleanup()
        np.random.set_state(caller)


def with_future(checkpoint, seed):
    """Replace only the RNG; matching arms start alike, later draws can diverge."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Future seed must be uint32')
    data = checkpoint.data()
    data['rng_state'] = encode_rng(np.random.RandomState(seed).get_state())
    return MarketCheckpoint.from_data(data)


def intervene(checkpoint, arm):
    """Bits: clear queues, flatten shared accepted-price memory, rebalance wealth.

    Rebalance active portfolios at the current price. Integer apportionment uses
    largest remainders, stable trader IDs and cash-affordability caps. Inactive
    portfolios and all historical wealth records remain untouched.
    """
    if arm not in ARMS:
        raise ValueError('Unknown intervention arm')
    data = checkpoint.data()
    before = [(t['cash'], t['assets']) for t in data['traders']]
    price = data['asset']['current_price']
    receipt = {'arm': arm, 'before_sha256': checkpoint.sha256, 'transfers': [],
               'cleared_bids': 0, 'cleared_asks': 0, 'memory_observations': 0,
               'cash_residual': 0., 'share_residual': 0, 'max_wealth_residual': 0.,
               'tolerance': '1e-8 absolute + 1e-12 relative'}
    if arm[0] == '1':
        receipt.update(cleared_bids=len(data['bids']), cleared_asks=len(data['asks']))
        data['bids'], data['asks'] = [], []
    if arm[1] == '1':
        history = data['asset']['price_history']
        count = min(20, len(history))
        history[-count:] = [price] * count
        receipt['memory_observations'] = count
    if arm[2] == '1':
        indices = [i for i, t in enumerate(data['traders']) if t['active']]
        wealth = [before[i][0] + price * before[i][1] for i in indices]
        total = fsum(wealth)
        shares = sum(int(before[i][1]) for i in indices)
        if total > 0 and indices:
            targets = [shares * (w / total) for w in wealth]
            # Cash-based caps avoid floor((cash + P*S)/P) rounding below S.
            caps = [int(before[i][1]) + int(before[i][0] / price) for i in indices]
            for j, i in enumerate(indices):
                while before[i][0] - price * (caps[j] - int(before[i][1])) < 0:
                    caps[j] -= 1
            assigned = [min(int(np.floor(t)), cap) for t, cap in zip(targets, caps)]
            remaining = shares - sum(assigned)
            ranking = sorted(range(len(indices)), key=lambda j: (-(targets[j] - np.floor(targets[j])), indices[j]))
            while remaining:
                eligible = [j for j in ranking if assigned[j] < caps[j]]
                if not eligible or remaining < 0:
                    raise ArithmeticError('Infeasible integer redistribution')
                for j in eligible[:remaining]:
                    assigned[j] += 1
                    remaining -= 1
            receipt['target_asset_fraction'] = float(price * shares / total)
            for j, i in enumerate(indices):
                trader = data['traders'][i]
                delta = assigned[j] - int(before[i][1])
                cash = before[i][0] - price * delta
                if cash < 0:
                    raise ArithmeticError('Redistribution created negative cash')
                if delta:
                    trader['cash'], trader['assets'] = cash, assigned[j]
                receipt['transfers'].append({'trader_id': i, 'share_delta': delta,
                    'cash_delta': float(cash - before[i][0]), 'target_shares': float(targets[j]),
                    'rounding_shares': float(assigned[j] - targets[j])})
    after = [(t['cash'], t['assets']) for t in data['traders']]
    receipt['cash_residual'] = fsum(c for c, _ in after) - fsum(c for c, _ in before)
    receipt['share_residual'] = int(sum(s for _, s in after) - sum(s for _, s in before))
    residuals = [float((c + price*s) - (a + price*b)) for (c, s), (a, b) in zip(after, before)]
    receipt['max_wealth_residual'] = max(map(abs, residuals), default=0.)
    if (receipt['share_residual'] or abs(receipt['cash_residual']) > 1e-8 + 1e-12 * fsum(c for c, _ in before)
            or any(abs(r) > 1e-8 + 1e-12 * abs(c + price*s) for r, (c, s) in zip(residuals, before))):
        raise ArithmeticError('Intervention violates resource/wealth conservation')
    result = MarketCheckpoint.from_data(data)
    receipt.update(after_sha256=result.sha256, changed=result.sha256 != checkpoint.sha256)
    return result, receipt
