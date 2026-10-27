"""Deterministic, pooled-portfolio hard-rule companion to the stochastic ABM.

The quote measure is discretized, not the matched ABM transition expectation.
Within-strategy portfolio dispersion is discarded. Integer decision rules are
evaluated on per-agent means before weighting; executed flows can be fractional.
See the reduction documentation in README for the scientific limitations.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import fsum, isfinite

import numpy as np

from src.scenarios import (NOISE_CASH, NOISE_ASSETS, CHARTIST_CASH, CHARTIST_ASSETS,
                           FUNDAMENTALIST_CASH, FUNDAMENTALIST_ASSETS, RISK_FREE_RATE)

STRATEGIES = ('noise', 'chartist', 'fundamentalist')
DIVIDEND = 0.016


@dataclass(frozen=True)
class ReductionConfig:
    counts: tuple[int, int, int] = (50, 20, 30)
    alpha: float = 0.1
    gamma: float = 2.0
    window: int = 20
    quadrature_points: int = 16
    order_lifetime: str = 'day'

    def __post_init__(self):
        object.__setattr__(self, 'counts', tuple(self.counts))
        if (len(self.counts) != 3 or any(type(n) is not int or n < 0 for n in self.counts)
                or sum(self.counts) == 0):
            raise ValueError('counts must contain three nonnegative integers with positive total')
        for name, valid in (('alpha', lambda x: 0 < x <= 1), ('gamma', lambda x: x > 0)):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) or not valid(x):
                raise ValueError(f'Invalid {name}')
        if type(self.window) is not int or self.window < 1:
            raise ValueError('window must be a positive integer')
        if type(self.quadrature_points) is not int or not 1 <= self.quadrature_points <= 64:
            raise ValueError('quadrature_points must be an integer in [1, 64]')
        if self.order_lifetime not in ('day', 'persistent'):
            raise ValueError('order_lifetime must be day or persistent')


@dataclass(frozen=True)
class Order:
    price: float
    quantity: float
    owner: int


@dataclass(frozen=True)
class ReductionState:
    cash: tuple[float, float, float]
    shares: tuple[float, float, float]
    memory: tuple[float, ...]
    bids: tuple[Order, ...] = ()
    asks: tuple[Order, ...] = ()

    def __post_init__(self):
        for name in ('cash', 'shares', 'memory', 'bids', 'asks'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def price(self):
        return self.memory[-1]


@dataclass
class ReductionResult:
    config: ReductionConfig
    initial_state: ReductionState
    final_state: ReductionState
    arrays: dict[str, np.ndarray]


def initial_state(memory, config=None):
    config = config or ReductionConfig()
    size = max(2, config.window)
    history = tuple(float(x) for x in memory)
    if len(history) < size:
        raise ValueError('Provide a complete accepted-price memory')
    state = ReductionState(
        tuple(float(n * c) for n, c in zip(config.counts, (NOISE_CASH, CHARTIST_CASH, FUNDAMENTALIST_CASH))),
        tuple(float(n * s) for n, s in zip(config.counts, (NOISE_ASSETS, CHARTIST_ASSETS, FUNDAMENTALIST_ASSETS))),
        history[-size:])
    validate_state(state, config)
    return state


def validate_state(state, config, *, queues=True):
    if len(state.cash) != 3 or len(state.shares) != 3 or len(state.memory) != max(2, config.window):
        raise ValueError('Invalid state dimensions')
    if any(not isfinite(x) or x < 0 for x in (*state.cash, *state.shares)):
        raise ValueError('Resources must be finite and nonnegative')
    if any(not isfinite(p) or p <= 0 for p in state.memory):
        raise ValueError('Prices must be finite and positive')
    if any(n == 0 and (c != 0 or s != 0) for n, c, s in zip(config.counts, state.cash, state.shares)):
        raise ValueError('An absent strategy cannot own resources')
    if config.order_lifetime == 'day' and (state.bids or state.asks):
        raise ValueError('A day-order boundary state must have an empty book')
    if queues:
        for orders, reverse in ((state.bids, True), (state.asks, False)):
            for order in orders:
                if (not isinstance(order, Order) or type(order.owner) is not int or not 0 <= order.owner < 3
                        or not config.counts[order.owner] or not isfinite(order.price) or order.price <= 0
                        or not isfinite(order.quantity) or order.quantity <= 0):
                    raise ValueError('Invalid queued order')
            if any((a.price < b.price if reverse else a.price > b.price) for a, b in zip(orders, orders[1:])):
                raise ValueError('Queue is not in price priority order')


@lru_cache(maxsize=64)
def quote_grid(points):
    """Immutable nodes/weights for Uniform(0, 1), with no RNG consumption."""
    x, w = np.polynomial.legendre.leggauss(points)
    return tuple(zip(((x + 1) / 2).tolist(), (w / 2).tolist()))


def conditional_order(kind, cash, shares, memory, config, u=0.5, branch='buy'):
    """Original per-agent rule at a prescribed conditional quote draw.

    Noise's categorical draw is integrated separately. This helper deliberately
    mirrors the code, including minimum-one buys with no holdings, the chartist
    sell fallback, and fundamentalist's asymmetric clipped allocation.
    """
    p = memory[-1]
    if cash <= 0 and shares <= 0:
        return None
    if kind == 0:
        if branch == 'buy' and cash > 0:
            price = p * (1 + 0.05 * u)
            maximum = int(cash / price)
            if maximum <= 0:
                return None
            qty = max(1, int(min(maximum, shares) * 0.1))
            return ('buy', price, qty) if cash >= price * qty else None
        if branch == 'sell' and shares > 0:
            return 'sell', p * (1 - 0.05 * u), min(shares, max(1, int(shares * 0.1)))
    elif kind == 1:
        prices = memory[-config.window:]
        sma = np.mean(prices)
        if p > sma and cash > 0:
            price = p * (1 + 0.02 * (1 + np.std(prices) / sma * 10) * u)
            maximum = int(cash * config.alpha / price)
            if maximum <= 0:
                return None
            qty = max(1, min(maximum, shares))
            return ('buy', price, qty) if cash >= price * qty else None
        if shares > 0:
            return 'sell', p * (1 - 0.02 * u), min(shares, max(1, int(shares * config.alpha)))
    elif kind == 2:
        excess = p / memory[-2] - 1 + DIVIDEND / memory[-2] - RISK_FREE_RATE
        if abs(excess) > 0.05:
            allocation = max(0.1, min(0.9, excess / (config.gamma * 0.1)))
            desired = int(allocation * (cash + shares * p) / p)
            if desired > shares and cash > 0:
                qty = max(1, min(desired - shares, int(cash / (p * 1.01))))
                return ('buy', p * 1.01, qty) if cash >= p * 1.01 * qty else None
            if desired < shares and shares > 0:
                qty = max(1, min(shares - desired, shares))
                return ('sell', p * 0.99, qty) if shares >= qty else None
    else:
        raise ValueError('Unknown strategy')
    return None


def _submit(state, config):
    bids, asks = [], []
    submitted = np.zeros((3, 2), dtype=float)
    for kind, count in enumerate(config.counts):
        if not count:
            continue
        cash, shares = state.cash[kind] / count, state.shares[kind] / count
        branches = (('buy', 0.4), ('sell', 0.4)) if kind == 0 else (('signal', 1.0),)
        grid = ((0.5, 1.0),) if kind == 2 else quote_grid(config.quadrature_points)
        for branch, probability in branches:
            for u, weight in grid:
                decision = conditional_order(kind, cash, shares, state.memory, config, u, branch)
                if decision is not None:
                    side, price, quantity = decision
                    mass = count * probability * weight
                    quantity = float(quantity * mass)
                    if quantity > 0:
                        (bids if side == 'buy' else asks).append(Order(float(price), quantity, kind))
                        submitted[kind, int(side == 'sell')] += mass
    return bids, asks, submitted


def _match(cash, shares, bids, asks):
    """Legacy clipping and midpoint clearing on pooled, weighted parcels."""
    bids = sorted(bids, key=lambda o: -o.price)
    asks = sorted(asks, key=lambda o: o.price)
    i = j = 0
    quantities, notionals = [], []
    bought, sold = np.zeros(3), np.zeros(3)
    within = 0.0
    while i < len(bids) and j < len(asks) and bids[i].price >= asks[j].price:
        bid, ask = bids[i], asks[j]
        b, s = bid.owner, ask.owner
        if cash[b] < bid.price * min(bid.quantity, ask.quantity):
            quantity = int(cash[b] / bid.price)
            if quantity <= 0:
                i += 1
                continue
            bid = bids[i] = Order(bid.price, float(quantity), b)
        if shares[s] < min(bid.quantity, ask.quantity):
            if shares[s] <= 0:
                j += 1
                continue
            ask = asks[j] = Order(ask.price, shares[s], s)
        price, quantity = (bid.price + ask.price) / 2, min(bid.quantity, ask.quantity)
        if quantity <= 0 or cash[b] < price * quantity or shares[s] < quantity:
            break
        if b != s:
            cash[b] -= price * quantity
            cash[s] += price * quantity
            shares[b] += quantity
            shares[s] -= quantity
        else:
            # Algebraically net-zero; avoid cancellation drift in one pooled owner.
            within += quantity
        quantities.append(quantity)
        notionals.append(price * quantity)
        bought[b] += quantity
        sold[s] += quantity
        if bid.quantity == quantity:
            i += 1
        else:
            bids[i] = Order(bid.price, bid.quantity - quantity, b)
        if ask.quantity == quantity:
            j += 1
        else:
            asks[j] = Order(ask.price, ask.quantity - quantity, s)
    return (tuple(bids[i:]), tuple(asks[j:]), fsum(quantities), fsum(notionals),
            len(quantities), within, bought, sold)


def _step(state, config):
    new_bids, new_asks, submitted = _submit(state, config)
    cash, shares = list(state.cash), list(state.shares)
    bids, asks, volume, notional, fills, within, bought, sold = _match(
        cash, shares, (*state.bids, *new_bids), (*state.asks, *new_asks))
    price = max(notional / volume, 0.01) if volume > 0 else state.price
    memory = (*state.memory[1:], price) if volume > 0 else state.memory
    diagnostics = {
        'price': price, 'previous_price': state.price, 'volume': volume,
        'notional': notional, 'trade_count': fills, 'within_strategy_volume': within,
        'accepted_price_update': volume > 0, 'outstanding_buy_orders': len(bids),
        'outstanding_sell_orders': len(asks),
        'outstanding_buy_quantity': fsum(o.quantity for o in bids),
        'outstanding_sell_quantity': fsum(o.quantity for o in asks),
        'total_cash': cash, 'total_assets': shares, 'buy_volume': bought, 'sell_volume': sold,
        'submitted_buy_mass': submitted[:, 0], 'submitted_sell_mass': submitted[:, 1],
    }
    if config.order_lifetime == 'day':
        bids = asks = ()
    next_state = ReductionState(cash, shares, memory, bids, asks)
    validate_state(next_state, config, queues=False)
    return next_state, diagnostics


def step_hard(state, config=None):
    """Return a new state and pre-expiry daily diagnostics, with no RNG calls."""
    config = config or ReductionConfig()
    validate_state(state, config)
    return _step(state, config)


def simulate_reduction(state, config=None, *, days=2000):
    config = config or ReductionConfig()
    if type(days) is not int or days < 1:
        raise ValueError('days must be a positive integer')
    validate_state(state, config)
    first, arrays = state, {}
    totals = np.array([fsum(first.cash), fsum(first.shares)])
    for day in range(days):
        state, diagnostic = _step(state, config)
        for name, value in diagnostic.items():
            value = np.asarray(value)
            if day == 0:
                arrays[name] = np.empty((days, *value.shape), dtype=value.dtype)
            arrays[name][day] = value
        if not np.allclose([fsum(state.cash), fsum(state.shares)], totals, rtol=1e-11, atol=1e-8):
            raise ArithmeticError('Reduction resource conservation failed')
    arrays['day'] = np.arange(days, dtype=np.int64)
    arrays['log_return'] = np.log(arrays['price']) - np.log(arrays['previous_price'])
    arrays['total_wealth'] = arrays['total_cash'] + arrays['total_assets'] * arrays['price'][:, None]
    return ReductionResult(config, first, state, arrays)
