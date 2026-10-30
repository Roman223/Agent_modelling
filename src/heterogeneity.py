"""Portfolio compression, retained-detail measures, and bounded observations."""

from collections import deque
from dataclasses import dataclass
from math import fsum

import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.experiment_artifacts import canonical
from src.market_checkpoints import MarketCheckpoint
from src.scenarios import (NOISE_CASH, NOISE_ASSETS, CHARTIST_CASH, CHARTIST_ASSETS,
                           FUNDAMENTALIST_CASH, FUNDAMENTALIST_ASSETS)

STRATEGIES = ('noise', 'chartist', 'fundamentalist')
SCALES = np.array([[NOISE_CASH, NOISE_ASSETS], [CHARTIST_CASH, CHARTIST_ASSETS],
                   [FUNDAMENTALIST_CASH, FUNDAMENTALIST_ASSETS]], dtype=float)
LAGS = (0, 1, 5, 20)
MARKET_FEATURES = ['log_price_relative_100', 'log_return', 'accepted_trend', 'accepted_volatility',
                   'log_volume', 'log_bid_count', 'log_ask_count', 'log_bid_quantity',
                   'log_ask_quantity', 'book_imbalance', 'relative_spread', 'empty_bid',
                   'empty_ask', 'accepted_update', 'log_no_trade_streak']
GROUP_FEATURES = ['cash_fraction', 'share_fraction', 'active_fraction', 'zero_share_fraction',
                  'one_share_cash_fraction', 'cash_share_correlation', 'correlation_undefined']
GROUP_FEATURES += [f'log_{resource}_q{q}' for resource in ('cash', 'shares') for q in (0, 25, 50, 75, 100)]
FEATURE_NAMES = MARKET_FEATURES + [f'{s}_{name}' for s in STRATEGIES for name in GROUP_FEATURES]


@dataclass(frozen=True)
class CompressionConfig:
    level: int | str = 'full'

    def __post_init__(self):
        if self.level != 'full' and (type(self.level) is not int or self.level < 1):
            raise ValueError('level must be full or a positive integer')


def _detail(before, after, eligible, requested, labels, price, scale):
    x, y = before / scale, after / scale
    variance = float(np.sum((x - x.mean(axis=0)) ** 2)) if len(x) else 0.0
    after_variance = float(np.sum((y - y.mean(axis=0)) ** 2)) if len(y) else 0.0
    if len(x) and not np.array_equal(x, y):
        # Equal-sized empirical measures with uniform individual weights: exact
        # W1 is the mean minimum-cost assignment. No LP solver/thread pool needed.
        costs = cdist(x, y)
        rows, columns = linear_sum_assignment(costs)
        error = float(np.mean(costs[rows, columns]))
    else:
        error = 0.0
    return {'requested': requested, 'eligible_count': len(eligible),
            'cluster_count': len(set(labels.tolist())) if len(labels) else 0,
            'realized_profiles': len(np.unique(after, axis=0)) if len(after) else 0,
            'retained_dispersion': after_variance / variance if variance > 0 else None,
            'dispersion_before': variance, 'dispersion_after': after_variance,
            'joint_wasserstein': error,
            'mean_squared_assignment_error': float(np.mean(np.sum((x - y)**2, axis=1))) if len(x) else 0,
            'zero_share_fraction_before': float(np.mean(before[:, 1] == 0)) if len(x) else None,
            'zero_share_fraction_after': float(np.mean(after[:, 1] == 0)) if len(x) else None,
            'unaffordable_fraction_before': float(np.mean(before[:, 0] < price)) if len(x) else None,
            'unaffordable_fraction_after': float(np.mean(after[:, 0] < price)) if len(x) else None}


def compress_checkpoint(checkpoint, config=None):
    """One conservative portfolio intervention; never changes queues or RNG.

    Ward partitions are nested for each state. Integer balancing may leave two
    share profiles per cluster. Assignment and cash residuals are recorded.
    """
    config = config or CompressionConfig()
    data = checkpoint.data()
    traders = data['traders']
    records = {}
    for kind, name in enumerate(STRATEGIES):
        indices = [i for i, t in enumerate(traders) if t['type'] == name]
        before = np.array([[traders[i]['cash'], traders[i]['assets']] for i in indices], dtype=float).reshape(-1, 2)
        eligible = [i for i in indices if traders[i]['active'] and (traders[i]['cash'] > 0 or traders[i]['assets'] > 0)]
        k = len(eligible) if config.level == 'full' else min(config.level, len(eligible))
        if k == len(eligible):
            labels = np.arange(len(eligible))
        elif k == 1:
            labels = np.zeros(len(eligible), dtype=int)
        else:
            x = np.array([[traders[i]['cash'], traders[i]['assets']] for i in eligible]) / SCALES[kind]
            labels = cut_tree(linkage(x, method='ward', optimal_ordering=False), n_clusters=[k])[:, 0]
        assignments = []
        for group in sorted(set(labels.tolist())):
            members = [i for i, label in zip(eligible, labels) if label == group]
            if len(members) > 1 and len({(traders[i]['cash'], traders[i]['assets']) for i in members}) > 1:
                total_cash = fsum(traders[i]['cash'] for i in members)
                total_shares = sum(int(traders[i]['assets']) for i in members)
                floor, remainder = divmod(total_shares, len(members))
                ranked = sorted(members, key=lambda i: (-traders[i]['assets'], i))
                extra = set(ranked[:remainder])
                for i in members:
                    scalar = np.float64 if isinstance(traders[i]['cash'], np.floating) else float
                    traders[i]['cash'] = scalar(total_cash / len(members))
                    traders[i]['assets'] = floor + int(i in extra)
                traders[members[-1]]['cash'] += total_cash - fsum(traders[i]['cash'] for i in members)
            assignments.append(members)
        after = np.array([[traders[i]['cash'], traders[i]['assets']] for i in indices], dtype=float).reshape(-1, 2)
        if len(before):
            if (sum(int(x) for x in before[:, 1]) != sum(int(x) for x in after[:, 1])
                    or not np.isclose(fsum(before[:, 0]), fsum(after[:, 0]), rtol=1e-13, atol=1e-8)
                    or np.any(after < 0)):
                raise ArithmeticError('Portfolio compression violated conservation')
        records[name] = _detail(before, after, eligible, config.level, labels,
                                data['asset']['current_price'], SCALES[kind])
        records[name]['members'] = assignments
    result = checkpoint if config.level == 'full' else MarketCheckpoint.from_data(data)
    return result, {'level': config.level, 'groups': records, 'source_sha256': checkpoint.sha256,
                    'compressed_sha256': result.sha256, 'changed': result != checkpoint}


class ObservationCollector:
    """Daily outcomes plus thinned current/lagged features; no random draws."""

    def __init__(self, days, start, streak=0):
        self.start, self.streak = start, streak
        self.values = {name: np.empty(days, dtype=float) for name in
                       ('price', 'previous_price', 'volume', 'within_strategy_volume', 'same_owner_volume', 'log_return')}
        self.values.update({name: np.empty((days, 3), dtype=float) for name in ('total_cash', 'total_assets')})
        self.values['day'] = np.arange(start, start + days, dtype=np.int64)
        self.values['no_trade_streak'] = np.empty(days, dtype=np.int64)
        self.buffer = deque(maxlen=21)
        self.feature_rows, self.feature_days = [], []

    def __call__(self, market, day, previous_price, trades):
        a, book, traders = market.asset, market.book, market.traders
        i, price = day - self.start, a.current_price
        volume = sum(q for _, q, _, _ in trades)
        self.streak = self.streak + 1 if volume == 0 else 0
        for name, value in (
            ('price', price), ('previous_price', previous_price), ('volume', volume),
            ('within_strategy_volume', sum(q for _, q, b, s in trades if b.type == s.type)),
            ('same_owner_volume', sum(q for _, q, b, s in trades if b is s)),
            ('log_return', np.log(price) - np.log(previous_price)), ('no_trade_streak', self.streak)):
            self.values[name][i] = value
        groups = [np.array([[t.cash, t.assets, t.active] for t in traders if t.type == name], dtype=float).reshape(-1, 3)
                  for name in STRATEGIES]
        cash = [fsum(g[:, 0]) for g in groups]
        shares = [fsum(g[:, 1]) for g in groups]
        self.values['total_cash'][i], self.values['total_assets'][i] = cash, shares
        # Only days that contribute to the fixed ten-day sample or its lags need features.
        if i % 10 not in (0, 5, 9):
            self.buffer.append(None)
            return
        history = a.price_history[-market.scenario.chartist_window:]
        mean = float(np.mean(history))
        bq = sum(o[1] for o in book.buy_orders)
        sq = sum(o[1] for o in book.sell_orders)
        spread = ((book.sell_orders[0][0] - book.buy_orders[0][0]) / price
                  if book.buy_orders and book.sell_orders else 0)
        feature = [np.log(price / 100), self.values['log_return'][i], price / mean - 1,
                   np.std(history) / mean, np.log1p(volume), np.log1p(len(book.buy_orders)),
                   np.log1p(len(book.sell_orders)), np.log1p(bq), np.log1p(sq),
                   (bq - sq) / (bq + sq) if bq + sq else 0, spread,
                   not bool(book.buy_orders), not bool(book.sell_orders), volume > 0, np.log1p(self.streak)]
        for k, g in enumerate(groups):
            if not len(g):
                feature.extend([0.] * len(GROUP_FEATURES))
                continue
            c, s = g[:, 0], g[:, 1]
            defined = np.std(c) > 0 and np.std(s) > 0
            feature.extend([cash[k] / fsum(cash) if fsum(cash) else 0,
                            shares[k] / fsum(shares) if fsum(shares) else 0,
                            np.mean(g[:, 2]), np.mean(s == 0), np.mean(c >= price),
                            np.corrcoef(c, s)[0, 1] if defined else 0, not defined])
            quantiles = np.quantile(g[:, :2] / SCALES[k], [0, .25, .5, .75, 1], axis=0)
            feature.extend(np.log1p(quantiles.T).ravel().tolist())
        row = np.asarray(feature, dtype=float)
        if not np.isfinite(row).all() or len(row) != len(FEATURE_NAMES):
            raise ArithmeticError('Invalid research features')
        self.buffer.append(row)
        if i >= 20 and i % 10 == 0:
            self.feature_rows.append(np.concatenate([self.buffer[-lag-1] for lag in LAGS]))
            self.feature_days.append(day)

    def arrays(self):
        self.values['feature_days'] = np.array(self.feature_days, dtype=np.int64)
        self.values['features'] = np.array(self.feature_rows, dtype=float).reshape(-1, len(FEATURE_NAMES) * len(LAGS))
        return self.values
