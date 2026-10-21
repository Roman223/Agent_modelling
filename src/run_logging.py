"""Structured run logs, streamed to temporary compressed files without RNG use."""

from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
import shutil
import tempfile
import weakref

import numpy as np


def json_safe(value):
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class LogOptions:
    """Detail filters use zero-based trading days and run-local trader indices."""

    level: str = 'summary'
    trader_ids: tuple[int, ...] = ()
    start_day: int = 0
    end_day: int | None = None
    max_detail_records: int = 100_000
    max_detail_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if self.level not in ('off', 'summary', 'trades', 'decisions'):
            raise ValueError('log level must be off, summary, trades, or decisions')
        values = (self.start_day, self.max_detail_records, self.max_detail_bytes, *self.trader_ids)
        if self.end_day is not None:
            values += (self.end_day,)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0
               for v in values):
            raise ValueError('log days, trader IDs, and limits must be nonnegative integers')
        if self.end_day is not None and self.end_day < self.start_day:
            raise ValueError('log end day must be at least the start day')
        object.__setattr__(self, 'trader_ids', tuple(self.trader_ids))


class RunLog:
    """Own compressed log files until exported or explicitly cleaned up.

    Summary/event records are complete. Detail caps apply across trades and
    decisions combined; omissions are counted, never silently called complete.
    """

    def __init__(self, options):
        self.options = options
        self.phase, self.day = 'warmup', 0
        self._directory = tempfile.TemporaryDirectory(prefix='abm-logs-') if self.enabled else None
        self._finalizer = weakref.finalize(self, self._directory.cleanup) if self.enabled else None
        self._streams = {}
        self.counts = {}
        self.omitted = {'trades': 0, 'decisions': 0}
        self._detail_records = self._detail_bytes = 0
        self._capped = False
        self._closed = False
        self._ids = {}

    @property
    def enabled(self):
        return self.options.level != 'off'

    def register(self, traders):
        self._ids.update({id(t): i for i, t in enumerate(traders)})

    def trader_id(self, trader):
        return self._ids[id(trader)]

    def wants_detail(self, kind, *ids):
        if self.options.level not in ('trades', 'decisions'):
            return False
        if kind == 'decisions' and self.options.level != 'decisions':
            return False
        return (self.phase == 'trading' and self.day >= self.options.start_day
                and (self.options.end_day is None or self.day <= self.options.end_day)
                and (not self.options.trader_ids or any(i in self.options.trader_ids for i in ids)))

    def write(self, kind, record, *, detail=False):
        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError('Run log is closed')
        if detail and self._capped:
            self.omitted[kind] += 1
            return
        line = json.dumps(json_safe(record), allow_nan=False, separators=(',', ':')) + '\n'
        if detail:
            size = len(line.encode('utf-8'))
            if (self._detail_records >= self.options.max_detail_records
                    or self._detail_bytes + size > self.options.max_detail_bytes):
                self._capped = True
                self.omitted[kind] += 1
                return
            self._detail_records += 1
            self._detail_bytes += size
        if kind not in self._streams:
            self._streams[kind] = gzip.open(Path(self._directory.name) / f'{kind}.jsonl.gz',
                                            'wt', encoding='utf-8')
        self._streams[kind].write(line)
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def event(self, event, **fields):
        self.write('events', {'phase': self.phase, 'day': self.day, 'event': event, **fields})

    def trade(self, price, quantity, buyer, seller, before):
        buyer_id, seller_id = self.trader_id(buyer), self.trader_id(seller)
        if self.wants_detail('trades', buyer_id, seller_id):
            self.write('trades', {
                'day': self.day, 'buyer_id': buyer_id, 'seller_id': seller_id,
                'buyer_type': buyer.type, 'seller_type': seller.type,
                'price': price, 'quantity': quantity, 'before': before,
                'after': {'buyer_cash': buyer.cash, 'buyer_assets': buyer.assets,
                          'seller_cash': seller.cash, 'seller_assets': seller.assets},
            }, detail=True)

    def decision(self, trader, trace, order, submitted):
        self.write('decisions', {
            'day': self.day, 'trader_id': self.trader_id(trader), 'type': trader.type,
            'cash': trader.cash, 'assets': trader.assets, 'active': trader.active,
            'order_type': order[0], 'limit_price': order[1], 'quantity': order[2],
            'submitted': submitted, **trace,
        }, detail=True)

    def close(self):
        for stream in self._streams.values():
            stream.close()
        self._closed = True

    def cleanup(self):
        """Release temporary files after saving/reading the logs."""
        self.close()
        if self._directory is not None:
            self._directory.cleanup()

    def iter_records(self, kind):
        """Read a completed log incrementally; an absent category yields no rows."""
        if not self._closed:
            raise RuntimeError('Read logs after the simulation completes')
        if kind in self.counts:
            with gzip.open(Path(self._directory.name) / f'{kind}.jsonl.gz', 'rt', encoding='utf-8') as stream:
                for line in stream:
                    yield json.loads(line)

    def metadata(self):
        return {'schema_version': 1, 'options': asdict(self.options),
                'records': dict(self.counts), 'omitted_by_cap': dict(self.omitted),
                'detail_truncated': self._capped, 'detail_bytes_uncompressed': self._detail_bytes,
                'files': {kind: f'{kind}.jsonl.gz' for kind in self.counts},
                'day_indexing': 'zero-based; phase distinguishes warmup from trading',
                'trader_ids': 'indices in traders.json; evolution appends new indices',
                'detail_scope': 'only matching days/traders; caps shared across detail categories'}

    def export(self, destination):
        self.close()
        destination = Path(destination)
        destination.mkdir()
        for kind in self.counts:
            shutil.copyfile(Path(self._directory.name) / f'{kind}.jsonl.gz',
                            destination / f'{kind}.jsonl.gz')
        (destination / 'metadata.json').write_text(
            json.dumps(json_safe(self.metadata()), indent=2, allow_nan=False) + '\n', encoding='utf-8')
