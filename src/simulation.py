"""Shared scenario loop and legacy-compatible simulation reports."""

from dataclasses import dataclass, field

import numpy as np
from .asset import Asset
from .order_book import OrderBook
from .traders import NoiseTrader, ChartistTrader, FundamentalistTrader
from .metrics import metric_result
from .run_logging import LogOptions, RunLog
from .scenarios import (Scenario, SCENARIOS, NOISE_CASH, NOISE_ASSETS, CHARTIST_CASH, CHARTIST_ASSETS, CHARTIST_ALPHA,
                        FUNDAMENTALIST_CASH, FUNDAMENTALIST_ASSETS, FUNDAMENTALIST_GAMMA, RISK_FREE_RATE)

TRADER_TYPES = ('noise', 'chartist', 'fundamentalist')
LOG_LABELS = {'noise': 'Noise', 'chartist': 'Chartist', 'fundamentalist': 'Fundamentalist'}

# Shared for notebook compatibility; each simulation overwrites these series.
wealth_history = {
    'days': [],
    'noise_avg': [],
    'chartist_avg': [],
    'fundamentalist_avg': []
}


@dataclass
class RunAnalysis:
    """Cached statistics and report text; formatting never consumes random draws."""

    summary: dict
    metrics: dict
    report_lines: list[str]
    rng_state_before: tuple
    rng_state_after: tuple


@dataclass
class SimulationResult:
    """One completed run, independent of subsequent runs and compatibility globals.

    Treat the contained asset/traders as read-only after running analysis: analysis
    is cached and is not invalidated by manual edits to model objects.
    """

    scenario: Scenario
    days: int
    asset: Asset
    traders: list
    wealth_history: dict
    initial_wealths: dict
    rng_state_before: tuple
    rng_state_after: tuple
    analysis: RunAnalysis | None = field(default=None, repr=False)
    run_log: RunLog | None = field(default=None, repr=False)

    @property
    def halt_events(self):
        return self.asset.halt_events

    @property
    def evolution_log(self):
        return self.asset.evolution_log

    @property
    def trader_counts_history(self):
        return self.asset.trader_counts_history


def run_simulation(scenario=SCENARIOS['baseline'], days=1000, verbose=True, *, log_options=None):
    """Legacy wrapper: preserve tuple return, global history, and RNG consumption."""
    result = simulate(scenario, days, verbose=verbose, log_options=log_options)
    # Keep the dictionary identity used by `from src import wealth_history`.
    wealth_history.clear()
    wealth_history.update({key: values.copy() for key, values in result.wealth_history.items()})
    analysis = analyze_result(result)
    if verbose:
        print_analysis(analysis)
    return result.asset, result.traders


def simulate(scenario=SCENARIOS['baseline'], days=1000, *, verbose=False, log_options=None):
    """Run the market loop without final analysis or mutation of global history.

    Uses the current global NumPy RNG stream. Call analyze_result immediately to
    reproduce legacy metric values, or retain its separately recorded RNG state.
    """
    if isinstance(days, (bool, np.bool_)) or not isinstance(days, (int, np.integer)) or days <= 0:
        raise ValueError('days must be a positive integer')
    scenario.validate()
    if log_options is not None and not isinstance(log_options, LogOptions):
        raise ValueError('log_options must be a LogOptions instance')
    run_log = RunLog(log_options or LogOptions())
    try:
        result = _simulate(scenario, days, verbose, run_log)
        run_log.close()
        if verbose and run_log.metadata()['detail_truncated']:
            print(f"Detail log limit reached; omitted records: {run_log.omitted}. "
                  "Summary and event logs remain complete.")
        return result
    except BaseException:
        run_log.cleanup()
        raise


def _simulate(scenario, days, verbose, run_log):
    rng_state_before = np.random.get_state()
    history = {key: [] for key in ('days', 'noise_avg', 'chartist_avg', 'fundamentalist_avg')}
    log = print if verbose else (lambda *args, **kwargs: None)
    asset = Asset(circuit_breaker=scenario.circuit_breaker_asset, log=log,
                  event=run_log.event if run_log.enabled else None)
    asset.run_log = run_log
    order_book = OrderBook()
    traders = _create_traders(scenario)
    run_log.register(traders)

    # Twenty exogenous GBM steps before endogenous trading starts.
    for warmup_day in range(20):
        run_log.day = warmup_day
        previous_price = asset.current_price
        new_price = asset.current_price * np.exp(0.0003 - 0.02**2/2 + 0.02*np.random.randn())
        accepted = asset.update_price(new_price)
        asset.update_dividend()
        if run_log.enabled:
            run_log.write('steps', {'phase': 'warmup', 'day': warmup_day,
                                   'previous_price': previous_price, 'price': asset.current_price,
                                   'return': asset.current_price / previous_price - 1,
                                   'proposed_price': new_price, 'price_update_accepted': accepted,
                                   'dividend': asset.dividend_history[-1]})

    initial_wealths = {id(trader): trader.wealth_history[0] for trader in traders}
    asset.halt_events = []
    asset.evolution_log = []
    asset.trader_counts_history = []

    for day in range(days):
        run_log.phase, run_log.day = 'trading', day
        previous_price = asset.current_price
        submitted = {kind: {'buy_orders': 0, 'sell_orders': 0,
                            'buy_quantity': 0, 'sell_quantity': 0} for kind in TRADER_TYPES}
        executed, price_formation = [], {}
        asset.update_dividend()

        # Strategy changes every 100 days.
        if scenario.evolution and day % 100 == 0 and day > 0:
            _evolve_traders(traders, day, asset.evolution_log, log, run_log)

        trading_halted = asset.check_halt_status() if scenario.halt_check else False

        if not trading_halted:
            for trader in traders:
                trace = {} if run_log.wants_detail('decisions', run_log.trader_id(trader)) else None
                if trader.cash <= 0 and trader.assets <= 0:
                    if trader.active:
                        run_log.event('trader_deactivated', trader_id=run_log.trader_id(trader),
                                      type=trader.type, reason='no_cash_and_no_holdings')
                    trader.active = False
                    if trace is not None:
                        run_log.decision(trader, {'reason': 'no_cash_and_no_holdings'},
                                         (None, None, None), False)
                    continue
                order_type, price, qty = trader.make_decision(asset, trace=trace)
                accepted_order = bool(order_type and price and qty and qty > 0)
                if trace is not None:
                    run_log.decision(trader, trace, (order_type, price, qty), accepted_order)
                if order_type and price and qty:
                    order_book.add_order(order_type, price, qty, id(trader))
                    if run_log.enabled and accepted_order:
                        submitted[trader.type][order_type + '_orders'] += 1
                        submitted[trader.type][order_type + '_quantity'] += qty
            executed = order_book.match_orders(
                traders, asset,
                on_trade=run_log.trade if run_log.options.level in ('trades', 'decisions') else None,
                on_price=price_formation.update if run_log.enabled else None)
        else:
            asset.halt_events.append(day)
            run_log.event('trading_skipped', reason='circuit_breaker', remaining=asset.halt_remaining)

        for trader in traders:
            trader.update_wealth(asset.current_price)

        if run_log.enabled:
            _log_step(run_log, asset, traders, order_book, previous_price, submitted,
                      executed, price_formation, trading_halted)

        # Sample group statistics every 100 days and on the final day.
        if day % 100 == 0 or day == days-1:
            _record_status(day, asset, traders, scenario, trading_halted, log, history)

    return SimulationResult(scenario, days, asset, traders, history, initial_wealths,
                            rng_state_before, np.random.get_state(), run_log=run_log)


def _log_step(run_log, asset, traders, book, previous_price, submitted, executed,
              price_formation, trading_skipped):
    groups = {kind: {'active_count': 0, 'total_cash': 0, 'total_assets': 0,
                     'total_wealth': 0, 'buy_volume': 0, 'sell_volume': 0,
                     'buy_notional': 0, 'sell_notional': 0, **submitted[kind]} for kind in TRADER_TYPES}
    for trader in traders:
        if trader.active:
            group = groups[trader.type]
            group['active_count'] += 1
            group['total_cash'] += trader.cash
            group['total_assets'] += trader.assets
            group['total_wealth'] += trader.cash + trader.assets * asset.current_price
    for price, qty, buyer, seller in executed:
        groups[buyer.type]['buy_volume'] += qty
        groups[seller.type]['sell_volume'] += qty
        groups[buyer.type]['buy_notional'] += price * qty
        groups[seller.type]['sell_notional'] += price * qty
    for kind, group in groups.items():
        count = group['active_count']
        run_log.write('strategies', {'day': run_log.day, 'type': kind, **group,
                                    'average_cash': group['total_cash'] / count if count else None,
                                    'average_wealth': group['total_wealth'] / count if count else None})
    best_bid = book.buy_orders[0][0] if book.buy_orders else None
    best_ask = book.sell_orders[0][0] if book.sell_orders else None
    run_log.write('steps', {
        'phase': 'trading', 'day': run_log.day, 'previous_price': previous_price,
        'price': asset.current_price, 'return': asset.current_price / previous_price - 1,
        'dividend': asset.dividend_history[-1], 'trade_count': len(executed),
        'volume': sum(qty for _, qty, _, _ in executed),
        'notional': sum(price * qty for price, qty, _, _ in executed),
        'best_bid': best_bid, 'best_ask': best_ask,
        'spread': best_ask - best_bid if best_bid is not None and best_ask is not None else None,
        'outstanding_buy_orders': len(book.buy_orders), 'outstanding_sell_orders': len(book.sell_orders),
        'outstanding_buy_quantity': sum(order[1] for order in book.buy_orders),
        'outstanding_sell_quantity': sum(order[1] for order in book.sell_orders),
        'trading_skipped': trading_skipped, 'halt_flag': asset.trading_halted,
        'halt_remaining': asset.halt_remaining, 'price_formation': price_formation or None,
    })


def _create_traders(scenario):
    """Build the population in the original noise/chartist/fundamentalist order."""
    traders = []
    for _ in range(scenario.noise):
        traders.append(NoiseTrader(cash=NOISE_CASH, assets=NOISE_ASSETS))
    for _ in range(scenario.chartists):
        traders.append(ChartistTrader(
            cash=CHARTIST_CASH, assets=CHARTIST_ASSETS,
            window=scenario.chartist_window, alpha=CHARTIST_ALPHA,
        ))
    for _ in range(scenario.fundamentalists):
        traders.append(FundamentalistTrader(
            cash=FUNDAMENTALIST_CASH, assets=FUNDAMENTALIST_ASSETS,
            gamma=FUNDAMENTALIST_GAMMA, rf=RISK_FREE_RATE,
        ))
    return traders


def _evolve_traders(traders, day, evolution_log, log, run_log=None):
    log(f"\nTrader evolution on day {day}")
    noise_traders_list = [t for t in traders if t.type == 'noise' and t.active]
    if len(noise_traders_list) >= 2:
        traders_to_evolve = np.random.choice(noise_traders_list, 2, replace=False)
        for trader in traders_to_evolve:
            new_chartist = ChartistTrader(cash=trader.cash, assets=trader.assets, window=20, alpha=CHARTIST_ALPHA)
            new_chartist.wealth_history = trader.wealth_history.copy()
            traders.append(new_chartist)
            trader.active = False
            if run_log is not None:
                _log_evolution(run_log, traders, trader, new_chartist)
            description = f"Noise trader evolved into a chartist (assets: {trader.assets}, cash: {trader.cash:.2f})"
            evolution_log.append({'day': day, 'description': description})
            log(description)
    chartist_traders_list = [t for t in traders if t.type == 'chartist' and t.active]
    if len(chartist_traders_list) >= 1:
        trader_to_evolve = np.random.choice(chartist_traders_list, 1)[0]
        new_fundamentalist = FundamentalistTrader(cash=trader_to_evolve.cash, assets=trader_to_evolve.assets,
                                                  gamma=FUNDAMENTALIST_GAMMA, rf=RISK_FREE_RATE)
        new_fundamentalist.wealth_history = trader_to_evolve.wealth_history.copy()
        traders.append(new_fundamentalist)
        trader_to_evolve.active = False
        if run_log is not None:
            _log_evolution(run_log, traders, trader_to_evolve, new_fundamentalist)
        description = (f"Chartist evolved into a fundamentalist "
                       f"(assets: {trader_to_evolve.assets}, cash: {trader_to_evolve.cash:.2f})")
        evolution_log.append({'day': day, 'description': description})
        log(description)


def _log_evolution(run_log, traders, old, new):
    run_log.register(traders)
    run_log.event('evolution', old_trader_id=run_log.trader_id(old),
                  new_trader_id=run_log.trader_id(new), old_type=old.type, new_type=new.type,
                  selection='uniform_random_among_active_source_type', selection_score=None,
                  cash=new.cash, assets=new.assets,
                  active_counts={kind: sum(t.active and t.type == kind for t in traders)
                                 for kind in TRADER_TYPES})


def _initial_counts(scenario):
    return {'noise': scenario.noise, 'chartist': scenario.chartists, 'fundamentalist': scenario.fundamentalists}


def _record_status(day, asset, traders, scenario, trading_halted, log, history):
    stats = {t: {'cash': [], 'assets': [], 'wealth': []} for t in TRADER_TYPES}
    for trader in traders:
        if trader.active:
            s = stats[trader.type]
            s['cash'].append(trader.cash)
            s['assets'].append(trader.assets)
            s['wealth'].append(trader.cash + trader.assets * asset.current_price)

    history['days'].append(day)
    for t in TRADER_TYPES:
        history[f'{t}_avg'].append(np.mean(stats[t]['wealth']) if stats[t]['wealth'] else 0)
    asset.trader_counts_history.append({'day': day, **{t: len(stats[t]['cash']) for t in TRADER_TYPES}})

    initial_counts = _initial_counts(scenario)
    log(f"\nDay {day}: Price = {asset.current_price:.2f}")
    if trading_halted:
        log(f"Trading halted! Days remaining: {asset.halt_remaining}")
    log("Active Traders - " + ", ".join(
        f"{LOG_LABELS[t]}: {len(stats[t]['cash'])}/{initial_counts[t]}" for t in TRADER_TYPES))
    for t in TRADER_TYPES:
        if stats[t]['cash']:
            log(f"{LOG_LABELS[t]} Traders: Avg Cash = {np.mean(stats[t]['cash']):.2f}, "
                f"Avg Assets = {np.mean(stats[t]['assets']):.2f}, "
                f"Avg Wealth = {history[f'{t}_avg'][-1]:.2f}")


def analyze_result(result):
    """Compute metrics once using the current RNG stream, in legacy call order."""
    if result.analysis is None:
        lines = []
        before = np.random.get_state()
        summary, metrics = _build_analysis(result.scenario, result.asset, result.traders,
                                           result.initial_wealths, lines.append)
        result.analysis = RunAnalysis(summary, metrics, lines, before, np.random.get_state())
    return result.analysis


def print_analysis(analysis):
    """Print a cached report without recomputing statistics or metrics."""
    for line in analysis.report_lines:
        print(line)


def _build_analysis(scenario, asset, traders, initial_wealths, log):
    log("\n" + "="*50)
    log(f"SIMULATION RESULTS ({scenario.title})")
    log("="*50)

    if scenario.halt_check:
        log(f"Days without trading (circuit breaker): {len(asset.halt_events)}")
    if scenario.evolution:
        log(f"Evolution events: {len(asset.evolution_log)}")

    # Evolved traders are new objects and absent from initial_wealths, so they are skipped here.
    original = [t for t in traders if id(t) in initial_wealths]
    initial_counts = _initial_counts(scenario)

    profitable = {t: 0 for t in TRADER_TYPES}
    wealth_change = {t: [] for t in TRADER_TYPES}
    for trader in original:
        initial_wealth = initial_wealths[id(trader)]
        final_wealth = trader.wealth_history[-1]
        if final_wealth > initial_wealth:
            profitable[trader.type] += 1
        wealth_change[trader.type].append((final_wealth - initial_wealth) / initial_wealth * 100)

    for t in TRADER_TYPES:
        if initial_counts[t]:
            log(f"{LOG_LABELS[t]} traders with a profit: {profitable[t]}/{initial_counts[t]} "
                f"({profitable[t]/initial_counts[t]*100:.1f}%)")
    total_profitable = sum(profitable.values())
    log(f"Total traders with a profit: {total_profitable}/{len(original)} "
        f"({total_profitable/len(original)*100:.1f}%)")
    for t in TRADER_TYPES:
        if wealth_change[t]:
            log(f"Mean wealth change for {LOG_LABELS[t]}: {np.mean(wealth_change[t]):.2f}%")

    log('')
    price_series = np.array(asset.price_history)
    hurst = metric_result('hurst', price_series)
    hurst_val = hurst.value
    if not np.isnan(hurst_val):
        log(f"HURST EXPONENT (H) = {hurst_val:.4f}")
    else:
        log("HURST EXPONENT: unavailable")

    lyapunov = metric_result('lyapunov', price_series, emb_dim=10, lag=1)
    lyap_val = lyapunov.value
    if not np.isnan(lyap_val):
        log(f"LYAPUNOV ESTIMATE (Rosenstein) = {lyap_val:.4f}")
    else:
        log("LYAPUNOV ESTIMATE: unavailable")

    sampen = metric_result('sampen', price_series, emb_dim=3)
    sample_entropy = sampen.value
    if not np.isnan(sample_entropy):
        log(f"SAMPLE ENTROPY (nats) = {sample_entropy:.4f}")
    else:
        log("SAMPLE ENTROPY: unavailable")

    for name, diagnostic in (('Hurst', hurst), ('Lyapunov', lyapunov), ('Sample Entropy', sampen)):
        if diagnostic.status != 'ok':
            log(f"{name} diagnostic [{diagnostic.status}]: {diagnostic.reason}")

    summary = {
        'initial_counts': initial_counts,
        'profitable_counts': profitable,
        'original_trader_count': len(original),
        'mean_wealth_change_percent': {
            t: float(np.mean(values)) if values else None for t, values in wealth_change.items()
        },
        'final_price': asset.current_price,
        'halt_days': len(asset.halt_events),
        'evolution_events': len(asset.evolution_log),
    }
    return summary, {'hurst': hurst, 'lyapunov': lyapunov, 'sampen': sampen}
