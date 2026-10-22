"""Scenario definitions and shared parameters from the original notebook."""

from dataclasses import dataclass
from numbers import Integral

# Per-agent endowments and strategy parameters shared by every scenario (notebook cells 18-47).
NOISE_CASH, NOISE_ASSETS = 100000, 1000
CHARTIST_CASH, CHARTIST_ASSETS = 150000, 1000
CHARTIST_ALPHA = 0.1
FUNDAMENTALIST_CASH, FUNDAMENTALIST_ASSETS = 1000000, 500
FUNDAMENTALIST_GAMMA = 2
RISK_FREE_RATE = 0.0003

# Trader.wealth_history[0] values every asset at 100, whatever the actual price.
INITIAL_WEALTH = {
    'noise': NOISE_CASH + NOISE_ASSETS * 100,
    'chartist': CHARTIST_CASH + CHARTIST_ASSETS * 100,
    'fundamentalist': FUNDAMENTALIST_CASH + FUNDAMENTALIST_ASSETS * 100,
}


@dataclass(frozen=True)
class Scenario:
    """Initial trader counts and switches for the common simulation loop."""

    name: str
    title: str
    noise: int = 50
    chartists: int = 20
    fundamentalists: int = 30
    chartist_window: int = 20
    circuit_breaker_asset: bool = False  # Asset rejects >20% moves (notebook cell 42)
    halt_check: bool = False             # call asset.check_halt_status() and skip trading while halted
    evolution: bool = False              # every 100 days: 2 noise -> chartists, 1 chartist -> fundamentalist

    def validate(self):
        """Validate public run inputs without rejecting legacy scenario switches."""
        for name in ('noise', 'chartists', 'fundamentalists', 'chartist_window'):
            value = getattr(self, name)
            minimum = 1 if name == 'chartist_window' else 0
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.noise + self.chartists + self.fundamentalists == 0:
            raise ValueError('scenario must contain at least one trader')


SCENARIOS = {s.name: s for s in [
    Scenario('baseline', 'Baseline'),
    Scenario('more_noise', 'More noise traders', noise=500),
    # The notebook's "more chartists" cell is 25/130/15, not a 10x chartist count.
    Scenario('more_chartists', 'More chartists', noise=25, chartists=130, fundamentalists=15),
    Scenario('more_fundamentalists', 'More fundamentalists', fundamentalists=300),
    Scenario('circuit_breaker', 'Circuit Breaker', circuit_breaker_asset=True, halt_check=True),
    # In the notebook, cell 42's circuit-breaker Asset was still defined when the evolution cell ran, but that
    # cell never called check_halt_status(). Kept that way so results match the notebook exactly.
    Scenario('evolution', 'Trader evolution', circuit_breaker_asset=True, evolution=True),
    Scenario('window_5', 'Moving-average window: 5 steps', chartist_window=5),
    Scenario('window_50', 'Moving-average window: 50 steps', chartist_window=50),
]}
