"""Price, wealth, and trader-population plots for simulation results."""

import os

import matplotlib.pyplot as plt

from .scenarios import INITIAL_WEALTH

_TYPE_STYLE = {
    'noise': ('b', 'Noise traders'),
    'chartist': ('g', 'Chartists'),
    'fundamentalist': ('r', 'Fundamentalists'),
}


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_price(asset, title=None, show_halts=False, save_path=None):
    """Plot recorded prices, including warm-up; halt markers use legacy offsets."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(asset.price_history, label='Price')
    if show_halts:
        # Marks simulation days; price_history also holds the 20 warm-up steps, so the offset is approximate.
        for i, halt_day in enumerate(getattr(asset, 'halt_events', [])):
            ax.axvline(x=halt_day, color='red', linestyle='--', alpha=0.7, linewidth=1,
                       label='Trading halted' if i == 0 else None)
    ax.set_xlabel('Day')
    ax.set_ylabel('Price')
    if title:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    return _finish(fig, save_path)


def plot_wealth_history(wealth_history, save_path=None):
    """Per-type mean wealth over active traders, sampled every 100 days, against the initial endowment."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    for ax, (trader_type, (color, label)) in zip(axes, _TYPE_STYLE.items()):
        ax.plot(wealth_history['days'], wealth_history[f'{trader_type}_avg'], f'{color}-', linewidth=2)
        ax.set_title(f'{label}: mean wealth over active traders')
        ax.set_xlabel('Day')
        ax.set_ylabel('Mean wealth')
        ax.grid(True)
        ax.axhline(y=INITIAL_WEALTH[trader_type], color='r', linestyle='--', alpha=0.5, label='Initial wealth')
        ax.legend()
    return _finish(fig, save_path)


def plot_wealth_histograms(traders, scale=1.0, save_path=None):
    """Final wealth of every trader (active or not), divided by `scale`."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (trader_type, (color, label)) in zip(axes, _TYPE_STYLE.items()):
        ax.hist([t.wealth_history[-1] / scale for t in traders if t.type == trader_type], bins=20, color=color)
        ax.set_title(f'{label}: wealth distribution')
        ax.set_xlabel('Wealth' if scale == 1.0 else f'Wealth / {scale:g}')
        ax.set_ylabel('Number of traders')
        ax.grid(True)
    return _finish(fig, save_path)


def plot_trader_counts(asset, save_path=None):
    """Active traders per type, sampled every 100 days (useful for the evolution scenario)."""
    history = asset.trader_counts_history
    days = [h['day'] for h in history]
    fig, ax = plt.subplots(figsize=(14, 5))
    for trader_type, marker in zip(_TYPE_STYLE, 'os^'):
        color, label = _TYPE_STYLE[trader_type]
        ax.plot(days, [h[trader_type] for h in history], f'{color}-{marker}', linewidth=2, markersize=4, label=label)
    ax.set_xlabel('Day')
    ax.set_ylabel('Active traders')
    ax.set_title('Trader population over time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _finish(fig, save_path)
