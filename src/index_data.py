"""Parse localized index exports and report legacy-compatible return metrics."""

import os

import numpy as np
import pandas as pd

from src.metrics import calculate_hurst_exponent, calculate_lyapunov_exponent, calculate_sample_entropy

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

INDEX_FILES = {
    'РТС': 'ртс.csv',
    'ММВБ': 'ммвб.csv',
    'S&P500': 's&p500.csv',
    'DJIA': 'djia.csv',
    'NASDAQ': 'nasdaq.csv',
}


def load_index(path):
    """Read an Investing.com CSV export into a (Дата, Цена) frame, parsed as in notebook cell 54.

    Rows keep the file's newest-first order: the notebook never re-sorted by date, so its index
    log-returns are time-reversed and sign-flipped (see CLAUDE.md, "Data"). Kept for reproducibility.
    """
    data = pd.read_csv(path)[['Дата', 'Цена']]
    data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%Y')
    data['Цена'] = data['Цена'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
    return data


def index_metrics(prices, emb_dim=10):
    prices = np.asarray(prices)
    return {
        'hurst': calculate_hurst_exponent(prices),
        'lyapunov': calculate_lyapunov_exponent(prices, emb_dim=emb_dim, lag=1),
        'sampen': calculate_sample_entropy(prices, emb_dim=3),
    }


def print_index_metrics(name, metrics):
    print(f"\n{name}")
    print(f"Hurst exponent: {metrics['hurst']:.4f}")
    print(f"Lyapunov estimate (Rosenstein): {metrics['lyapunov']:.6f}")
    print(f"Sample Entropy (nats): {metrics['sampen']:.6f}")
