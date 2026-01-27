"""
Trade-based Bootstrap

Resamples individual trades (complete trade sequences) rather than returns.
Preserves within-trade structure while randomizing trade order.
"""

import numpy as np
from typing import List, Callable


def _identify_trades(signal: np.ndarray, returns: np.ndarray) -> List[dict]:
    """Identify individual trades from signal and returns."""
    trades = []
    trade_start = None
    prev_sig = 0

    for i in range(len(signal)):
        sig = signal[i]

        if sig != 0 and prev_sig == 0:
            trade_start = i
        elif (sig == 0 and prev_sig != 0) or (sig != prev_sig and sig != 0 and prev_sig != 0):
            if trade_start is not None:
                trades.append({
                    'start': trade_start,
                    'end': i,
                    'returns': returns[trade_start:i].copy(),
                    'signals': signal[trade_start:i].copy(),
                })
            trade_start = i if sig != 0 else None

        prev_sig = sig

    if trade_start is not None:
        trades.append({
            'start': trade_start,
            'end': len(signal),
            'returns': returns[trade_start:].copy(),
            'signals': signal[trade_start:].copy(),
        })

    return trades


def trade_bootstrap(
    returns: np.ndarray,
    signal: np.ndarray,
    strategy_fn: Callable,
    n_iterations: int = 1000,
    seed: int = None
) -> dict:
    """
    Trade-based Bootstrap for strategy returns.

    Resamples complete trades with replacement.

    Args:
        returns: Original strategy returns array
        signal: Original trading signal array
        strategy_fn: Function to calculate strategy metric
        n_iterations: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dict with bootstrap distribution and statistics
    """
    if seed is not None:
        np.random.seed(seed)

    trades = _identify_trades(signal, returns)

    if len(trades) < 2:
        return {
            'method': 'trade_bootstrap',
            'error': 'Insufficient trades for bootstrap',
            'n_trades': len(trades),
        }

    real_metric = strategy_fn(returns, signal)
    n_trades = len(trades)

    bootstrap_metrics = []

    for _ in range(n_iterations):
        # Resample trades with replacement
        sampled_indices = np.random.choice(n_trades, size=n_trades, replace=True)

        # Reconstruct returns and signals
        boot_returns = np.concatenate([trades[i]['returns'] for i in sampled_indices])
        boot_signal = np.concatenate([trades[i]['signals'] for i in sampled_indices])

        metric = strategy_fn(boot_returns, boot_signal)
        bootstrap_metrics.append(metric)

    bootstrap_metrics = np.array(bootstrap_metrics)

    p_value = np.mean(bootstrap_metrics >= real_metric)
    ci_lower = np.percentile(bootstrap_metrics, 2.5)
    ci_upper = np.percentile(bootstrap_metrics, 97.5)

    return {
        'method': 'trade_bootstrap',
        'real_metric': float(real_metric),
        'bootstrap_metrics': bootstrap_metrics,
        'p_value': float(p_value),
        'mean': float(np.mean(bootstrap_metrics)),
        'median': float(np.median(bootstrap_metrics)),
        'std': float(np.std(bootstrap_metrics)),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'n_iterations': n_iterations,
        'n_trades': n_trades,
    }
