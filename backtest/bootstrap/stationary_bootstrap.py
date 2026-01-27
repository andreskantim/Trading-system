"""
Stationary Bootstrap (Politis & Romano, 1994)

Uses random block lengths drawn from geometric distribution.
Produces stationary resampled series while preserving dependencies.
"""

import numpy as np
from typing import Callable


def _generate_stationary_indices(n: int, expected_block_length: float) -> np.ndarray:
    """Generate bootstrap indices using stationary bootstrap with geometric block lengths."""
    p = 1.0 / expected_block_length  # Probability of starting new block
    indices = []
    pos = np.random.randint(0, n)

    while len(indices) < n:
        indices.append(pos)
        if np.random.random() < p:
            pos = np.random.randint(0, n)
        else:
            pos = (pos + 1) % n

    return np.array(indices[:n])


def stationary_bootstrap(
    returns: np.ndarray,
    signal: np.ndarray,
    strategy_fn: Callable,
    n_iterations: int = 1000,
    expected_block_length: float = 20.0,
    seed: int = None
) -> dict:
    """
    Stationary Bootstrap for strategy returns.

    Args:
        returns: Original strategy returns array
        signal: Original trading signal array
        strategy_fn: Function to calculate strategy metric
        n_iterations: Number of bootstrap iterations
        expected_block_length: Expected block length (inverse of geometric parameter)
        seed: Random seed for reproducibility

    Returns:
        Dict with bootstrap distribution and statistics
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(returns)
    real_metric = strategy_fn(returns, signal)

    bootstrap_metrics = []

    for _ in range(n_iterations):
        indices = _generate_stationary_indices(n, expected_block_length)
        boot_returns = returns[indices]
        boot_signal = signal[indices]

        metric = strategy_fn(boot_returns, boot_signal)
        bootstrap_metrics.append(metric)

    bootstrap_metrics = np.array(bootstrap_metrics)

    p_value = np.mean(bootstrap_metrics >= real_metric)
    ci_lower = np.percentile(bootstrap_metrics, 2.5)
    ci_upper = np.percentile(bootstrap_metrics, 97.5)

    return {
        'method': 'stationary_bootstrap',
        'real_metric': float(real_metric),
        'bootstrap_metrics': bootstrap_metrics,
        'p_value': float(p_value),
        'mean': float(np.mean(bootstrap_metrics)),
        'median': float(np.median(bootstrap_metrics)),
        'std': float(np.std(bootstrap_metrics)),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'n_iterations': n_iterations,
        'expected_block_length': expected_block_length,
    }
