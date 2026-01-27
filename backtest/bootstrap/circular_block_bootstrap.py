"""
Circular Block Bootstrap (CBB)

Resamples blocks of consecutive returns while preserving autocorrelation.
Blocks wrap around circularly to maintain stationarity.
"""

import numpy as np
from typing import List, Tuple, Callable


def _generate_cbb_indices(n: int, block_size: int) -> np.ndarray:
    """Generate bootstrap indices using circular block bootstrap."""
    n_blocks = int(np.ceil(n / block_size))
    indices = []

    for _ in range(n_blocks):
        start = np.random.randint(0, n)
        block = [(start + i) % n for i in range(block_size)]
        indices.extend(block)

    return np.array(indices[:n])


def circular_block_bootstrap(
    returns: np.ndarray,
    signal: np.ndarray,
    strategy_fn: Callable,
    n_iterations: int = 1000,
    block_size: int = 20,
    seed: int = None
) -> dict:
    """
    Circular Block Bootstrap for strategy returns.

    Args:
        returns: Original strategy returns array
        signal: Original trading signal array
        strategy_fn: Function to calculate strategy metric (receives resampled returns)
        n_iterations: Number of bootstrap iterations
        block_size: Size of each block for resampling
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
        # Generate block bootstrap indices
        indices = _generate_cbb_indices(n, block_size)

        # Resample returns and signal
        boot_returns = returns[indices]
        boot_signal = signal[indices]

        # Calculate metric on resampled data
        metric = strategy_fn(boot_returns, boot_signal)
        bootstrap_metrics.append(metric)

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Calculate statistics
    p_value = np.mean(bootstrap_metrics >= real_metric)
    ci_lower = np.percentile(bootstrap_metrics, 2.5)
    ci_upper = np.percentile(bootstrap_metrics, 97.5)

    return {
        'method': 'circular_block_bootstrap',
        'real_metric': float(real_metric),
        'bootstrap_metrics': bootstrap_metrics,
        'p_value': float(p_value),
        'mean': float(np.mean(bootstrap_metrics)),
        'median': float(np.median(bootstrap_metrics)),
        'std': float(np.std(bootstrap_metrics)),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'n_iterations': n_iterations,
        'block_size': block_size,
    }


def calculate_profit_factor(returns: np.ndarray, signal: np.ndarray = None) -> float:
    """Calculate profit factor from returns."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses > 0 else (float('inf') if gains > 0 else 0.0)
