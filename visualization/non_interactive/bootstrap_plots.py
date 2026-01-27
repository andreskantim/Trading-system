#!/usr/bin/env python3
"""
Bootstrap visualization functions.

Generates plots for individual ticker and batch bootstrap analysis.
Adapted from MCPT plotting functions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_bootstrap_results(
    ticker: str,
    strategy: str,
    bootstrap_type: str,
    index: pd.DatetimeIndex,
    real_cum_rets: np.ndarray,
    bootstrap_metrics: np.ndarray,
    real_metric: float,
    p_value: float,
    output_dir: Path,
    ci_95: tuple = None,
):
    """
    Generate plots for single ticker bootstrap results.

    Generates:
    1. Bootstrap distribution histogram with real metric
    2. Cumulative returns plot

    Args:
        ticker: Ticker symbol
        strategy: Strategy name
        bootstrap_type: Bootstrap method name
        index: DatetimeIndex for time series
        real_cum_rets: Real cumulative returns
        bootstrap_metrics: Array of bootstrap metric values
        real_metric: Real strategy metric (e.g., profit factor)
        p_value: Bootstrap p-value
        output_dir: Output directory for plots
        ci_95: Optional 95% confidence interval tuple
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')

    prefix = f'{ticker}_bootstrap_{bootstrap_type}'

    # Plot 1: Bootstrap Distribution
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(bootstrap_metrics, bins=50, color='steelblue', alpha=0.7,
            edgecolor='white', label='Bootstrap Distribution')
    ax.axvline(real_metric, color='red', linestyle='--', linewidth=2.5,
               label=f'Real: {real_metric:.4f}')
    ax.axvline(np.mean(bootstrap_metrics), color='yellow', linestyle=':',
               linewidth=2, alpha=0.7, label=f'Mean: {np.mean(bootstrap_metrics):.4f}')
    ax.axvline(np.median(bootstrap_metrics), color='cyan', linestyle=':',
               linewidth=2, alpha=0.7, label=f'Median: {np.median(bootstrap_metrics):.4f}')

    if ci_95:
        ax.axvline(ci_95[0], color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(ci_95[1], color='green', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]')

    ax.set_xlabel('Metric Value (Profit Factor)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{ticker} {bootstrap_type.replace("_", " ").title()} Bootstrap ({strategy}) | p={p_value:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_file = output_dir / f'{prefix}_distribution.png'
    plt.savefig(hist_file, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Distribution: {hist_file}")

    # Plot 2: Cumulative Returns
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(index, real_cum_rets, color='red', linewidth=2,
            label=f'Real (PF={real_metric:.4f})')
    ax.axhline(0, color='yellow', linewidth=1, alpha=0.5)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Cumulative Log Return', fontsize=11)
    ax.set_title(f'{ticker} Cumulative Returns ({strategy}) | Bootstrap p={p_value:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    cum_file = output_dir / f'{prefix}_cumulative.png'
    plt.savefig(cum_file, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Cumulative: {cum_file}")

    return [hist_file, cum_file]


def plot_bootstrap_batch_comparison(
    batch_name: str,
    strategy: str,
    ticker_results: List[dict],
    output_dir: Path,
    bootstrap_type: str = 'bootstrap',
):
    """
    Generate comparison plot for batch bootstrap results.

    Shows all tickers' p-values and metrics sorted.

    Args:
        batch_name: Batch identifier
        strategy: Strategy name
        ticker_results: List of ticker result dicts
        output_dir: Output directory
        bootstrap_type: Bootstrap method name
    """
    if not ticker_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')

    tickers = [r.get('ticker', f'T{i}') for i, r in enumerate(ticker_results)]
    p_values = [r.get('p_value', 1.0) for r in ticker_results]

    # Sort by p-value (ascending - best first)
    sorted_pairs = sorted(zip(tickers, p_values), key=lambda x: x[1])
    sorted_tickers = [p[0] for p in sorted_pairs]
    sorted_pvalues = [p[1] for p in sorted_pairs]

    colors = ['green' if p < 0.05 else 'red' for p in sorted_pvalues]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(range(len(sorted_tickers)), sorted_pvalues, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(0.05, color='yellow', linestyle='--', linewidth=2, label='p=0.05')
    ax.axhline(np.mean(sorted_pvalues), color='red', linestyle=':', linewidth=2,
               label=f'Mean: {np.mean(sorted_pvalues):.4f}')

    ax.set_xticks(range(len(sorted_tickers)))
    ax.set_xticklabels(sorted_tickers, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Ticker', fontsize=11)
    ax.set_ylabel('P-Value', fontsize=11)
    ax.set_title(f'{batch_name} Bootstrap P-Values ({strategy})', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'{batch_name}_{bootstrap_type}_pvalues.png'
    plt.savefig(output_file, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  P-Values: {output_file}")

    return output_file
