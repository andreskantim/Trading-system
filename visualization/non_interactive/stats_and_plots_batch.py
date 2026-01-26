#!/usr/bin/env python3
"""
Batch-level statistics and plotting functions.

Functions:
- calculate_batch_statistics: Returns mean and median for all metrics
- plot_batch_results: Generates histograms and time-series plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_batch_statistics(ticker_stats_list: List[Dict]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics for a batch of tickers.

    Args:
        ticker_stats_list: List of statistics dicts from calculate_ticker_statistics

    Returns:
        Dict with mean and median for all metrics
    """
    if not ticker_stats_list:
        return {}

    def flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}.{k}' if parent_key else k
            if isinstance(v, dict) and not any(isinstance(vv, dict) for vv in v.values()):
                items.extend(flatten(v, new_key).items())
            elif isinstance(v, (int, float)) and v is not None:
                items.append((new_key, v))
        return dict(items)

    # Flatten all ticker stats
    flat_stats = [flatten(s) for s in ticker_stats_list]

    # Get all keys
    all_keys = set()
    for fs in flat_stats:
        all_keys.update(fs.keys())

    # Calculate aggregates
    aggregated = {}
    for key in sorted(all_keys):
        vals = [fs.get(key) for fs in flat_stats if fs.get(key) is not None]
        vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]

        if vals:
            aggregated[key] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)) if len(vals) > 1 else 0.0,
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'n': len(vals),
            }

    return aggregated


def _plot_histogram(ax, values: List[float], tickers: List[str], title: str, xlabel: str,
                    threshold: float = None, threshold_label: str = None):
    """Helper to plot histogram with mean/median lines."""
    colors = ['green' if v > (threshold or 0) else 'red' for v in values] if threshold else 'steelblue'

    ax.bar(range(len(tickers)), values, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
    ax.axhline(np.median(values), color='blue', linestyle=':', linewidth=2, label=f'Median: {np.median(values):.4f}')

    if threshold is not None:
        ax.axhline(threshold, color='yellow', linestyle='-', linewidth=1, alpha=0.7, label=threshold_label)

    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Ticker')
    ax.set_ylabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def plot_batch_results(batch_name: str, strategy: str, ticker_results: List[Dict],
                       batch_stats: Dict, output_dir: Path):
    """
    Generate plots for batch results.

    Plots:
    1. 7 histograms: p-value, PnL, CAGR, Max DD%, Max DD duration, Sharpe, SQN
    2. 1 time-series: Cumulative log return (thin lines=tickers, bold=mean/median)

    Args:
        batch_name: Name of the batch run
        strategy: Strategy name
        ticker_results: List of individual ticker result dicts
        batch_stats: Aggregated batch statistics
        output_dir: Output directory for plots
    """
    if not ticker_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')

    tickers = [r.get('ticker', f'T{i}') for i, r in enumerate(ticker_results)]

    # Extract metrics
    p_values = [r.get('p_value', 0) for r in ticker_results]
    pnls = []
    cagrs = []
    max_dds = []
    max_dd_durs = []
    sharpes = []
    sqns = []

    for r in ticker_results:
        stats = r.get('stats', {})
        perf = stats.get('performance', {})
        risk = stats.get('risk', {})
        rr = stats.get('risk_return', {})

        pnls.append(perf.get('total_pnl', perf.get('cumulative_log_return', 0)))
        cagrs.append(perf.get('cagr_pct', 0))
        max_dds.append(abs(risk.get('max_dd_pct', risk.get('max_drawdown_pct', 0))))
        max_dd_durs.append(risk.get('max_dd_duration', risk.get('max_drawdown_duration', 0)))
        sharpes.append(rr.get('sharpe_ratio', 0))
        sqns.append(rr.get('sqn', 0))

    # Plot 1: 7 Histograms (2 rows x 4 cols, last cell empty)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    metrics = [
        (p_values, 'P-Value', 'p-value', 0.05, 'p=0.05'),
        (pnls, 'Total PnL (log)', 'PnL', 0, 'break-even'),
        (cagrs, 'CAGR (%)', 'CAGR', 0, 'break-even'),
        (max_dds, 'Max Drawdown (%)', 'Max DD', None, None),
        (max_dd_durs, 'Max DD Duration (bars)', 'Duration', None, None),
        (sharpes, 'Sharpe Ratio', 'Sharpe', 0, 'zero'),
        (sqns, 'SQN', 'SQN', 0, 'zero'),
    ]

    for i, (vals, title, ylabel, thresh, thresh_label) in enumerate(metrics):
        _plot_histogram(axes[i], vals, tickers, f'{title} by Ticker', ylabel, thresh, thresh_label)

    # Hide last subplot
    axes[7].axis('off')

    plt.suptitle(f'{batch_name} - {strategy} Batch Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{batch_name}_histograms.png', dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Histograms: {output_dir / f'{batch_name}_histograms.png'}")

    # Plot 2: Cumulative Returns Time Series
    # Check if we have cumulative return data
    has_cum_data = False
    for r in ticker_results:
        if 'cum_rets' in r or ('stats' in r and 'temporal' in r['stats']):
            has_cum_data = True
            break

    if not has_cum_data:
        # Create simplified plot using final cumulative returns
        fig, ax = plt.subplots(figsize=(14, 8))

        # Simple bar chart of final cumulative returns
        ax.bar(range(len(tickers)), pnls, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axhline(np.mean(pnls), color='red', linewidth=2, linestyle='--', label=f'Mean: {np.mean(pnls):.4f}')
        ax.axhline(np.median(pnls), color='blue', linewidth=2, linestyle=':', label=f'Median: {np.median(pnls):.4f}')
        ax.axhline(0, color='yellow', linewidth=1, alpha=0.5)

        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        ax.set_xlabel('Ticker')
        ax.set_ylabel('Cumulative Log Return')
        ax.set_title(f'{batch_name} Final Cumulative Returns', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{batch_name}_cumulative.png', dpi=150, facecolor='#0d1117')
        plt.close()
        print(f"  Cumulative: {output_dir / f'{batch_name}_cumulative.png'}")

    # Plot 3: Summary Statistics Table (optional)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    table_data = [
        ['Metric', 'Mean', 'Median', 'Std', 'Min', 'Max'],
        ['P-Value', f'{np.mean(p_values):.4f}', f'{np.median(p_values):.4f}',
         f'{np.std(p_values):.4f}', f'{np.min(p_values):.4f}', f'{np.max(p_values):.4f}'],
        ['PnL', f'{np.mean(pnls):.4f}', f'{np.median(pnls):.4f}',
         f'{np.std(pnls):.4f}', f'{np.min(pnls):.4f}', f'{np.max(pnls):.4f}'],
        ['CAGR (%)', f'{np.mean(cagrs):.2f}', f'{np.median(cagrs):.2f}',
         f'{np.std(cagrs):.2f}', f'{np.min(cagrs):.2f}', f'{np.max(cagrs):.2f}'],
        ['Max DD (%)', f'{np.mean(max_dds):.2f}', f'{np.median(max_dds):.2f}',
         f'{np.std(max_dds):.2f}', f'{np.min(max_dds):.2f}', f'{np.max(max_dds):.2f}'],
        ['Sharpe', f'{np.mean(sharpes):.4f}', f'{np.median(sharpes):.4f}',
         f'{np.std(sharpes):.4f}', f'{np.min(sharpes):.4f}', f'{np.max(sharpes):.4f}'],
        ['SQN', f'{np.mean(sqns):.4f}', f'{np.median(sqns):.4f}',
         f'{np.std(sqns):.4f}', f'{np.min(sqns):.4f}', f'{np.max(sqns):.4f}'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(6):
        table[(0, j)].set_facecolor('#1f2937')
        table[(0, j)].set_text_props(weight='bold', color='white')

    ax.set_title(f'{batch_name} Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / f'{batch_name}_summary.png', dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Summary: {output_dir / f'{batch_name}_summary.png'}")

    return [
        output_dir / f'{batch_name}_histograms.png',
        output_dir / f'{batch_name}_cumulative.png',
        output_dir / f'{batch_name}_summary.png',
    ]
