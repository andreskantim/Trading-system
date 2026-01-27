#!/usr/bin/env python3
"""
Batch-level statistics and plotting functions.

Functions:
- calculate_batch_statistics: Returns mean and median for all metrics
- plot_batch_results: Generates individual metric histograms and cumulative plot
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

    flat_stats = [flatten(s) for s in ticker_stats_list]

    all_keys = set()
    for fs in flat_stats:
        all_keys.update(fs.keys())

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


# Metrics that should be sorted ascending (lower is better)
ASCENDING_METRICS = {'p_value', 'max_dd', 'max_dd_duration', 'max_drawdown_pct', 'max_drawdown_duration'}


def _sort_data(tickers: List[str], values: List[float], metric_name: str) -> tuple:
    """Sort tickers and values by metric. Lower is better for some metrics."""
    ascending = any(m in metric_name.lower() for m in ASCENDING_METRICS)
    sorted_pairs = sorted(zip(tickers, values), key=lambda x: x[1], reverse=not ascending)
    return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]


def _plot_single_metric(values: List[float], tickers: List[str], title: str, ylabel: str,
                        threshold: float, thresh_label: str, output_path: Path,
                        metric_name: str):
    """Plot single metric histogram with proper sorting."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    sorted_tickers, sorted_values = _sort_data(tickers, values, metric_name)

    colors = []
    for v in sorted_values:
        if threshold is not None:
            if metric_name in ASCENDING_METRICS:
                colors.append('green' if v <= threshold else 'red')
            else:
                colors.append('green' if v >= threshold else 'red')
        else:
            colors.append('steelblue')

    ax.bar(range(len(sorted_tickers)), sorted_values, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(np.mean(sorted_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(sorted_values):.4f}')
    ax.axhline(np.median(sorted_values), color='cyan', linestyle=':', linewidth=2,
               label=f'Median: {np.median(sorted_values):.4f}')

    if threshold is not None:
        ax.axhline(threshold, color='yellow', linestyle='-', linewidth=1.5, alpha=0.8,
                   label=thresh_label)

    ax.set_xticks(range(len(sorted_tickers)))
    ax.set_xticklabels(sorted_tickers, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Ticker', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  {metric_name}: {output_path}")


def plot_batch_cumulative(ticker_results: List[Dict], batch_name: str, prefix: str,
                          output_dir: Path) -> Path:
    """
    Generate cumulative log return plot for batch.

    All individual tickers as thin lines + mean + median.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9))

    cum_returns_list = []
    indices = []

    for r in ticker_results:
        ticker = r.get('ticker', 'Unknown')
        cum_rets = r.get('cum_rets')
        index = r.get('index')

        if cum_rets is not None and len(cum_rets) > 0:
            ax.plot(index if index is not None else range(len(cum_rets)),
                    cum_rets, alpha=0.3, linewidth=0.8, label=ticker)
            cum_returns_list.append(cum_rets)
            if index is not None:
                indices.append(index)

    if cum_returns_list:
        # Align all series to same length (use shortest)
        min_len = min(len(cr) for cr in cum_returns_list)
        aligned = np.array([cr[:min_len] for cr in cum_returns_list])

        mean_cum = np.mean(aligned, axis=0)
        median_cum = np.median(aligned, axis=0)

        x_axis = indices[0][:min_len] if indices else range(min_len)

        ax.plot(x_axis, mean_cum, color='red', linewidth=2.5, linestyle='--',
                label=f'Mean ({mean_cum[-1]:.4f})', zorder=100)
        ax.plot(x_axis, median_cum, color='cyan', linewidth=2.5, linestyle=':',
                label=f'Median ({median_cum[-1]:.4f})', zorder=100)

    ax.axhline(0, color='yellow', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Cumulative Log Return', fontsize=11)
    ax.set_title(f'{batch_name} {prefix.title()} Cumulative Returns', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'cumulative_log_{prefix}_{batch_name}.png'
    plt.savefig(output_path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  cumulative_log: {output_path}")
    return output_path


def plot_batch_results(batch_name: str, strategy: str, ticker_results: List[Dict],
                       batch_stats: Dict, output_dir: Path, prefix: str = 'insample'):
    """
    Generate plots for batch results.

    Generates individual histograms for each metric:
    - p_value_{prefix}_{batch}.png
    - pnl_{prefix}_{batch}.png
    - cagr_{prefix}_{batch}.png
    - max_dd_{prefix}_{batch}.png
    - max_dd_duration_{prefix}_{batch}.png
    - sharpe_{prefix}_{batch}.png
    - sqn_{prefix}_{batch}.png
    - cumulative_log_{prefix}_{batch}.png

    Args:
        batch_name: Name of the batch run
        strategy: Strategy name
        ticker_results: List of individual ticker result dicts
        batch_stats: Aggregated batch statistics
        output_dir: Output directory for plots
        prefix: 'insample' or 'walkforward'
    """
    if not ticker_results:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [r.get('ticker', f'T{i}') for i, r in enumerate(ticker_results)]

    # Extract metrics
    p_values = [r.get('p_value', 0) for r in ticker_results]
    pnls, cagrs, max_dds, max_dd_durs, sharpes, sqns = [], [], [], [], [], []

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

    output_files = []

    # Individual metric plots
    metrics_config = [
        (p_values, 'P-Value', 'p-value', 0.05, 'p=0.05', 'p_value'),
        (pnls, 'Total PnL (log)', 'PnL', 0, 'break-even', 'pnl'),
        (cagrs, 'CAGR (%)', 'CAGR %', 0, 'break-even', 'cagr'),
        (max_dds, 'Max Drawdown (%)', 'Max DD %', 20, 'threshold', 'max_dd'),
        (max_dd_durs, 'Max DD Duration (bars)', 'Duration (bars)', None, None, 'max_dd_duration'),
        (sharpes, 'Sharpe Ratio', 'Sharpe', 0.5, 'threshold', 'sharpe'),
        (sqns, 'SQN', 'SQN', 1.5, 'threshold', 'sqn'),
    ]

    for vals, title, ylabel, thresh, thresh_label, metric_name in metrics_config:
        out_path = output_dir / f'{metric_name}_{prefix}_{batch_name}.png'
        _plot_single_metric(vals, tickers, f'{title} - {batch_name} ({strategy})',
                            ylabel, thresh, thresh_label, out_path, metric_name)
        output_files.append(out_path)

    # Cumulative log returns plot (all tickers + mean + median)
    cum_path = plot_batch_cumulative(ticker_results, batch_name, prefix, output_dir)
    output_files.append(cum_path)

    return output_files
