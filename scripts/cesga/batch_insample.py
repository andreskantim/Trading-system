#!/usr/bin/env python3
"""
Batch In-Sample Backtest with aggregation.

Usage:
    python batch_insample.py --group crypto_10 --strategy hawkes
    python batch_insample.py --tickers BTC ETH SOL --strategy donchian --n-permutations 10000
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import os

import numpy as np

if 'TRADING_ROOT' in os.environ:
    project_root = Path(os.environ['TRADING_ROOT'])
else:
    project_root = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(project_root))

from config.tickers import get_ticker_group, TICKER_GROUPS
from config.paths import ensure_batch_OUTPUTS_DIRs, get_ticker_OUTPUTS_DIR
from visualization.non_interactive.stats_and_plots_batch import calculate_batch_statistics, plot_batch_results


def load_ticker_results(strategy: str, ticker: str) -> dict:
    results_dir = get_ticker_OUTPUTS_DIR(strategy, ticker, 'results')
    results_file = results_dir / f'{ticker}_insample_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def save_batch_report(batch_dirs: dict, batch_name: str, strategy: str,
                      successful: list, failed: list, all_results: list, batch_stats: dict):
    """Save comprehensive batch report."""
    report = {
        'batch_name': batch_name,
        'strategy': strategy,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total': len(successful) + len(failed),
            'successful': len(successful),
            'failed': len(failed),
            'n_significant': sum(1 for r in all_results if r.get('significant', False)),
        },
        'tickers': {
            'successful': successful,
            'failed': failed,
        },
        'batch_statistics': batch_stats,
    }

    # Save to results
    with open(batch_dirs['results'] / f'{batch_name}_insample_results.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save to reports (human-readable)
    with open(batch_dirs['reports'] / f'{batch_name}_insample_report.json', 'w') as f:
        json.dump(report, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch in-sample backtest')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--group', type=str, choices=list(TICKER_GROUPS.keys()))
    group.add_argument('--tickers', type=str, nargs='+')

    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-workers', type=int, default=16)

    args = parser.parse_args()

    tickers = get_ticker_group(args.group) if args.group else [t.upper() for t in args.tickers]
    batch_name = f"{args.group}" if args.group else f"custom_{len(tickers)}"

    print(f"\n{'='*70}\nBATCH IN-SAMPLE\n{'='*70}")
    print(f"Tickers: {len(tickers)} | Strategy: {args.strategy} | Perms: {args.n_permutations}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    successful, failed = [], []

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] {ticker}")

        cmd = [sys.executable, str(project_root / 'scripts' / 'laptop' / 'insample_permutation.py'),
               '--ticker', ticker, '--strategy', args.strategy,
               '--n-permutations', str(args.n_permutations), '--n-workers', str(args.n_workers)]

        if args.start:
            cmd.extend(['--start', args.start])
        if args.end:
            cmd.extend(['--end', args.end])

        try:
            subprocess.run(cmd, check=True)
            successful.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"  ERROR: {e}")

    # Aggregate results
    print(f"\n{'='*70}\nAGREGATING RESULTS\n{'='*70}")

    all_results = [r for t in successful if (r := load_ticker_results(args.strategy, t))]

    if all_results:
        # Calculate batch statistics from all ticker stats
        ticker_stats = [r['stats'] for r in all_results if 'stats' in r]
        batch_stats = calculate_batch_statistics(ticker_stats)

        # Prepare ticker results for plotting
        ticker_results = [{'ticker': r['ticker'], 'p_value': r['p_value'], 'stats': r.get('stats', {})}
                          for r in all_results]

        batch_dirs = ensure_batch_OUTPUTS_DIRs(args.strategy, batch_name)

        # Save reports
        save_batch_report(batch_dirs, batch_name, args.strategy, successful, failed, all_results, batch_stats)

        # Generate plots
        print("\nGenerating visualizations...")
        plot_batch_results(batch_name, args.strategy, ticker_results, batch_stats, batch_dirs['figures'])

    # Summary
    duration = datetime.now() - start_time
    print(f"\n{'='*70}\nBATCH COMPLETED in {duration}\n{'='*70}")
    print(f"Successful: {len(successful)}/{len(tickers)} | Failed: {len(failed)}")
    if all_results:
        print(f"Significant: {sum(1 for r in all_results if r.get('significant'))}/{len(all_results)}")
        print(f"P-Value: mean={np.mean([r['p_value'] for r in all_results]):.4f}")
        print(f"PF: mean={np.mean([r['real_pf'] for r in all_results]):.4f}")


if __name__ == '__main__':
    main()
