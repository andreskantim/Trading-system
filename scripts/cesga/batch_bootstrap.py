#!/usr/bin/env python3
"""
Batch Bootstrap Analysis for multiple tickers.

Usage:
    python batch_bootstrap.py --group crypto_10 --strategy hawkes --bootstrap-type circular_block
    python batch_bootstrap.py --tickers BTC ETH SOL --strategy donchian --bootstrap-type trade_based
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
from visualization.non_interactive.batch_plots import calculate_batch_statistics, plot_batch_results
from backtest.report import generate_batch_report


def load_ticker_results(strategy: str, ticker: str, bootstrap_type: str) -> dict:
    results_dir = get_ticker_OUTPUTS_DIR(strategy, ticker, 'results')
    results_file = results_dir / f'{ticker}_bootstrap_{bootstrap_type}_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch Bootstrap Analysis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--group', type=str, choices=list(TICKER_GROUPS.keys()))
    group.add_argument('--tickers', type=str, nargs='+')

    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--bootstrap-type', type=str, required=True,
                        choices=['circular_block', 'stationary', 'trade_based'])
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--block-size', type=int, default=20)

    args = parser.parse_args()

    tickers = get_ticker_group(args.group) if args.group else [t.upper() for t in args.tickers]
    batch_name = f"{args.group}" if args.group else f"custom_{len(tickers)}"
    batch_type = f"{batch_name}_bootstrap_{args.bootstrap_type}"

    print(f"\n{'='*70}\nBATCH BOOTSTRAP ANALYSIS\n{'='*70}")
    print(f"Tickers: {len(tickers)} | Strategy: {args.strategy}")
    print(f"Bootstrap: {args.bootstrap_type} | Iterations: {args.n_iterations}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    successful, failed = [], []

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] {ticker}")

        cmd = [
            sys.executable,
            str(project_root / 'scripts' / 'laptop' / 'bootstrap.py'),
            '--ticker', ticker,
            '--strategy', args.strategy,
            '--bootstrap-type', args.bootstrap_type,
            '--n-iterations', str(args.n_iterations),
            '--block-size', str(args.block_size),
        ]

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
    print(f"\n{'='*70}\nAGGREGATING RESULTS\n{'='*70}")

    all_results = [r for t in successful if (r := load_ticker_results(args.strategy, t, args.bootstrap_type))]

    if all_results:
        ticker_stats = [r['stats'] for r in all_results if 'stats' in r]
        batch_stats = calculate_batch_statistics(ticker_stats)

        ticker_results = [{'ticker': r['ticker'], 'p_value': r['p_value'], 'stats': r.get('stats', {})}
                          for r in all_results]

        batch_dirs = ensure_batch_OUTPUTS_DIRs(args.strategy, batch_name)

        # Save JSON report
        report = {
            'batch_name': batch_type,
            'strategy': args.strategy,
            'bootstrap_type': args.bootstrap_type,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': len(successful) + len(failed),
                'successful': len(successful),
                'failed': len(failed),
                'n_significant': sum(1 for r in all_results if r.get('significant', False)),
            },
            'tickers': {'successful': successful, 'failed': failed},
            'batch_statistics': batch_stats,
        }

        with open(batch_dirs['results'] / f'{batch_type}_results.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        generate_batch_report(batch_type, args.strategy, ticker_results, batch_stats, batch_dirs['reports'])

        # Generate plots
        print("\nGenerating visualizations...")
        plot_batch_results(batch_type, args.strategy, ticker_results, batch_stats,
                           batch_dirs['figures'], prefix=f'bootstrap_{args.bootstrap_type}')

    # Summary
    duration = datetime.now() - start_time
    print(f"\n{'='*70}\nBATCH COMPLETED in {duration}\n{'='*70}")
    print(f"Successful: {len(successful)}/{len(tickers)} | Failed: {len(failed)}")
    if all_results:
        print(f"Significant: {sum(1 for r in all_results if r.get('significant'))}/{len(all_results)}")
        print(f"P-Value: mean={np.mean([r['p_value'] for r in all_results]):.4f}")


if __name__ == '__main__':
    main()
