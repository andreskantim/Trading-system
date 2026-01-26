#!/usr/bin/env python3
"""
Batch Walk-Forward Backtest with aggregation and visualization.

Usage:
    python batch_walkforward.py --group crypto_10 --strategy hawkes
    python batch_walkforward.py --tickers BTC ETH SOL --strategy donchian
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.tickers import get_ticker_group, TICKER_GROUPS
from config.paths import ensure_batch_output_dirs, get_ticker_output_dir
from utils.stats_calculator import calculate_batch_stats
from visualization.non_interactive.stats_and_plots_batch import plot_batch_results


def load_ticker_results(strategy: str, ticker: str) -> dict:
    results_dir = get_ticker_output_dir(strategy, ticker, 'results')
    results_file = results_dir / f'{ticker}_walkforward_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def aggregate_results(results: list) -> dict:
    if not results:
        return {}

    p_values = [r['p_value'] for r in results if 'p_value' in r]
    pfs = [r['real_pf'] for r in results if 'real_pf' in r]

    aggregated = {
        'n_tickers': len(results),
        'n_significant': sum(1 for r in results if r.get('significant', False)),
        'p_value': {'mean': float(np.mean(p_values)), 'median': float(np.median(p_values)),
                    'std': float(np.std(p_values)) if len(p_values) > 1 else None},
        'profit_factor': {'mean': float(np.mean(pfs)), 'median': float(np.median(pfs)),
                         'std': float(np.std(pfs)) if len(pfs) > 1 else None},
        'tickers': {r['ticker']: {'p_value': r['p_value'], 'pf': r['real_pf'], 'significant': r['significant']}
                    for r in results if 'ticker' in r},
    }

    stats_results = [r for r in results if 'stats' in r]
    if stats_results:
        aggregated['stats_aggregated'] = calculate_batch_stats(stats_results)

    return aggregated


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch walk-forward backtest')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--group', type=str, choices=list(TICKER_GROUPS.keys()))
    group.add_argument('--tickers', type=str, nargs='+')

    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--start-train', type=str)
    parser.add_argument('--end-train', type=str)
    parser.add_argument('--start-walk', type=str)
    parser.add_argument('--end-walk', type=str)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--n-workers', type=int, default=None)

    args = parser.parse_args()

    if args.group:
        tickers = get_ticker_group(args.group)
        batch_name = f"{args.group}_walkforward"
        print(f"\n{'='*70}\nBATCH WALK-FORWARD\n{'='*70}")
        print(f"Grupo: {args.group} ({len(tickers)} tickers)")
    else:
        tickers = [t.upper() for t in args.tickers]
        batch_name = f"custom_{len(tickers)}_walkforward"
        print(f"\n{'='*70}\nBATCH WALK-FORWARD\n{'='*70}")
        print(f"Tickers: {', '.join(tickers)}")

    print(f"Estrategia: {args.strategy}")
    print(f"Permutaciones: {args.n_permutations}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    successful, failed = [], []

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] {ticker}")

        cmd = [sys.executable, str(project_root / 'scripts' / 'laptop' / 'walkforward_permutation.py'),
               '--ticker', ticker, '--strategy', args.strategy,
               '--n-permutations', str(args.n_permutations)]

        if args.start_train:
            cmd.extend(['--start-train', args.start_train, '--end-train', args.end_train,
                       '--start-walk', args.start_walk, '--end-walk', args.end_walk])
        if args.n_workers:
            cmd.extend(['--n-workers', str(args.n_workers)])

        try:
            subprocess.run(cmd, check=True, capture_output=False)
            successful.append(ticker)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            failed.append(ticker)
            print(f"  ERROR: {e}")

    # Aggregate
    print(f"\n{'='*70}\nAGREGANDO RESULTADOS\n{'='*70}")
    all_results = [r for t in successful if (r := load_ticker_results(args.strategy, t))]

    if all_results:
        aggregated = aggregate_results(all_results)
        batch_dirs = ensure_batch_output_dirs(args.strategy, batch_name)

        batch_file = batch_dirs['results'] / f'{batch_name}_results.json'
        with open(batch_file, 'w') as f:
            json.dump({'batch_name': batch_name, 'strategy': args.strategy,
                      'timestamp': datetime.now().isoformat(), 'successful': successful,
                      'failed': failed, 'aggregated': aggregated}, f, indent=2)

        print("\nGenerando visualizaciones...")
        plot_batch_results(batch_name, args.strategy, all_results, aggregated, batch_dirs['figures'])

    # Summary
    duration = datetime.now() - start_time
    print(f"\n{'='*70}\nBATCH COMPLETADO\n{'='*70}")
    print(f"Tiempo: {duration} | Exitosos: {len(successful)}/{len(tickers)}")
    if all_results:
        print(f"Significativos: {sum(1 for r in all_results if r.get('significant'))}/{len(all_results)}")
        print(f"P-Value: mean={np.mean([r['p_value'] for r in all_results]):.4f}")
        print(f"PF: mean={np.mean([r['real_pf'] for r in all_results]):.4f}")


if __name__ == '__main__':
    main()
