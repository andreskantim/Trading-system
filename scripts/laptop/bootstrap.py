#!/usr/bin/env python3
"""
Bootstrap Analysis Script

Performs bootstrap analysis on trading strategies using three methods:
- circular_block: Circular Block Bootstrap (CBB)
- stationary: Stationary Bootstrap (Politis & Romano)
- trade_based: Trade-based Bootstrap

Usage:
    python bootstrap.py --ticker BTC --strategy hawkes --bootstrap-type circular_block
    python bootstrap.py --ticker ETH --strategy donchian --bootstrap-type stationary --n-iterations 2000
    python bootstrap.py --ticker SOL --strategy hawkes --bootstrap-type trade_based --block-size 30
"""

import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
import importlib
import argparse

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import ensure_directories, ensure_ticker_OUTPUTS_DIRs
from backtest.bootstrap import circular_block_bootstrap, stationary_bootstrap, trade_bootstrap
from backtest.bootstrap.circular_block_bootstrap import calculate_profit_factor
from utils.data_loader import load_ticker_data, get_available_date_range
from utils.stats_calculator import calculate_all_stats
from visualization.non_interactive.report import generate_bootstrap_ticker_report
from visualization.non_interactive.bootstrap_plots import plot_bootstrap_results


def parse_args():
    parser = argparse.ArgumentParser(description='Bootstrap Analysis')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    parser.add_argument('--start', type=str, help='Start date DD/MM/YYYY')
    parser.add_argument('--end', type=str, help='End date DD/MM/YYYY')
    parser.add_argument('--bootstrap-type', type=str, required=True,
                        choices=['circular_block', 'stationary', 'trade_based'],
                        help='Bootstrap method')
    parser.add_argument('--n-iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--block-size', type=int, default=20, help='Block size for CBB/Stationary')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("BOOTSTRAP ANALYSIS")
    print("=" * 70 + "\n")

    strategy_name = args.strategy.replace('.py', '')

    try:
        strategy = importlib.import_module(f'models.strategies.{strategy_name}')
        print(f"Strategy loaded: {strategy_name}")
    except ModuleNotFoundError:
        print(f"ERROR: Strategy '{strategy_name}' not found")
        sys.exit(1)

    if not hasattr(strategy, 'optimize') or not hasattr(strategy, 'signal'):
        print(f"ERROR: '{strategy_name}' must have 'optimize' and 'signal' functions")
        sys.exit(1)

    ensure_directories()

    # Load data
    print(f"\nLoading data for {args.ticker}...")

    if not args.start and not args.end:
        start_available, end_available = get_available_date_range(args.ticker)
        if start_available:
            print(f"  Available range: {start_available} - {end_available}")

    try:
        df = load_ticker_data(args.ticker, start_date=args.start, end_date=args.end)
        print(f"  Data loaded: {len(df):,} candles")
        print(f"  From: {df.index.min()}")
        print(f"  To: {df.index.max()}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Bootstrap type: {args.bootstrap_type}")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  Block size: {args.block_size}")
    print("=" * 70 + "\n")

    # Optimize strategy
    print("Running strategy optimization...")
    result = strategy.optimize(df)
    *best_params, best_pf = result

    signal = strategy.signal(df, *best_params)
    r = np.log(df['close']).diff().shift(-1)
    returns = (signal * r).values
    signal_arr = signal.values

    print(f"  Best Parameters: {best_params}")
    print(f"  Best PF: {best_pf:.4f}")
    print("=" * 70 + "\n")

    # Run bootstrap
    print(f"Running {args.bootstrap_type} bootstrap with {args.n_iterations} iterations...")
    start_time = datetime.now()

    if args.bootstrap_type == 'circular_block':
        boot_result = circular_block_bootstrap(
            returns=returns,
            signal=signal_arr,
            strategy_fn=calculate_profit_factor,
            n_iterations=args.n_iterations,
            block_size=args.block_size,
        )
    elif args.bootstrap_type == 'stationary':
        boot_result = stationary_bootstrap(
            returns=returns,
            signal=signal_arr,
            strategy_fn=calculate_profit_factor,
            n_iterations=args.n_iterations,
            expected_block_length=float(args.block_size),
        )
    else:  # trade_based
        boot_result = trade_bootstrap(
            returns=returns,
            signal=signal_arr,
            strategy_fn=calculate_profit_factor,
            n_iterations=args.n_iterations,
        )

    duration = (datetime.now() - start_time).total_seconds()

    # Calculate full stats
    print("\nCalculating statistics...")
    full_stats = calculate_all_stats(
        returns=returns,
        signal=signal_arr,
        p_value_mcpt=boot_result['p_value'],
        periods_per_year=8760
    )

    # Prepare results
    results_dict = {
        'ticker': args.ticker,
        'strategy': strategy_name,
        'type': 'bootstrap',
        'bootstrap_method': args.bootstrap_type,
        'period': {
            'start': str(df.index.min()),
            'end': str(df.index.max())
        },
        'n_candles': len(df),
        'n_iterations': args.n_iterations,
        'best_params': best_params,
        'real_pf': float(best_pf),
        'p_value': boot_result['p_value'],
        'mean_bootstrap_pf': boot_result['mean'],
        'median_bootstrap_pf': boot_result['median'],
        'std_bootstrap_pf': boot_result['std'],
        'ci_95': boot_result['ci_95'],
        'execution_time_seconds': duration,
        'significant': boot_result['p_value'] < 0.05,
        'stats': full_stats,
    }

    # Save results
    output_dirs = ensure_ticker_OUTPUTS_DIRs(strategy_name, args.ticker)
    results_file = output_dirs['results'] / f'{args.ticker}_bootstrap_{args.bootstrap_type}_results.json'

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")

    # Generate markdown report
    print("\nGenerating report...")
    report_path = generate_bootstrap_ticker_report(results_dict, output_dirs['reports'], args.bootstrap_type)
    print(f"  Report: {report_path}")

    # Generate plots
    print("\nGenerating plots...")
    cum_rets = np.nancumsum(returns)

    plot_bootstrap_results(
        ticker=args.ticker,
        strategy=strategy_name,
        bootstrap_type=args.bootstrap_type,
        index=df.index,
        real_cum_rets=cum_rets,
        bootstrap_metrics=boot_result['bootstrap_metrics'],
        real_metric=boot_result['real_metric'],
        p_value=boot_result['p_value'],
        output_dir=output_dirs['figures'],
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BOOTSTRAP ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Ticker: {args.ticker}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Method: {args.bootstrap_type}")
    print(f"  Real PF: {best_pf:.4f}")
    print(f"  P-Value: {boot_result['p_value']:.4f}")
    print(f"  95% CI: [{boot_result['ci_95'][0]:.4f}, {boot_result['ci_95'][1]:.4f}]")
    print(f"  Significant: {'Yes' if boot_result['p_value'] < 0.05 else 'No'}")
    print(f"  Time: {duration:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
