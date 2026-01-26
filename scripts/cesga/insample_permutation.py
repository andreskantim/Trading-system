#!/usr/bin/env python3
"""
CESGA Cluster Version - In-Sample MCPT Analysis

Adapted for SLURM job scheduler and CESGA Finisterrae III cluster.
Uses MPI for distributed computing across nodes.

Usage:
    Direct: python insample_permutation.py <strategy> [options]
    SLURM:  sbatch run_insample.sbatch <strategy>

Environment variables (set by SLURM or manually):
    SLURM_NTASKS: Number of MPI tasks
    SLURM_CPUS_PER_TASK: CPUs per task
    SCRATCH: Scratch directory for temporary files
    STORE: Persistent storage directory
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import importlib
import argparse

# CESGA-specific path configuration
# Adjust these paths according to your CESGA project structure
CESGA_PROJECT_ROOT = Path(os.getenv('CESGA_PROJECT_ROOT', '/mnt/lustre/scratch/nlsas/home/usc/ec/ahe/Trading-system'))
CESGA_DATA_DIR = Path(os.getenv('CESGA_DATA_DIR', CESGA_PROJECT_ROOT / 'data'))
CESGA_OUTPUT_DIR = Path(os.getenv('CESGA_OUTPUT_DIR', CESGA_PROJECT_ROOT / 'outputs'))

# Add project root to path
sys.path.insert(0, str(CESGA_PROJECT_ROOT))

# Import project modules
from config.paths import (
    ensure_directories, ensure_ticker_output_dirs,
    TICKERS, get_ticker_data_paths
)
from backtest.mcpt.bar_permute import get_permutation

# Optional: Interactive visualization (usually disabled on cluster)
try:
    from visualization.interactive.lightweight_charts_viewer import create_interactive_chart
    HAS_INTERACTIVE_VIS = True
except ImportError:
    HAS_INTERACTIVE_VIS = False


# Global variables for worker initialization
_strategy_module = None
_train_df = None
_best_real_pf = None


def _init_worker(strategy_name, train_data, train_index, train_columns, best_real_pf):
    """Initialize worker with strategy module and data"""
    global _strategy_module, _train_df, _best_real_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    _train_df = pd.DataFrame(train_data, index=train_index, columns=train_columns)
    _best_real_pf = best_real_pf


def process_permutation_generic(perm_i):
    """Process a single permutation - optimized version"""
    strategy = _strategy_module
    train_df = _train_df
    best_real_pf = _best_real_pf

    # Execute permutation
    train_perm = get_permutation(train_df, seed=perm_i)
    result = strategy.optimize(train_perm)

    # Unpack result: last value is pf, rest are parameters
    *best_params, best_perm_pf = result

    # Calculate cumulative returns for this permutation
    sig = strategy.signal(train_perm, *best_params)
    r = np.log(train_perm['close']).diff().shift(-1)
    perm_rets = sig * r
    cum_rets = perm_rets.cumsum().values

    is_better = 1 if best_perm_pf >= best_real_pf else 0
    return best_perm_pf, is_better, cum_rets


def get_slurm_config():
    """Get configuration from SLURM environment variables"""
    config = {
        'job_id': os.getenv('SLURM_JOB_ID', 'local'),
        'job_name': os.getenv('SLURM_JOB_NAME', 'mcpt_insample'),
        'ntasks': int(os.getenv('SLURM_NTASKS', '1')),
        'cpus_per_task': int(os.getenv('SLURM_CPUS_PER_TASK', str(cpu_count()))),
        'node_name': os.getenv('SLURM_NODELIST', 'localhost'),
        'scratch': Path(os.getenv('SCRATCH', '/tmp')),
        'store': Path(os.getenv('STORE', CESGA_OUTPUT_DIR)),
    }
    # Total workers = cpus_per_task (single node) or use MPI for multi-node
    config['n_workers'] = config['cpus_per_task']
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CESGA Cluster - MCPT In-Sample Analysis'
    )
    parser.add_argument(
        'strategy',
        type=str,
        help='Strategy name (e.g., donchian, hawkes, moving_average)'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default='BTCUSD',
        help=f'Ticker symbol (default: BTCUSD). Available: {", ".join(TICKERS.keys())}'
    )
    parser.add_argument(
        '--train-start',
        type=str,
        default='2016-01-01',
        help='Training period start date YYYY-MM-DD (default: 2016-01-01)'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        default='2020-01-01',
        help='Training period end date YYYY-MM-DD (default: 2020-01-01)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Number of permutations (default: 1000)'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of workers (default: auto-detect from SLURM or CPU count)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: project outputs/backtest/figures)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    slurm_config = get_slurm_config()

    print("\n" + "="*70)
    print("CESGA CLUSTER - MCPT IN-SAMPLE ANALYSIS")
    print("="*70)
    print(f"SLURM Job ID: {slurm_config['job_id']}")
    print(f"Node: {slurm_config['node_name']}")
    print("="*70 + "\n")

    strategy_file = args.strategy
    strategy_name = strategy_file.replace('.py', '') if strategy_file.endswith('.py') else strategy_file

    # Load strategy module
    try:
        strategy = importlib.import_module(f'models.strategies.{strategy_name}')
        print(f"Strategy loaded: {strategy_name}")
    except ModuleNotFoundError:
        print(f"ERROR: Strategy module '{strategy_name}' not found")
        sys.exit(1)

    if not hasattr(strategy, 'optimize') or not hasattr(strategy, 'signal'):
        print(f"ERROR: Module '{strategy_name}' must have 'optimize' and 'signal' functions")
        sys.exit(1)

    # Ensure output directories
    ensure_directories()

    # Worker configuration
    if args.n_workers:
        n_workers = args.n_workers
    else:
        n_workers = slurm_config['n_workers']

    n_permutations = args.n_permutations

    # Get ticker data paths
    try:
        ticker_paths = get_ticker_data_paths(args.ticker)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Ticker:           {args.ticker}")
    print(f"  Training period:  {args.train_start} to {args.train_end}")
    print(f"  Workers:          {n_workers}")
    print(f"  Permutations:     {n_permutations}")
    print(f"  Scratch dir:      {slurm_config['scratch']}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Load data
    print(f"Loading data for {args.ticker}...")
    parquet_path = ticker_paths['parquet']
    csv_path = ticker_paths['csv']

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.index = df.index.astype('datetime64[s]')
    elif csv_path.exists():
        print("Converting CSV to Parquet...")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df.to_parquet(parquet_path)
    else:
        print(f"ERROR: No data files found for {args.ticker}")
        sys.exit(1)

    print(f"Data loaded: {len(df):,} rows\n")

    # In-sample analysis
    print("="*70)
    print("IN-SAMPLE OPTIMIZATION")
    print("="*70)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    train_df = df[(df.index >= train_start) & (df.index < train_end)]

    if len(train_df) == 0:
        print(f"ERROR: No data found in training period {args.train_start} to {args.train_end}")
        sys.exit(1)

    print(f"  Training period:    {train_start} to {train_end}")
    print(f"  Training samples:   {len(train_df):,}")

    result = strategy.optimize(train_df)
    *best_params, best_real_pf = result

    # Calculate cumulative returns for real strategy
    real_signal = strategy.signal(train_df, *best_params)
    real_r = np.log(train_df['close']).diff().shift(-1)
    real_rets = real_signal * real_r
    real_cum_rets = real_rets.cumsum()

    print(f"  Best Parameters:    {best_params}")
    print(f"  Best Profit Factor: {best_real_pf:.4f}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # MCPT with multiprocessing
    print(f"Running MCPT with {n_permutations} permutations using {n_workers} workers...")
    print()
    sys.stdout.flush()

    # Prepare data for worker initialization
    train_data = train_df.values
    train_index = train_df.index
    train_columns = train_df.columns.tolist()
    args_list = list(range(n_permutations))

    print("="*70)
    print("PROGRESS")
    print("="*70)
    print(f"  Start: {time.strftime('%H:%M:%S')}")
    print("="*70 + "\n")

    start_time = time.time()
    results = []
    chunksize = max(1, n_permutations // (n_workers * 10))

    # Use multiprocessing Pool
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(strategy_name, train_data, train_index, train_columns, best_real_pf)) as pool:
        for result in tqdm(pool.imap_unordered(process_permutation_generic, args_list, chunksize=chunksize),
                          total=len(args_list),
                          desc="Processing permutations",
                          ncols=80):
            results.append(result)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.0f}s")

    # Analyze results
    permuted_pfs = [pf for pf, _, _ in results]
    perm_better_count = 1 + sum(is_better for _, is_better, _ in results)
    perm_cum_rets = [cum_rets for _, _, cum_rets in results]
    insample_mcpt_pval = perm_better_count / n_permutations

    print("="*70)
    print("MCPT RESULTS")
    print("="*70)
    print(f"  Permutations:     {len(results)}")
    print(f"  Better than real: {perm_better_count}")
    print(f"  P-Value:          {insample_mcpt_pval:.4f}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Speed:            {len(results)/total_time:.1f} tasks/s")

    if insample_mcpt_pval < 0.05:
        print(f"  SIGNIFICANT (p < 0.05)")
    else:
        print(f"  NOT significant (p >= 0.05)")

    print("="*70 + "\n")
    sys.stdout.flush()

    # Generate plots
    print("Generating plots...")
    plt.style.use('dark_background')

    # Output directory - new structure
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir
    else:
        output_dirs = ensure_ticker_output_dirs(strategy_name, args.ticker)
        output_dir = output_dirs['figures']
        results_dir = output_dirs['results']

    # Histogram plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(permuted_pfs, bins=50, color='steelblue', alpha=0.7, edgecolor='white', label='Permutations')
    ax.axvline(best_real_pf, color='red', linestyle='--', linewidth=2.5, label=f'Real PF: {best_real_pf:.4f}')
    mean_perm = np.mean(permuted_pfs)
    ax.axvline(mean_perm, color='yellow', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Perm: {mean_perm:.4f}')
    ax.set_xlabel("Profit Factor", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"In-sample MCPT ({strategy_name}) | P-Value: {insample_mcpt_pval:.4f} | "
                 f"{'Significant' if insample_mcpt_pval < 0.05 else 'Not Significant'}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    output_file = output_dir / f'{args.ticker}_insample_mcpt.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Plot saved: {output_file}")
    plt.close()

    # Cumulative returns plot
    fig, ax = plt.subplots(figsize=(14, 8))
    perm_matrix = np.array(perm_cum_rets).T
    ax.plot(train_df.index, perm_matrix, color='white', alpha=0.02, linewidth=0.5)
    ax.plot(train_df.index, real_cum_rets, color='red', linewidth=2.5,
            label=f'Real Strategy (PF={best_real_pf:.4f})', zorder=100)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"{args.ticker} In-sample Cumulative Returns ({strategy_name})",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()

    output_file_cum = output_dir / f'{args.ticker}_insample_cumulative.png'
    plt.savefig(output_file_cum, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Cumulative returns plot saved: {output_file_cum}")
    plt.close()

    # Save results to CSV for later analysis
    results_df = pd.DataFrame({
        'permutation': range(len(permuted_pfs)),
        'profit_factor': permuted_pfs,
        'better_than_real': [1 if pf >= best_real_pf else 0 for pf in permuted_pfs]
    })
    results_csv = results_dir / f'{args.ticker}_insample_results_{slurm_config["job_id"]}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved: {results_csv}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETED")
    print("="*70)
    print(f"\nFigures: {output_dir}")
    print(f"  - {args.ticker}_insample_mcpt.png")
    print(f"  - {args.ticker}_insample_cumulative.png")
    print(f"Results: {results_dir}")
    print(f"  - {args.ticker}_insample_results_{slurm_config['job_id']}.csv")
    print("="*70 + "\n")
