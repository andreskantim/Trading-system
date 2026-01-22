#!/usr/bin/env python3
"""
CESGA Cluster Version - Walk-Forward MCPT Analysis

Adapted for SLURM job scheduler and CESGA Finisterrae III cluster.
Uses multiprocessing for distributed computing within a node.

Usage:
    Direct: python walkforward_permutation.py <strategy> [options]
    SLURM:  sbatch run_walkforward.sbatch <strategy>

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
CESGA_PROJECT_ROOT = Path(os.getenv('CESGA_PROJECT_ROOT', '/mnt/lustre/scratch/nlsas/home/usc/ec/ahe/Trading-system'))
CESGA_DATA_DIR = Path(os.getenv('CESGA_DATA_DIR', CESGA_PROJECT_ROOT / 'data'))
CESGA_OUTPUT_DIR = Path(os.getenv('CESGA_OUTPUT_DIR', CESGA_PROJECT_ROOT / 'outputs'))

# Add project root to path
sys.path.insert(0, str(CESGA_PROJECT_ROOT))

# Import project modules
from config.paths import (
    BITCOIN_PARQUET,
    BACKTEST_FIGURES,
    get_plot_path, ensure_directories,
    TICKERS, get_ticker_data_paths
)

from backtest.mcpt.bar_permute import get_permutation


# Global variables for worker initialization
_strategy_module = None
_df = None
_train_window = None
_real_wf_pf = None


def _init_worker(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf):
    """Initialize worker with strategy module and data"""
    global _strategy_module, _df, _train_window, _real_wf_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    _df = pd.DataFrame(df_data, index=df_index, columns=df_columns)
    _train_window = train_window
    _real_wf_pf = real_wf_pf


def walkforward_strategy(ohlc: pd.DataFrame, strategy, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
    """
    Implements walk-forward optimization

    Args:
        ohlc: DataFrame with OHLC data
        strategy: Strategy module with signal and optimize functions
        train_lookback: Training window in bars
        train_step: Step between re-optimizations in bars

    Returns:
        Array with walk-forward signals
    """
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            result = strategy.optimize(ohlc.iloc[i-train_lookback:i])
            best_params = result[:-1]  # Last value is always PF
            tmp_signal = strategy.signal(ohlc, *best_params)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


def process_walkforward_permutation_generic(perm_i):
    """Process a single walk-forward permutation - optimized version"""
    strategy = _strategy_module
    df_perm = _df.copy()
    train_window = _train_window
    real_wf_pf = _real_wf_pf

    # Execute permutation with seed for reproducibility
    wf_perm = get_permutation(df_perm, start_index=train_window, seed=perm_i)

    # Calculate returns and signals
    wf_perm['r'] = np.log(wf_perm['close']).diff().shift(-1)
    wf_perm_sig = walkforward_strategy(wf_perm, strategy, train_lookback=train_window)
    perm_rets = wf_perm['r'] * wf_perm_sig

    # Calculate profit factor
    pos = perm_rets[perm_rets > 0].sum()
    neg = perm_rets[perm_rets < 0].abs().sum()
    if neg == 0:
        perm_pf = np.inf if pos > 0 else 0.0
    else:
        perm_pf = pos / neg

    # Calculate cumulative returns
    cum_rets = perm_rets.cumsum().values

    is_better = 1 if perm_pf >= real_wf_pf else 0
    return perm_pf, is_better, cum_rets


def get_slurm_config():
    """Get configuration from SLURM environment variables"""
    config = {
        'job_id': os.getenv('SLURM_JOB_ID', 'local'),
        'job_name': os.getenv('SLURM_JOB_NAME', 'mcpt_walkforward'),
        'ntasks': int(os.getenv('SLURM_NTASKS', '1')),
        'cpus_per_task': int(os.getenv('SLURM_CPUS_PER_TASK', str(cpu_count()))),
        'node_name': os.getenv('SLURM_NODELIST', 'localhost'),
        'scratch': Path(os.getenv('SCRATCH', '/tmp')),
        'store': Path(os.getenv('STORE', CESGA_OUTPUT_DIR)),
    }
    config['n_workers'] = config['cpus_per_task']
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CESGA Cluster - Walk-Forward MCPT Analysis'
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
        help=f'Ticker symbol (default: BTCUSD)'
    )
    parser.add_argument(
        '--data-start',
        type=str,
        default='2018-01-01',
        help='Data start date YYYY-MM-DD (default: 2018-01-01)'
    )
    parser.add_argument(
        '--data-end',
        type=str,
        default='2024-01-01',
        help='Data end date YYYY-MM-DD (default: 2024-01-01)'
    )
    parser.add_argument(
        '--train-window-years',
        type=float,
        default=4.0,
        help='Training window in years (default: 4.0)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=200,
        help='Number of permutations (default: 200)'
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
    print("CESGA CLUSTER - WALK-FORWARD MCPT ANALYSIS")
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

    if not hasattr(strategy, 'signal') or not hasattr(strategy, 'optimize'):
        print(f"ERROR: Module '{strategy_name}' must have 'signal' and 'optimize' functions")
        sys.exit(1)

    # Ensure output directories
    ensure_directories()

    # Worker configuration
    if args.n_workers:
        n_workers = args.n_workers
    else:
        n_workers = slurm_config['n_workers']

    n_permutations = args.n_permutations

    print(f"\nConfiguration:")
    print(f"  Ticker:           {args.ticker}")
    print(f"  Data period:      {args.data_start} to {args.data_end}")
    print(f"  Train window:     {args.train_window_years} years")
    print(f"  Workers:          {n_workers}")
    print(f"  Permutations:     {n_permutations}")
    print(f"  Scratch dir:      {slurm_config['scratch']}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Load data
    print(f"Loading data for {args.ticker}...")
    try:
        ticker_paths = get_ticker_data_paths(args.ticker)
        parquet_path = ticker_paths['parquet']
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            print(f"ERROR: Data file not found: {parquet_path}")
            sys.exit(1)
    except ValueError:
        # Fallback to default Bitcoin data
        df = pd.read_parquet(BITCOIN_PARQUET)

    df.index = df.index.astype('datetime64[s]')

    # Filter by date range
    data_start = pd.Timestamp(args.data_start)
    data_end = pd.Timestamp(args.data_end)
    df = df[(df.index >= data_start) & (df.index < data_end)]

    print(f"Data loaded: {len(df):,} rows ({args.data_start} to {args.data_end})\n")

    # Walk-forward configuration
    train_window = int(24 * 365 * args.train_window_years)  # Hours in years

    print("="*70)
    print("WALK-FORWARD ANALYSIS")
    print("="*70)
    print(f"  Total data:     {len(df):,} periods ({len(df)/24/365:.1f} years)")
    print(f"  Train window:   {train_window:,} periods ({train_window/24/365:.1f} years)")

    # Calculate real strategy
    df['r'] = np.log(df['close']).diff().shift(-1)
    df['wf_signal'] = walkforward_strategy(df, strategy, train_lookback=train_window)
    wf_rets = df['wf_signal'] * df['r']
    real_wf_pf = wf_rets[wf_rets > 0].sum() / wf_rets[wf_rets < 0].abs().sum()
    real_cum_rets = wf_rets.cumsum()

    print(f"  Real Profit Factor: {real_wf_pf:.4f}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # MCPT with multiprocessing
    print(f"Running Walk-Forward MCPT with {n_permutations} permutations using {n_workers} workers...")
    print()
    sys.stdout.flush()

    # Prepare data for worker initialization
    df_data = df.values
    df_index = df.index
    df_columns = df.columns.tolist()
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
              initargs=(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf)) as pool:
        for result in tqdm(pool.imap_unordered(process_walkforward_permutation_generic, args_list, chunksize=chunksize),
                          total=n_permutations,
                          desc="Processing permutations",
                          ncols=80):
            results.append(result)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.0f}s")

    # Analyze results
    permuted_pfs = [pf for pf, _, _ in results]
    perm_better_count = 1 + sum(is_better for _, is_better, _ in results)
    perm_cum_rets = [cum_rets for _, _, cum_rets in results]
    walkforward_mcpt_pval = perm_better_count / n_permutations

    print("="*70)
    print("MCPT RESULTS")
    print("="*70)
    print(f"  Permutations:     {len(results)}")
    print(f"  Better than real: {perm_better_count}")
    print(f"  P-Value:          {walkforward_mcpt_pval:.4f}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Speed:            {len(results)/total_time:.1f} tasks/s")

    if walkforward_mcpt_pval < 0.05:
        print(f"  SIGNIFICANT (p < 0.05)")
    else:
        print(f"  NOT significant (p >= 0.05)")

    print("="*70 + "\n")
    sys.stdout.flush()

    # Generate plots
    print("Generating plots...")
    plt.style.use('dark_background')

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / strategy_name
    else:
        output_dir = BACKTEST_FIGURES / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Histogram plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(permuted_pfs, bins=50, color='steelblue', alpha=0.7, edgecolor='white', label='Permutations')
    ax.axvline(real_wf_pf, color='red', linestyle='--', linewidth=2.5, label=f'Real PF: {real_wf_pf:.4f}')
    mean_perm = np.mean(permuted_pfs)
    ax.axvline(mean_perm, color='yellow', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Perm: {mean_perm:.4f}')
    ax.set_xlabel("Profit Factor", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Walk-Forward MCPT ({strategy_name}) | P-Value: {walkforward_mcpt_pval:.4f} | "
                 f"{'Significant' if walkforward_mcpt_pval < 0.05 else 'Not Significant'}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    output_file = output_dir / 'walkforward_mcpt.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Plot saved: {output_file}")
    plt.close()

    # Cumulative returns plot
    fig, ax = plt.subplots(figsize=(14, 8))
    perm_matrix = np.array(perm_cum_rets).T
    ax.plot(df.index, perm_matrix, color='white', alpha=0.05, linewidth=0.5)
    ax.plot(df.index, real_cum_rets, color='red', linewidth=2.5,
            label=f'Real Strategy (PF={real_wf_pf:.4f})', zorder=100)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"Walk-Forward Cumulative Returns ({strategy_name}) | Real vs {len(perm_cum_rets)} Permutations",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()

    output_file_cum = output_dir / 'walkforward_cumulative_mcpt.png'
    plt.savefig(output_file_cum, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Cumulative returns plot saved: {output_file_cum}")
    plt.close()

    # Save results to CSV for later analysis
    results_df = pd.DataFrame({
        'permutation': range(len(permuted_pfs)),
        'profit_factor': permuted_pfs,
        'better_than_real': [1 if pf >= real_wf_pf else 0 for pf in permuted_pfs]
    })
    results_csv = output_dir / f'walkforward_results_{slurm_config["job_id"]}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved: {results_csv}")

    print("\n" + "="*70)
    print("WALK-FORWARD ANALYSIS COMPLETED")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - walkforward_mcpt.png")
    print(f"  - walkforward_cumulative_mcpt.png")
    print(f"  - walkforward_results_{slurm_config['job_id']}.csv")
    print("="*70 + "\n")
