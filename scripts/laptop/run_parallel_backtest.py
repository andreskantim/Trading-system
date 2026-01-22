#!/usr/bin/env python3
"""
Main execution script for parallel backtest runs.

Orchestrates parallel execution of backtests and MCPT analysis
on a 16-core local workstation.

Usage:
    python run_parallel_backtest.py --strategy donchian --symbols BTCUSD ETHUSD
    python run_parallel_backtest.py --test-type walkforward --backend multiprocess
    python run_parallel_backtest.py --config config/parallel_config.yaml

Examples:
    # Run in-sample MCPT with default settings
    python scripts/run_parallel_backtest.py --test-type insample

    # Run walk-forward with specific symbols
    python scripts/run_parallel_backtest.py --test-type walkforward --symbols BTCUSD

    # Run with Dask backend
    python scripts/run_parallel_backtest.py --backend dask --n-workers 12
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import subprocess
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATA_DIR,
    BACKTEST_RESULTS,
    BACKTEST_FIGURES,
    BACKTEST_REPORTS,
    CONFIG_DIR,
    SCRIPTS_DIR,
    ensure_directories
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parallel backtests on 16-core workstation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--backend',
        choices=['sequential', 'multiprocess', 'dask'],
        default='multiprocess',
        help='Execution backend (default: multiprocess)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTCUSD'],
        help='List of ticker symbols to process (default: BTCUSD)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='donchian',
        help='Strategy name to use (default: donchian)'
    )

    parser.add_argument(
        '--n-workers',
        type=int,
        default=14,
        help='Number of worker processes (default: 14, max: 16)'
    )

    parser.add_argument(
        '--test-type',
        choices=['insample', 'walkforward', 'both'],
        default='insample',
        help='Type of MCPT test to run (default: insample)'
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Number of permutations for MCPT (default: 1000)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (default: outputs/backtest/results)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )

    return parser.parse_args()


def load_config(config_path: Optional[Path]) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        default_config = CONFIG_DIR / 'parallel_config.yaml'
        if default_config.exists():
            config_path = default_config
        else:
            return {}

    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def run_insample_mcpt(
    strategy: str,
    n_workers: int,
    n_permutations: int,
    verbose: bool = False,
    dry_run: bool = False
) -> int:
    """
    Run in-sample MCPT analysis.

    Args:
        strategy: Strategy name
        n_workers: Number of worker processes
        n_permutations: Number of permutations
        verbose: Enable verbose output
        dry_run: Print command without executing

    Returns:
        Return code from subprocess
    """
    script_path = SCRIPTS_DIR / 'insample_permutation.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--strategy', strategy,
        '--n_permutations', str(n_permutations),
        '--n_workers', str(n_workers)
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return 0

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_walkforward_mcpt(
    strategy: str,
    n_workers: int,
    n_permutations: int,
    verbose: bool = False,
    dry_run: bool = False
) -> int:
    """
    Run walk-forward MCPT analysis.

    Args:
        strategy: Strategy name
        n_workers: Number of worker processes
        n_permutations: Number of permutations
        verbose: Enable verbose output
        dry_run: Print command without executing

    Returns:
        Return code from subprocess
    """
    script_path = SCRIPTS_DIR / 'walkforward_permutation.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--strategy', strategy,
        '--n_permutations', str(n_permutations),
        '--n_workers', str(n_workers)
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return 0

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_parallel_backtest(
    symbols: List[str],
    strategy: str,
    backend: str,
    n_workers: int,
    test_type: str,
    n_permutations: int,
    verbose: bool = False,
    dry_run: bool = False
) -> int:
    """
    Run parallel backtest using orchestration layer.

    Args:
        symbols: List of symbols to process
        strategy: Strategy name
        backend: Execution backend
        n_workers: Number of workers
        test_type: Type of test (insample/walkforward/both)
        n_permutations: Number of permutations
        verbose: Enable verbose output
        dry_run: Print without executing

    Returns:
        0 on success, non-zero on failure
    """
    from orchestration import get_orchestrator

    # Limit workers to max 16
    n_workers = min(n_workers, 16)

    if verbose:
        print(f"Backend: {backend}")
        print(f"Workers: {n_workers}")
        print(f"Strategy: {strategy}")
        print(f"Symbols: {symbols}")
        print(f"Test type: {test_type}")

    if dry_run:
        print("[DRY RUN] Would run parallel backtest with above settings")
        return 0

    # Run the appropriate test type
    return_code = 0

    if test_type in ['insample', 'both']:
        if verbose:
            print("\n" + "=" * 50)
            print("Running In-Sample MCPT")
            print("=" * 50)

        rc = run_insample_mcpt(
            strategy=strategy,
            n_workers=n_workers,
            n_permutations=n_permutations,
            verbose=verbose,
            dry_run=dry_run
        )
        if rc != 0:
            return_code = rc

    if test_type in ['walkforward', 'both']:
        if verbose:
            print("\n" + "=" * 50)
            print("Running Walk-Forward MCPT")
            print("=" * 50)

        rc = run_walkforward_mcpt(
            strategy=strategy,
            n_workers=n_workers,
            n_permutations=n_permutations,
            verbose=verbose,
            dry_run=dry_run
        )
        if rc != 0:
            return_code = rc

    return return_code


def main():
    """Main entry point."""
    args = parse_args()

    # Ensure output directories exist
    ensure_directories()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    n_workers = args.n_workers
    if 'default' in config and 'n_workers' in config['default']:
        n_workers = config['default']['n_workers']
    n_workers = min(args.n_workers, 16)  # Cap at 16 cores

    if args.verbose:
        print("=" * 60)
        print("Trading-System Parallel Backtest Runner")
        print("=" * 60)
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Results directory: {BACKTEST_RESULTS}")
        print()

    # Run backtest
    return_code = run_parallel_backtest(
        symbols=args.symbols,
        strategy=args.strategy,
        backend=args.backend,
        n_workers=n_workers,
        test_type=args.test_type,
        n_permutations=args.n_permutations,
        verbose=args.verbose,
        dry_run=args.dry_run
    )

    if args.verbose:
        print()
        if return_code == 0:
            print("Backtest completed successfully!")
            print(f"Results saved to: {BACKTEST_RESULTS}")
        else:
            print(f"Backtest failed with return code: {return_code}")

    sys.exit(return_code)


if __name__ == '__main__':
    main()
