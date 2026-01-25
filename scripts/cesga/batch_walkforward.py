#!/usr/bin/env python3
"""
Batch Walk-Forward Backtest para CESGA

Versión optimizada para cluster HPC.
Ejecuta walkforward_permutation.py para múltiples tickers.

Usage (en nodo de cómputo):
    python batch_walkforward.py --group crypto_10 --strategy hawkes --n-permutations 1000
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import os

# Detectar directorio del proyecto
if 'TRADING_ROOT' in os.environ:
    project_root = Path(os.environ['TRADING_ROOT'])
else:
    project_root = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(project_root))

from config.tickers import get_ticker_group, TICKER_GROUPS


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch walk-forward backtest para CESGA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--group', type=str, choices=list(TICKER_GROUPS.keys()))
    group.add_argument('--tickers', type=str, nargs='+')

    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--start-train', type=str)
    parser.add_argument('--end-train', type=str)
    parser.add_argument('--start-walk', type=str)
    parser.add_argument('--end-walk', type=str)
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-workers', type=int, default=16)

    args = parser.parse_args()

    # Obtener tickers
    tickers = get_ticker_group(args.group) if args.group else [t.upper() for t in args.tickers]

    print(f"\n{'='*70}")
    print(f"CESGA BATCH WALK-FORWARD")
    print(f"{'='*70}")
    print(f"Tickers: {len(tickers)}")
    print(f"Estrategia: {args.strategy}")
    print(f"Permutaciones: {args.n_permutations}")
    print(f"Workers: {args.n_workers}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    successful, failed = [], []

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] {ticker}")

        cmd = [
            sys.executable,
            str(project_root / 'scripts' / 'laptop' / 'walkforward_permutation.py'),
            '--ticker', ticker,
            '--strategy', args.strategy,
            '--n-permutations', str(args.n_permutations),
            '--n-workers', str(args.n_workers)
        ]

        if args.start_train and args.end_train and args.start_walk and args.end_walk:
            cmd.extend(['--start-train', args.start_train])
            cmd.extend(['--end-train', args.end_train])
            cmd.extend(['--start-walk', args.start_walk])
            cmd.extend(['--end-walk', args.end_walk])

        try:
            subprocess.run(cmd, check=True)
            successful.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"  ERROR: {e}")

    duration = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETADO en {duration}")
    print(f"Exitosos: {len(successful)}, Fallidos: {len(failed)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
