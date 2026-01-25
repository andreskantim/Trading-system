#!/usr/bin/env python3
"""
Batch In-Sample Backtest para CESGA

Versión optimizada para cluster HPC.
Ejecuta insample_permutation.py para múltiples tickers.

Usage (en nodo de cómputo):
    python batch_insample.py --group crypto_10 --strategy hawkes --n-permutations 10000
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
        description='Batch in-sample backtest para CESGA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--group', type=str, choices=list(TICKER_GROUPS.keys()))
    group.add_argument('--tickers', type=str, nargs='+')

    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--n-permutations', type=int, default=10000)
    parser.add_argument('--n-workers', type=int, default=16)

    args = parser.parse_args()

    # Obtener tickers
    tickers = get_ticker_group(args.group) if args.group else [t.upper() for t in args.tickers]

    print(f"\n{'='*70}")
    print(f"CESGA BATCH IN-SAMPLE")
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
            str(project_root / 'scripts' / 'laptop' / 'insample_permutation.py'),
            '--ticker', ticker,
            '--strategy', args.strategy,
            '--n-permutations', str(args.n_permutations),
            '--n-workers', str(args.n_workers)
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

    duration = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETADO en {duration}")
    print(f"Exitosos: {len(successful)}, Fallidos: {len(failed)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
