#!/usr/bin/env python3
"""
Batch In-Sample Backtest

Ejecuta insample_permutation.py para múltiples tickers secuencialmente.

Usage:
    python batch_insample.py --group crypto_10 --strategy hawkes
    python batch_insample.py --tickers BTC ETH SOL --strategy donchian
    python batch_insample.py --group crypto_25 --strategy moving_average --start 01/01/2020 --end 31/12/2023
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.tickers import get_ticker_group, TICKER_GROUPS


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch in-sample backtest para múltiples tickers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Grupo o lista de tickers
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--group',
        type=str,
        choices=list(TICKER_GROUPS.keys()),
        help='Grupo de tickers predefinido (crypto_10, crypto_25, crypto_all)'
    )
    group.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Lista de tickers específicos (ej: BTC ETH SOL)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Estrategia a testear (ej: donchian, hawkes)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Fecha inicio DD/MM/YYYY (opcional)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='Fecha fin DD/MM/YYYY (opcional)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Número de permutaciones (default: 1000)'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Número de workers por ticker (default: auto)'
    )

    args = parser.parse_args()

    # Obtener lista de tickers
    if args.group:
        tickers = get_ticker_group(args.group)
        print(f"\n{'='*70}")
        print(f"BATCH IN-SAMPLE BACKTEST")
        print(f"{'='*70}")
        print(f"Grupo: {args.group} ({len(tickers)} tickers)")
    else:
        tickers = [t.upper() for t in args.tickers]
        print(f"\n{'='*70}")
        print(f"BATCH IN-SAMPLE BACKTEST")
        print(f"{'='*70}")
        print(f"Tickers: {', '.join(tickers)}")

    print(f"Estrategia: {args.strategy}")
    print(f"Permutaciones: {args.n_permutations}")
    if args.start:
        print(f"Período: {args.start} - {args.end if args.end else 'presente'}")
    print(f"{'='*70}\n")

    # Ejecutar para cada ticker
    start_time = datetime.now()
    successful = []
    failed = []

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(tickers)}] Procesando {ticker}")
        print(f"{'='*70}")

        cmd = [
            sys.executable,
            str(project_root / 'scripts' / 'laptop' / 'insample_permutation.py'),
            '--ticker', ticker,
            '--strategy', args.strategy,
            '--n-permutations', str(args.n_permutations)
        ]

        if args.start:
            cmd.extend(['--start', args.start])
        if args.end:
            cmd.extend(['--end', args.end])
        if args.n_workers:
            cmd.extend(['--n-workers', str(args.n_workers)])

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            successful.append(ticker)
            print(f"\n[{ticker}] Completado")
        except subprocess.CalledProcessError as e:
            failed.append(ticker)
            print(f"\n[{ticker}] ERROR: {e}")
            continue
        except FileNotFoundError as e:
            failed.append(ticker)
            print(f"\n[{ticker}] ERROR: No hay datos - {e}")
            continue

    # Resumen final
    duration = datetime.now() - start_time

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETADO")
    print(f"{'='*70}")
    print(f"Tiempo total: {duration}")
    print(f"Exitosos: {len(successful)}/{len(tickers)}")
    if successful:
        print(f"  {', '.join(successful)}")
    if failed:
        print(f"Fallidos: {len(failed)}/{len(tickers)}")
        print(f"  {', '.join(failed)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
