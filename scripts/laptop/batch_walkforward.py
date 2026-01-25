#!/usr/bin/env python3
"""
Batch Walk-Forward Backtest

Ejecuta walkforward_permutation.py para múltiples tickers secuencialmente.

Usage:
    python batch_walkforward.py --group crypto_10 --strategy hawkes
    python batch_walkforward.py --tickers BTC ETH SOL --strategy donchian
    python batch_walkforward.py --group crypto_25 --strategy moving_average --n-permutations 500
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
        description='Batch walk-forward backtest para múltiples tickers',
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
        '--start-train',
        type=str,
        help='Fecha inicio entrenamiento DD/MM/YYYY (opcional, auto-split si no se da)'
    )
    parser.add_argument(
        '--end-train',
        type=str,
        help='Fecha fin entrenamiento DD/MM/YYYY'
    )
    parser.add_argument(
        '--start-walk',
        type=str,
        help='Fecha inicio walk-forward DD/MM/YYYY'
    )
    parser.add_argument(
        '--end-walk',
        type=str,
        help='Fecha fin walk-forward DD/MM/YYYY'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=200,
        help='Número de permutaciones (default: 200, walk-forward es lento)'
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
        print(f"BATCH WALK-FORWARD BACKTEST")
        print(f"{'='*70}")
        print(f"Grupo: {args.group} ({len(tickers)} tickers)")
    else:
        tickers = [t.upper() for t in args.tickers]
        print(f"\n{'='*70}")
        print(f"BATCH WALK-FORWARD BACKTEST")
        print(f"{'='*70}")
        print(f"Tickers: {', '.join(tickers)}")

    print(f"Estrategia: {args.strategy}")
    print(f"Permutaciones: {args.n_permutations}")
    if args.start_train:
        print(f"Train: {args.start_train} - {args.end_train}")
        print(f"Walk:  {args.start_walk} - {args.end_walk}")
    else:
        print(f"Fechas: Auto-split 50/50")
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
            str(project_root / 'scripts' / 'laptop' / 'walkforward_permutation.py'),
            '--ticker', ticker,
            '--strategy', args.strategy,
            '--n-permutations', str(args.n_permutations)
        ]

        if args.start_train and args.end_train and args.start_walk and args.end_walk:
            cmd.extend(['--start-train', args.start_train])
            cmd.extend(['--end-train', args.end_train])
            cmd.extend(['--start-walk', args.start_walk])
            cmd.extend(['--end-walk', args.end_walk])

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
