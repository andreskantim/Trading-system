#!/usr/bin/env python3
"""
Batch In-Sample Backtest

Ejecuta insample_permutation.py para múltiples tickers secuencialmente.
Después de completar todos los tickers, genera estadísticas agregadas y visualizaciones.

Usage:
    python batch_insample.py --group crypto_10 --strategy hawkes
    python batch_insample.py --tickers BTC ETH SOL --strategy donchian
    python batch_insample.py --group crypto_25 --strategy moving_average --start 01/01/2020 --end 31/12/2023
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
    """Load results from individual ticker run."""
    results_dir = get_ticker_output_dir(strategy, ticker, 'results')
    results_file = results_dir / f'{ticker}_insample_results.json'

    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def aggregate_results(results: list) -> dict:
    """
    Aggregate results from multiple ticker runs.

    Returns dict with mean, median, and per-ticker values.
    """
    if not results:
        return {}

    # Basic metrics aggregation
    p_values = [r['p_value'] for r in results if 'p_value' in r]
    pfs = [r['real_pf'] for r in results if 'real_pf' in r]

    aggregated = {
        'n_tickers': len(results),
        'n_significant': sum(1 for r in results if r.get('significant', False)),
        'p_value': {
            'mean': float(np.mean(p_values)) if p_values else None,
            'median': float(np.median(p_values)) if p_values else None,
            'std': float(np.std(p_values)) if len(p_values) > 1 else None,
        },
        'profit_factor': {
            'mean': float(np.mean(pfs)) if pfs else None,
            'median': float(np.median(pfs)) if pfs else None,
            'std': float(np.std(pfs)) if len(pfs) > 1 else None,
        },
        'tickers': {r['ticker']: {'p_value': r['p_value'], 'pf': r['real_pf'], 'significant': r['significant']}
                    for r in results if 'ticker' in r},
    }

    # Full stats aggregation
    stats_results = [r for r in results if 'stats' in r]
    if stats_results:
        aggregated['stats_aggregated'] = calculate_batch_stats(stats_results)

    return aggregated




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
        batch_name = f"{args.group}_insample"
        print(f"\n{'='*70}")
        print(f"BATCH IN-SAMPLE BACKTEST")
        print(f"{'='*70}")
        print(f"Grupo: {args.group} ({len(tickers)} tickers)")
    else:
        tickers = [t.upper() for t in args.tickers]
        batch_name = f"custom_{len(tickers)}_insample"
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

    # Aggregate results
    print(f"\n{'='*70}")
    print("AGREGANDO RESULTADOS")
    print(f"{'='*70}")

    all_results = []
    for ticker in successful:
        result = load_ticker_results(args.strategy, ticker)
        if result:
            all_results.append(result)
            print(f"  Cargado: {ticker}")

    if all_results:
        # Calculate aggregated statistics
        aggregated = aggregate_results(all_results)

        # Save batch results
        batch_dirs = ensure_batch_output_dirs(args.strategy, batch_name)

        batch_results = {
            'batch_name': batch_name,
            'strategy': args.strategy,
            'timestamp': datetime.now().isoformat(),
            'n_tickers': len(tickers),
            'successful': successful,
            'failed': failed,
            'aggregated': aggregated,
        }

        batch_file = batch_dirs['results'] / f'{batch_name}_results.json'
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"\n  Batch results: {batch_file}")

        # Generate plots
        print(f"\nGenerando visualizaciones...")
        plot_batch_results(batch_name, args.strategy, all_results, aggregated, batch_dirs['figures'])

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

    if all_results:
        print(f"\nEstadísticas agregadas:")
        print(f"  Significativos (p<0.05): {sum(1 for r in all_results if r.get('significant', False))}/{len(all_results)}")
        p_values = [r['p_value'] for r in all_results]
        pfs = [r['real_pf'] for r in all_results]
        print(f"  P-Value medio:    {np.mean(p_values):.4f}")
        print(f"  P-Value mediana:  {np.median(p_values):.4f}")
        print(f"  PF medio:         {np.mean(pfs):.4f}")
        print(f"  PF mediana:       {np.median(pfs):.4f}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
