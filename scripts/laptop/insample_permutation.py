#!/usr/bin/env python3
"""
In-Sample MCPT Analysis

Backtest in-sample con test de permutaciones Monte Carlo (MCPT).
Carga datos desde data/operative/ con filtrado por fechas.

Usage:
    python insample_permutation.py --ticker BTC --strategy donchian
    python insample_permutation.py --ticker ETH --strategy hawkes --start 01/01/2018 --end 31/12/2022
    python insample_permutation.py --ticker SOL --strategy moving_average --n-permutations 5000 --n-workers 8
"""

import numpy as np
import sys
import json
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import importlib
import argparse

# Agregar directorio raíz al path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import ensure_directories, ensure_ticker_output_dirs
from backtest.mcpt.bar_permute import get_permutation
from utils.data_loader import load_ticker_data, get_available_date_range
from utils.stats_calculator import calculate_all_stats
from visualization.non_interactive.stats_and_plots_ticker import plot_ticker_results


# Variables globales para el worker
_strategy_module = None
_train_df = None
_best_real_pf = None


def _init_worker(strategy_name, train_data, train_index, train_columns, best_real_pf):
    """Inicializa el worker con el módulo de estrategia y datos"""
    global _strategy_module, _train_df, _best_real_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    _train_df = pd.DataFrame(train_data, index=train_index, columns=train_columns)
    _best_real_pf = best_real_pf


def process_permutation(perm_i):
    """Procesa una permutación individual"""
    strategy = _strategy_module
    train_df = _train_df
    best_real_pf = _best_real_pf

    # Ejecutar permutación
    train_perm = get_permutation(train_df, seed=perm_i)
    result = strategy.optimize(train_perm)

    # Desempaquetar: último valor es pf, resto son parámetros
    *best_params, best_perm_pf = result

    # Calcular cumulative returns
    sig = strategy.signal(train_perm, *best_params)
    r = np.log(train_perm['close']).diff().shift(-1)
    perm_rets = sig * r
    cum_rets = perm_rets.cumsum().values

    is_better = 1 if best_perm_pf >= best_real_pf else 0
    return best_perm_pf, is_better, cum_rets


def save_results(results_dict: dict, ticker: str, strategy_name: str, output_dir: Path):
    """Guarda resultados en JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f'{ticker}_insample_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"  Resultados guardados: {results_file}")
    return results_file


def print_summary(results_dict: dict):
    """Imprime resumen de resultados"""
    print("\n" + "=" * 70)
    print("RESUMEN MCPT IN-SAMPLE")
    print("=" * 70)
    print(f"  Ticker:            {results_dict['ticker']}")
    print(f"  Estrategia:        {results_dict['strategy']}")
    print(f"  Período:           {results_dict['period']['start']} - {results_dict['period']['end']}")
    print(f"  Velas:             {results_dict['n_candles']:,}")
    print(f"  Permutaciones:     {results_dict['n_permutations']}")
    print(f"  Best Parameters:   {results_dict['best_params']}")
    print(f"  Real PF:           {results_dict['real_pf']:.4f}")
    print(f"  P-Value:           {results_dict['p_value']:.4f}")

    if results_dict['p_value'] < 0.05:
        print(f"  Significativo (p < 0.05)")
    else:
        print(f"  NO significativo (p >= 0.05)")

    # Print key statistics if available
    if 'stats' in results_dict:
        stats = results_dict['stats']
        print("\n" + "-" * 70)
        print("ESTADÍSTICAS CLAVE")
        print("-" * 70)

        if 'performance' in stats:
            perf = stats['performance']
            print(f"  Total Return:      {perf.get('total_return_pct', 0):.2f}%")
            print(f"  CAGR:              {perf.get('cagr_pct', 0):.2f}%")

        if 'risk' in stats:
            risk = stats['risk']
            print(f"  Max Drawdown:      {risk.get('max_drawdown_pct', 0):.2f}%")
            print(f"  Max DD Duration:   {risk.get('max_drawdown_duration', 0):,} bars")

        if 'risk_return' in stats:
            rr = stats['risk_return']
            print(f"  Sharpe Ratio:      {rr.get('sharpe_ratio', 0):.4f}")
            print(f"  Sortino Ratio:     {rr.get('sortino_ratio', 0):.4f}")
            print(f"  SQN:               {rr.get('sqn', 0):.4f}")

        if 'trades' in stats:
            trades = stats['trades']
            print(f"  N Trades:          {trades.get('n_trades', 0)}")
            print(f"  Win Rate:          {trades.get('win_rate_pct', 0):.2f}%")

    print("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MCPT In-Sample Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Ticker a analizar (ej: BTC, ETH, SOL)'
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
        help='Número de workers (default: auto)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    import pandas as pd

    args = parse_args()

    print("\n" + "=" * 70)
    print("MCPT IN-SAMPLE ANALYSIS")
    print("=" * 70 + "\n")

    strategy_name = args.strategy.replace('.py', '')

    # Importar estrategia
    try:
        strategy = importlib.import_module(f'models.strategies.{strategy_name}')
        print(f"Estrategia cargada: {strategy_name}")
    except ModuleNotFoundError:
        print(f"ERROR: No se encontró el módulo '{strategy_name}'")
        sys.exit(1)

    if not hasattr(strategy, 'optimize') or not hasattr(strategy, 'signal'):
        print(f"ERROR: '{strategy_name}' debe tener funciones 'optimize' y 'signal'")
        sys.exit(1)

    ensure_directories()

    # Configuración de workers
    total_cpus = cpu_count()
    n_workers = args.n_workers if args.n_workers else min(15, total_cpus)
    n_permutations = args.n_permutations

    # Cargar datos
    print(f"\nCargando datos de {args.ticker}...")

    if not args.start and not args.end:
        start_available, end_available = get_available_date_range(args.ticker)
        if start_available:
            print(f"  Rango disponible: {start_available} - {end_available}")

    try:
        train_df = load_ticker_data(args.ticker, start_date=args.start, end_date=args.end)
        print(f"  Datos cargados: {len(train_df):,} velas")
        print(f"  Desde: {train_df.index.min()}")
        print(f"  Hasta: {train_df.index.max()}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nConfiguración:")
    print(f"  Workers:       {n_workers}")
    print(f"  Permutaciones: {n_permutations}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    # Optimización in-sample
    print("Ejecutando optimización in-sample...")
    result = strategy.optimize(train_df)
    *best_params, best_real_pf = result

    # Calcular cumulative returns reales
    real_signal = strategy.signal(train_df, *best_params)
    real_r = np.log(train_df['close']).diff().shift(-1)
    real_rets = real_signal * real_r
    real_cum_rets = real_rets.cumsum()

    print(f"  Best Parameters: {best_params}")
    print(f"  Best PF:         {best_real_pf:.4f}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    # MCPT
    print(f"Ejecutando MCPT con {n_permutations} permutaciones...")

    train_data = train_df.values
    train_index = train_df.index
    train_columns = train_df.columns.tolist()
    args_list = list(range(n_permutations))

    start_time = time.time()
    results = []
    chunksize = max(1, n_permutations // (n_workers * 10))

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(strategy_name, train_data, train_index, train_columns, best_real_pf)) as pool:
        for result in tqdm(pool.imap_unordered(process_permutation, args_list, chunksize=chunksize),
                          total=len(args_list),
                          desc="Procesando",
                          ncols=80):
            results.append(result)

    total_time = time.time() - start_time

    # Análisis de resultados
    permuted_pfs = [pf for pf, _, _ in results]
    perm_better_count = 1 + sum(is_better for _, is_better, _ in results)
    perm_cum_rets = [cum_rets for _, _, cum_rets in results]
    p_value = perm_better_count / n_permutations

    # Calcular estadísticas completas
    print("\nCalculando estadísticas completas...")
    full_stats = calculate_all_stats(
        returns=real_rets.values,
        signal=real_signal.values,
        p_value_mcpt=p_value,
        periods_per_year=8760  # Hourly data
    )

    # Preparar resultados
    results_dict = {
        'ticker': args.ticker,
        'strategy': strategy_name,
        'period': {
            'start': str(train_df.index.min()),
            'end': str(train_df.index.max())
        },
        'n_candles': len(train_df),
        'n_permutations': n_permutations,
        'best_params': best_params,
        'real_pf': float(best_real_pf),
        'p_value': float(p_value),
        'mean_perm_pf': float(np.mean(permuted_pfs)),
        'std_perm_pf': float(np.std(permuted_pfs)),
        'perm_better_count': int(perm_better_count),
        'execution_time_seconds': float(total_time),
        'significant': p_value < 0.05,
        'stats': full_stats,
    }

    # Guardar resultados usando nueva estructura de directorios
    output_dirs = ensure_ticker_output_dirs(strategy_name, args.ticker)
    save_results(results_dict, args.ticker, strategy_name, output_dirs['results'])

    # Imprimir resumen
    print_summary(results_dict)

    # Generar gráficos
    print("\nGenerando gráficos...")
    plot_ticker_results(
        ticker=args.ticker,
        strategy=strategy_name,
        index=train_df.index,
        real_cum_rets=real_cum_rets.values,
        perm_cum_rets=perm_cum_rets,
        perm_pfs=permuted_pfs,
        real_pf=best_real_pf,
        p_value=p_value,
        output_dir=output_dirs['figures'],
        prefix='insample'
    )

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"  Tiempo total: {total_time:.1f}s")
    print(f"  Velocidad:    {len(results)/total_time:.1f} tareas/s")
