#!/usr/bin/env python3
"""
Walk-Forward MCPT Analysis

Backtest walk-forward con test de permutaciones Monte Carlo (MCPT).
Optimiza en período de entrenamiento y evalúa en período de walk-forward.

Si no se especifican fechas, divide los datos 50/50 automáticamente.

Usage:
    python walkforward_permutation.py --ticker BTC --strategy donchian
    python walkforward_permutation.py --ticker ETH --strategy hawkes --n-permutations 500
    python walkforward_permutation.py --ticker SOL --strategy moving_average \\
        --start-train 01/01/2018 --end-train 31/12/2020 \\
        --start-walk 01/01/2021 --end-walk 31/12/2023
"""

import pandas as pd
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
_df = None
_train_window = None
_real_wf_pf = None


def _init_worker(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf):
    """Inicializa el worker con el módulo de estrategia y datos"""
    global _strategy_module, _df, _train_window, _real_wf_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    _df = pd.DataFrame(df_data, index=df_index, columns=df_columns)
    _train_window = train_window
    _real_wf_pf = real_wf_pf


def walkforward_strategy(ohlc: pd.DataFrame, strategy, train_lookback: int, train_step: int = None):
    """
    Implementa walk-forward optimization genérica

    Args:
        ohlc: DataFrame con datos OHLC
        strategy: Módulo de estrategia
        train_lookback: Ventana de entrenamiento en barras
        train_step: Paso entre re-optimizaciones (default: train_lookback // 12)

    Returns:
        Array con señales walk-forward
    """
    if train_step is None:
        train_step = max(train_lookback // 12, 24 * 30)  # Mínimo 30 días

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            result = strategy.optimize(ohlc.iloc[i - train_lookback:i])
            best_params = result[:-1]  # Último valor es PF
            tmp_signal = strategy.signal(ohlc, *best_params)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


def process_walkforward_permutation(perm_i):
    """Procesa una permutación individual de walk-forward"""
    strategy = _strategy_module
    df_perm = _df.copy()
    train_window = _train_window
    real_wf_pf = _real_wf_pf

    # Permutar desde el inicio del período de walk-forward
    wf_perm = get_permutation(df_perm, start_index=train_window, seed=perm_i)

    # Calcular returns y señales
    wf_perm['r'] = np.log(wf_perm['close']).diff().shift(-1)
    wf_perm_sig = walkforward_strategy(wf_perm, strategy, train_lookback=train_window)
    perm_rets = wf_perm['r'] * wf_perm_sig

    # Calcular profit factor
    pos = perm_rets[perm_rets > 0].sum()
    neg = perm_rets[perm_rets < 0].abs().sum()
    if neg == 0:
        perm_pf = np.inf if pos > 0 else 0.0
    else:
        perm_pf = pos / neg

    # Cumulative returns
    cum_rets = perm_rets.cumsum().values

    is_better = 1 if perm_pf >= real_wf_pf else 0
    return perm_pf, is_better, cum_rets


def save_results(results_dict: dict, ticker: str, strategy_name: str, output_dir: Path):
    """Guarda resultados en JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f'{ticker}_walkforward_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"  Resultados guardados: {results_file}")
    return results_file


def print_summary(results_dict: dict):
    """Imprime resumen de resultados"""
    print("\n" + "=" * 70)
    print("RESUMEN MCPT WALK-FORWARD")
    print("=" * 70)
    print(f"  Ticker:            {results_dict['ticker']}")
    print(f"  Estrategia:        {results_dict['strategy']}")
    print(f"  Período total:     {results_dict['period']['start']} - {results_dict['period']['end']}")
    print(f"  Velas totales:     {results_dict['n_candles']:,}")
    print(f"  Train window:      {results_dict['train_window']:,} velas")
    print(f"  Permutaciones:     {results_dict['n_permutations']}")
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
        print("ESTADÍSTICAS CLAVE (Walk-Forward)")
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
        description='MCPT Walk-Forward Analysis',
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
        '--start-train',
        type=str,
        help='Fecha inicio entrenamiento DD/MM/YYYY'
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
        default=1000,
        help='Número de permutaciones (default: 1000, walk-forward es lento)'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Número de workers (default: auto)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("\n" + "=" * 70)
    print("MCPT WALK-FORWARD ANALYSIS")
    print("=" * 70 + "\n")

    strategy_name = args.strategy.replace('.py', '')

    # Importar estrategia
    try:
        strategy = importlib.import_module(f'models.strategies.{strategy_name}')
        print(f"Estrategia cargada: {strategy_name}")
    except ModuleNotFoundError:
        print(f"ERROR: No se encontró el módulo '{strategy_name}'")
        sys.exit(1)

    if not hasattr(strategy, 'signal') or not hasattr(strategy, 'optimize'):
        print(f"ERROR: '{strategy_name}' debe tener funciones 'signal' y 'optimize'")
        sys.exit(1)

    ensure_directories()

    # Configuración de workers
    total_cpus = cpu_count()
    n_workers = args.n_workers if args.n_workers else min(15, total_cpus)
    n_permutations = args.n_permutations

    # Cargar datos
    print(f"\nCargando datos de {args.ticker}...")

    start_available, end_available = get_available_date_range(args.ticker)
    if start_available:
        print(f"  Rango disponible: {start_available} - {end_available}")

    # Determinar fechas - auto-split 50/50 si no se especifican
    if args.start_train and args.end_train and args.start_walk and args.end_walk:
        # Usar fechas proporcionadas
        df_full = load_ticker_data(args.ticker, start_date=args.start_train, end_date=args.end_walk)
        train_start = pd.Timestamp(args.start_train, dayfirst=True)
        train_end = pd.Timestamp(args.end_train, dayfirst=True)
        walk_start = pd.Timestamp(args.start_walk, dayfirst=True)
        walk_end = pd.Timestamp(args.end_walk, dayfirst=True)
        print(f"  Usando fechas proporcionadas")
    else:
        # Auto-split 50/50
        try:
            df_full = load_ticker_data(args.ticker)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        mid_idx = len(df_full) // 2
        mid_date = df_full.index[mid_idx]

        train_start = df_full.index.min()
        train_end = mid_date
        walk_start = df_full.index[mid_idx + 1]
        walk_end = df_full.index.max()

        print(f"  Auto-split 50/50:")
        print(f"    Train: {train_start.strftime('%d/%m/%Y')} - {train_end.strftime('%d/%m/%Y')}")
        print(f"    Walk:  {walk_start.strftime('%d/%m/%Y')} - {walk_end.strftime('%d/%m/%Y')}")

    print(f"  Datos cargados: {len(df_full):,} velas")
    print(f"  Desde: {df_full.index.min()}")
    print(f"  Hasta: {df_full.index.max()}")

    # Calcular train_window (número de velas en período de train)
    df_train = df_full[df_full.index <= train_end]
    train_window = len(df_train)

    print(f"\nConfiguración:")
    print(f"  Workers:       {n_workers}")
    print(f"  Permutaciones: {n_permutations}")
    print(f"  Train window:  {train_window:,} velas ({train_window/24/365:.1f} años)")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    # Walk-forward analysis real
    print("Ejecutando walk-forward real...")

    df_full['r'] = np.log(df_full['close']).diff().shift(-1)
    df_full['wf_signal'] = walkforward_strategy(df_full, strategy, train_lookback=train_window)
    wf_rets = df_full['wf_signal'] * df_full['r']

    # Profit factor
    pos = wf_rets[wf_rets > 0].sum()
    neg = wf_rets[wf_rets < 0].abs().sum()
    real_wf_pf = pos / neg if neg != 0 else (np.inf if pos > 0 else 0.0)
    real_cum_rets = wf_rets.cumsum()

    print(f"  Real PF: {real_wf_pf:.4f}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    # MCPT
    print(f"Ejecutando Walk-Forward MCPT con {n_permutations} permutaciones...")

    df_data = df_full.values
    df_index = df_full.index
    df_columns = df_full.columns.tolist()
    args_list = list(range(n_permutations))

    start_time = time.time()
    results = []
    chunksize = max(1, n_permutations // (n_workers * 10))

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf)) as pool:
        for result in tqdm(pool.imap_unordered(process_walkforward_permutation, args_list, chunksize=chunksize),
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
        returns=wf_rets.values,
        signal=df_full['wf_signal'].values,
        p_value_mcpt=p_value,
        periods_per_year=8760  # Hourly data
    )

    # Preparar resultados
    results_dict = {
        'ticker': args.ticker,
        'strategy': strategy_name,
        'period': {
            'start': str(df_full.index.min()),
            'end': str(df_full.index.max())
        },
        'train_period': {
            'start': str(train_start),
            'end': str(train_end)
        },
        'walk_period': {
            'start': str(walk_start),
            'end': str(walk_end)
        },
        'n_candles': len(df_full),
        'train_window': train_window,
        'n_permutations': n_permutations,
        'real_pf': float(real_wf_pf),
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
        index=df_full.index,
        real_cum_rets=real_cum_rets.values,
        perm_cum_rets=perm_cum_rets,
        perm_pfs=permuted_pfs,
        real_pf=real_wf_pf,
        p_value=p_value,
        output_dir=output_dirs['figures'],
        prefix='walkforward',
        vlines=[(train_end, 'cyan', 'Train/Walk split')]
    )

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Velocidad:    {len(results)/total_time:.2f} tareas/s")
