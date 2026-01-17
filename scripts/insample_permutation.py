"""
Versión GENÉRICA del análisis MCPT con soporte para Dask y visualización interactiva.
Permite elegir la estrategia y backend desde línea de comandos.

Uso: python insample_permutation.py <estrategia> [--backend dask|multiprocess]
Ejemplo: python insample_permutation.py donchian
         python insample_permutation.py hawkes --backend dask
         python insample_permutation.py moving_average --backend multiprocess
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import importlib
import argparse

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar configuración de rutas
from config.paths import (
    BACKTEST_FIGURES,
    ensure_directories,
    TICKERS, get_ticker_data_paths
)

from backtest.mcpt.bar_permute import get_permutation

# Import for interactive visualization (optional)
try:
    from visualization.interactive.lightweight_charts_viewer import create_interactive_chart
    HAS_INTERACTIVE_VIS = True
except ImportError:
    HAS_INTERACTIVE_VIS = False

# Import for Dask orchestration (optional)
try:
    from config.parallel_config import load_config as load_parallel_config
    from orchestration.dask_runner import DaskOrchestrator
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


# Variables globales para el worker (cargadas una sola vez por worker)
_strategy_module = None
_train_df = None
_best_real_pf = None

def _init_worker(strategy_name, train_data, train_index, train_columns, best_real_pf):
    """Inicializa el worker con el módulo de estrategia y datos"""
    global _strategy_module, _train_df, _best_real_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    # Crear DataFrame UNA SOLA VEZ por worker
    _train_df = pd.DataFrame(train_data, index=train_index, columns=train_columns)
    _best_real_pf = best_real_pf


# Función para procesar una permutación (debe estar en nivel módulo para pickle)
def process_permutation_generic(perm_i):
    """Procesa una permutación individual - VERSIÓN OPTIMIZADA REAL"""
    # Usar variables globales (ya cargadas en initializer)
    strategy = _strategy_module
    train_df = _train_df
    best_real_pf = _best_real_pf

    # Ejecutar permutación
    train_perm = get_permutation(train_df, seed=perm_i)
    result = strategy.optimize(train_perm)

    # Desempaquetar resultado: último valor es pf, resto son parámetros
    *best_params, best_perm_pf = result

    # Calcular cumulative returns para esta permutación
    sig = strategy.signal(train_perm, *best_params)
    r = np.log(train_perm['close']).diff().shift(-1)
    perm_rets = sig * r
    cum_rets = perm_rets.cumsum().values  # Convertir a numpy array

    is_better = 1 if best_perm_pf >= best_real_pf else 0
    return best_perm_pf, is_better, cum_rets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MCPT In-Sample Analysis with Dask support and interactive visualization'
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
        '--backend',
        choices=['multiprocess', 'dask'],
        default='multiprocess',
        help='Parallelization backend (default: multiprocess)'
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
        help='Number of workers (default: auto-detect)'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive visualization generation'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("\n" + "="*70)
    print(f"MCPT - VERSIÓN GENÉRICA (Backend: {args.backend.upper()})")
    print("="*70 + "\n")

    strategy_file = args.strategy

    # Remover extensión .py si está presente
    if strategy_file.endswith('.py'):
        strategy_name = strategy_file[:-3]
    else:
        strategy_name = strategy_file

    # Intentar importar el módulo de la estrategia
    try:
        strategy = importlib.import_module(f'models.strategies.{strategy_name}')
        print(f"✓ Estrategia cargada: {strategy_file}")
    except ModuleNotFoundError:
        print(f"ERROR: No se encontró el módulo '{strategy_name}.py'")
        print("Estrategias disponibles: donchian, moving_average, tree_strat, hawkes, donchian_aft_loss")
        sys.exit(1)

    # Verificar que el módulo tiene las funciones requeridas
    if not hasattr(strategy, 'optimize') or not hasattr(strategy, 'signal'):
        print(f"ERROR: El módulo '{strategy_name}' debe tener las funciones 'optimize' y 'signal'")
        sys.exit(1)

    # Asegurar directorios
    ensure_directories()

    # Configuración de workers
    total_cpus = cpu_count()
    if args.n_workers:
        n_workers = min(args.n_workers, total_cpus)
    else:
        n_workers = int(os.getenv('N_WORKERS', min(15, total_cpus)))

    # Número de permutaciones
    n_permutations = args.n_permutations

    # Get ticker data paths
    try:
        ticker_paths = get_ticker_data_paths(args.ticker)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nConfiguración:")
    print(f"  Ticker:           {args.ticker}")
    print(f"  Training period:  {args.train_start} to {args.train_end}")
    print(f"  CPUs disponibles: {total_cpus}")
    print(f"  Workers a usar:   {n_workers}")
    print(f"  Backend:          {args.backend}")
    print(f"  Permutaciones:    {n_permutations}")
    if HAS_DASK and args.backend == 'dask':
        print(f"  Dask disponible:  ✓")
    elif args.backend == 'dask' and not HAS_DASK:
        print(f"  Dask disponible:  ✗ (usando multiprocess)")
        args.backend = 'multiprocess'
    print("="*70 + "\n")
    sys.stdout.flush()

    # Cargar datos
    print(f"Cargando datos de {args.ticker}...")
    parquet_path = ticker_paths['parquet']
    csv_path = ticker_paths['csv']

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.index = df.index.astype('datetime64[s]')
    elif csv_path.exists():
        print(f"Convirtiendo CSV a Parquet...")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df.to_parquet(parquet_path)
    else:
        print(f"ERROR: No data files found for {args.ticker}")
        print(f"  Expected parquet: {parquet_path}")
        print(f"  Expected csv: {csv_path}")
        sys.exit(1)

    print(f"✓ Datos cargados: {len(df)} filas\n")

    # Análisis in-sample
    print("="*70)
    print("OPTIMIZACIÓN IN-SAMPLE")
    print("="*70)

    # Filter by training period
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    train_df = df[(df.index >= train_start) & (df.index < train_end)]

    if len(train_df) == 0:
        print(f"ERROR: No data found in training period {args.train_start} to {args.train_end}")
        sys.exit(1)

    print(f"  Training period:    {train_start} to {train_end}")
    print(f"  Training samples:   {len(train_df)}")
    result = strategy.optimize(train_df)

    # Desempaquetar resultado: último valor es pf, resto son parámetros
    *best_params, best_real_pf = result

    # Calcular cumulative returns de la estrategia real
    real_signal = strategy.signal(train_df, *best_params)
    real_r = np.log(train_df['close']).diff().shift(-1)
    real_rets = real_signal * real_r
    real_cum_rets = real_rets.cumsum()

    print(f"  Best Parameters:    {best_params}")
    print(f"  Best Profit Factor: {best_real_pf:.4f}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # MCPT
    print(f"Ejecutando MCPT con {n_permutations} permutaciones usando {n_workers} workers...")
    print()
    sys.stdout.flush()

    # Preparar datos para el initializer
    train_data = train_df.values
    train_index = train_df.index
    train_columns = train_df.columns.tolist()

    # OPTIMIZACIÓN CLAVE: Solo pasar índices, los datos van en initializer
    args_list = list(range(n_permutations))

    print("="*70)
    print("PROGRESO")
    print("="*70)
    print(f"  Inicio: {time.strftime('%H:%M:%S')}")
    print("="*70 + "\n")

    start_time = time.time()
    results = []  # lista para almacenar resultados

    # Chunksize más pequeño para mejor distribución de carga
    chunksize = max(1, n_permutations // (n_workers * 10))

    # Select backend for parallel processing
    if args.backend == 'dask' and HAS_DASK:
        # Dask backend
        print("Usando Dask para paralelización...")
        from dask import delayed

        config = load_parallel_config()
        dask_config = config.get('dask', {})
        dask_config['n_workers'] = n_workers

        with DaskOrchestrator(dask_config) as orchestrator:
            # Initialize global variables in the main process for delayed tasks
            _init_worker(strategy_name, train_data, train_index, train_columns, best_real_pf)

            # Create delayed tasks
            tasks = [delayed(process_permutation_generic)(i) for i in args_list]

            # Show dashboard info
            dashboard_link = orchestrator.get_dashboard_link()
            print(f"\n{'='*70}")
            print(f"DASK DASHBOARD")
            print(f"{'='*70}")
            print(f"  Monitor progress at: {dashboard_link}")
            print(f"  (Open in browser to view task progress, memory usage, etc.)")
            print(f"{'='*70}\n")
            sys.stdout.flush()

            # Compute all tasks with progress bar
            futures = orchestrator.client.compute(tasks)
            for future in tqdm(orchestrator.client.gather(futures),
                             total=len(tasks),
                             desc="Procesando tareas (Dask)",
                             ncols=80):
                results.append(future)
    else:
        # Multiprocessing backend (default)
        # CLAVE: Pasar datos en initializer, no en args
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(strategy_name, train_data, train_index, train_columns, best_real_pf)) as pool:
            for result in tqdm(pool.imap_unordered(process_permutation_generic, args_list, chunksize=chunksize),
                            total=len(args_list),
                            desc="Procesando tareas",
                            ncols=80):
                results.append(result)

    total_time = time.time() - start_time
    print(f"\nTiempo total: {total_time:.0f}s")

    # Análisis de resultados
    permuted_pfs = [pf for pf, _, _ in results]
    perm_better_count = 1 + sum(is_better for _, is_better, _ in results)
    perm_cum_rets = [cum_rets for _, _, cum_rets in results]
    insample_mcpt_pval = perm_better_count / n_permutations

    print("="*70)
    print("RESULTADOS MCPT")
    print("="*70)
    print(f"  Permutaciones:     {len(results)}")
    print(f"  Mejores que real:  {perm_better_count}")
    print(f"  P-Value:           {insample_mcpt_pval:.4f}")
    print(f"  Tiempo total:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Velocidad:         {len(results)/total_time:.1f} tareas/s")

    if insample_mcpt_pval < 0.05:
        print(f"  ✅ Significativo (p < 0.05)")
    else:
        print(f"  ⚠️  NO significativo (p >= 0.05)")

    print("="*70 + "\n")
    sys.stdout.flush()

    # Generar gráfico
    print("Generando gráfico...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(permuted_pfs, bins=50, color='steelblue',
            alpha=0.7, edgecolor='white', label='Permutations')

    ax.axvline(best_real_pf, color='red', linestyle='--',
               linewidth=2.5, label=f'Real PF: {best_real_pf:.4f}')

    mean_perm = np.mean(permuted_pfs)
    ax.axvline(mean_perm, color='yellow', linestyle=':',
               linewidth=2, alpha=0.7, label=f'Mean Perm: {mean_perm:.4f}')

    ax.set_xlabel("Profit Factor", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"In-sample MCPT ({strategy_file.replace('.py', '')}) | P-Value: {insample_mcpt_pval:.4f} | "
                 f"{'Significant' if insample_mcpt_pval < 0.05 else 'Not Significant'}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    # Crear directorio de salida para esta estrategia
    strategy_name_clean = strategy_file.replace(".py", "")
    output_dir = BACKTEST_FIGURES / strategy_name_clean
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f'insample_mcpt.png'
    output_file = output_dir / output_filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico guardado: {output_file}")
    print(f"  (p-value: {insample_mcpt_pval:.4f})\n")
    plt.close()

    # Generar gráfico de cumulative returns
    print("Generando gráfico de cumulative returns...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Graficar todas las permutaciones de manera más eficiente
    # Convertir lista de arrays a matriz numpy y plotear de una vez
    perm_matrix = np.array(perm_cum_rets).T  # Transponer para tener tiempo en filas
    ax.plot(train_df.index, perm_matrix, color='white', alpha=0.02, linewidth=0.5)

    # Graficar la estrategia real en rojo
    ax.plot(train_df.index, real_cum_rets, color='red', linewidth=2.5,
            label=f'Real Strategy (PF={best_real_pf:.4f})', zorder=100)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"In-sample Cumulative Returns ({strategy_file.replace('.py', '')}) | Real vs {len(perm_cum_rets)} Permutations",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()

    output_filename_cum = f'insample_cumulative_mcpt.png'
    output_file_cum = output_dir / output_filename_cum
    plt.savefig(output_file_cum, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico cumulative returns guardado: {output_file_cum}\n")
    plt.close()

    # Generate interactive visualization if available and not disabled
    if HAS_INTERACTIVE_VIS and not args.no_interactive:
        print("Generando visualización interactiva...")

        # Check if strategy has visualization() method
        if hasattr(strategy, 'visualization'):
            try:
                vis_data = strategy.visualization(train_df, *best_params)
                interactive_path = output_dir / 'interactive_chart.html'

                result_path = create_interactive_chart(
                    ohlc_data=train_df,
                    vis_data=vis_data,
                    strategy_name=strategy_name,
                    params=tuple(best_params),
                    output_path=interactive_path
                )

                if result_path:
                    print(f"✓ Visualización interactiva: {result_path}\n")
                else:
                    print("  (No se pudo generar la visualización interactiva)\n")
            except Exception as e:
                print(f"  Error generando visualización interactiva: {e}\n")
        else:
            print(f"  Estrategia '{strategy_name}' no tiene método visualization()\n")
    elif not HAS_INTERACTIVE_VIS:
        print("Visualización interactiva no disponible (instalar: pip install lightweight-charts)\n")

    print("="*70)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*70)
    print(f"\nArchivos generados en: {output_dir}")
    print(f"  - insample_mcpt.png")
    print(f"  - insample_cumulative_mcpt.png")
    if HAS_INTERACTIVE_VIS and not args.no_interactive and hasattr(strategy, 'visualization'):
        print(f"  - interactive_chart.html")
