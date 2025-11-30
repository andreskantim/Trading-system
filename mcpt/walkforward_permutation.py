"""
Versión GENÉRICA del análisis Walk-Forward MCPT sin Dask.
Permite elegir la estrategia desde línea de comandos.
Uso: python walkforward_permutation.py <estrategia>
Ejemplo: python walkforward_permutation.py donchian
         python walkforward_permutation.py moving_average
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


# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar configuración de rutas
from config.path import (
    BITCOIN_PARQUET,
    get_plot_path, ensure_directories
)

from bar_permute import get_permutation


# Variables globales para el worker (cargadas una sola vez por worker)
_strategy_module = None
_df = None
_train_window = None
_real_wf_pf = None

def _init_worker(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf):
    """Inicializa el worker con el módulo de estrategia y datos"""
    global _strategy_module, _df, _train_window, _real_wf_pf
    _strategy_module = importlib.import_module(f'strategies.{strategy_name}')
    # Crear DataFrame UNA SOLA VEZ por worker
    _df = pd.DataFrame(df_data, index=df_index, columns=df_columns)
    _train_window = train_window
    _real_wf_pf = real_wf_pf


# Función para procesar una permutación (debe estar en nivel módulo para pickle)
def walkforward_strategy(ohlc: pd.DataFrame, strategy, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
    """
    Implementa walk-forward optimization genérica

    Args:
        ohlc: DataFrame con datos OHLC
        strategy: Módulo de estrategia con funciones signal y optimize
        train_lookback: Ventana de entrenamiento en barras
        train_step: Paso entre re-optimizaciones en barras

    Returns:
        Array con señales walk-forward
    """
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            result = strategy.optimize(ohlc.iloc[i-train_lookback:i])
            # Último valor es siempre el PF, el resto son parámetros
            best_params = result[:-1]
            tmp_signal = strategy.signal(ohlc, *best_params)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


def process_walkforward_permutation_generic(perm_i):
    """Procesa una permutación individual de walk-forward - VERSIÓN OPTIMIZADA REAL"""
    # Usar variables globales (ya cargadas en initializer)
    strategy = _strategy_module
    df_perm = _df.copy()  # Copiar para no modificar el original
    train_window = _train_window
    real_wf_pf = _real_wf_pf

    # Ejecutar permutación CON SEED para reproducibilidad
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

    # Calcular cumulative returns
    cum_rets = perm_rets.cumsum().values  # Convertir a numpy array

    is_better = 1 if perm_pf >= real_wf_pf else 0
    return perm_pf, is_better, cum_rets


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WALK-FORWARD MCPT - VERSIÓN GENÉRICA (sin Dask)")
    print("="*70 + "\n")

    # Validar argumentos de línea de comandos
    if len(sys.argv) < 2:
        print("ERROR: Debes especificar la estrategia")
        print("Uso: python walkforward_permutation.py <estrategia.py>")
        print("Ejemplo: python walkforward_permutation.py donchian.py")
        print("         python walkforward_permutation.py moving_average.py")
        sys.exit(1)

    strategy_file = sys.argv[1]

    # Remover extensión .py si está presente
    if strategy_file.endswith('.py'):
        strategy_name = strategy_file[:-3]
    else:
        strategy_name = strategy_file

    # Intentar importar el módulo de la estrategia
    try:
        strategy = importlib.import_module(f'strategies.{strategy_name}')
        print(f"✓ Estrategia cargada: {strategy_file}")
    except ModuleNotFoundError:
        print(f"ERROR: No se encontró el módulo '{strategy_name}.py'")
        print("Estrategias disponibles: donchian, moving_average, tree_strat, hawkes, donchian_aft_loss")
        sys.exit(1)

    # Verificar que el módulo tiene las funciones requeridas
    if not hasattr(strategy, 'signal') or not hasattr(strategy, 'optimize'):
        print(f"ERROR: El módulo '{strategy_name}' debe tener las funciones 'signal' y 'optimize'")
        sys.exit(1)

    # Asegurar directorios
    ensure_directories()

    # Configuración de workers - usar TODOS los cores
    total_cpus = cpu_count()
    n_workers = int(os.getenv('N_WORKERS', total_cpus))

    print(f"\nConfiguración:")
    print(f"  CPUs disponibles: {total_cpus}")
    print(f"  Workers a usar:   {n_workers}")
    print(f"  (Cambiar con: N_WORKERS=4 python {Path(__file__).name} {strategy_file})")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Cargar datos
    print("Cargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2018) & (df.index.year < 2024)]
    print(f"✓ Datos cargados: {len(df)} filas (2018-2023)\n")

    # Configuración walk-forward
    train_window = 24 * 365 * 4  # 4 years of hourly data

    print("="*70)
    print("ANÁLISIS WALK-FORWARD")
    print("="*70)
    print(f"  Datos totales: {len(df)} períodos ({len(df)/24/365:.1f} años)")
    print(f"  Train window: {train_window} períodos ({train_window/24/365:.1f} años)")

    # Calcular estrategia real
    df['r'] = np.log(df['close']).diff().shift(-1)
    df['wf_signal'] = walkforward_strategy(df, strategy, train_lookback=train_window)
    wf_rets = df['wf_signal'] * df['r']
    real_wf_pf = wf_rets[wf_rets > 0].sum() / wf_rets[wf_rets < 0].abs().sum()
    real_cum_rets = wf_rets.cumsum()

    print(f"  Real Profit Factor: {real_wf_pf:.4f}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # MCPT
    n_permutations = 200
    print(f"Ejecutando Walk-Forward MCPT con {n_permutations} permutaciones usando {n_workers} workers...")
    print()
    sys.stdout.flush()

    # Preparar datos para el initializer
    df_data = df.values
    df_index = df.index
    df_columns = df.columns.tolist()

    # OPTIMIZACIÓN CLAVE: Solo pasar índices, los datos van en initializer
    args_list = list(range(n_permutations))

    # Procesar con multiprocessing
    start_time = time.time()

    print("="*70)
    print("PROGRESO")
    print("="*70)
    print(f"  Inicio: {time.strftime('%H:%M:%S')}")
    print("="*70 + "\n")

    n_tasks = len(args_list)
    results = []  # crea la lista vacía donde se guardarán los resultados

    # Chunksize más pequeño para mejor distribución de carga
    chunksize = max(1, n_permutations // (n_workers * 10))

    # CLAVE: Pasar datos en initializer, no en args
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf)) as pool:
        # tqdm envuelve pool.imap_unordered para mostrar progreso
        for result in tqdm(pool.imap_unordered(process_walkforward_permutation_generic, args_list, chunksize=chunksize),
                        total=n_tasks,
                        desc="Procesando tareas",
                        ncols=80):
            results.append(result)

    total_time = time.time() - start_time
    print(f"\nTiempo total: {total_time:.0f}s")

    # Análisis de resultados
    permuted_pfs = [pf for pf, _, _ in results]
    perm_better_count = 1 + sum(is_better for _, is_better, _ in results)
    perm_cum_rets = [cum_rets for _, _, cum_rets in results]
    walkforward_mcpt_pval = perm_better_count / n_permutations

    print("="*70)
    print("RESULTADOS MCPT")
    print("="*70)
    print(f"  Permutaciones:     {len(results)}")
    print(f"  Mejores que real:  {perm_better_count}")
    print(f"  P-Value:           {walkforward_mcpt_pval:.4f}")
    print(f"  Tiempo total:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Velocidad:         {len(results)/total_time:.1f} tareas/s")

    if walkforward_mcpt_pval < 0.05:
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

    ax.axvline(real_wf_pf, color='red', linestyle='--',
               linewidth=2.5, label=f'Real PF: {real_wf_pf:.4f}')

    mean_perm = np.mean(permuted_pfs)
    ax.axvline(mean_perm, color='yellow', linestyle=':',
               linewidth=2, alpha=0.7, label=f'Mean Perm: {mean_perm:.4f}')

    ax.set_xlabel("Profit Factor", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Walk-Forward MCPT ({strategy_file.replace('.py', '')}) | P-Value: {walkforward_mcpt_pval:.4f} | "
                 f"{'Significant' if walkforward_mcpt_pval < 0.05 else 'Not Significant'}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    # Crear directorio de salida para esta estrategia
    strategy_name_clean = strategy_file.replace(".py", "")
    output_dir = Path(__file__).resolve().parent.parent / "output" / strategy_name_clean
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f'walkforward_mcpt.png'
    output_file = output_dir / output_filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico guardado: {output_file}")
    print(f"  (p-value: {walkforward_mcpt_pval:.4f})\n")
    plt.close()

    # Generar gráfico de cumulative returns
    print("Generando gráfico de cumulative returns...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Graficar todas las permutaciones de manera más eficiente
    # Convertir lista de arrays a matriz numpy y plotear de una vez
    perm_matrix = np.array(perm_cum_rets).T  # Transponer para tener tiempo en filas
    ax.plot(df.index, perm_matrix, color='white', alpha=0.05, linewidth=0.5)

    # Graficar la estrategia real en rojo
    ax.plot(df.index, real_cum_rets, color='red', linewidth=2.5,
            label=f'Real Strategy (PF={real_wf_pf:.4f})', zorder=100)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"Walk-Forward Cumulative Returns ({strategy_file.replace('.py', '')}) | Real vs {len(perm_cum_rets)} Permutations",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()

    output_filename_cum = f'walkforward_cumulative_mcpt.png'
    output_file_cum = output_dir / output_filename_cum
    plt.savefig(output_file_cum, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico cumulative returns guardado: {output_file_cum}\n")
    plt.close()

    print("="*70)
    print("✓ ANÁLISIS WALK-FORWARD COMPLETADO")
    print("="*70)
