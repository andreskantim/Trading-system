"""
Versión GENÉRICA del análisis MCPT sin Dask.
Permite elegir la estrategia desde línea de comandos.
Uso: python insample_permutation.py <estrategia>
Ejemplo: python insample_permutation.py donchian
         python insample_permutation.py moving_average
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
    BITCOIN_CSV, BITCOIN_PARQUET,
    get_plot_path, ensure_directories
)

from bar_permute import get_permutation


# Variables globales para el worker (cargadas una sola vez por worker)
_strategy_module = None
_train_df = None
_best_real_pf = None

def _init_worker(strategy_name, train_data, train_index, train_columns, best_real_pf):
    """Inicializa el worker con el módulo de estrategia y datos"""
    global _strategy_module, _train_df, _best_real_pf
    _strategy_module = importlib.import_module(f'strategies.{strategy_name}')
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


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MCPT - VERSIÓN GENÉRICA (sin Dask)")
    print("="*70 + "\n")

    # Validar argumentos de línea de comandos
    if len(sys.argv) < 2:
        print("ERROR: Debes especificar la estrategia")
        print("Uso: python insample_permutation.py <estrategia.py>")
        print("Ejemplo: python insample_permutation.py donchian.py")
        print("         python insample_permutation.py moving_average.py")
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
    if not hasattr(strategy, 'optimize') or not hasattr(strategy, 'signal'):
        print(f"ERROR: El módulo '{strategy_name}' debe tener las funciones 'optimize' y 'signal'")
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
    if BITCOIN_PARQUET.exists():
        df = pd.read_parquet(BITCOIN_PARQUET)
        df.index = df.index.astype('datetime64[s]')
    else:
        print(f"Convirtiendo CSV a Parquet...")
        df = pd.read_csv(BITCOIN_CSV, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df.to_parquet(BITCOIN_PARQUET)

    print(f"✓ Datos cargados: {len(df)} filas\n")

    # Análisis in-sample
    print("="*70)
    print("OPTIMIZACIÓN IN-SAMPLE")
    print("="*70)

    train_df = df[(df.index.year >= 2016) & (df.index.year < 2020)]
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
    n_permutations = 1000
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

    # Procesamiento con multiprocessing y barra de progreso
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
    output_dir = Path(__file__).resolve().parent.parent / "output" / strategy_name_clean
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

    print("="*70)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*70)
