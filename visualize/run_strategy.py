"""
Algoritmo para ejecutar y visualizar estrategias de trading

Ejecuta una estrategia con optimización insample y walkforward,
generando gráficos y métricas de rendimiento.

Uso:
    python run_strategy.py <estrategia> [--filters <filtros>]

Ejemplo:
    python run_strategy.py donchian
    python run_strategy.py moving_average
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import importlib
from typing import Optional
import argparse

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar configuración de rutas
from config.path import BITCOIN_PARQUET, ensure_directories


def get_output_dir(strategy_name: str) -> Path:
    """
    Crea y retorna el directorio de salida para una estrategia

    Args:
        strategy_name: Nombre de la estrategia

    Returns:
        Path al directorio de salida
    """
    output_dir = Path(__file__).resolve().parent.parent / "output" / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def walkforward(ohlc: pd.DataFrame, strategy, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
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


def plot_parameter_space(ohlc: pd.DataFrame, strategy, strategy_name: str, output_dir: Path):
    """
    Genera gráfico de log return vs parámetros de la estrategia

    Args:
        ohlc: DataFrame con datos OHLC
        strategy: Módulo de estrategia
        strategy_name: Nombre de la estrategia
        output_dir: Directorio de salida
    """
    print("\nGenerando gráfico de espacio de parámetros...")

    r = np.log(ohlc['close']).diff().shift(-1)

    # Detectar número de parámetros probando optimize
    result = strategy.optimize(ohlc)
    n_params = len(result) - 1  # Excluir PF

    if strategy_name == 'donchian' or strategy_name == 'donchian_aft_loss':
        # 1 parámetro: lookback
        lookbacks = range(12, 169)
        cum_rets = []

        for lb in lookbacks:
            sig = strategy.signal(ohlc, lb)
            sig_rets = sig * r
            cum_ret = sig_rets.sum()
            cum_rets.append(cum_ret)

        # Plot 1D
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(lookbacks, cum_rets, color='cyan', linewidth=2)
        ax.set_xlabel('Lookback Period', fontsize=12)
        ax.set_ylabel('Cumulative Log Return', fontsize=12)
        ax.set_title(f'{strategy_name.title()} - Parameter Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = output_dir / "parameter_space.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"✓ Gráfico de parámetros guardado: {output_file}")

    elif strategy_name == 'moving_average':
        # 2 parámetros: fast, slow
        fast_range = range(15, 26)
        slow_range = range(140, 161)
        cum_rets = np.zeros((len(fast_range), len(slow_range)))

        for i, fast in enumerate(fast_range):
            for j, slow in enumerate(slow_range):
                if fast >= slow:
                    cum_rets[i, j] = np.nan
                else:
                    try:
                        sig = strategy.signal(ohlc, fast, slow)
                        sig_rets = sig * r
                        cum_rets[i, j] = sig_rets.sum()
                    except:
                        cum_rets[i, j] = np.nan

        # Plot 2D heatmap
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cum_rets, cmap='RdYlGn', center=0,
                    xticklabels=list(slow_range)[::5],
                    yticklabels=list(fast_range)[::2],
                    cbar_kws={'label': 'Cumulative Log Return'},
                    ax=ax)
        ax.set_xlabel('Slow MA Period', fontsize=12)
        ax.set_ylabel('Fast MA Period', fontsize=12)
        ax.set_title(f'{strategy_name.title()} - Parameter Space Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = output_dir / "parameter_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"✓ Heatmap de parámetros guardado: {output_file}")

    elif strategy_name == 'hawkes':
        # 2 parámetros pero con rango limitado - similar a MA
        kappa_values = [0.075, 0.1, 0.125, 0.15]
        lb_values = [96, 120, 144, 168, 169, 192]
        cum_rets = np.zeros((len(kappa_values), len(lb_values)))

        for i, kappa in enumerate(kappa_values):
            for j, lb in enumerate(lb_values):
                sig = strategy.signal(ohlc, kappa, lb)
                sig_rets = sig * r
                cum_rets[i, j] = sig_rets.sum()

        # Plot 2D heatmap
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cum_rets, cmap='RdYlGn', center=0,
                    xticklabels=[str(lb) for lb in lb_values],
                    yticklabels=[f'{k:.3f}' for k in kappa_values],
                    cbar_kws={'label': 'Cumulative Log Return'},
                    ax=ax, annot=True, fmt='.3f')
        ax.set_xlabel('Lookback Period', fontsize=12)
        ax.set_ylabel('Kappa', fontsize=12)
        ax.set_title(f'{strategy_name.title()} - Parameter Space Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = output_dir / "parameter_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"✓ Heatmap de parámetros guardado: {output_file}")

    else:
        print(f"  Espacio de parámetros no soportado para {strategy_name}")


def run_strategy(strategy_name: str, filters: Optional[list] = None):
    """
    Ejecuta una estrategia con optimización y genera visualizaciones

    Args:
        strategy_name: Nombre de la estrategia (módulo en strategies/)
        filters: Lista de filtros a aplicar (opcional)
    """
    print("\n" + "="*70)
    print(f"EJECUTANDO ESTRATEGIA: {strategy_name.upper()}")
    print("="*70 + "\n")

    # Cargar estrategia
    try:
        strategy = importlib.import_module(f'strategies.{strategy_name}')
        print(f"✓ Estrategia cargada: {strategy_name}")
    except ModuleNotFoundError:
        print(f"ERROR: No se encontró la estrategia '{strategy_name}'")
        print("Estrategias disponibles: donchian, moving_average, tree_strat, hawkes, donchian_aft_loss")
        sys.exit(1)

    # Verificar funciones requeridas
    if not hasattr(strategy, 'signal') or not hasattr(strategy, 'optimize'):
        print(f"ERROR: La estrategia '{strategy_name}' debe tener las funciones 'signal' y 'optimize'")
        sys.exit(1)

    # Crear directorio de salida
    output_dir = get_output_dir(strategy_name)
    print(f"✓ Directorio de salida: {output_dir}")

    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2018) & (df.index.year < 2023)]
    print(f"✓ Datos cargados: {len(df)} filas (2018-2022)")

    # Optimización in-sample
    print("\n" + "="*70)
    print("OPTIMIZACIÓN IN-SAMPLE")
    print("="*70)

    train_window = 24*365*4  # 4 años
    result = strategy.optimize(df.iloc[:train_window])
    best_params = result[:-1]
    best_pf = result[-1]

    print(f"  Parámetros óptimos: {best_params}")
    print(f"  Profit Factor:      {best_pf:.4f}")

    # Señales y retornos in-sample
    signal_in = strategy.signal(df.iloc[:train_window], *best_params)
    df_in = df.iloc[:train_window].copy()
    df_in['r'] = np.log(df_in['close']).diff().shift(-1)
    df_in['strategy_ret'] = signal_in * df_in['r']

    # Métricas in-sample
    cum_in_total = df_in['strategy_ret'].sum()
    pos_in = df_in['strategy_ret'][df_in['strategy_ret'] > 0].sum()
    neg_in = df_in['strategy_ret'][df_in['strategy_ret'] < 0].abs().sum()
    pf_in = pos_in / neg_in if neg_in > 0 else np.inf

    print(f"  Cumulative Return:  {cum_in_total:.4f}")
    print("="*70)

    # Walk-forward out-of-sample
    print("\n" + "="*70)
    print("WALK-FORWARD OUT-OF-SAMPLE")
    print("="*70)

    df_out = df.iloc[train_window:].copy()
    wf_signal = walkforward(df, strategy, train_lookback=train_window, train_step=24*60)
    df_out['r'] = np.log(df_out['close']).diff().shift(-1)
    wf_signal_series = pd.Series(wf_signal, index=df.index)
    df_out['strategy_ret'] = wf_signal_series.iloc[train_window:] * df_out['r']

    # Métricas out-of-sample
    cum_out_total = df_out['strategy_ret'].sum()
    pos_out = df_out['strategy_ret'][df_out['strategy_ret'] > 0].sum()
    neg_out = df_out['strategy_ret'][df_out['strategy_ret'] < 0].abs().sum()
    pf_out = pos_out / neg_out if neg_out > 0 else np.inf

    print(f"  Profit Factor:      {pf_out:.4f}")
    print(f"  Cumulative Return:  {cum_out_total:.4f}")
    print("="*70)

    # Gráfico de cumulative returns
    print("\nGenerando gráficos...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))

    cum_in = df_in['strategy_ret'].cumsum()
    cum_out = df_out['strategy_ret'].cumsum()

    ax.plot(df_in.index, cum_in, color='orange', linewidth=2,
            label=f'In-Sample (PF={pf_in:.3f})')
    ax.plot(df_out.index, cum_out, color='red', linewidth=2,
            label=f'Out-of-Sample (PF={pf_out:.3f})')

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"{strategy_name.title()} Strategy: In-Sample vs Out-of-Sample",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_file = output_dir / "cumulative_returns.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"✓ Gráfico de retornos guardado: {output_file}")

    # Gráfico de espacio de parámetros
    if strategy_name != 'tree_strat':  # Tree no tiene espacio de parámetros simple
        plot_parameter_space(df.iloc[:train_window], strategy, strategy_name, output_dir)

    # Guardar métricas en archivo
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"ESTRATEGIA: {strategy_name.upper()}\n")
        f.write("="*70 + "\n\n")
        f.write("IN-SAMPLE\n")
        f.write(f"  Parámetros óptimos: {best_params}\n")
        f.write(f"  Profit Factor:      {pf_in:.4f}\n")
        f.write(f"  Cumulative Return:  {cum_in_total:.4f}\n\n")
        f.write("OUT-OF-SAMPLE (Walk-Forward)\n")
        f.write(f"  Profit Factor:      {pf_out:.4f}\n")
        f.write(f"  Cumulative Return:  {cum_out_total:.4f}\n")

    print(f"✓ Métricas guardadas: {metrics_file}")

    print("\n" + "="*70)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ejecutar y visualizar estrategias de trading')
    parser.add_argument('strategy', type=str, help='Nombre de la estrategia')
    parser.add_argument('--filters', nargs='*', help='Filtros a aplicar (opcional)')

    args = parser.parse_args()

    ensure_directories()
    run_strategy(args.strategy, args.filters)
