"""
Estrategia Donchian Breakout con filtro "After Loser"

Esta estrategia solo opera cuando el último trade fue perdedor,
aprovechando el sesgo mean-reversion detectado en el análisis de
dependencia de trades.

Hallazgos del análisis (Bitcoin 2018-2022):
- Profit Factor después de perder: 1.044
- Profit Factor después de ganar: 0.980
- Mejora: +6.58% operando solo después de perder

Compatible con:
- insample_permutation.py
- walkforward_permutation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Agregar directorio padre al path para importar config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar rutas relativas y funciones de salida
from config.path import BITCOIN_PARQUET, get_plot_path, ensure_directories


def donchian_base_signal(ohlc: pd.DataFrame, lookback: int):
    """
    Genera señales Donchian básicas (sin filtro)

    Args:
        ohlc: DataFrame con columna 'close'
        lookback: Periodo de lookback para canales Donchian

    Returns:
        Series con señales: 1 (long), -1 (short), ffill para mantener posición
    """
    upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['close'].rolling(lookback - 1).min().shift(1)
    sig = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    sig.loc[ohlc['close'] > upper] = 1
    sig.loc[ohlc['close'] < lower] = -1
    sig = sig.ffill()
    return sig


def apply_after_loser_filter(ohlc: pd.DataFrame, base_signal: pd.Series):
    """
    Aplica filtro "After Loser" a señales base.
    Solo opera cuando el último trade fue perdedor.

    Args:
        ohlc: DataFrame con columna 'close'
        base_signal: Series con señales base (1, -1, 0)

    Returns:
        Series con señales filtradas
    """
    # Convertir a numpy para evitar warnings
    signal = base_signal.to_numpy()
    close = ohlc['close'].to_numpy()

    # Señal filtrada (inicialmente todo en 0)
    filtered_signal = np.zeros(len(signal))

    # Tracking de trades
    long_entry_price = np.nan
    short_entry_price = np.nan
    last_long_result = np.nan  # 1: ganó, -1: perdió
    last_short_result = np.nan

    last_sig = 0.0

    for i in range(len(close)):
        # Detectar entrada long
        if signal[i] == 1.0 and last_sig != 1.0:
            long_entry_price = close[i]
            # Si había short abierto, cerrarlo y registrar resultado
            if not np.isnan(short_entry_price):
                # Short profit = (entry_price - exit_price) / entry_price
                # Si precio sube (close[i] > short_entry_price) → pierde
                # Si precio baja (close[i] < short_entry_price) → gana
                last_short_result = np.sign(short_entry_price - close[i])
                short_entry_price = np.nan

        # Detectar entrada short
        if signal[i] == -1.0 and last_sig != -1.0:
            short_entry_price = close[i]
            # Si había long abierto, cerrarlo y registrar resultado
            if not np.isnan(long_entry_price):
                # Long profit = (exit_price - entry_price) / entry_price
                # Si precio sube (close[i] > long_entry_price) → gana
                # Si precio baja (close[i] < long_entry_price) → pierde
                last_long_result = np.sign(close[i] - long_entry_price)
                long_entry_price = np.nan

        last_sig = signal[i]

        # Aplicar filtro: solo operar si el ÚLTIMO TRADE DEL LADO OPUESTO fue perdedor
        # Si queremos entrar long, miramos el último resultado del short
        # Si queremos entrar short, miramos el último resultado del long

        if signal[i] == 1.0 and last_short_result == -1:
            # Entrar long solo si último short perdió
            filtered_signal[i] = 1.0
        elif signal[i] == -1.0 and last_long_result == -1:
            # Entrar short solo si último long perdió
            filtered_signal[i] = -1.0

    return pd.Series(filtered_signal, index=ohlc.index)


def signal(ohlc: pd.DataFrame, lookback: int):
    """
    Genera señales Donchian con filtro "After Loser"

    Esta es la función principal que llaman los scripts de permutación.

    Args:
        ohlc: DataFrame con columna 'close'
        lookback: Periodo de lookback para canales Donchian

    Returns:
        Series con señales filtradas: 1 (long), -1 (short), 0 (flat)
    """
    # Generar señales base
    base_sig = donchian_base_signal(ohlc, lookback)

    # Aplicar filtro "After Loser"
    filtered_sig = apply_after_loser_filter(ohlc, base_sig)

    return filtered_sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza el lookback de Donchian con filtro "After Loser"

    Busca el lookback que maximiza el Profit Factor en el rango [12, 169)

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (best_lookback, best_pf)
    """
    best_pf = 0
    best_lookback = -1
    r = np.log(ohlc['close']).diff().shift(-1)

    for lookback in range(12, 169):
        sig = signal(ohlc, lookback)
        sig_rets = sig * r
        pos = sig_rets[sig_rets > 0].sum()
        neg = sig_rets[sig_rets < 0].abs().sum()

        if neg == 0:
            sig_pf = np.inf if pos > 0 else 0.0
        else:
            sig_pf = pos / neg

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf


def walkforward(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
    """
    Implementación walk-forward de la estrategia

    Args:
        ohlc: DataFrame con datos OHLC
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
            best_lookback, _ = optimize(ohlc.iloc[i-train_lookback:i])
            tmp_signal = signal(ohlc, best_lookback)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


# Alias para compatibilidad
donchian_aft_loss_signal = signal
optimize_donchian_aft_loss = optimize
walkforward_donchian_aft_loss = walkforward


# --- Main ---
if __name__ == "__main__":
    ensure_directories()

    print("\n" + "="*70)
    print("DONCHIAN BREAKOUT CON FILTRO 'AFTER LOSER'")
    print("="*70 + "\n")

    # Cargar datos
    print("Cargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2018) & (df.index.year < 2023)]
    print(f"✓ Datos cargados: {len(df)} filas (2018-2022)")

    # Optimización in-sample
    train_window = 24*365*4  # 4 años de datos horarios
    print(f"\nOptimizando in-sample ({train_window} barras)...")
    best_lookback, best_pf = optimize(df.iloc[:train_window])
    print(f"✓ Mejor lookback: {best_lookback}")
    print(f"✓ Profit Factor in-sample: {best_pf:.4f}")

    # Señales y retornos in-sample
    signal_in = signal(df.iloc[:train_window], best_lookback)
    df_in = df.iloc[:train_window].copy()
    df_in['r'] = np.log(df_in['close']).diff().shift(-1)
    df_in['strategy_ret'] = signal_in * df_in['r']

    # Señales y retornos out-of-sample (walkforward)
    print("\nEjecutando walk-forward out-of-sample...")
    df_out = df.iloc[train_window:].copy()
    wf_signal = walkforward(df, train_lookback=train_window, train_step=24*60)
    df_out['r'] = np.log(df_out['close']).diff().shift(-1)
    wf_signal = pd.Series(wf_signal, index=df.index)
    df_out['strategy_ret'] = wf_signal.iloc[train_window:] * df_out['r']

    # Cumulative returns
    cum_in = df_in['strategy_ret'].cumsum()
    cum_out = df_out['strategy_ret'].cumsum()

    # Cumulative returns totales
    cum_in_total = df_in['strategy_ret'].sum()
    cum_out_total = df_out['strategy_ret'].sum()

    # Profit Factor in-sample
    pos_in = df_in['strategy_ret'][df_in['strategy_ret'] > 0].sum()
    neg_in = df_in['strategy_ret'][df_in['strategy_ret'] < 0].abs().sum()
    pf_in = pos_in / neg_in if neg_in > 0 else np.inf

    # Profit Factor out-of-sample
    pos_out = df_out['strategy_ret'][df_out['strategy_ret'] > 0].sum()
    neg_out = df_out['strategy_ret'][df_out['strategy_ret'] < 0].abs().sum()
    pf_out = pos_out / neg_out if neg_out > 0 else np.inf

    # Número de trades
    n_trades_in = len(df_in[df_in['strategy_ret'] != 0])
    n_trades_out = len(df_out[df_out['strategy_ret'] != 0])

    # Sharpe ratio aproximado
    sharpe_in = df_in['strategy_ret'].mean() / df_in['strategy_ret'].std() if df_in['strategy_ret'].std() > 0 else 0
    sharpe_out = df_out['strategy_ret'].mean() / df_out['strategy_ret'].std() if df_out['strategy_ret'].std() > 0 else 0

    # Resultados
    print("\n" + "="*70)
    print("RESULTADOS DE LA ESTRATEGIA")
    print("="*70)
    print(f"\nIN-SAMPLE (Entrenamiento - {len(df_in)} horas):")
    print(f"  Lookback óptimo:       {best_lookback}")
    print(f"  Cumulative Log Return: {cum_in_total:.4f}")
    print(f"  Profit Factor:         {pf_in:.4f}")
    print(f"  Sharpe Ratio:          {sharpe_in:.4f}")
    print(f"  Número de trades:      {n_trades_in}")

    print(f"\nOUT-OF-SAMPLE (Walk-Forward - {len(df_out)} horas):")
    print(f"  Cumulative Log Return: {cum_out_total:.4f}")
    print(f"  Profit Factor:         {pf_out:.4f}")
    print(f"  Sharpe Ratio:          {sharpe_out:.4f}")
    print(f"  Número de trades:      {n_trades_out}")
    print("\n" + "="*70)

    # Comparación con estrategia base (sin filtro)
    print("\nCOMPARACIÓN CON DONCHIAN BASE (sin filtro After Loser):")
    print("="*70)

    # Importar donchian base
    import donchian as donch_base

    # In-sample base
    signal_base_in = donch_base.signal(df.iloc[:train_window], best_lookback)
    df_base_in = df.iloc[:train_window].copy()
    df_base_in['r'] = np.log(df_base_in['close']).diff().shift(-1)
    df_base_in['strategy_ret'] = signal_base_in * df_base_in['r']

    pos_base_in = df_base_in['strategy_ret'][df_base_in['strategy_ret'] > 0].sum()
    neg_base_in = df_base_in['strategy_ret'][df_base_in['strategy_ret'] < 0].abs().sum()
    pf_base_in = pos_base_in / neg_base_in if neg_base_in > 0 else np.inf
    cum_base_in_total = df_base_in['strategy_ret'].sum()
    n_trades_base_in = len(df_base_in[df_base_in['strategy_ret'] != 0])

    print(f"\nDonchian Base IN-SAMPLE:")
    print(f"  Profit Factor:    {pf_base_in:.4f}")
    print(f"  Cum Log Return:   {cum_base_in_total:.4f}")
    print(f"  Número de trades: {n_trades_base_in}")

    print(f"\nMejora con filtro 'After Loser' IN-SAMPLE:")
    print(f"  Profit Factor:    {((pf_in/pf_base_in - 1)*100):+.2f}%")
    print(f"  Cum Log Return:   {((cum_in_total/cum_base_in_total - 1)*100):+.2f}%")
    print(f"  Reducción trades: {((n_trades_in/n_trades_base_in - 1)*100):+.2f}%")
    print("="*70)

    # Plot comparativo
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Cumulative returns
    ax = axes[0]
    ax.plot(df_in.index, cum_in, color='#4CAF50', linewidth=2,
            label=f'After Loser (PF={pf_in:.3f}, trades={n_trades_in})')
    ax.plot(df_base_in.index, df_base_in['strategy_ret'].cumsum(),
            color='#888888', linewidth=2, alpha=0.7,
            label=f'Base (PF={pf_base_in:.3f}, trades={n_trades_base_in})')

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Cumulative Log Return", fontsize=11)
    ax.set_title("IN-SAMPLE: Donchian 'After Loser' vs Base Strategy",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 2: Out-of-sample
    ax = axes[1]
    ax.plot(df_out.index, cum_out, color='#F44336', linewidth=2,
            label=f'After Loser OOS (PF={pf_out:.3f}, trades={n_trades_out})')

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Cumulative Log Return", fontsize=11)
    ax.set_title("OUT-OF-SAMPLE: Walk-Forward Performance",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    output_file = get_plot_path("donchian_aft_loss_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n✓ Gráfico guardado: {output_file}")

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70 + "\n")
