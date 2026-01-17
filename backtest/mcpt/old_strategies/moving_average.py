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

def signal(ohlc: pd.DataFrame, fast: int, slow: int):
    if fast >= slow:
        raise ValueError("fast MA debe ser menor que slow MA")
    fast_ma = ohlc['close'].rolling(fast).mean()
    slow_ma = ohlc['close'].rolling(slow).mean()
    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)
    sig[fast_ma > slow_ma] = 1
    sig[fast_ma < slow_ma] = -1
    return sig.ffill()

def optimize(ohlc: pd.DataFrame, fast_range=(15, 25), slow_range=(140, 160)):
    best_pf = 0
    best_fast = -1
    best_slow = -1
    r = np.log(ohlc['close']).diff().shift(-1)
    for fast in range(fast_range[0], fast_range[1] + 1):
        for slow in range(slow_range[0], slow_range[1] + 1):
            if fast >= slow:
                continue
            sig = signal(ohlc, fast, slow)
            sig_rets = sig * r
            pos = sig_rets[sig_rets > 0].sum()
            neg = sig_rets[sig_rets < 0].abs().sum()
            if neg == 0:
                sig_pf = np.inf if pos > 0 else 0.0
            else:
                sig_pf = pos / neg

            if sig_pf > best_pf:
                best_pf = sig_pf
                best_fast = fast
                best_slow = slow

    return best_fast, best_slow, best_pf

def walkforward(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_fast, best_slow, _ = optimize(ohlc.iloc[i-train_lookback:i])
            tmp_signal = signal(ohlc, best_fast, best_slow)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal

# Alias para compatibilidad con código existente
ma_signal = signal
optimize_ma = optimize
walkforward_ma = walkforward

# --- Main ---
if __name__ == "__main__":
    ensure_directories()

    # Cargar datos
    print("Cargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2018) & (df.index.year < 2023)]
    print(f"✓ Datos cargados: {len(df)} filas (2018-2022)")

    # Optimización in-sample
    train_window = 24*365*4  # 4 años de datos horarios
    best_fast, best_slow, best_pf = optimize_ma(df.iloc[:train_window])

    # Señales y retornos in-sample
    signal_in = ma_signal(df.iloc[:train_window], best_fast, best_slow)
    df_in = df.iloc[:train_window].copy()
    df_in['r'] = np.log(df_in['close']).diff().shift(-1)
    df_in['strategy_ret'] = signal_in * df_in['r']

    # Señales y retornos out-of-sample (walkforward)
    df_out = df.iloc[train_window:].copy()
    wf_signal = walkforward_ma(df, train_lookback=train_window, train_step=24*60)
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

    # Resultados
    print("\n" + "="*60)
    print("RESULTADOS DE LA ESTRATEGIA MOVING AVERAGE CROSSOVER")
    print("="*60)
    print(f"\nIN-SAMPLE (Entrenamiento - {len(df_in)} horas):")
    print(f"  Cumulative Log Return: {cum_in_total:.4f}")
    print(f"  Profit Factor: {pf_in:.4f}")
    print(f"\nOUT-OF-SAMPLE (Walk-Forward - {len(df_out)} horas):")
    print(f"  Cumulative Log Return: {cum_out_total:.4f}")
    print(f"  Profit Factor: {pf_out:.4f}")
    print("\n" + "="*60)

    # Plot
    plt.style.use("dark_background")
    plt.figure(figsize=(14,7))
    plt.plot(df_in.index, cum_in, color='orange', linewidth=2, label='In-Sample (4 años)')
    plt.plot(df_out.index, cum_out, color='red', linewidth=2, label='Out-of-Sample (último año)')

    plt.xlabel("Time")
    plt.ylabel("Cumulative Log Return")
    plt.title("Moving Average Crossover Strategy: In-Sample vs Out-of-Sample")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = get_plot_path("cumulative_insample_outsample_ma.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"✓ Gráfico guardado: {output_file}")
