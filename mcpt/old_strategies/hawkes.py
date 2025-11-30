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

# --------------------------
# Hawkes Process
# --------------------------
def hawkes_process(data: pd.Series, kappa: float):
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    out = np.zeros(len(arr))
    out[:] = np.nan
    for i in range(1, len(arr)):
        if np.isnan(out[i-1]):
            out[i] = arr[i]
        else:
            out[i] = out[i-1] * alpha + arr[i]
    return pd.Series(out, index=data.index) * kappa

# --------------------------
# Señal basada en Hawkes
# --------------------------
def hawkes_vol_signal(close: pd.Series, v_hawk: pd.Series, lookback: int):
    signal = np.zeros(len(close))
    q05 = v_hawk.rolling(lookback).quantile(0.05)
    q95 = v_hawk.rolling(lookback).quantile(0.95)
    last_below = -1
    curr_sig = 0
    for i in range(1, len(close)):
        if v_hawk.iloc[i] < q05.iloc[i]:
            last_below = i
            curr_sig = 0
        if v_hawk.iloc[i] > q95.iloc[i] and v_hawk.iloc[i-1] <= q95.iloc[i-1] and last_below > 0:
            change = close.iloc[i] - close.iloc[last_below]
            curr_sig = 1 if change > 0 else -1
        signal[i] = curr_sig
    return pd.Series(signal, index=close.index)

# --------------------------
# Función principal de señal
# --------------------------
def signal(ohlc: pd.DataFrame, kappa: float, lookback: int):
    high = np.log(ohlc["high"])
    low = np.log(ohlc["low"])
    close = np.log(ohlc["close"])
    hl_range = high - low
    atr = hl_range.rolling(336).mean()
    norm_range = hl_range / atr
    v_hawk = hawkes_process(norm_range, kappa)
    sig = hawkes_vol_signal(ohlc['close'], v_hawk, lookback)
    return sig

# --------------------------
# Optimización
# --------------------------
def optimize(ohlc: pd.DataFrame):
    r = np.log(ohlc["close"]).diff().shift(-1)
    best_pf = 0
    best_kappa = 0.1
    best_lb = 169

    # Barrer valores de kappa alrededor de 0.1
    # kappa_values = [0.075, 0.1, 0.125, 0.15]
    kappa_values = [0.125]

    # Barrer valores de lb alrededor de 169
    # lb_values = [96, 120, 144, 168, 169, 192]
    lb_values = [96, 120, 144, 168, 169]

    for kappa in kappa_values:
        for lb in lb_values:
            sig = signal(ohlc, kappa, lb)
            strat = sig * r
            pos = strat[strat > 0].sum()
            neg = strat[strat < 0].abs().sum()
            if neg == 0:
                pf = np.inf if pos > 0 else 0
            else:
                pf = pos / neg
            if pf > best_pf:
                best_pf = pf
                best_kappa = kappa
                best_lb = lb

    return best_kappa, best_lb, best_pf

# --------------------------
# Walkforward
# --------------------------
def walkforward(ohlc: pd.DataFrame, train_lookback: int = 24*365*4, train_step: int = 24*30):
    n = len(ohlc)
    wf_sig = np.full(n, np.nan)
    next_train = train_lookback
    tmp_sig = None
    for i in range(next_train, n):
        if i == next_train:
            best_kappa, best_lb, _ = optimize(ohlc.iloc[i-train_lookback:i])
            tmp_sig = signal(ohlc, best_kappa, best_lb)
            next_train += train_step
        wf_sig[i] = tmp_sig.iloc[i]
    return wf_sig

# Aliases
hawkes_signal = signal
optimize_hawkes = optimize
walkforward_hawkes = walkforward

# --- Main ---
if __name__ == "__main__":
    ensure_directories()

    print("\n" + "="*70)
    print("HAWKES STRATEGY")
    print("="*70 + "\n")

    # Cargar datos
    print("Cargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2018) & (df.index.year < 2023)]
    print(f"✓ Datos cargados: {len(df)} filas (2018-2022)")

    # Optimización in-sample
    train_window = 24*365*4
    print(f"\nOptimizando in-sample ({train_window} barras)...")
    best_kappa, best_lookback, best_pf = optimize(df.iloc[:train_window])
    print(f"✓ Mejor kappa: {best_kappa}")
    print(f"✓ Mejor lookback: {best_lookback}")
    print(f"✓ Mejor Profit Factor: {best_pf:.4f}")

    # Señales y retornos in-sample
    signal_in = signal(df.iloc[:train_window], best_kappa, best_lookback)
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

    # Resultados
    cum_in_total = df_in['strategy_ret'].sum()
    cum_out_total = df_out['strategy_ret'].sum()
    pos_in = df_in['strategy_ret'][df_in['strategy_ret'] > 0].sum()
    neg_in = df_in['strategy_ret'][df_in['strategy_ret'] < 0].abs().sum()
    pf_in = pos_in / neg_in if neg_in > 0 else np.inf
    pos_out = df_out['strategy_ret'][df_out['strategy_ret'] > 0].sum()
    neg_out = df_out['strategy_ret'][df_out['strategy_ret'] < 0].abs().sum()
    pf_out = pos_out / neg_out if neg_out > 0 else np.inf
    n_trades_in = len(df_in[df_in['strategy_ret'] != 0])
    n_trades_out = len(df_out[df_out['strategy_ret'] != 0])
    sharpe_in = df_in['strategy_ret'].mean() / df_in['strategy_ret'].std() if df_in['strategy_ret'].std() > 0 else 0
    sharpe_out = df_out['strategy_ret'].mean() / df_out['strategy_ret'].std() if df_out['strategy_ret'].std() > 0 else 0

    print("\n" + "="*70)
    print("RESULTADOS DE LA ESTRATEGIA HAWKES")
    print("="*70)
    print(f"\nIN-SAMPLE (Entrenamiento - {len(df_in)} horas):")
    print(f"  Kappa óptimo:          {best_kappa}")
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
    print("="*70)

    # Plot comparativo
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_in.index, cum_in, color='#4CAF50', linewidth=2, label='In-Sample')
    ax.plot(df_out.index, cum_out, color='#F44336', linewidth=2, label='Out-of-Sample')
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_title("Hawkes Strategy Performance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_file = get_plot_path("hawkes_strategy_performance.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n✓ Gráfico guardado: {output_file}")
    print("\nANÁLISIS COMPLETADO\n")
