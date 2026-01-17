import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

# Agregar directorio padre al path para importar config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar rutas relativas y funciones de salida
from config.path import BITCOIN_PARQUET, get_plot_path, ensure_directories

def signal(ohlc: pd.DataFrame, model):
    """
    Genera señales de trading usando un modelo de árbol de decisión entrenado

    Args:
        ohlc: DataFrame con columna 'close'
        model: Modelo DecisionTreeClassifier entrenado

    Returns:
        Series con señales (1 o -1)
    """
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff168 = log_c.diff(168)

    dataset = pd.concat([diff6, diff24, diff168], axis=1)
    dataset.columns = ['diff6', 'diff24', 'diff168']

    dataset = dataset.dropna()

    pred = model.predict(dataset.to_numpy())
    pred = pd.Series(pred, index=dataset.index)

    # Reindex to actual data
    pred = pred.reindex(ohlc.index)

    # Make predictions tradable: 0 -> -1, 1 -> 1
    sig = np.where(pred > 0, 1, -1)
    sig = pd.Series(sig, index=ohlc.index)

    return sig

def optimize(ohlc: pd.DataFrame):
    """
    Entrena un modelo de árbol de decisión y retorna el modelo y su profit factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        tuple: (model, profit_factor)
    """
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff168 = log_c.diff(168)

    # -1 or 1 if next 24 hours go up/down
    target = np.sign(log_c.diff(24).shift(-24))

    # Transform to -1, 1 to 0, 1
    target = (target + 1) / 2

    dataset = pd.concat([diff6, diff24, diff168, target], axis=1)
    dataset.columns = ['diff6', 'diff24', 'diff168', 'target']

    train_data = dataset.dropna()
    train_x = train_data[['diff6', 'diff24', 'diff168']].to_numpy()
    train_y = train_data['target'].astype(int).to_numpy()

    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=69)
    model.fit(train_x, train_y)

    # Calcular profit factor del modelo entrenado
    sig = signal(ohlc, model)
    r = np.log(ohlc['close']).diff().shift(-1)
    sig_rets = sig * r
    pos = sig_rets[sig_rets > 0].sum()
    neg = sig_rets[sig_rets < 0].abs().sum()
    if neg == 0:
        pf = np.inf if pos > 0 else 0.0
    else:
        pf = pos / neg

    return model, pf

def walkforward(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
    """
    Implementa walk-forward optimization para la estrategia de árbol de decisión

    Args:
        ohlc: DataFrame con columna 'close'
        train_lookback: Tamaño de la ventana de entrenamiento
        train_step: Frecuencia de re-entrenamiento

    Returns:
        numpy array con señales
    """
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_model, _ = optimize(ohlc.iloc[i-train_lookback:i])
            tmp_signal = signal(ohlc, best_model)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal

# Alias para compatibilidad con código existente
train_tree = lambda ohlc: optimize(ohlc)[0]
tree_strategy = lambda ohlc, model: (signal(ohlc, model), optimize(ohlc)[1])

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
    best_model, best_pf = optimize(df.iloc[:train_window])

    # Señales y retornos in-sample
    signal_in = signal(df.iloc[:train_window], best_model)
    df_in = df.iloc[:train_window].copy()
    df_in['r'] = np.log(df_in['close']).diff().shift(-1)
    df_in['strategy_ret'] = signal_in * df_in['r']

    # Señales y retornos out-of-sample (walkforward)
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

    # Resultados
    print("\n" + "="*60)
    print("RESULTADOS DE LA ESTRATEGIA DECISION TREE")
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
    plt.title("Decision Tree Strategy: In-Sample vs Out-of-Sample")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = get_plot_path("cumulative_insample_outsample_tree.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"✓ Gráfico guardado: {output_file}")

