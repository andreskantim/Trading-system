import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# FUNCIONES COPIADAS DE TradeDependenceRunsTest
# ============================================================================

def donchian_breakout(df: pd.DataFrame, lookback: int):
    """Calcula señales de Donchian breakout"""
    df['upper'] = df['close'].rolling(lookback - 1).max().shift(1)
    df['lower'] = df['close'].rolling(lookback - 1).min().shift(1)
    df['signal'] = np.nan
    df.loc[df['close'] > df['upper'], 'signal'] = 1
    df.loc[df['close'] < df['lower'], 'signal'] = -1
    df['signal'] = df['signal'].ffill()


def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    """Extrae trades individuales desde una señal"""
    long_trades = []
    short_trades = []

    # Asegurar que signal es numpy array
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy()

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index

    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0:  # Long entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)
            open_trade = [idx[i], close_arr[i], -1, np.nan]

        if signal[i] == -1.0 and last_sig != -1.0:  # Short entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)
            open_trade = [idx[i], close_arr[i], -1, np.nan]

        if signal[i] == 0.0 and last_sig == -1.0:  # Short exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            short_trades.append(open_trade)
            open_trade = None

        if signal[i] == 0.0 and last_sig == 1.0:  # Long exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            long_trades.append(open_trade)
            open_trade = None

        last_sig = signal[i]

    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    long_trades['return'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
    short_trades['return'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
    long_trades = long_trades.set_index('entry_time')
    short_trades = short_trades.set_index('entry_time')

    long_trades['type'] = 1
    short_trades['type'] = -1
    all_trades = pd.concat([long_trades, short_trades])
    all_trades = all_trades.sort_index()

    return long_trades, short_trades, all_trades


def last_trade_adj_signal(ohlc: pd.DataFrame, signal: np.array, last_winner: bool = False):
    """Ajusta señal para solo operar después de ganador/perdedor"""
    last_type = -1
    if last_winner:
        last_type = 1

    # Asegurar que signal es numpy array
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy()

    close = ohlc['close'].to_numpy()
    mod_signal = np.zeros(len(signal))

    long_entry_p = np.nan
    short_entry_p = np.nan
    last_long = np.nan
    last_short = np.nan

    last_sig = 0.0
    for i in range(len(close)):
        if signal[i] == 1.0 and last_sig != 1.0:  # Long entry
            long_entry_p = close[i]
            if not np.isnan(short_entry_p):
                last_short = np.sign(short_entry_p - close[i])
                short_entry_p = np.nan

        if signal[i] == -1.0 and last_sig != -1.0:  # Short entry
            short_entry_p = close[i]
            if not np.isnan(long_entry_p):
                last_long = np.sign(close[i] - long_entry_p)
                long_entry_p = np.nan

        last_sig = signal[i]

        if signal[i] == 1.0 and last_short == last_type:
            mod_signal[i] = 1.0
        if signal[i] == -1.0 and last_long == last_type:
            mod_signal[i] = -1.0

    return mod_signal


def runs_test(signs: np.array):
    """Test estadístico de rachas (runs test)"""
    assert len(signs) >= 2

    n_pos = len(signs[signs > 0])
    n_neg = len(signs[signs < 0])
    n = len(signs)

    # Mean number of expected runs
    mean = 2 * n_pos * n_neg / n + 1
    # Standard deviation of expected runs
    std = (mean - 1) * (mean - 2) / (n - 1)  # Variance
    std = std ** 0.5

    # Count observed runs
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1  # Streak broken

    # Z-Score
    z = (runs - mean) / std
    return z


def load_bitcoin_data(start_year=2018, end_year=2022):
    """Carga datos de Bitcoin para el periodo especificado"""
    data = pd.read_csv('bitcoin_hourly.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')

    # Filtrar por años
    data = data[(data.index.year >= start_year) & (data.index.year <= end_year)]
    data = data.dropna()

    # Calcular retornos logarítmicos
    data['r'] = np.log(data['close']).diff().shift(-1)

    print(f"Datos cargados: {len(data)} barras")
    print(f"Periodo: {data.index[0]} a {data.index[-1]}")

    return data


def analyze_trade_dependence_single_lookback(data, lookback=24):
    """Analiza la dependencia de trades para un lookback específico"""

    # Calcular señales
    donchian_breakout(data, lookback)
    long_trades, short_trades, all_trades = get_trades_from_signal(data, data['signal'])

    # Calcular si cada trade fue ganador (1) o perdedor (-1)
    all_trades['win_lose'] = np.sign(all_trades['return'])
    all_trades['prev_win_lose'] = all_trades['win_lose'].shift(1)

    # Análisis condicional
    results = {}

    # Todos los trades
    results['all_trades'] = {
        'mean_return': all_trades['return'].mean(),
        'win_rate': (all_trades['return'] > 0).mean(),
        'profit_factor': all_trades[all_trades['return'] > 0]['return'].sum() /
                        abs(all_trades[all_trades['return'] < 0]['return'].sum())
    }

    # Trades después de ganar
    after_win = all_trades[all_trades['prev_win_lose'] == 1]
    if len(after_win) > 0:
        results['after_win'] = {
            'mean_return': after_win['return'].mean(),
            'win_rate': (after_win['return'] > 0).mean(),
            'profit_factor': after_win[after_win['return'] > 0]['return'].sum() /
                            abs(after_win[after_win['return'] < 0]['return'].sum()) if len(after_win[after_win['return'] < 0]) > 0 else np.inf,
            'n_trades': len(after_win)
        }

    # Trades después de perder
    after_lose = all_trades[all_trades['prev_win_lose'] == -1]
    if len(after_lose) > 0:
        results['after_lose'] = {
            'mean_return': after_lose['return'].mean(),
            'win_rate': (after_lose['return'] > 0).mean(),
            'profit_factor': after_lose[after_lose['return'] > 0]['return'].sum() /
                            abs(after_lose[after_lose['return'] < 0]['return'].sum()) if len(after_lose[after_lose['return'] < 0]) > 0 else np.inf,
            'n_trades': len(after_lose)
        }

    # Runs test
    signs = np.sign(all_trades['return']).to_numpy()
    results['runs_z_score'] = runs_test(signs)

    return results, all_trades


def process_single_lookback(args):
    """Procesa un solo lookback - función auxiliar para paralelización"""
    lookback, data_dict = args

    # Reconstruir DataFrame desde dict
    data = pd.DataFrame(data_dict)
    data.index = pd.to_datetime(data.index)

    # Crear copia de datos
    df = data.copy()

    results_list = []

    # Calcular señales
    donchian_breakout(df, lookback)
    df['last_lose'] = last_trade_adj_signal(df, df['signal'].to_numpy(), last_winner=False)
    df['last_win'] = last_trade_adj_signal(df, df['signal'].to_numpy(), last_winner=True)

    # Calcular returns por estrategia
    orig = df['r'] * df['signal']
    lose = df['r'] * df['last_lose']
    win = df['r'] * df['last_win']

    # Calcular métricas - Todos los trades
    if orig[orig != 0].sum() != 0:
        pf_all = orig[orig > 0].sum() / abs(orig[orig < 0].sum()) if orig[orig < 0].sum() != 0 else np.inf
        results_list.append({
            'lookback': lookback,
            'type': 'All Trades',
            'profit_factor': pf_all,
            'log_profit_factor': np.log(pf_all) if pf_all > 0 and np.isfinite(pf_all) else 0,
            'mean_return': orig[orig != 0].mean(),
            'sharpe': orig[orig != 0].mean() / orig[orig != 0].std() if orig[orig != 0].std() > 0 else 0,
            'n_trades': len(orig[orig != 0])
        })

    # Último perdedor
    if lose[lose != 0].sum() != 0:
        pf_lose = lose[lose > 0].sum() / abs(lose[lose < 0].sum()) if lose[lose < 0].sum() != 0 else np.inf
        results_list.append({
            'lookback': lookback,
            'type': 'After Loser',
            'profit_factor': pf_lose,
            'log_profit_factor': np.log(pf_lose) if pf_lose > 0 and np.isfinite(pf_lose) else 0,
            'mean_return': lose[lose != 0].mean(),
            'sharpe': lose[lose != 0].mean() / lose[lose != 0].std() if lose[lose != 0].std() > 0 else 0,
            'n_trades': len(lose[lose != 0])
        })

    # Último ganador
    if win[win != 0].sum() != 0:
        pf_win = win[win > 0].sum() / abs(win[win < 0].sum()) if win[win < 0].sum() != 0 else np.inf
        results_list.append({
            'lookback': lookback,
            'type': 'After Winner',
            'profit_factor': pf_win,
            'log_profit_factor': np.log(pf_win) if pf_win > 0 and np.isfinite(pf_win) else 0,
            'mean_return': win[win != 0].mean(),
            'sharpe': win[win != 0].mean() / win[win != 0].std() if win[win != 0].std() > 0 else 0,
            'n_trades': len(win[win != 0])
        })

    # Calcular runs test
    _, _, all_trades = get_trades_from_signal(df, df['signal'])
    if len(all_trades) > 0:
        signs = np.sign(all_trades['return']).to_numpy()
        runs_z = runs_test(signs)
        results_list.append({
            'lookback': lookback,
            'type': 'Runs Z-Score',
            'runs_z_score': runs_z
        })

    return results_list


def analyze_multiple_lookbacks(data, lookbacks=None, n_jobs=None):
    """Analiza la dependencia para múltiples lookbacks usando paralelización"""

    if lookbacks is None:
        lookbacks = list(range(12, 169, 6))

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Dejar 1 CPU libre

    print(f"Usando {n_jobs} procesos paralelos para {len(lookbacks)} lookbacks...")

    # Convertir data a dict para serialización
    data_dict = data.to_dict('series')
    data_dict['index'] = data.index.astype(str)

    # Preparar argumentos para cada proceso
    args_list = [(lb, data_dict) for lb in lookbacks]

    # Ejecutar en paralelo
    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(process_single_lookback, args_list)

    # Aplanar lista de resultados
    results_list = []
    for result in results_nested:
        results_list.extend(result)

    print(f"✓ Análisis completado para {len(lookbacks)} lookbacks")

    return pd.DataFrame(results_list)


def plot_profit_factor_by_lookback(results_df, save_path='figures/profit_factor_by_lookback.png'):
    """Gráfico de Profit Factor por lookback y tipo"""

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filtrar solo los tipos de estrategia (no runs z-score)
    plot_data = results_df[results_df['type'].isin(['All Trades', 'After Loser', 'After Winner'])].copy()

    # Crear gráfico de barras agrupadas
    sns.barplot(
        data=plot_data,
        x='lookback',
        y='log_profit_factor',
        hue='type',
        palette=['#888888', '#4CAF50', '#F44336'],
        ax=ax
    )

    ax.axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Lookback Period', fontsize=12)
    ax.set_ylabel('Log(Profit Factor)', fontsize=12)
    ax.set_title('Profit Factor by Lookback and Previous Trade Result\nBitcoin 2018-2022', fontsize=14, fontweight='bold')
    ax.legend(title='Strategy Type', fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    Path('figures').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_runs_test_by_lookback(results_df, save_path='figures/runs_test_by_lookback.png'):
    """Gráfico del Z-score del Runs Test por lookback"""

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filtrar solo runs z-score
    runs_data = results_df[results_df['type'] == 'Runs Z-Score'].copy()

    if len(runs_data) > 0:
        ax.plot(runs_data['lookback'], runs_data['runs_z_score'],
                marker='o', linewidth=2, markersize=6, color='#FF9800')

        # Líneas de referencia para significancia estadística
        ax.axhline(1.96, color='red', linewidth=1, linestyle='--', alpha=0.7, label='95% Confidence')
        ax.axhline(-1.96, color='red', linewidth=1, linestyle='--', alpha=0.7)
        ax.axhline(0, color='white', linewidth=1, linestyle='-', alpha=0.3)

        ax.set_xlabel('Lookback Period', fontsize=12)
        ax.set_ylabel('Runs Test Z-Score', fontsize=12)
        ax.set_title('Trade Independence Test (Runs Test) by Lookback\nBitcoin 2018-2022\nZ > 2: Mean-reverting | Z < -2: Trending',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
        plt.show()


def plot_mean_return_comparison(results_df, save_path='figures/mean_return_comparison.png'):
    """Comparación de retorno medio por tipo de estrategia"""

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    plot_data = results_df[results_df['type'].isin(['All Trades', 'After Loser', 'After Winner'])].copy()

    sns.barplot(
        data=plot_data,
        x='lookback',
        y='mean_return',
        hue='type',
        palette=['#888888', '#4CAF50', '#F44336'],
        ax=ax
    )

    ax.axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Lookback Period', fontsize=12)
    ax.set_ylabel('Mean Return per Trade', fontsize=12)
    ax.set_title('Average Trade Return by Lookback and Previous Result\nBitcoin 2018-2022',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Strategy Type', fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_sharpe_comparison(results_df, save_path='figures/sharpe_comparison.png'):
    """Comparación de Sharpe ratio por tipo"""

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    plot_data = results_df[results_df['type'].isin(['All Trades', 'After Loser', 'After Winner'])].copy()

    sns.barplot(
        data=plot_data,
        x='lookback',
        y='sharpe',
        hue='type',
        palette=['#888888', '#4CAF50', '#F44336'],
        ax=ax
    )

    ax.axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Lookback Period', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Sharpe Ratio by Lookback and Previous Result\nBitcoin 2018-2022',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Strategy Type', fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_trade_count_comparison(results_df, save_path='figures/trade_count.png'):
    """Número de trades por lookback y tipo"""

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    plot_data = results_df[results_df['type'].isin(['All Trades', 'After Loser', 'After Winner'])].copy()

    sns.barplot(
        data=plot_data,
        x='lookback',
        y='n_trades',
        hue='type',
        palette=['#888888', '#4CAF50', '#F44336'],
        ax=ax
    )

    ax.set_xlabel('Lookback Period', fontsize=12)
    ax.set_ylabel('Number of Trades', fontsize=12)
    ax.set_title('Trade Count by Lookback and Filter Type\nBitcoin 2018-2022',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Strategy Type', fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_detailed_analysis_single_lookback(data, lookback=24, save_path='figures/detailed_analysis_lb24.png'):
    """Análisis detallado para un lookback específico"""

    results, all_trades = analyze_trade_dependence_single_lookback(data, lookback)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Distribución de returns
    ax = axes[0, 0]
    all_trades['return'].hist(bins=50, ax=ax, color='#2196F3', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Trade Return', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution of Trade Returns (Lookback {lookback})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # 2. Scatter plot: Previous vs Current return
    ax = axes[0, 1]
    all_trades_clean = all_trades.dropna(subset=['prev_win_lose'])
    colors = ['#F44336' if x == -1 else '#4CAF50' for x in all_trades_clean['prev_win_lose']]
    ax.scatter(all_trades_clean.index, all_trades_clean['return'],
               c=colors, alpha=0.6, s=30)
    ax.axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Trade Number', fontsize=11)
    ax.set_ylabel('Trade Return', fontsize=11)
    ax.set_title(f'Trade Returns Over Time (Green=After Win, Red=After Loss)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # 3. Box plot: Returns after win vs loss
    ax = axes[1, 0]
    after_win = all_trades[all_trades['prev_win_lose'] == 1]['return']
    after_lose = all_trades[all_trades['prev_win_lose'] == -1]['return']

    bp = ax.boxplot([after_win, after_lose],
                     tick_labels=['After Winner', 'After Loser'],
                     patch_artist=True,
                     medianprops=dict(color='yellow', linewidth=2),
                     boxprops=dict(facecolor='#1976D2', alpha=0.7))
    ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_ylabel('Trade Return', fontsize=11)
    ax.set_title(f'Return Distribution by Previous Trade Result', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # 4. Estadísticas resumidas
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
OVERALL STATISTICS (Lookback {lookback})
{'='*45}

All Trades:
  • Mean Return: {results['all_trades']['mean_return']:.4f}
  • Win Rate: {results['all_trades']['win_rate']:.2%}
  • Profit Factor: {results['all_trades']['profit_factor']:.3f}
  • Runs Z-Score: {results['runs_z_score']:.3f}

After Winner:
  • Mean Return: {results['after_win']['mean_return']:.4f}
  • Win Rate: {results['after_win']['win_rate']:.2%}
  • Profit Factor: {results['after_win']['profit_factor']:.3f}
  • N Trades: {results['after_win']['n_trades']}

After Loser:
  • Mean Return: {results['after_lose']['mean_return']:.4f}
  • Win Rate: {results['after_lose']['win_rate']:.2%}
  • Profit Factor: {results['after_lose']['profit_factor']:.3f}
  • N Trades: {results['after_lose']['n_trades']}

INTERPRETATION:
Runs Z-Score > 2: Mean-reverting (win after loss)
Runs Z-Score < -2: Trending (win after win)
|Z| < 2: Independent trades
"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))

    plt.suptitle(f'Detailed Trade Dependence Analysis - Lookback {lookback}\nBitcoin 2018-2022',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {save_path}")
    plt.show()


def main():
    """Función principal de análisis"""

    print("\n" + "="*60)
    print("ANÁLISIS DE DEPENDENCIA DE TRADES - DONCHIAN BREAKOUT")
    print("Bitcoin 2018-2022")
    print("="*60 + "\n")

    # Cargar datos
    print("1. Cargando datos...")
    data = load_bitcoin_data(2018, 2022)

    # Análisis para múltiples lookbacks
    print("\n2. Analizando múltiples lookbacks...")
    lookbacks = list(range(12, 169, 6))
    results_df = analyze_multiple_lookbacks(data, lookbacks)

    # Guardar resultados
    results_df.to_csv('results/trade_dependence_results.csv', index=False)
    print("\nResultados guardados en: results/trade_dependence_results.csv")

    # Generar gráficos
    print("\n3. Generando gráficos...")

    print("  - Profit Factor por lookback...")
    plot_profit_factor_by_lookback(results_df)

    print("  - Runs Test por lookback...")
    plot_runs_test_by_lookback(results_df)

    print("  - Mean Return comparison...")
    plot_mean_return_comparison(results_df)

    print("  - Sharpe Ratio comparison...")
    plot_sharpe_comparison(results_df)

    print("  - Trade count...")
    plot_trade_count_comparison(results_df)

    print("\n4. Análisis detallado para lookback 24...")
    plot_detailed_analysis_single_lookback(data, lookback=24)

    print("\n5. Análisis detallado para lookback 48...")
    plot_detailed_analysis_single_lookback(data, lookback=48,
                                           save_path='figures/detailed_analysis_lb48.png')

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print("\nResumen de hallazgos:")

    # Análisis de resultados
    strat_data = results_df[results_df['type'].isin(['All Trades', 'After Loser', 'After Winner'])]

    # Profit Factor promedio por tipo
    avg_pf = strat_data.groupby('type')['profit_factor'].mean()
    print("\nProfit Factor Promedio:")
    for strat_type, pf in avg_pf.items():
        print(f"  {strat_type}: {pf:.3f}")

    # Runs test promedio
    runs_data = results_df[results_df['type'] == 'Runs Z-Score']
    if len(runs_data) > 0:
        avg_runs_z = runs_data['runs_z_score'].mean()
        print(f"\nRuns Z-Score Promedio: {avg_runs_z:.3f}")
        if avg_runs_z > 2:
            print("  → Evidencia de MEAN-REVERSION (ganar después de perder)")
        elif avg_runs_z < -2:
            print("  → Evidencia de TRENDING (ganar después de ganar)")
        else:
            print("  → Trades relativamente INDEPENDIENTES")

    print("\n")


if __name__ == '__main__':
    # Crear directorios necesarios
    Path('figures').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    main()
