"""
Análisis completo de la estrategia Hawkes Volatility

Genera:
1. Grid search de parámetros (kappa, lookback) en in-sample
2. Métricas detalladas por combinación de parámetros
3. Walk-forward con mejores parámetros
4. Gráfico interactivo con precio, percentil y señales
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Añadir path del proyecto
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from strategies import hawkes


def calculate_metrics(ohlc: pd.DataFrame, signals: pd.Series):
    """
    Calcula métricas detalladas de una estrategia

    Returns:
        dict con todas las métricas
    """
    # Retornos logarítmicos
    log_returns = np.log(ohlc["close"]).diff()

    # Estrategia: señal * retorno futuro
    strat_returns = signals * log_returns.shift(-1)
    strat_returns = strat_returns.dropna()

    # Cumulative returns
    cum_returns = strat_returns.cumsum()
    log_cum_returns = cum_returns.iloc[-1] if len(cum_returns) > 0 else 0

    # Separar trades ganadores y perdedores
    winning_trades = strat_returns[strat_returns > 0]
    losing_trades = strat_returns[strat_returns < 0]

    # Básicos
    total_trades = len(strat_returns[strat_returns != 0])
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)

    win_ratio = winning_count / total_trades if total_trades > 0 else 0

    # Profit Factor
    total_profit = winning_trades.sum()
    total_loss = abs(losing_trades.sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else (np.inf if total_profit > 0 else 0)

    # Mejor win y peor loss
    best_win = winning_trades.max() if len(winning_trades) > 0 else 0
    worst_loss = losing_trades.min() if len(losing_trades) > 0 else 0

    # Calcular rachas (streaks)
    # Crear serie de wins/losses
    trade_results = strat_returns.copy()
    trade_results[trade_results > 0] = 1  # Win
    trade_results[trade_results < 0] = -1  # Loss
    trade_results[trade_results == 0] = 0  # No trade

    # Eliminar no-trades para calcular rachas
    trades_only = trade_results[trade_results != 0]

    # Calcular rachas
    if len(trades_only) > 0:
        streaks = []
        current_streak = 1
        current_type = trades_only.iloc[0]

        for i in range(1, len(trades_only)):
            if trades_only.iloc[i] == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = trades_only.iloc[i]

        # Añadir última racha
        streaks.append((current_type, current_streak))

        # Separar rachas ganadoras y perdedoras
        winning_streaks = [s[1] for s in streaks if s[0] == 1]
        losing_streaks = [s[1] for s in streaks if s[0] == -1]

        max_winning_streak = max(winning_streaks) if winning_streaks else 0
        max_losing_streak = max(losing_streaks) if losing_streaks else 0
        avg_winning_streak = np.mean(winning_streaks) if winning_streaks else 0
        avg_losing_streak = np.mean(losing_streaks) if losing_streaks else 0
    else:
        max_winning_streak = 0
        max_losing_streak = 0
        avg_winning_streak = 0
        avg_losing_streak = 0

    # Separar trades long y short
    long_returns = strat_returns[signals.shift(1) == 1]
    short_returns = strat_returns[signals.shift(1) == -1]

    long_wins = len(long_returns[long_returns > 0])
    long_losses = len(long_returns[long_returns < 0])
    short_wins = len(short_returns[short_returns > 0])
    short_losses = len(short_returns[short_returns < 0])

    return {
        'log_cum_returns': log_cum_returns,
        'total_trades': total_trades,
        'winning_trades': winning_count,
        'losing_trades': losing_count,
        'win_ratio': win_ratio,
        'profit_factor': profit_factor,
        'best_win': best_win,
        'worst_loss': worst_loss,
        'max_winning_streak': max_winning_streak,
        'max_losing_streak': max_losing_streak,
        'avg_winning_streak': avg_winning_streak,
        'avg_losing_streak': avg_losing_streak,
        'long_wins': long_wins,
        'long_losses': long_losses,
        'short_wins': short_wins,
        'short_losses': short_losses,
        'total_profit': total_profit,
        'total_loss': total_loss
    }


def grid_search_parameters(ohlc: pd.DataFrame, kappa_values, lookback_values):
    """
    Grid search sobre parámetros de Hawkes

    Returns:
        DataFrame con resultados para cada combinación
    """
    results = []

    total_combinations = len(kappa_values) * len(lookback_values)
    print(f"\n🔍 Grid search con {total_combinations} combinaciones...")

    for i, kappa in enumerate(kappa_values):
        for j, lookback in enumerate(lookback_values):
            print(f"  [{i*len(lookback_values)+j+1}/{total_combinations}] Testing kappa={kappa:.3f}, lookback={lookback}", end='\r')

            # Generar señales
            sig = hawkes.signal(ohlc, kappa, lookback)

            # Calcular métricas
            metrics = calculate_metrics(ohlc, sig)

            # Añadir parámetros
            metrics['kappa'] = kappa
            metrics['lookback'] = lookback

            results.append(metrics)

    print()  # Nueva línea después del loop

    return pd.DataFrame(results)


def plot_parameter_heatmaps(results_df, output_dir):
    """
    Genera heatmaps de métricas vs parámetros
    """
    metrics_to_plot = [
        ('log_cum_returns', 'Log Cumulative Returns'),
        ('profit_factor', 'Profit Factor'),
        ('win_ratio', 'Win Ratio'),
        ('max_winning_streak', 'Max Winning Streak'),
        ('max_losing_streak', 'Max Losing Streak')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics_to_plot):
        # Crear pivot table
        pivot = results_df.pivot_table(
            values=metric,
            index='kappa',
            columns='lookback',
            aggfunc='mean'
        )

        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f' if metric != 'max_winning_streak' and metric != 'max_losing_streak' else '.0f',
            cmap='RdYlGn' if metric != 'max_losing_streak' else 'RdYlGn_r',
            ax=axes[idx],
            cbar_kws={'label': title}
        )

        axes[idx].set_title(f'{title}\n(kappa vs lookback)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Lookback (hours)', fontsize=10)
        axes[idx].set_ylabel('Kappa', fontsize=10)

    # Eliminar subplot extra
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"✅ Heatmaps guardados en {output_dir / 'parameter_heatmaps.png'}")
    plt.close()


def plot_metrics_comparison(results_df, output_dir):
    """
    Gráfico de barras comparando diferentes métricas
    """
    # Encontrar mejor configuración por Profit Factor
    best_idx = results_df['profit_factor'].idxmax()
    best_row = results_df.loc[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top 10 configuraciones por Profit Factor
    top10 = results_df.nlargest(10, 'profit_factor')
    labels = [f"κ={row['kappa']:.3f}\nLB={row['lookback']}" for _, row in top10.iterrows()]

    axes[0, 0].barh(range(len(top10)), top10['profit_factor'], color='green', alpha=0.7)
    axes[0, 0].set_yticks(range(len(top10)))
    axes[0, 0].set_yticklabels(labels, fontsize=8)
    axes[0, 0].set_xlabel('Profit Factor', fontsize=10)
    axes[0, 0].set_title('Top 10 Configurations by Profit Factor', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Breakeven')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()

    # 2. Win Ratio vs Profit Factor scatter
    axes[0, 1].scatter(results_df['win_ratio'], results_df['profit_factor'],
                       c=results_df['total_trades'], cmap='viridis', s=100, alpha=0.6)
    axes[0, 1].set_xlabel('Win Ratio', fontsize=10)
    axes[0, 1].set_ylabel('Profit Factor', fontsize=10)
    axes[0, 1].set_title('Win Ratio vs Profit Factor\n(color = total trades)', fontsize=12, fontweight='bold')
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Total Trades', fontsize=9)

    # 3. Rachas ganadoras vs perdedoras
    axes[1, 0].scatter(results_df['max_winning_streak'], results_df['max_losing_streak'],
                       c=results_df['profit_factor'], cmap='RdYlGn', s=100, alpha=0.6)
    axes[1, 0].set_xlabel('Max Winning Streak', fontsize=10)
    axes[1, 0].set_ylabel('Max Losing Streak', fontsize=10)
    axes[1, 0].set_title('Winning vs Losing Streaks\n(color = Profit Factor)', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar2.set_label('Profit Factor', fontsize=9)

    # 4. Long vs Short performance
    results_df['long_wr'] = results_df['long_wins'] / (results_df['long_wins'] + results_df['long_losses'])
    results_df['short_wr'] = results_df['short_wins'] / (results_df['short_wins'] + results_df['short_losses'])
    results_df['long_wr'] = results_df['long_wr'].fillna(0)
    results_df['short_wr'] = results_df['short_wr'].fillna(0)

    axes[1, 1].scatter(results_df['long_wr'], results_df['short_wr'],
                       c=results_df['log_cum_returns'], cmap='coolwarm', s=100, alpha=0.6)
    axes[1, 1].set_xlabel('Long Win Ratio', fontsize=10)
    axes[1, 1].set_ylabel('Short Win Ratio', fontsize=10)
    axes[1, 1].set_title('Long vs Short Performance\n(color = Log Cum Returns)', fontsize=12, fontweight='bold')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal performance')
    axes[1, 1].legend()
    cbar3 = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar3.set_label('Log Cum Returns', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Comparación de métricas guardada en {output_dir / 'metrics_comparison.png'}")
    plt.close()


def save_best_parameters(results_df, output_dir):
    """
    Guarda los mejores parámetros y sus métricas en un archivo de texto
    """
    best_idx = results_df['profit_factor'].idxmax()
    best_row = results_df.loc[best_idx]

    output_file = output_dir / 'best_parameters.txt'

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MEJORES PARÁMETROS HAWKES - IN-SAMPLE (2018-2022)\n")
        f.write("=" * 70 + "\n\n")

        f.write("PARÁMETROS ÓPTIMOS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Kappa:    {best_row['kappa']:.4f}\n")
        f.write(f"  Lookback: {int(best_row['lookback'])} hours\n\n")

        f.write("MÉTRICAS DE RENDIMIENTO:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Log Cumulative Returns: {best_row['log_cum_returns']:.4f}\n")
        f.write(f"  Profit Factor:          {best_row['profit_factor']:.4f}\n")
        f.write(f"  Win Ratio:              {best_row['win_ratio']:.2%}\n\n")

        f.write("ESTADÍSTICAS DE TRADES:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Trades:     {int(best_row['total_trades'])}\n")
        f.write(f"  Winning Trades:   {int(best_row['winning_trades'])}\n")
        f.write(f"  Losing Trades:    {int(best_row['losing_trades'])}\n\n")

        f.write("LONG vs SHORT:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Long Wins:        {int(best_row['long_wins'])}\n")
        f.write(f"  Long Losses:      {int(best_row['long_losses'])}\n")
        f.write(f"  Short Wins:       {int(best_row['short_wins'])}\n")
        f.write(f"  Short Losses:     {int(best_row['short_losses'])}\n\n")

        f.write("MEJORES/PEORES TRADES:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Best Win:         {best_row['best_win']:.4f}\n")
        f.write(f"  Worst Loss:       {best_row['worst_loss']:.4f}\n\n")

        f.write("RACHAS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Max Winning Streak:     {int(best_row['max_winning_streak'])}\n")
        f.write(f"  Max Losing Streak:      {int(best_row['max_losing_streak'])}\n")
        f.write(f"  Avg Winning Streak:     {best_row['avg_winning_streak']:.2f}\n")
        f.write(f"  Avg Losing Streak:      {best_row['avg_losing_streak']:.2f}\n\n")

        f.write("PROFIT/LOSS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Profit:     {best_row['total_profit']:.4f}\n")
        f.write(f"  Total Loss:       {best_row['total_loss']:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"✅ Mejores parámetros guardados en {output_file}")

    return best_row['kappa'], best_row['lookback']


def plot_interactive_signals(ohlc: pd.DataFrame, kappa: float, lookback: int, output_dir):
    """
    Genera gráfico interactivo con precio, percentil y señales Hawkes
    """
    # Calcular componentes de la estrategia
    high = np.log(ohlc["high"])
    low = np.log(ohlc["low"])
    hl_range = high - low
    atr = hl_range.rolling(336).mean()
    norm_range = hl_range / atr
    v_hawk = hawkes.hawkes_process(norm_range, kappa)

    # Calcular percentiles
    q05 = v_hawk.rolling(lookback).quantile(0.05)
    q95 = v_hawk.rolling(lookback).quantile(0.95)

    # Generar señales
    signals = hawkes.signal(ohlc, kappa, lookback)

    # Encontrar puntos de entrada
    entries_long = signals[(signals == 1) & (signals.shift(1) != 1)]
    entries_short = signals[(signals == -1) & (signals.shift(1) != -1)]
    exits = signals[(signals == 0) & (signals.shift(1) != 0)]

    # Crear figura con subplots
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    # 1. Precio de Bitcoin
    axes[0].plot(ohlc.index, ohlc['close'], label='BTC Price', color='black', linewidth=1.5)

    # Marcar entradas y salidas
    for idx in entries_long.index:
        axes[0].scatter(idx, ohlc.loc[idx, 'close'], color='green', s=100, marker='^',
                       zorder=5, label='Long Entry' if idx == entries_long.index[0] else '')

    for idx in entries_short.index:
        axes[0].scatter(idx, ohlc.loc[idx, 'close'], color='red', s=100, marker='v',
                       zorder=5, label='Short Entry' if idx == entries_short.index[0] else '')

    for idx in exits.index:
        axes[0].scatter(idx, ohlc.loc[idx, 'close'], color='blue', s=50, marker='x',
                       zorder=5, alpha=0.6, label='Exit' if idx == exits.index[0] else '')

    axes[0].set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Bitcoin Price & Trading Signals\n(kappa={kappa:.4f}, lookback={lookback}h)',
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 2. Proceso de Hawkes con percentiles
    axes[1].plot(v_hawk.index, v_hawk, label='Hawkes Process', color='purple', linewidth=1.5, alpha=0.8)
    axes[1].plot(q05.index, q05, label='5th Percentile (q05)', color='blue', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].plot(q95.index, q95, label='95th Percentile (q95)', color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Sombrear regiones
    axes[1].fill_between(v_hawk.index, q05, q95, alpha=0.1, color='gray', label='Normal Range')

    axes[1].set_ylabel('Hawkes Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Hawkes Volatility Process & Percentile Thresholds', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 3. Señales de trading
    axes[2].fill_between(signals.index, 0, signals, where=(signals > 0),
                        color='green', alpha=0.3, label='Long Position', step='post')
    axes[2].fill_between(signals.index, 0, signals, where=(signals < 0),
                        color='red', alpha=0.3, label='Short Position', step='post')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    axes[2].set_ylabel('Signal', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=12, fontweight='bold')
    axes[2].set_title('Trading Signals Timeline', fontsize=14, fontweight='bold')
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_yticklabels(['Short', 'Flat', 'Long'])
    axes[2].legend(loc='upper left', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'interactive_signals.png', dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de señales guardado en {output_dir / 'interactive_signals.png'}")
    plt.close()


def walkforward_analysis(ohlc: pd.DataFrame, best_kappa: float, best_lookback: int,
                        train_years: int = 4, test_months: int = 2, output_dir=None):
    """
    Análisis walk-forward con los mejores parámetros
    """
    print("\n📊 Ejecutando Walk-Forward Analysis...")

    train_bars = 24 * 365 * train_years
    test_bars = 24 * 30 * test_months

    results = []

    start_idx = train_bars
    while start_idx + test_bars < len(ohlc):
        # Período de test
        test_start = start_idx
        test_end = start_idx + test_bars

        test_data = ohlc.iloc[test_start:test_end]

        # Generar señales con parámetros óptimos
        sig = hawkes.signal(test_data, best_kappa, best_lookback)

        # Calcular métricas
        metrics = calculate_metrics(test_data, sig)
        metrics['period_start'] = test_data.index[0]
        metrics['period_end'] = test_data.index[-1]

        results.append(metrics)

        # Avanzar ventana
        start_idx += test_bars

    wf_df = pd.DataFrame(results)

    # Gráfico de walk-forward
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Cumulative Returns por período
    axes[0].plot(range(len(wf_df)), wf_df['log_cum_returns'].cumsum(),
                marker='o', linewidth=2, markersize=6, color='blue', label='Cumulative Returns')
    axes[0].set_ylabel('Cumulative Log Returns', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Walk-Forward Analysis (kappa={best_kappa:.4f}, lookback={best_lookback}h)\n' +
                     f'Train={train_years}Y, Test={test_months}M', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].legend()

    # 2. Profit Factor por período
    axes[1].bar(range(len(wf_df)), wf_df['profit_factor'],
               color=['green' if pf > 1 else 'red' for pf in wf_df['profit_factor']],
               alpha=0.7, edgecolor='black')
    axes[1].axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='Breakeven')
    axes[1].set_ylabel('Profit Factor', fontsize=11, fontweight='bold')
    axes[1].set_title('Profit Factor per Period', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()

    # 3. Win Ratio por período
    axes[2].plot(range(len(wf_df)), wf_df['win_ratio'],
                marker='s', linewidth=2, markersize=6, color='orange', label='Win Ratio')
    axes[2].axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50%')
    axes[2].set_ylabel('Win Ratio', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Period', fontsize=11, fontweight='bold')
    axes[2].set_title('Win Ratio per Period', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim([0, 1])

    for ax in axes:
        ax.set_xticks(range(len(wf_df)))
        ax.set_xticklabels([f"P{i+1}" for i in range(len(wf_df))], rotation=0)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'walkforward_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Walk-forward analysis guardado en {output_dir / 'walkforward_analysis.png'}")
    plt.close()

    # Guardar estadísticas de walk-forward
    if output_dir:
        with open(output_dir / 'walkforward_stats.txt', 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("WALK-FORWARD ANALYSIS STATISTICS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Número de períodos:        {len(wf_df)}\n")
            f.write(f"Train window:              {train_years} años\n")
            f.write(f"Test window:               {test_months} meses\n\n")

            f.write("PROFIT FACTOR:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Media:        {wf_df['profit_factor'].mean():.4f}\n")
            f.write(f"  Mediana:      {wf_df['profit_factor'].median():.4f}\n")
            f.write(f"  Std Dev:      {wf_df['profit_factor'].std():.4f}\n")
            f.write(f"  Min:          {wf_df['profit_factor'].min():.4f}\n")
            f.write(f"  Max:          {wf_df['profit_factor'].max():.4f}\n")
            f.write(f"  Períodos > 1: {(wf_df['profit_factor'] > 1).sum()}/{len(wf_df)}\n\n")

            f.write("WIN RATIO:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Media:        {wf_df['win_ratio'].mean():.2%}\n")
            f.write(f"  Mediana:      {wf_df['win_ratio'].median():.2%}\n")
            f.write(f"  Std Dev:      {wf_df['win_ratio'].std():.4f}\n")
            f.write(f"  Min:          {wf_df['win_ratio'].min():.2%}\n")
            f.write(f"  Max:          {wf_df['win_ratio'].max():.2%}\n\n")

            f.write("LOG CUMULATIVE RETURNS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total:        {wf_df['log_cum_returns'].sum():.4f}\n")
            f.write(f"  Media:        {wf_df['log_cum_returns'].mean():.4f}\n")
            f.write(f"  Std Dev:      {wf_df['log_cum_returns'].std():.4f}\n")
            f.write(f"  Períodos +:   {(wf_df['log_cum_returns'] > 0).sum()}/{len(wf_df)}\n")
            f.write("=" * 70 + "\n")

        print(f"✅ Estadísticas walk-forward guardadas en {output_dir / 'walkforward_stats.txt'}")

    return wf_df


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETO DE ESTRATEGIA HAWKES")
    print("="*70)

    # Cargar datos
    data_path = project_root / "mcpt" / "data" / "BTCUSD3600.pq"
    print(f"\n📂 Cargando datos desde {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"✅ Datos cargados: {len(df)} barras desde {df.index[0]} hasta {df.index[-1]}")

    # Período in-sample: 2018-2022
    insample_start = "2018-01-01"
    insample_end = "2022-12-31"
    df_insample = df.loc[insample_start:insample_end]
    print(f"\n📅 In-sample period: {df_insample.index[0]} a {df_insample.index[-1]} ({len(df_insample)} barras)")

    # Crear directorio de salida
    output_dir = project_root / "output" / "hawkes_complete"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Grid search de parámetros
    kappa_values = [0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    lookback_values = [96, 120, 144, 168, 192, 216, 240]

    print(f"\n🔬 Parámetros a explorar:")
    print(f"   Kappa: {kappa_values}")
    print(f"   Lookback: {lookback_values}")

    results_df = grid_search_parameters(df_insample, kappa_values, lookback_values)

    # Guardar resultados completos
    results_csv = output_dir / 'grid_search_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Resultados completos guardados en {results_csv}")

    # Generar visualizaciones
    print("\n📊 Generando visualizaciones...")
    plot_parameter_heatmaps(results_df, output_dir)
    plot_metrics_comparison(results_df, output_dir)

    # Guardar mejores parámetros
    best_kappa, best_lookback = save_best_parameters(results_df, output_dir)

    print(f"\n🏆 MEJORES PARÁMETROS:")
    print(f"   Kappa:    {best_kappa:.4f}")
    print(f"   Lookback: {int(best_lookback)} hours")

    # Gráfico interactivo con mejores parámetros
    print("\n🎨 Generando gráfico interactivo de señales...")
    plot_interactive_signals(df_insample, best_kappa, int(best_lookback), output_dir)

    # Walk-forward analysis
    wf_results = walkforward_analysis(df, best_kappa, int(best_lookback),
                                     train_years=4, test_months=2, output_dir=output_dir)

    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*70)
    print(f"\n📁 Todos los resultados guardados en: {output_dir}")
    print("\nArchivos generados:")
    print("  - grid_search_results.csv       : Resultados completos del grid search")
    print("  - parameter_heatmaps.png        : Heatmaps de métricas vs parámetros")
    print("  - metrics_comparison.png        : Comparación de métricas")
    print("  - best_parameters.txt           : Mejores parámetros y métricas")
    print("  - interactive_signals.png       : Gráfico de precio, percentil y señales")
    print("  - walkforward_analysis.png      : Análisis walk-forward")
    print("  - walkforward_stats.txt         : Estadísticas walk-forward")
    print()


if __name__ == "__main__":
    main()
