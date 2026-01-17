"""
Ejemplo: Cómo correlacionar patrones ordinales con estrategias de trading

Este script muestra cómo usar los patrones ordinales para:
1. Filtrar señales de estrategias existentes
2. Mejorar timing de entradas/salidas
3. Ajustar position sizing dinámicamente
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from perm_entropy_enhanced import ordinal_patterns

def simple_donchian_strategy(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Estrategia Donchian simple.

    Long: cuando precio rompe máximo de N periodos
    Short: cuando precio rompe mínimo de N periodos
    """
    high_channel = data['high'].rolling(window).max()
    low_channel = data['low'].rolling(window).min()

    signals = pd.Series(0, index=data.index)

    # Long cuando cierre > canal superior
    signals[data['close'] > high_channel.shift(1)] = 1

    # Short cuando cierre < canal inferior
    signals[data['close'] < low_channel.shift(1)] = -1

    return signals

def pattern_regime(pattern: int) -> str:
    """
    Identifica régimen basado en patrón ordinal.

    Returns: 'bullish', 'bearish', o 'neutral'
    """
    if pattern in [0, 1, 2]:
        return 'bearish'
    elif pattern in [3, 4, 5]:
        return 'bullish'
    else:
        return 'neutral'

def filter_by_regime(signals: pd.Series, patterns: pd.Series) -> pd.Series:
    """
    Filtra señales de trading por régimen de patrones ordinales.

    Solo permite:
    - Señales LONG en régimen alcista
    - Señales SHORT en régimen bajista
    """
    filtered_signals = signals.copy()

    for i in range(len(signals)):
        pattern = patterns.iloc[i]
        signal = signals.iloc[i]

        if np.isnan(pattern):
            filtered_signals.iloc[i] = 0
            continue

        regime = pattern_regime(int(pattern))

        # Filtrar señales contrarias al régimen
        if regime == 'bearish' and signal > 0:  # Long en régimen bajista
            filtered_signals.iloc[i] = 0
        elif regime == 'bullish' and signal < 0:  # Short en régimen alcista
            filtered_signals.iloc[i] = 0

    return filtered_signals

def dynamic_position_sizing(signals: pd.Series, patterns: pd.Series,
                           transition_matrix: np.ndarray) -> pd.Series:
    """
    Ajusta tamaño de posición según probabilidad de transición.

    Mayor probabilidad de continuar tendencia → mayor posición
    """
    position_sizes = pd.Series(1.0, index=signals.index)

    for i in range(1, len(signals)):
        if signals.iloc[i] == 0:
            position_sizes.iloc[i] = 0
            continue

        current_pattern = patterns.iloc[i]
        if np.isnan(current_pattern):
            continue

        pattern_int = int(current_pattern)

        # Calcular fuerza del patrón (auto-transición)
        self_transition_prob = transition_matrix[pattern_int, pattern_int]

        # Escalar posición (1.0 = baseline, max 2.0)
        # Si auto-transición es alta, aumentar posición
        position_sizes.iloc[i] = 1.0 + self_transition_prob

    return position_sizes

def calculate_returns(data: pd.DataFrame, signals: pd.Series,
                     position_sizes: pd.Series = None) -> pd.Series:
    """
    Calcula retornos de estrategia.

    Args:
        data: DataFrame con precios
        signals: Serie con señales (1=long, -1=short, 0=flat)
        position_sizes: Serie con tamaños de posición (opcional)
    """
    # Retornos del activo
    returns = data['close'].pct_change()

    if position_sizes is None:
        position_sizes = pd.Series(1.0, index=signals.index)

    # Retornos de estrategia
    strategy_returns = signals.shift(1) * returns * position_sizes.shift(1)

    return strategy_returns

def calculate_metrics(returns: pd.Series) -> dict:
    """
    Calcula métricas de performance.
    """
    returns_clean = returns.dropna()

    total_return = (1 + returns_clean).prod() - 1
    annual_return = (1 + total_return) ** (365*24 / len(returns_clean)) - 1
    annual_vol = returns_clean.std() * np.sqrt(365*24)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Drawdown
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns_clean > 0).sum() / len(returns_clean)

    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Num Trades': (returns_clean != 0).sum()
    }

if __name__ == '__main__':
    print("="*80)
    print("CORRELACIÓN DE PATRONES ORDINALES CON ESTRATEGIA DONCHIAN")
    print("="*80)

    # Cargar datos procesados
    data = pd.read_csv('PermutationEntropy/results/BTCUSDT3600_processed.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    print(f"\nDatos: {len(data)} velas horarias")

    # Cargar matriz de transición
    trans_matrix = pd.read_csv('PermutationEntropy/results/transition_matrix_close.csv', index_col=0).values

    # Estrategia Donchian baseline
    print("\n" + "="*80)
    print("1. ESTRATEGIA DONCHIAN BASELINE (sin filtro)")
    print("="*80)

    donchian_signals = simple_donchian_strategy(data, window=20)
    donchian_returns = calculate_returns(data, donchian_signals)
    baseline_metrics = calculate_metrics(donchian_returns)

    print("\nMétricas:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")

    # Estrategia con filtro de régimen
    print("\n" + "="*80)
    print("2. ESTRATEGIA DONCHIAN + FILTRO DE RÉGIMEN")
    print("="*80)

    filtered_signals = filter_by_regime(donchian_signals, data['pattern_close'])
    filtered_returns = calculate_returns(data, filtered_signals)
    filtered_metrics = calculate_metrics(filtered_returns)

    print("\nMétricas:")
    for metric, value in filtered_metrics.items():
        improvement = ""
        if metric in baseline_metrics:
            if metric in ['Sharpe Ratio', 'Total Return', 'Annual Return', 'Win Rate']:
                diff = value - baseline_metrics[metric]
                improvement = f" ({diff:+.4f})"
            elif metric == 'Max Drawdown':
                diff = value - baseline_metrics[metric]
                improvement = f" ({diff:+.4f})"  # Más negativo = peor

        print(f"  {metric:20s}: {value:.4f}{improvement}")

    # Estrategia con position sizing dinámico
    print("\n" + "="*80)
    print("3. ESTRATEGIA DONCHIAN + RÉGIMEN + POSITION SIZING")
    print("="*80)

    position_sizes = dynamic_position_sizing(filtered_signals, data['pattern_close'], trans_matrix)
    sized_returns = calculate_returns(data, filtered_signals, position_sizes)
    sized_metrics = calculate_metrics(sized_returns)

    print("\nMétricas:")
    for metric, value in sized_metrics.items():
        improvement = ""
        if metric in baseline_metrics:
            if metric in ['Sharpe Ratio', 'Total Return', 'Annual Return', 'Win Rate']:
                diff = value - baseline_metrics[metric]
                improvement = f" ({diff:+.4f})"
            elif metric == 'Max Drawdown':
                diff = value - baseline_metrics[metric]
                improvement = f" ({diff:+.4f})"

        print(f"  {metric:20s}: {value:.4f}{improvement}")

    # Comparación visual
    print("\n" + "="*80)
    print("GENERANDO GRÁFICOS COMPARATIVOS...")
    print("="*80)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Cumulative returns
    cumulative_baseline = (1 + donchian_returns).cumprod()
    cumulative_filtered = (1 + filtered_returns).cumprod()
    cumulative_sized = (1 + sized_returns).cumprod()

    ax1 = axes[0]
    ax1.plot(cumulative_baseline.index, cumulative_baseline, label='Donchian Baseline', linewidth=1.5)
    ax1.plot(cumulative_filtered.index, cumulative_filtered, label='Donchian + Filtro Régimen', linewidth=1.5)
    ax1.plot(cumulative_sized.index, cumulative_sized, label='Donchian + Régimen + Sizing', linewidth=1.5)
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.set_title('Comparación de Retornos Acumulados', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    dd_baseline = (cumulative_baseline / cumulative_baseline.expanding().max() - 1)
    dd_filtered = (cumulative_filtered / cumulative_filtered.expanding().max() - 1)
    dd_sized = (cumulative_sized / cumulative_sized.expanding().max() - 1)

    ax2.fill_between(dd_baseline.index, 0, dd_baseline, alpha=0.3, label='Baseline')
    ax2.fill_between(dd_filtered.index, 0, dd_filtered, alpha=0.3, label='+ Filtro')
    ax2.fill_between(dd_sized.index, 0, dd_sized, alpha=0.3, label='+ Sizing')
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.set_title('Comparación de Drawdowns', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Rolling Sharpe
    ax3 = axes[2]
    window_sharpe = 720  # 30 días

    sharpe_baseline = (donchian_returns.rolling(window_sharpe).mean() /
                      donchian_returns.rolling(window_sharpe).std() * np.sqrt(365*24))
    sharpe_filtered = (filtered_returns.rolling(window_sharpe).mean() /
                      filtered_returns.rolling(window_sharpe).std() * np.sqrt(365*24))
    sharpe_sized = (sized_returns.rolling(window_sharpe).mean() /
                   sized_returns.rolling(window_sharpe).std() * np.sqrt(365*24))

    ax3.plot(sharpe_baseline.index, sharpe_baseline, label='Baseline', alpha=0.7, linewidth=1)
    ax3.plot(sharpe_filtered.index, sharpe_filtered, label='+ Filtro', alpha=0.7, linewidth=1)
    ax3.plot(sharpe_sized.index, sharpe_sized, label='+ Sizing', alpha=0.7, linewidth=1)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Fecha', fontsize=11)
    ax3.set_ylabel('Rolling Sharpe Ratio (30d)', fontsize=11)
    ax3.set_title('Comparación de Sharpe Ratios Rolling', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight')

    print("\nGráfico guardado: results/strategy_comparison.png")

    # Análisis de señales filtradas
    print("\n" + "="*80)
    print("ANÁLISIS DE FILTRADO")
    print("="*80)

    total_signals = (donchian_signals != 0).sum()
    filtered_out = total_signals - (filtered_signals != 0).sum()

    print(f"\nSeñales originales: {total_signals}")
    print(f"Señales filtradas: {(filtered_signals != 0).sum()}")
    print(f"Señales eliminadas: {filtered_out} ({filtered_out/total_signals*100:.1f}%)")

    # Desglose por tipo
    long_signals = (donchian_signals > 0).sum()
    short_signals = (donchian_signals < 0).sum()
    long_filtered = long_signals - (filtered_signals > 0).sum()
    short_filtered = short_signals - (filtered_signals < 0).sum()

    print(f"\nSeñales LONG eliminadas: {long_filtered}/{long_signals} ({long_filtered/long_signals*100:.1f}%)")
    print(f"Señales SHORT eliminadas: {short_filtered}/{short_signals} ({short_filtered/short_signals*100:.1f}%)")

    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)

    print("\nComparación Sharpe Ratio:")
    print(f"  Baseline:        {baseline_metrics['Sharpe Ratio']:.4f}")
    print(f"  + Filtro:        {filtered_metrics['Sharpe Ratio']:.4f} "
          f"({filtered_metrics['Sharpe Ratio'] - baseline_metrics['Sharpe Ratio']:+.4f})")
    print(f"  + Sizing:        {sized_metrics['Sharpe Ratio']:.4f} "
          f"({sized_metrics['Sharpe Ratio'] - baseline_metrics['Sharpe Ratio']:+.4f})")

    print("\nComparación Max Drawdown:")
    print(f"  Baseline:        {baseline_metrics['Max Drawdown']:.4f}")
    print(f"  + Filtro:        {filtered_metrics['Max Drawdown']:.4f} "
          f"({filtered_metrics['Max Drawdown'] - baseline_metrics['Max Drawdown']:+.4f})")
    print(f"  + Sizing:        {sized_metrics['Max Drawdown']:.4f} "
          f"({sized_metrics['Max Drawdown'] - baseline_metrics['Max Drawdown']:+.4f})")

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print("\nNOTA: Estos resultados son IN-SAMPLE. Validar en out-of-sample antes de trading real.")
