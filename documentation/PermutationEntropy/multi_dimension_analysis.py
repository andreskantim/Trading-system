"""
Análisis de Patrones Ordinales Multi-dimensional

Genera análisis para d=3, d=4, y d=5 con escalas logarítmicas
"""

import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from perm_entropy_enhanced import (ordinal_patterns, permutation_entropy,
                                   pattern_frequencies, test_uniformity,
                                   plot_pattern_frequencies)

def analyze_dimension(data: pd.DataFrame, d: int, mult: int = 28, full_analysis: bool = True):
    """
    Analiza patrones ordinales para una dimensión específica.

    Args:
        data: DataFrame con datos OHLCV
        d: Dimensión de patrones
        mult: Multiplicador para ventana de entropía
        full_analysis: Si False, solo genera gráficos de entropía
    """
    n_patterns = math.factorial(d)

    print("\n" + "="*80)
    print(f"ANÁLISIS CON DIMENSIÓN d={d}")
    print("="*80)
    print(f"Número de patrones posibles: {n_patterns}")

    # Calcular patrones ordinales
    print(f"\nCalculando patrones ordinales (d={d})...")
    patterns_close = ordinal_patterns(data['close'].to_numpy(), d)
    patterns_volume = ordinal_patterns(data['volume'].to_numpy(), d)

    # Calcular entropía de permutación
    print(f"Calculando entropía de permutación (d={d}, mult={mult})...")
    perm_entropy_close = permutation_entropy(data['close'].to_numpy(), d, mult)
    perm_entropy_volume = permutation_entropy(data['volume'].to_numpy(), d, mult)

    # GRÁFICOS
    print(f"\nGenerando gráficos para d={d}...")

    if full_analysis:
        # 1. Frecuencias de patrones - CLOSE
        freq_close = pattern_frequencies(data['close'].to_numpy(), d)
        chi2_close, p_close, uniform_close = test_uniformity(freq_close)

        fig1 = plot_pattern_frequencies(
            freq_close,
            f'Frecuencias de Patrones Ordinales - CLOSE (d={d}, {n_patterns} patrones)\n'
            f'Chi2={chi2_close:.2f}, p={p_close:.2e}',
            f'results/pattern_frequencies_close_d{d}.png'
        )
        plt.close(fig1)

        # 2. Frecuencias de patrones - VOLUME
        freq_volume = pattern_frequencies(data['volume'].to_numpy(), d)
        chi2_volume, p_volume, uniform_volume = test_uniformity(freq_volume)

        fig2 = plot_pattern_frequencies(
            freq_volume,
            f'Frecuencias de Patrones Ordinales - VOLUME (d={d}, {n_patterns} patrones)\n'
            f'Chi2={chi2_volume:.2f}, p={p_volume:.2e}',
            f'results/pattern_frequencies_volume_d{d}.png'
        )
        plt.close(fig2)

        print(f"  ✓ Gráficos de frecuencias guardados")

        # Mostrar estadísticas
        print(f"\n  Test de uniformidad - CLOSE:")
        print(f"    Chi2 = {chi2_close:.4f}, p-value = {p_close:.2e}")
        print(f"    {'UNIFORME' if uniform_close else 'NO UNIFORME (HAY ESTRUCTURA!)'}")

        print(f"\n  Test de uniformidad - VOLUME:")
        print(f"    Chi2 = {chi2_volume:.4f}, p-value = {p_volume:.2e}")
        print(f"    {'UNIFORME' if uniform_volume else 'NO UNIFORME (HAY ESTRUCTURA!)'}")

    # 3. Serie temporal con entropía (SIEMPRE SE GENERA)
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Close + Entropía (ESCALA LOG)
    ax1_twin = ax1.twinx()

    # Precio en escala logarítmica
    ax1.plot(data.index, data['close'], color='steelblue',
             linewidth=1, label='Close Price', alpha=0.7)
    ax1.set_yscale('log')

    # Entropía en escala lineal
    ax1_twin.plot(data.index, perm_entropy_close,
                  color='orange', linewidth=1.5,
                  label='Permutation Entropy', alpha=0.8)

    ax1.set_ylabel('Close Price (USD) - LOG SCALE', fontsize=11, color='steelblue')
    ax1_twin.set_ylabel('Permutation Entropy', fontsize=11, color='orange')
    ax1.set_title(f'Bitcoin Close Price y Permutation Entropy (d={d})',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Subplot 2: Volume + Entropía (ESCALA LOG)
    ax2_twin = ax2.twinx()

    # Volumen en escala logarítmica
    ax2.plot(data.index, data['volume'], color='green',
             linewidth=1, label='Volume', alpha=0.7)
    ax2.set_yscale('log')

    # Entropía en escala lineal
    ax2_twin.plot(data.index, perm_entropy_volume,
                  color='red', linewidth=1.5,
                  label='Permutation Entropy (Volume)', alpha=0.8)

    ax2.set_xlabel('Fecha', fontsize=11)
    ax2.set_ylabel('Volume - LOG SCALE', fontsize=11, color='green')
    ax2_twin.set_ylabel('Permutation Entropy', fontsize=11, color='red')
    ax2.set_title(f'Bitcoin Volume y Permutation Entropy (d={d})',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f'results/timeseries_entropy_d{d}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico de serie temporal guardado")

    # Guardar datos procesados
    result_data = data.copy()
    result_data[f'pattern_close_d{d}'] = patterns_close
    result_data[f'pattern_volume_d{d}'] = patterns_volume
    result_data[f'perm_entropy_close_d{d}'] = perm_entropy_close
    result_data[f'perm_entropy_volume_d{d}'] = perm_entropy_volume

    return result_data

if __name__ == '__main__':
    print("="*80)
    print("ANÁLISIS MULTI-DIMENSIONAL DE PATRONES ORDINALES")
    print("="*80)
    print("\nNOTA: d! = número de patrones posibles")
    print("  d=3 → 3! = 6 patrones")
    print("  d=4 → 4! = 24 patrones")
    print("  d=5 → 5! = 120 patrones")

    # Cargar datos
    print("\nCargando datos...")
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.dropna()

    print(f"Datos cargados: {len(data)} velas horarias")
    print(f"Período: {data.index[0]} a {data.index[-1]}")

    # d=3: Análisis completo
    print("\n" + "="*80)
    print("DIMENSIÓN d=3 (ANÁLISIS COMPLETO)")
    print("="*80)
    data_d3 = analyze_dimension(data, d=3, mult=28, full_analysis=True)

    # d=4: Análisis completo
    print("\n" + "="*80)
    print("DIMENSIÓN d=4 (ANÁLISIS COMPLETO)")
    print("="*80)
    data_d4 = analyze_dimension(data, d=4, mult=28, full_analysis=True)

    # d=5: Solo entropía (muchos patrones para analizar individualmente)
    print("\n" + "="*80)
    print("DIMENSIÓN d=5 (SOLO ENTROPÍA)")
    print("="*80)
    print("NOTA: Con 120 patrones, solo generamos gráficos de entropía")
    data_d5 = analyze_dimension(data, d=5, mult=28, full_analysis=False)

    # Comparación de entropías entre dimensiones
    print("\n" + "="*80)
    print("COMPARACIÓN ENTRE DIMENSIONES")
    print("="*80)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Close
    ax1.plot(data_d3.index, data_d3['perm_entropy_close_d3'],
            label='d=3 (6 patrones)', alpha=0.8, linewidth=1.5)
    ax1.plot(data_d4.index, data_d4['perm_entropy_close_d4'],
            label='d=4 (24 patrones)', alpha=0.8, linewidth=1.5)
    ax1.plot(data_d5.index, data_d5['perm_entropy_close_d5'],
            label='d=5 (120 patrones)', alpha=0.8, linewidth=1.5)

    ax1.set_ylabel('Permutation Entropy', fontsize=11)
    ax1.set_title('Comparación de Entropía de Permutación - CLOSE',
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Volume
    ax2.plot(data_d3.index, data_d3['perm_entropy_volume_d3'],
            label='d=3 (6 patrones)', alpha=0.8, linewidth=1.5)
    ax2.plot(data_d4.index, data_d4['perm_entropy_volume_d4'],
            label='d=4 (24 patrones)', alpha=0.8, linewidth=1.5)
    ax2.plot(data_d5.index, data_d5['perm_entropy_volume_d5'],
            label='d=5 (120 patrones)', alpha=0.8, linewidth=1.5)

    ax2.set_xlabel('Fecha', fontsize=11)
    ax2.set_ylabel('Permutation Entropy', fontsize=11)
    ax2.set_title('Comparación de Entropía de Permutación - VOLUME',
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('results/entropy_comparison_all_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Gráfico comparativo guardado: results/entropy_comparison_all_dimensions.png")

    # Estadísticas comparativas
    print("\n" + "="*80)
    print("ESTADÍSTICAS COMPARATIVAS")
    print("="*80)

    data_by_d = {3: data_d3, 4: data_d4, 5: data_d5}

    for d in [3, 4, 5]:
        close_col = f'perm_entropy_close_d{d}'
        volume_col = f'perm_entropy_volume_d{d}'
        current_data = data_by_d[d]

        print(f"\nd={d} ({math.factorial(d)} patrones):")
        print(f"  Entropía CLOSE:")
        print(f"    Media: {current_data[close_col].mean():.4f}")
        print(f"    Std:   {current_data[close_col].std():.4f}")
        print(f"    Min:   {current_data[close_col].min():.4f}")
        print(f"    Max:   {current_data[close_col].max():.4f}")

        print(f"  Entropía VOLUME:")
        print(f"    Media: {current_data[volume_col].mean():.4f}")
        print(f"    Std:   {current_data[volume_col].std():.4f}")
        print(f"    Min:   {current_data[volume_col].min():.4f}")
        print(f"    Max:   {current_data[volume_col].max():.4f}")

    # Guardar datos combinados
    print("\n" + "="*80)
    print("GUARDANDO DATOS PROCESADOS")
    print("="*80)

    # Combinar todas las columnas
    combined_data = data.copy()

    # Añadir columnas de d=3
    for col in data_d3.columns:
        if 'd3' in col:
            combined_data[col] = data_d3[col]

    # Añadir columnas de d=4
    for col in data_d4.columns:
        if 'd4' in col:
            combined_data[col] = data_d4[col]

    # Añadir columnas de d=5
    for col in data_d5.columns:
        if 'd5' in col:
            combined_data[col] = data_d5[col]

    combined_data.to_csv('results/BTCUSDT3600_all_dimensions.csv')
    print("\n✓ Datos guardados: results/BTCUSDT3600_all_dimensions.csv")

    print("\n" + "="*80)
    print("ARCHIVOS GENERADOS")
    print("="*80)
    print("\nd=3 (6 patrones):")
    print("  - results/pattern_frequencies_close_d3.png")
    print("  - results/pattern_frequencies_volume_d3.png")
    print("  - results/timeseries_entropy_d3.png")

    print("\nd=4 (24 patrones):")
    print("  - results/pattern_frequencies_close_d4.png")
    print("  - results/pattern_frequencies_volume_d4.png")
    print("  - results/timeseries_entropy_d4.png")

    print("\nd=5 (120 patrones):")
    print("  - results/timeseries_entropy_d5.png (solo entropía)")

    print("\nComparación:")
    print("  - results/entropy_comparison_all_dimensions.png")

    print("\nDatos:")
    print("  - results/BTCUSDT3600_all_dimensions.csv")

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print("\nNOTA: Todos los gráficos usan escala LOGARÍTMICA para precio y volumen")
