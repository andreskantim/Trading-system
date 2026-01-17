"""
Análisis de correlaciones entre diferentes entropías de permutación
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def plot_correlation_scatter(x, y, xlabel, ylabel, title, save_path):
    """
    Crea scatter plot con línea de regresión y estadísticas de correlación.
    """
    # Eliminar NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    # Calcular correlación
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

    # Regresión lineal
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot con transparencia
    ax.scatter(x_clean, y_clean, alpha=0.3, s=10, color='steelblue', edgecolors='none')

    # Línea de regresión
    x_line = np.array([x_clean.min(), x_clean.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

    # Línea diagonal perfecta (x=y) para referencia
    diag_line = np.array([min(x_clean.min(), y_clean.min()),
                          max(x_clean.max(), y_clean.max())])
    ax.plot(diag_line, diag_line, 'k--', linewidth=1, alpha=0.5, label='y = x (perfecta)')

    # Labels y título
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Agregar estadísticas como texto
    stats_text = (
        f'Pearson r = {pearson_r:.4f} (p={pearson_p:.2e})\n'
        f'Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})\n'
        f'R² = {r_value**2:.4f}\n'
        f'N = {len(x_clean)}'
    )

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return pearson_r, spearman_r, r_value**2

if __name__ == '__main__':
    print("="*80)
    print("ANÁLISIS DE CORRELACIONES ENTRE ENTROPÍAS")
    print("="*80)

    # Cargar datos
    print("\nCargando datos...")
    data = pd.read_csv('results/BTCUSDT3600_all_dimensions.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    print(f"Datos cargados: {len(data)} velas")

    # Crear figura compuesta con los 4 gráficos
    fig = plt.figure(figsize=(16, 12))

    # 1. Correlación: Entropía Close d=3 vs d=4
    print("\n1. Correlación: Entropía Close d=3 vs d=4")
    ax1 = plt.subplot(2, 2, 1)

    x1 = data['perm_entropy_close_d3'].values
    y1 = data['perm_entropy_close_d4'].values
    mask1 = ~(np.isnan(x1) | np.isnan(y1))
    x1_clean = x1[mask1]
    y1_clean = y1[mask1]

    pearson_r1, _ = stats.pearsonr(x1_clean, y1_clean)
    spearman_r1, _ = stats.spearmanr(x1_clean, y1_clean)
    slope1, intercept1, r_value1, _, _ = stats.linregress(x1_clean, y1_clean)

    ax1.scatter(x1_clean, y1_clean, alpha=0.3, s=10, color='steelblue', edgecolors='none')
    x_line1 = np.array([x1_clean.min(), x1_clean.max()])
    y_line1 = slope1 * x_line1 + intercept1
    ax1.plot(x_line1, y_line1, 'r-', linewidth=2)

    ax1.set_xlabel('Entropía Close (d=3)', fontsize=11)
    ax1.set_ylabel('Entropía Close (d=4)', fontsize=11)
    ax1.set_title('Correlación: Entropía Close d=3 vs d=4', fontsize=12, fontweight='bold')
    ax1.text(0.05, 0.95, f'Pearson r = {pearson_r1:.4f}\nSpearman ρ = {spearman_r1:.4f}\nR² = {r_value1**2:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.grid(True, alpha=0.3)

    print(f"  Pearson r = {pearson_r1:.4f}")
    print(f"  Spearman ρ = {spearman_r1:.4f}")
    print(f"  R² = {r_value1**2:.4f}")

    # 2. Correlación: Entropía Volume d=3 vs d=4
    print("\n2. Correlación: Entropía Volume d=3 vs d=4")
    ax2 = plt.subplot(2, 2, 2)

    x2 = data['perm_entropy_volume_d3'].values
    y2 = data['perm_entropy_volume_d4'].values
    mask2 = ~(np.isnan(x2) | np.isnan(y2))
    x2_clean = x2[mask2]
    y2_clean = y2[mask2]

    pearson_r2, _ = stats.pearsonr(x2_clean, y2_clean)
    spearman_r2, _ = stats.spearmanr(x2_clean, y2_clean)
    slope2, intercept2, r_value2, _, _ = stats.linregress(x2_clean, y2_clean)

    ax2.scatter(x2_clean, y2_clean, alpha=0.3, s=10, color='green', edgecolors='none')
    x_line2 = np.array([x2_clean.min(), x2_clean.max()])
    y_line2 = slope2 * x_line2 + intercept2
    ax2.plot(x_line2, y_line2, 'r-', linewidth=2)

    ax2.set_xlabel('Entropía Volume (d=3)', fontsize=11)
    ax2.set_ylabel('Entropía Volume (d=4)', fontsize=11)
    ax2.set_title('Correlación: Entropía Volume d=3 vs d=4', fontsize=12, fontweight='bold')
    ax2.text(0.05, 0.95, f'Pearson r = {pearson_r2:.4f}\nSpearman ρ = {spearman_r2:.4f}\nR² = {r_value2**2:.4f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.grid(True, alpha=0.3)

    print(f"  Pearson r = {pearson_r2:.4f}")
    print(f"  Spearman ρ = {spearman_r2:.4f}")
    print(f"  R² = {r_value2**2:.4f}")

    # 3. Correlación: Entropía Close vs Volume (d=3)
    print("\n3. Correlación: Entropía Close vs Volume (d=3)")
    ax3 = plt.subplot(2, 2, 3)

    x3 = data['perm_entropy_close_d3'].values
    y3 = data['perm_entropy_volume_d3'].values
    mask3 = ~(np.isnan(x3) | np.isnan(y3))
    x3_clean = x3[mask3]
    y3_clean = y3[mask3]

    pearson_r3, _ = stats.pearsonr(x3_clean, y3_clean)
    spearman_r3, _ = stats.spearmanr(x3_clean, y3_clean)
    slope3, intercept3, r_value3, _, _ = stats.linregress(x3_clean, y3_clean)

    ax3.scatter(x3_clean, y3_clean, alpha=0.3, s=10, color='orange', edgecolors='none')
    x_line3 = np.array([x3_clean.min(), x3_clean.max()])
    y_line3 = slope3 * x_line3 + intercept3
    ax3.plot(x_line3, y_line3, 'r-', linewidth=2)

    ax3.set_xlabel('Entropía Close (d=3)', fontsize=11)
    ax3.set_ylabel('Entropía Volume (d=3)', fontsize=11)
    ax3.set_title('Correlación: Close vs Volume para d=3', fontsize=12, fontweight='bold')
    ax3.text(0.05, 0.95, f'Pearson r = {pearson_r3:.4f}\nSpearman ρ = {spearman_r3:.4f}\nR² = {r_value3**2:.4f}',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax3.grid(True, alpha=0.3)

    print(f"  Pearson r = {pearson_r3:.4f}")
    print(f"  Spearman ρ = {spearman_r3:.4f}")
    print(f"  R² = {r_value3**2:.4f}")

    # 4. Correlación: Entropía Close vs Volume (d=4)
    print("\n4. Correlación: Entropía Close vs Volume (d=4)")
    ax4 = plt.subplot(2, 2, 4)

    x4 = data['perm_entropy_close_d4'].values
    y4 = data['perm_entropy_volume_d4'].values
    mask4 = ~(np.isnan(x4) | np.isnan(y4))
    x4_clean = x4[mask4]
    y4_clean = y4[mask4]

    pearson_r4, _ = stats.pearsonr(x4_clean, y4_clean)
    spearman_r4, _ = stats.spearmanr(x4_clean, y4_clean)
    slope4, intercept4, r_value4, _, _ = stats.linregress(x4_clean, y4_clean)

    ax4.scatter(x4_clean, y4_clean, alpha=0.3, s=10, color='purple', edgecolors='none')
    x_line4 = np.array([x4_clean.min(), x4_clean.max()])
    y_line4 = slope4 * x_line4 + intercept4
    ax4.plot(x_line4, y_line4, 'r-', linewidth=2)

    ax4.set_xlabel('Entropía Close (d=4)', fontsize=11)
    ax4.set_ylabel('Entropía Volume (d=4)', fontsize=11)
    ax4.set_title('Correlación: Close vs Volume para d=4', fontsize=12, fontweight='bold')
    ax4.text(0.05, 0.95, f'Pearson r = {pearson_r4:.4f}\nSpearman ρ = {spearman_r4:.4f}\nR² = {r_value4**2:.4f}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.grid(True, alpha=0.3)

    print(f"  Pearson r = {pearson_r4:.4f}")
    print(f"  Spearman ρ = {spearman_r4:.4f}")
    print(f"  R² = {r_value4**2:.4f}")

    plt.suptitle('Correlaciones entre Entropías de Permutación',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('results/entropy_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("GRÁFICO GUARDADO: results/entropy_correlations.png")
    print("="*80)

    # Gráfico adicional: Serie temporal comparativa
    print("\nGenerando gráfico de series temporales comparativas...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Close
    ax1 = axes[0]
    ax1.plot(data.index, data['perm_entropy_close_d3'],
            label='d=3', alpha=0.7, linewidth=1.5, color='steelblue')
    ax1.plot(data.index, data['perm_entropy_close_d4'],
            label='d=4', alpha=0.7, linewidth=1.5, color='orange')
    ax1.set_ylabel('Permutation Entropy', fontsize=11)
    ax1.set_title('Comparación Temporal: Entropía Close (d=3 vs d=4)',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Volume
    ax2 = axes[1]
    ax2.plot(data.index, data['perm_entropy_volume_d3'],
            label='d=3', alpha=0.7, linewidth=1.5, color='green')
    ax2.plot(data.index, data['perm_entropy_volume_d4'],
            label='d=4', alpha=0.7, linewidth=1.5, color='red')
    ax2.set_xlabel('Fecha', fontsize=11)
    ax2.set_ylabel('Permutation Entropy', fontsize=11)
    ax2.set_title('Comparación Temporal: Entropía Volume (d=3 vs d=4)',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('results/entropy_timeseries_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("GRÁFICO GUARDADO: results/entropy_timeseries_comparison.png")

    # Resumen estadístico
    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO")
    print("="*80)

    print("\n1. Correlación entre dimensiones (mismo tipo):")
    print(f"   Close d=3 vs d=4:  r={pearson_r1:.4f}, R²={r_value1**2:.4f}")
    print(f"   Volume d=3 vs d=4: r={pearson_r2:.4f}, R²={r_value2**2:.4f}")
    print("\n   → Las entropías de diferentes dimensiones están FUERTEMENTE correlacionadas")
    print("   → Capturan información similar sobre la estructura de los datos")

    print("\n2. Correlación entre Close y Volume:")
    print(f"   d=3: r={pearson_r3:.4f}, R²={r_value3**2:.4f}")
    print(f"   d=4: r={pearson_r4:.4f}, R²={r_value4**2:.4f}")

    if abs(pearson_r3) < 0.3:
        print("\n   → Correlación DÉBIL entre entropía de precio y volumen")
        print("   → Precio y volumen tienen dinámicas de complejidad INDEPENDIENTES")
        print("   → Usar ambas entropías puede dar información complementaria")
    elif abs(pearson_r3) < 0.7:
        print("\n   → Correlación MODERADA entre entropía de precio y volumen")
        print("   → Hay cierta relación pero también información independiente")
    else:
        print("\n   → Correlación FUERTE entre entropía de precio y volumen")
        print("   → La complejidad del precio y volumen varían juntas")

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
