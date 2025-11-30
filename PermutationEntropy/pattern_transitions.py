"""
Análisis de transiciones entre patrones ordinales (Cadenas de Markov)

Este script analiza cómo los patrones ordinales transicionan entre sí,
lo cual puede revelar estructura predictiva en los datos.
"""

import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from perm_entropy_enhanced import ordinal_patterns, interpret_pattern

def build_transition_matrix(patterns: np.array, n_states: int) -> np.ndarray:
    """
    Construye matriz de transición de patrones.

    Args:
        patterns: Array de patrones ordinales
        n_states: Número de estados posibles (d!)

    Returns:
        Matriz de transición normalizada (n_states x n_states)
    """
    # Eliminar NaNs
    patterns_clean = patterns[~np.isnan(patterns)].astype(int)

    # Inicializar matriz de conteo
    transition_counts = np.zeros((n_states, n_states))

    # Contar transiciones
    for i in range(len(patterns_clean) - 1):
        current = patterns_clean[i]
        next_pattern = patterns_clean[i + 1]
        transition_counts[current, next_pattern] += 1

    # Normalizar filas (probabilidades condicionales)
    transition_matrix = np.zeros_like(transition_counts, dtype=float)
    for i in range(n_states):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum

    return transition_matrix

def expected_uniform_matrix(n_states: int) -> np.ndarray:
    """Matriz de transición esperada si las transiciones fueran uniformes."""
    return np.ones((n_states, n_states)) / n_states

def compute_deviation_matrix(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Calcula desviación porcentual de lo observado vs esperado."""
    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = (observed - expected) / expected * 100
        deviation[np.isnan(deviation)] = 0
        deviation[np.isinf(deviation)] = 0
    return deviation

def test_markov_property(patterns: np.array, lag: int = 1) -> float:
    """
    Test si el proceso tiene propiedad de Markov.

    Args:
        patterns: Array de patrones
        lag: Lag para el test (default=1, primera orden)

    Returns:
        Score de Markov (0 = no Markov, 1 = perfecto Markov)
    """
    patterns_clean = patterns[~np.isnan(patterns)].astype(int)

    # Calcular entropía condicional H(X_t | X_{t-1})
    # vs H(X_t | X_{t-1}, X_{t-2}, ..., X_{t-lag})

    # Para simplificar, usamos un test aproximado:
    # Si es Markov de orden 1, conocer X_{t-2} no añade info sobre X_t dado X_{t-1}

    # Por ahora retornamos placeholder
    return 0.0  # TODO: implementar test completo

def find_strongest_transitions(transition_matrix: np.ndarray, deviation_matrix: np.ndarray,
                              d: int, top_n: int = 10) -> list:
    """
    Encuentra las transiciones más fuertes (más desviadas de lo esperado).

    Args:
        transition_matrix: Matriz de transición observada
        deviation_matrix: Matriz de desviación porcentual
        d: Dimensión de patrones
        top_n: Número de transiciones a retornar

    Returns:
        Lista de tuplas (from_pattern, to_pattern, probability, deviation_pct)
    """
    n_states = transition_matrix.shape[0]
    transitions = []

    for i in range(n_states):
        for j in range(n_states):
            prob = transition_matrix[i, j]
            dev = deviation_matrix[i, j]
            if abs(dev) > 1:  # Solo considerar desviaciones > 1%
                transitions.append((i, j, prob, dev))

    # Ordenar por valor absoluto de desviación
    transitions.sort(key=lambda x: abs(x[3]), reverse=True)

    return transitions[:top_n]

def plot_transition_matrix(matrix: np.ndarray, title: str, d: int,
                          save_path: str = None, cmap='RdYlGn'):
    """
    Visualiza matriz de transición como heatmap.

    Args:
        matrix: Matriz a visualizar
        title: Título del gráfico
        d: Dimensión de patrones
        save_path: Ruta para guardar
        cmap: Colormap a usar
    """
    n_states = matrix.shape[0]

    # Labels con interpretación
    labels = [f"P{i}\n{interpret_pattern(i, d).split('(')[0]}" for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valor', rotation=270, labelpad=20)

    # Ticks y labels
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel('Patrón Siguiente', fontsize=11)
    ax.set_ylabel('Patrón Actual', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # Añadir valores en celdas
    for i in range(n_states):
        for j in range(n_states):
            value = matrix[i, j]
            if abs(value) > matrix.max() * 0.1:  # Solo mostrar valores significativos
                color = 'white' if abs(value) > matrix.max() * 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=color, fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

if __name__ == '__main__':
    print("="*80)
    print("ANÁLISIS DE TRANSICIONES DE PATRONES ORDINALES")
    print("="*80)

    # Cargar datos procesados
    import os
    # Buscar archivo procesado en varios lugares posibles
    possible_paths = [
        'results/BTCUSDT3600_processed.csv',
        'PermutationEntropy/results/BTCUSDT3600_processed.csv',
        '../results/BTCUSDT3600_processed.csv'
    ]
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError("No se encontró BTCUSDT3600_processed.csv")

    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    d = 3
    n_states = math.factorial(d)

    print(f"\nDatos cargados: {len(data)} velas")
    print(f"Dimensión: d={d}")
    print(f"Estados posibles: {n_states}")

    # Análisis para CLOSE
    print("\n" + "="*80)
    print("ANÁLISIS DE TRANSICIONES - CLOSE")
    print("="*80)

    patterns_close = data['pattern_close'].values
    trans_matrix_close = build_transition_matrix(patterns_close, n_states)
    expected_matrix = expected_uniform_matrix(n_states)
    deviation_close = compute_deviation_matrix(trans_matrix_close, expected_matrix)

    print("\nMatriz de Transición (Close):")
    print("-" * 60)
    print(pd.DataFrame(trans_matrix_close,
                      columns=[f'P{i}' for i in range(n_states)],
                      index=[f'P{i}' for i in range(n_states)]).round(4))

    print("\nTransiciones más fuertes (Close):")
    print("-" * 60)
    top_transitions_close = find_strongest_transitions(trans_matrix_close, deviation_close, d, top_n=15)
    for from_p, to_p, prob, dev in top_transitions_close:
        print(f"  {interpret_pattern(from_p, d):30s} → {interpret_pattern(to_p, d):30s}: "
              f"P={prob:.4f} ({dev:+.1f}%)")

    # Análisis para VOLUME
    print("\n" + "="*80)
    print("ANÁLISIS DE TRANSICIONES - VOLUME")
    print("="*80)

    patterns_volume = data['pattern_volume'].values
    trans_matrix_volume = build_transition_matrix(patterns_volume, n_states)
    deviation_volume = compute_deviation_matrix(trans_matrix_volume, expected_matrix)

    print("\nMatriz de Transición (Volume):")
    print("-" * 60)
    print(pd.DataFrame(trans_matrix_volume,
                      columns=[f'P{i}' for i in range(n_states)],
                      index=[f'P{i}' for i in range(n_states)]).round(4))

    print("\nTransiciones más fuertes (Volume):")
    print("-" * 60)
    top_transitions_volume = find_strongest_transitions(trans_matrix_volume, deviation_volume, d, top_n=15)
    for from_p, to_p, prob, dev in top_transitions_volume:
        print(f"  {interpret_pattern(from_p, d):30s} → {interpret_pattern(to_p, d):30s}: "
              f"P={prob:.4f} ({dev:+.1f}%)")

    # Análisis de auto-transiciones (persistencia)
    print("\n" + "="*80)
    print("PERSISTENCIA DE PATRONES (Auto-transiciones)")
    print("="*80)

    print("\nCLOSE:")
    for i in range(n_states):
        auto_trans = trans_matrix_close[i, i]
        expected = 1.0 / n_states
        dev = (auto_trans - expected) / expected * 100
        print(f"  {interpret_pattern(i, d):35s}: P(self→self)={auto_trans:.4f} ({dev:+.1f}%)")

    print("\nVOLUME:")
    for i in range(n_states):
        auto_trans = trans_matrix_volume[i, i]
        expected = 1.0 / n_states
        dev = (auto_trans - expected) / expected * 100
        print(f"  {interpret_pattern(i, d):35s}: P(self→self)={auto_trans:.4f} ({dev:+.1f}%)")

    # Calcular entropía de cada fila (predictibilidad)
    print("\n" + "="*80)
    print("PREDICTIBILIDAD DE PATRONES")
    print("="*80)
    print("(Menor entropía = más predictible siguiente patrón)")

    def row_entropy(row):
        """Calcula entropía de una fila de la matriz de transición."""
        row_clean = row[row > 0]
        if len(row_clean) == 0:
            return 0
        return -np.sum(row_clean * np.log2(row_clean))

    max_entropy = math.log2(n_states)

    print("\nCLOSE:")
    for i in range(n_states):
        ent = row_entropy(trans_matrix_close[i])
        norm_ent = ent / max_entropy
        print(f"  {interpret_pattern(i, d):35s}: Entropía={ent:.3f}/{max_entropy:.3f} "
              f"(norm={norm_ent:.3f})")

    print("\nVOLUME:")
    for i in range(n_states):
        ent = row_entropy(trans_matrix_volume[i])
        norm_ent = ent / max_entropy
        print(f"  {interpret_pattern(i, d):35s}: Entropía={ent:.3f}/{max_entropy:.3f} "
              f"(norm={norm_ent:.3f})")

    # GRÁFICOS
    print("\n" + "="*80)
    print("GENERANDO GRÁFICOS DE TRANSICIONES...")
    print("="*80)

    # 1. Matriz de transición CLOSE
    print("\n1. Matriz de transición (CLOSE)...")
    fig1 = plot_transition_matrix(
        trans_matrix_close,
        'Matriz de Transición de Patrones - CLOSE',
        d,
        'results/transition_matrix_close.png',
        cmap='YlOrRd'
    )

    # 2. Matriz de desviación CLOSE
    print("2. Matriz de desviación (CLOSE)...")
    # Limitar desviaciones para mejor visualización
    deviation_close_clipped = np.clip(deviation_close, -50, 50)
    fig2 = plot_transition_matrix(
        deviation_close_clipped,
        'Desviación de Transiciones (%) - CLOSE\n(valores limitados a ±50%)',
        d,
        'results/transition_deviation_close.png',
        cmap='RdBu_r'
    )

    # 3. Matriz de transición VOLUME
    print("3. Matriz de transición (VOLUME)...")
    fig3 = plot_transition_matrix(
        trans_matrix_volume,
        'Matriz de Transición de Patrones - VOLUME',
        d,
        'results/transition_matrix_volume.png',
        cmap='YlOrRd'
    )

    # 4. Matriz de desviación VOLUME
    print("4. Matriz de desviación (VOLUME)...")
    deviation_volume_clipped = np.clip(deviation_volume, -50, 50)
    fig4 = plot_transition_matrix(
        deviation_volume_clipped,
        'Desviación de Transiciones (%) - VOLUME\n(valores limitados a ±50%)',
        d,
        'results/transition_deviation_volume.png',
        cmap='RdBu_r'
    )

    # 5. Comparación de auto-transiciones
    print("5. Gráfico de auto-transiciones...")
    fig5, ax = plt.subplots(figsize=(12, 6))

    patterns_idx = np.arange(n_states)
    width = 0.35

    auto_close = np.diag(trans_matrix_close)
    auto_volume = np.diag(trans_matrix_volume)
    expected_auto = 1.0 / n_states

    bars1 = ax.bar(patterns_idx - width/2, auto_close, width, label='Close', alpha=0.8)
    bars2 = ax.bar(patterns_idx + width/2, auto_volume, width, label='Volume', alpha=0.8)
    ax.axhline(y=expected_auto, color='red', linestyle='--', linewidth=2,
              label=f'Esperado (uniforme) = {expected_auto:.4f}')

    ax.set_xlabel('Patrón', fontsize=11)
    ax.set_ylabel('Probabilidad de Auto-transición', fontsize=11)
    ax.set_title('Persistencia de Patrones (P(patrón → mismo patrón))',
                fontsize=13, fontweight='bold')
    ax.set_xticks(patterns_idx)
    ax.set_xticklabels([f'P{i}\n{interpret_pattern(i, d).split("(")[0]}' for i in range(n_states)],
                       rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/auto_transitions.png', dpi=300, bbox_inches='tight')

    # Guardar matrices como CSV
    print("\n6. Guardando matrices como CSV...")
    pd.DataFrame(trans_matrix_close,
                columns=[f'To_P{i}' for i in range(n_states)],
                index=[f'From_P{i}' for i in range(n_states)]).to_csv(
                    'results/transition_matrix_close.csv')

    pd.DataFrame(trans_matrix_volume,
                columns=[f'To_P{i}' for i in range(n_states)],
                index=[f'From_P{i}' for i in range(n_states)]).to_csv(
                    'results/transition_matrix_volume.csv')

    print("\n" + "="*80)
    print("ARCHIVOS GUARDADOS:")
    print("="*80)
    print("  - results/transition_matrix_close.png")
    print("  - results/transition_deviation_close.png")
    print("  - results/transition_matrix_volume.png")
    print("  - results/transition_deviation_volume.png")
    print("  - results/auto_transitions.png")
    print("  - results/transition_matrix_close.csv")
    print("  - results/transition_matrix_volume.csv")

    print("\n" + "="*80)
    print("ANÁLISIS DE TRANSICIONES COMPLETADO")
    print("="*80)
