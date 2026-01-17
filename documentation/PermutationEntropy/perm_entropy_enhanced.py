import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

def ordinal_patterns(arr: np.array, d: int) -> np.array:
    """
    Convierte una serie temporal en patrones ordinales.

    Para cada ventana de d elementos, asigna un número único (0 a d!-1)
    basado en el orden relativo de los elementos.

    Args:
        arr: Array de datos de la serie temporal
        d: Dimensión del patrón (número de elementos en cada ventana)

    Returns:
        Array con el patrón ordinal para cada punto

    Ejemplo con d=3:
        Si tenemos [1.0, 2.0, 1.5], el orden relativo es [0, 2, 1]
        correspondiente al patrón donde el primero es menor, el tercero
        es medio, y el segundo es mayor.
    """
    assert(d >= 2)
    fac = math.factorial(d)
    d1 = d - 1
    mults = []
    for i in range(1, d):
        mult = fac / math.factorial(i + 1)
        mults.append(mult)

    # Crear array para patrones ordinales
    ordinals = np.empty(len(arr))
    ordinals[:] = np.nan

    for i in range(d1, len(arr)):
        dat = arr[i - d1:  i+1]
        pattern_ordinal = 0
        for l in range(1, d):
            count = 0
            for r in range(l):
                if dat[d1 - l] >= dat[d1 - r]:
                   count += 1

            pattern_ordinal += count * mults[l - 1]
        ordinals[i] = int(pattern_ordinal)

    return ordinals

def permutation_entropy(arr: np.array, d:int, mult: int) -> np.array:
    """
    Calcula la entropía de permutación normalizada (0-1).

    La entropía de permutación mide la complejidad/aleatoriedad de una serie:
    - Valor cercano a 1: alta aleatoriedad (muchos patrones diferentes)
    - Valor cercano a 0: baja aleatoriedad (pocos patrones dominantes)

    Args:
        arr: Array de datos de la serie temporal
        d: Dimensión del patrón
        mult: Multiplicador para el tamaño de ventana (lookback = d! * mult)

    Returns:
        Array con la entropía de permutación para cada punto
    """
    fac = math.factorial(d)
    lookback = fac * mult

    ent = np.empty(len(arr))
    ent[:] = np.nan
    ordinals = ordinal_patterns(arr, d)

    for i in range(lookback + d - 1, len(arr)):
        window = ordinals[i - lookback + 1 :i+1]

        # Crear distribución de frecuencias
        freqs = pd.Series(window).value_counts().to_dict()
        for j in range(fac):
            if j in freqs:
                freqs[j] = freqs[j] / lookback

        # Calcular entropía
        perm_entropy = 0.0
        for k, v in freqs.items():
            perm_entropy += v * math.log2(v)

        # Normalizar a 0-1
        perm_entropy = -1. * (1. / math.log2(fac)) * perm_entropy
        ent[i] = perm_entropy

    return ent

def pattern_frequencies(arr: np.array, d: int) -> dict:
    """
    Calcula las frecuencias relativas de cada patrón ordinal.

    Args:
        arr: Array de datos de la serie temporal
        d: Dimensión del patrón

    Returns:
        Diccionario con pattern_id -> frecuencia_relativa
    """
    ordinals = ordinal_patterns(arr, d)
    # Eliminar NaNs
    ordinals_clean = ordinals[~np.isnan(ordinals)]

    # Contar frecuencias
    counter = Counter(ordinals_clean)
    total = len(ordinals_clean)

    # Calcular frecuencias relativas
    freq_dict = {}
    fac = math.factorial(d)
    for i in range(fac):
        freq_dict[i] = counter.get(i, 0) / total

    return freq_dict

def test_uniformity(frequencies: dict) -> tuple:
    """
    Test de chi-cuadrado para verificar uniformidad de la distribución.

    H0: La distribución es uniforme (datos son ruido)
    H1: La distribución NO es uniforme (hay estructura en los datos)

    Args:
        frequencies: Diccionario con frecuencias relativas

    Returns:
        (chi2_statistic, p_value, is_uniform)
        - is_uniform: True si no podemos rechazar uniformidad (p > 0.05)
    """
    n_patterns = len(frequencies)
    observed = np.array(list(frequencies.values()))
    expected = np.ones(n_patterns) / n_patterns

    # Total de observaciones (aproximado)
    total = 1.0 / min([f for f in observed if f > 0])
    observed_counts = observed * total
    expected_counts = expected * total

    chi2_stat = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    df = n_patterns - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, p_value > 0.05

def plot_pattern_frequencies(frequencies: dict, title: str, save_path: str = None):
    """
    Grafica las frecuencias de los patrones ordinales.

    Args:
        frequencies: Diccionario con frecuencias relativas
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    n_patterns = len(frequencies)
    expected_freq = 1.0 / n_patterns

    # Preparar datos
    patterns = sorted(frequencies.keys())
    freqs = [frequencies[p] for p in patterns]

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))

    # Barras
    bars = ax.bar(patterns, freqs, alpha=0.7, color='steelblue', edgecolor='black')

    # Línea de uniformidad esperada
    ax.axhline(y=expected_freq, color='red', linestyle='--',
               linewidth=2, label=f'Distribución Uniforme ({expected_freq:.4f})')

    # Test de uniformidad
    chi2, p_val, is_uniform = test_uniformity(frequencies)

    # Colorear barras según desviación
    for i, bar in enumerate(bars):
        if freqs[i] > expected_freq * 1.2:
            bar.set_color('darkgreen')
        elif freqs[i] < expected_freq * 0.8:
            bar.set_color('darkred')

    ax.set_xlabel('Patrón Ordinal', fontsize=12)
    ax.set_ylabel('Frecuencia Relativa', fontsize=12)
    ax.set_title(f'{title}\n' +
                 f'Chi2={chi2:.2f}, p-value={p_val:.2e} ' +
                 f'{"(Uniforme)" if is_uniform else "(NO Uniforme - Hay Estructura!)"}',
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Añadir valores encima de barras que se desvían mucho
    for i, (pattern, freq) in enumerate(zip(patterns, freqs)):
        if abs(freq - expected_freq) > expected_freq * 0.3:
            ax.text(pattern, freq, f'{freq:.4f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def interpret_pattern(pattern_id: int, d: int) -> str:
    """
    Interpreta un patrón ordinal (para d=3 especialmente).

    Args:
        pattern_id: ID del patrón (0 a d!-1)
        d: Dimensión del patrón

    Returns:
        Descripción textual del patrón
    """
    if d == 3:
        interpretations = {
            0: "↓↓ (Descenso continuo)",
            1: "↓→ (Descenso luego estable/sube)",
            2: "→↓ (Estable luego baja)",
            3: "V (Baja-Sube)",
            4: "→↑ (Estable luego sube)",
            5: "↑↑ (Ascenso continuo)"
        }
        return interpretations.get(pattern_id, f"Patrón {pattern_id}")
    else:
        return f"Patrón {pattern_id}"

if __name__ == '__main__':
    # Cargar datos
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    # Eliminar primera fila si tiene NaNs
    data = data.dropna()

    print("="*80)
    print("ANÁLISIS DE PATRONES ORDINALES EN BITCOIN (PERMUTATION ENTROPY)")
    print("="*80)
    print(f"\nDataset: {len(data)} velas horarias")
    print(f"Período: {data.index[0]} a {data.index[-1]}")

    # Parámetros
    d = 3  # Dimensión (miramos 3 velas consecutivas)
    mult = 28  # Multiplicador para ventana de entropía

    print(f"\nParámetros:")
    print(f"  - Dimensión (d): {d}")
    print(f"  - Número de patrones posibles: {math.factorial(d)}")
    print(f"  - Ventana para entropía: {math.factorial(d) * mult} velas")

    # Calcular patrones ordinales
    print("\n" + "="*80)
    print("CALCULANDO PATRONES ORDINALES...")
    print("="*80)

    data['pattern_close'] = ordinal_patterns(data['close'].to_numpy(), d)
    data['pattern_volume'] = ordinal_patterns(data['volume'].to_numpy(), d)

    # Calcular frecuencias
    freq_close = pattern_frequencies(data['close'].to_numpy(), d)
    freq_volume = pattern_frequencies(data['volume'].to_numpy(), d)

    # Mostrar frecuencias
    print("\nFRECUENCIAS RELATIVAS - CLOSE:")
    print("-" * 60)
    for pattern_id in sorted(freq_close.keys()):
        freq = freq_close[pattern_id]
        expected = 1.0 / math.factorial(d)
        deviation = (freq - expected) / expected * 100
        print(f"  Patrón {pattern_id} {interpret_pattern(pattern_id, d):30s}: "
              f"{freq:.6f} ({deviation:+.2f}%)")

    print("\nFRECUENCIAS RELATIVAS - VOLUME:")
    print("-" * 60)
    for pattern_id in sorted(freq_volume.keys()):
        freq = freq_volume[pattern_id]
        expected = 1.0 / math.factorial(d)
        deviation = (freq - expected) / expected * 100
        print(f"  Patrón {pattern_id} {interpret_pattern(pattern_id, d):30s}: "
              f"{freq:.6f} ({deviation:+.2f}%)")

    # Tests de uniformidad
    print("\n" + "="*80)
    print("TEST DE UNIFORMIDAD (Chi-cuadrado)")
    print("="*80)

    chi2_close, p_close, uniform_close = test_uniformity(freq_close)
    print(f"\nCLOSE:")
    print(f"  Chi2 = {chi2_close:.4f}")
    print(f"  p-value = {p_close:.2e}")
    print(f"  Conclusión: {'Distribución UNIFORME (datos parecen ruido)' if uniform_close else 'Distribución NO UNIFORME (HAY ESTRUCTURA!)'}")

    chi2_volume, p_volume, uniform_volume = test_uniformity(freq_volume)
    print(f"\nVOLUME:")
    print(f"  Chi2 = {chi2_volume:.4f}")
    print(f"  p-value = {p_volume:.2e}")
    print(f"  Conclusión: {'Distribución UNIFORME (datos parecen ruido)' if uniform_volume else 'Distribución NO UNIFORME (HAY ESTRUCTURA!)'}")

    # Calcular entropía de permutación
    print("\n" + "="*80)
    print("CALCULANDO ENTROPÍA DE PERMUTACIÓN...")
    print("="*80)

    data['perm_entropy_close'] = permutation_entropy(data['close'].to_numpy(), d, mult)
    data['perm_entropy_volume'] = permutation_entropy(data['volume'].to_numpy(), d, mult)

    # Guardar datos procesados
    data.to_csv('BTCUSDT3600_processed.csv')
    print("\nDatos procesados guardados en: BTCUSDT3600_processed.csv")

    # GRÁFICOS
    print("\n" + "="*80)
    print("GENERANDO GRÁFICOS...")
    print("="*80)

    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            pass  # Usar estilo por defecto

    # 1. Frecuencias de patrones - CLOSE
    print("\n1. Gráfico de frecuencias de patrones (CLOSE)...")
    fig1 = plot_pattern_frequencies(
        freq_close,
        'Frecuencias de Patrones Ordinales - CLOSE (Bitcoin)',
        'pattern_frequencies_close.png'
    )

    # 2. Frecuencias de patrones - VOLUME
    print("2. Gráfico de frecuencias de patrones (VOLUME)...")
    fig2 = plot_pattern_frequencies(
        freq_volume,
        'Frecuencias de Patrones Ordinales - VOLUME (Bitcoin)',
        'pattern_frequencies_volume.png'
    )

    # 3. Serie temporal con entropía
    print("3. Gráfico de serie temporal con entropía...")
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Close + Entropía
    ax1_twin = ax1.twinx()
    ax1.plot(data.index, data['close'], color='steelblue',
             linewidth=1, label='Close Price', alpha=0.7)
    ax1_twin.plot(data.index, data['perm_entropy_close'],
                  color='orange', linewidth=1.5,
                  label='Permutation Entropy', alpha=0.8)

    ax1.set_ylabel('Close Price (USD)', fontsize=11, color='steelblue')
    ax1_twin.set_ylabel('Permutation Entropy', fontsize=11, color='orange')
    ax1.set_title('Bitcoin Close Price y Permutation Entropy', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Volume + Entropía
    ax2_twin = ax2.twinx()
    ax2.plot(data.index, data['volume'], color='green',
             linewidth=1, label='Volume', alpha=0.7)
    ax2_twin.plot(data.index, data['perm_entropy_volume'],
                  color='red', linewidth=1.5,
                  label='Permutation Entropy (Volume)', alpha=0.8)

    ax2.set_xlabel('Fecha', fontsize=11)
    ax2.set_ylabel('Volume', fontsize=11, color='green')
    ax2_twin.set_ylabel('Permutation Entropy', fontsize=11, color='red')
    ax2.set_title('Bitcoin Volume y Permutation Entropy', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('timeseries_entropy.png', dpi=300, bbox_inches='tight')

    # 4. Distribución de patrones en el tiempo
    print("4. Gráfico de evolución de patrones en el tiempo...")
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Close patterns
    for pattern_id in range(math.factorial(d)):
        mask = data['pattern_close'] == pattern_id
        pattern_counts = data[mask].resample('W').size()
        pattern_freq = pattern_counts / data.resample('W').size()
        ax1.plot(pattern_freq.index, pattern_freq,
                label=f'P{pattern_id}', alpha=0.7, linewidth=1)

    ax1.set_ylabel('Frecuencia Relativa', fontsize=11)
    ax1.set_title('Evolución Temporal de Patrones - CLOSE', fontsize=13, fontweight='bold')
    ax1.legend(ncol=6, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1/math.factorial(d), color='red', linestyle='--',
                linewidth=1, alpha=0.5, label='Uniforme')

    # Volume patterns
    for pattern_id in range(math.factorial(d)):
        mask = data['pattern_volume'] == pattern_id
        pattern_counts = data[mask].resample('W').size()
        pattern_freq = pattern_counts / data.resample('W').size()
        ax2.plot(pattern_freq.index, pattern_freq,
                label=f'P{pattern_id}', alpha=0.7, linewidth=1)

    ax2.set_xlabel('Fecha', fontsize=11)
    ax2.set_ylabel('Frecuencia Relativa', fontsize=11)
    ax2.set_title('Evolución Temporal de Patrones - VOLUME', fontsize=13, fontweight='bold')
    ax2.legend(ncol=6, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1/math.factorial(d), color='red', linestyle='--',
                linewidth=1, alpha=0.5, label='Uniforme')

    plt.tight_layout()
    plt.savefig('pattern_evolution.png', dpi=300, bbox_inches='tight')

    print("\n" + "="*80)
    print("GRÁFICOS GUARDADOS:")
    print("="*80)
    print("  - pattern_frequencies_close.png")
    print("  - pattern_frequencies_volume.png")
    print("  - timeseries_entropy.png")
    print("  - pattern_evolution.png")

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print("\nNOTA: Los gráficos se han guardado como imágenes PNG.")
    print("Usa 'plt.show()' si quieres visualizarlos interactivamente.")
