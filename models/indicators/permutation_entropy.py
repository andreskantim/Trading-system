"""
Indicador Permutation Entropy

Módulo independiente para cálculo y visualización de Entropía de Permutación.
Basado en la implementación de documentation/PermutationEntropy/perm_entropy.py
"""

import pandas as pd
import numpy as np
import math


def ordinal_patterns(arr: np.array, d: int) -> np.array:
    """
    Calcula los patrones ordinales de una serie temporal.

    Args:
        arr: Array numpy de valores
        d: Dimensión del patrón (número de puntos, 3 o 4)

    Returns:
        np.array con índices de patrones ordinales
    """
    assert d >= 2, "d debe ser >= 2"
    fac = math.factorial(d)
    d1 = d - 1
    mults = []
    for i in range(1, d):
        mult = fac / math.factorial(i + 1)
        mults.append(mult)

    # Create array to put ordinal pattern in
    ordinals = np.empty(len(arr))
    ordinals[:] = np.nan

    for i in range(d1, len(arr)):
        dat = arr[i - d1: i + 1]
        pattern_ordinal = 0
        for l in range(1, d):
            count = 0
            for r in range(l):
                if dat[d1 - l] >= dat[d1 - r]:
                    count += 1

            pattern_ordinal += count * mults[l - 1]
        ordinals[i] = int(pattern_ordinal)

    return ordinals


def calculate_permutation_entropy(close: pd.Series, n: int, mult: int = 14) -> pd.Series:
    """
    Calcula la Permutation Entropy de una serie de precios.

    Args:
        close: Serie de precios de cierre
        n: Número de velas previas (dimensión del patrón, típicamente 3 o 4)
        mult: Multiplicador para el lookback (default: 28)
             lookback = factorial(n) * mult

    Returns:
        pd.Series con valores de entropía normalizados (0-1)
        - Valores cercanos a 1: alta entropía (mercado aleatorio/eficiente)
        - Valores cercanos a 0: baja entropía (patrones predecibles)
    """
    d = n
    fac = math.factorial(d)
    lookback = fac * mult

    arr = close.to_numpy()
    ent = np.empty(len(arr))
    ent[:] = np.nan
    ordinals = ordinal_patterns(arr, d)

    for i in range(lookback + d - 1, len(arr)):
        window = ordinals[i - lookback + 1: i + 1]

        # Create distribution
        freqs = pd.Series(window).value_counts().to_dict()
        for j in range(fac):
            if j in freqs:
                freqs[j] = freqs[j] / lookback

        # Calculate entropy
        perm_entropy = 0.0
        for k, v in freqs.items():
            perm_entropy += v * math.log2(v)

        # Normalize to 0-1
        perm_entropy = -1. * (1. / math.log2(fac)) * perm_entropy
        ent[i] = perm_entropy

    return pd.Series(ent, index=close.index)


def visualization(ohlc: pd.DataFrame, n: int = 3, mult: int = 14):
    """
    Datos para visualización del indicador Permutation Entropy.

    Args:
        ohlc: DataFrame con OHLC data
        n: Número de velas previas (dimensión del patrón, 3 o 4)
        mult: Multiplicador para el lookback (default: 28)

    Returns:
        dict con estructura:
        {
            'indicators_in_price': {},
            'indicators_off_price': {
                'perm_entropy': {'data': pd.Series, 'color': str}
            }
        }
    """
    pe = calculate_permutation_entropy(ohlc['close'], n, mult)

    color = 'cyan' if n == 3 else 'orange'
    name = f'pe{n}'

    return {
        'indicators_in_price': {},
        'indicators_off_price': {
            name: {'data': pe, 'color': color}
        }
    }


def visualization3(ohlc: pd.DataFrame, mult: int = 14):
    """
    Permutation Entropy con 3 velas anteriores.

    Args:
        ohlc: DataFrame con OHLC data
        mult: Multiplicador para el lookback (default: 28)

    Returns:
        dict para visualización
    """
    pe3 = calculate_permutation_entropy(ohlc['close'], 3, mult)
    return {
        'indicators_in_price': {},
        'indicators_off_price': {
            'pe3': {'data': pe3, 'color': 'cyan'}
        }
    }


def visualization4(ohlc: pd.DataFrame, mult: int = 14):
    """
    Permutation Entropy con 4 velas anteriores.

    Args:
        ohlc: DataFrame con OHLC data
        mult: Multiplicador para el lookback (default: 28)

    Returns:
        dict para visualización
    """
    pe4 = calculate_permutation_entropy(ohlc['close'], 4, mult)
    return {
        'indicators_in_price': {},
        'indicators_off_price': {
            'pe4': {'data': pe4, 'color': 'orange'}
        }
    }
