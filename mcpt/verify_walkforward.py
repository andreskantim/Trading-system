"""
Script de verificación para comparar walkforward original vs moderno

Compara que ambas implementaciones generen resultados idénticos
con los mismos datos y parámetros.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.path import BITCOIN_PARQUET
from bar_permute import get_permutation
import importlib

# Cargar estrategia
strategy = importlib.import_module('strategies.donchian')

# Implementación walkforward moderna (copiada de walkforward_permutation.py)
def walkforward_modern(ohlc: pd.DataFrame, strategy, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
    """Implementación moderna"""
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            result = strategy.optimize(ohlc.iloc[i-train_lookback:i])
            best_params = result[:-1]
            tmp_signal = strategy.signal(ohlc, *best_params)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal

# Implementación walkforward original (de mcpt/old_strategies/donchian.py)
def walkforward_original(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
    """Implementación original"""
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_lookback, _ = strategy.optimize(ohlc.iloc[i-train_lookback:i])
            tmp_signal = strategy.signal(ohlc, best_lookback)
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


if __name__ == '__main__':
    print("\n" + "="*70)
    print("VERIFICACIÓN: WALKFORWARD ORIGINAL VS MODERNO")
    print("="*70 + "\n")

    # Cargar datos (mismos que el original)
    print("Cargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2016) & (df.index.year < 2021)]
    print(f"✓ Datos cargados: {len(df)} filas (2016-2020)\n")

    train_window = 24 * 365 * 4

    # Ejecutar ambas implementaciones
    print("Ejecutando walkforward original...")
    wf_original = walkforward_original(df, train_lookback=train_window)

    print("Ejecutando walkforward moderno...")
    wf_modern = walkforward_modern(df, strategy, train_lookback=train_window)

    # Comparar resultados
    print("\n" + "="*70)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*70)

    # Calcular returns
    df['r'] = np.log(df['close']).diff().shift(-1)

    rets_original = wf_original * df['r']
    rets_modern = wf_modern * df['r']

    # Profit factors
    pf_original = rets_original[rets_original > 0].sum() / rets_original[rets_original < 0].abs().sum()
    pf_modern = rets_modern[rets_modern > 0].sum() / rets_modern[rets_modern < 0].abs().sum()

    print(f"Profit Factor Original: {pf_original:.6f}")
    print(f"Profit Factor Moderno:  {pf_modern:.6f}")
    print(f"Diferencia absoluta:    {abs(pf_original - pf_modern):.10f}")

    # Comparar señales
    diff = np.sum(wf_original != wf_modern)
    print(f"\nSeñales diferentes: {diff}/{len(wf_original)}")

    if diff == 0 and abs(pf_original - pf_modern) < 1e-10:
        print("\n✅ VERIFICACIÓN EXITOSA: Ambas implementaciones son IDÉNTICAS")
    else:
        print("\n⚠️  ADVERTENCIA: Hay diferencias entre las implementaciones")
        print("   Revisar lógica de walkforward")

    print("="*70 + "\n")

    # Test con permutación
    print("Probando con 1 permutación...")
    df_perm = get_permutation(df, start_index=train_window, seed=42)

    wf_perm_original = walkforward_original(df_perm, train_lookback=train_window)
    wf_perm_modern = walkforward_modern(df_perm, strategy, train_lookback=train_window)

    df_perm['r'] = np.log(df_perm['close']).diff().shift(-1)
    pf_perm_orig = (wf_perm_original * df_perm['r']).dropna()
    pf_perm_mod = (wf_perm_modern * df_perm['r']).dropna()

    pf_o = pf_perm_orig[pf_perm_orig > 0].sum() / pf_perm_orig[pf_perm_orig < 0].abs().sum()
    pf_m = pf_perm_mod[pf_perm_mod > 0].sum() / pf_perm_mod[pf_perm_mod < 0].abs().sum()

    print(f"Permutación - PF Original: {pf_o:.6f}")
    print(f"Permutación - PF Moderno:  {pf_m:.6f}")
    print(f"Diferencia: {abs(pf_o - pf_m):.10f}")

    if abs(pf_o - pf_m) < 1e-10:
        print("✅ Test con permutación: IDÉNTICO\n")
    else:
        print("⚠️  Test con permutación: DIFERENTE\n")
