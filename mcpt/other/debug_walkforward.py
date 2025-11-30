"""
Script de debug para verificar paso a paso que walkforward_permutation.py
es idéntico a walkforward_donchian_mod.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.path import BITCOIN_PARQUET
from bar_permute import get_permutation
import importlib

# Cargar estrategia
strategy = importlib.import_module('strategies.donchian')

# Implementación walkforward moderna
def walkforward_modern(ohlc: pd.DataFrame, strategy, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
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

# Implementación walkforward original (de donchian.py)
def walkforward_original(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 60):
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
    print("DEBUG: VERIFICACIÓN PASO A PASO")
    print("="*70 + "\n")

    # Cargar datos (EXACTAMENTE como en ambos códigos)
    print("1. Cargando datos...")
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2018) & (df.index.year < 2024)]
    print(f"   ✓ Datos: {len(df)} filas\n")

    train_window = 24 * 365 * 4

    # PASO 1: Verificar walkforward REAL (sin permutación)
    print("2. Verificando walkforward REAL (sin permutación)...")

    wf_original = walkforward_original(df, train_lookback=train_window)
    wf_modern = walkforward_modern(df, strategy, train_lookback=train_window)

    df['r'] = np.log(df['close']).diff().shift(-1)

    rets_original = wf_original * df['r']
    rets_modern = wf_modern * df['r']

    pf_orig = rets_original[rets_original > 0].sum() / rets_original[rets_original < 0].abs().sum()
    pf_mod = rets_modern[rets_modern > 0].sum() / rets_modern[rets_modern < 0].abs().sum()

    print(f"   PF Original: {pf_orig:.6f}")
    print(f"   PF Moderno:  {pf_mod:.6f}")
    print(f"   Diferencia:  {abs(pf_orig - pf_mod):.10f}")

    if abs(pf_orig - pf_mod) < 1e-10:
        print("   ✅ Walkforward REAL es IDÉNTICO\n")
    else:
        print("   ⚠️  Walkforward REAL es DIFERENTE - REVISAR\n")
        sys.exit(1)

    # PASO 2: Verificar 1 permutación CON SEED
    print("3. Verificando 1 permutación CON SEED=42...")

    df_perm1_orig = get_permutation(df.copy(), start_index=train_window, seed=42)
    df_perm1_mod = get_permutation(df.copy(), start_index=train_window, seed=42)

    # Verificar que las permutaciones son idénticas
    diff_close = np.sum(df_perm1_orig['close'].values != df_perm1_mod['close'].values)
    print(f"   Diferencias en close: {diff_close}")

    if diff_close == 0:
        print("   ✓ Permutaciones con mismo seed son idénticas")
    else:
        print("   ✗ ERROR: Permutaciones con mismo seed son diferentes!")

    # Calcular walkforward en permutación
    wf_perm_orig = walkforward_original(df_perm1_orig, train_lookback=train_window)
    wf_perm_mod = walkforward_modern(df_perm1_mod, strategy, train_lookback=train_window)

    df_perm1_orig['r'] = np.log(df_perm1_orig['close']).diff().shift(-1)
    df_perm1_mod['r'] = np.log(df_perm1_mod['close']).diff().shift(-1)

    pf_perm_orig = (wf_perm_orig * df_perm1_orig['r'])
    pf_perm_mod = (wf_perm_mod * df_perm1_mod['r'])

    pf_po = pf_perm_orig[pf_perm_orig > 0].sum() / pf_perm_orig[pf_perm_orig < 0].abs().sum()
    pf_pm = pf_perm_mod[pf_perm_mod > 0].sum() / pf_perm_mod[pf_perm_mod < 0].abs().sum()

    print(f"   PF Permutación Original: {pf_po:.6f}")
    print(f"   PF Permutación Moderno:  {pf_pm:.6f}")
    print(f"   Diferencia:              {abs(pf_po - pf_pm):.10f}")

    if abs(pf_po - pf_pm) < 1e-10:
        print("   ✅ Permutación con seed da resultados IDÉNTICOS\n")
    else:
        print("   ⚠️  Permutación con seed da resultados DIFERENTES\n")

    # PASO 3: Conclusión
    print("="*70)
    print("CONCLUSIÓN")
    print("="*70)
    print()
    print("Si walkforward REAL y permutaciones CON SEED son idénticos,")
    print("entonces la única diferencia es el uso (o no) de seed.")
    print()
    print("Para resultados reproducibles y comparables:")
    print("  → walkforward_donchian_mod.py debe usar seed=i")
    print("  → walkforward_permutation.py debe usar seed=perm_i")
    print()
    print("Para resultados aleatorios (no reproducibles):")
    print("  → Ambos NO deben usar seed (variación estadística esperada)")
    print()
