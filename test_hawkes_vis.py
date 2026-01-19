#!/usr/bin/env python3
"""Script para debuggear qué devuelve hawkes.visualization()"""
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.strategies import hawkes
from config.paths import get_ticker_data_paths

# Cargar datos
paths = get_ticker_data_paths('BTCUSD')
df = pd.read_parquet(paths['parquet'])

print("="*60)
print("TEST: ¿Qué devuelve hawkes.visualization()?")
print("="*60)

# Optimizar
result = hawkes.optimize(df)
kappa, lookback, pf = result
print(f"\nParámetros optimizados: kappa={kappa}, lookback={lookback}, pf={pf:.4f}")

# Llamar a visualization
vis_data = hawkes.visualization(df, kappa, lookback)

print(f"\n📋 Estructura de vis_data:")
print(f"  Keys: {list(vis_data.keys())}")

if 'indicators' in vis_data:
    print(f"\n✓ Tiene 'indicators' (formato viejo):")
    for name, spec in vis_data['indicators'].items():
        data = spec.get('data')
        panel = spec.get('panel', 'unknown')
        color = spec.get('color', 'unknown')
        print(f"    - {name}:")
        print(f"        panel: {panel}")
        print(f"        color: {color}")
        print(f"        data: {len(data) if data is not None else 0} puntos")

if 'indicators_in_price' in vis_data:
    print(f"\n✓ Tiene 'indicators_in_price':")
    print(f"    {list(vis_data['indicators_in_price'].keys())}")

if 'indicators_off_price' in vis_data:
    print(f"\n✓ Tiene 'indicators_off_price':")
    print(f"    {list(vis_data['indicators_off_price'].keys())}")

if 'signals' in vis_data:
    signals = vis_data['signals']
    print(f"\n✓ Tiene 'signals': {len(signals)} puntos")
    print(f"    Valores únicos: {signals.unique()}")
    print(f"    Conteo: {signals.value_counts().to_dict()}")
else:
    print(f"\n✗ NO tiene 'signals'")

print("\n" + "="*60)
