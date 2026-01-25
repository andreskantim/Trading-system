#!/usr/bin/env python3
"""
Visualizador de múltiples indicadores técnicos

Permite visualizar uno o más indicadores simultáneamente sobre datos OHLC.

Usage:
    python indic_vis.py rsi
    python indic_vis.py rsi stochastic
    python indic_vis.py moving_average bollinger_bands --ticker BTCUSD
    python indic_vis.py hawkes permutation_entropy3 permutation_entropy4
"""

import argparse
import importlib
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from visualization.interactive.lightweight_indicator import create_indicator_chart
from config.paths import get_ticker_data_paths, BACKTEST_FIGURES


def main():
    parser = argparse.ArgumentParser(
        description='Visualizacion de indicadores tecnicos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'indicators',
        nargs='+',
        help='Nombres de indicadores a visualizar (ej: rsi stochastic)'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default='BTCUSD',
        help='Ticker symbol (default: BTCUSD)'
    )
    args = parser.parse_args()

    # Cargar datos
    print(f"\n{'='*60}")
    print(f"VISUALIZACION DE INDICADORES")
    print(f"{'='*60}")

    paths = get_ticker_data_paths(args.ticker)
    if paths['parquet'].exists():
        df = pd.read_parquet(paths['parquet'])
    else:
        df = pd.read_csv(paths['csv'])

    print(f"Datos cargados: {len(df):,} filas")
    print(f"Indicadores: {', '.join(args.indicators)}")

    # Cargar y combinar indicadores
    combined_vis = {
        'indicators_in_price': {},
        'indicators_off_price': {}
    }

    # Mapeo de nombres especiales
    indicator_module_map = {
        'permutation_entropy3': ('permutation_entropy', 'visualization3'),
        'permutation_entropy4': ('permutation_entropy', 'visualization4'),
    }

    for ind_name in args.indicators:
        try:
            # Verificar si es un caso especial
            if ind_name in indicator_module_map:
                module_name, func_name = indicator_module_map[ind_name]
                ind_module = importlib.import_module(f'models.indicators.{module_name}')
                vis_func = getattr(ind_module, func_name)
                vis_data = vis_func(df)
            else:
                # Importar módulo del indicador normalmente
                ind_module = importlib.import_module(f'models.indicators.{ind_name}')
                vis_data = ind_module.visualization(df)

            # Combinar indicadores
            combined_vis['indicators_in_price'].update(vis_data.get('indicators_in_price', {}))
            combined_vis['indicators_off_price'].update(vis_data.get('indicators_off_price', {}))

            print(f"{ind_name} cargado")

        except ModuleNotFoundError:
            print(f"Error: Indicador '{ind_name}' no encontrado")
            print(f"Indicadores disponibles: rsi, stochastic, moving_average, hawkes, bollinger_bands, permutation_entropy, permutation_entropy3, permutation_entropy4")
            return
        except Exception as e:
            print(f"Error en '{ind_name}': {e}")
            return

    # Generar visualización
    output_dir = BACKTEST_FIGURES / 'indicators'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{"_".join(args.indicators)}.html'

    result = create_indicator_chart(
        ohlc_data=df,
        vis_data=combined_vis,
        indicator_names=args.indicators,
        output_path=output_path
    )

    if result:
        print(f"\nVisualizacion generada: {result}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
