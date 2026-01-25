#!/usr/bin/env python3
"""
Visualizador de múltiples indicadores técnicos

Permite visualizar uno o más indicadores simultáneamente sobre datos OHLC.

Usage:
    python indicator_vis.py --ticker BTC --indicators rsi
    python indicator_vis.py --ticker ETH --indicators rsi stochastic
    python indicator_vis.py --ticker SOL --indicators moving_average bollinger_bands --start 01/01/2020
    python indicator_vis.py --ticker BTC --indicators hawkes permutation_entropy3 permutation_entropy4
"""

import argparse
import importlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from visualization.interactive.lightweight_indicator import create_indicator_chart
from config.paths import BACKTEST_FIGURES
from utils.data_loader import load_ticker_data, get_available_date_range


def main():
    parser = argparse.ArgumentParser(
        description='Visualizacion de indicadores tecnicos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Ticker a visualizar (ej: BTC, ETH, SOL)'
    )
    parser.add_argument(
        '--indicators',
        nargs='+',
        required=True,
        help='Nombres de indicadores a visualizar (ej: rsi stochastic)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Fecha inicio DD/MM/YYYY (opcional)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='Fecha fin DD/MM/YYYY (opcional)'
    )
    args = parser.parse_args()

    # Cargar datos
    print(f"\n{'='*60}")
    print(f"VISUALIZACIÓN DE INDICADORES")
    print(f"{'='*60}")

    print(f"\nCargando datos de {args.ticker}...")

    if not args.start and not args.end:
        start_available, end_available = get_available_date_range(args.ticker)
        if start_available:
            print(f"  Rango disponible: {start_available} - {end_available}")

    try:
        df = load_ticker_data(args.ticker, start_date=args.start, end_date=args.end)
        print(f"  Datos cargados: {len(df):,} velas")
        print(f"  Desde: {df.index.min()}")
        print(f"  Hasta: {df.index.max()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ejecuta primero: python scripts/laptop/operative_data.py --ticker " + args.ticker)
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nIndicadores: {', '.join(args.indicators)}")

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

            print(f"  {ind_name} cargado")

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
    output_path = output_dir / f'{args.ticker}_{"_".join(args.indicators)}.html'

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
