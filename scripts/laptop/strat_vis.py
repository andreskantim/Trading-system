#!/usr/bin/env python3
"""
Visualizador de estrategias de trading

Genera gráficos interactivos de estrategias con datos desde data/operative/

Usage:
    python strat_vis.py --ticker BTC --strategy hawkes_volatility
    python strat_vis.py --ticker ETH --strategy donchian --start 01/01/2020 --end 31/12/2023
    python strat_vis.py --ticker SOL --strategy moving_average --params 10 50
"""

import pandas as pd
import importlib
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from visualization.interactive.lightweight_strategy import create_interactive_chart
from config.paths import ensure_ticker_OUTPUTS_DIRs
from utils.data_loader import load_ticker_data, get_available_date_range


def main():
    parser = argparse.ArgumentParser(
        description='Generador de gráfico interactivo de estrategias',
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
        '--strategy',
        type=str,
        required=True,
        help='Nombre de la estrategia (ej: donchian, hawkes_volatility)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Fecha inicio DD/MM/YYYY (opcional, usa todo el rango si no se especifica)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='Fecha fin DD/MM/YYYY (opcional)'
    )
    parser.add_argument(
        '--params',
        type=float,
        nargs='+',
        help='Parámetros manuales (opcional, si no se dan se optimiza)'
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"VISUALIZACIÓN DE ESTRATEGIA")
    print(f"{'='*60}")

    # 1. Cargar Estrategia
    strat_name = args.strategy.replace('.py', '')
    try:
        strat = importlib.import_module(f'models.strategies.{strat_name}')
        print(f"Estrategia cargada: {strat_name}")
    except ModuleNotFoundError:
        print(f"Error: No se encontró la estrategia '{strat_name}'")
        return

    # 2. Cargar Datos
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
        print("Ejecuta primero: python utils/operative_data.py --ticker " + args.ticker)
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # 3. Obtener Parámetros
    if args.params:
        best_params = tuple(args.params)
        print(f"\nUsando parámetros manuales: {best_params}")
    else:
        print("\nOptimizando para obtener mejores parámetros...")
        result = strat.optimize(df)
        best_params = tuple(result[:-1])  # Último valor es PF
        pf = result[-1]
        print(f"  Mejores parámetros: {best_params}")
        print(f"  Profit Factor: {pf:.4f}")

    # 4. Generar Visualización
    if hasattr(strat, 'visualization'):
        vis_data = strat.visualization(df, *best_params)

        output_dirs = ensure_ticker_OUTPUTS_DIRs(strat_name, args.ticker)
        output_path = output_dirs['figures'] / f'{args.ticker}_chart.html'

        res = create_interactive_chart(
            ohlc_data=df,
            vis_data=vis_data,
            strategy_name=strat_name,
            params=best_params,
            output_path=output_path
        )

        if res:
            print(f"\nGráfico generado: {res}")
    else:
        print(f"\nLa estrategia '{strat_name}' no tiene método .visualization()")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
