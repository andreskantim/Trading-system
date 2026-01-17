import pandas as pd
import importlib
import argparse
import sys
from pathlib import Path

# Configurar rutas del proyecto
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from visualization.interactive.lightweight_charts_viewer import create_interactive_chart
from config.paths import get_ticker_data_paths, BACKTEST_FIGURES

def main():
    parser = argparse.ArgumentParser(description='Generador rápido de gráfico interactivo')
    parser.add_argument('strategy', type=str, help='Nombre de la estrategia (ej: donchian)')
    parser.add_argument('--ticker', type=str, default='BTCUSD')
    parser.add_argument('--params', type=float, nargs='+', help='Parámetros manuales (opcional)')
    args = parser.parse_args()

    # 1. Cargar Estrategia
    strat_name = args.strategy.replace('.py', '')
    try:
        strat = importlib.import_module(f'models.strategies.{strat_name}')
    except ModuleNotFoundError:
        print(f"Error: No se encontró la estrategia '{strat_name}'")
        return

    # 2. Cargar Datos
    paths = get_ticker_data_paths(args.ticker)
    df = pd.read_parquet(paths['parquet']) if paths['parquet'].exists() else pd.read_csv(paths['csv'], index_col=0, parse_dates=True)
    
    # 3. Obtener Parámetros (Optimizar si no se proveen)
    if args.params:
        best_params = tuple(args.params)
        print(f"Usando parámetros manuales: {best_params}")
    else:
        print("Optimizando para obtener mejores parámetros...")
        *best_params, _ = strat.optimize(df)
        best_params = tuple(best_params)

    # 4. Generar Visualización
    if hasattr(strat, 'visualization'):
        vis_data = strat.visualization(df, *best_params)
        output_path = BACKTEST_FIGURES / strat_name / 'quick_chart.html'
        
        res = create_interactive_chart(
            ohlc_data=df,
            vis_data=vis_data,
            strategy_name=strat_name,
            params=best_params,
            output_path=output_path
        )
        
        if res: print(f"✅ Gráfico generado en: {res}")
    else:
        print(f"La estrategia '{strat_name}' no tiene el método .visualization()")

if __name__ == "__main__":
    main()