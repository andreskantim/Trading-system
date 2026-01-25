"""
Data Loader Utilities

Funciones para cargar datos desde data/operative/ con filtrado por fechas.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.tickers import get_operative_path


def load_ticker_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Carga datos de un ticker desde data/operative/{ticker}/

    Args:
        ticker: Símbolo del ticker (ej: 'BTC')
        start_date: Fecha inicio formato 'DD/MM/YYYY' (opcional)
        end_date: Fecha fin formato 'DD/MM/YYYY' (opcional)

    Returns:
        DataFrame con datos OHLCV filtrados por fechas

    Example:
        >>> df = load_ticker_data('BTC', start_date='01/01/2020', end_date='31/12/2023')
    """
    operative_path = get_operative_path(ticker)

    if not operative_path.exists():
        raise FileNotFoundError(f"No existen datos operative para {ticker} en {operative_path}")

    # Cargar todos los parquets del ticker
    parquet_files = sorted(operative_path.glob(f"{ticker}_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No hay archivos parquet para {ticker} en {operative_path}")

    # Concatenar todos los años
    dfs = []
    for file in parquet_files:
        df_year = pd.read_parquet(file)
        dfs.append(df_year)

    df = pd.concat(dfs).sort_index()

    # Eliminar duplicados si existen
    df = df[~df.index.duplicated(keep='first')]

    # Filtrar por fechas si se especifican
    if start_date:
        start_ts = pd.Timestamp(datetime.strptime(start_date, '%d/%m/%Y'))
        df = df[df.index >= start_ts]

    if end_date:
        end_ts = pd.Timestamp(datetime.strptime(end_date, '%d/%m/%Y'))
        df = df[df.index <= end_ts]

    if df.empty:
        raise ValueError(f"No hay datos para {ticker} en el rango {start_date} - {end_date}")

    return df


def get_available_date_range(ticker: str) -> tuple:
    """
    Obtiene rango de fechas disponibles para un ticker.

    Args:
        ticker: Símbolo del ticker

    Returns:
        (fecha_inicio, fecha_fin) como strings 'DD/MM/YYYY'
    """
    operative_path = get_operative_path(ticker)
    parquet_files = list(operative_path.glob(f"{ticker}_*.parquet"))

    if not parquet_files:
        return None, None

    # Cargar primer y último archivo
    sorted_files = sorted(parquet_files)
    first_file = sorted_files[0]
    last_file = sorted_files[-1]

    df_first = pd.read_parquet(first_file)
    df_last = pd.read_parquet(last_file)

    start_date = df_first.index.min().strftime('%d/%m/%Y')
    end_date = df_last.index.max().strftime('%d/%m/%Y')

    return start_date, end_date


def get_available_tickers() -> list:
    """
    Lista tickers con datos disponibles en data/operative/

    Returns:
        Lista de símbolos de tickers con datos
    """
    from config.tickers import CONSOLIDATION

    operative_dir = CONSOLIDATION['output_dir']
    if not operative_dir.exists():
        return []

    tickers = []
    for ticker_dir in operative_dir.iterdir():
        if ticker_dir.is_dir():
            parquet_files = list(ticker_dir.glob("*.parquet"))
            if parquet_files:
                tickers.append(ticker_dir.name)

    return sorted(tickers)


def print_ticker_info(ticker: str):
    """
    Imprime información sobre un ticker.
    """
    try:
        start_date, end_date = get_available_date_range(ticker)
        operative_path = get_operative_path(ticker)
        parquet_files = list(operative_path.glob(f"{ticker}_*.parquet"))

        print(f"Ticker: {ticker}")
        print(f"  Path: {operative_path}")
        print(f"  Archivos: {len(parquet_files)}")
        print(f"  Rango: {start_date} - {end_date}")
    except Exception as e:
        print(f"Error obteniendo info de {ticker}: {e}")


if __name__ == '__main__':
    # Mostrar tickers disponibles
    available = get_available_tickers()
    print(f"Tickers disponibles: {len(available)}")
    for ticker in available[:5]:
        print_ticker_info(ticker)
        print()
