"""
Indicador Bollinger Bands

Módulo independiente para cálculo y visualización de Bandas de Bollinger.
"""

import pandas as pd
import numpy as np


def calculate_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    """
    Calcula las Bandas de Bollinger

    Args:
        close: Serie de precios de cierre
        period: Periodo para la media móvil (default: 20)
        num_std: Número de desviaciones estándar (default: 2.0)

    Returns:
        tuple: (middle_band, upper_band, lower_band) como pd.Series
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle, upper, lower


def visualization(ohlc: pd.DataFrame, period: int = 20, num_std: float = 2.0):
    """
    Datos para visualización del indicador Bollinger Bands.

    Args:
        ohlc: DataFrame con OHLC data
        period: Periodo para la media móvil (default: 20)
        num_std: Número de desviaciones estándar (default: 2.0)

    Returns:
        dict con estructura:
        {
            'indicators_in_price': {
                'bb_upper': {'data': pd.Series, 'color': str},
                'bb_middle': {'data': pd.Series, 'color': str},
                'bb_lower': {'data': pd.Series, 'color': str}
            },
            'indicators_off_price': {}
        }
    """
    period = int(period)

    middle, upper, lower = calculate_bollinger(ohlc['close'], period, num_std)

    return {
        'indicators_in_price': {
            'bb_upper': {'data': upper, 'color': 'red'},
            'bb_middle': {'data': middle, 'color': 'yellow'},
            'bb_lower': {'data': lower, 'color': 'green'}
        },
        'indicators_off_price': {}
    }
