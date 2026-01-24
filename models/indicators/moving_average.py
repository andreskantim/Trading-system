"""
Indicador Moving Average

Módulo independiente para cálculo y visualización de medias móviles.
"""

import pandas as pd
import numpy as np


def calculate_moving_average(close: pd.Series, period: int, ma_type: str = 'SMA') -> pd.Series:
    """
    Calcula una Media Móvil (SMA o EMA)

    Args:
        close: Serie de precios de cierre
        period: Periodo de la media móvil
        ma_type: Tipo de media móvil ('SMA' o 'EMA')

    Returns:
        pd.Series con los valores de la media móvil
    """
    ma_type = ma_type.upper()

    if ma_type == 'SMA':
        return close.rolling(period).mean()
    elif ma_type == 'EMA':
        return close.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError(f"ma_type debe ser 'SMA' o 'EMA', no '{ma_type}'")


def visualization(ohlc: pd.DataFrame, fast: int = 20, slow: int = 50, ma_type: str = 'SMA'):
    """
    Datos para visualización del indicador Moving Average.

    Args:
        ohlc: DataFrame con OHLC data
        fast: Periodo de la media móvil rápida (default: 20)
        slow: Periodo de la media móvil lenta (default: 50)
        ma_type: Tipo de media móvil ('SMA' o 'EMA')

    Returns:
        dict con estructura:
        {
            'indicators_in_price': {
                'fast_ma': {'data': pd.Series, 'color': str},
                'slow_ma': {'data': pd.Series, 'color': str}
            },
            'indicators_off_price': {}
        }
    """
    fast = int(fast)
    slow = int(slow)

    fast_ma = calculate_moving_average(ohlc['close'], fast, ma_type)
    slow_ma = calculate_moving_average(ohlc['close'], slow, ma_type)

    return {
        'indicators_in_price': {
            'fast_ma': {'data': fast_ma, 'color': 'cyan'},
            'slow_ma': {'data': slow_ma, 'color': 'orange'}
        },
        'indicators_off_price': {}
    }
