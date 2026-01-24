"""
Indicador RSI (Relative Strength Index)

Módulo independiente para cálculo y visualización del RSI.
"""

import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula el RSI (Relative Strength Index)

    Args:
        close: Serie de precios de cierre
        period: Periodo para el cálculo del RSI (default: 14)

    Returns:
        pd.Series con valores RSI (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def visualization(ohlc: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70):
    """
    Datos para visualización del indicador RSI.

    Args:
        ohlc: DataFrame con OHLC data
        period: Periodo RSI (default: 14)
        oversold: Nivel de sobreventa (default: 30)
        overbought: Nivel de sobrecompra (default: 70)

    Returns:
        dict con estructura:
        {
            'indicators_in_price': {},
            'indicators_off_price': {
                'rsi': {'data': pd.Series, 'color': str},
                'oversold': {'data': pd.Series, 'color': str},
                'overbought': {'data': pd.Series, 'color': str}
            }
        }
    """
    period = int(period)
    oversold = int(oversold)
    overbought = int(overbought)

    rsi = calculate_rsi(ohlc['close'], period)

    # Create constant lines for levels
    oversold_line = pd.Series(oversold, index=ohlc.index)
    overbought_line = pd.Series(overbought, index=ohlc.index)

    return {
        'indicators_in_price': {},
        'indicators_off_price': {
            'rsi': {'data': rsi, 'color': 'yellow'},
            'oversold': {'data': oversold_line, 'color': 'green'},
            'overbought': {'data': overbought_line, 'color': 'red'}
        }
    }
