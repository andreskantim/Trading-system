"""
Indicador Stochastic Oscillator

Módulo independiente para cálculo y visualización del oscilador estocástico.
"""

import pandas as pd
import numpy as np


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Calcula el oscilador estocástico

    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
        k_period: Periodo para %K (default: 14)
        d_period: Periodo para %D, suavizado de %K (default: 3)

    Returns:
        tuple: (%K, %D) como pd.Series
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100

    # %D = SMA of %K
    stoch_d = stoch_k.rolling(d_period).mean()

    return stoch_k, stoch_d


def visualization(ohlc: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                  oversold: int = 20, overbought: int = 80):
    """
    Datos para visualización del indicador Stochastic.

    Args:
        ohlc: DataFrame con OHLC data
        k_period: Periodo para %K (default: 14)
        d_period: Periodo para %D (default: 3)
        oversold: Nivel de sobreventa (default: 20)
        overbought: Nivel de sobrecompra (default: 80)

    Returns:
        dict con estructura:
        {
            'indicators_in_price': {},
            'indicators_off_price': {
                'stoch_k': {'data': pd.Series, 'color': str},
                'stoch_d': {'data': pd.Series, 'color': str},
                'oversold': {'data': pd.Series, 'color': str},
                'overbought': {'data': pd.Series, 'color': str}
            }
        }
    """
    k_period = int(k_period)
    d_period = int(d_period)
    oversold = int(oversold)
    overbought = int(overbought)

    stoch_k, stoch_d = calculate_stochastic(
        ohlc['high'], ohlc['low'], ohlc['close'],
        k_period, d_period
    )

    # Reference lines
    oversold_line = pd.Series(oversold, index=ohlc.index)
    overbought_line = pd.Series(overbought, index=ohlc.index)

    return {
        'indicators_in_price': {},
        'indicators_off_price': {
            'stoch_k': {'data': stoch_k, 'color': 'cyan'},
            'stoch_d': {'data': stoch_d, 'color': 'orange'},
            'oversold': {'data': oversold_line, 'color': 'green'},
            'overbought': {'data': overbought_line, 'color': 'red'}
        }
    }
