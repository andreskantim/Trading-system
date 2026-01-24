"""
Estrategia Bollinger Bands

Genera señales de trading basadas en rupturas de las bandas de Bollinger.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def calculate_bollinger(close: pd.Series, period: int, num_std: float):
    """
    Calcula las bandas de Bollinger

    Args:
        close: Serie de precios de cierre
        period: Periodo para la media móvil
        num_std: Número de desviaciones estándar

    Returns:
        Tupla (middle_band, upper_band, lower_band)
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle, upper, lower


def signal(ohlc: pd.DataFrame, period: int, num_std: float):
    """
    Genera señales de trading basadas en Bollinger Bands breakout

    Args:
        ohlc: DataFrame con columna 'close'
        period: Periodo para la media móvil
        num_std: Número de desviaciones estándar

    Returns:
        Series con señales: 1 (long), -1 (short)
    """
    period = int(period)

    middle, upper, lower = calculate_bollinger(ohlc['close'], period, num_std)

    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)

    # Long cuando precio rompe por encima de la banda superior
    # Short cuando precio rompe por debajo de la banda inferior
    sig[ohlc['close'] > upper] = 1
    sig[ohlc['close'] < lower] = -1

    # Mantener posición con ffill
    sig = sig.replace(0, np.nan).ffill().fillna(0)

    return sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza los parámetros de Bollinger para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (best_period, best_num_std, best_pf)
    """
    best_pf = 0
    best_period = 20
    best_num_std = 2.0
    r = np.log(ohlc['close']).diff().shift(-1)

    for period in [10, 15, 20, 25, 30, 40, 50]:
        for num_std in [1.5, 2.0, 2.5, 3.0]:
            sig = signal(ohlc, period, num_std)
            sig_rets = sig * r
            pos = sig_rets[sig_rets > 0].sum()
            neg = sig_rets[sig_rets < 0].abs().sum()
            if neg == 0:
                sig_pf = np.inf if pos > 0 else 0.0
            else:
                sig_pf = pos / neg

            if sig_pf > best_pf:
                best_pf = sig_pf
                best_period = period
                best_num_std = num_std

    return best_period, best_num_std, best_pf


def visualization(ohlc: pd.DataFrame, period: int, num_std: float):
    """
    Calculate all indicators needed for interactive visualization.

    Args:
        ohlc: DataFrame with OHLC data
        period: Period for moving average
        num_std: Number of standard deviations

    Returns:
        dict with indicators and signals
    """
    period = int(period)

    middle, upper, lower = calculate_bollinger(ohlc['close'], period, num_std)

    signals = signal(ohlc, period, num_std)

    return {
        'indicators_in_price': {
            'bb_upper': {'data': upper, 'color': 'red'},
            'bb_middle': {'data': middle, 'color': 'yellow'},
            'bb_lower': {'data': lower, 'color': 'green'}
        },
        'indicators_off_price': {
            # No off-price indicators for Bollinger
        },
        'signals': signals
    }
