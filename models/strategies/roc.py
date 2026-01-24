"""
Estrategia Rate of Change (ROC)

Genera señales de trading basadas en el momentum medido por el Rate of Change.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def calculate_roc(close: pd.Series, period: int) -> pd.Series:
    """
    Calcula el Rate of Change (ROC)

    Args:
        close: Serie de precios de cierre
        period: Periodo para el cálculo del ROC

    Returns:
        Serie con valores ROC (porcentaje de cambio)
    """
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return roc


def signal(ohlc: pd.DataFrame, period: int, threshold: float):
    """
    Genera señales de trading basadas en ROC

    Args:
        ohlc: DataFrame con columna 'close'
        period: Periodo para el cálculo del ROC
        threshold: Umbral para generar señales (ej: 0 para cruce de cero)

    Returns:
        Series con señales: 1 (long), -1 (short)
    """
    period = int(period)

    roc = calculate_roc(ohlc['close'], period)

    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)

    # Long cuando ROC está por encima del umbral
    # Short cuando ROC está por debajo del umbral negativo
    sig[roc > threshold] = 1
    sig[roc < -threshold] = -1

    return sig.ffill()


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza los parámetros del ROC para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (best_period, best_threshold, best_pf)
    """
    best_pf = 0
    best_period = 12
    best_threshold = 0.0
    r = np.log(ohlc['close']).diff().shift(-1)

    for period in [6, 12, 24, 48, 72]:
        for threshold in [2.0, 3.0, 4.0, 5.0]:
            sig = signal(ohlc, period, threshold)
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
                best_threshold = threshold

    return best_period, best_threshold, best_pf


def visualization(ohlc: pd.DataFrame, period: int, threshold: float):
    """
    Calculate all indicators needed for interactive visualization.

    Args:
        ohlc: DataFrame with OHLC data
        period: ROC period
        threshold: Signal threshold

    Returns:
        dict with indicators and signals
    """
    period = int(period)

    roc = calculate_roc(ohlc['close'], period)

    # Reference lines
    zero_line = pd.Series(0, index=ohlc.index)
    upper_threshold = pd.Series(threshold, index=ohlc.index)
    lower_threshold = pd.Series(-threshold, index=ohlc.index)

    signals = signal(ohlc, period, threshold)

    return {
        'indicators_in_price': {
            # No in-price indicators for ROC
        },
        'indicators_off_price': {
            'roc': {'data': roc, 'color': 'cyan'},
            'zero': {'data': zero_line, 'color': 'gray'},
            'upper_threshold': {'data': upper_threshold, 'color': 'green'},
            'lower_threshold': {'data': lower_threshold, 'color': 'red'}
        },
        'signals': signals
    }
