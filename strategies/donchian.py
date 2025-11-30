"""
Estrategia Donchian Breakout

Genera señales de trading basadas en rupturas de los canales de Donchian.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def signal(ohlc: pd.DataFrame, lookback: int):
    """
    Genera señales de trading basadas en Donchian Breakout

    Args:
        ohlc: DataFrame con columna 'close'
        lookback: Periodo de lookback para los canales Donchian

    Returns:
        Series con señales: 1 (long), -1 (short), manteniendo posición con ffill
    """
    upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['close'].rolling(lookback - 1).min().shift(1)
    sig = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    sig.loc[ohlc['close'] > upper] = 1
    sig.loc[ohlc['close'] < lower] = -1
    sig = sig.ffill()
    return sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza el parámetro lookback para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (best_lookback, best_pf)
    """
    best_pf = 0
    best_lookback = -1
    r = np.log(ohlc['close']).diff().shift(-1)

    for lookback in range(12, 169):
        sig = signal(ohlc, lookback)
        sig_rets = sig * r
        pos = sig_rets[sig_rets > 0].sum()
        neg = sig_rets[sig_rets < 0].abs().sum()
        if neg == 0:
            sig_pf = np.inf if pos > 0 else 0.0
        else:
            sig_pf = pos / neg

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf
