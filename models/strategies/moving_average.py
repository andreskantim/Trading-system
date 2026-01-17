"""
Estrategia Moving Average Crossover

Genera señales de trading basadas en el cruce de medias móviles.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def signal(ohlc: pd.DataFrame, fast: int, slow: int):
    """
    Genera señales de trading basadas en cruce de medias móviles

    Args:
        ohlc: DataFrame con columna 'close'
        fast: Periodo de la media móvil rápida
        slow: Periodo de la media móvil lenta

    Returns:
        Series con señales: 1 (long), -1 (short)
    """
    if fast >= slow:
        raise ValueError("fast MA debe ser menor que slow MA")

    fast_ma = ohlc['close'].rolling(fast).mean()
    slow_ma = ohlc['close'].rolling(slow).mean()
    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)
    sig[fast_ma > slow_ma] = 1
    sig[fast_ma < slow_ma] = -1
    return sig.ffill()


def optimize(ohlc: pd.DataFrame, fast_range=(15, 25), slow_range=(140, 160)):
    """
    Optimiza los parámetros fast y slow para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columna 'close'
        fast_range: Tupla (min, max) para el rango de la MA rápida
        slow_range: Tupla (min, max) para el rango de la MA lenta

    Returns:
        Tupla (best_fast, best_slow, best_pf)
    """
    best_pf = 0
    best_fast = -1
    best_slow = -1
    r = np.log(ohlc['close']).diff().shift(-1)

    for fast in range(fast_range[0], fast_range[1] + 1):
        for slow in range(slow_range[0], slow_range[1] + 1):
            if fast >= slow:
                continue
            sig = signal(ohlc, fast, slow)
            sig_rets = sig * r
            pos = sig_rets[sig_rets > 0].sum()
            neg = sig_rets[sig_rets < 0].abs().sum()
            if neg == 0:
                sig_pf = np.inf if pos > 0 else 0.0
            else:
                sig_pf = pos / neg

            if sig_pf > best_pf:
                best_pf = sig_pf
                best_fast = fast
                best_slow = slow

    return best_fast, best_slow, best_pf
