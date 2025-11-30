"""
Estrategia Hawkes Volatility

Genera señales de trading basadas en procesos de Hawkes aplicados a la volatilidad.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def hawkes_process(data: pd.Series, kappa: float):
    """
    Aplica un proceso de Hawkes a una serie temporal

    Args:
        data: Serie temporal de entrada
        kappa: Parámetro de decaimiento del proceso

    Returns:
        Serie temporal procesada por Hawkes
    """
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    out = np.zeros(len(arr))
    out[:] = np.nan
    for i in range(1, len(arr)):
        if np.isnan(out[i-1]):
            out[i] = arr[i]
        else:
            out[i] = out[i-1] * alpha + arr[i]
    return pd.Series(out, index=data.index) * kappa


def hawkes_vol_signal(close: pd.Series, v_hawk: pd.Series, lookback: int):
    """
    Genera señales basadas en volatilidad Hawkes

    Args:
        close: Serie de precios de cierre
        v_hawk: Serie de volatilidad procesada por Hawkes
        lookback: Periodo de lookback para cuantiles

    Returns:
        Serie con señales
    """
    signal = np.zeros(len(close))
    q05 = v_hawk.rolling(lookback).quantile(0.05)
    q95 = v_hawk.rolling(lookback).quantile(0.95)
    last_below = -1
    curr_sig = 0
    for i in range(1, len(close)):
        if v_hawk.iloc[i] < q05.iloc[i]:
            last_below = i
            curr_sig = 0
        if v_hawk.iloc[i] > q95.iloc[i] and v_hawk.iloc[i-1] <= q95.iloc[i-1] and last_below > 0:
            change = close.iloc[i] - close.iloc[last_below]
            curr_sig = 1 if change > 0 else -1
        signal[i] = curr_sig
    return pd.Series(signal, index=close.index)


def signal(ohlc: pd.DataFrame, kappa: float, lookback: int):
    """
    Genera señales de trading basadas en Hawkes Volatility

    Args:
        ohlc: DataFrame con columnas 'high', 'low', 'close'
        kappa: Parámetro de decaimiento del proceso de Hawkes
        lookback: Periodo de lookback para cuantiles

    Returns:
        Series con señales: 1 (long), -1 (short), 0 (flat)
    """
    high = np.log(ohlc["high"])
    low = np.log(ohlc["low"])
    close = np.log(ohlc["close"])
    hl_range = high - low
    atr = hl_range.rolling(336).mean()
    norm_range = hl_range / atr
    v_hawk = hawkes_process(norm_range, kappa)
    sig = hawkes_vol_signal(ohlc['close'], v_hawk, lookback)
    return sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza los parámetros kappa y lookback para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columnas 'high', 'low', 'close'

    Returns:
        Tupla (best_kappa, best_lookback, best_pf)
    """
    r = np.log(ohlc["close"]).diff().shift(-1)
    best_pf = 0
    best_kappa = 0.1
    best_lb = 169

    # Barrer valores de kappa
    kappa_values = [0.125]

    # Barrer valores de lookback
    lb_values = [96, 120, 144, 168, 169]

    for kappa in kappa_values:
        for lb in lb_values:
            sig = signal(ohlc, kappa, lb)
            strat = sig * r
            pos = strat[strat > 0].sum()
            neg = strat[strat < 0].abs().sum()
            if neg == 0:
                pf = np.inf if pos > 0 else 0
            else:
                pf = pos / neg
            if pf > best_pf:
                best_pf = pf
                best_kappa = kappa
                best_lb = lb

    return best_kappa, best_lb, best_pf
