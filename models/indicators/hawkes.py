"""
Indicador Hawkes Volatility

Módulo independiente para cálculo y visualización de volatilidad Hawkes.
"""

import pandas as pd
import numpy as np


def hawkes_process(data: pd.Series, kappa: float) -> pd.Series:
    """
    Aplica un proceso de Hawkes a una serie temporal

    Args:
        data: Serie temporal de entrada
        kappa: Parámetro de decaimiento del proceso

    Returns:
        pd.Series: Serie temporal procesada por Hawkes
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


def calculate_hawkes(ohlc: pd.DataFrame, kappa: float = 0.125, lookback: int = 169) -> tuple:
    """
    Calcula v_hawkes y percentiles.

    Args:
        ohlc: DataFrame con OHLC data
        kappa: Parámetro de decaimiento del proceso de Hawkes (default: 0.125)
        lookback: Periodo de lookback para cuantiles (default: 169)

    Returns:
        tuple: (v_hawk, q05, q95) como pd.Series
    """
    lookback = int(lookback)

    high = np.log(ohlc["high"])
    low = np.log(ohlc["low"])
    hl_range = high - low
    atr = hl_range.rolling(336).mean()
    norm_range = hl_range / atr
    v_hawk = hawkes_process(norm_range, kappa)

    # Forzar NaN durante todo el periodo de warmup de ATR
    v_hawk.iloc[:336] = np.nan

    q05 = v_hawk.rolling(lookback).quantile(0.05)
    q95 = v_hawk.rolling(lookback).quantile(0.95)

    # Los quantiles necesitan warmup adicional de lookback sobre v_hawk
    q05.iloc[:336 + lookback - 1] = np.nan
    q95.iloc[:336 + lookback - 1] = np.nan

    return v_hawk, q05, q95


def visualization(ohlc: pd.DataFrame, kappa: float = 0.125, lookback: int = 169):
    """
    Datos para visualización del indicador Hawkes Volatility.

    Args:
        ohlc: DataFrame con OHLC data
        kappa: Parámetro de decaimiento del proceso de Hawkes (default: 0.125)
        lookback: Periodo de lookback para cuantiles (default: 169)

    Returns:
        dict con estructura:
        {
            'indicators_in_price': {},
            'indicators_off_price': {
                'v_hawkes': {'data': pd.Series, 'color': str},
                'q05': {'data': pd.Series, 'color': str},
                'q95': {'data': pd.Series, 'color': str}
            }
        }
    """
    v_hawk, q05, q95 = calculate_hawkes(ohlc, kappa, lookback)

    return {
        'indicators_in_price': {},
        'indicators_off_price': {
            'v_hawkes': {'data': v_hawk, 'color': 'yellow'},
            'q05': {'data': q05, 'color': 'red'},
            'q95': {'data': q95, 'color': 'cyan'}
        }
    }
