"""
Estrategia Stochastic Oscillator

Genera señales de trading basadas en el oscilador estocástico (%K y %D).
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int, d_period: int):
    """
    Calcula el oscilador estocástico

    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
        k_period: Periodo para %K
        d_period: Periodo para %D (suavizado de %K)

    Returns:
        Tupla (%K, %D)
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100

    # %D = SMA of %K
    stoch_d = stoch_k.rolling(d_period).mean()

    return stoch_k, stoch_d


def signal(ohlc: pd.DataFrame, k_period: int, d_period: int, oversold: int, overbought: int):
    """
    Genera señales de trading basadas en Stochastic Oscillator

    Args:
        ohlc: DataFrame con columnas 'high', 'low', 'close'
        k_period: Periodo para %K
        d_period: Periodo para %D
        oversold: Nivel de sobreventa (ej: 20)
        overbought: Nivel de sobrecompra (ej: 80)

    Returns:
        Series con señales: 1 (long), -1 (short), 0 (flat)
    """
    k_period = int(k_period)
    d_period = int(d_period)
    oversold = int(oversold)
    overbought = int(overbought)

    stoch_k, stoch_d = calculate_stochastic(ohlc['high'], ohlc['low'], ohlc['close'],
                                             k_period, d_period)

    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)

    # Long: %K cruza por encima de %D en zona de sobreventa
    long_cross = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
    long_condition = long_cross & (stoch_k < oversold + 10)

    # Short: %K cruza por debajo de %D en zona de sobrecompra
    short_cross = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))
    short_condition = short_cross & (stoch_k > overbought - 10)

    sig[long_condition] = 1
    sig[short_condition] = -1

    # Mantener posición hasta señal contraria
    sig = sig.replace(0, np.nan).ffill().fillna(0)

    return sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza los parámetros del Stochastic para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columnas 'high', 'low', 'close'

    Returns:
        Tupla (best_k_period, best_d_period, best_oversold, best_overbought, best_pf)
    """
    best_pf = 0
    best_k = 14
    best_d = 3
    best_oversold = 20
    best_overbought = 80
    r = np.log(ohlc['close']).diff().shift(-1)

    for k_period in [5, 9, 14, 21]:
        for d_period in [3, 5, 7]:
            for oversold in [15, 20, 25, 30]:
                for overbought in [70, 75, 80, 85]:
                    sig = signal(ohlc, k_period, d_period, oversold, overbought)
                    sig_rets = sig * r
                    pos = sig_rets[sig_rets > 0].sum()
                    neg = sig_rets[sig_rets < 0].abs().sum()
                    if neg == 0:
                        sig_pf = np.inf if pos > 0 else 0.0
                    else:
                        sig_pf = pos / neg

                    if sig_pf > best_pf:
                        best_pf = sig_pf
                        best_k = k_period
                        best_d = d_period
                        best_oversold = oversold
                        best_overbought = overbought

    return best_k, best_d, best_oversold, best_overbought, best_pf


def visualization(ohlc: pd.DataFrame, k_period: int, d_period: int,
                  oversold: int, overbought: int):
    """
    Calculate all indicators needed for interactive visualization.

    Args:
        ohlc: DataFrame with OHLC data
        k_period: %K period
        d_period: %D period
        oversold: Oversold level
        overbought: Overbought level

    Returns:
        dict with indicators and signals
    """
    k_period = int(k_period)
    d_period = int(d_period)
    oversold = int(oversold)
    overbought = int(overbought)

    stoch_k, stoch_d = calculate_stochastic(ohlc['high'], ohlc['low'], ohlc['close'],
                                             k_period, d_period)

    # Reference lines
    oversold_line = pd.Series(oversold, index=ohlc.index)
    overbought_line = pd.Series(overbought, index=ohlc.index)

    signals = signal(ohlc, k_period, d_period, oversold, overbought)

    return {
        'indicators_in_price': {
            # No in-price indicators for Stochastic
        },
        'indicators_off_price': {
            'stoch_k': {'data': stoch_k, 'color': 'cyan'},
            'stoch_d': {'data': stoch_d, 'color': 'orange'},
            'oversold': {'data': oversold_line, 'color': 'green'},
            'overbought': {'data': overbought_line, 'color': 'red'}
        },
        'signals': signals
    }
