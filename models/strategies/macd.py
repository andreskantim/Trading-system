"""
Estrategia MACD (Moving Average Convergence Divergence)

Genera señales de trading basadas en cruces del MACD con su línea de señal.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def calculate_macd(close: pd.Series, fast: int, slow: int, signal_period: int):
    """
    Calcula el MACD y su línea de señal

    Args:
        close: Serie de precios de cierre
        fast: Periodo EMA rápida
        slow: Periodo EMA lenta
        signal_period: Periodo para la línea de señal

    Returns:
        Tupla (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def signal(ohlc: pd.DataFrame, fast: int, slow: int, signal_period: int):
    """
    Genera señales de trading basadas en cruces del MACD

    Args:
        ohlc: DataFrame con columna 'close'
        fast: Periodo EMA rápida
        slow: Periodo EMA lenta
        signal_period: Periodo para la línea de señal

    Returns:
        Series con señales: 1 (long), -1 (short)
    """
    fast = int(fast)
    slow = int(slow)
    signal_period = int(signal_period)

    if fast >= slow:
        raise ValueError("fast EMA debe ser menor que slow EMA")

    macd_line, signal_line, _ = calculate_macd(ohlc['close'], fast, slow, signal_period)

    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)

    # Long cuando MACD cruza por encima de la señal
    # Short cuando MACD cruza por debajo de la señal
    sig[macd_line > signal_line] = 1
    sig[macd_line < signal_line] = -1

    return sig.ffill()


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza los parámetros del MACD para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (best_fast, best_slow, best_signal, best_pf)
    """
    best_pf = 0
    best_fast = 12
    best_slow = 26
    best_signal = 9
    r = np.log(ohlc['close']).diff().shift(-1)

    for fast in [8, 10, 12, 14, 16]:
        for slow in [20, 24, 26, 30, 34]:
            for sig_period in [7, 9, 11, 13]:
                if fast >= slow:
                    continue
                sig = signal(ohlc, fast, slow, sig_period)
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
                    best_signal = sig_period

    return best_fast, best_slow, best_signal, best_pf


def visualization(ohlc: pd.DataFrame, fast: int, slow: int, signal_period: int):
    """
    Calculate all indicators needed for interactive visualization.

    Args:
        ohlc: DataFrame with OHLC data
        fast: Fast EMA period
        slow: Slow EMA period
        signal_period: Signal line period

    Returns:
        dict with indicators and signals
    """
    fast = int(fast)
    slow = int(slow)
    signal_period = int(signal_period)

    macd_line, signal_line, histogram = calculate_macd(ohlc['close'], fast, slow, signal_period)

    # Zero line for reference
    zero_line = pd.Series(0, index=ohlc.index)

    signals = signal(ohlc, fast, slow, signal_period)

    return {
        'indicators_in_price': {
            # No in-price indicators for MACD
        },
        'indicators_off_price': {
            'macd': {'data': macd_line, 'color': 'cyan'},
            'signal_line': {'data': signal_line, 'color': 'orange'},
            'zero': {'data': zero_line, 'color': 'gray'}
        },
        'signals': signals
    }
