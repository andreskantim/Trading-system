"""
Estrategia RSI (Relative Strength Index)

Genera señales de trading basadas en niveles de sobrecompra/sobreventa del RSI.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Calcula el RSI (Relative Strength Index)

    Args:
        close: Serie de precios de cierre
        period: Periodo para el cálculo del RSI

    Returns:
        Serie con valores RSI (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signal(ohlc: pd.DataFrame, period: int, oversold: int, overbought: int):
    """
    Genera señales de trading basadas en RSI

    Args:
        ohlc: DataFrame con columna 'close'
        period: Periodo para el cálculo del RSI
        oversold: Nivel de sobreventa (ej: 30)
        overbought: Nivel de sobrecompra (ej: 70)

    Returns:
        Series con señales: 1 (long), -1 (short), 0 (flat)
    """
    period = int(period)
    oversold = int(oversold)
    overbought = int(overbought)

    rsi = calculate_rsi(ohlc['close'], period)

    sig = pd.Series(np.zeros(len(ohlc)), index=ohlc.index)

    # Long cuando RSI cruza por encima del nivel de sobreventa
    # Short cuando RSI cruza por debajo del nivel de sobrecompra
    long_condition = (rsi > oversold) & (rsi.shift(1) <= oversold)
    short_condition = (rsi < overbought) & (rsi.shift(1) >= overbought)

    sig[long_condition] = 1
    sig[short_condition] = -1

    # Mantener posición hasta señal contraria
    sig = sig.replace(0, np.nan).ffill().fillna(0)

    return sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza los parámetros del RSI para maximizar el Profit Factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (best_period, best_oversold, best_overbought, best_pf)
    """
    best_pf = 0
    best_period = 14
    best_oversold = 30
    best_overbought = 70
    r = np.log(ohlc['close']).diff().shift(-1)

    for period in [7, 14, 21, 28]:
        for oversold in [20, 25, 30, 35]:
            for overbought in [65, 70, 75, 80]:
                sig = signal(ohlc, period, oversold, overbought)
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
                    best_oversold = oversold
                    best_overbought = overbought

    return best_period, best_oversold, best_overbought, best_pf


def visualization(ohlc: pd.DataFrame, period: int, oversold: int, overbought: int):
    """
    Calculate all indicators needed for interactive visualization.

    Args:
        ohlc: DataFrame with OHLC data
        period: RSI period
        oversold: Oversold level
        overbought: Overbought level

    Returns:
        dict with indicators and signals
    """
    period = int(period)
    oversold = int(oversold)
    overbought = int(overbought)

    rsi = calculate_rsi(ohlc['close'], period)

    # Create constant lines for levels
    oversold_line = pd.Series(oversold, index=ohlc.index)
    overbought_line = pd.Series(overbought, index=ohlc.index)

    signals = signal(ohlc, period, oversold, overbought)

    return {
        'indicators_in_price': {
            # No in-price indicators for RSI
        },
        'indicators_off_price': {
            'rsi': {'data': rsi, 'color': 'yellow'},
            'oversold': {'data': oversold_line, 'color': 'green'},
            'overbought': {'data': overbought_line, 'color': 'red'}
        },
        'signals': signals
    }
