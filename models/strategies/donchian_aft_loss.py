"""
Estrategia Donchian Breakout con filtro "After Loser"

Esta estrategia solo opera cuando el último trade fue perdedor,
aprovechando el sesgo mean-reversion detectado en el análisis de
dependencia de trades.

Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np


def donchian_base_signal(ohlc: pd.DataFrame, lookback: int):
    """
    Genera señales Donchian básicas (sin filtro)

    Args:
        ohlc: DataFrame con columna 'close'
        lookback: Periodo de lookback para canales Donchian

    Returns:
        Series con señales: 1 (long), -1 (short), ffill para mantener posición
    """
    upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['close'].rolling(lookback - 1).min().shift(1)
    sig = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    sig.loc[ohlc['close'] > upper] = 1
    sig.loc[ohlc['close'] < lower] = -1
    sig = sig.ffill()
    return sig


def apply_after_loser_filter(ohlc: pd.DataFrame, base_signal: pd.Series):
    """
    Aplica filtro "After Loser" a señales base.
    Solo opera cuando el último trade fue perdedor.

    Args:
        ohlc: DataFrame con columna 'close'
        base_signal: Series con señales base (1, -1, 0)

    Returns:
        Series con señales filtradas
    """
    # Convertir a numpy para evitar warnings
    signal = base_signal.to_numpy()
    close = ohlc['close'].to_numpy()

    # Señal filtrada (inicialmente todo en 0)
    filtered_signal = np.zeros(len(signal))

    # Tracking de trades
    long_entry_price = np.nan
    short_entry_price = np.nan
    last_long_result = np.nan  # 1: ganó, -1: perdió
    last_short_result = np.nan

    last_sig = 0.0

    for i in range(len(close)):
        # Detectar entrada long
        if signal[i] == 1.0 and last_sig != 1.0:
            long_entry_price = close[i]
            # Si había short abierto, cerrarlo y registrar resultado
            if not np.isnan(short_entry_price):
                last_short_result = np.sign(short_entry_price - close[i])
                short_entry_price = np.nan

        # Detectar entrada short
        if signal[i] == -1.0 and last_sig != -1.0:
            short_entry_price = close[i]
            # Si había long abierto, cerrarlo y registrar resultado
            if not np.isnan(long_entry_price):
                last_long_result = np.sign(close[i] - long_entry_price)
                long_entry_price = np.nan

        last_sig = signal[i]

        # Aplicar filtro: solo operar si el ÚLTIMO TRADE DEL LADO OPUESTO fue perdedor
        if signal[i] == 1.0 and last_short_result == -1:
            # Entrar long solo si último short perdió
            filtered_signal[i] = 1.0
        elif signal[i] == -1.0 and last_long_result == -1:
            # Entrar short solo si último long perdió
            filtered_signal[i] = -1.0

    return pd.Series(filtered_signal, index=ohlc.index)


def signal(ohlc: pd.DataFrame, lookback: int):
    """
    Genera señales Donchian con filtro "After Loser"

    Args:
        ohlc: DataFrame con columna 'close'
        lookback: Periodo de lookback para canales Donchian

    Returns:
        Series con señales filtradas: 1 (long), -1 (short), 0 (flat)
    """
    # Generar señales base
    base_sig = donchian_base_signal(ohlc, lookback)

    # Aplicar filtro "After Loser"
    filtered_sig = apply_after_loser_filter(ohlc, base_sig)

    return filtered_sig


def optimize(ohlc: pd.DataFrame):
    """
    Optimiza el lookback de Donchian con filtro "After Loser"

    Busca el lookback que maximiza el Profit Factor en el rango [12, 169)

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
