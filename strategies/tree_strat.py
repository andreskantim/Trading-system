"""
Estrategia Decision Tree

Genera señales de trading usando un árbol de decisión entrenado con características técnicas.
Compatible con los scripts de visualización y MCPT.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def signal(ohlc: pd.DataFrame, model):
    """
    Genera señales de trading usando un modelo de árbol de decisión entrenado

    Args:
        ohlc: DataFrame con columna 'close'
        model: Modelo DecisionTreeClassifier entrenado

    Returns:
        Series con señales (1 o -1)
    """
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff168 = log_c.diff(168)

    dataset = pd.concat([diff6, diff24, diff168], axis=1)
    dataset.columns = ['diff6', 'diff24', 'diff168']
    dataset = dataset.dropna()

    pred = model.predict(dataset.to_numpy())
    pred = pd.Series(pred, index=dataset.index)

    # Reindex to actual data
    pred = pred.reindex(ohlc.index)

    # Make predictions tradable: 0 -> -1, 1 -> 1
    sig = np.where(pred > 0, 1, -1)
    sig = pd.Series(sig, index=ohlc.index)

    return sig


def optimize(ohlc: pd.DataFrame):
    """
    Entrena un modelo de árbol de decisión y retorna el modelo y su profit factor

    Args:
        ohlc: DataFrame con columna 'close'

    Returns:
        Tupla (model, profit_factor)
    """
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff168 = log_c.diff(168)

    # -1 or 1 if next 24 hours go up/down
    target = np.sign(log_c.diff(24).shift(-24))

    # Transform to -1, 1 to 0, 1
    target = (target + 1) / 2

    dataset = pd.concat([diff6, diff24, diff168, target], axis=1)
    dataset.columns = ['diff6', 'diff24', 'diff168', 'target']

    train_data = dataset.dropna()
    train_x = train_data[['diff6', 'diff24', 'diff168']].to_numpy()
    train_y = train_data['target'].astype(int).to_numpy()

    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=69)
    model.fit(train_x, train_y)

    # Calcular profit factor del modelo entrenado
    sig = signal(ohlc, model)
    r = np.log(ohlc['close']).diff().shift(-1)
    sig_rets = sig * r
    pos = sig_rets[sig_rets > 0].sum()
    neg = sig_rets[sig_rets < 0].abs().sum()
    if neg == 0:
        pf = np.inf if pos > 0 else 0.0
    else:
        pf = pos / neg

    return model, pf
