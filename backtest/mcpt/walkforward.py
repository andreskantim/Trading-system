"""Core walk-forward MCPT functions."""

import numpy as np
import pandas as pd
from backtest.mcpt.bar_permute import get_permutation

# Global worker state
_strategy_module = None
_df = None
_train_window = None
_real_wf_pf = None


def walkforward_strategy(ohlc: pd.DataFrame, strategy, train_lookback: int, train_step: int = None):
    """
    Walk-forward optimization.

    Args:
        ohlc: DataFrame with OHLC data
        strategy: Strategy module with optimize() and signal()
        train_lookback: Training window in bars
        train_step: Steps between re-optimizations (default: train_lookback // 12)

    Returns:
        Array with walk-forward signals
    """
    if train_step is None:
        train_step = max(train_lookback // 12, 24 * 30)

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            result = strategy.optimize(ohlc.iloc[i - train_lookback:i])
            best_params = result[:-1]
            tmp_signal = strategy.signal(ohlc, *best_params)
            next_train += train_step
        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


def init_walkforward_worker(strategy_name, df_data, df_index, df_columns, train_window, real_wf_pf):
    """Initialize worker with strategy module and data."""
    import importlib
    global _strategy_module, _df, _train_window, _real_wf_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    _df = pd.DataFrame(df_data, index=df_index, columns=df_columns)
    _train_window = train_window
    _real_wf_pf = real_wf_pf


def process_walkforward_permutation(perm_i):
    """Process single walk-forward permutation."""
    strategy = _strategy_module
    df_perm = _df.copy()
    train_window = _train_window
    real_wf_pf = _real_wf_pf

    wf_perm = get_permutation(df_perm, start_index=train_window, seed=perm_i)
    wf_perm['r'] = np.log(wf_perm['close']).diff().shift(-1)
    wf_perm_sig = walkforward_strategy(wf_perm, strategy, train_lookback=train_window)
    perm_rets = wf_perm['r'] * wf_perm_sig

    pos = perm_rets[perm_rets > 0].sum()
    neg = perm_rets[perm_rets < 0].abs().sum()
    perm_pf = pos / neg if neg != 0 else (np.inf if pos > 0 else 0.0)

    cum_rets = perm_rets.cumsum().values
    is_better = 1 if perm_pf >= real_wf_pf else 0
    return perm_pf, is_better, cum_rets
