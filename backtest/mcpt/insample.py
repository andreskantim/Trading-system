"""Core in-sample MCPT functions."""

import numpy as np
import pandas as pd
from backtest.mcpt.bar_permute import get_permutation

# Global worker state
_strategy_module = None
_train_df = None
_best_real_pf = None


def init_insample_worker(strategy_name, train_data, train_index, train_columns, best_real_pf):
    """Initialize worker with strategy module and data."""
    import importlib
    global _strategy_module, _train_df, _best_real_pf
    _strategy_module = importlib.import_module(f'models.strategies.{strategy_name}')
    _train_df = pd.DataFrame(train_data, index=train_index, columns=train_columns)
    _best_real_pf = best_real_pf


def process_permutation(perm_i):
    """Process single in-sample permutation."""
    strategy = _strategy_module
    train_df = _train_df
    best_real_pf = _best_real_pf

    train_perm = get_permutation(train_df, seed=perm_i)
    result = strategy.optimize(train_perm)
    *best_params, best_perm_pf = result

    sig = strategy.signal(train_perm, *best_params)
    r = np.log(train_perm['close']).diff().shift(-1)
    perm_rets = sig * r
    cum_rets = perm_rets.cumsum().values

    is_better = 1 if best_perm_pf >= best_real_pf else 0
    return best_perm_pf, is_better, cum_rets
