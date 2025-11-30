"""
Módulo de estrategias de trading

Este módulo contiene las implementaciones core de diferentes estrategias de trading.
Cada estrategia debe implementar:
  - signal(ohlc, *params): Genera señales de trading
  - optimize(ohlc): Optimiza parámetros y retorna (params..., profit_factor)
"""

from . import donchian
from . import moving_average
from . import tree_strat
from . import hawkes
from . import donchian_aft_loss

__all__ = [
    'donchian',
    'moving_average',
    'tree_strat',
    'hawkes',
    'donchian_aft_loss',
]
