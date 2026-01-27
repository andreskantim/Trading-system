"""
Módulo de visualización de estrategias

Contiene herramientas para ejecutar, analizar y visualizar estrategias de trading.
"""

from . import run_strategy
from .ticker_plots import calculate_ticker_statistics, plot_ticker_results
from .batch_plots import calculate_batch_statistics, plot_batch_results

__all__ = [
    'run_strategy',
    'calculate_ticker_statistics',
    'plot_ticker_results',
    'calculate_batch_statistics',
    'plot_batch_results',
]
