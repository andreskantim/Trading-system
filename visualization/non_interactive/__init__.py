"""
Módulo de visualización de estrategias

Contiene herramientas para ejecutar, analizar y visualizar estrategias de trading.
"""

from . import run_strategy
from .stats_and_plots_ticker import calculate_ticker_statistics, plot_ticker_results
from .stats_and_plots_batch import calculate_batch_statistics, plot_batch_results

__all__ = [
    'run_strategy',
    'calculate_ticker_statistics',
    'plot_ticker_results',
    'calculate_batch_statistics',
    'plot_batch_results',
]
