"""
Módulo de visualización de estrategias

Contiene herramientas para ejecutar, analizar y visualizar estrategias de trading.
"""

from . import run_strategy
from .ticker_plots import calculate_ticker_statistics, plot_ticker_results
from .batch_plots import calculate_batch_statistics, plot_batch_results
from .bootstrap_plots import plot_bootstrap_results, plot_bootstrap_batch_comparison
from .report import (
    generate_mcpt_ticker_report,
    generate_mcpt_batch_report,
    generate_bootstrap_ticker_report,
    generate_bootstrap_batch_report,
)

__all__ = [
    'run_strategy',
    # Ticker plots
    'calculate_ticker_statistics',
    'plot_ticker_results',
    # Batch plots
    'calculate_batch_statistics',
    'plot_batch_results',
    # Bootstrap plots
    'plot_bootstrap_results',
    'plot_bootstrap_batch_comparison',
    # Reports
    'generate_mcpt_ticker_report',
    'generate_mcpt_batch_report',
    'generate_bootstrap_ticker_report',
    'generate_bootstrap_batch_report',
]
