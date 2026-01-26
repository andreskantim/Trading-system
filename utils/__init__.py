"""
Utils module - utility scripts and data handling.

Contains:
- data_loader: Load ticker data from operative directory
- stats_calculator: Comprehensive statistics for backtest results
- download_crypto: Download crypto data from Kraken and Binance
- operative_data: Consolidate exchange data into operative datasets
- run_parallel_backtest: Orchestrate parallel backtest execution
- test_migration: Test script for verifying project structure
"""

from utils.data_loader import (
    load_ticker_data,
    get_available_date_range,
    get_available_tickers,
)

from utils.stats_calculator import (
    calculate_all_stats,
    calculate_batch_stats,
    flatten_stats,
    get_key_metrics,
)

__all__ = [
    'load_ticker_data',
    'get_available_date_range',
    'get_available_tickers',
    'calculate_all_stats',
    'calculate_batch_stats',
    'flatten_stats',
    'get_key_metrics',
]
