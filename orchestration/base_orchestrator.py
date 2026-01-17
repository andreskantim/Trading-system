"""
Base orchestrator abstract class for parallel backtest execution.

Provides the interface that all orchestrator implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseOrchestrator(ABC):
    """
    Abstract base class for orchestrating parallel backtest execution.

    All orchestrator implementations (sequential, multiprocess, dask) must
    inherit from this class and implement its abstract methods.
    """

    def __init__(self, n_workers: int = 14, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator.

        Args:
            n_workers: Number of worker processes to use (default: 14, leaving 2 cores for system)
            config: Optional configuration dictionary
        """
        self.n_workers = n_workers
        self.config = config or {}
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the execution backend (create pools, clusters, etc.).
        Must be called before running any backtests.
        """
        pass

    @abstractmethod
    def backtest_parallel(
        self,
        symbols: List[str],
        strategy_params: Dict[str, Any],
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Run backtests in parallel across multiple symbols.

        Args:
            symbols: List of ticker symbols to backtest
            strategy_params: Dictionary containing strategy configuration
            start: Start date for backtest (YYYY-MM-DD format)
            end: End date for backtest (YYYY-MM-DD format)

        Returns:
            DataFrame containing backtest results for all symbols
        """
        pass

    @abstractmethod
    def optimize_parameters(
        self,
        symbol: str,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters in parallel.

        Args:
            symbol: Ticker symbol to optimize for
            param_grid: Dictionary mapping parameter names to lists of values to test

        Returns:
            DataFrame containing optimization results for all parameter combinations
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources (close pools, shutdown clusters, etc.).
        Should be called when done with the orchestrator.
        """
        pass

    def __enter__(self):
        """Context manager entry - initialize the backend."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.close()
        return False
