"""
Multiprocessing-based runner for CPU-bound parallel backtest execution.

Optimized for 16-core local workstation using Python's multiprocessing.Pool.
"""

from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import product
import pandas as pd

from orchestration.base_orchestrator import BaseOrchestrator


class MultiprocessRunner(BaseOrchestrator):
    """
    Multiprocessing runner using Python's multiprocessing.Pool.

    Optimized for CPU-bound tasks on a 16-core workstation.
    Default configuration uses 14 workers, leaving 2 cores for system.
    """

    def __init__(
        self,
        n_workers: int = 14,
        config: Optional[Dict[str, Any]] = None,
        maxtasksperchild: int = 10
    ):
        """
        Initialize multiprocessing runner.

        Args:
            n_workers: Number of worker processes (default: 14)
            config: Optional configuration dictionary
            maxtasksperchild: Max tasks per worker before restart (prevents memory leaks)
        """
        # Limit workers to available CPUs minus 2
        max_workers = max(1, cpu_count() - 2)
        n_workers = min(n_workers, max_workers)

        super().__init__(n_workers=n_workers, config=config)
        self.maxtasksperchild = maxtasksperchild
        self._pool: Optional[Pool] = None

    def initialize(self) -> None:
        """Initialize the multiprocessing pool."""
        if self._pool is not None:
            self.close()

        self._pool = Pool(
            processes=self.n_workers,
            maxtasksperchild=self.maxtasksperchild
        )
        self._is_initialized = True

    def backtest_parallel(
        self,
        symbols: List[str],
        strategy_params: Dict[str, Any],
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Run backtests in parallel using multiprocessing.

        Args:
            symbols: List of ticker symbols to backtest
            strategy_params: Dictionary containing strategy configuration
            start: Start date for backtest
            end: End date for backtest

        Returns:
            DataFrame containing backtest results for all symbols
        """
        if not self._is_initialized or self._pool is None:
            self.initialize()

        # Create partial function with fixed parameters
        backtest_func = partial(
            _run_backtest_worker,
            strategy_params=strategy_params,
            start=start,
            end=end
        )

        # Execute in parallel
        results = self._pool.map(backtest_func, symbols)

        return pd.DataFrame(results)

    def optimize_parameters(
        self,
        symbol: str,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters in parallel.

        Args:
            symbol: Ticker symbol to optimize for
            param_grid: Dictionary mapping parameter names to lists of values

        Returns:
            DataFrame containing optimization results
        """
        if not self._is_initialized or self._pool is None:
            self.initialize()

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]

        # Create partial function with fixed symbol
        optimize_func = partial(_run_optimization_worker, symbol=symbol)

        # Execute in parallel
        results = self._pool.map(optimize_func, combinations)

        return pd.DataFrame(results)

    def map(self, func, iterable, chunksize: int = 1) -> List[Any]:
        """
        Generic parallel map function.

        Args:
            func: Function to apply to each element
            iterable: Iterable of inputs
            chunksize: Number of items per worker task

        Returns:
            List of results
        """
        if not self._is_initialized or self._pool is None:
            self.initialize()

        return self._pool.map(func, iterable, chunksize=chunksize)

    def starmap(self, func, iterable, chunksize: int = 1) -> List[Any]:
        """
        Generic parallel starmap function.

        Args:
            func: Function to apply to each tuple of arguments
            iterable: Iterable of argument tuples
            chunksize: Number of items per worker task

        Returns:
            List of results
        """
        if not self._is_initialized or self._pool is None:
            self.initialize()

        return self._pool.starmap(func, iterable, chunksize=chunksize)

    def close(self) -> None:
        """Close the multiprocessing pool and clean up."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        self._is_initialized = False


# Worker functions must be defined at module level for pickling
def _run_backtest_worker(
    symbol: str,
    strategy_params: Dict[str, Any],
    start: str,
    end: str
) -> Dict[str, Any]:
    """
    Worker function for running a single backtest.

    Override or replace this function to implement actual backtest logic.
    """
    try:
        # Placeholder - implement actual backtest logic
        return {
            'symbol': symbol,
            'start': start,
            'end': end,
            'status': 'completed',
            'returns': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e),
            'status': 'failed'
        }


def _run_optimization_worker(
    params: Dict[str, Any],
    symbol: str
) -> Dict[str, Any]:
    """
    Worker function for running a single optimization iteration.

    Override or replace this function to implement actual optimization logic.
    """
    try:
        result = {
            'symbol': symbol,
            'status': 'completed',
            'objective': 0.0
        }
        result.update(params)
        return result
    except Exception as e:
        result = {
            'symbol': symbol,
            'error': str(e),
            'status': 'failed'
        }
        result.update(params)
        return result
