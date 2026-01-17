"""
Dask-based runner for advanced parallel workflows.

Uses Dask with local scheduler for optional advanced parallelization.
Configured for 16-core workstation with processes scheduler.
"""

from typing import List, Dict, Any, Optional
from itertools import product
import pandas as pd

from orchestration.base_orchestrator import BaseOrchestrator


class DaskRunner(BaseOrchestrator):
    """
    Dask runner using local scheduler with processes.

    For advanced workflows that benefit from Dask's features:
    - Task graphs and lazy evaluation
    - Memory-efficient processing
    - Dynamic task scheduling
    """

    def __init__(
        self,
        n_workers: int = 14,
        config: Optional[Dict[str, Any]] = None,
        threads_per_worker: int = 1,
        memory_limit: str = '4GB'
    ):
        """
        Initialize Dask runner.

        Args:
            n_workers: Number of worker processes (default: 14)
            config: Optional configuration dictionary
            threads_per_worker: Threads per worker (default: 1 for CPU-bound tasks)
            memory_limit: Memory limit per worker (default: 4GB)
        """
        super().__init__(n_workers=n_workers, config=config)
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self._client = None
        self._cluster = None

    def initialize(self) -> None:
        """Initialize Dask local cluster."""
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError:
            raise ImportError(
                "Dask distributed is required for DaskRunner. "
                "Install with: pip install dask[distributed]"
            )

        if self._client is not None:
            self.close()

        self._cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            processes=True  # Use processes for CPU-bound tasks
        )
        self._client = Client(self._cluster)
        self._is_initialized = True

    def backtest_parallel(
        self,
        symbols: List[str],
        strategy_params: Dict[str, Any],
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Run backtests in parallel using Dask.

        Args:
            symbols: List of ticker symbols to backtest
            strategy_params: Dictionary containing strategy configuration
            start: Start date for backtest
            end: End date for backtest

        Returns:
            DataFrame containing backtest results for all symbols
        """
        if not self._is_initialized or self._client is None:
            self.initialize()

        from dask import delayed
        import dask

        # Create delayed tasks
        tasks = []
        for symbol in symbols:
            task = delayed(_dask_backtest_worker)(
                symbol, strategy_params, start, end
            )
            tasks.append(task)

        # Execute all tasks
        results = dask.compute(*tasks)

        return pd.DataFrame(results)

    def optimize_parameters(
        self,
        symbol: str,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters in parallel using Dask.

        Args:
            symbol: Ticker symbol to optimize for
            param_grid: Dictionary mapping parameter names to lists of values

        Returns:
            DataFrame containing optimization results
        """
        if not self._is_initialized or self._client is None:
            self.initialize()

        from dask import delayed
        import dask

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Create delayed tasks
        tasks = []
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            task = delayed(_dask_optimization_worker)(symbol, params)
            tasks.append(task)

        # Execute all tasks
        results = dask.compute(*tasks)

        return pd.DataFrame(results)

    def submit(self, func, *args, **kwargs):
        """
        Submit a single task to the Dask cluster.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Future object representing the pending result
        """
        if not self._is_initialized or self._client is None:
            self.initialize()

        return self._client.submit(func, *args, **kwargs)

    def map(self, func, iterable) -> List[Any]:
        """
        Map a function over an iterable using Dask.

        Args:
            func: Function to apply to each element
            iterable: Iterable of inputs

        Returns:
            List of results
        """
        if not self._is_initialized or self._client is None:
            self.initialize()

        futures = self._client.map(func, iterable)
        return self._client.gather(futures)

    def get_dashboard_link(self) -> Optional[str]:
        """Get the Dask dashboard URL if available."""
        if self._cluster is not None:
            return self._cluster.dashboard_link
        return None

    def close(self) -> None:
        """Close Dask client and cluster."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None
        self._is_initialized = False


def _dask_backtest_worker(
    symbol: str,
    strategy_params: Dict[str, Any],
    start: str,
    end: str
) -> Dict[str, Any]:
    """
    Worker function for Dask backtest execution.

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


def _dask_optimization_worker(
    symbol: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Worker function for Dask optimization execution.

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
