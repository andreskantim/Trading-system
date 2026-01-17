"""
Sequential execution runner for debugging and testing.

Runs backtests sequentially without parallelization, useful for debugging
strategy logic and ensuring correct behavior before parallel execution.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from itertools import product

from orchestration.base_orchestrator import BaseOrchestrator


class SequentialRunner(BaseOrchestrator):
    """
    Sequential runner that executes backtests one at a time.

    Use this for:
    - Debugging strategy logic
    - Testing individual symbol behavior
    - Profiling before parallel execution
    """

    def __init__(self, n_workers: int = 1, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sequential runner.

        Args:
            n_workers: Ignored for sequential execution (always 1)
            config: Optional configuration dictionary
        """
        super().__init__(n_workers=1, config=config)

    def initialize(self) -> None:
        """Initialize the sequential runner (no-op for sequential execution)."""
        self._is_initialized = True

    def backtest_parallel(
        self,
        symbols: List[str],
        strategy_params: Dict[str, Any],
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Run backtests sequentially for each symbol.

        Args:
            symbols: List of ticker symbols to backtest
            strategy_params: Dictionary containing strategy configuration
            start: Start date for backtest
            end: End date for backtest

        Returns:
            DataFrame containing backtest results for all symbols
        """
        if not self._is_initialized:
            self.initialize()

        results = []

        for symbol in symbols:
            try:
                result = self._run_single_backtest(symbol, strategy_params, start, end)
                results.append(result)
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'status': 'failed'
                })

        return pd.DataFrame(results)

    def optimize_parameters(
        self,
        symbol: str,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters sequentially.

        Args:
            symbol: Ticker symbol to optimize for
            param_grid: Dictionary mapping parameter names to lists of values

        Returns:
            DataFrame containing optimization results
        """
        if not self._is_initialized:
            self.initialize()

        results = []

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combination in product(*param_values):
            params = dict(zip(param_names, combination))

            try:
                result = self._run_single_optimization(symbol, params)
                result.update(params)
                results.append(result)
            except Exception as e:
                result = {'symbol': symbol, 'error': str(e), 'status': 'failed'}
                result.update(params)
                results.append(result)

        return pd.DataFrame(results)

    def _run_single_backtest(
        self,
        symbol: str,
        strategy_params: Dict[str, Any],
        start: str,
        end: str
    ) -> Dict[str, Any]:
        """
        Run a single backtest.

        Override this method to implement actual backtest logic.
        """
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

    def _run_single_optimization(
        self,
        symbol: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single optimization iteration.

        Override this method to implement actual optimization logic.
        """
        # Placeholder - implement actual optimization logic
        return {
            'symbol': symbol,
            'status': 'completed',
            'objective': 0.0
        }

    def close(self) -> None:
        """Clean up resources (no-op for sequential execution)."""
        self._is_initialized = False
