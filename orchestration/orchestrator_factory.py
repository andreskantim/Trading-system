"""
Factory for creating orchestrator instances.

Provides a unified interface to create the appropriate orchestrator
based on the requested backend.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml

from orchestration.base_orchestrator import BaseOrchestrator
from orchestration.sequential_runner import SequentialRunner
from orchestration.multiprocess_runner import MultiprocessRunner
from orchestration.dask_runner import DaskRunner


BACKEND_MAP = {
    'sequential': SequentialRunner,
    'multiprocess': MultiprocessRunner,
    'dask': DaskRunner
}


def get_orchestrator(
    backend: str = 'multiprocess',
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    **kwargs
) -> BaseOrchestrator:
    """
    Factory function to create an orchestrator instance.

    Args:
        backend: Execution backend ('sequential', 'multiprocess', 'dask')
                 Default is 'multiprocess' for 16-core workstation
        config: Configuration dictionary, or path to YAML config file
        **kwargs: Additional keyword arguments passed to orchestrator constructor

    Returns:
        BaseOrchestrator instance configured for the specified backend

    Raises:
        ValueError: If backend is not recognized

    Example:
        >>> orchestrator = get_orchestrator('multiprocess', n_workers=14)
        >>> with orchestrator:
        ...     results = orchestrator.backtest_parallel(symbols, params, start, end)
    """
    # Normalize backend name
    backend = backend.lower().strip()

    if backend not in BACKEND_MAP:
        available = ', '.join(BACKEND_MAP.keys())
        raise ValueError(
            f"Unknown backend: '{backend}'. Available backends: {available}"
        )

    # Load config if it's a file path
    if isinstance(config, (str, Path)):
        config = load_config(config)

    # Merge config with kwargs (kwargs take precedence)
    if config is not None:
        # Get backend-specific config
        backend_config = config.get(backend, {})
        # Merge with default config
        default_config = config.get('default', {})
        merged_config = {**default_config, **backend_config, **kwargs}
    else:
        merged_config = kwargs

    # Create and return orchestrator
    orchestrator_class = BACKEND_MAP[backend]
    return orchestrator_class(**merged_config)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_available_backends() -> list:
    """
    Get list of available backend names.

    Returns:
        List of available backend names
    """
    return list(BACKEND_MAP.keys())
