"""
Parallel configuration loader.

Loads configuration from parallel_config.yaml for use in orchestration modules.
"""

import yaml
from pathlib import Path


def load_config() -> dict:
    """
    Load parallel configuration from YAML file.

    Returns:
        dict: Configuration dictionary with keys:
            - default: Default settings
            - multiprocess: Multiprocessing settings
            - dask: Dask settings
            - backtest: Backtest-specific settings
            - mcpt: MCPT settings
            - walkforward: Walk-forward settings
    """
    config_path = Path(__file__).parent / 'parallel_config.yaml'

    if not config_path.exists():
        # Return default config if file doesn't exist
        return get_default_config()

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config() -> dict:
    """
    Return default configuration for 16-core workstation.

    Returns:
        dict: Default configuration
    """
    return {
        'default': {
            'backend': 'multiprocess',
            'n_workers': 15,
            'chunk_size': 'auto'
        },
        'multiprocess': {
            'n_workers': 15,
            'maxtasksperchild': 10
        },
        'dask': {
            'scheduler': 'processes',
            'n_workers': 15,
            'threads_per_worker': 1,
            'memory_limit': '4GB'
        },
        'backtest': {
            'default_symbols': ['BTCUSD', 'ETHUSD'],
            'default_start': '2020-01-01',
            'default_end': '2024-01-01',
            'save_results': True,
            'save_figures': True,
            'figure_format': 'png',
            'figure_dpi': 150
        },
        'mcpt': {
            'n_permutations': 1000,
            'confidence_level': 0.95,
            'parallel_permutations': True
        },
        'walkforward': {
            'train_window': 365,
            'test_window': 30,
            'step_size': 30
        }
    }


def get_dask_config() -> dict:
    """
    Get Dask-specific configuration.

    Returns:
        dict: Dask configuration
    """
    config = load_config()
    return config.get('dask', get_default_config()['dask'])


def get_multiprocess_config() -> dict:
    """
    Get multiprocessing-specific configuration.

    Returns:
        dict: Multiprocessing configuration
    """
    config = load_config()
    return config.get('multiprocess', get_default_config()['multiprocess'])
