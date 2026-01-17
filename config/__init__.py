"""
Módulo de configuración del proyecto Trading-System.

Importa las rutas principales para facilitar su uso en otros scripts.
Updated for new project organization.
"""

# Import from both old path.py (for backward compatibility) and new paths.py
try:
    from config.paths import (
        PROJECT_ROOT,
        CONFIG_DIR,
        MCPT_DIR,
        SCRIPTS_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        OUTPUTS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        BITCOIN_CSV,
        BITCOIN_PARQUET,
        MODELS_DIR,
        STRATEGIES_DIR,
        FILTERS_DIR,
        BACKTEST_DIR,
        BACKTEST_RESULTS,
        BACKTEST_REPORTS,
        BACKTEST_FIGURES,
        BACKTEST_CODE_DIR,
        VISUALIZATION_DIR,
        VIS_NON_INTERACTIVE,
        VIS_INTERACTIVE,
        DOCUMENTATION_DIR,
        SCREENING_DIR,
        ensure_directories,
        get_plot_path,
        get_output_path,
        get_data_path,
        print_paths
    )
except ImportError:
    # Fallback to old path.py if paths.py not available
    from config.path import (
        PROJECT_ROOT,
        CONFIG_DIR,
        MCPT_DIR,
        SCRIPTS_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        BITCOIN_CSV,
        BITCOIN_PARQUET,
        ensure_directories,
        get_plot_path,
        get_output_path,
        print_paths
    )
    # Define missing variables for compatibility
    OUTPUTS_DIR = OUTPUT_DIR
    MODELS_DIR = PROJECT_ROOT / "models"
    STRATEGIES_DIR = MODELS_DIR / "strategies"
    FILTERS_DIR = MODELS_DIR / "filters"
    BACKTEST_DIR = OUTPUT_DIR / "backtest"
    BACKTEST_RESULTS = BACKTEST_DIR / "results"
    BACKTEST_REPORTS = BACKTEST_DIR / "reports"
    BACKTEST_FIGURES = BACKTEST_DIR / "figures"
    BACKTEST_CODE_DIR = PROJECT_ROOT / "backtest"
    VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
    VIS_NON_INTERACTIVE = VISUALIZATION_DIR / "non_interactive"
    VIS_INTERACTIVE = VISUALIZATION_DIR / "interactive"
    DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
    SCREENING_DIR = PROJECT_ROOT / "screening"
    get_data_path = lambda symbol, filename: DATA_DIR / symbol / filename

__all__ = [
    'PROJECT_ROOT',
    'CONFIG_DIR',
    'MCPT_DIR',
    'SCRIPTS_DIR',
    'DATA_DIR',
    'OUTPUT_DIR',
    'OUTPUTS_DIR',
    'PLOTS_DIR',
    'LOGS_DIR',
    'BITCOIN_CSV',
    'BITCOIN_PARQUET',
    'MODELS_DIR',
    'STRATEGIES_DIR',
    'FILTERS_DIR',
    'BACKTEST_DIR',
    'BACKTEST_RESULTS',
    'BACKTEST_REPORTS',
    'BACKTEST_FIGURES',
    'BACKTEST_CODE_DIR',
    'VISUALIZATION_DIR',
    'VIS_NON_INTERACTIVE',
    'VIS_INTERACTIVE',
    'DOCUMENTATION_DIR',
    'SCREENING_DIR',
    'ensure_directories',
    'get_plot_path',
    'get_output_path',
    'get_data_path',
    'print_paths'
]
