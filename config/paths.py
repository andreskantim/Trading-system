"""
Configuración de rutas del proyecto Trading-System.

Solo rutas de directorios. La configuración de tickers está en tickers.py.
"""

from pathlib import Path

# ====================================================================
# RUTAS BASE DEL PROYECTO
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# ====================================================================
# DIRECTORIOS DE DATOS
# ====================================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OPERATIVE_DATA_DIR = DATA_DIR / "operative"

# Legacy paths (compatibilidad)
BTCUSD_DATA_DIR = DATA_DIR / "BTCUSD"
ETHUSD_DATA_DIR = DATA_DIR / "ETHUSD"
BITCOIN_CSV = BTCUSD_DATA_DIR / "bitcoin_hourly.csv"
BITCOIN_PARQUET = BTCUSD_DATA_DIR / "BTCUSD3600.pq"

# Legacy TICKERS dict (compatibilidad con scripts existentes)
TICKERS = {
    'BTCUSD': {
        'csv': DATA_DIR / 'BTCUSD' / 'bitcoin_hourly.csv',
        'parquet': DATA_DIR / 'BTCUSD' / 'BTCUSD3600.pq'
    },
    'ETHUSD': {
        'csv': DATA_DIR / 'ETHUSD' / 'ethereum_hourly.csv',
        'parquet': DATA_DIR / 'ETHUSD' / 'ETHUSD3600.pq'
    },
}

# ====================================================================
# DIRECTORIOS PRINCIPALES
# ====================================================================

MODELS_DIR = PROJECT_ROOT / "models"
STRATEGIES_DIR = MODELS_DIR / "strategies"
FILTERS_DIR = MODELS_DIR / "filters"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ORCHESTRATION_DIR = PROJECT_ROOT / "orchestration"

# ====================================================================
# DIRECTORIOS DE SALIDA
# ====================================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
BACKTEST_RESULTS = BACKTEST_DIR / "results"
BACKTEST_REPORTS = BACKTEST_DIR / "reports"
BACKTEST_FIGURES = BACKTEST_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Alias para compatibilidad
OUTPUT_DIR = OUTPUTS_DIR
PLOTS_DIR = BACKTEST_FIGURES

# ====================================================================
# OTROS DIRECTORIOS
# ====================================================================

BACKTEST_CODE_DIR = PROJECT_ROOT / "backtest"
MCPT_DIR = BACKTEST_CODE_DIR / "mcpt"
BOOTSTRAP_DIR = BACKTEST_CODE_DIR / "bootstrap"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
VIS_NON_INTERACTIVE = VISUALIZATION_DIR / "non_interactive"
VIS_INTERACTIVE = VISUALIZATION_DIR / "interactive"
VIS_UTILS = VISUALIZATION_DIR / "utils"
DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
PERM_ENTROPY_DIR = DOCUMENTATION_DIR / "PermutationEntropy"
TRADE_DEPENDENCE_DIR = DOCUMENTATION_DIR / "TradeDependenceRunTest"
VOLATILITY_HAWKES_DIR = DOCUMENTATION_DIR / "VolatilityHawkes"
SCREENING_DIR = PROJECT_ROOT / "screening"

# ====================================================================
# PARÁMETROS GENERALES
# ====================================================================

OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
PARQUET_COMPRESSION = 'snappy'
PARQUET_ENGINE = 'pyarrow'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

def ensure_directories():
    """Crea los directorios necesarios si no existen."""
    dirs_to_create = [
        DATA_DIR,
        RAW_DATA_DIR,
        OPERATIVE_DATA_DIR,
        BTCUSD_DATA_DIR,
        ETHUSD_DATA_DIR,
        OUTPUTS_DIR,
        BACKTEST_DIR,
        BACKTEST_RESULTS,
        BACKTEST_REPORTS,
        BACKTEST_FIGURES,
        LOGS_DIR,
        SCREENING_DIR,
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_ticker_data_paths(ticker_name: str) -> dict:
    """Get data paths for a specific ticker (legacy)."""
    ticker_upper = ticker_name.upper()
    if ticker_upper not in TICKERS:
        available = ', '.join(TICKERS.keys())
        raise ValueError(f"Ticker '{ticker_name}' not configured. Available: {available}")
    return TICKERS[ticker_upper]


def get_plot_path(filename: str) -> Path:
    """Genera ruta para guardar un gráfico."""
    return BACKTEST_FIGURES / filename


def get_output_path(filename: str) -> Path:
    """Genera ruta para guardar archivo de salida."""
    return BACKTEST_RESULTS / filename


def get_data_path(symbol: str, filename: str) -> Path:
    """Genera ruta para archivo de datos de un símbolo."""
    return DATA_DIR / symbol / filename


# ====================================================================
# NEW OUTPUT STRUCTURE FUNCTIONS
# ====================================================================

def get_strategy_output_dir(strategy: str) -> Path:
    """Get base output directory for a strategy."""
    return OUTPUTS_DIR / strategy


def get_ticker_output_dir(strategy: str, ticker: str, subdir: str = None) -> Path:
    """
    Get output directory for a specific ticker under a strategy.

    Args:
        strategy: Strategy name (e.g., 'hawkes', 'donchian')
        ticker: Ticker symbol (e.g., 'BTC', 'ETH')
        subdir: Optional subdirectory ('figures', 'results', 'reports')

    Returns:
        Path to the ticker output directory

    Structure:
        outputs/{strategy}/ticker/{ticker}/[subdir]
    """
    base_path = OUTPUTS_DIR / strategy / "ticker" / ticker
    if subdir:
        return base_path / subdir
    return base_path


def get_batch_output_dir(strategy: str, batch_name: str, subdir: str = None) -> Path:
    """
    Get output directory for a batch run under a strategy.

    Args:
        strategy: Strategy name (e.g., 'hawkes', 'donchian')
        batch_name: Batch identifier (e.g., 'crypto_10_insample', '2024_01_15')
        subdir: Optional subdirectory ('figures', 'results', 'reports')

    Returns:
        Path to the batch output directory

    Structure:
        outputs/{strategy}/batch/{batch_name}/[subdir]
    """
    base_path = OUTPUTS_DIR / strategy / "batch" / batch_name
    if subdir:
        return base_path / subdir
    return base_path


def ensure_ticker_output_dirs(strategy: str, ticker: str) -> dict:
    """
    Create and return ticker output directories.

    Args:
        strategy: Strategy name
        ticker: Ticker symbol

    Returns:
        Dict with paths to figures, results, reports directories
    """
    dirs = {
        'figures': get_ticker_output_dir(strategy, ticker, 'figures'),
        'results': get_ticker_output_dir(strategy, ticker, 'results'),
        'reports': get_ticker_output_dir(strategy, ticker, 'reports'),
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_batch_output_dirs(strategy: str, batch_name: str) -> dict:
    """
    Create and return batch output directories.

    Args:
        strategy: Strategy name
        batch_name: Batch identifier

    Returns:
        Dict with paths to figures, results, reports directories
    """
    dirs = {
        'figures': get_batch_output_dir(strategy, batch_name, 'figures'),
        'results': get_batch_output_dir(strategy, batch_name, 'results'),
        'reports': get_batch_output_dir(strategy, batch_name, 'reports'),
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def print_paths():
    """Imprime las rutas configuradas."""
    print("=" * 70)
    print("RUTAS DEL PROYECTO")
    print("=" * 70)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR:     {DATA_DIR}")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"LOGS_DIR:     {LOGS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_paths()
    ensure_directories()
    print("Directorios creados.")
