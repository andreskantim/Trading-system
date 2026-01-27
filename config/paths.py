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

# ====================================================================
# DIRECTORIOS PRINCIPALES
# ====================================================================

MODELS_DIR = PROJECT_ROOT / "models"
STRATEGIES_DIR = MODELS_DIR / "strategies"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ORCHESTRATION_DIR = PROJECT_ROOT / "orchestration"


# ====================================================================
# DIRECTORIOS DE SALIDA
# ====================================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs" 
BACKTEST_OUTPUTS_DIR = OUTPUTS_DIR / "backtest"
LOGS_DIR = PROJECT_ROOT / "logs"

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
        OUTPUTS_DIR,
        BACKTEST_OUTPUTS_DIR, 
        LOGS_DIR,
        SCREENING_DIR,
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


# ====================================================================
# NEW OUTPUT STRUCTURE FUNCTIONS
# ====================================================================

def get_strategy_OUTPUTS_DIR(strategy: str) -> Path:
    """Get base output directory for a strategy."""
    return OUTPUTS_DIR / strategy


def get_ticker_OUTPUTS_DIR(strategy: str, ticker: str, subdir: str = None) -> Path:
    """outputs/backtest/{strategy}/{ticker}/[figures|interactive|results|reports]"""
    base_path = BACKTEST_OUTPUTS_DIR / strategy / ticker
    return base_path / subdir if subdir else base_path


def get_batch_OUTPUTS_DIR(strategy: str, batch_name: str, subdir: str = None) -> Path:
    """outputs/backtest/{strategy}/{batch_name}/[figures|results|reports]"""
    base_path = BACKTEST_OUTPUTS_DIR / strategy / batch_name
    return base_path / subdir if subdir else base_path


def ensure_ticker_OUTPUTS_DIRs(strategy: str, ticker: str) -> dict:
    """Create ticker output directories: figures, interactive, results, reports"""
    dirs = {
        'figures': get_ticker_OUTPUTS_DIR(strategy, ticker, 'figures'),
        'interactive': get_ticker_OUTPUTS_DIR(strategy, ticker, 'interactive'),
        'results': get_ticker_OUTPUTS_DIR(strategy, ticker, 'results'),
        'reports': get_ticker_OUTPUTS_DIR(strategy, ticker, 'reports'),
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_batch_OUTPUTS_DIRs(strategy: str, batch_name: str) -> dict:
    """
    Create and return batch output directories.

    Args:
        strategy: Strategy name
        batch_name: Batch identifier

    Returns:
        Dict with paths to figures, results, reports directories
    """
    dirs = {
        'figures': get_batch_OUTPUTS_DIR(strategy, batch_name, 'figures'),
        'results': get_batch_OUTPUTS_DIR(strategy, batch_name, 'results'),
        'reports': get_batch_OUTPUTS_DIR(strategy, batch_name, 'reports'),
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
