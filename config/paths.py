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
