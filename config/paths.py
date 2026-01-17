"""
Configuración de rutas del proyecto Trading-System.

Este módulo centraliza todas las rutas clave del proyecto usando paths relativos,
lo que permite mover el proyecto a cualquier máquina manteniendo su estructura.

Updated for new project organization optimized for 16-core local workstation.
"""

from pathlib import Path

# ====================================================================
# RUTAS BASE DEL PROYECTO
# ====================================================================

# Raíz del proyecto (directorio que contiene config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio de configuración
CONFIG_DIR = PROJECT_ROOT / "config"

# ====================================================================
# DIRECTORIOS PRINCIPALES
# ====================================================================

# Directorio de modelos (estrategias y filtros)
MODELS_DIR = PROJECT_ROOT / "models"
STRATEGIES_DIR = MODELS_DIR / "strategies"
FILTERS_DIR = MODELS_DIR / "filters"

# Directorio de scripts ejecutables
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Directorio de orquestación
ORCHESTRATION_DIR = PROJECT_ROOT / "orchestration"

# ====================================================================
# DIRECTORIOS DE DATOS
# ====================================================================

# Directorio principal de datos
DATA_DIR = PROJECT_ROOT / "data"
BTCUSD_DATA_DIR = DATA_DIR / "BTCUSD"
ETHUSD_DATA_DIR = DATA_DIR / "ETHUSD"

# Archivos de datos principales
BITCOIN_CSV = BTCUSD_DATA_DIR / "bitcoin_hourly.csv"
BITCOIN_PARQUET = BTCUSD_DATA_DIR / "BTCUSD3600.pq"

# ====================================================================
# DIRECTORIOS DE SALIDA
# ====================================================================

# Directorio principal de salidas
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Subdirectorios de backtest
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
BACKTEST_RESULTS = BACKTEST_DIR / "results"
BACKTEST_REPORTS = BACKTEST_DIR / "reports"
BACKTEST_FIGURES = BACKTEST_DIR / "figures"

# Alias para compatibilidad
OUTPUT_DIR = OUTPUTS_DIR
PLOTS_DIR = BACKTEST_FIGURES
LOGS_DIR = BACKTEST_DIR / "logs"

# ====================================================================
# DIRECTORIOS DE BACKTEST Y MCPT
# ====================================================================

# Directorio de backtest (código y utilidades)
BACKTEST_CODE_DIR = PROJECT_ROOT / "backtest"
MCPT_DIR = BACKTEST_CODE_DIR / "mcpt"
BOOTSTRAP_DIR = BACKTEST_CODE_DIR / "bootstrap"

# ====================================================================
# DIRECTORIOS DE VISUALIZACIÓN
# ====================================================================

VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
VIS_NON_INTERACTIVE = VISUALIZATION_DIR / "non_interactive"
VIS_INTERACTIVE = VISUALIZATION_DIR / "interactive"
VIS_UTILS = VISUALIZATION_DIR / "utils"

# ====================================================================
# DIRECTORIOS DE DOCUMENTACIÓN
# ====================================================================

DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
PERM_ENTROPY_DIR = DOCUMENTATION_DIR / "PermutationEntropy"
TRADE_DEPENDENCE_DIR = DOCUMENTATION_DIR / "TradeDependenceRunTest"
VOLATILITY_HAWKES_DIR = DOCUMENTATION_DIR / "VolatilityHawkes"

# ====================================================================
# DIRECTORIO DE SCREENING
# ====================================================================

SCREENING_DIR = PROJECT_ROOT / "screening"

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

def ensure_directories():
    """
    Crea los directorios necesarios si no existen.
    Llamar esta función al inicio de los scripts principales.
    """
    dirs_to_create = [
        DATA_DIR,
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


def get_plot_path(filename: str) -> Path:
    """
    Genera una ruta completa para guardar un gráfico.

    Args:
        filename (str): Nombre del archivo (puede incluir subdirectorios)

    Returns:
        Path: Ruta completa al archivo de gráfico

    Example:
        >>> plot_path = get_plot_path('insample_mcpt_pval_0.0450.png')
        >>> plt.savefig(plot_path)
    """
    return BACKTEST_FIGURES / filename


def get_output_path(filename: str) -> Path:
    """
    Genera una ruta completa para guardar cualquier archivo de salida.

    Args:
        filename (str): Nombre del archivo

    Returns:
        Path: Ruta completa al archivo de salida
    """
    return BACKTEST_RESULTS / filename


def get_data_path(symbol: str, filename: str) -> Path:
    """
    Genera una ruta completa para un archivo de datos de un símbolo específico.

    Args:
        symbol (str): Símbolo del activo (e.g., 'BTCUSD', 'ETHUSD')
        filename (str): Nombre del archivo de datos

    Returns:
        Path: Ruta completa al archivo de datos
    """
    return DATA_DIR / symbol / filename


# ====================================================================
# INFORMACIÓN DEL PROYECTO
# ====================================================================

def print_paths():
    """
    Imprime todas las rutas configuradas (útil para debugging).
    """
    print("=" * 70)
    print("CONFIGURACIÓN DE RUTAS DEL PROYECTO")
    print("=" * 70)
    print(f"PROJECT_ROOT:       {PROJECT_ROOT}")
    print(f"CONFIG_DIR:         {CONFIG_DIR}")
    print()
    print("Modelos:")
    print(f"  MODELS_DIR:       {MODELS_DIR}")
    print(f"  STRATEGIES_DIR:   {STRATEGIES_DIR}")
    print(f"  FILTERS_DIR:      {FILTERS_DIR}")
    print()
    print("Datos:")
    print(f"  DATA_DIR:         {DATA_DIR}")
    print(f"  BITCOIN_CSV:      {BITCOIN_CSV}")
    print(f"  BITCOIN_PARQUET:  {BITCOIN_PARQUET}")
    print()
    print("Salidas:")
    print(f"  OUTPUTS_DIR:      {OUTPUTS_DIR}")
    print(f"  BACKTEST_RESULTS: {BACKTEST_RESULTS}")
    print(f"  BACKTEST_REPORTS: {BACKTEST_REPORTS}")
    print(f"  BACKTEST_FIGURES: {BACKTEST_FIGURES}")
    print()
    print("Código de backtest:")
    print(f"  BACKTEST_CODE_DIR:{BACKTEST_CODE_DIR}")
    print(f"  MCPT_DIR:         {MCPT_DIR}")
    print()
    print("Visualización:")
    print(f"  VISUALIZATION_DIR:{VISUALIZATION_DIR}")
    print(f"  VIS_NON_INTERACTIVE: {VIS_NON_INTERACTIVE}")
    print(f"  VIS_INTERACTIVE:  {VIS_INTERACTIVE}")
    print()
    print("Scripts:")
    print(f"  SCRIPTS_DIR:      {SCRIPTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    # Si se ejecuta directamente, muestra las rutas configuradas
    print_paths()

    # Crear directorios si no existen
    print("\nCreando directorios necesarios...")
    ensure_directories()
    print("Directorios creados correctamente.")
