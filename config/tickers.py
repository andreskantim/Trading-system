"""
Configuración de tickers y exchanges - ÚNICA fuente de verdad

Este archivo contiene:
- Top 100 cryptos con mapeo entre exchanges (Kraken/Binance)
- Configuración de exchanges (URLs, rate limits, intervalos)
- Configuración de consolidación (priorización, ajuste de volumen)
"""

from pathlib import Path
import sys

# Import paths
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA_DIR, RAW_DATA_DIR, OPERATIVE_DATA_DIR

# ====================================================================
# TOP 100 CRYPTO TICKERS
# ====================================================================
# Mapeo entre exchanges. Si un exchange no tiene el ticker, usar None.

CRYPTO_TICKERS = [
    # Top 1-10
    {'symbol': 'BTC', 'kraken': 'XXBTZUSD', 'binance': 'BTCUSDT', 'kraken_start': '2013-10-01', 'binance_start': '2017-07-14'},
    {'symbol': 'ETH', 'kraken': 'XETHZUSD', 'binance': 'ETHUSDT', 'kraken_start': '2015-08-01', 'binance_start': '2017-08-01'},
    {'symbol': 'BNB', 'kraken': 'BNBUSD', 'binance': 'BNBUSDT', 'kraken_start': '2021-01-01', 'binance_start': '2017-07-14'},
    {'symbol': 'XRP', 'kraken': 'XXRPZUSD', 'binance': 'XRPUSDT', 'kraken_start': '2013-10-01', 'binance_start': '2017-07-14'},
    {'symbol': 'SOL', 'kraken': 'SOLUSD', 'binance': 'SOLUSDT', 'kraken_start': '2021-06-01', 'binance_start': '2020-08-11'},
    {'symbol': 'DOGE', 'kraken': 'XDGUSD', 'binance': 'DOGEUSDT', 'kraken_start': '2014-02-01', 'binance_start': '2019-07-05'},
    {'symbol': 'ADA', 'kraken': 'ADAUSD', 'binance': 'ADAUSDT', 'kraken_start': '2017-10-01', 'binance_start': '2017-10-01'},
    {'symbol': 'TRX', 'kraken': 'TRXUSD', 'binance': 'TRXUSDT', 'kraken_start': '2019-01-01', 'binance_start': '2017-10-01'},
    {'symbol': 'AVAX', 'kraken': 'AVAXUSD', 'binance': 'AVAXUSDT', 'kraken_start': '2020-09-22', 'binance_start': '2020-09-22'},
    {'symbol': 'LINK', 'kraken': 'LINKUSD', 'binance': 'LINKUSDT', 'kraken_start': '2019-03-01', 'binance_start': '2019-01-16'},
    # Top 11-20
    {'symbol': 'TON', 'kraken': 'TONUSD', 'binance': 'TONUSDT', 'kraken_start': '2023-01-01', 'binance_start': '2023-11-01'},
    {'symbol': 'SHIB', 'kraken': 'SHIBUSD', 'binance': 'SHIBUSDT', 'kraken_start': '2021-11-01', 'binance_start': '2021-05-10'},
    {'symbol': 'XLM', 'kraken': 'XXLMZUSD', 'binance': 'XLMUSDT', 'kraken_start': '2014-08-01', 'binance_start': '2017-10-01'},
    {'symbol': 'DOT', 'kraken': 'DOTUSD', 'binance': 'DOTUSDT', 'kraken_start': '2020-08-19', 'binance_start': '2020-08-19'},
    {'symbol': 'BCH', 'kraken': 'BCHUSD', 'binance': 'BCHUSDT', 'kraken_start': '2017-08-01', 'binance_start': '2017-08-01'},
    {'symbol': 'HBAR', 'kraken': 'HBARUSD', 'binance': 'HBARUSDT', 'kraken_start': '2020-11-01', 'binance_start': '2019-09-17'},
    {'symbol': 'UNI', 'kraken': 'UNIUSD', 'binance': 'UNIUSDT', 'kraken_start': '2020-09-17', 'binance_start': '2020-09-17'},
    {'symbol': 'LTC', 'kraken': 'XLTCZUSD', 'binance': 'LTCUSDT', 'kraken_start': '2013-10-01', 'binance_start': '2017-07-14'},
    {'symbol': 'PEPE', 'kraken': 'PEPEUSD', 'binance': 'PEPEUSDT', 'kraken_start': '2023-05-05', 'binance_start': '2023-05-05'},
    {'symbol': 'NEAR', 'kraken': 'NEARUSD', 'binance': 'NEARUSDT', 'kraken_start': '2020-10-14', 'binance_start': '2020-10-14'},
    # Top 21-30
    {'symbol': 'LEO', 'kraken': None, 'binance': 'LEOUSDT', 'kraken_start': None, 'binance_start': '2019-05-20'},
    {'symbol': 'APT', 'kraken': 'APTUSD', 'binance': 'APTUSDT', 'kraken_start': '2022-10-19', 'binance_start': '2022-10-19'},
    {'symbol': 'ICP', 'kraken': 'ICPUSD', 'binance': 'ICPUSDT', 'kraken_start': '2021-05-10', 'binance_start': '2021-05-10'},
    {'symbol': 'ETC', 'kraken': 'XETCZUSD', 'binance': 'ETCUSDT', 'kraken_start': '2016-07-24', 'binance_start': '2018-06-01'},
    {'symbol': 'RENDER', 'kraken': 'RENDERUSD', 'binance': 'RENDERUSDT', 'kraken_start': '2023-11-01', 'binance_start': '2023-11-27'},
    {'symbol': 'FET', 'kraken': 'FETUSD', 'binance': 'FETUSDT', 'kraken_start': '2019-03-01', 'binance_start': '2019-02-28'},
    {'symbol': 'ATOM', 'kraken': 'ATOMUSD', 'binance': 'ATOMUSDT', 'kraken_start': '2019-04-22', 'binance_start': '2019-04-22'},
    {'symbol': 'CRO', 'kraken': 'CROUSD', 'binance': 'CROUSDT', 'kraken_start': '2021-11-08', 'binance_start': '2021-11-08'},
    {'symbol': 'AAVE', 'kraken': 'AAVEUSD', 'binance': 'AAVEUSDT', 'kraken_start': '2020-10-03', 'binance_start': '2020-10-15'},
    {'symbol': 'FIL', 'kraken': 'FILUSD', 'binance': 'FILUSDT', 'kraken_start': '2020-10-15', 'binance_start': '2020-10-15'},
    # Top 31-40
    {'symbol': 'IMX', 'kraken': 'IMXUSD', 'binance': 'IMXUSDT', 'kraken_start': '2022-01-01', 'binance_start': '2021-11-11'},
    {'symbol': 'XMR', 'kraken': 'XXMRZUSD', 'binance': None, 'kraken_start': '2014-05-21', 'binance_start': None},
    {'symbol': 'STX', 'kraken': 'STXUSD', 'binance': 'STXUSDT', 'kraken_start': '2021-01-01', 'binance_start': '2019-10-25'},
    {'symbol': 'ARB', 'kraken': 'ARBUSD', 'binance': 'ARBUSDT', 'kraken_start': '2023-03-23', 'binance_start': '2023-03-23'},
    {'symbol': 'OP', 'kraken': 'OPUSD', 'binance': 'OPUSDT', 'kraken_start': '2022-06-01', 'binance_start': '2022-06-01'},
    {'symbol': 'VET', 'kraken': 'VETUSD', 'binance': 'VETUSDT', 'kraken_start': '2021-04-01', 'binance_start': '2018-07-23'},
    {'symbol': 'MKR', 'kraken': 'MKRUSD', 'binance': 'MKRUSDT', 'kraken_start': '2017-09-15', 'binance_start': '2020-06-08'},
    {'symbol': 'THETA', 'kraken': 'THETAUSD', 'binance': 'THETAUSDT', 'kraken_start': '2021-03-25', 'binance_start': '2019-03-13'},
    {'symbol': 'GRT', 'kraken': 'GRTUSD', 'binance': 'GRTUSDT', 'kraken_start': '2020-12-18', 'binance_start': '2020-12-18'},
    {'symbol': 'INJ', 'kraken': 'INJUSD', 'binance': 'INJUSDT', 'kraken_start': '2020-10-21', 'binance_start': '2020-10-21'},
    # Top 41-50
    {'symbol': 'FTM', 'kraken': 'FTMUSD', 'binance': 'FTMUSDT', 'kraken_start': '2021-04-06', 'binance_start': '2019-06-11'},
    {'symbol': 'ALGO', 'kraken': 'ALGOUSD', 'binance': 'ALGOUSDT', 'kraken_start': '2019-06-21', 'binance_start': '2019-06-22'},
    {'symbol': 'MATIC', 'kraken': 'MATICUSD', 'binance': 'MATICUSDT', 'kraken_start': '2021-02-24', 'binance_start': '2019-04-26'},
    {'symbol': 'SEI', 'kraken': 'SEIUSD', 'binance': 'SEIUSDT', 'kraken_start': '2023-08-15', 'binance_start': '2023-08-15'},
    {'symbol': 'WIF', 'kraken': 'WIFUSD', 'binance': 'WIFUSDT', 'kraken_start': '2024-03-05', 'binance_start': '2024-03-05'},
    {'symbol': 'RUNE', 'kraken': 'RUNEUSD', 'binance': 'RUNEUSDT', 'kraken_start': '2021-04-13', 'binance_start': '2019-07-23'},
    {'symbol': 'BONK', 'kraken': 'BONKUSD', 'binance': 'BONKUSDT', 'kraken_start': '2023-12-14', 'binance_start': '2023-12-14'},
    {'symbol': 'FLOKI', 'kraken': 'FLOKIUSD', 'binance': 'FLOKIUSDT', 'kraken_start': '2023-01-01', 'binance_start': '2021-11-01'},
    {'symbol': 'TIA', 'kraken': 'TIAUSD', 'binance': 'TIAUSDT', 'kraken_start': '2023-10-31', 'binance_start': '2023-10-31'},
    {'symbol': 'JUP', 'kraken': 'JUPUSD', 'binance': 'JUPUSDT', 'kraken_start': '2024-01-31', 'binance_start': '2024-01-31'},
    # Top 51-60
    {'symbol': 'LDO', 'kraken': 'LDOUSD', 'binance': 'LDOUSDT', 'kraken_start': '2022-03-10', 'binance_start': '2022-03-10'},
    {'symbol': 'SAND', 'kraken': 'SANDUSD', 'binance': 'SANDUSDT', 'kraken_start': '2021-04-29', 'binance_start': '2020-08-14'},
    {'symbol': 'MANA', 'kraken': 'MANAUSD', 'binance': 'MANAUSDT', 'kraken_start': '2017-10-28', 'binance_start': '2020-08-06'},
    {'symbol': 'GALA', 'kraken': 'GALAUSD', 'binance': 'GALAUSDT', 'kraken_start': '2021-09-15', 'binance_start': '2021-09-13'},
    {'symbol': 'FLOW', 'kraken': 'FLOWUSD', 'binance': 'FLOWUSDT', 'kraken_start': '2021-01-27', 'binance_start': '2021-01-27'},
    {'symbol': 'AXS', 'kraken': 'AXSUSD', 'binance': 'AXSUSDT', 'kraken_start': '2021-08-09', 'binance_start': '2020-11-05'},
    {'symbol': 'EGLD', 'kraken': 'EGLDUSD', 'binance': 'EGLDUSDT', 'kraken_start': '2020-09-03', 'binance_start': '2020-09-03'},
    {'symbol': 'EOS', 'kraken': 'EOSUSD', 'binance': 'EOSUSDT', 'kraken_start': '2017-07-01', 'binance_start': '2018-05-03'},
    {'symbol': 'SNX', 'kraken': 'SNXUSD', 'binance': 'SNXUSDT', 'kraken_start': '2020-09-01', 'binance_start': '2020-02-11'},
    {'symbol': 'XTZ', 'kraken': 'XTZUSD', 'binance': 'XTZUSDT', 'kraken_start': '2018-10-17', 'binance_start': '2019-06-04'},
    # Top 61-70
    {'symbol': 'PYTH', 'kraken': 'PYTHUSD', 'binance': 'PYTHUSDT', 'kraken_start': '2023-11-20', 'binance_start': '2023-11-20'},
    {'symbol': 'QNT', 'kraken': 'QNTUSD', 'binance': 'QNTUSDT', 'kraken_start': '2022-06-13', 'binance_start': '2019-08-21'},
    {'symbol': 'IOTA', 'kraken': 'IOTAUSD', 'binance': 'IOTAUSDT', 'kraken_start': '2017-12-19', 'binance_start': '2017-06-13'},
    {'symbol': 'KCS', 'kraken': None, 'binance': 'KCSUSDT', 'kraken_start': None, 'binance_start': '2021-03-10'},
    {'symbol': 'NEO', 'kraken': 'NEOUSD', 'binance': 'NEOUSDT', 'kraken_start': '2019-06-06', 'binance_start': '2017-07-14'},
    {'symbol': 'KAVA', 'kraken': 'KAVAUSD', 'binance': 'KAVAUSDT', 'kraken_start': '2020-10-08', 'binance_start': '2019-10-25'},
    {'symbol': 'ONDO', 'kraken': 'ONDOUSD', 'binance': 'ONDOUSDT', 'kraken_start': '2024-01-18', 'binance_start': '2024-01-18'},
    {'symbol': 'STRK', 'kraken': 'STRKUSD', 'binance': 'STRKUSDT', 'kraken_start': '2024-02-20', 'binance_start': '2024-02-20'},
    {'symbol': 'ZEC', 'kraken': 'XZECZUSD', 'binance': 'ZECUSDT', 'kraken_start': '2016-10-29', 'binance_start': '2018-05-16'},
    {'symbol': 'DASH', 'kraken': 'DASHUSD', 'binance': 'DASHUSDT', 'kraken_start': '2014-04-01', 'binance_start': '2018-03-21'},
    # Top 71-80
    {'symbol': 'ENS', 'kraken': 'ENSUSD', 'binance': 'ENSUSDT', 'kraken_start': '2021-11-09', 'binance_start': '2021-11-09'},
    {'symbol': 'COMP', 'kraken': 'COMPUSD', 'binance': 'COMPUSDT', 'kraken_start': '2020-06-16', 'binance_start': '2020-06-25'},
    {'symbol': 'CRV', 'kraken': 'CRVUSD', 'binance': 'CRVUSDT', 'kraken_start': '2020-08-14', 'binance_start': '2020-08-25'},
    {'symbol': '1INCH', 'kraken': '1INCHUSD', 'binance': '1INCHUSDT', 'kraken_start': '2020-12-25', 'binance_start': '2020-12-25'},
    {'symbol': 'CAKE', 'kraken': 'CAKEUSD', 'binance': 'CAKEUSDT', 'kraken_start': '2021-04-21', 'binance_start': '2020-09-29'},
    {'symbol': 'ZIL', 'kraken': 'ZILUSD', 'binance': 'ZILUSDT', 'kraken_start': '2020-08-24', 'binance_start': '2018-03-19'},
    {'symbol': 'CHZ', 'kraken': 'CHZUSD', 'binance': 'CHZUSDT', 'kraken_start': '2021-02-04', 'binance_start': '2019-07-01'},
    {'symbol': 'WAVES', 'kraken': 'WAVESUSD', 'binance': 'WAVESUSDT', 'kraken_start': '2017-08-01', 'binance_start': '2018-07-16'},
    {'symbol': 'BAT', 'kraken': 'BATUSD', 'binance': 'BATUSDT', 'kraken_start': '2017-09-22', 'binance_start': '2019-04-16'},
    {'symbol': 'SUSHI', 'kraken': 'SUSHIUSD', 'binance': 'SUSHIUSDT', 'kraken_start': '2020-09-01', 'binance_start': '2020-09-01'},
    # Top 81-90
    {'symbol': 'CELO', 'kraken': 'CELOUSD', 'binance': 'CELOUSDT', 'kraken_start': '2021-04-08', 'binance_start': '2020-08-31'},
    {'symbol': 'YFI', 'kraken': 'YFIUSD', 'binance': 'YFIUSDT', 'kraken_start': '2020-08-31', 'binance_start': '2020-07-26'},
    {'symbol': 'ROSE', 'kraken': 'ROSEUSD', 'binance': 'ROSEUSDT', 'kraken_start': '2020-11-19', 'binance_start': '2020-11-19'},
    {'symbol': 'KSM', 'kraken': 'KSMUSD', 'binance': 'KSMUSDT', 'kraken_start': '2020-08-18', 'binance_start': '2020-08-31'},
    {'symbol': 'ZRX', 'kraken': 'ZRXUSD', 'binance': 'ZRXUSDT', 'kraken_start': '2017-08-16', 'binance_start': '2018-08-21'},
    {'symbol': 'ICX', 'kraken': 'ICXUSD', 'binance': 'ICXUSDT', 'kraken_start': '2018-04-17', 'binance_start': '2017-10-27'},
    {'symbol': 'MINA', 'kraken': 'MINAUSD', 'binance': 'MINAUSDT', 'kraken_start': '2021-06-01', 'binance_start': '2021-06-01'},
    {'symbol': 'ONE', 'kraken': 'ONEUSD', 'binance': 'ONEUSDT', 'kraken_start': '2020-12-03', 'binance_start': '2019-05-31'},
    {'symbol': 'ENJ', 'kraken': 'ENJUSD', 'binance': 'ENJUSDT', 'kraken_start': '2021-01-21', 'binance_start': '2019-04-18'},
    {'symbol': 'ANKR', 'kraken': 'ANKRUSD', 'binance': 'ANKRUSDT', 'kraken_start': '2021-03-18', 'binance_start': '2019-03-07'},
    # Top 91-100
    {'symbol': 'GLM', 'kraken': 'GLMUSD', 'binance': 'GLMUSDT', 'kraken_start': '2016-11-18', 'binance_start': '2020-11-19'},
    {'symbol': 'LRC', 'kraken': 'LRCUSD', 'binance': 'LRCUSDT', 'kraken_start': '2020-08-31', 'binance_start': '2019-02-27'},
    {'symbol': 'STORJ', 'kraken': 'STORJUSD', 'binance': 'STORJUSDT', 'kraken_start': '2017-08-31', 'binance_start': '2019-05-24'},
    {'symbol': 'OMG', 'kraken': 'OMGUSD', 'binance': 'OMGUSDT', 'kraken_start': '2017-09-07', 'binance_start': '2018-05-04'},
    {'symbol': 'SKL', 'kraken': 'SKLUSD', 'binance': 'SKLUSDT', 'kraken_start': '2020-12-01', 'binance_start': '2020-12-01'},
    {'symbol': 'OCEAN', 'kraken': 'OCEANUSD', 'binance': 'OCEANUSDT', 'kraken_start': '2020-12-02', 'binance_start': '2020-04-28'},
    {'symbol': 'SXP', 'kraken': 'SXPUSD', 'binance': 'SXPUSDT', 'kraken_start': '2020-11-19', 'binance_start': '2020-07-28'},
    {'symbol': 'REN', 'kraken': 'RENUSD', 'binance': 'RENUSDT', 'kraken_start': '2020-10-08', 'binance_start': '2020-02-20'},
    {'symbol': 'MASK', 'kraken': 'MASKUSD', 'binance': 'MASKUSDT', 'kraken_start': '2021-03-31', 'binance_start': '2021-02-24'},
    {'symbol': 'DYDX', 'kraken': 'DYDXUSD', 'binance': 'DYDXUSDT', 'kraken_start': '2021-09-08', 'binance_start': '2021-09-08'},
]

# ====================================================================
# CONFIGURACIÓN DE EXCHANGES
# ====================================================================

EXCHANGES = {
    'kraken': {
        'url': 'https://api.kraken.com/0/public/OHLC',
        'interval': 60,           # Datos horarios (60 min)
        'rate_limit': 1.0,        # Segundos entre requests
        'max_retries': 3,
        'timeout': 30,
        'data_dir': RAW_DATA_DIR / 'kraken'
    },
    'binance': {
        'url': 'https://api.binance.com/api/v3/klines',
        'interval': '1h',         # Datos horarios (string '1h')
        'rate_limit': 0.5,        # Segundos entre requests
        'max_retries': 3,
        'timeout': 30,
        'data_dir': RAW_DATA_DIR / 'binance'
    }
}

# ====================================================================
# CONFIGURACIÓN DE CONSOLIDACIÓN
# ====================================================================

CONSOLIDATION = {
    'primary': 'binance',         # Prioritario desde cutoff_date
    'secondary': 'kraken',        # Pre-2017 con ajuste de volumen
    'cutoff_date': '2017-07-14',  # Binance empieza a ser prioritario
    'min_overlap_candles': 720,   # Mínimo 30 días de overlap (720 velas @ 1h)
    'min_correlation': 0.95,      # Correlación mínima para validar ajuste
    'output_dir': OPERATIVE_DATA_DIR / 'cryptocurrencies'
}

# ====================================================================
# FUNCIONES HELPER
# ====================================================================

def get_all_symbols():
    """Retorna lista de todos los símbolos normalizados."""
    return [t['symbol'] for t in CRYPTO_TICKERS]


def get_ticker(symbol):
    """Obtiene configuración de un ticker por símbolo."""
    return next((t for t in CRYPTO_TICKERS if t['symbol'] == symbol), None)


def get_exchange_symbol(symbol, exchange):
    """Obtiene el símbolo específico de un exchange."""
    ticker = get_ticker(symbol)
    if ticker:
        return ticker.get(exchange)
    return None


def is_available(symbol, exchange):
    """Verifica si un ticker está disponible en un exchange."""
    return get_exchange_symbol(symbol, exchange) is not None


def get_start_date(symbol, exchange):
    """Obtiene la fecha de inicio de datos para un ticker en un exchange."""
    ticker = get_ticker(symbol)
    if ticker:
        return ticker.get(f'{exchange}_start')
    return None


def get_raw_path(symbol, exchange):
    """Retorna el path para datos raw de un ticker."""
    return EXCHANGES[exchange]['data_dir'] / symbol


def get_operative_path(symbol):
    """Retorna el path para datos operativos de un ticker."""
    return CONSOLIDATION['output_dir'] / symbol


def ensure_exchange_dirs():
    """Crea directorios de exchanges si no existen."""
    for config in EXCHANGES.values():
        config['data_dir'].mkdir(parents=True, exist_ok=True)
    CONSOLIDATION['output_dir'].mkdir(parents=True, exist_ok=True)


# Crear directorios al importar
ensure_exchange_dirs()


# ====================================================================
# GRUPOS DE TICKERS POR IMPORTANCIA
# ====================================================================
# Basado en volumen y capitalización de mercado

# Top 10 cryptos (máxima liquidez y volumen)
CRYPTO_10 = [
    'BTC',   # Bitcoin
    'ETH',   # Ethereum
    'BNB',   # Binance Coin
    'SOL',   # Solana
    'XRP',   # Ripple
    'ADA',   # Cardano
    'DOGE',  # Dogecoin
    'TRX',   # Tron
    'AVAX',  # Avalanche
    'DOT'    # Polkadot
]

# Top 25 cryptos (alta liquidez)
CRYPTO_25 = CRYPTO_10 + [
    'MATIC', # Polygon
    'LINK',  # Chainlink
    'UNI',   # Uniswap
    'LTC',   # Litecoin
    'ATOM',  # Cosmos
    'XLM',   # Stellar
    'ETC',   # Ethereum Classic
    'FIL',   # Filecoin
    'HBAR',  # Hedera
    'VET',   # VeChain
    'ALGO',  # Algorand
    'AAVE',  # Aave
    'GRT',   # The Graph
    'SAND',  # Sandbox
    'MANA'   # Decentraland
]

# Todos los cryptos configurados
CRYPTO_ALL = get_all_symbols()

# Diccionario de grupos para fácil acceso
TICKER_GROUPS = {
    'crypto_10': CRYPTO_10,
    'crypto_25': CRYPTO_25,
    'crypto_all': CRYPTO_ALL
}


def get_ticker_group(group_name: str) -> list:
    """
    Obtiene lista de tickers de un grupo.

    Args:
        group_name: Nombre del grupo ('crypto_10', 'crypto_25', 'crypto_all')

    Returns:
        Lista de símbolos de tickers
    """
    return TICKER_GROUPS.get(group_name, [])


if __name__ == '__main__':
    print(f"Total tickers: {len(CRYPTO_TICKERS)}")
    print(f"Exchanges: {list(EXCHANGES.keys())}")
    print(f"\nGrupos disponibles:")
    for name, tickers in TICKER_GROUPS.items():
        print(f"  {name}: {len(tickers)} tickers")
    print(f"\nTop 5 tickers:")
    for t in CRYPTO_TICKERS[:5]:
        print(f"  {t['symbol']}: kraken={t['kraken']}, binance={t['binance']}")
