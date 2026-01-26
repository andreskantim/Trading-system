#!/usr/bin/env python3
"""
Download Crypto Data from Kraken and Binance

Descarga datos históricos 1h de las top 100 criptomonedas desde:
- Kraken: TODO su rango histórico (ej: 2013-presente para BTC)
- Binance: TODO su rango histórico (2017-presente)

IMPORTANTE: Descarga TODO incluyendo períodos comunes (overlap) que se usan
para calcular ajustes de volumen en la fase de consolidación.

Los datos se guardan en formato Parquet, organizados por exchange, ticker y año.

Usage:
    python download_crypto.py                    # Descarga todos los tickers
    python download_crypto.py --ticker BTC       # Solo un ticker
    python download_crypto.py --test             # Modo test (top 3)
    python download_crypto.py --exchange kraken  # Solo un exchange
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path para imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.paths import (
    OHLCV_COLUMNS,
    PARQUET_COMPRESSION,
    PARQUET_ENGINE,
    LOGS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    ensure_directories,
)

from config.tickers import (
    CRYPTO_TICKERS,
    EXCHANGES,
    get_all_symbols,
    get_ticker,
    get_exchange_symbol,
    is_available,
    get_start_date,
    get_raw_path,
    ensure_exchange_dirs,
)

import time
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

ensure_directories()
ensure_exchange_dirs()
log_file = LOGS_DIR / f"download_crypto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoDownloader:
    """
    Clase para descargar datos de crypto desde múltiples exchanges.

    Descarga datos históricos OHLCV de Kraken y Binance,
    guardándolos en formato Parquet organizados por año.
    """

    def __init__(self):
        """
        Inicializa el downloader usando configuración centralizada.
        Toda la configuración viene de config.tickers y config.paths.
        """
        logger.info("=" * 80)
        logger.info("Crypto Downloader Initialized")
        logger.info("=" * 80)
        logger.info(f"Total tickers configurados: {len(CRYPTO_TICKERS)}")
        logger.info(f"Exchanges: {list(EXCHANGES.keys())}")

        # Crear directorios para cada exchange
        for exchange_name, config in EXCHANGES.items():
            config['data_dir'].mkdir(parents=True, exist_ok=True)
            logger.info(f"Data dir {exchange_name}: {config['data_dir']}")

        # Estadísticas de tickers disponibles por exchange
        for exchange_name in EXCHANGES.keys():
            available = sum(1 for t in CRYPTO_TICKERS if t.get(exchange_name) is not None)
            logger.info(f"Tickers disponibles en {exchange_name}: {available}")

        self.failed_downloads = []
        self.session = requests.Session()

    def _make_request(self, url: str, params: dict, exchange_name: str) -> dict:
        """
        Hace una request HTTP con reintentos y backoff exponencial.

        Args:
            url: URL del endpoint
            params: Parámetros de la request
            exchange_name: Nombre del exchange para rate limiting

        Returns:
            dict: Respuesta JSON

        Raises:
            Exception: Si falla después de todos los reintentos
        """
        config = EXCHANGES[exchange_name]
        max_retries = config['max_retries']
        timeout = config['timeout']

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                wait_time = (2 ** attempt) * config['rate_limit']
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        return {}

    def download_kraken(self, ticker_symbol: str) -> pd.DataFrame:
        """
        Descarga TODOS los datos históricos de Kraken para un ticker.

        Descarga desde fecha de disponibilidad hasta PRESENTE (HOY),
        incluyendo overlap con Binance para ajuste de volumen.

        Args:
            ticker_symbol: Símbolo normalizado (ej: 'BTC')

        Returns:
            DataFrame con columnas: timestamp (index), open, high, low, close, volume
        """
        if not is_available(ticker_symbol, 'kraken'):
            logger.warning(f"{ticker_symbol} no disponible en Kraken, skipping...")
            return pd.DataFrame()

        kraken_symbol = get_exchange_symbol(ticker_symbol, 'kraken')
        start_date_str = get_start_date(ticker_symbol, 'kraken')

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.now()

        kraken_config = EXCHANGES['kraken']
        base_url = kraken_config['url']

        logger.info(f"Descargando {ticker_symbol} desde Kraken...")
        logger.info(f"  Simbolo Kraken: {kraken_symbol}")
        logger.info(f"  Rango: {start_date.date()} a {end_date.date()}")

        all_data = []
        since = int(start_date.timestamp())
        request_count = 0

        while True:
            params = {
                'pair': kraken_symbol,
                'interval': kraken_config['interval'],
                'since': since
            }

            try:
                data = self._make_request(base_url, params, 'kraken')
            except Exception as e:
                logger.error(f"Error en request Kraken: {e}")
                break

            # Verificar errores de API
            if data.get('error') and len(data['error']) > 0:
                logger.error(f"Kraken API error: {data['error']}")
                break

            result = data.get('result', {})
            # Kraken devuelve el resultado con la key del par
            ohlc_data = None
            for key, value in result.items():
                if key != 'last' and isinstance(value, list):
                    ohlc_data = value
                    break

            if not ohlc_data or len(ohlc_data) == 0:
                logger.info(f"  No mas datos disponibles")
                break

            # Formato Kraken: [time, open, high, low, close, vwap, volume, count]
            for row in ohlc_data:
                ts = int(row[0])
                if ts >= int(end_date.timestamp()):
                    continue
                all_data.append({
                    'timestamp': pd.Timestamp(ts, unit='s'),
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[6])
                })

            # Actualizar since para siguiente request
            last_ts = int(ohlc_data[-1][0])
            if last_ts <= since:
                break
            since = last_ts

            request_count += 1
            if request_count % 50 == 0:
                logger.info(f"  {request_count} requests, {len(all_data):,} velas...")

            time.sleep(kraken_config['rate_limit'])

        if not all_data:
            logger.warning(f"No se obtuvieron datos de Kraken para {ticker_symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.set_index('timestamp').sort_index()

        logger.info(f"  Total: {len(df):,} velas descargadas de Kraken")
        return df

    def download_binance(self, ticker_symbol: str) -> pd.DataFrame:
        """
        Descarga TODOS los datos históricos de Binance para un ticker.

        Args:
            ticker_symbol: Símbolo normalizado (ej: 'BTC')

        Returns:
            DataFrame con columnas: timestamp (index), open, high, low, close, volume
        """
        if not is_available(ticker_symbol, 'binance'):
            logger.warning(f"{ticker_symbol} no disponible en Binance, skipping...")
            return pd.DataFrame()

        binance_symbol = get_exchange_symbol(ticker_symbol, 'binance')
        start_date_str = get_start_date(ticker_symbol, 'binance')

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.now()

        binance_config = EXCHANGES['binance']
        base_url = binance_config['url']

        logger.info(f"Descargando {ticker_symbol} desde Binance...")
        logger.info(f"  Simbolo Binance: {binance_symbol}")
        logger.info(f"  Rango: {start_date.date()} a {end_date.date()}")

        all_data = []
        start_time = int(start_date.timestamp() * 1000)
        end_time = int(end_date.timestamp() * 1000)
        request_count = 0

        while start_time < end_time:
            params = {
                'symbol': binance_symbol,
                'interval': binance_config['interval'],
                'startTime': start_time,
                'limit': 1000
            }

            try:
                data = self._make_request(base_url, params, 'binance')
            except Exception as e:
                logger.error(f"Error en request Binance: {e}")
                break

            if not data or len(data) == 0:
                break

            # Formato Binance: [open_time, open, high, low, close, volume, close_time, ...]
            for row in data:
                ts = int(row[0])
                all_data.append({
                    'timestamp': pd.Timestamp(ts, unit='ms'),
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5])
                })

            # Siguiente página
            last_ts = int(data[-1][0])
            if last_ts <= start_time:
                break
            start_time = last_ts + 1

            request_count += 1
            if request_count % 50 == 0:
                logger.info(f"  {request_count} requests, {len(all_data):,} velas...")

            time.sleep(binance_config['rate_limit'])

        if not all_data:
            logger.warning(f"No se obtuvieron datos de Binance para {ticker_symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.set_index('timestamp').sort_index()

        logger.info(f"  Total: {len(df):,} velas descargadas de Binance")
        return df

    def save_by_year(self, df: pd.DataFrame, ticker_symbol: str, exchange_name: str):
        """
        Divide DataFrame por años naturales y guarda en Parquet.

        Args:
            df: DataFrame con índice timestamp
            ticker_symbol: Símbolo normalizado
            exchange_name: Nombre del exchange ('kraken' o 'binance')
        """
        if df.empty:
            logger.warning(f"DataFrame vacio para {ticker_symbol}/{exchange_name}")
            return

        output_path = get_raw_path(ticker_symbol, exchange_name)
        output_path.mkdir(parents=True, exist_ok=True)

        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year

        years = sorted(df_copy['year'].unique())

        for year in years:
            year_data = df_copy[df_copy['year'] == year].drop('year', axis=1)

            filename = f"{ticker_symbol}_{year}.parquet"
            filepath = output_path / filename

            year_data.to_parquet(
                filepath,
                compression=PARQUET_COMPRESSION,
                engine=PARQUET_ENGINE
            )

            logger.info(f"    Guardado: {filepath.name} ({len(year_data):,} velas)")

        logger.info(f"  Total guardado: {len(years)} años, {len(df):,} velas")

    def download_ticker(self, ticker_symbol: str, exchange_filter: str = None):
        """
        Descarga datos de un ticker desde TODOS los exchanges disponibles.

        Args:
            ticker_symbol: Símbolo normalizado
            exchange_filter: Si se especifica, solo descarga de ese exchange
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DESCARGANDO: {ticker_symbol}")
        logger.info(f"{'=' * 80}")

        ticker_config = get_ticker(ticker_symbol)
        if not ticker_config:
            logger.error(f"Ticker {ticker_symbol} no encontrado en configuracion")
            return

        exchanges_to_download = [exchange_filter] if exchange_filter else EXCHANGES.keys()

        for exchange_name in exchanges_to_download:
            if exchange_name not in EXCHANGES:
                logger.warning(f"Exchange {exchange_name} no configurado")
                continue

            try:
                if is_available(ticker_symbol, exchange_name):
                    if exchange_name == 'kraken':
                        df = self.download_kraken(ticker_symbol)
                    elif exchange_name == 'binance':
                        df = self.download_binance(ticker_symbol)
                    else:
                        logger.warning(f"Exchange {exchange_name} no implementado")
                        continue

                    if not df.empty:
                        self.save_by_year(df, ticker_symbol, exchange_name)
                        logger.info(f"  {exchange_name.upper()}: {len(df):,} velas descargadas")
                    else:
                        logger.warning(f"  {exchange_name.upper()}: Sin datos")
                else:
                    logger.info(f"  {exchange_name.upper()}: No disponible para {ticker_symbol}")

            except Exception as e:
                logger.error(f"Error descargando {ticker_symbol} de {exchange_name}: {e}")
                self.failed_downloads.append({
                    'ticker': ticker_symbol,
                    'exchange': exchange_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue

    def download_all(self, exchange_filter: str = None):
        """
        Descarga todos los tickers configurados.

        Args:
            exchange_filter: Si se especifica, solo descarga de ese exchange
        """
        all_tickers = get_all_symbols()
        total = len(all_tickers)

        logger.info(f"\n{'#' * 80}")
        logger.info(f"INICIO DESCARGA MASIVA: {total} tickers")
        if exchange_filter:
            logger.info(f"Exchange filter: {exchange_filter}")
        logger.info(f"{'#' * 80}\n")

        start_time = datetime.now()

        for idx, ticker_symbol in enumerate(all_tickers, 1):
            logger.info(f"\n[{idx}/{total}] Procesando {ticker_symbol}...")

            try:
                self.download_ticker(ticker_symbol, exchange_filter)
            except Exception as e:
                logger.error(f"Error fatal descargando {ticker_symbol}: {e}")
                self.failed_downloads.append({
                    'ticker': ticker_symbol,
                    'exchange': 'all',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue

        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info(f"\n{'#' * 80}")
        logger.info(f"DESCARGA COMPLETADA")
        logger.info(f"{'#' * 80}")
        logger.info(f"Tiempo total: {duration}")
        logger.info(f"Tickers procesados: {total}")
        logger.info(f"Fallos: {len(self.failed_downloads)}")

        if self.failed_downloads:
            failed_log = LOGS_DIR / f"failed_downloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_log, 'w') as f:
                json.dump(self.failed_downloads, f, indent=2)
            logger.warning(f"Log de fallos guardado en: {failed_log}")

        logger.info(f"Log completo en: {log_file}")


def main():
    """Punto de entrada principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Descarga datos historicos de crypto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--ticker',
        type=str,
        help='Descargar solo un ticker especifico (ej: BTC)'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        choices=['kraken', 'binance'],
        help='Descargar solo de un exchange especifico'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Modo test: solo descarga top 3 tickers'
    )

    args = parser.parse_args()

    downloader = CryptoDownloader()

    if args.test:
        logger.info("MODO TEST: Descargando solo top 3 tickers")
        for ticker in get_all_symbols()[:3]:
            downloader.download_ticker(ticker, args.exchange)
    elif args.ticker:
        logger.info(f"Descargando ticker especifico: {args.ticker}")
        downloader.download_ticker(args.ticker.upper(), args.exchange)
    else:
        downloader.download_all(args.exchange)


if __name__ == '__main__':
    main()
