"""
Binance Downloader - Descarga datos horarios del último mes
Mantiene solo el último mes de datos en SQLite (ventana deslizante)
"""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config.tickers import CRYPTO_TICKERS, get_exchange_symbol, get_operative_path


class BinanceDownloader:
    """Descarga y almacena datos horarios de Binance (último mes)"""

    BASE_URL = "https://api.binance.com/api/v3/klines"
    RATE_LIMIT = 0.5  # Segundos entre requests
    MAX_RETRIES = 3
    TIMEOUT = 30

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: Directorio base para almacenar datos (default: ../data/operative/)
        """
        if data_dir is None:
            self.data_dir = project_root / "data" / "operative"
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_ticker_db_path(self, symbol: str) -> Path:
        """Retorna path a screening.db: data/operative/{label}/{ticker}/screening.db"""
        ticker_dir = get_operative_path(symbol)
        return ticker_dir / "screening.db"

    def create_table(self, conn):
        """Crea tabla OHLCV si no existe"""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp INTEGER PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            )
        """)
        conn.commit()

    def delete_old_data(self, conn, cutoff_timestamp: int):
        """
        Elimina datos más antiguos que cutoff_timestamp

        Args:
            conn: Conexión SQLite
            cutoff_timestamp: Timestamp en milisegundos
        """
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ohlcv WHERE timestamp < ?", (cutoff_timestamp,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted

    def fetch_binance_data(self, symbol: str, start_time: int, end_time: int) -> list:
        """
        Descarga datos de Binance API

        Args:
            symbol: Símbolo en formato Binance (ej: 'BTCUSDT')
            start_time: Timestamp inicio en milisegundos
            end_time: Timestamp fin en milisegundos

        Returns:
            Lista de velas OHLCV
        """
        params = {
            'symbol': symbol,
            'interval': '1h',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000  # Máximo por request
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.TIMEOUT
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Error en request (intento {attempt+1}/{self.MAX_RETRIES}): {e}")
                    print(f"  Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Falló descarga después de {self.MAX_RETRIES} intentos: {e}")

        return []

    def download_ticker(self, symbol: str, ticker_binance: str, verbose: bool = True) -> dict:
        """
        Descarga datos del último mes para un ticker

        Args:
            symbol: Símbolo genérico (ej: 'BTC')
            ticker_binance: Nombre en Binance (ej: 'BTCUSDT')
            verbose: Si imprimir logs de progreso

        Returns:
            Dict con estadísticas de descarga
        """
        if verbose:
            print(f"\n[{symbol}] Iniciando descarga...")

        # Calcular timestamps (último mes)
        now = datetime.utcnow()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(days=30)).timestamp() * 1000)

        # Conectar a BD
        db_path = self.get_ticker_db_path(symbol)
        conn = sqlite3.connect(str(db_path))

        try:
            # Crear tabla
            self.create_table(conn)

            # Eliminar datos antiguos (> 30 días)
            deleted = self.delete_old_data(conn, start_time)
            if verbose and deleted > 0:
                print(f"  Eliminados {deleted} registros antiguos")

            # Descargar datos en chunks de 1000 velas
            all_data = []
            current_start = start_time

            while current_start < end_time:
                # Respetar rate limit
                time.sleep(self.RATE_LIMIT)

                # Fetch data
                data = self.fetch_binance_data(ticker_binance, current_start, end_time)

                if not data:
                    break

                all_data.extend(data)

                # Actualizar timestamp para siguiente chunk
                last_timestamp = data[-1][0]
                current_start = last_timestamp + 1

                # Si recibimos menos de 1000, ya terminamos
                if len(data) < 1000:
                    break

            # Insertar datos (INSERT OR REPLACE para evitar duplicados)
            cursor = conn.cursor()
            inserted = 0

            for candle in all_data:
                timestamp = candle[0]
                open_price = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                volume = float(candle[5])

                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv (timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (timestamp, open_price, high, low, close, volume))
                inserted += 1

            conn.commit()

            # Estadísticas
            cursor.execute("SELECT COUNT(*) FROM ohlcv")
            total_records = cursor.fetchone()[0]

            if verbose:
                print(f"  Insertados: {inserted} registros")
                print(f"  Total en BD: {total_records} registros")
                print(f"  BD ubicada en: {db_path}")

            return {
                'symbol': symbol,
                'success': True,
                'inserted': inserted,
                'deleted': deleted,
                'total_records': total_records,
                'db_path': str(db_path)
            }

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }

        finally:
            conn.close()

    def download_all_tickers(self, tickers_list: list = None, verbose: bool = True) -> list:
        """
        Descarga datos para todos los tickers configurados

        Args:
            tickers_list: Lista de símbolos genéricos (si None, usa todos)
            verbose: Si imprimir logs

        Returns:
            Lista de resultados por ticker
        """
        if tickers_list is None:
            # Usar todos los tickers que tengan Binance
            tickers_list = [
                t['symbol'] for t in CRYPTO_TICKERS
                if t.get('binance') is not None
            ]

        results = []
        total = len(tickers_list)

        print(f"\n{'='*60}")
        print(f"Descargando {total} tickers desde Binance")
        print(f"{'='*60}")

        for idx, symbol in enumerate(tickers_list, 1):
            # Obtener nombre de Binance
            ticker_binance = get_exchange_symbol(symbol, 'binance')

            if ticker_binance is None:
                print(f"\n[{idx}/{total}] {symbol}: No disponible en Binance (SKIP)")
                continue

            print(f"\n[{idx}/{total}] Procesando {symbol} ({ticker_binance})")

            result = self.download_ticker(symbol, ticker_binance, verbose=verbose)
            results.append(result)

        # Resumen final
        print(f"\n{'='*60}")
        print(f"RESUMEN DE DESCARGA")
        print(f"{'='*60}")

        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"Total: {len(results)}")
        print(f"Exitosos: {successful}")
        print(f"Fallidos: {failed}")

        if failed > 0:
            print(f"\nTickers fallidos:")
            for r in results:
                if not r['success']:
                    print(f"  - {r['symbol']}: {r.get('error', 'Unknown error')}")

        return results


def main():
    """Función principal para testing"""
    downloader = BinanceDownloader()

    # Descargar solo top 3 para testing
    test_tickers = ['BTC', 'ETH', 'BNB']

    print("TESTING: Descargando top 3 tickers...")
    results = downloader.download_all_tickers(test_tickers)

    print(f"\n\nResultados guardados en: {downloader.data_dir}")


if __name__ == '__main__':
    main()
