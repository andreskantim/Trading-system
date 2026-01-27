#!/usr/bin/env python3
"""
Ejemplos de uso del Screening System
Muestra cómo usar programáticamente cada componente
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from screening.download.binance_downloader import BinanceDownloader
from screening.generate_signal.signal_detector import SignalDetector


def ejemplo_1_descarga_basica():
    """Ejemplo 1: Descarga básica de un ticker"""
    print("="*70)
    print("EJEMPLO 1: Descarga básica de datos")
    print("="*70)

    downloader = BinanceDownloader()

    # Descargar un solo ticker
    result = downloader.download_ticker('BTCUSDT', verbose=True)

    if result['success']:
        print(f"\nDescarga exitosa!")
        print(f"  - Insertados: {result['inserted']} registros")
        print(f"  - Total en BD: {result['total_records']} registros")
        print(f"  - Ubicación: {result['db_path']}")


def ejemplo_2_leer_datos_descargados():
    """Ejemplo 2: Leer datos descargados desde SQLite"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Leer datos descargados")
    print("="*70)

    db_path = project_root / "data" / "operative" / "BTCUSDT" / "data.db"

    if not db_path.exists():
        print("ERROR: Primero ejecuta ejemplo_1_descarga_basica()")
        return

    conn = sqlite3.connect(str(db_path))

    # Leer como DataFrame
    df = pd.read_sql_query(
        "SELECT * FROM ohlcv ORDER BY timestamp DESC LIMIT 10",
        conn
    )

    # Convertir timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    print("\nÚltimas 10 velas:")
    print(df.to_string(index=False))

    # Estadísticas
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM ohlcv")
    min_ts, max_ts, count = cursor.fetchone()

    print(f"\nEstadísticas:")
    print(f"  - Total registros: {count}")
    print(f"  - Desde: {pd.to_datetime(min_ts, unit='ms', utc=True)}")
    print(f"  - Hasta: {pd.to_datetime(max_ts, unit='ms', utc=True)}")

    conn.close()


def ejemplo_3_detectar_senales():
    """Ejemplo 3: Detectar señales en datos descargados"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Detección de señales")
    print("="*70)

    detector = SignalDetector()

    # Detectar señales para un ticker con estrategia específica
    result = detector.process_ticker('BTCUSDT', 'hawkes', verbose=True)

    print(f"\nResultado:")
    print(f"  - Ticker: {result['ticker']}")
    print(f"  - Estrategia: {result['strategy']}")
    print(f"  - Éxito: {result['success']}")
    if result['success']:
        print(f"  - Señales detectadas: {result['detected']}")
        print(f"  - Señales nuevas: {result['new_signals']}")


def ejemplo_4_leer_senales_guardadas():
    """Ejemplo 4: Leer y analizar señales guardadas"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Analizar señales guardadas")
    print("="*70)

    csv_path = project_root / "screening" / "signals" / "hawkes" / "signals.csv"

    if not csv_path.exists():
        print("ERROR: No hay señales guardadas. Ejecuta ejemplo_3_detectar_senales() primero")
        return

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    print(f"\nTotal señales guardadas: {len(df)}")

    # Agrupar por tipo de señal
    print("\nPor tipo de señal:")
    print(df['signal_type'].value_counts())

    # Agrupar por ticker
    print("\nPor ticker:")
    print(df['ticker'].value_counts())

    # Últimas 5 señales
    print("\nÚltimas 5 señales:")
    print(df.tail(5).to_string(index=False))


def ejemplo_5_pipeline_completo():
    """Ejemplo 5: Pipeline completo con múltiples tickers"""
    print("\n" + "="*70)
    print("EJEMPLO 5: Pipeline completo")
    print("="*70)

    tickers = ['BTC', 'ETH', 'SOL']

    # 1. Descargar datos
    print("\n[PASO 1] Descargando datos...")
    downloader = BinanceDownloader()
    download_results = downloader.download_all_tickers(tickers, verbose=False)

    successful = sum(1 for r in download_results if r['success'])
    print(f"Descarga: {successful}/{len(download_results)} exitosos")

    # 2. Detectar señales
    print("\n[PASO 2] Detectando señales...")
    detector = SignalDetector()
    signal_results = detector.process_all_tickers(
        tickers_list=tickers,
        strategies=['hawkes', 'bollinger_b2b'],
        verbose=False
    )

    # 3. Resumen
    print("\n[RESUMEN]")
    for strategy, results in signal_results.items():
        successful_results = [r for r in results if r['success']]
        total_new = sum(r.get('new_signals', 0) for r in successful_results)
        print(f"  {strategy}: {total_new} nuevas señales")


def ejemplo_6_filtrar_senales_recientes():
    """Ejemplo 6: Filtrar señales de las últimas 24 horas"""
    print("\n" + "="*70)
    print("EJEMPLO 6: Señales de últimas 24 horas")
    print("="*70)

    # Leer señales de Bollinger
    csv_path = project_root / "screening" / "signals" / "bollinger_b2b" / "signals.csv"

    if not csv_path.exists():
        print("ERROR: No hay señales de Bollinger guardadas")
        return

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    # Filtrar últimas 24h
    cutoff = pd.Timestamp.now('UTC') - pd.Timedelta(hours=24)
    recent = df[df['timestamp'] > cutoff]

    print(f"\nSeñales en últimas 24h: {len(recent)}/{len(df)}")

    if len(recent) > 0:
        print("\nSeñales recientes:")
        print(recent[['timestamp', 'ticker', 'signal_type', 'price']].to_string(index=False))


def main():
    """Ejecuta todos los ejemplos"""
    print("SCREENING SYSTEM - EJEMPLOS DE USO")
    print("="*70)

    # Descomenta los ejemplos que quieras ejecutar:

    ejemplo_1_descarga_basica()
    ejemplo_2_leer_datos_descargados()
    ejemplo_3_detectar_senales()
    ejemplo_4_leer_senales_guardadas()
    ejemplo_5_pipeline_completo()
    ejemplo_6_filtrar_senales_recientes()

    print("\n" + "="*70)
    print("EJEMPLOS COMPLETADOS")
    print("="*70)


if __name__ == '__main__':
    main()
