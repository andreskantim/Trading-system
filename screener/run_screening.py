#!/usr/bin/env python3
"""
Pipeline completo de Screening System
Ejecuta descarga + detección + broadcast en un solo comando
Por defecto usa CRYPTO_25 para todos los procesos
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from screener.actualize_data.binance_downloader import BinanceDownloader
from screener.generate_signal.signal_detector import SignalDetector
from screener.broadcast.email_sender import EmailBroadcaster
from config.tickers import CRYPTO_ALL, CRYPTO_10, CRYPTO_25


def setup_logging():
    log_dir = project_root / "logs" / "screening"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"screening_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Screening System - Download + Signals + Broadcast')

    parser.add_argument(
        '--tickers',
        choices=['crypto_10', 'crypto_25', 'crypto_all'],
        default='crypto_25',  # CRYPTO_25 por defecto
        help='Grupo de tickers (default: crypto_25)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Saltar descarga de datos'
    )
    parser.add_argument(
        '--skip-signals',
        action='store_true',
        help='Saltar detección de señales'
    )
    parser.add_argument(
        '--skip-broadcast',
        action='store_true',
        help='Saltar envío de señales por email'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['hawkes', 'bollinger_b2b'],
        default=['hawkes', 'bollinger_b2b'],
        help='Estrategias a aplicar (default: todas)'
    )
    parser.add_argument(
        '--email',
        type=str,
        default='dummy@example.com',
        help='Email destinatario para broadcast'
    )

    args = parser.parse_args()
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("SCREENING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Grupo de tickers: {args.tickers}")
    logger.info(f"Estrategias: {', '.join(args.strategies)}")

    # Seleccionar tickers (mismo batch para todo el pipeline)
    if args.tickers == 'crypto_10':
        tickers_list = CRYPTO_10
    elif args.tickers == 'crypto_25':
        tickers_list = CRYPTO_25
    else:
        tickers_list = CRYPTO_ALL

    logger.info(f"Total tickers: {len(tickers_list)}")

    signal_results = {}

    # FASE 1: Descarga
    if not args.skip_download:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FASE 1: DESCARGA DE DATOS")
        logger.info("=" * 80)

        downloader = BinanceDownloader()
        download_results = downloader.download_all_tickers(tickers_list, verbose=True)

        successful_downloads = sum(1 for r in download_results if r['success'])
        logger.info(f"\nDescarga: {successful_downloads}/{len(download_results)} exitosos")
    else:
        logger.info("\nDESCARGA OMITIDA (--skip-download)")

    # FASE 2: Señales
    if not args.skip_signals:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FASE 2: DETECCIÓN DE SEÑALES")
        logger.info("=" * 80)

        detector = SignalDetector()
        signal_results = detector.process_all_tickers(
            tickers_list=tickers_list,
            strategies=args.strategies,
            verbose=True
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("RESUMEN SEÑALES")
        logger.info("=" * 80)

        total_new = 0
        for strategy_name, results in signal_results.items():
            successful = [r for r in results if r['success']]
            new_count = sum(r.get('new_signals', 0) for r in successful)
            total_new += new_count
            logger.info(f"[{strategy_name}] Nuevas señales: {new_count}")

    else:
        logger.info("\nDETECCIÓN OMITIDA (--skip-signals)")

    # FASE 3: Broadcast
    if not args.skip_broadcast and signal_results:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FASE 3: BROADCAST")
        logger.info("=" * 80)

        broadcaster = EmailBroadcaster(recipient_email=args.email)
        broadcast_result = broadcaster.broadcast_new_signals(signal_results)

        if broadcast_result.get('sent'):
            logger.info(f"Email enviado: {broadcast_result.get('count')} señales")
        else:
            logger.info(f"No enviado: {broadcast_result.get('reason', 'sin señales nuevas')}")

    elif args.skip_broadcast:
        logger.info("\nBROADCAST OMITIDO (--skip-broadcast)")

    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETADO")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
