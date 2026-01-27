#!/usr/bin/env python3
"""
Script de actualización periódica de datos de Binance
Diseñado para ejecutarse via cron/scheduler
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from screening.download.binance_downloader import BinanceDownloader
from config.tickers import CRYPTO_ALL


def setup_logging(log_dir: Path = None):
    """Configura logging para el proceso de actualización"""
    if log_dir is None:
        log_dir = project_root / "logs" / "screening"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Nombre de log con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"update_{timestamp}.log"

    # Configurar formato
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
    """
    Ejecuta actualización completa de todos los tickers
    """
    logger = setup_logging()

    logger.info("="*70)
    logger.info("INICIO DE ACTUALIZACIÓN DE DATOS - BINANCE")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Total tickers a procesar: {len(CRYPTO_ALL)}")

    # Inicializar downloader
    downloader = BinanceDownloader()

    try:
        # Descargar todos los tickers
        results = downloader.download_all_tickers(
            tickers_list=CRYPTO_ALL,
            verbose=False  # Logs más compactos para producción
        )

        # Analizar resultados
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        # Calcular estadísticas agregadas
        total_inserted = sum(r.get('inserted', 0) for r in successful)
        total_deleted = sum(r.get('deleted', 0) for r in successful)

        logger.info("")
        logger.info("="*70)
        logger.info("RESUMEN DE ACTUALIZACIÓN")
        logger.info("="*70)
        logger.info(f"Total procesados: {len(results)}")
        logger.info(f"Exitosos: {len(successful)}")
        logger.info(f"Fallidos: {len(failed)}")
        logger.info(f"Total registros insertados: {total_inserted}")
        logger.info(f"Total registros eliminados (antiguos): {total_deleted}")

        if failed:
            logger.warning("")
            logger.warning("Tickers con errores:")
            for r in failed:
                logger.warning(f"  - {r['ticker']}: {r.get('error', 'Unknown')}")

        # Detalles de tickers exitosos
        logger.info("")
        logger.info("Tickers actualizados exitosamente:")
        for r in successful:
            logger.info(
                f"  {r['ticker']}: "
                f"+{r['inserted']} nuevos, "
                f"-{r['deleted']} antiguos, "
                f"total={r['total_records']}"
            )

        logger.info("")
        logger.info("="*70)
        logger.info("ACTUALIZACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("="*70)

        return 0  # Exit code success

    except Exception as e:
        logger.error("")
        logger.error("="*70)
        logger.error("ERROR CRÍTICO EN ACTUALIZACIÓN")
        logger.error("="*70)
        logger.error(f"Error: {e}", exc_info=True)
        return 1  # Exit code error


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
