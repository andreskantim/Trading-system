#!/usr/bin/env python3
"""
Scheduler automático para screening
Ejecuta descarga + señales + broadcast cada hora
Recupera datos faltantes al iniciar
"""

import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config.tickers import CRYPTO_25
from screening.download.binance_downloader import BinanceDownloader
from screening.generate_signal.signal_detector import SignalDetector
from screening.broadcast.email_sender import EmailBroadcaster

# Configuración
DEFAULT_TICKERS = CRYPTO_25
INTERVAL_SECONDS = 3600  # 1 hora
LOG_DIR = project_root / "logs" / "screening"


class ScreeningScheduler:
    """Scheduler que ejecuta el pipeline cada hora"""

    def __init__(
        self,
        tickers: list = None,
        strategies: list = None,
        recipient_email: str = "dummy@example.com"
    ):
        self.tickers = tickers or DEFAULT_TICKERS
        self.strategies = strategies or ['hawkes', 'bollinger_b2b']
        self.running = True

        self.downloader = BinanceDownloader()
        self.detector = SignalDetector()
        self.broadcaster = EmailBroadcaster(recipient_email=recipient_email)

        self._setup_logging()
        self._setup_signal_handlers()

    def _setup_logging(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / "scheduler.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.logger.info("Señal de parada recibida. Finalizando...")
        self.running = False

    def run_pipeline(self) -> dict:
        """Ejecuta pipeline completo: download -> signals -> broadcast"""
        timestamp = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"PIPELINE INICIADO: {timestamp.isoformat()}")
        self.logger.info(f"Tickers: {len(self.tickers)}, Estrategias: {self.strategies}")

        results = {'download': None, 'signals': None, 'broadcast': None}

        # 1. Descarga (recupera datos faltantes automáticamente)
        try:
            self.logger.info("[1/3] Descargando datos...")
            download_results = self.downloader.download_all_tickers(self.tickers, verbose=False)
            successful = sum(1 for r in download_results if r['success'])
            results['download'] = {'success': successful, 'total': len(download_results)}
            self.logger.info(f"  Descarga: {successful}/{len(download_results)} OK")
        except Exception as e:
            self.logger.error(f"  Error en descarga: {e}")
            results['download'] = {'error': str(e)}

        # 2. Detección de señales
        try:
            self.logger.info("[2/3] Detectando señales...")
            signal_results = self.detector.process_all_tickers(
                tickers_list=self.tickers,
                strategies=self.strategies,
                verbose=False
            )
            total_new = sum(
                sum(r.get('new_signals', 0) for r in res if r.get('success'))
                for res in signal_results.values()
            )
            results['signals'] = {'new_signals': total_new, 'by_strategy': {
                s: sum(r.get('new_signals', 0) for r in res if r.get('success'))
                for s, res in signal_results.items()
            }}
            self.logger.info(f"  Señales nuevas: {total_new}")
        except Exception as e:
            self.logger.error(f"  Error en señales: {e}")
            results['signals'] = {'error': str(e)}
            signal_results = {}

        # 3. Broadcast
        try:
            self.logger.info("[3/3] Broadcasting...")
            if signal_results and total_new > 0:
                broadcast_result = self.broadcaster.broadcast_new_signals(signal_results)
                results['broadcast'] = broadcast_result
                if broadcast_result.get('sent'):
                    self.logger.info(f"  Email enviado: {broadcast_result.get('count')} señales")
                else:
                    self.logger.info(f"  No enviado: {broadcast_result.get('reason', 'unknown')}")
            else:
                results['broadcast'] = {'sent': False, 'reason': 'no_new_signals'}
                self.logger.info("  Sin señales nuevas para enviar")
        except Exception as e:
            self.logger.error(f"  Error en broadcast: {e}")
            results['broadcast'] = {'error': str(e)}

        self.logger.info(f"PIPELINE COMPLETADO en {(datetime.now() - timestamp).seconds}s")
        return results

    def _seconds_until_next_hour(self) -> int:
        """Calcula segundos hasta la próxima hora en punto"""
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
        return int((next_hour - now).total_seconds())

    def start(self, run_immediately: bool = True):
        """
        Inicia el scheduler

        Args:
            run_immediately: Si True, ejecuta pipeline al iniciar (recupera datos)
        """
        self.logger.info("="*60)
        self.logger.info("SCHEDULER INICIADO")
        self.logger.info(f"Tickers: {self.tickers}")
        self.logger.info(f"Intervalo: {INTERVAL_SECONDS}s ({INTERVAL_SECONDS//60} min)")
        self.logger.info("="*60)

        if run_immediately:
            self.logger.info("Ejecutando pipeline inicial (recuperación de datos)...")
            self.run_pipeline()

        while self.running:
            wait_seconds = self._seconds_until_next_hour()
            self.logger.info(f"Próxima ejecución en {wait_seconds//60} minutos...")

            # Esperar en chunks para poder responder a señales
            for _ in range(wait_seconds):
                if not self.running:
                    break
                time.sleep(1)

            if self.running:
                self.run_pipeline()

        self.logger.info("Scheduler detenido.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Screening Scheduler')
    parser.add_argument('--no-immediate', action='store_true', help='No ejecutar al iniciar')
    parser.add_argument('--email', type=str, default='dummy@example.com', help='Email destinatario')
    args = parser.parse_args()

    scheduler = ScreeningScheduler(recipient_email=args.email)
    scheduler.start(run_immediately=not args.no_immediate)


if __name__ == '__main__':
    main()
