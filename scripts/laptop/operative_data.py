#!/usr/bin/env python3
"""
Operative Data Consolidator

Consolida datos de múltiples exchanges en dataset operativo único:
- Prioriza exchange principal según configuración (Binance para crypto)
- Ajusta volumen de exchange secundario usando correlaciones del overlap
- Combina datos sin duplicar períodos

Usage:
    python operative_data.py                    # Consolida todo
    python operative_data.py --ticker BTC       # Solo un ticker
    python operative_data.py --test             # Modo test (top 3)
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
project_root = Path(__file__).resolve().parent.parent.parent
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
    CONSOLIDATION,
    get_all_symbols,
    get_ticker,
    is_available,
    get_raw_path,
    get_operative_path,
    ensure_exchange_dirs,
)

import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

ensure_directories()
ensure_exchange_dirs()
log_file = LOGS_DIR / f"operative_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataConsolidator:
    """
    Consolida datos de múltiples fuentes en dataset operativo.

    Prioriza Binance (desde 2017) y usa Kraken para datos anteriores
    con ajuste de volumen basado en el período de overlap.
    """

    def __init__(self):
        """Inicializa el consolidador con configuración de CONSOLIDATION."""
        self.primary_source = CONSOLIDATION['primary']
        self.secondary_source = CONSOLIDATION['secondary']
        self.cutoff_date = CONSOLIDATION['cutoff_date']
        self.min_overlap_candles = CONSOLIDATION['min_overlap_candles']
        self.min_correlation = CONSOLIDATION['min_correlation']
        self.output_dir = CONSOLIDATION['output_dir']

        logger.info("=" * 80)
        logger.info("Data Consolidator Initialized")
        logger.info("=" * 80)
        logger.info(f"Primary source: {self.primary_source}")
        logger.info(f"Secondary source: {self.secondary_source}")
        logger.info(f"Cutoff date: {self.cutoff_date}")
        logger.info(f"Min overlap candles: {self.min_overlap_candles}")
        logger.info(f"Min correlation: {self.min_correlation}")
        logger.info(f"Output: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.consolidation_results = []

    def get_available_years(self, ticker_symbol: str, exchange_name: str) -> list:
        """
        Lista años disponibles para un ticker en un exchange.

        Args:
            ticker_symbol: Símbolo normalizado
            exchange_name: Nombre del exchange

        Returns:
            list: Lista de años disponibles (ordenada)
        """
        data_path = get_raw_path(ticker_symbol, exchange_name)

        if not data_path.exists():
            return []

        pattern = f"{ticker_symbol}_*.parquet"
        files = list(data_path.glob(pattern))

        years = []
        for file in files:
            try:
                year = int(file.stem.split('_')[-1])
                years.append(year)
            except ValueError:
                logger.warning(f"Archivo con formato inesperado: {file}")
                continue

        return sorted(years)

    def load_year_data(self, ticker_symbol: str, exchange_name: str, year: int) -> pd.DataFrame:
        """
        Carga datos de un año específico.

        Args:
            ticker_symbol: Símbolo normalizado
            exchange_name: Nombre del exchange
            year: Año a cargar

        Returns:
            DataFrame con datos del año
        """
        data_path = get_raw_path(ticker_symbol, exchange_name)
        filename = f"{ticker_symbol}_{year}.parquet"
        filepath = data_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        df = pd.read_parquet(filepath)
        return df

    def load_all_data(self, ticker_symbol: str, exchange_name: str) -> pd.DataFrame:
        """
        Carga todos los datos de un ticker de un exchange.

        Args:
            ticker_symbol: Símbolo normalizado
            exchange_name: Nombre del exchange

        Returns:
            DataFrame con todos los datos concatenados
        """
        years = self.get_available_years(ticker_symbol, exchange_name)
        if not years:
            return pd.DataFrame()

        dfs = []
        for year in years:
            try:
                df = self.load_year_data(ticker_symbol, exchange_name, year)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error cargando {ticker_symbol}/{exchange_name}/{year}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs).sort_index()

    def calculate_volume_adjustment(self, ticker_symbol: str) -> dict:
        """
        Calcula factor de ajuste de volumen entre exchanges.

        Usa período de overlap para calcular:
        1. Correlación de precios (validación de consistencia)
        2. Ratio promedio de volúmenes: primary/secondary
        3. Desviación estándar del ratio

        Args:
            ticker_symbol: Símbolo normalizado

        Returns:
            dict con estadísticas de ajuste
        """
        logger.info(f"\nCalculando ajuste de volumen para {ticker_symbol}...")

        # Cargar datos de ambos exchanges
        primary_df = self.load_all_data(ticker_symbol, self.primary_source)
        secondary_df = self.load_all_data(ticker_symbol, self.secondary_source)

        if primary_df.empty or secondary_df.empty:
            logger.warning(f"  No hay datos suficientes para ajuste")
            return {
                'adjustment_factor': 1.0,
                'correlation': np.nan,
                'std_ratio': np.nan,
                'overlap_candles': 0,
                'overlap_start': None,
                'overlap_end': None,
                'valid': False,
                'reason': 'Insufficient data'
            }

        # Encontrar período de overlap
        overlap_start = max(primary_df.index.min(), secondary_df.index.min())
        overlap_end = min(primary_df.index.max(), secondary_df.index.max())

        if overlap_start >= overlap_end:
            logger.warning(f"  No hay periodo de overlap")
            return {
                'adjustment_factor': 1.0,
                'correlation': np.nan,
                'std_ratio': np.nan,
                'overlap_candles': 0,
                'overlap_start': None,
                'overlap_end': None,
                'valid': False,
                'reason': 'No overlap period'
            }

        # Filtrar datos del overlap
        primary_overlap = primary_df.loc[overlap_start:overlap_end]
        secondary_overlap = secondary_df.loc[overlap_start:overlap_end]

        # Merge por timestamp
        merged = pd.merge(
            primary_overlap[['close', 'volume']],
            secondary_overlap[['close', 'volume']],
            left_index=True,
            right_index=True,
            suffixes=('_primary', '_secondary')
        )

        overlap_candles = len(merged)
        logger.info(f"  Overlap: {overlap_start} a {overlap_end}")
        logger.info(f"  Velas en overlap: {overlap_candles:,}")

        if overlap_candles < self.min_overlap_candles:
            logger.warning(f"  Overlap insuficiente: {overlap_candles} < {self.min_overlap_candles}")
            return {
                'adjustment_factor': 1.0,
                'correlation': np.nan,
                'std_ratio': np.nan,
                'overlap_candles': overlap_candles,
                'overlap_start': str(overlap_start),
                'overlap_end': str(overlap_end),
                'valid': False,
                'reason': f'Overlap < {self.min_overlap_candles} candles'
            }

        # Calcular correlación de close prices
        correlation, p_value = stats.pearsonr(
            merged['close_primary'].dropna(),
            merged['close_secondary'].dropna()
        )
        logger.info(f"  Correlacion de precios: {correlation:.4f}")

        if correlation < self.min_correlation:
            logger.warning(f"  Correlacion muy baja: {correlation:.4f} < {self.min_correlation}")
            return {
                'adjustment_factor': 1.0,
                'correlation': float(correlation),
                'std_ratio': np.nan,
                'overlap_candles': overlap_candles,
                'overlap_start': str(overlap_start),
                'overlap_end': str(overlap_end),
                'valid': False,
                'reason': f'Correlation {correlation:.4f} < {self.min_correlation}'
            }

        # Calcular ratio de volúmenes (evitar división por cero)
        merged_vol = merged[(merged['volume_primary'] > 0) & (merged['volume_secondary'] > 0)]
        if len(merged_vol) < 100:
            logger.warning(f"  Pocos puntos con volumen valido: {len(merged_vol)}")
            return {
                'adjustment_factor': 1.0,
                'correlation': float(correlation),
                'std_ratio': np.nan,
                'overlap_candles': overlap_candles,
                'overlap_start': str(overlap_start),
                'overlap_end': str(overlap_end),
                'valid': False,
                'reason': 'Insufficient valid volume points'
            }

        volume_ratio = merged_vol['volume_primary'] / merged_vol['volume_secondary']
        adjustment_factor = float(volume_ratio.median())  # Usar mediana para robustez
        std_ratio = float(volume_ratio.std())

        logger.info(f"  Factor de ajuste: {adjustment_factor:.4f}")
        logger.info(f"  Std del ratio: {std_ratio:.4f}")

        # Validar que el factor es razonable
        if adjustment_factor < 0.01 or adjustment_factor > 100:
            logger.warning(f"  Factor de ajuste fuera de rango razonable: {adjustment_factor}")
            return {
                'adjustment_factor': adjustment_factor,
                'correlation': float(correlation),
                'std_ratio': std_ratio,
                'overlap_candles': overlap_candles,
                'overlap_start': str(overlap_start),
                'overlap_end': str(overlap_end),
                'valid': False,
                'reason': f'Adjustment factor {adjustment_factor} out of range [0.01, 100]'
            }

        return {
            'adjustment_factor': adjustment_factor,
            'correlation': float(correlation),
            'std_ratio': std_ratio,
            'overlap_candles': overlap_candles,
            'overlap_start': str(overlap_start),
            'overlap_end': str(overlap_end),
            'valid': True,
            'reason': None
        }

    def adjust_secondary_volume(self, df: pd.DataFrame, adjustment_stats: dict) -> pd.DataFrame:
        """
        Ajusta volumen de datos secundarios.

        Args:
            df: DataFrame con datos secundarios
            adjustment_stats: dict retornado por calculate_volume_adjustment

        Returns:
            DataFrame con volumen ajustado
        """
        if not adjustment_stats['valid']:
            logger.warning("  Ajuste no valido, retornando datos sin ajustar")
            return df.copy()

        df_copy = df.copy()
        df_copy['volume'] = df_copy['volume'] * adjustment_stats['adjustment_factor']

        logger.info(f"  Volumen ajustado con factor: {adjustment_stats['adjustment_factor']:.4f}")

        return df_copy

    def consolidate_ticker(self, ticker_symbol: str):
        """
        Consolida datos de un ticker específico.

        Args:
            ticker_symbol: Símbolo normalizado
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"CONSOLIDANDO: {ticker_symbol}")
        logger.info(f"{'=' * 80}")

        # Obtener años disponibles de cada exchange
        primary_years = self.get_available_years(ticker_symbol, self.primary_source)
        secondary_years = self.get_available_years(ticker_symbol, self.secondary_source)

        if not primary_years and not secondary_years:
            logger.error(f"No hay datos para {ticker_symbol} en ningun exchange")
            self.consolidation_results.append({
                'ticker': ticker_symbol,
                'years': 0,
                'status': 'failed',
                'error': 'No data available'
            })
            return

        logger.info(f"Primary ({self.primary_source}): {len(primary_years)} anios - {primary_years}")
        logger.info(f"Secondary ({self.secondary_source}): {len(secondary_years)} anios - {secondary_years}")

        # Calcular ajuste de volumen si hay datos en ambos exchanges
        adjustment_stats = None
        if primary_years and secondary_years:
            adjustment_stats = self.calculate_volume_adjustment(ticker_symbol)

        # Crear directorio operative para este ticker
        output_dir = get_operative_path(ticker_symbol)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Consolidar año por año
        all_years = sorted(set(primary_years + secondary_years))
        metadata_by_year = {}

        cutoff_ts = pd.Timestamp(self.cutoff_date)

        for year in all_years:
            logger.info(f"\n  Procesando anio {year}...")

            year_start = pd.Timestamp(f"{year}-01-01")

            # Determinar qué fuente usar
            if year_start >= cutoff_ts and year in primary_years:
                # Usar primary
                source = self.primary_source
                df_year = self.load_year_data(ticker_symbol, source, year)
                volume_adjusted = False
                adjustment_factor = None
                logger.info(f"    Fuente: {source} (primary)")

            elif year in secondary_years:
                # Usar secondary (con ajuste si disponible)
                source = self.secondary_source
                df_year = self.load_year_data(ticker_symbol, source, year)

                if adjustment_stats and adjustment_stats['valid']:
                    df_year = self.adjust_secondary_volume(df_year, adjustment_stats)
                    volume_adjusted = True
                    adjustment_factor = adjustment_stats['adjustment_factor']
                    logger.info(f"    Fuente: {source} (secondary, ajustado x{adjustment_factor:.4f})")
                else:
                    volume_adjusted = False
                    adjustment_factor = None
                    logger.info(f"    Fuente: {source} (secondary, sin ajuste)")

            elif year in primary_years:
                # Año antes de cutoff pero tenemos datos de primary
                source = self.primary_source
                df_year = self.load_year_data(ticker_symbol, source, year)
                volume_adjusted = False
                adjustment_factor = None
                logger.info(f"    Fuente: {source} (primary, antes de cutoff)")
            else:
                logger.warning(f"    Anio {year} no disponible en ninguna fuente")
                continue

            # Asegurar columnas estándar
            df_clean = df_year.reset_index()
            df_clean = df_clean.rename(columns={'index': 'timestamp'})
            if 'timestamp' not in df_clean.columns:
                df_clean['timestamp'] = df_year.index
            df_clean = df_clean[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df_clean = df_clean.set_index('timestamp')

            # Guardar en operative
            filename = f"{ticker_symbol}_{year}.parquet"
            filepath = output_dir / filename

            df_clean.to_parquet(filepath, compression=PARQUET_COMPRESSION, engine=PARQUET_ENGINE)

            # Metadata del año
            metadata_by_year[str(year)] = {
                'source': source,
                'volume_adjusted': volume_adjusted,
                'adjustment_factor': adjustment_factor,
                'candles': len(df_year),
                'date_range': {
                    'start': str(df_year.index.min()),
                    'end': str(df_year.index.max())
                }
            }

            logger.info(f"    Guardado: {filename} ({len(df_year):,} velas)")

        # Guardar metadata completa
        metadata = {
            'ticker': ticker_symbol,
            'consolidation_date': datetime.now().isoformat(),
            'total_years': len(all_years),
            'primary_source': self.primary_source,
            'secondary_source': self.secondary_source,
            'cutoff_date': self.cutoff_date,
            'sources_by_year': metadata_by_year,
            'volume_adjustment': adjustment_stats if adjustment_stats else {
                'applied': False,
                'reason': 'No overlap data'
            }
        }

        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\nConsolidacion completa para {ticker_symbol}")
        logger.info(f"  Metadata: {metadata_file}")

        self.consolidation_results.append({
            'ticker': ticker_symbol,
            'years': len(all_years),
            'status': 'success'
        })

    def consolidate_all(self):
        """Consolida todos los tickers disponibles."""
        all_tickers = get_all_symbols()
        total = len(all_tickers)

        logger.info(f"\n{'#' * 80}")
        logger.info(f"INICIO CONSOLIDACION MASIVA: {total} tickers")
        logger.info(f"{'#' * 80}\n")

        start_time = datetime.now()

        for idx, ticker_symbol in enumerate(all_tickers, 1):
            logger.info(f"\n[{idx}/{total}] Procesando {ticker_symbol}...")

            try:
                self.consolidate_ticker(ticker_symbol)
            except Exception as e:
                logger.error(f"Error consolidando {ticker_symbol}: {e}")
                self.consolidation_results.append({
                    'ticker': ticker_symbol,
                    'years': 0,
                    'status': 'failed',
                    'error': str(e)
                })
                continue

        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time

        successful = sum(1 for r in self.consolidation_results if r['status'] == 'success')
        failed = sum(1 for r in self.consolidation_results if r['status'] == 'failed')

        logger.info(f"\n{'#' * 80}")
        logger.info(f"CONSOLIDACION COMPLETADA")
        logger.info(f"{'#' * 80}")
        logger.info(f"Tiempo total: {duration}")
        logger.info(f"Tickers procesados: {total}")
        logger.info(f"Exitosos: {successful}")
        logger.info(f"Fallidos: {failed}")

        # Guardar resumen
        summary = {
            'consolidation_date': datetime.now().isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_tickers': total,
            'successful': successful,
            'failed': failed,
            'results': self.consolidation_results
        }

        summary_file = self.output_dir / 'consolidation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Resumen guardado en: {summary_file}")
        logger.info(f"Log completo en: {log_file}")


def main():
    """Punto de entrada principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Consolida datos de multiples exchanges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--ticker',
        type=str,
        help='Consolidar solo un ticker especifico'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Modo test: solo top 3 tickers'
    )

    args = parser.parse_args()

    consolidator = DataConsolidator()

    if args.test:
        logger.info("MODO TEST: Consolidando solo top 3 tickers")
        for ticker in get_all_symbols()[:3]:
            consolidator.consolidate_ticker(ticker)
    elif args.ticker:
        logger.info(f"Consolidando ticker especifico: {args.ticker}")
        consolidator.consolidate_ticker(args.ticker.upper())
    else:
        consolidator.consolidate_all()


if __name__ == '__main__':
    main()
