"""
Signal Detector - Detecta nuevas señales de trading en tiempo real
Aplica estrategias sobre datos descargados y guarda solo señales NUEVAS
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config.tickers import CRYPTO_TICKERS, get_exchange_symbol
from models.strategies import hawkes, bollinger_b2b


class SignalDetector:
    """Detecta y almacena nuevas señales de trading"""

    # Configuración de estrategias: (módulo, parámetros)
    STRATEGIES = {
        'hawkes': {
            'module': hawkes,
            'params': {'kappa': 0.125, 'lookback': 169}
        },
        'bollinger_b2b': {
            'module': bollinger_b2b,
            'params': {'period': 20, 'num_std': 2.0}
        }
    }

    def __init__(self, data_dir: str = None, signals_dir: str = None):
        """
        Args:
            data_dir: Directorio con datos descargados (SQLite)
            signals_dir: Directorio donde guardar señales CSV
        """
        if data_dir is None:
            self.data_dir = project_root / "data" / "operative"
        else:
            self.data_dir = Path(data_dir)

        if signals_dir is None:
            self.signals_dir = project_root / "screening" / "signals"
        else:
            self.signals_dir = Path(signals_dir)

        # Crear directorios por estrategia
        for strategy_name in self.STRATEGIES.keys():
            (self.signals_dir / strategy_name).mkdir(parents=True, exist_ok=True)

    def load_data_from_sqlite(self, symbol: str) -> pd.DataFrame:
        """Carga datos desde SQLite para un símbolo genérico (ej: 'BTC')"""
        db_path = self.data_dir / symbol / "screening.db"

        if not db_path.exists():
            raise FileNotFoundError(f"No se encontró BD para {symbol}: {db_path}")

        conn = sqlite3.connect(str(db_path))

        try:
            df = pd.read_sql_query(
                "SELECT * FROM ohlcv ORDER BY timestamp ASC",
                conn
            )

            # Convertir timestamp (milisegundos) a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            return df

        finally:
            conn.close()

    def get_signals_csv_path(self, strategy_name: str) -> Path:
        """Retorna path al CSV de señales para una estrategia"""
        return self.signals_dir / strategy_name / "signals.csv"

    def load_existing_signals(self, strategy_name: str) -> pd.DataFrame:
        """
        Carga señales existentes desde CSV

        Returns:
            DataFrame con señales previas (vacío si no existe)
        """
        csv_path = self.get_signals_csv_path(strategy_name)

        if not csv_path.exists():
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'signal_type', 'price', 'strategy', 'metadata'
            ])

        return pd.read_csv(csv_path, parse_dates=['timestamp'])

    def detect_new_signals(self, symbol: str, strategy_name: str, ohlc_data: pd.DataFrame) -> list:
        """Detecta señales nuevas comparando con estado anterior"""
        strategy_config = self.STRATEGIES[strategy_name]
        strategy_module = strategy_config['module']
        params = strategy_config['params']

        signals = strategy_module.signal(ohlc_data, **params)
        signal_changes = signals.diff()
        change_indices = signal_changes[signal_changes != 0].index[1:]

        new_signals = []
        for idx in change_indices:
            prev_signal = signals.loc[:idx].iloc[-2]
            curr_signal = signals.loc[idx]
            price = ohlc_data.loc[idx, 'close']

            if curr_signal == 1 and prev_signal != 1:
                signal_type = 'entry_long'
            elif curr_signal == -1 and prev_signal != -1:
                signal_type = 'entry_short'
            elif curr_signal == 0 and prev_signal != 0:
                signal_type = 'exit'
            else:
                continue

            new_signals.append({
                'timestamp': idx,
                'symbol': symbol,
                'signal_type': signal_type,
                'price': price,
                'strategy': strategy_name,
                'metadata': str(params)
            })

        return new_signals

    def filter_truly_new_signals(self, new_signals: list, existing_signals: pd.DataFrame) -> list:
        """Filtra señales que realmente son nuevas (no existen en CSV)"""
        if existing_signals.empty:
            return new_signals

        existing_set = set(
            existing_signals[['timestamp', 'symbol', 'signal_type']].itertuples(index=False, name=None)
        )

        truly_new = []
        for sig in new_signals:
            key = (sig['timestamp'], sig['symbol'], sig['signal_type'])
            if key not in existing_set:
                truly_new.append(sig)

        return truly_new

    def save_signals(self, strategy_name: str, new_signals: list):
        """
        Guarda señales nuevas en CSV (append mode)

        Args:
            strategy_name: Nombre de estrategia
            new_signals: Lista de nuevas señales a guardar
        """
        if not new_signals:
            return

        csv_path = self.get_signals_csv_path(strategy_name)

        # Convertir a DataFrame
        df_new = pd.DataFrame(new_signals)

        # Append o crear nuevo
        if csv_path.exists():
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(csv_path, index=False)

    def process_ticker(self, symbol: str, strategy_name: str, verbose: bool = True) -> dict:
        """Procesa un ticker para una estrategia específica"""
        try:
            if verbose:
                print(f"  [{strategy_name}] Procesando {symbol}...")

            ohlc_data = self.load_data_from_sqlite(symbol)

            if len(ohlc_data) < 500:
                if verbose:
                    print(f"    SKIP: Datos insuficientes ({len(ohlc_data)} velas)")
                return {'symbol': symbol, 'strategy': strategy_name, 'success': False, 'reason': 'insufficient_data'}

            detected_signals = self.detect_new_signals(symbol, strategy_name, ohlc_data)
            existing_signals = self.load_existing_signals(strategy_name)
            truly_new = self.filter_truly_new_signals(detected_signals, existing_signals)
            self.save_signals(strategy_name, truly_new)

            if verbose:
                print(f"    Detectadas: {len(detected_signals)}, Nuevas: {len(truly_new)}")

            return {'symbol': symbol, 'strategy': strategy_name, 'success': True, 'detected': len(detected_signals), 'new_signals': len(truly_new)}

        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            return {'symbol': symbol, 'strategy': strategy_name, 'success': False, 'error': str(e)}

    def process_all_tickers(
        self,
        tickers_list: list = None,
        strategies: list = None,
        verbose: bool = True
    ) -> dict:
        """
        Procesa todos los tickers para todas las estrategias

        Args:
            tickers_list: Lista de símbolos genéricos (None = todos)
            strategies: Lista de estrategias (None = todas)
            verbose: Si imprimir logs

        Returns:
            Dict con resultados agregados
        """
        if tickers_list is None:
            tickers_list = [
                t['symbol'] for t in CRYPTO_TICKERS
                if t.get('binance') is not None
            ]

        if strategies is None:
            strategies = list(self.STRATEGIES.keys())

        print(f"\n{'='*70}")
        print(f"DETECCIÓN DE SEÑALES")
        print(f"{'='*70}")
        print(f"Tickers: {len(tickers_list)}")
        print(f"Estrategias: {', '.join(strategies)}")

        results = {strategy: [] for strategy in strategies}

        for idx, symbol in enumerate(tickers_list, 1):
            if get_exchange_symbol(symbol, 'binance') is None:
                continue

            print(f"\n[{idx}/{len(tickers_list)}] {symbol}")

            for strategy_name in strategies:
                result = self.process_ticker(symbol, strategy_name, verbose=verbose)
                results[strategy_name].append(result)

        # Resumen por estrategia
        print(f"\n{'='*70}")
        print(f"RESUMEN DE SEÑALES")
        print(f"{'='*70}")

        for strategy_name, strategy_results in results.items():
            successful = [r for r in strategy_results if r['success']]
            total_new_signals = sum(r.get('new_signals', 0) for r in successful)

            print(f"\n[{strategy_name}]")
            print(f"  Tickers procesados: {len(successful)}/{len(strategy_results)}")
            print(f"  Nuevas señales: {total_new_signals}")

            if total_new_signals > 0:
                print(f"  Tickers con señales nuevas:")
                for r in successful:
                    if r.get('new_signals', 0) > 0:
                        print(f"    - {r['symbol']}: {r['new_signals']} señales")

        return results


def main():
    """Función principal para testing"""
    detector = SignalDetector()

    # Test con top 3 tickers
    test_tickers = ['BTC', 'ETH', 'BNB']

    print("TESTING: Detectando señales en top 3 tickers...")
    results = detector.process_all_tickers(test_tickers)

    print(f"\n\nSeñales guardadas en: {detector.signals_dir}")


if __name__ == '__main__':
    main()
