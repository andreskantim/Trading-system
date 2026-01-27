# Screening System

Sistema automatizado de descarga de datos y detección de señales de trading en tiempo real para criptomonedas.

## Estructura del Proyecto

```
screening/
├── download/
│   ├── binance_downloader.py    # Descarga datos horarios de Binance (último mes)
│   └── run_update.py            # Script de actualización periódica
│
├── generate_signal/
│   └── signal_detector.py       # Detección de señales nuevas
│
├── signals/                     # Señales generadas (CSV)
│   ├── hawkes/
│   │   └── signals.csv
│   └── bollinger_b2b/
│       └── signals.csv
│
├── broadcast/
│   ├── email_sender.py          # Envío de señales por email
│   └── sent/                    # Logs de emails enviados (JSON)
│
├── scheduler.py                 # Ejecución automática cada hora
└── run_screening.py             # Pipeline completo
```

## Características

### 1. Download System
- **Fuente**: Binance API (sin autenticación requerida)
- **Timeframe**: Velas horarias (1h)
- **Ventana**: Último mes (ventana deslizante)
- **Storage**: SQLite (`data/operative/{TICKER_BINANCE}/data.db`)
- **Rate Limiting**: 0.5s entre requests
- **Retries**: 3 intentos con backoff exponencial

### 2. Signal Detection
- **Estrategias disponibles**:
  - `hawkes`: Volatilidad basada en procesos de Hawkes
  - `bollinger_b2b`: Breakout de Bandas de Bollinger

- **Detección inteligente**: Solo guarda señales NUEVAS (compara con estado previo)
- **Tipos de señal**:
  - `entry_long`: Entrada en posición larga
  - `entry_short`: Entrada en posición corta
  - `exit`: Salida a flat

### 3. Signals Storage
Formato CSV con columnas:
```csv
timestamp,ticker,signal_type,price,strategy,metadata
2026-01-20 22:00:00+00:00,BTCUSDT,entry_short,88509.18,hawkes,"{'kappa': 0.125, 'lookback': 169}"
```

## Instalación

### Dependencias
```bash
pip install requests pandas numpy
```

### Verificar instalación
```bash
cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/Trading-system
python3 screening/download/binance_downloader.py  # Test downloader
python3 screening/generate_signal/signal_detector.py  # Test detector
```

## Uso

### Opción 1: Scheduler Automático (Recomendado)

```bash
# Ejecutar scheduler (cada hora, CRYPTO_25 por defecto)
python3 screening/scheduler.py

# Con email específico
python3 screening/scheduler.py --email tu_email@gmail.com

# Sin ejecución inicial (solo esperar próxima hora)
python3 screening/scheduler.py --no-immediate
```

### Opción 2: Pipeline Manual

```bash
# Ejecutar una vez (CRYPTO_25 por defecto)
python3 screening/run_screening.py

# Con email para broadcast
python3 screening/run_screening.py --email tu_email@gmail.com

# Solo top 10 cryptos
python3 screening/run_screening.py --tickers crypto_10

# Todos los tickers (~100)
python3 screening/run_screening.py --tickers crypto_all

# Sin broadcast
python3 screening/run_screening.py --skip-broadcast

# Solo descarga (sin señales ni broadcast)
python3 screening/run_screening.py --skip-signals --skip-broadcast
```

### Opción 2: Módulos Individuales

#### Descargar datos manualmente
```bash
python3 screening/download/run_update.py
```

#### Detectar señales manualmente
```bash
python3 screening/generate_signal/signal_detector.py
```

### Opción 3: Cron (alternativa al scheduler)

```bash
# Editar crontab
crontab -e

# Ejecutar cada hora (minuto 5), CRYPTO_25 por defecto
5 * * * * cd /path/to/Trading-system && python3 screening/run_screening.py --email tu@email.com
```

## Uso Programático

### Descargar datos desde Python

```python
from screener.actualize_data.binance_downloader import BinanceDownloader

downloader = BinanceDownloader()

# Descargar un ticker específico
result = downloader.download_ticker('BTCUSDT')

# Descargar múltiples tickers
results = downloader.download_all_tickers(['BTC', 'ETH', 'BNB'])
```

### Detectar señales desde Python

```python
from screener.generate_signal.signal_detector import SignalDetector

detector = SignalDetector()

# Procesar un ticker con una estrategia
result = detector.process_ticker('BTCUSDT', 'hawkes')

# Procesar todos los tickers
results = detector.process_all_tickers(
    tickers_list=['BTC', 'ETH'],
    strategies=['hawkes', 'bollinger_b2b']
)
```

### Leer señales guardadas

```python
import pandas as pd

# Leer señales de Hawkes
df_hawkes = pd.read_csv('screening/signals/hawkes/signals.csv', parse_dates=['timestamp'])

# Filtrar señales recientes (últimas 24h)
recent = df_hawkes[df_hawkes['timestamp'] > pd.Timestamp.now() - pd.Timedelta(hours=24)]

# Filtrar por ticker específico
btc_signals = df_hawkes[df_hawkes['ticker'] == 'BTCUSDT']
```

## Configuración

### Tickers disponibles
Los tickers se configuran en `config/tickers.py`:
- `CRYPTO_ALL`: Todos los tickers (~100)
- `CRYPTO_10`: Top 10 por capitalización
- `CRYPTO_25`: Top 25 por capitalización

### Parámetros de estrategias
Configurados en `screening/generate_signal/signal_detector.py`:

```python
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
```

### Agregar nueva estrategia

1. Crear estrategia en `models/strategies/mi_estrategia.py`:
```python
def signal(ohlc: pd.DataFrame, param1: float, param2: int):
    # Tu lógica aquí
    return pd.Series(...)  # 1, -1, 0
```

2. Registrar en `signal_detector.py`:
```python
from models.strategies import mi_estrategia

STRATEGIES = {
    'mi_estrategia': {
        'module': mi_estrategia,
        'params': {'param1': 1.5, 'param2': 20}
    }
}
```

3. Crear directorio para señales:
```bash
mkdir -p screening/signals/mi_estrategia
```

## Logs

Los logs se guardan en:
```
logs/screening/
├── screening_20260127_040500.log
├── update_20260127_040000.log
└── ...
```

## Testing

### Test rápido con 3 tickers
```bash
# Editar main() en binance_downloader.py o signal_detector.py
# Cambiar test_tickers = ['BTC', 'ETH', 'BNB']
python3 screening/download/binance_downloader.py
python3 screening/generate_signal/signal_detector.py
```

### Verificar datos descargados
```bash
# Contar registros en BD
sqlite3 data/operative/BTCUSDT/data.db "SELECT COUNT(*) FROM ohlcv;"

# Ver primeras filas
sqlite3 data/operative/BTCUSDT/data.db "SELECT datetime(timestamp/1000, 'unixepoch') as dt, close FROM ohlcv LIMIT 5;"
```

### Verificar señales generadas
```bash
# Ver señales de Hawkes
head -20 screening/signals/hawkes/signals.csv

# Contar señales por ticker
cat screening/signals/bollinger_b2b/signals.csv | cut -d',' -f2 | sort | uniq -c
```

## Troubleshooting

### Error: "No se encontró BD para XXXUSDT"
**Causa**: No se han descargado datos para ese ticker.
**Solución**: Ejecutar primero el downloader:
```bash
python3 screening/download/run_update.py
```

### Error: "Datos insuficientes (XXX velas)"
**Causa**: El ticker tiene menos de 500 velas (necesario para estrategias).
**Solución**: Normal para tickers recién listados, se omiten automáticamente.

### Error: "Rate limit exceeded"
**Causa**: Demasiados requests a Binance API.
**Solución**: El sistema ya tiene rate limiting (0.5s), pero si persiste:
- Aumentar `RATE_LIMIT` en `binance_downloader.py`
- Procesar menos tickers a la vez

### Señales duplicadas
**Causa**: Ejecutar el detector múltiples veces sin cambios en datos.
**Solución**: El sistema filtra automáticamente duplicados, pero puedes limpiar CSVs:
```bash
rm screening/signals/*/signals.csv
```

## Roadmap

### Fase 1: Download + Signal Detection ✅
- [x] Descarga desde Binance (último mes)
- [x] Detección de señales (hawkes, bollinger_b2b)
- [x] Storage en SQLite + CSV
- [x] Pipeline unificado

### Fase 2: Scheduler + Broadcast ✅
- [x] Scheduler automático cada hora
- [x] Recuperación de datos faltantes al iniciar
- [x] Envío de señales por email (SMTP)
- [x] Logs de envíos en `broadcast/sent/`
- [x] CRYPTO_25 como batch por defecto

### Fase 3: Mejoras (Futuro)
- [ ] Dashboard web (Flask/Streamlit)
- [ ] Backtesting de señales detectadas
- [ ] Alertas por Telegram/Discord

## Notas Importantes

1. **Nomenclatura de tickers**:
   - Configuración: Usa símbolos genéricos (`'BTC'`, `'ETH'`)
   - Storage: Usa nomenclatura Binance (`'BTCUSDT'`, `'ETHUSDT'`)
   - Función `get_exchange_symbol(symbol, 'binance')` hace la conversión

2. **Ventana deslizante**:
   - Se mantienen SOLO últimos 30 días en SQLite
   - Datos antiguos se eliminan automáticamente en cada actualización

3. **Timezone**:
   - Todos los timestamps son UTC
   - Binance devuelve timestamps en milisegundos

4. **Rendimiento**:
   - ~100 tickers toman ~10-15 minutos (descarga + señales)
   - Rate limit de 0.5s entre requests = ~200 requests/min

## Soporte

Para reportar bugs o solicitar features:
1. Revisar logs en `logs/screening/`
2. Verificar configuración en `config/tickers.py`
3. Consultar este README
