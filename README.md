# NeuroTrader - Trading Research Framework

Framework modular para investigación y desarrollo de estrategias de trading algorítmico con validación estadística mediante Monte Carlo Permutation Test (MCPT).

## Estructura del Proyecto (Reorganizada)

```
neurotrader/
├── strategies/          # Estrategias de trading (código core)
│   ├── donchian.py               # Donchian Breakout
│   ├── moving_average.py         # MA Crossover
│   ├── tree_strat.py             # Decision Tree
│   ├── hawkes.py                 # Hawkes Volatility
│   └── donchian_aft_loss.py      # Donchian con filtro "After Loser"
│
├── visualize/           # Ejecución y visualización de estrategias
│   └── run_strategy.py           # Ejecutar estrategias con optimización
│
├── mcpt/                # Algoritmos de MCPT
│   ├── bar_permute.py            # Permutación de barras OHLC
│   ├── insample_permutation.py   # Análisis MCPT in-sample
│   ├── walkforward_permutation.py # Análisis MCPT walk-forward
│   ├── old_strategies/           # Backup de estrategias antiguas
│   └── other/                    # Códigos de referencia originales
│
├── filters/             # Filtros para estrategias (en desarrollo)
│
├── config/              # Configuración de rutas y parámetros
│   ├── path.py
│   └── example_usage.py
│
├── output/              # Resultados organizados por estrategia
│   ├── donchian/
│   │   ├── cumulative_returns.png
│   │   ├── parameter_space.png
│   │   ├── insample_mcpt.png
│   │   ├── walkforward_mcpt.png
│   │   └── metrics.txt
│   └── ...
│
├── TradeDependenceRunsTest/     # Análisis de dependencia de trades
├── PermutationEntropy/          # Análisis de entropía
└── VolatilityHawkes/            # Análisis de volatilidad
```

## Componentes Principales

### 1. Estrategias (`strategies/`)

Cada estrategia implementa la interfaz estándar:

```python
def signal(ohlc: pd.DataFrame, *params) -> pd.Series:
    """Genera señales: 1 (long), -1 (short), 0 (flat)"""
    ...

def optimize(ohlc: pd.DataFrame) -> tuple:
    """Optimiza parámetros. Retorna (param1, param2, ..., profit_factor)"""
    ...
```

**Estrategias disponibles:**

1. **Donchian Breakout** - Rupturas de canales de Donchian
2. **Moving Average** - Cruce de medias móviles
3. **Decision Tree** - ML con características técnicas
4. **Hawkes Volatility** - Proceso de Hawkes en volatilidad
5. **Donchian After Loser** - Donchian con filtro de dependencia

**Ejemplo de uso:**
```python
from strategies import donchian

# Optimizar parámetros
best_lookback, best_pf = donchian.optimize(df)

# Generar señales
signals = donchian.signal(df, lookback=best_lookback)
```

### 2. Visualización (`visualize/`)

Ejecuta estrategias con optimización completa y genera visualizaciones.

**Uso:**
```bash
cd visualize
python run_strategy.py donchian
python run_strategy.py moving_average
python run_strategy.py hawkes
```

**Genera:**
- Gráfico de cumulative returns (in-sample vs out-of-sample)
- Espacio de parámetros:
  - **1D**: Gráfico de línea (1 parámetro)
  - **2D**: Heatmap (2 parámetros)
- Métricas en `output/{estrategia}/metrics.txt`

### 3. Análisis MCPT (`mcpt/`)

Validación estadística mediante permutaciones Monte Carlo.

#### In-Sample MCPT

```bash
cd mcpt
python insample_permutation.py donchian
```

Valida si la estrategia supera significativamente al azar en datos históricos.

**Genera:**
- Histograma de Profit Factors (permutaciones vs real)
- Gráfico de cumulative returns
- P-value del test

#### Walk-Forward MCPT

```bash
cd mcpt
python walkforward_permutation.py donchian
```

Valida robustez temporal con optimización walk-forward.

**Genera:**
- Histograma de Profit Factors walk-forward
- Gráfico de cumulative returns
- P-value del test

**Interpretación:**
- **p < 0.05**: Estrategia estadísticamente significativa
- **p ≥ 0.05**: No hay evidencia de ventaja sobre azar

### 4. Filtros (`filters/`)

**En desarrollo.** Módulo para filtros aplicables a cualquier estrategia:
- Filtros de volatilidad
- Filtros de tendencia
- Filtros de dependencia de trades

## Workflow Completo

### 1. Ejecutar y Visualizar Estrategia

```bash
cd visualize
python run_strategy.py donchian
```

Revisa resultados en `output/donchian/`:
- `cumulative_returns.png` - Performance in-sample vs out-of-sample
- `parameter_space.png` - Sensibilidad a parámetros
- `metrics.txt` - Profit Factor, Cumulative Returns, etc.

### 2. Validar con MCPT In-Sample

```bash
cd ../mcpt
python insample_permutation.py donchian
```

Revisa `output/donchian/insample_mcpt.png`:
- Si **p < 0.05**: Estrategia significativa en datos históricos

### 3. Validar con MCPT Walk-Forward

```bash
python walkforward_permutation.py donchian
```

Revisa `output/donchian/walkforward_mcpt.png`:
- Si **p < 0.05**: Estrategia robusta temporalmente

### 4. Analizar Resultados

**Criterios de éxito:**
- ✅ P-value < 0.05 (in-sample y walk-forward)
- ✅ Profit Factor > 1.2
- ✅ Degradación OOS < 30%

## Crear Nueva Estrategia

1. **Crear archivo** `strategies/mi_estrategia.py`:

```python
import pandas as pd
import numpy as np

def signal(ohlc: pd.DataFrame, param1: int):
    """Genera señales de trading"""
    # Implementar lógica
    sig = pd.Series(...)
    return sig

def optimize(ohlc: pd.DataFrame):
    """Optimiza parámetros"""
    best_param1 = ...
    best_pf = ...
    return best_param1, best_pf
```

2. **Ejecutar con herramientas existentes:**

```bash
# Visualización
cd visualize
python run_strategy.py mi_estrategia

# MCPT
cd ../mcpt
python insample_permutation.py mi_estrategia
python walkforward_permutation.py mi_estrategia
```

## Configuración

### Datos

Configurados en `config/path.py`:
- Archivo: `mcpt/data/BTCUSD3600.pq`
- Formato: Parquet (OHLC hourly)
- Periodo por defecto: 2018-2022

### Parámetros Walk-Forward

```python
train_lookback = 24 * 365 * 4  # 4 años de entrenamiento
train_step = 24 * 60           # Re-optimizar cada 60 días
```

### Permutaciones

```python
# In-sample
n_permutations = 1000

# Walk-forward (más costoso computacionalmente)
n_permutations = 400
```

### Paralelización

```bash
# Usar todos los cores (por defecto)
python insample_permutation.py donchian

# Especificar número de workers
N_WORKERS=8 python insample_permutation.py donchian
```

## Módulos Adicionales

### Trade Dependence Analysis

Análisis de dependencia serial entre trades.

```bash
cd TradeDependenceRunsTest
python analyze_trade_dependence.py
```

**Hallazgos (Bitcoin 2018-2022):**
- Sesgo mean-reversion confirmado
- Filtro "After Loser" mejora PF +6.58%
- Implementado en `strategies/donchian_aft_loss.py`

### Permutation Entropy

Análisis de complejidad y predictibilidad.

```bash
cd PermutationEntropy
python perm_entropy.py
```

### Volatility Hawkes

Modelado de volatilidad con procesos de Hawkes.

```bash
cd VolatilityHawkes
python hawkes_visualization.py
```

## Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm pyarrow
```

## Interpretación de Resultados

### Profit Factor

- **PF > 1**: Estrategia rentable
- **PF > 1.2**: Buena estrategia
- **PF > 1.5**: Excelente (verificar overfitting)

### P-Value MCPT

- **p < 0.01**: Muy significativo
- **p < 0.05**: Significativo
- **p ≥ 0.05**: No significativo

### Degradación Out-of-Sample

```
Degradación = (PF_insample - PF_oos) / PF_insample * 100%
```

- **< 20%**: Excelente robustez
- **20-30%**: Buena robustez
- **> 50%**: Posible overfitting

## Buenas Prácticas

1. **Siempre validar con MCPT** antes de confiar en una estrategia
2. **Verificar robustez walk-forward** para asegurar estabilidad temporal
3. **Analizar espacio de parámetros** para evitar overfitting
4. **Comparar in-sample vs out-of-sample** para detectar degradación
5. **Usar múltiples activos** para validar generalización

## Referencias

- **MCPT**: Monte Carlo Permutation Test para trading
- **Walk-Forward**: Optimización out-of-sample con ventanas móviles
- **Bar Permutation**: Preserva estructura intrabar y distribución

## Licencia

Ver LICENSE en cada subdirectorio.

---

**Nota:** Este framework es para investigación y educación. No constituye asesoramiento financiero. El rendimiento pasado no garantiza resultados futuros.
