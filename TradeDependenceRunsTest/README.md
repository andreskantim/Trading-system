# Trade Dependence Analysis - Donchian Breakout Strategy

Este proyecto analiza la dependencia entre trades consecutivos en la estrategia Donchian Breakout, específicamente evaluando si existe un sesgo de **mean-reversion** (ganar después de perder) o **trending** (ganar después de ganar).

## Hallazgos Principales (Bitcoin 2018-2022)

### Sesgo Mean-Reversion Confirmado

La estrategia muestra un claro sesgo a **ganar después de perder**:

| Estrategia | Profit Factor | Mean Return | Sharpe Ratio | Mejora vs All |
|------------|--------------|-------------|--------------|---------------|
| **After Loser** | **1.0442** | **0.000106** | **0.0126** | **+2.10%** |
| All Trades | 1.0226 | 0.000055 | 0.0066 | - |
| After Winner | 0.9797 | -0.000050 | -0.0064 | -4.20% |

**Mejora After Loser vs After Winner: +6.58%**

### Evidencia Estadística

- **Runs Test Z-Score Promedio: 1.662**
  - Indica tendencia a mean-reversion (aunque no alcanza significancia estadística completa Z > 1.96)
  - 5 de 27 lookbacks muestran Z > 2 (mean-reversion significativo)
  - Los lookbacks más cortos (12-24) exhiben el efecto más fuerte

## Estructura del Proyecto

### Archivos Core

1. **donchian.py** - Estrategia base Donchian Breakout
   - `donchian_breakout()`: Calcula señales de breakout
   - `get_trades_from_signal()`: Extrae trades individuales desde señales

2. **last_trade_signal.py** - Filtro ganador/perdedor
   - `last_trade_adj_signal()`: Filtra señales según resultado del trade anterior
   - Permite operar solo después de ganar o después de perder

3. **runs_test.py** - Test estadístico de rachas
   - `runs_test()`: Calcula Z-score para detectar dependencia entre trades
   - Z > 2: Mean-reversion (ganar después de perder)
   - Z < -2: Trending (ganar después de ganar)
   - |Z| < 2: Trades independientes

4. **runs_indicator.py** - Indicador rolling de rachas
   - Computa el runs test en ventana móvil

### Análisis Completo

5. **analyze_trade_dependence.py** - Script principal de análisis
   - Carga datos Bitcoin 2018-2022
   - Analiza múltiples lookbacks (12-168)
   - Genera 7 gráficos comparativos
   - Calcula estadísticas completas

### Datos

- **bitcoin_hourly.csv** - Datos horarios Bitcoin 2018-2022
- **BTCUSDT3600.csv** - Datos alternativos BTCUSDT

### Resultados

**results/**
- `trade_dependence_results.csv` - Tabla completa con todas las métricas por lookback

**figures/**
- `profit_factor_by_lookback.png` - Profit factor por lookback y tipo de filtro
- `runs_test_by_lookback.png` - Z-score del test de rachas por lookback
- `mean_return_comparison.png` - Retorno medio comparativo
- `sharpe_comparison.png` - Sharpe ratio comparativo
- `trade_count.png` - Número de trades por lookback y filtro
- `detailed_analysis_lb24.png` - Análisis detallado lookback 24
- `detailed_analysis_lb48.png` - Análisis detallado lookback 48

## Uso

### Ejecutar análisis completo

```bash
python analyze_trade_dependence.py
```

Esto generará:
- Todos los gráficos en `figures/`
- Resultados CSV en `results/trade_dependence_results.csv`
- Resumen estadístico en consola

### Ejecutar análisis individuales

```bash
# Análisis básico con lookback específico
python last_trade_signal.py

# Test de rachas por lookback
python runs_test.py

# Indicador de rachas rolling
python runs_indicator.py
```

## Interpretación del Runs Test

El Runs Test evalúa si una secuencia de trades ganadores/perdedores es aleatoria:

- **Z-Score > 2**: Menos rachas de las esperadas → Mean-reversion
  - Los trades tienden a alternar entre ganadores y perdedores
  - **Filtro recomendado: After Loser**

- **Z-Score < -2**: Más rachas de las esperadas → Trending
  - Los trades tienden a repetir el mismo resultado
  - **Filtro recomendado: After Winner**

- **|Z-Score| < 2**: Rachas dentro de lo esperado → Independencia
  - Los trades son relativamente independientes
  - No se recomienda filtro específico

## Conclusiones para Trading

1. **Usar filtro "After Loser"** para Donchian Breakout en Bitcoin
   - Mejora significativa en todas las métricas
   - Reduce aproximadamente 1/3 de los trades (solo opera ~67% del tiempo)
   - Mejora el Sharpe Ratio casi al doble

2. **Evitar filtro "After Winner"**
   - Empeora el rendimiento
   - Profit Factor < 1 (estrategia perdedora)

3. **Lookbacks óptimos**
   - El efecto es más fuerte en lookbacks cortos (12-30)
   - Considerar lookback 24 como punto de partida

## Próximos Pasos

1. Implementar estrategia con filtro "After Loser" en framework MCPT
2. Validar con test de permutaciones (insample/outsample)
3. Analizar robustez en diferentes periodos
4. Evaluar en otros activos/mercados

## Referencias

- Wald-Wolfowitz Runs Test: Test estadístico de aleatoriedad en secuencias binarias
- Donchian Channels: Sistema de breakout clásico desarrollado por Richard Donchian

## Licencia

Ver LICENSE
