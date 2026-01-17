# ANÁLISIS COMPLETO: ESTRATEGIA HAWKES VOLATILITY

**Fecha de análisis:** 30 de Noviembre de 2025
**Período analizado:** 2018-2022 (In-Sample) + Walk-Forward hasta 2025
**Activo:** Bitcoin (BTCUSD) - Datos horarios

---

## 🎯 RESUMEN EJECUTIVO

Se ha realizado un análisis exhaustivo de la estrategia **Hawkes Volatility** mediante:

1. ✅ **Grid Search de Parámetros** (42 combinaciones)
2. ✅ **Optimización In-Sample** (2018-2022)
3. ✅ **Análisis Walk-Forward** (25 períodos de 2 meses)
4. ✅ **Visualizaciones Estáticas e Interactivas**

---

## 🏆 PARÁMETROS ÓPTIMOS (IN-SAMPLE)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Kappa** | `0.1250` | Factor de decaimiento exponencial del proceso de Hawkes |
| **Lookback** | `120 horas` | Ventana móvil para calcular percentiles q05 y q95 (5 días) |

---

## 📊 MÉTRICAS DE RENDIMIENTO (IN-SAMPLE 2018-2022)

### Rendimiento General

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Log Cumulative Returns** | `5.0921` | Retornos acumulados logarítmicos totales |
| **Profit Factor** | `1.0956` | Rentable (>1), pero margen moderado |
| **Win Ratio** | `51.35%` | Ligeramente mejor que el azar |
| **Total Trades** | `21,278` | Alta frecuencia de trading |

### Distribución de Trades

| Categoría | Ganadores | Perdedores | Win Ratio |
|-----------|-----------|------------|-----------|
| **Totales** | 10,927 | 10,351 | 51.35% |
| **Long** | 4,912 | 4,455 | 52.43% |
| **Short** | 5,908 | 5,792 | 50.48% |

**Observación:** Los trades **LONG** tienen un win ratio ligeramente superior a los **SHORT**.

### Rachas (Streaks)

| Métrica | Valor |
|---------|-------|
| **Max Winning Streak** | 11 trades consecutivos ganadores |
| **Max Losing Streak** | 9 trades consecutivos perdedores |
| **Avg Winning Streak** | 1.89 trades |
| **Avg Losing Streak** | 1.79 trades |

**Observación:** Las rachas son relativamente cortas, indicando alternancia entre wins/losses.

### Mejor/Peor Trade

| Tipo | Valor | Porcentaje |
|------|-------|------------|
| **Mejor Win** | `0.2010` | +22.3% |
| **Peor Loss** | `-0.1603` | -14.8% |

---

## 🔄 RESULTADOS WALK-FORWARD (2018-2025)

El análisis walk-forward valida la **robustez temporal** de la estrategia.

**Configuración:**
- Train window: 4 años
- Test window: 2 meses
- Número de períodos: 25

### Estadísticas Agregadas

| Métrica | Media | Mediana | Std Dev | Min | Max |
|---------|-------|---------|---------|-----|-----|
| **Profit Factor** | 1.0594 | 1.0449 | 0.1900 | 0.7553 | 1.5622 |
| **Win Ratio** | 50.59% | 51.01% | 2.15% | 45.90% | 54.25% |
| **Log Cum Returns** | 0.0461 | - | 0.1412 | - | - |

### Consistencia

| Indicador | Resultado |
|-----------|-----------|
| **Períodos rentables (PF > 1)** | 15 de 25 (60%) |
| **Períodos con retornos positivos** | 15 de 25 (60%) |
| **Retornos acumulados totales** | 1.1520 |

**Conclusión Walk-Forward:** La estrategia muestra **robustez moderada**. En el 60% de los períodos out-of-sample, la estrategia es rentable. Sin embargo, hay **variabilidad significativa** entre períodos (Std Dev del PF = 0.19).

---

## 📈 ANÁLISIS DEL ESPACIO DE PARÁMETROS

### Top 5 Combinaciones por Profit Factor

| Ranking | Kappa | Lookback | PF | Log Cum Returns | Win Ratio |
|---------|-------|----------|-------|-----------------|-----------|
| 1 | 0.1250 | 120h | 1.0956 | 5.0921 | 51.35% |
| 2 | 0.1750 | 120h | 1.0924 | 5.0279 | 51.18% |
| 3 | 0.1500 | 144h | 1.0899 | 4.8443 | 51.05% |
| 4 | 0.1500 | 120h | 1.0873 | 4.7418 | 51.21% |
| 5 | 0.1250 | 144h | 1.0934 | 4.9401 | 51.27% |

**Observaciones:**
- **Kappa entre 0.125-0.175** ofrece los mejores resultados
- **Lookback de 120-144 horas** (5-6 días) es óptimo
- Los mejores parámetros son **robustos**: pequeñas variaciones mantienen buen rendimiento

### Sensibilidad a Parámetros

El análisis de heatmaps revela:

1. **Kappa demasiado bajo** (0.075): Proceso de Hawkes con memoria excesivamente larga
2. **Kappa demasiado alto** (0.200): Proceso demasiado reactivo, pierde señal
3. **Lookback muy corto** (<96h): Percentiles demasiado sensibles al ruido
4. **Lookback muy largo** (>216h): Señales retrasadas, pérdida de oportunidades

---

## 🎨 VISUALIZACIONES GENERADAS

Se han generado los siguientes archivos en `output/hawkes_complete/`:

### 1. Grid Search Results
**Archivo:** `grid_search_results.csv`
**Descripción:** Resultados completos de las 42 combinaciones de parámetros con todas las métricas.

### 2. Parameter Heatmaps
**Archivo:** `parameter_heatmaps.png`
**Descripción:** 5 heatmaps mostrando:
- Log Cumulative Returns
- Profit Factor
- Win Ratio
- Max Winning Streak
- Max Losing Streak

### 3. Metrics Comparison
**Archivo:** `metrics_comparison.png`
**Descripción:** 4 gráficos comparativos:
- Top 10 configuraciones por Profit Factor
- Win Ratio vs Profit Factor scatter
- Rachas ganadoras vs perdedoras
- Long vs Short performance

### 4. Interactive Signals (Estático)
**Archivo:** `interactive_signals.png`
**Descripción:** 3 paneles mostrando:
- Precio de Bitcoin con señales de entrada/salida
- Proceso de Hawkes con percentiles q05/q95
- Timeline de señales (Long/Short/Flat)

### 5. Walk-Forward Analysis
**Archivo:** `walkforward_analysis.png`
**Descripción:** Evolución temporal de:
- Cumulative Returns por período
- Profit Factor por período
- Win Ratio por período

### 6. Gráfico Interactivo HTML ⭐
**Archivo:** `hawkes_interactive.html` (13 MB)
**Descripción:** Gráfico interactivo con Plotly que permite:
- Zoom y pan en cualquier región
- Hover para ver detalles de cada punto
- Candlestick chart de Bitcoin
- Señales de trading interactivas
- Proceso de Hawkes en tiempo real

**💡 Para visualizar:** Abre `hawkes_interactive.html` en cualquier navegador web.

---

## 🔍 ANÁLISIS DETALLADO DE LA ESTRATEGIA

### Lógica de Trading

La estrategia Hawkes opera de la siguiente manera:

1. **Cálculo de Volatilidad Normalizada:**
   ```
   hl_range = log(high) - log(low)
   atr = hl_range.rolling(336).mean()  # 14 días
   norm_range = hl_range / atr
   ```

2. **Aplicación del Proceso de Hawkes:**
   ```
   alpha = exp(-kappa)
   v_hawk[t] = v_hawk[t-1] * alpha + norm_range[t]
   v_hawk = v_hawk * kappa
   ```

3. **Cálculo de Umbrales Dinámicos:**
   ```
   q05 = v_hawk.rolling(lookback).quantile(0.05)  # Umbral bajo
   q95 = v_hawk.rolling(lookback).quantile(0.95)  # Umbral alto
   ```

4. **Generación de Señales:**
   - Cuando `v_hawk < q05`: Marcar punto de referencia (flat)
   - Cuando `v_hawk > q95` (cruce):
     - Si precio subió desde último q05: **LONG**
     - Si precio bajó desde último q05: **SHORT**
   - Mantener posición hasta próximo cruce de q05

### Interpretación de Señales

- **q05 (Percentil 5%)**: Volatilidad muy baja → Mercado en calma
- **q95 (Percentil 95%)**: Volatilidad muy alta → Potencial inicio de movimiento
- **Cruce de q95**: Confirmación de breakout de volatilidad
- **Dirección**: Determinada por cambio de precio durante período de baja volatilidad

---

## ⚠️ ADVERTENCIAS Y LIMITACIONES

### 1. Profit Factor Moderado (1.0956)

El PF de ~1.10 indica rentabilidad **marginal**:
- Costos de trading (comisiones, slippage) pueden **eliminar** el edge
- En mercados reales, el PF neto podría ser **cercano a 1.0**

### 2. Alta Frecuencia de Trading

Con **21,278 trades** en 5 años:
- Promedio: ~12 trades/día
- Esto implica **altos costos de transacción**
- Sensible a latencia de ejecución

### 3. Variabilidad Walk-Forward

La **desviación estándar del PF** (0.19) es alta:
- Algunos períodos tienen PF < 0.8 (pérdidas)
- El rendimiento **no es estable** en el tiempo
- Posible **régimen-dependencia**

### 4. No se probó permutación (MCPT)

Este análisis **NO incluye** Monte Carlo Permutation Test:
- No se valida significancia estadística
- Para completar la validación, ejecutar:
  ```bash
  cd mcpt
  python insample_permutation.py hawkes
  python walkforward_permutation.py hawkes
  ```

---

## 🎯 RECOMENDACIONES

### 1. Validación Estadística
- [ ] Ejecutar **MCPT in-sample** (p-value < 0.05?)
- [ ] Ejecutar **MCPT walk-forward** (p-value < 0.05?)
- [ ] Solo confiar en la estrategia si ambos p-values son significativos

### 2. Optimización de Costos
- [ ] Incorporar **costos de transacción** realistas (0.05-0.10% por trade)
- [ ] Recalcular métricas con costos incluidos
- [ ] Considerar **filtros de trades** para reducir frecuencia

### 3. Mejoras Potenciales
- [ ] Añadir **filtro de tendencia** (evitar lateralidad)
- [ ] Implementar **stop-loss/take-profit** dinámicos
- [ ] Explorar **parámetros adaptativos** (kappa, lookback variables)

### 4. Análisis Complementario
- [ ] **Régimen de mercado**: ¿La estrategia funciona mejor en bull/bear?
- [ ] **Dependencia de trades**: Aplicar filtro "After Loser"
- [ ] **Drawdown analysis**: Calcular max drawdown y duración

---

## 📁 ARCHIVOS GENERADOS

Todos los resultados están en: `/neurotrader/output/hawkes_complete/`

```
hawkes_complete/
├── best_parameters.txt              # Parámetros óptimos y métricas in-sample
├── grid_search_results.csv          # Resultados de 42 combinaciones
├── parameter_heatmaps.png           # Heatmaps de métricas vs parámetros
├── metrics_comparison.png           # Comparación de métricas
├── interactive_signals.png          # Señales estáticas
├── walkforward_analysis.png         # Análisis walk-forward
├── walkforward_stats.txt            # Estadísticas walk-forward
├── hawkes_interactive.html          # Gráfico interactivo (⭐ PRINCIPAL)
└── RESUMEN_EJECUTIVO.md             # Este documento
```

---

## 📝 CONCLUSIONES FINALES

### ✅ Fortalezas

1. **Parámetros estables**: Pequeñas variaciones en kappa/lookback mantienen rendimiento
2. **Lógica interpretable**: Basada en volatilidad y procesos de auto-excitación
3. **60% de períodos rentables** en walk-forward (consistencia moderada)
4. **Longs ligeramente mejores**: Win ratio 52.4% vs 50.5% shorts

### ⚠️ Debilidades

1. **Profit Factor marginal** (~1.10): Vulnerable a costos de transacción
2. **Alta frecuencia**: 12 trades/día → altos costos operativos
3. **Variabilidad temporal**: Algunos períodos con pérdidas significativas
4. **Sin validación MCPT**: Falta prueba de significancia estadística

### 🎓 Veredicto

La estrategia **Hawkes Volatility** muestra:
- ✅ Rentabilidad teórica en backtesting
- ⚠️ Edge **marginal** que requiere **ejecución perfecta**
- ⚠️ **No recomendada para trading real** hasta:
  1. Validar con MCPT (p < 0.05)
  2. Incorporar costos realistas
  3. Reducir frecuencia de trading

**Uso recomendado:** Investigación académica y base para estrategias mejoradas.

---

**Autor:** Análisis generado con Claude Code
**Framework:** NeuroTrader
**Versión:** 1.0
**Fecha:** 30/11/2025
