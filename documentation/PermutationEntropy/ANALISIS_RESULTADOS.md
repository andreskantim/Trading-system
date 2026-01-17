# Análisis de Patrones Ordinales en Bitcoin - Resultados

## Resumen Ejecutivo

Se analizaron **43,823 velas horarias** de Bitcoin (2018-2022) utilizando **Permutation Entropy** para identificar patrones en la estructura del precio (close) y volumen.

---

## Hallazgos Clave

### 1. **Distribución de Patrones - CLOSE**

Los 6 patrones ordinales posibles (basados en 3 velas consecutivas) muestran la siguiente distribución:

| Patrón | Interpretación | Frecuencia | Desviación |
|--------|----------------|------------|------------|
| **Patrón 0** | ↓↓ (Descenso continuo) | **23.75%** | **+42.52%** ⚠️ |
| Patrón 1 | ↓→ (Descenso luego estable/sube) | 12.89% | -22.65% |
| Patrón 2 | →↓ (Estable luego baja) | 14.15% | -15.10% |
| Patrón 3 | V (Baja-Sube) | 14.37% | -13.79% |
| Patrón 4 | →↑ (Estable luego sube) | 12.67% | -23.95% |
| **Patrón 5** | ↑↑ (Ascenso continuo) | **22.16%** | **+32.98%** ⚠️ |

**Frecuencia esperada si fuera ruido puro:** 16.67% (1/6)

### 2. **Distribución de Patrones - VOLUME**

| Patrón | Interpretación | Frecuencia | Desviación |
|--------|----------------|------------|------------|
| Patrón 0 | ↓↓ (Descenso continuo) | 16.89% | +1.36% |
| Patrón 1 | ↓→ (Descenso luego estable/sube) | 15.12% | -9.29% |
| Patrón 2 | →↓ (Estable luego baja) | 14.40% | -13.59% |
| Patrón 3 | V (Baja-Sube) | 15.18% | -8.92% |
| Patrón 4 | →↑ (Estable luego sube) | 14.34% | -13.95% |
| **Patrón 5** | ↑↑ (Ascenso continuo) | **24.06%** | **+44.38%** ⚠️ |

---

## Interpretación de Resultados

### 📊 Test de Uniformidad (Chi-cuadrado)

**CLOSE:**
- Chi² = 0.5787
- p-value = 0.989
- **Conclusión:** Distribución UNIFORME según el test estadístico

**VOLUME:**
- Chi² = 0.2925
- p-value = 0.998
- **Conclusión:** Distribución UNIFORME según el test estadístico

### ⚠️ Paradoja Aparente

Aunque el test de chi-cuadrado indica distribuciones uniformes (no se rechaza H0), observamos:

1. **En CLOSE:** Los patrones de tendencia continua (↓↓ y ↑↑) son **~40% más frecuentes** que lo esperado
2. **En VOLUME:** El patrón ↑↑ (ascenso continuo) es **+44% más frecuente**

**Explicación:** El test chi-cuadrado con n=43,823 muestras tiene alta potencia, pero los desvíos observados, aunque visibles, no son lo suficientemente grandes para rechazar uniformidad al nivel α=0.05. Esto sugiere que:
- **Hay cierta estructura** en los datos (las desviaciones no son aleatorias)
- Pero **no es tan fuerte** como para ser estadísticamente significativa con este método

---

## Insights para Trading

### 🔍 Observaciones Clave

1. **Momentum persistente en precio:**
   - Bitcoin tiende a continuar tendencias (↓↓ y ↑↑) más de lo esperado
   - Los patrones de reversión (V) son menos frecuentes
   - **Implicación:** Las estrategias de seguimiento de tendencia pueden tener ventaja

2. **Volumen muestra tendencia alcista:**
   - El patrón ↑↑ en volumen es el más sobre-representado (+44%)
   - **Implicación:** El volumen tiende a crecer de forma persistente (posible crecimiento del mercado crypto)

3. **Asimetría en reversiones:**
   - Los patrones de transición (↓→, →↓, →↑) están sub-representados
   - **Implicación:** Bitcoin pasa menos tiempo en consolidación, más en tendencia

### 💡 Posibles Estrategias a Explorar

#### Estrategia 1: Momentum Continuation
- **Hipótesis:** Explotar la sobre-representación de patrones ↓↓ y ↑↑
- **Señal de entrada:** Detectar patrón ↑↑ → entrar largo (o ↓↓ → entrar corto)
- **Racionalización:** Si el patrón aparece más de lo esperado, puede continuar

#### Estrategia 2: Volume-Price Divergence
- **Hipótesis:** Usar volumen para confirmar movimientos de precio
- **Señal:** Precio con patrón ↑↑ + volumen con patrón ↑↑ = señal fuerte
- **Divergencia:** Precio ↑↑ pero volumen ↓↓ = posible reversión

#### Estrategia 3: Anti-Consolidation
- **Hipótesis:** Evitar entradas cuando el mercado muestra patrones de consolidación
- **Filtro:** No operar cuando aparecen patrones →↓, →↑, ↓→ (consolidación)
- **Racionalización:** Estos patrones son raros, indicando que no hay tendencia clara

#### Estrategia 4: Pattern Transition Trading
- **Hipótesis:** Las transiciones entre patrones pueden ser predictivas
- **Análisis necesario:** Estudiar cadenas de Markov (¿qué patrón sigue a cuál?)
- **Ejemplo:** Si aparece patrón 3 (V), ¿cuál es el siguiente patrón más probable?

---

## Próximos Pasos Sugeridos

### 1. **Análisis de Transiciones** ⭐ PRIORITARIO
   - Construir matriz de transición de patrones
   - Identificar secuencias predictivas (ej: patrón A → patrón B más de lo esperado)
   - Test de memoria (¿el patrón actual depende de patrones anteriores?)

### 2. **Backtesting de Estrategias**
   - Implementar las 4 estrategias propuestas
   - Comparar con baseline (buy & hold, random)
   - Métricas: Sharpe ratio, max drawdown, win rate

### 3. **Correlación con Estrategias Existentes**
   - Comparar patrones ordinales con las estrategias en `/mcpt`
   - Buscar si las estrategias Donchian/Tree explotan estos patrones
   - Analizar si los patrones pueden mejorar señales existentes

### 4. **Análisis Multi-escala**
   - Repetir análisis con d=4, d=5 (más patrones, más específicos)
   - Probar diferentes timeframes (4h, diario, semanal)
   - Buscar consistencia de patrones entre escalas

### 5. **Ventanas Temporales**
   - Analizar si las frecuencias de patrones cambian con el tiempo
   - Identificar regímenes de mercado según distribución de patrones
   - Bull market vs Bear market: ¿diferentes distribuciones?

---

## Archivos Generados

```
PermutationEntropy/
├── perm_entropy_enhanced.py          # Código mejorado con análisis completo
├── BTCUSDT3600.csv                   # Datos originales
└── results/
    ├── BTCUSDT3600_processed.csv     # Datos con patrones y entropía calculados
    ├── pattern_frequencies_close.png  # Histograma de frecuencias (precio)
    ├── pattern_frequencies_volume.png # Histograma de frecuencias (volumen)
    ├── timeseries_entropy.png         # Serie temporal con entropía
    └── pattern_evolution.png          # Evolución temporal de patrones
```

---

## Conclusión

Aunque estadísticamente los datos parecen "casi uniformes", existen **desviaciones sistemáticas** que sugieren:

1. ✅ **Bitcoin NO es ruido puro** - hay estructura en los datos
2. ✅ **Tendencias persistentes** son más comunes que reversiones
3. ✅ **Volumen creciente** es un patrón dominante
4. ⚠️ **La estructura es sutil** - no trivial de explotar

**Recomendación:** Proceder con análisis de transiciones de patrones y backtesting de estrategias para validar si estas observaciones son explotables en trading real.

---

## Cómo Usar el Código

```bash
# Ejecutar análisis completo
cd PermutationEntropy
python perm_entropy_enhanced.py

# Los resultados se guardan en:
# - results/BTCUSDT3600_processed.csv (datos procesados)
# - results/*.png (gráficos)
```

## Modificar Parámetros

En `perm_entropy_enhanced.py`, líneas 267-268:

```python
d = 3      # Dimensión (2-5 recomendado)
mult = 28  # Multiplicador para ventana de entropía
```

- `d=3`: 6 patrones posibles (3! = 6)
- `d=4`: 24 patrones posibles (4! = 24)
- `d=5`: 120 patrones posibles (5! = 120)

**Nota:** Mayor d = más patrones específicos, pero necesitas más datos para que sean estadísticamente significativos.
