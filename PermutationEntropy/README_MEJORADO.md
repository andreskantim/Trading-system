# Permutation Entropy - Análisis Mejorado

## 📚 Descripción

Este módulo implementa análisis avanzado de **Permutation Entropy** para identificar patrones en series temporales de Bitcoin. A diferencia del código original, incluye:

1. ✅ Cálculo de frecuencias relativas de patrones ordinales
2. ✅ Test estadístico de uniformidad (chi-cuadrado)
3. ✅ Análisis de transiciones entre patrones (Cadenas de Markov)
4. ✅ Visualizaciones completas
5. ✅ Documentación detallada de hallazgos

---

## 🗂️ Estructura de Archivos

```
PermutationEntropy/
├── BTCUSDT3600.csv                   # Datos originales (43,823 velas horarias 2018-2022)
│
├── perm_entropy.py                   # Código original (simple)
├── perm_entropy_enhanced.py          # Código mejorado ⭐
├── pattern_transitions.py            # Análisis de transiciones ⭐
│
├── results/                          # Resultados generados
│   ├── BTCUSDT3600_processed.csv     # Datos con patrones y entropía
│   │
│   ├── pattern_frequencies_close.png # Histograma de frecuencias (precio)
│   ├── pattern_frequencies_volume.png# Histograma de frecuencias (volumen)
│   ├── timeseries_entropy.png        # Serie temporal + entropía
│   ├── pattern_evolution.png         # Evolución temporal de patrones
│   │
│   ├── transition_matrix_close.png   # Matriz de transición (precio)
│   ├── transition_matrix_volume.png  # Matriz de transición (volumen)
│   ├── transition_deviation_close.png# Desviación de transiciones (precio)
│   ├── transition_deviation_volume.png# Desviación de transiciones (volumen)
│   ├── auto_transitions.png          # Persistencia de patrones
│   │
│   ├── transition_matrix_close.csv   # Matriz de transición en CSV
│   └── transition_matrix_volume.csv  # Matriz de transición en CSV
│
├── ANALISIS_RESULTADOS.md            # Resumen de resultados ⭐
├── HALLAZGOS_CRITICOS.md             # Hallazgos clave para trading ⭐⭐⭐
└── README_MEJORADO.md                # Este archivo
```

---

## 🚀 Uso Rápido

### 1. Ejecutar Análisis Completo

```bash
cd PermutationEntropy

# Análisis de frecuencias y entropía
python perm_entropy_enhanced.py

# Análisis de transiciones
python pattern_transitions.py
```

### 2. Ver Resultados

Los gráficos se guardan automáticamente en `results/`

**Archivos clave para revisar:**
1. `HALLAZGOS_CRITICOS.md` - Descubrimientos principales
2. `ANALISIS_RESULTADOS.md` - Análisis detallado
3. `results/*.png` - Visualizaciones

---

## 📊 Qué Hace Cada Script

### `perm_entropy_enhanced.py`

**Calcula:**
- Patrones ordinales para cada vela (6 patrones con d=3)
- Frecuencias relativas de cada patrón
- Test de uniformidad (chi-cuadrado)
- Entropía de permutación (medida de aleatoriedad)

**Genera:**
- 4 gráficos PNG
- 1 CSV con datos procesados
- Estadísticas en consola

**Tiempo de ejecución:** ~30 segundos

### `pattern_transitions.py`

**Calcula:**
- Matriz de transición 6x6 (probabilidad patrón_i → patrón_j)
- Desviación vs distribución uniforme
- Persistencia de patrones (auto-transiciones)
- Predictibilidad (entropía de cada fila)

**Genera:**
- 5 gráficos PNG
- 2 CSV con matrices de transición
- Estadísticas en consola

**Tiempo de ejecución:** ~20 segundos

---

## 🔬 Conceptos Clave

### ¿Qué es un Patrón Ordinal?

Con dimensión d=3, miramos 3 velas consecutivas y clasificamos su **orden relativo**:

**Los 6 patrones posibles (d=3):**

| ID | Interpretación | Significado |
|----|----------------|-------------|
| P0 | ↓↓ | Descenso continuo (v1 > v2 > v3) |
| P1 | ↓→ | Descenso luego estable/sube |
| P2 | →↓ | Estable luego baja |
| P3 | V | Baja-Sube (reversión alcista) |
| P4 | →↑ | Estable luego sube |
| P5 | ↑↑ | Ascenso continuo (v1 < v2 < v3) |

**Ejemplo:**
- Velas: [100, 105, 110] → Orden: [bajo, medio, alto] → **P5 (↑↑)**
- Velas: [100, 95, 90] → Orden: [alto, medio, bajo] → **P0 (↓↓)**
- Velas: [100, 90, 95] → Orden: [alto, bajo, medio] → **P3 (V)**

### ¿Qué es la Permutation Entropy?

Mide la **complejidad/aleatoriedad** de la serie temporal:

- **Entropía alta (~1.0):** Los 6 patrones aparecen con frecuencias similares → Alta aleatoriedad
- **Entropía baja (~0.0):** Uno o pocos patrones dominan → Baja aleatoriedad, alta estructura

### ¿Qué es la Matriz de Transición?

Matriz 6x6 donde `M[i][j]` = probabilidad de que el patrón i sea seguido por el patrón j.

**Si los datos fueran ruido puro:** Todas las entradas serían 1/6 = 16.67%

**En Bitcoin:** Muchas entradas son 0% y otras son >40% → **HAY ESTRUCTURA**

---

## 🎯 Hallazgos Principales

### 1. **Los Datos NO son Ruido**

- 18 de 36 transiciones son **imposibles** (P=0%)
- Las transiciones permitidas son **muy fuertes** (hasta 52%)
- Esto indica estructura determinística

### 2. **Existen 2 Regímenes Mutuamente Excluyentes**

**Grupo BAJISTA:** {P0, P1, P2}
**Grupo ALCISTA:** {P3, P4, P5}

**Regla crítica:** NO hay transiciones entre grupos

### 3. **Tendencias Persisten**

- P0→P0: 43.6% (descenso continuo persiste)
- P5→P5: 41.6% (ascenso continuo persiste)

### 4. **Consolidaciones Predicen Breakouts**

- P2→P0: 52.3% (estable→baja se convierte en descenso fuerte)
- P4→P5: 44.6% (estable→sube se convierte en ascenso fuerte)

### 5. **Reversiones Predicen Tendencias Alcistas**

- P3→P5: 50.8% (patrón V lleva a ascenso continuo)

**Ver `HALLAZGOS_CRITICOS.md` para detalles completos.**

---

## 🛠️ Modificar Parámetros

### Cambiar Dimensión de Patrones

En `perm_entropy_enhanced.py` y `pattern_transitions.py`:

```python
d = 3  # Cambiar a 4 o 5
```

**Nota:**
- d=2 → 2 patrones (muy simple)
- d=3 → 6 patrones ✅ (recomendado)
- d=4 → 24 patrones (requiere más datos)
- d=5 → 120 patrones (difícil de interpretar)

### Cambiar Ventana de Entropía

En `perm_entropy_enhanced.py`:

```python
mult = 28  # lookback = d! * mult
```

Con d=3 y mult=28 → ventana de 6*28 = 168 velas (1 semana)

---

## 📈 Aplicaciones para Trading

### 1. **Filtro de Régimen**

```python
if pattern in [0, 1, 2]:  # Bajista
    allow_short = True
    allow_long = False
elif pattern in [3, 4, 5]:  # Alcista
    allow_short = False
    allow_long = True
```

### 2. **Momentum Continuation**

```python
if pattern == 0:  # ↓↓
    signal = "SHORT" # 43.6% probabilidad de continuar
elif pattern == 5:  # ↑↑
    signal = "LONG"  # 41.6% probabilidad de continuar
```

### 3. **Consolidation Breakout**

```python
if pattern == 2:  # →↓
    signal = "SHORT" # 52.3% probabilidad → ↓↓
elif pattern == 4:  # →↑
    signal = "LONG"  # 44.6% probabilidad → ↑↑
```

### 4. **Reversal Trading**

```python
if pattern == 3:  # V
    signal = "LONG"  # 50.8% probabilidad → ↑↑
```

---

## 🔗 Próximos Pasos

### Correlación con Estrategias Existentes

1. **`/mcpt/donchian.py`**
   - ¿Los breakouts de Donchian coinciden con transiciones P2→P0 o P4→P5?

2. **`/mcpt/tree_strat.py`**
   - ¿Podemos mejorar el árbol añadiendo patrones ordinales como features?

3. **`/VolatilityHawkes`**
   - ¿Los procesos de Hawkes capturan la auto-excitación de P0→P0 y P5→P5?

### Experimentos Sugeridos

1. **Backtesting simple:**
   - Long solo en régimen alcista {P3,P4,P5}
   - Short solo en régimen bajista {P0,P1,P2}

2. **Combinación con indicadores:**
   - RSI + Patrón ordinal
   - MACD + Patrón ordinal

3. **Position sizing dinámico:**
   - Ajustar tamaño según probabilidad de transición

4. **Walk-forward validation:**
   - Validar matriz de transición out-of-sample

---

## 📚 Referencias

**Permutation Entropy:**
- Bandt, C., & Pompe, B. (2002). "Permutation entropy: a natural complexity measure for time series." Physical review letters, 88(17), 174102.

**Aplicaciones en Finanzas:**
- Zunino, L., et al. (2009). "Permutation entropy of fractional Brownian motion and fractional Gaussian noise." Physics Letters A, 372(27-28), 4768-4774.

**Markov Chains:**
- Norris, J. R. (1998). "Markov chains." Cambridge university press.

---

## 🤝 Contribuciones

Para mejorar este análisis:

1. Implementar backtesting de estrategias propuestas
2. Añadir análisis multi-escala (diferentes timeframes)
3. Comparar con otras métricas de entropía (Shannon, Tsallis)
4. Integrar con estrategias existentes en `/mcpt`

---

## 📄 Licencia

Ver archivo `LICENSE` en el directorio raíz.

---

## ✉️ Contacto

Para preguntas sobre este análisis, revisar:
1. `ANALISIS_RESULTADOS.md` - Explicación detallada
2. `HALLAZGOS_CRITICOS.md` - Insights clave
3. Código fuente (bien comentado)

---

**¡Happy Trading! 📈**
