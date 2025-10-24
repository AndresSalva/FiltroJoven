# Conceptos clave para el proyecto de rejuvenecimiento facial con algoritmos genéticos

Este documento resume los fundamentos teóricos y prácticos necesarios para comprender el código del repositorio y reproducir los experimentos.

## 1. Flujo general del sistema

1. **Entrada**: imagen de un rostro (`input_images/`).
2. **Detección y segmentación**: `core/haar_detector.py` calcula un cuadro facial y máscaras para piel, ojos, labios y regiones a preservar.
3. **Transformación parametrizada**: `transformations/face_manipulator.py` aplica filtros (suavizado bilateral, ajuste de gamma y enfoque selectivo) controlados por cinco parámetros continuos/discretos.
4. **Algoritmo genético (AG)**:
   - Inicializa una población de individuos, cada uno con un conjunto de parámetros.
   - Evalúa cada individuo mediante una **función de aptitud** diseñada para correlacionarse con “apariencia joven”.
   - Repite selección → cruce → mutación → evaluación durante varias generaciones para optimizar la aptitud.
5. **Salida**: imagen rejuvenecida y parámetros óptimos; los resultados opcionales se guardan bajo `output_images/` o `benchmark_results/images/`.

## 2. Fundamentos de algoritmos genéticos

1. **Representación del individuo**
   - Genotipo: vector de parámetros (`bilateral_d`, `sigma_color`, `sigma_space`, `gamma`, `unsharp_amount`).
   - Fenotipo: imagen resultante tras aplicar la transformación con esos parámetros.

2. **Población y generación**
   - Conjunto de individuos evaluados en cada iteración. El tamaño se define con `--pop`.
   - Los individuos se mantienen en memoria y el mejor se preserva (elitismo).

3. **Funciones de aptitud (fitness)**
   - **original**: pondera reducción de arrugas, preservación de bordes, uniformidad y similitud estructural (`ga/fitness_functions.py:18`).
   - **ideal**: aproxima la piel a un referente suavizado, penalizando zonas planas y premiando bordes en ojos/labios.
   - **color**: combina similitud de histograma en el espacio Lab y textura deseada.
   - También existen combinaciones ponderadas mediante `WeightedCompositeFitness`.

4. **Operadores genéticos**
   - **Selección** (`ga/selection_operators.py`): torneo, ruleta, ranking y muestreo universal estocástico (SUS).
   - **Crossover** (`ga/crossover_operators.py`): punto único, dos puntos, k-puntos y uniforme.
   - **Mutación** (`ga/mutation_operators.py`): agrega ruido gaussiano controlado por la amplitud de cada parámetro.
   - Todos los operadores actúan sobre diccionarios; `clip_params` asegura que los valores se mantengan dentro de límites físicos.

5. **Control de semillas**
   - `run_genetic_algorithm()` recibe `seed` y sincroniza `random` y `numpy`. Esto vuelve las corridas reproducibles cuando el benchmark fija `base_seed`.

6. **Métricas adicionales**
   - **Historial de aptitud**: mejor aptitud histórica y mejor de la generación.
   - **Diversidad poblacional**: desviación estándar promedio de los parámetros por generación.
   - **Duración**: tiempo empleado en cada corrida (útil para reportes).

## 3. Métricas de análisis

1. **Criterios cuantitativos**:
   - Mejor aptitud alcanzada (por corrida/configuración).
   - Varianza entre corridas (se observa en `ga_benchmark_summary.csv`).
   - Curvas de convergencia (`benchmark_results/figures/convergence_*.png`).
   - Diversidad poblacional (`diversity_*.png`), requisito explícito del PDF.

2. **Criterios cualitativos**:
   - Calidad visual subjetiva: comparar imágenes guardadas (`output_images/` o `benchmark_results/images/`).
   - Comparación con un ideal (máscara facial limpia) o contra otras configuraciones.

3. **Registro de resultados**:
   - `ga_benchmark_runs.csv`: JSON en columnas `best_params`, `metrics`, `history`, `diversity_history`.
   - `ga_benchmark_summary.csv`: promedios y desviaciones por combinación.
   - `ga_benchmark_history.csv`: series temporales por corrida para análisis avanzado.

## 4. Transformaciones de imagen utilizadas

1. **Filtro bilateral**: suaviza la piel preservando bordes; controlado por `bilateral_d`, `sigma_color`, `sigma_space`.
2. **Corrección gamma**: aclarado/oscurecimiento (`gamma`).
3. **Enfoque selectivo (unsharp masking)**: refuerza bordes en zonas fuera de piel o en ojos/labios (`unsharp_amount`).
4. **Máscara de conservación**: aseguran que ojos, labios y fondo se mantengan nítidos (`keep_mask`).

Estas operaciones se aplican en `apply_transformation`, y las máscaras provienen de `compute_masks`.

## 5. Relación con los requisitos de la práctica

1. **Implementación básica**:
   - Representación → parámetros (sección 2.1).
   - Transformación → `apply_transformation`.
   - AG básico → `run_genetic_algorithm`.

2. **Funciones de aptitud múltiples**: tres variantes más combinaciones ponderadas (`ga/fitness_functions.py`).

3. **Operadores genéticos variados**: selección/cruce/mutación definidos en `ga/`.

4. **Experimentación sistemática**:
   - Benchmark configurable (`experiments/benchmark.py`).
   - Ejecuta 10 corridas por combinación para evaluar estabilidad y varianza.
   - Gráficas de convergencia y diversidad generadas automáticamente.

5. **Análisis y evaluación**:
   - Los CSV/figuras permiten elaborar los apartados cuantitativos.
   - Imágenes guardadas sirven para evaluación visual y discusión cualitativa.

6. **Documentación**:
   - `docs/benchmark_guide.md` explica el uso del benchmark.
   - Este documento resume los conceptos teóricos exigidos.

## 6. Pasos sugeridos para entender y presentar resultados

1. **Ejecutar el benchmark completo** (ver `docs/benchmark_guide.md`).
2. **Revisar CSV y figuras** para completar el informe con datos cuantitativos.
3. **Comparar imágenes resultado** para describir hallazgos visuales.
4. **Redactar informe final**:
   - Metodología (cómo se configuró el benchmark).
   - Resultados (tablas y gráficas relevantes).
   - Discusión (respuestas a las preguntas del PDF).
   - Conclusiones y mejoras futuras.

Con estos conceptos y referencias podrás comprender el proyecto, explicar sus componentes y defender los resultados frente a los criterios de evaluación del magíster.
