# Guía Detallada del Benchmark GA

## 1. Propósito general

El benchmark incluido en `experiments/benchmark.py` automatiza la evaluación de múltiples configuraciones del algoritmo genético que impulsa el filtro de rejuvenecimiento facial. Está diseñado para responder de forma objetiva a tres preguntas clave:

1. ¿Qué combinación de operadores (selección, cruce, mutación) funciona mejor para cada función de aptitud?  
2. ¿Qué tan estable es cada configuración cuando se repite con semillas distintas?  
3. ¿Cómo evoluciona la calidad de las soluciones y la diversidad de la población a lo largo de las generaciones?

Para ello, ejecuta un barrido exhaustivo de combinaciones, recopila métricas cuantitativas comparables y produce figuras y reportes que facilitan el análisis.

## 2. Flujo operativo

1. **Carga y preparación**  
   - Se carga la imagen objetivo (`input_images/<archivo>.jpg`).  
   - Se calculan máscaras de piel, ojos, labios y fondo mediante `compute_masks` para restringir la aplicación de filtros.

2. **Definición de configuraciones**  
   - El script combina todas las funciones de aptitud solicitadas con los operadores de selección y cruce especificados.  
   - En el ejemplo completo se evaluaron 3 funciones (`original`, `ideal`, `color`), 4 selecciones (`tournament`, `roulette`, `rank`, `sus`) y 4 cruces (`single_point`, `two_point`, `k_point`, `uniform`), lo que resultó en 48 configuraciones.

3. **Ejecución multihilo**  
   - Cada configuración se ejecuta `--runs` veces (10 en el barrido final).  
   - Las corridas se paralelizan en hilos y cada una recibe una semilla única (`base_seed + run_id`), garantizando reproducibilidad.

4. **Registro de resultados**  
   - Se guarda la mejor aptitud, los parámetros finales, los historiales por generación, la diversidad poblacional y la duración de cada corrida.  
   - Opcionalmente (`--save-images`), se almacena la imagen producida por cada ejecución.

5. **Generación de salidas**  
   - CSV con datos detallados y agregados.  
   - Figuras en PNG que resumen comportamientos clave.  
   - Reporte Markdown (`benchmark_report.md`) con tablas y referencias a las figuras.

## 3. Métricas y archivos de salida

### 3.1 Datos en CSV (`benchmark_results_assignment/data/`)

- **`ga_benchmark_runs.csv`**  
  - Una fila por corrida. Incluye `fitness_label`, operadores usados, `best_fitness`, parámetros finales (`best_params`), historial de aptitud y diversidad (`history`, `diversity_history`), duración (`duration_sec`) y semilla empleada.

- **`ga_benchmark_summary.csv`**  
  - Agregado por combinación (`fitness`, `selection`, `crossover`). Contiene medias y desviaciones de la mejor aptitud, diversidad final, valores normalizados (`mean_cdf`, `mean_z`) y número de corridas. Es la base para comparar configuraciones.

- **`ga_benchmark_history.csv`**  
  - Registra, por generación, la mejor aptitud y la diversidad de cada corrida. Permite reconstruir la curva de convergencia o analizar la evolución de la población.

### 3.2 Figuras en PNG (`benchmark_results_assignment/figures/`)

| Archivo | Qué muestra | Interpretación |
| --- | --- | --- |
| `fitness_boxplot_cdf.png` | Distribución de la mejor aptitud normalizada (CDF 0-1) por fitness y operador de selección. | Identifica combinaciones con percentiles altos y variabilidad reducida. |
| `heatmap_<fitness>_cdf.png` | Promedio de CDF por selección × cruce para un fitness. | Muestra la “intensidad” del rendimiento; colores más brillantes indican mejores resultados. |
| `convergence_<fitness>_cdf.png` | Curvas de convergencia promedio por generación (CDF 0-1), con banda ±1 desviación. | Permite ver la rapidez y estabilidad con la que mejora cada combinación. |
| `convergence_all_cdf.png` | Todas las funciones de aptitud superpuestas en una sola curva normalizada. | Facilita comparar tasas de mejora entre funciones. |
| `diversity_<fitness>.png` | Evolución de la diversidad poblacional (desviación típica media de los parámetros). | Sirve para detectar estancamiento prematuro o exploración excesiva. |
| `diversity_original.png` (y equivalentes) | Especialmente útil si se sospecha pérdida de diversidad en algún fitness. | Diferencias marcadas entre líneas indican configuraciones más o menos exploratorias. |

### 3.3 Reporte autogenerado (`benchmark_results_assignment/reports/benchmark_report.md`)

Incluye:
- Configuración base del experimento (imagen, generaciones, poblaciones, etc.).  
- Tabla con promedio de CDF por etiqueta de fitness.  
- Top 10 combinaciones según CDF.  
- Listado de figuras generadas.  

Es un resumen rápido para validar que el benchmark se ejecutó correctamente antes de un análisis más profundo.

## 4. Normalización y comparación

Las tres funciones de aptitud (`original`, `ideal`, `color`) operan en escalas distintas. Para compararlas de forma justa, el benchmark realiza una normalización en dos etapas:

1. **Calibración previa**  
   - Se generan `--calibration-samples` configuraciones aleatorias por fitness.  
   - Para cada una se evalúa la aptitud, estimando media y desviación (µ, σ).

2. **Normalización durante las corridas**  
   - Cada resultado se transforma a Z-score: `z = (score - µ) / σ`.  
   - Luego se aplica la función de distribución acumulada (CDF) de la normal estándar: `cdf = Φ(z)`, obteniendo valores entre 0 y 1.

Gracias a esta normalización, gráficos y tablas se expresan en la misma escala (CDF 0-1). Así, un 0.99 en `color` y un 0.99 en `ideal` representan percentiles equivalentes.

## 5. Interpretación de resultados

1. **Rendimiento promedio**  
   - Revisar `ga_benchmark_summary.csv` y el boxplot para encontrar combinaciones con CDF alto y baja desviación (configuraciones estables).

2. **Análisis por operador**  
   - Los heatmaps resaltan qué “filas” (selecciones) y “columnas” (cruces) dominan para cada función de aptitud.

3. **Velocidad de convergencia**  
   - Las curvas `convergence_*` muestran qué configuraciones alcanzan altos percentiles rápidamente. La banda ±1 desviación ayuda a detectar comportamientos erráticos.

4. **Diversidad**  
   - Las gráficas `diversity_*` permiten monitorear si una combinación se queda sin diversidad prematuramente (posible sobreajuste) o si mantiene exploración suficiente.

5. **Selección de imágenes**  
   - Aunque Git ignora las imágenes finales (para evitar peso), estas se regeneran al ejecutar el benchmark con `--save-images`. Son fundamentales para la evaluación cualitativa y presentación de resultados.

## 6. Recomendaciones prácticas

- **Exploración rápida**: arrancar con valores bajos (`--runs 2`, `--gens 5`) y sin guardar imágenes para validar dependencias y tiempos aproximados.  
- **Barrido formal**: usar 10 corridas, 20 generaciones y guardar imágenes solo cuando ya se haya confirmado la estabilidad del entorno.  
- **Selección de resultados**: tras el benchmark completo, usar las tablas de sumarizado para elegir la mejor configuración, y complementar con observación visual de sus imágenes asociadas.  
- **Reporte final**: combinar el presente documento, el informe de asignación (`docs/benchmark_assignment_report.md`) y el reporte autogenerado para una documentación integral.

Con esta guía es posible comprender cómo se ejecuta el benchmark, qué salidas produce y de qué manera se comparan las configuraciones de manera objetiva y repetible.
