# Guia Detallada del Benchmark GA

## 1. Proposito general

El benchmark incluido en `experiments/benchmark.py` automatiza la evaluacion de multiples configuraciones del algoritmo genetico que impulsa el filtro de rejuvenecimiento facial. Esta disenado para responder de forma objetiva a tres preguntas clave:

1. Que combinacion de operadores (seleccion, cruce, mutacion) funciona mejor para cada funcion de aptitud?
2. Que tan estable es cada configuracion cuando se repite con semillas distintas?
3. Como evoluciona la calidad de las soluciones y la diversidad de la poblacion a lo largo de las generaciones?

Para ello, ejecuta un barrido exhaustivo de combinaciones, recopila metricas cuantitativas comparables y produce figuras y reportes que facilitan el analisis.

## 2. Flujo operativo

1. **Carga y preparacion**
   - Se carga la imagen objetivo (`input_images/<archivo>.jpg`).
   - Se calculan mascaras de piel, ojos, labios y fondo mediante `compute_masks` para restringir la aplicacion de filtros.

2. **Definicion de configuraciones**
   - El script combina todas las funciones de aptitud solicitadas con los operadores de seleccion y cruce especificados.
   - En el ejemplo completo se evaluaron 3 funciones (`original`, `ideal`, `color`), 4 selecciones (`tournament`, `roulette`, `rank`, `sus`) y 4 cruces (`single_point`, `two_point`, `k_point`, `uniform`), lo que resulto en 48 configuraciones.

3. **Ejecucion multihilo**
   - Cada configuracion se ejecuta `--runs` veces (10 en el barrido final).
   - Las corridas se paralelizan en hilos y cada una recibe una semilla unica (`base_seed + run_id`), garantizando reproducibilidad.

4. **Registro de resultados**
   - Se guarda la mejor aptitud, los parametros finales, los historiales por generacion, la diversidad poblacional y la duracion de cada corrida.
   - Opcionalmente (`--save-images`), se almacena la imagen producida por cada ejecucion.

5. **Generacion de salidas**
   - CSV con datos detallados y agregados.
   - Figuras en PNG que resumen comportamientos clave.
   - Reporte Markdown (`benchmark_report.md`) con tablas y referencias a las figuras.

## 3. Metricas y archivos de salida

### 3.1 Datos en CSV (`benchmark_results_assignment/data/`)

- **`ga_benchmark_runs.csv`**
  - Una fila por corrida con operadores usados, `best_fitness`, parametros finales (`best_params`), historiales (`history`, `diversity_history`), duracion (`duration_sec`) y semilla.

- **`ga_benchmark_summary.csv`**
  - Agregado por combinacion (`fitness`, `selection`, `crossover`). Contiene medias y desviaciones de la aptitud, diversidad final y valores normalizados (`mean_cdf`, `mean_z`).

- **`ga_benchmark_history.csv`**
  - Registra, por generacion, la mejor aptitud y la diversidad de cada corrida para reconstruir curvas de convergencia.

### 3.2 Figuras en PNG (`benchmark_results_assignment/figures/`)

| Archivo | Que muestra | Interpretacion |
| --- | --- | --- |
| `fitness_boxplot_cdf.png` | Distribucion de la mejor aptitud normalizada (CDF 0-1) por fitness y seleccion. | Identifica combinaciones con percentiles altos y baja varianza. |
| `heatmap_<fitness>_cdf.png` | Promedio de CDF por seleccion x cruce. | Colores intensos señalan combinaciones superiores. |
| `convergence_<fitness>_cdf.png` | Curva promedio de convergencia (CDF 0-1) con banda +/-1 desviacion. | Muestra rapidez y estabilidad de mejora. |
| `convergence_all_cdf.png` | Todas las funciones en una sola curva normalizada. | Facilita comparar ritmos entre fitness. |
| `diversity_<fitness>.png` | Evolucion de la diversidad (desviacion tipica media). | Detecta estancamiento o exploracion excesiva. |
| `diversity_original.png` | Visor general de diversidad por fitness. | Diferencias marcadas indican operadores mas o menos exploratorios. |

### 3.3 Reporte autogenerado

El archivo `benchmark_results_assignment/reports/benchmark_report.md` resume la configuracion usada, muestra el promedio de CDF por etiqueta de fitness, lista el top 10 de combinaciones y referencia todas las figuras generadas.

## 4. Normalizacion y comparacion

1. **Calibracion previa:** `--calibration-samples` toma configuraciones aleatorias para estimar media (`mu`) y desviacion (`sigma`) por fitness.
2. **Durante las corridas:** cada resultado se convierte a Z-score (`z = (score - mu) / sigma`) y luego a percentil (`cdf = Phi(z)`), quedando en escala 0-1.

Asi se comparan objetos diferentes (`original`, `ideal`, `color`) en una misma escala de calidad.

## 5. Interpretacion practica

1. Ordena `ga_benchmark_summary.csv` por `mean_cdf` para encontrar combinaciones dominantes.
2. Usa los heatmaps para detectar que selecciones y cruces destacan en cada fitness.
3. Consulta las curvas `convergence_*_cdf.png` para evaluar la velocidad y estabilidad del progreso.
4. Observa `diversity_*` para verificar que la poblacion no pierda variedad demasiado pronto.
5. Revisa las imagenes generadas para confirmar la calidad subjetiva del resultado.

## 6. Recomendaciones finales

- Itera con corridas cortas (`--runs 2`, `--gens 5`) y reserva el barrido completo para la entrega final.
- Combina esta guia con `docs/benchmark_assignment_report.md` y `docs/benchmark_guide.md` para documentar resultados.
- Si agregas nuevos operadores o funciones, actualiza `experiments/benchmark.py` y repite el benchmark.
- El `.gitignore` ya excluye `benchmark_results_*/images/`; evita versionar las imagenes finales.
